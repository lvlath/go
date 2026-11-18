// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion owns the single positive-intensity trajectory kernel.
// This file keeps RNG ordering, wide log accumulation, and atomic publication.
package jumpdiffusion

import (
	"context"
	"math/big"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/rng"
)

// stepWorkspace stores all mutable precision-fixed operands for one path.
// No workspace is shared among independent simulations or retained afterwards.
// The price anchor is kept outside the log-growth state to avoid log(S0) error.
type stepWorkspace struct {
	previous, current, interval                  *big.Float
	intensity, mean, maxMean                     *big.Float
	drift, volatility, scale, ito, shock, normal *big.Float
	half, increment, growth, candidate           *big.Float
	jump, jumpCount, jumpScale, jumpZ, jumpMean  *big.Float
	jumpStd                                      *big.Float
}

// newStepWorkspace preallocates one private extended-precision numeric workspace.
// Compensator arithmetic stays wide even when intensity*compensator exceeds float64.
//
// Implementation:
//   - Stage 1: allocate fixed-precision accumulators and copy model coefficients.
//   - Stage 2: subtract the compensated proportional jump drift when selected.
//   - Stage 3: return reusable buffers for all observation intervals.
//
// Inputs: p has already passed canonical validation; compensator is finite.
// Returns: owned mutable step workspace; Errors: none after prevalidation.
// Determinism: fixed operand precision, allocation, and compensation order.
// Complexity: Time O(1), Space O(1) at fixed working precision.
// Notes: internal operands are overwritten each interval; not goroutine-safe.
// AI-Hints: no float64 multiplication is allowed before the drift subtraction.
func newStepWorkspace(p Parameters, compensator float64) *stepWorkspace {
	w := &stepWorkspace{
		previous: newWide(), current: newWide(), interval: newWide(),
		intensity: wide(p.Intensity), mean: newWide(), maxMean: wide(maxPoissonMean),
		drift: wide(p.Diffusion.Drift), volatility: wide(p.Diffusion.Volatility),
		scale: newWide(), ito: newWide(), shock: newWide(), normal: newWide(),
		half: wide(0.5), increment: newWide(), growth: newWide(), candidate: newWide(),
		jump: newWide(), jumpCount: newWide(), jumpScale: newWide(),
		jumpZ: newWide(), jumpMean: wide(p.LogJumpMean), jumpStd: wide(p.LogJumpStdDev),
	}
	if p.Convention == Compensated {
		compensation := newWide().Mul(wide(p.Intensity), wide(compensator))
		w.drift.Sub(w.drift, compensation)
	}
	return w
}

// candidateLogGrowth evaluates a full Merton log increment at wide precision.
// It leaves committed growth untouched until the price is validated by the caller.
//
// Implementation:
//   - Stage 1: compute sigma*sqrt(dt) and its Ito half-square correction.
//   - Stage 2: combine drift*dt, diffusion shock, and the aggregated log jump.
//   - Stage 3: add the increment to committed growth into candidate storage.
//
// Inputs: interval is positive; diffusionNormal finite; jump computed for this step.
// Returns: workspace-owned candidate log growth, overwritten on the next step.
// Errors: none; input sampling and price conversion have separate failure gates.
// Determinism: ordered wide arithmetic with no floating-point reduction races.
// Complexity: O(1) fixed-precision time/space per timestep.
// Notes: candidate may be outside the representable price domain until checked.
// AI-Hints: never update growth in place before positivePrice returns successfully.
func (w *stepWorkspace) candidateLogGrowth(diffusionNormal float64) *big.Float {
	// A diffusion scale of zero makes both the Brownian shock and Ito term
	// exactly zero. Avoid the wide square root in this deterministic regime;
	// the Poisson and aggregate-jump draws have already been performed.
	if w.volatility.Sign() == 0 {
		w.increment.Mul(w.drift, w.interval)
		w.increment.Add(w.increment, w.jump)
		return w.candidate.Add(w.growth, w.increment)
	}

	w.scale.Sqrt(w.interval)
	w.scale.Mul(w.scale, w.volatility)
	w.ito.Mul(w.scale, w.scale)
	w.ito.Mul(w.ito, w.half)
	w.normal.SetFloat64(diffusionNormal)
	w.shock.Mul(w.scale, w.normal)

	w.increment.Mul(w.drift, w.interval)
	w.increment.Sub(w.increment, w.ito)
	w.increment.Add(w.increment, w.shock)
	w.increment.Add(w.increment, w.jump)
	return w.candidate.Add(w.growth, w.increment)
}

// simulateValidated evolves a positive-intensity path over the validated grid.
// Every publication represents a completed stochastic interval, never a half-step.
//
// Implementation:
//   - Stage 1: publish the detached exact initial point and allocate one workspace.
//   - Stage 2: preflight the wide Poisson mean, draw diffusion/Poisson/jump in order.
//   - Stage 3: append time and price atomically, swapping workspaces after commit.
//
// Behavior highlights:
//   - Poisson mean>2^40 is rejected before the diffusion draw for that interval.
//   - A mathematically positive mean rounding to binary64 zero returns ErrOverflow.
//   - A canceled/rejected step retains no output, but consumed RNG draws persist.
//
// Inputs:
//   - ctx: active validated context; p: fixed validated Merton model.
//   - grid: valid strictly rising timestamps in p.Diffusion.Unit.
//   - random: single-path owned RNG; compensator: finite k or zero for Uncompensated.
//
// Returns: detached full result, or non-nil completed prefix with Complete=false.
// Errors: ErrDomain for mean>2^40, ErrOverflow for mean underflow or price
// conversion, ErrNonFinite for a malformed Normal, ErrBudget, or ctx.Err().
// Determinism: ascending grid indices; Normal(diffusion), UniformOpen, Normal(jump).
// Complexity: O(n*B) worst-case, O(n) output plus fixed-size numeric scratch.
// Notes: this kernel is entered only when Intensity>0; GBM parity uses the facade.
// An exceptionally small positive product is rejected rather than drawing no Poisson.
// AI-Hints: validate mean before any entropy draw and append only after conversion.
func simulateValidated(ctx context.Context, p Parameters, grid gbm.TimeGrid,
	random rng.RNG, compensator float64,
) (*Result, error) {
	count := len(grid.Values)
	result := &Result{
		Times:  make([]float64, 1, count),
		Values: make([]float64, 1, count),
	}
	result.Times[0] = grid.Values[0]
	result.Values[0] = p.Diffusion.Initial

	// A one-point grid has no intervals, so neither a workspace nor entropy is needed.
	if count == 1 {
		result.Complete = true
		return result, nil
	}

	work := newStepWorkspace(p, compensator)
	for index := 1; index < count; index++ {
		if err := ctx.Err(); err != nil {
			return result, err
		}

		// Wide subtraction preserves tiny positive increments even when the
		// two finite timestamps are very large and extremely close together.
		work.previous.SetFloat64(grid.Values[index-1])
		work.current.SetFloat64(grid.Values[index])
		work.interval.Sub(work.current, work.previous)

		// Preserve the exact binary64 input product until it has passed the
		// 2^40 domain gate. An ordinary float64 multiplication could round a
		// tiny positive Poisson mean to zero (silently omitting RNG draws) or
		// round a marginally excessive mean back into the accepted range.
		work.mean.Mul(work.intensity, work.interval)
		if work.mean.Cmp(work.maxMean) > 0 {
			return result, ErrDomain
		}
		mean, _ := work.mean.Float64()
		if mean == 0 {
			// The true mean remains strictly positive. Silently skipping a
			// Poisson draw here would violate the intensity>0 RNG protocol.
			return result, ErrOverflow
		}

		// Draw order is an API law: a diffusion Normal always precedes the
		// Poisson uniforms, which always precede the optional jump Normal.
		diffusionNormal := 0.0
		if p.Diffusion.Volatility > 0 {
			diffusionNormal = random.Normal()
			if !finite(diffusionNormal) {
				return result, ErrNonFinite
			}
		}

		jumpCount, err := samplePoisson(ctx, mean, random)
		if err != nil {
			return result, err
		}
		if err = ctx.Err(); err != nil {
			return result, err
		}
		if err = aggregateJump(jumpCount, random,
			work.jump, work.jumpCount, work.jumpScale, work.jumpZ,
			work.jumpMean, work.jumpStd,
		); err != nil {
			return result, err
		}
		if err = ctx.Err(); err != nil {
			return result, err
		}

		candidate := work.candidateLogGrowth(diffusionNormal)
		price, err := positivePrice(p.Diffusion.Initial, candidate)
		if err != nil {
			return result, err
		}
		if err = ctx.Err(); err != nil {
			return result, err
		}

		// Only completed time-price pairs become visible. The workspace
		// swap is likewise deferred until the candidate price is committed.
		result.Times = append(result.Times, grid.Values[index])
		result.Values = append(result.Values, price)
		work.growth, work.candidate = work.candidate, work.growth
	}

	result.Complete = true
	return result, nil
}
