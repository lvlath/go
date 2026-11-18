// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm implements the single exact-in-distribution trajectory kernel.
// This file governs step order, cancellation, and atomic prefix publication.
package gbm

import (
	"context"
	"github.com/lvlath/go/rng"
)

// simulateValidated evolves cumulative logarithmic growth on the requested grid.
// The caller-facing price is derived from the exact starting float64 and this
// growth, preventing avoidable intermediate over/underflow or log(S0) rounding.
//
// Implementation:
//   - Stage 1: publish detached initial point and create local wide log state.
//   - Stage 2: check context, draw at most one Normal, and evaluate each increment.
//   - Stage 3: append time and value together only after successful conversion.
//
// Behavior highlights:
//   - Zero-volatility trajectories use the analytic mu*t endpoint at every time.
//   - A failed step never appends its time or value; no RNG rollback is promised.
//
// Inputs:
//   - ctx: active context, validated before entering this private kernel.
//   - p: finite, valid parameters with units matching grid.Unit.
//   - grid: validated strictly increasing, read-only, nonempty time array.
//   - random: caller-owned non-nil stream; not shared across simultaneous paths.
//
// Returns: detached full Result, or completed prefix and a runtime error.
// Errors: ctx.Err(), ErrNonFinite for a bad draw, ErrOverflow for a bad price.
// Determinism: strictly ascending time index; one Normal per stochastic step.
// Complexity: Time O(n), output space O(n), fixed-size numeric scratch O(1).
// Notes: complete prefixes include time zero; no synchronization protects random.
// AI-Hints: do not update published slices until the new price passes validation.
func simulateValidated(ctx context.Context, p Parameters, grid TimeGrid, random rng.RNG) (*Result, error) {
	count := len(grid.Values)
	result := &Result{
		Times:  make([]float64, 1, count),
		Values: make([]float64, 1, count),
	}
	result.Times[0] = grid.Values[0]
	result.Values[0] = p.Initial

	// A valid one-point grid represents the current price without evolution,
	// and must not consume RNG entropy even if volatility is positive.
	if count == 1 {
		result.Complete = true
		return result, nil
	}

	// The log-growth state is ln(S(t)/S0). Together with the unchanged
	// initial anchor it is algebraically the same log-price process as
	// ln(S(t))=ln(S0)+growth, without approximating ln(S0) in float64.
	growth := newWide()
	candidateGrowth := newWide()
	work := newStepArithmetic(p.Drift, p.Volatility)
	for index := 1; index < count; index++ {
		// Cancellation is checked before starting each step; cancellation
		// inside a Normal call can be observed at the next step boundary.
		if err := ctx.Err(); err != nil {
			return result, err
		}

		z := 0.0
		if p.Volatility > 0 {
			z = random.Normal()
			if !finite(z) {
				return result, ErrNonFinite
			}
		}

		if p.Volatility == 0 {
			// Independent analytic endpoints eliminate cumulative summation
			// drift and preserve zero random-number consumption. The same
			// preallocated buffer is overwritten for each endpoint.
			work.current.SetFloat64(grid.Values[index])
			candidateGrowth.Mul(work.mu, work.current)
		} else {
			// Compute the positive timestamp difference, Ito correction,
			// and shock in a reusable workspace. Accumulate the candidate
			// without modifying the committed growth state.
			increment := work.logIncrement(grid.Values[index-1], grid.Values[index], z)
			candidateGrowth.Add(growth, increment)
		}

		// Publication is atomic at the pair level: failed conversion cannot
		// leak the unfinished timestamp, and existing values are untouched.
		price, err := priceFromGrowth(p.Initial, candidateGrowth)
		if err != nil {
			return result, err
		}
		result.Times = append(result.Times, grid.Values[index])
		result.Values = append(result.Values, price)

		// Ping-pong the two owned buffers only after committing the point;
		// no per-step growth allocation or alias survives to the next step.
		growth, candidateGrowth = candidateGrowth, growth
	}
	result.Complete = true
	return result, nil
}
