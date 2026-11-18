// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion exposes the canonical simulation facade and thin wrappers.
// This file owns user-facing composition; stochastic mathematics lives in kernels.
package jumpdiffusion

import (
	"context"
	"errors"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/rng"
)

// Simulate generates one Merton compound-Poisson lognormal jump-diffusion path.
// It preserves GBM's bitwise path and exact draw sequence when Intensity is zero.
//
// Implementation:
//   - Stage 1: validate context, unit, RNG, time order, and all coefficients.
//   - Stage 2: dispatch zero-intensity paths to GBM or calculate the compensator.
//   - Stage 3: run the canonical jump kernel and publish detached completed points.
//
// Behavior highlights:
//   - Compensated drift removes Intensity*expm1(m+s^2/2), not Intensity*m.
//   - Positive-intensity steps draw diffusion Normal, Poisson uniforms, then jump Normal.
//   - No intrastep jump event times or Gaussian Poisson approximations are produced.
//
// Inputs:
//   - ctx: non-nil active context; cancellation is checked during rejection loops.
//   - p: GBM Initial>0, finite coefficients, nonnegative vol/intensity/stddev.
//   - grid: read-only finite times starting at zero, strictly increasing in p.Diffusion.Unit.
//   - random: non-nil single-owner RNG; concurrent access to one stream is unsupported.
//
// Returns:
//   - *Result: detached full path or completed prefix on runtime error.
//   - error: nil exactly on Complete=true; preflight errors return nil Result.
//
// Errors:
//   - ErrInvalidOption: nil ctx, invalid convention/unit, or unit mismatch.
//   - ErrNilInput/ErrEmptyInput/ErrOrder: RNG or time grid rejected in preflight.
//   - ErrNonFinite/ErrDomain: nonfinite or out-of-domain input/draw/count.
//   - ErrOverflow: compensator or computed price cannot be represented.
//   - ErrBudget: Poisson attempt exhaustion; ctx.Err(): cancellation.
//   - GBM dependency errors are preserved through errors.Is for zero intensity.
//
// Determinism:
//   - Strict ascending grid index; per step Normal(diffusion), Poisson, Normal(jump).
//   - The zero-intensity branch delegates all arithmetic and draws to gbm.Simulate.
//
// Complexity:
//   - Time O(n*B) worst-case, Space O(n) output plus O(1) fixed-precision scratch.
//   - B=2^20 attempts per interval; actual expected work is much lower.
//
// Notes:
//   - No input is retained; successful prices are strictly positive finite float64.
//   - A runtime error does not restore the RNG position of the abandoned step.
//
// AI-Hints:
//   - Do not rebuild the GBM lambda-zero numerical path independently.
//   - Never feed an incomplete path into a full-horizon montecarlo.Sampler.
func Simulate(ctx context.Context, p Parameters, grid gbm.TimeGrid, random rng.RNG) (*Result, error) {
	if err := validateInputs(ctx, p, grid, random); err != nil {
		return nil, err
	}

	// Delegation is essential: GBM's high-precision arithmetic and RNG spare
	// semantics must remain byte-for-byte identical at zero jump intensity.
	if p.Intensity == 0 {
		path, err := gbm.Simulate(ctx, p.Diffusion, grid, random)
		if err != nil {
			switch {
			case errors.Is(err, gbm.ErrOverflow):
				err = errors.Join(ErrOverflow, err)
			case errors.Is(err, gbm.ErrNonFinite):
				err = errors.Join(ErrNonFinite, err)
			}
		}
		if path == nil {
			return nil, err
		}
		return &Result{Times: path.Times, Values: path.Values, Complete: path.Complete}, err
	}

	// Only compensated models require a representable jump mean multiplier;
	// computing it for lambda=0 or Uncompensated would introduce false errors.
	compensator := 0.0
	if p.Convention == Compensated {
		var err error
		compensator, err = jumpCompensator(p.LogJumpMean, p.LogJumpStdDev)
		if err != nil {
			return nil, err
		}
	}
	return simulateValidated(ctx, p, grid, random, compensator)
}

// SimulatePath constructs a grid in the model's explicit unit and simulates it.
// It removes only TimeGrid boilerplate without changing error or RNG semantics.
//
// Implementation:
//   - Stage 1: associate read-only timestamps with p.Diffusion.Unit.
//   - Stage 2: delegate all validation and computation to Simulate.
//   - Stage 3: return the identical full/partial result and error unchanged.
//
// Behavior highlights: no calendar, 252-session, or year-to-second conversion.
// Inputs:
//   - ctx: non-nil context; p: complete physical and jump parameters.
//   - times: finite strictly rising elapsed times starting at zero, in model unit.
//   - random: one path-owned RNG implementing rng.RNG.
//
// Returns: canonical *Result and error without lost completion metadata.
// Errors: exactly the Simulate categories and precedence for the constructed grid.
// Determinism: ascending times, identical RNG calls as Simulate.
// Complexity: Time O(n*B), Space O(n), n=len(times), B=2^20.
// Notes: times is borrowed only for the duration of Simulate; output is detached.
// AI-Hints: if the supplied times use a different unit, convert explicitly first.
func SimulatePath(ctx context.Context, p Parameters, times []float64, random rng.RNG) (*Result, error) {
	return Simulate(ctx, p, gbm.TimeGrid{Values: times, Unit: p.Diffusion.Unit}, random)
}

// SimulateTerminal returns the initial and terminal observations of one horizon.
// It avoids generating an unneeded dense path for terminal-only financial payoffs.
//
// Implementation:
//   - Stage 1: construct [0] if horizon=0, otherwise [0,horizon].
//   - Stage 2: validate and generate through the canonical Simulate facade.
//   - Stage 3: return the complete or incomplete Result without dropping errors.
//
// Behavior highlights:
//   - Horizon zero produces the initial price and zero draws.
//   - Positive horizons have exactly one jump-count draw sequence, not one per day.
//
// Inputs:
//   - ctx: non-nil context; p: all Merton parameters with an explicit time unit.
//   - horizon: finite nonnegative elapsed duration expressed in p.Diffusion.Unit.
//   - random: path-owned RNG, never reused concurrently.
//
// Returns: canonical *Result of length one or two, or a partial prefix on failure.
// Errors: Simulate sentinels; negative horizon is ErrOrder, nonfinite ErrNonFinite.
// Determinism: the first interval uses the first appropriate source draws.
// Complexity: O(B) worst-case time, O(1) output/scratch, B=2^20.
// Notes: path-dependent barriers and drawdowns require a denser explicit grid.
// AI-Hints: a two-point path misses intrahorizon barrier crossings by design.
func SimulateTerminal(ctx context.Context, p Parameters, horizon float64, random rng.RNG) (*Result, error) {
	if horizon == 0 {
		return Simulate(ctx, p, gbm.TimeGrid{Values: []float64{0}, Unit: p.Diffusion.Unit}, random)
	}
	return Simulate(ctx, p, gbm.TimeGrid{Values: []float64{0, horizon}, Unit: p.Diffusion.Unit}, random)
}

// SimulateSeeded creates a deterministic independent stream for one logical path.
// The wrapper makes reproducible scenarios convenient without any global RNG.
//
// Implementation:
//   - Stage 1: derive a fresh rng.NewStream from explicit seed and pathID.
//   - Stage 2: delegate validation, draw sequence, and arithmetic to Simulate.
//   - Stage 3: return its complete or partial Result and unchanged error.
//
// Behavior highlights: identical seed/pathID and RNG version replay the same path.
// Inputs:
//   - ctx: non-nil context; p/grid: same model and unit constraints as Simulate.
//   - seed: full uint64 experiment ID; pathID: full uint64 logical trajectory ID.
//
// Returns: canonical *Result/error without retaining the created RNG.
// Errors: exactly those of Simulate, including wrapped GBM roots and cancellation.
// Determinism: rng.NewStream(seed,pathID), then canonical ascending step order.
// Complexity: Time O(n*B), Space O(n) plus O(1) independent stream state.
// Notes: pathID must be independent of worker scheduling to preserve replay.
// AI-Hints: never use a goroutine counter or wall clock as a hidden path ID.
func SimulateSeeded(ctx context.Context, p Parameters, grid gbm.TimeGrid, seed, pathID uint64) (*Result, error) {
	return Simulate(ctx, p, grid, rng.NewStream(seed, pathID))
}

// SimulateTerminalSeeded simulates a single expiry using a new reproducible stream.
// It combines explicit logical-path seeding with the minimal terminal grid.
//
// Implementation:
//   - Stage 1: construct a fresh stream from the complete seed and pathID.
//   - Stage 2: delegate horizon semantics and stochastic work to SimulateTerminal.
//   - Stage 3: return its canonical full/partial Result and original error.
//
// Behavior highlights:
//   - A zero horizon publishes only the initial price and consumes no draws.
//   - There is no hidden global seed, calendar conversion, or worker identity.
//
// Inputs:
//   - ctx: active non-nil context; p: immutable Merton coefficients and unit.
//   - horizon: nonnegative finite elapsed duration in p.Diffusion.Unit.
//   - seed: experiment identifier; pathID: scheduler-independent scenario ID.
//
// Returns: detached Result and error exactly as SimulateTerminal publishes them.
// Errors: identical sentinel identity and precedence to the canonical facade.
// Determinism: rng.NewStream(seed,pathID), then the one-interval draw order.
// Complexity: O(B) worst-case work and O(1) output/scratch, B=2^20.
// Notes: terminal-only paths do not resolve intrahorizon barrier events.
// AI-Hints: keep Monte Carlo workers' supplied RNG in SimulateTerminal rather
// than recreating a seed-derived stream within montecarlo.Sampler.Sample.
func SimulateTerminalSeeded(ctx context.Context, p Parameters, horizon float64, seed, pathID uint64) (*Result, error) {
	return SimulateTerminal(ctx, p, horizon, rng.NewStream(seed, pathID))
}
