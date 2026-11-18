// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm exposes exact-in-distribution GBM simulation and thin facades.
// This file assembles caller intent and delegates all mathematics to one kernel.
package gbm

import (
	"context"

	"github.com/lvlath/go/rng"
)

// Simulate generates one GBM price path over a strictly increasing time grid.
// The conditional log increments are exact in distribution for the supplied
// normal stream; neither market calendars nor parameter fitting are involved.
//
// Implementation:
//   - Stage 1: validate context, units, input presence, time order and finiteness.
//   - Stage 2: evolve cumulative logarithmic growth through simulateValidated.
//   - Stage 3: publish only completed, detached time/price pairs.
//
// Behavior highlights:
//   - Volatility zero follows the analytic deterministic exponential and draws no RNG.
//   - Every positive-volatility step calls random.Normal exactly once in time order.
//   - A successful price may be subnormal, but never zero, NaN, or infinite.
//
// Inputs:
//   - ctx: non-nil cancellation context; pre-canceled calls return nil result.
//   - p: S0>0, finite drift, finite sigma>=0, all expressed in p.Unit.
//   - grid: finite, read-only timestamps beginning at zero, strictly ascending.
//   - random: non-nil path-owned RNG, never used concurrently by this package.
//
// Returns:
//   - *Result: detached full path, or completed prefix on runtime failure.
//   - error: nil exactly when Complete=true; validation errors return nil result.
//
// Errors:
//   - ErrInvalidOption: nil ctx, invalid enum, or mismatched parameter/grid units.
//   - ErrNilInput/ErrEmptyInput: missing stream or absent/empty time grid.
//   - ErrOrder: nonzero origin or nonincreasing timestamps.
//   - ErrNonFinite: NaN/Inf in parameters, grid, or a Normal sample.
//   - ErrDomain: nonpositive initial price or negative volatility.
//   - ErrOverflow: computed price rounds to zero or infinity in float64.
//   - ctx.Err(): canceled context, preserving errors.Is identity.
//
// Determinism:
//   - Ascending input timestamp order; no worker-dependent or global randomness.
//   - Identical floating inputs and identical Normal sequence reproduce the path.
//
// Complexity:
//   - Time O(n), returned memory O(n), fixed-precision scratch O(1).
//   - Input validation and output construction are both included in the bound.
//
// Notes:
//   - Input slices are not retained and must not be mutated during the call.
//   - An incomplete path is not a complete-horizon Monte Carlo observation.
//
// AI-Hints:
//   - Never replace anchored log evolution with S_previous*exp(increment).
//   - Do not silently rescale annual drift or volatility to grid seconds.
func Simulate(ctx context.Context, p Parameters, grid TimeGrid, random rng.RNG) (*Result, error) {
	if err := validateInputs(ctx, p, grid, random); err != nil {
		return nil, err
	}
	return simulateValidated(ctx, p, grid, random)
}

// SimulatePath constructs a GBM model and irregular grid from scalar inputs.
// It saves application code from assembling two structs while retaining an
// explicit physical unit and the complete canonical Simulate error contract.
//
// Implementation:
//   - Stage 1: assemble Parameters and TimeGrid using the same explicit unit.
//   - Stage 2: delegate all validation and lognormal work to Simulate.
//   - Stage 3: forward its full or partial result without dropping errors.
//
// Behavior highlights:
//   - No annualization, defaults, RNG allocation, or automatic interpolation.
//
// Inputs:
//   - ctx: non-nil context, checked by Simulate before other inputs.
//   - initial: finite positive starting price in caller-defined monetary units.
//   - drift: finite drift per unit; volatility: finite >=0 per sqrt(unit).
//   - unit: one of TimeYears, TimeDays, or TimeSeconds.
//   - times: read-only timestamps in unit, first exactly zero and strictly rising.
//   - random: one caller-owned RNG for this trajectory.
//
// Returns: the detached *Result and error from Simulate without modification.
// Errors: same sentinels, context identity, and error precedence as Simulate.
// Determinism: identical index order and RNG call sequence as Simulate.
// Complexity: Time O(n), output memory O(n), scratch O(1), n=len(times).
// Notes: explicit unit prevents confusing fractional seconds with model years.
// AI-Hints: this wrapper does not infer units from the magnitude of times.
func SimulatePath(ctx context.Context, initial, drift, volatility float64, unit TimeUnit,
	times []float64, random rng.RNG,
) (*Result, error) {
	return Simulate(ctx,
		Parameters{Initial: initial, Drift: drift, Volatility: volatility, Unit: unit},
		TimeGrid{Values: times, Unit: unit}, random)
}

// SimulateTerminal simulates the starting and terminal prices for one horizon.
// This minimal two-point grid is useful for European-option payoffs that do not
// depend on the intermediate path; horizon zero is a valid initial-only grid.
//
// Implementation:
//   - Stage 1: build [0,horizon] (or [0] for a zero horizon) in p.Unit.
//   - Stage 2: delegate validation, RNG consumption and arithmetic to Simulate.
//   - Stage 3: return the full Result rather than silently dropping metadata.
//
// Behavior highlights:
//   - A positive horizon consumes one Normal when Volatility>0, otherwise zero.
//   - A zero horizon consumes no RNG; negative horizon yields ErrOrder.
//
// Inputs:
//   - ctx: non-nil cancellation context.
//   - p: finite physical GBM parameters; p.Unit also defines horizon units.
//   - horizon: finite nonnegative elapsed duration in p.Unit, not a calendar date.
//   - random: separate mutable RNG stream for this logical path.
//
// Returns: canonical *Result with one or two points, or nil/partial as applicable.
// Errors: exactly those from Simulate with the constructed TimeGrid.
// Determinism: the terminal observation, if any, uses the first Normal draw.
// Complexity: Time O(1), output memory O(1), wide arithmetic scratch O(1).
// Notes: path-dependent derivatives must use Simulate on a fuller explicit grid.
// AI-Hints: risk-neutral drift for dividends is r-q, not r-q-sigma^2/2.
func SimulateTerminal(ctx context.Context, p Parameters, horizon float64, random rng.RNG) (*Result, error) {
	if horizon == 0 {
		return Simulate(ctx, p, TimeGrid{Values: []float64{0}, Unit: p.Unit}, random)
	}
	return Simulate(ctx, p, TimeGrid{Values: []float64{0, horizon}, Unit: p.Unit}, random)
}

// SimulateSeeded creates an independent deterministic stream for a logical path.
// It is a convenience for replayable stand-alone scenarios; Monte Carlo workers
// should pass the already-created path stream into the canonical Simulate instead.
//
// Implementation:
//   - Stage 1: create a fresh RNG from explicit seed and logical pathID.
//   - Stage 2: delegate all input checks and trajectory arithmetic to Simulate.
//   - Stage 3: return the identical canonical Result and error protocol.
//
// Behavior highlights:
//   - Identical seed/pathID and inputs replay the same path on one RNG revision.
//   - The factory uses no global RNG, system clock, or worker-dependent counter.
//
// Inputs:
//   - ctx: non-nil, checked by Simulate before reading other input fields.
//   - p: validated physical parameters and unit (via Simulate).
//   - grid: read-only irregular time grid whose Unit must equal p.Unit.
//   - seed: full uint64 experiment seed; pathID: full uint64 logical path key.
//
// Returns: complete *Result or permitted partial prefix, plus original error.
// Errors: same classification and precedence as Simulate.
// Determinism: rng.NewStream(seed,pathID) then ascending timestamp traversal.
// Complexity: Time O(n), output memory O(n), stream setup O(1).
// Notes: replay identity also depends on the version of the rng package.
// AI-Hints: never derive pathID from a goroutine number or scheduler order.
func SimulateSeeded(ctx context.Context, p Parameters, grid TimeGrid, seed, pathID uint64) (*Result, error) {
	return Simulate(ctx, p, grid, rng.NewStream(seed, pathID))
}
