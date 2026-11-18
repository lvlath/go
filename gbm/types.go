// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm defines the physical unit, inputs, and detached trajectory result.
// All exported data shapes are fixed here; no hidden model or stream is retained.
package gbm

import "github.com/lvlath/go/core"

// TimeUnit identifies the time dimension shared by drift, volatility, and grid.
// It selects an interpretation of numeric timestamps, not a conversion formula.
//
// Implementation:
//   - Stage 1: select an explicit enumerated unit in Parameters and TimeGrid.
//   - Stage 2: Simulate rejects invalid enum values or mismatched units.
//   - Stage 3: return timestamps in precisely the supplied unit.
//
// Behavior highlights:
//   - Drift is inverse-unit; volatility is inverse-square-root-unit.
//
// Inputs: use only TimeYears, TimeDays, or TimeSeconds.
// Returns: output times preserve the caller's exact input representation.
// Errors: invalid/mismatched values cause ErrInvalidOption in Simulate.
// Determinism: strict equality of enum values, with no unit inference.
// Complexity: Time O(1), Space O(1).
// Notes: elapsed days are not exchange sessions or business days.
// AI-Hints: 252-session annualization is always an external modeling decision.
type TimeUnit uint8

const (
	// TimeYears interprets one timestamp unit as one elapsed model year.
	// The caller, not gbm, defines how calendar dates map into year fractions.
	TimeYears TimeUnit = iota

	// TimeDays interprets one unit as an elapsed day (not a trading session).
	// No 252- or 365-day conversion is applied by Simulate.
	TimeDays

	// TimeSeconds interprets one unit as one elapsed second.
	// Fractional seconds express irregular subsecond ticks without interpolation.
	TimeSeconds
)

// TimeGrid declares the exact observation schedule for a GBM trajectory.
// Values must be finite, nonempty, begin at zero, and increase strictly.
//
// Implementation:
//   - Stage 1: Simulate checks unit, presence and timestamp order.
//   - Stage 2: each consecutive pair defines one positive stochastic interval.
//   - Stage 3: completed timestamps are copied to the output Result.
//
// Behavior highlights: [0] is valid and never consumes a Gaussian sample.
// Inputs: Values are read-only float64 timestamps in Unit for the full call.
// Returns: Result.Times is a detached prefix or full copy of Values.
// Errors: ErrNilInput, ErrEmptyInput, ErrOrder, ErrNonFinite, ErrInvalidOption.
// Determinism: original ascending slice order, never sorted or deduplicated.
// Complexity: validation O(n), output time ownership O(n).
// Notes: concurrent mutation by the caller is unsupported.
// AI-Hints: a Unix timestamp must be rebased to elapsed time starting at zero.
type TimeGrid struct {
	// Values holds elapsed observation times. Values[0] must equal +/-0;
	// every later time must be finite and strictly greater than the previous.
	Values []float64

	// Unit gives the physical meaning of Values and must match Parameters.Unit.
	Unit TimeUnit
}

// Parameters contains the physical coefficients and starting state of GBM.
// They are caller-supplied fixed model assumptions, not fitted market data.
//
// Implementation:
//   - Stage 1: validate finite initial/drift/volatility and their domains.
//   - Stage 2: apply each coefficient to time increments in its declared unit.
//   - Stage 3: retain the initial value as the exact price-conversion anchor.
//
// Behavior highlights: sigma=0 means deterministic S0*exp(mu*t), zero RNG calls.
// Inputs: Initial>0, Volatility>=0, Drift any finite float64, valid Unit.
// Returns: coefficients are read by value; no parameter state is retained.
// Errors: ErrNonFinite, ErrDomain, ErrInvalidOption on invalid inputs.
// Determinism: all fields are fixed throughout one simulation.
// Complexity: copying and validation O(1).
// Notes: drift alone does not assert risk-neutral or real-world measure.
// AI-Hints: supply r-q for risk-neutral dividend-paying share-price dynamics.
type Parameters struct {
	// Initial is the strictly positive, finite starting price S0.
	Initial float64

	// Drift is mu, the finite proportional drift per one Unit of elapsed time.
	Drift float64

	// Volatility is sigma, the finite nonnegative diffusion coefficient per sqrt(Unit).
	Volatility float64

	// Unit defines the measurement scale for both Drift and Volatility.
	// It must exactly equal the associated TimeGrid.Unit value.
	Unit TimeUnit
}

// Result describes the completed observation prefix of one GBM trajectory.
// Both slices are caller-owned and detached from the input grid and RNG state.
//
// Implementation:
//   - Stage 1: publish time zero and the exact initial price.
//   - Stage 2: append time/value pairs only after a successful numeric step.
//   - Stage 3: set Complete only after every requested step is committed.
//
// Behavior highlights:
//   - Times and Values always have equal length and share the same indices.
//   - All published prices are positive and finite, including legal subnormals.
//   - On runtime failure only completed indices [0,k] are meaningful.
//
// Inputs: created by Simulate; callers may subsequently mutate returned slices.
// Returns: IsNil classifies receiver absence independently of completeness.
// Errors: Simulate returns the failure error alongside an incomplete Result.
// Determinism: output order equals the original time-grid order.
// Complexity: owned storage O(n) and IsNil O(1).
// Notes: a partial result must not be treated as a full-horizon MC sample.
// AI-Hints: always check err and Complete before using Values[len-1] as terminal.
type Result struct {
	// Times is the detached array of completed timestamps, beginning at zero.
	Times []float64

	// Values is the detached, strictly positive finite price at each Times index.
	Values []float64

	// Complete is true only when every grid point succeeded; false denotes
	// an interrupted or numerically failed computation with a valid prefix.
	Complete bool
}

// IsNil reports whether the Result receiver pointer is absent.
// The method is safe on a typed-nil *Result stored behind core.Nilable.
//
// Implementation:
//   - Stage 1: compare the receiver pointer to nil without dereferencing it.
//   - Stage 2: return the identity boolean without reading mutable fields.
//   - Stage 3: perform no allocation, lock acquisition, or state mutation.
//
// Behavior highlights: a non-nil incomplete or empty Result still returns false.
// Inputs: the receiver may be nil or non-nil.
// Returns: true only for a nil *Result pointer.
// Errors: none; nil receivers are expressly valid for this identity operation.
// Determinism: identity comparison only, independent of timestamps or workers.
// Complexity: Time O(1), Space O(1).
// Notes: check Complete and the returned error separately from IsNil.
// AI-Hints: never use IsNil as a proxy for successful stochastic simulation.
func (r *Result) IsNil() bool {
	return r == nil
}

var _ core.Nilable = (*Result)(nil)
