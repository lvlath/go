// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion defines the model's immutable parameters and path artifact.
// This file fixes drift policy, units, output ownership, and nil-result semantics.
package jumpdiffusion

import (
	"github.com/lvlath/go/core"
	"github.com/lvlath/go/gbm"
)

// DriftConvention selects the meaning of the GBM proportional drift parameter.
// It determines whether the mean relative jump is subtracted from log growth.
//
// Implementation:
//   - Stage 1: validate the closed enum before simulation.
//   - Stage 2: choose whether to subtract intensity*E[exp(Y)-1].
//   - Stage 3: preserve the selected law throughout the complete trajectory.
//
// Behavior highlights: the zero-value policy is Compensated, not automatic fitting.
// Inputs: only Compensated or Uncompensated are supported values.
// Returns: output expectation follows the selected documented convention.
// Errors: ErrInvalidOption for any unsupported value before RNG use.
// Determinism: pure enum comparison; Complexity: Time O(1), Space O(1).
// Notes: changing the convention changes the stochastic model, not its RNG.
// AI-Hints: the jump mean m is not the arithmetic compensator k.
type DriftConvention uint8

const (
	// Compensated subtracts intensity*E[exp(Y)-1] from proportional drift.
	// Provided the model expectation exists, E[S(t)]=Initial*exp(Drift*t).
	Compensated DriftConvention = iota

	// Uncompensated keeps GBM's proportional drift without a jump correction.
	// The expectation becomes Initial*exp((Drift+Intensity*k)*t).
	Uncompensated
)

// Parameters fixes the coefficients for one Merton lognormal jump-diffusion.
// Every field is interpreted in Diffusion.Unit without implicit rescaling.
//
// Implementation:
//   - Stage 1: validate GBM domains, jump moments, intensity, and drift policy.
//   - Stage 2: use the fixed coefficients in independent grid increments.
//   - Stage 3: publish a result without retaining or modifying Parameters.
//
// Behavior highlights:
//   - Intensity=0 delegates directly to GBM after validating both jump moments.
//   - Standard deviations of zero suppress only their corresponding Normal draws.
//
// Inputs: Diffusion.Initial>0; Volatility, Intensity, LogJumpStdDev >=0; all finite.
// Returns: parameter values are copied, never fitted or mutated by Simulate.
// Errors: ErrInvalidOption, ErrNonFinite, ErrDomain, or ErrOverflow as documented.
// Determinism: immutable coefficients per call; Complexity: O(1) copy and storage.
// Notes: jump magnitudes describe model assumptions, not calibrated market tails.
// AI-Hints: intensity is in inverse Diffusion.Unit; never reinterpret seconds as years.
type Parameters struct {
	// Diffusion fixes the strictly positive starting price, proportional drift,
	// nonnegative Brownian volatility, and physical time unit.
	Diffusion gbm.Parameters

	// Intensity is the finite nonnegative expected jump count per Diffusion.Unit.
	// Each grid interval has Poisson mean Intensity * elapsed time.
	Intensity float64

	// LogJumpMean is E[log(S_after/S_before)] for one jump, dimensionless.
	// It is not the arithmetic mean percentage jump and may be negative.
	LogJumpMean float64

	// LogJumpStdDev is the finite nonnegative dimensionless standard deviation
	// of the normal log jump distribution, not a price percentage standard deviation.
	LogJumpStdDev float64

	// Convention explicitly selects compensated or uncompensated drift.
	// The zero value Compensated preserves E[S(t)]=Initial*exp(Drift*t).
	Convention DriftConvention
}

// Result contains the independently owned completed grid observations of one path.
// A failed runtime step cannot expose its candidate timestamp or price.
//
// Implementation:
//   - Stage 1: publish the input grid's zero timestamp and exact initial price.
//   - Stage 2: append validated positive finite prices in ascending grid order.
//   - Stage 3: set Complete only after all requested intervals have succeeded.
//
// Behavior highlights:
//   - Times and Values have equal positive lengths whenever Result is returned.
//   - All Values are strictly positive and finite, including legal subnormals.
//
// Inputs: Simulate creates the result; caller owns both exported slices afterwards.
// Returns: a full path or the completed prefix [0,k] with Complete=false on error.
// Errors: errors are returned by Simulate, never stored inside Result.
// Determinism: original grid index order, with no sorting or interpolation.
// Complexity: Time O(n) and memory O(n) for n published observations.
// Notes: RNG draws from an abandoned step are not rolled back; no slice aliases grid.
// AI-Hints: never use Values[len(Values)-1] as a full-horizon sample if !Complete.
type Result struct {
	// Times contains detached exact input timestamps for completed observations.
	// No partially computed future timestamp is published on runtime failure.
	Times []float64

	// Values contains the corresponding detached positive finite price observations.
	// Values[0] is always the original Diffusion.Initial on a non-nil result.
	Values []float64

	// Complete is true exactly when the entire grid succeeded with no error.
	// An allocated partial Result can still have a single valid initial point.
	Complete bool
}

// IsNil reports whether the Result pointer itself is nil.
// It implements core.Nilable without reading any mutable trajectory fields.
//
// Implementation:
//   - Stage 1: compare the receiver pointer with nil.
//   - Stage 2: return the identity result without dereferencing the receiver.
//   - Stage 3: leave both result slices and completion metadata untouched.
//
// Behavior highlights: a non-nil incomplete or empty Result returns false.
// Inputs: receiver may be an ordinary or typed-nil *Result.
// Returns: true for a nil receiver only; Errors: none.
// Determinism: pointer identity alone; Complexity: Time O(1), Space O(1).
// Notes: nilability is distinct from simulation error and completion.
// AI-Hints: do not substitute IsNil for Complete or an explicit err check.
func (r *Result) IsNil() bool {
	return r == nil
}

var _ core.Nilable = (*Result)(nil)
