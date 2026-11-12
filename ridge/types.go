// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge defines the immutable fitted coefficients and public fit result.
// All numerical diagnostics use the same explicitly normalized objective.
package ridge

import "github.com/lvlath/go/core"

// Options selects the regularization strength, hypothesis space, and rank policy.
// It is a value-only configuration: the zero value is intentionally invalid.
//
// Implementation:
//   - Stage 1: Fit validates both numeric options before any matrix access.
//   - Stage 2: the design kernel interprets intercept and standardization choices.
//   - Stage 3: the equilibrated augmented QR resolves the relative rank tolerance.
//
// Behavior highlights:
//   - Lambda>0 is required; with Standardize, Lambda penalizes scaled slopes.
//   - NoIntercept=true enforces a physical zero intercept and scale-only transforms.
//   - RankTolerance=0 uses the matrix solver's dimension-scaled machine epsilon.
//   - This rank cutoff applies after deterministic row permutation and positive
//     column equilibration, not to raw original-feature column magnitudes.
//
// Inputs:
//   - Lambda: finite positive L2 penalty strength in the chosen feature coordinates.
//   - NoIntercept: exclude the unpenalized intercept column and any centering.
//   - Standardize: fit population DDOF=0 predictor scales from training data.
//   - RankTolerance: finite nonnegative relative QR rank cutoff; zero means automatic.
//
// Returns:
//   - Value-only policy copied by Fit; no external slices or resources are retained.
//
// Errors:
//   - Fit returns ErrInvalidOption for invalid Lambda or RankTolerance.
//
// Determinism:
//   - Fixed fields select one basis and one solver path, without automatic retries.
//
// Complexity:
//   - Time O(1), Space O(1) for validation and copying.
//
// Notes:
//   - Changing Options after Fit cannot mutate an already fitted Model.
//
// AI-Hints:
//   - Lambda=0 is not a request for OLS; use the separate ols package.
//   - Penalty units change with Standardize; changing that flag changes the objective.
//
// The exported field order remains stable for source compatibility.
// Use keyed struct literals when supplying Options.
type Options struct {
	// Lambda is the strictly positive, finite regularization coefficient.
	// Its squared-slope term is Lambda/2 times the optimization-basis norm.
	Lambda float64

	// NoIntercept constrains b=+0 in original physical coordinates.
	// Standardize then uses scales only; feature means are never subtracted.
	NoIntercept bool

	// Standardize uses train population (DDOF=0) feature scales.
	// Exact-constant features retain scale=1 under stats.UnitScaleConstant.
	Standardize bool

	// RankTolerance selects a finite nonnegative relative QR diagonal cutoff.
	// The cutoff is applied to the equilibrated slope-first augmented system.
	// Zero delegates the automatic tolerance to matrix.SolveLeastSquares.
	RankTolerance float64
}

// Model retains a fitted physical-unit ridge predictor and its training transform.
// Fit is its sole supported constructor; the zero value is not a fitted model.
//
// Implementation:
//   - Stage 1: undo QR column equilibration and recover original-unit beta and b.
//   - Stage 2: detach and preserve training means/scales for provenance.
//   - Stage 3: Predict reads only frozen fields, allocating workspace per invocation.
//
// Behavior highlights:
//   - Inference uses physical beta and b, without reapplying the scaling transform.
//   - Concurrent read-only Predict and Coefficients calls on a fitted Model are safe.
//
// Inputs:
//   - Produced only by successful Fit; no public construction or setters exist.
//
// Returns:
//   - Coefficients returns caller-owned slope copies; Predict returns fresh output slices.
//
// Errors:
//   - Predict rejects nil/unfitted receivers with ErrNilInput.
//
// Determinism:
//   - Predictor order is the original feature column order; no maps or randomization.
//
// Complexity:
//   - Persistent O(p) coefficient and training-transform storage.
//
// Notes:
//   - Caller mutation of input matrices during inference is unsupported.
//   - FitResult.Model is an exported pointer, but Model has no mutable exported state.
//
// AI-Hints:
//   - Never expose beta, means, or scales by returning their backing slices.
//   - Do not refit a Standardizer on future data before calling Predict.
type Model struct {
	p         int       // Expected physical feature width for all future batches.
	beta      []float64 // Immutable original-unit slopes in original column order.
	intercept float64   // Physical-unit affine offset; positive zero if disabled.
	means     []float64 // Frozen train centers; all zeros when NoIntercept=true.
	scales    []float64 // Frozen positive population scales; ones when unstandardized.
	fitted    bool      // Distinguishes a completed fit from the zero-value Model.
}

// FitResult publishes one successful ridge fit and its checked objective terms.
// It is present only after every coefficient and diagnostic is finite.
//
// Implementation:
//   - Stage 1: the QR solver accepts the loss-equivalent equilibrated design.
//   - Stage 2: Fit verifies residual energy and regularization in their own units.
//   - Stage 3: atomically returns the fitted Model and all three scalar diagnostics.
//
// Behavior highlights:
//   - ResidualSumSquares is unnormalized SSE in original target units squared.
//   - Penalty is Lambda/2*||theta||^2 for optimization slopes, excluding intercept.
//   - Objective is ResidualSumSquares/(2*n)+Penalty with the original train n.
//
// Inputs:
//   - Returned by Fit after complete verification; no partial success states exist.
//
// Returns:
//   - Public Model pointer and scalar metadata; no external training input aliases.
//
// Errors:
//   - Not returned with any Fit error; callers inspect a nil pointer on failure.
//
// Determinism:
//   - Objective parts derive from one accepted QR solution and fixed sample ordering.
//
// Complexity:
//   - O(1) result metadata; independently owned Model retains O(p) data.
//
// Notes:
//   - Callers may mutate FitResult scalar fields without changing Model predictions.
//   - IsNil checks pointer presence, not whether the exported fields are populated.
//
// AI-Hints:
//   - Do not compute Penalty from physical beta when Standardize=true.
//   - Do not interpret a small train Objective as proof of out-of-sample performance.
type FitResult struct {
	// Model is the immutable fitted regression, non-nil after successful Fit.
	// Its private parameters are not aliased with the training inputs.
	Model *Model

	// ResidualSumSquares is sum_i(y_i-Predict(x_i))^2, before division by n.
	// It is a finite nonnegative scalar in squared original response units.
	ResidualSumSquares float64

	// Penalty is the unpenalized-intercept L2 term Lambda*sum_j(theta_j^2)/2.
	// Theta is in the training optimization basis, not necessarily physical beta.
	Penalty float64

	// Objective is ResidualSumSquares/(2*n)+Penalty with the fitted sample count.
	// It is finite, nonnegative, and never recomputed after publication.
	Objective float64
}

// IsNil classifies a nil FitResult pointer, including typed-nil interfaces.
// An allocated zero-value FitResult is present even before it contains a model.
//
// Implementation:
//   - Stage 1: compare the receiver pointer with nil.
//   - Stage 2: return the comparison without dereferencing fields.
//   - Stage 3: allocate no memory and mutate no state.
//
// Behavior highlights:
//   - Safe when *FitResult(nil) is stored as core.Nilable.
//
// Inputs:
//   - r: possibly nil receiver, including an interface-held typed-nil pointer.
//
// Returns:
//   - bool: true exactly for a nil pointer receiver.
//
// Errors:
//   - None; this method is safe on a nil receiver.
//
// Determinism:
//   - Pure pointer comparison without iteration or shared state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The method deliberately does not inspect FitResult.Model or diagnostics.
//
// AI-Hints:
//   - Do not replace this with reflection-based interface nil detection.
func (r *FitResult) IsNil() bool { return r == nil }

// Compile-time evidence of the library-wide nil-state interface contract.
var _ core.Nilable = (*FitResult)(nil)
