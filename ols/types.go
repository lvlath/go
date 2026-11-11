// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols defines the public fit policy, immutable model, and diagnostics.
// This file fixes physical units, ownership, nil-state, and the QR rank convention.
package ols

import "github.com/lvlath/go/core"

// Options selects the regression basis and numerical rank policy.
// Its zero value fits an explicit intercept, keeps predictor scales unmodified,
// and delegates automatic numerical-rank tolerance to matrix.FactorQR.
//
// Implementation:
//   - Stage 1: the canonical Fit facade validates the numeric policy.
//   - Stage 2: the design builder applies the intercept and scaling choices.
//   - Stage 3: the QR solver applies the relative rank cutoff before publication.
//
// Behavior highlights:
//   - NoIntercept=true fixes b=+0; Standardize then uses scale-only predictors.
//   - RankTolerance=0 uses the matrix solver's dimension-scaled machine epsilon.
//
// Inputs:
//   - NoIntercept: when true, omit the column of ones and all centering.
//   - Standardize: use train population DDOF=0 standard deviations with unit constant scales.
//   - RankTolerance: finite, nonnegative, dimensionless, relative pivot cutoff.
//
// Returns:
//   - Value-only configuration snapshot consumed by Fit; no model references retained.
//
// Errors:
//   - ErrInvalidOption from Fit when RankTolerance is negative, NaN, or infinite.
//
// Determinism:
//   - No implicit option interpolation; identical field values select identical operations.
//
// Complexity:
//   - Time O(1), Space O(1) for policy storage and validation.
//
// Notes:
//   - Centering with an intercept preserves the affine model after back-conversion.
//   - Changing an Options variable after Fit cannot change the resulting Model.
//
// AI-Hints:
//   - Do not use a fitted Standardizer's centering transform under NoIntercept.
//   - RankTolerance is relative to the largest |R_ii|, not an absolute unit-scale cutoff.
type Options struct {
	// NoIntercept removes the explicit column of ones and fixes b to +0.0.
	// This also forbids centering even if Standardize is enabled.
	NoIntercept bool

	// Standardize fits DDOF=0 predictor scales from training rows only.
	// Constant columns get unit scale; QR decides whether they reduce the rank.
	Standardize bool

	// RankTolerance is a dimensionless relative QR rank cutoff.
	// Zero selects the matrix solver's default; negative and nonfinite values fail.
	RankTolerance float64
}

// Model is a fitted immutable affine regression expressed in physical units.
// Successful Fit is its only supported constructor; input matrices, y, and
// returned coefficient slices never alias the privately owned model state.
//
// Implementation:
//   - Stage 1: Fit converts QR coefficients back into original units.
//   - Stage 2: Fit retains copies of the training transform for provenance.
//   - Stage 3: Predict reads coefficients without any shared mutable workspace.
//
// Behavior highlights:
//   - Zero-value Model is untrained and is rejected by Predict.
//   - A model trained with NoIntercept always has intercept=+0.
//
// Inputs:
//   - Constructed solely by successful Fit; no caller-owned memory is retained.
//
// Returns:
//   - Coefficients supplies a detached slope vector and a copied scalar intercept.
//   - Predict supplies a detached prediction slice per call.
//
// Errors:
//   - Predict returns ErrNilInput for nil/untrained receivers.
//
// Determinism:
//   - Slopes preserve original predictor order; inference accumulates features ascending.
//
// Complexity:
//   - Persistent space O(p); Predict O(n*p) time and O(n) output space.
//
// Notes:
//   - Concurrent read-only Predict/Coefficients calls are safe with stable inputs.
//   - Fitted mean/scale vectors are provenance only; inference uses original-unit beta.
//
// AI-Hints:
//   - Never expose beta, means, or scales backing arrays through a public accessor.
//   - Do not refit scalers for heldout batches; doing so shifts the learned model.
type Model struct {
	p         int       // Number of original feature columns accepted at inference.
	beta      []float64 // Private fitted slopes, ordered as input columns.
	means     []float64 // Frozen training centers; zero in no-intercept mode.
	scales    []float64 // Frozen training population scales; one when not standardized.
	intercept float64   // Physical-unit affine offset; exact +0 when disabled.
	fitted    bool      // Distinguishes a fitted model from an allocated zero value.
}

// FitResult publishes a successfully fitted full-rank regression and its diagnostics.
// Each diagnostic refers to the training design and the model's original-unit
// prediction law; on every error Fit returns a nil pointer instead.
//
// Implementation:
//   - Stage 1: QR verifies that the accepted design has numerical rank q.
//   - Stage 2: original-unit predictions yield the finite training SSE.
//   - Stage 3: Fit publishes an immutable Model pointer and scalar diagnostics.
//
// Behavior highlights:
//   - Rank=q, where q=p for NoIntercept or p+1 with an explicit intercept.
//   - ConditionEstimate is the QR solver's finite kappa_1(R), not kappa_2(x).
//
// Inputs:
//   - Returned from Fit; callers may inspect or replace public result fields.
//
// Returns:
//   - Model: independently owned fitted object safe for concurrent read-only inference.
//   - ResidualSumSquares: unnormalized sum of squared training residuals in y units squared.
//
// Errors:
//   - No error-producing methods; any Fit error yields nil result, not partial metadata.
//
// Determinism:
//   - Rank and diagnostics reflect the same fixed QR and row-order processing.
//
// Complexity:
//   - O(1) result metadata plus the O(p) owned fitted Model.
//
// Notes:
//   - Mutating FitResult.Model does not mutate the previously returned Model itself.
//   - IsNil checks pointer presence, not whether exported fields are populated.
//
// AI-Hints:
//   - Do not interpret SSE alone as a scale-free goodness-of-fit metric.
//   - Do not infer p-values or confidence intervals without a residual covariance model.
type FitResult struct {
	// Model is the independently owned fitted regression, non-nil after success.
	// No partial Model is published on training errors or cancellation.
	Model *Model

	// Rank is the QR numerical rank of the transformed design and equals q on success.
	// It includes the explicit intercept column when NoIntercept is false.
	Rank int

	// ResidualSumSquares is sum_i (y_i - Predict(x_i))^2 in physical y units squared.
	// It is finite and nonnegative; it is never divided by n or degrees of freedom.
	ResidualSumSquares float64

	// ConditionEstimate is QR's finite 1-norm condition estimate kappa_1(R).
	// It describes the transformed/pivoted triangular factor, not the raw input x.
	ConditionEstimate float64
}

// IsNil reports whether a FitResult pointer is nil, including typed-nil interfaces.
// Presence is independent of the exported diagnostic values and Model field.
//
// Implementation:
//   - Stage 1: compare the receiver pointer with nil.
//   - Stage 2: return the boolean without dereferencing fields.
//   - Stage 3: perform no allocation, mutation, or external work.
//
// Behavior highlights:
//   - &FitResult{} is present and returns false even though it is not a fit result.
//
// Inputs:
//   - r: potentially nil result pointer, including when stored as core.Nilable.
//
// Returns:
//   - bool: true exactly when the pointer is nil.
//
// Errors:
//   - None; nil receivers are safe.
//
// Determinism:
//   - Pure pointer comparison without data access or ordering.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocations.
//
// Notes:
//   - Check Fit's error to establish whether fitting succeeded.
//
// AI-Hints:
//   - Do not substitute r.Model == nil or r.Rank == 0 for the nil-state contract.
func (r *FitResult) IsNil() bool { return r == nil }

// Compile-time verification anchors the library-wide nil-state interface.
var _ core.Nilable = (*FitResult)(nil)
