// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca declares numerical policies, privately owned fitted models, and
// detached result diagnostics for covariance-based principal component analysis.
package pca

import "github.com/lvlath/go/core"

// Solver identifies the requested factorization algorithm at Fit admission.
// It prevents callers from mistaking covariance diagonalization for direct SVD.
//
// Implementation:
//   - Stage 1: Fit validates the exact enum value before reading observations.
//   - Stage 2: CovarianceEigen delegates to matrix.Covariance and matrix.EigenSym.
//   - Stage 3: unsupported selections fail without returning a fitted Model.
//
// Behavior highlights:
//   - The zero value selects the implemented reference covariance eigensolver.
//   - ThinSVD is not currently executable and never falls back to covariance.
//
// Inputs:
//   - Solver is selected through Options.Solver; no string-based parsing occurs.
//
// Returns:
//   - The enum selects Fit behavior; no direct Solver method produces a value.
//
// Errors:
//   - ErrUnsupportedSolver for ThinSVD; ErrInvalidOption for unrecognized values.
//
// Determinism:
//   - Equality is an exact uint8 comparison; there is no automatic solver dispatch.
//
// Complexity:
//   - Time O(1), space O(1) for solver selection.
//
// Notes:
//   - Covariance squaring of the condition number limits reliable rank detection.
//
// AI-Hints:
//   - Do not implement ThinSVD by forwarding it to CovarianceEigen.
//   - Choose a direct SVD library separately for strongly ill-conditioned data.
type Solver uint8

const (
	// CovarianceEigen selects symmetric eigen-decomposition of sample covariance.
	// The solver uses deterministic Jacobi pivots with an explicit rotation limit.
	// It is the default because its enum value is zero; rank is eigenvalue-relative.
	CovarianceEigen Solver = iota

	// ThinSVD denotes an unavailable direct thin singular-value decomposition.
	// Fit returns ErrUnsupportedSolver immediately without accessing observations.
	// This constant is reserved for compatibility and cannot currently train a model.
	ThinSVD
)

// Options defines the complete training, numerical-rank, and solver policy.
// Its value is read at Fit admission and never retained as mutable caller state.
//
// Implementation:
//   - Stage 1: validate enum, nonnegative counts, and finite nonnegative tolerances.
//   - Stage 2: resolve zero defaults for rank cutoff, Jacobi tolerance, and rotations.
//   - Stage 3: execute the same canonical covariance kernel for every valid policy.
//
// Behavior highlights:
//   - Standardize changes the training metric, not the original input units.
//   - Components is bounded by min(n-1,d) and the detected covariance numerical rank.
//   - RankTolerance is in covariance eigenvalue coordinates, not singular-value units.
//
// Inputs:
//   - All fields are caller-supplied by value, no pointers or retained aliases.
//
// Returns:
//   - The resolved policy governs Fit; the struct itself has no returned slices.
//
// Errors:
//   - ErrInvalidOption for negative or non-finite fields and unknown solver.
//   - ErrUnsupportedSolver if Solver is ThinSVD, even when observations are nil.
//   - ErrOverflow if the default 50*d*d rotation count cannot fit int.
//
// Determinism:
//   - Zero defaults and all comparisons are pure functions of opts and shape.
//
// Complexity:
//   - Time O(1), space O(1) for validation; solver work is documented by Fit.
//
// Notes:
//   - Components=0 is a rank-selection policy; RankTolerance=0 is not rank zero.
//
// AI-Hints:
//   - Do not reinterpret EigenTolerance as an absolute covariance threshold.
//   - Do not silently turn a requested component count into min(k,rank).
type Options struct {
	// Components is k retained latent axes; zero requests detected numerical rank.
	// Negative values are invalid and positive values above rank fail explicitly.
	Components int

	// Standardize selects population-standardized training coordinates (DDOF=0).
	// Every feature is always centered; true additionally requires nonconstant columns.
	Standardize bool

	// Solver selects CovarianceEigen (zero) or unavailable ThinSVD.
	// No value switches the implementation silently on numerical failure.
	Solver Solver

	// RankTolerance is a relative eigenvalue cutoff: keep lambda>tau*lambdaMax.
	// Zero defaults to machine epsilon times max(n,d), not a singular-value cutoff.
	RankTolerance float64

	// EigenTolerance is the relative max-pivot Jacobi convergence criterion.
	// Zero selects 1e-12; the adapter converts it to matrix's absolute threshold.
	// Looser settings may fail an independent eigenpair accuracy check: they do
	// not authorize publishing a numerically inconsistent factorization.
	EigenTolerance float64

	// MaxRotations is the maximum number of individual Jacobi pivot rotations.
	// Zero selects checked 50*d*d; it does not indicate sweeps or unlimited work.
	MaxRotations int
}

// Model holds a fitted affine preprocessing map and an orthonormal loading basis.
// All parameters are private, independently owned, and immutable after Fit.
//
// Implementation:
//   - Stage 1: Fit freezes training means and positive population scales.
//   - Stage 2: it copies retained canonical loadings into private row-major storage.
//   - Stage 3: Transform, InverseTransform, and Components read but never mutate it.
//
// Behavior highlights:
//   - Transform uses (x-mean)/scale and projects onto the retained loading axes.
//   - InverseTransform restores original units after reconstruction.
//   - Each invocation owns its temporary buffers; the Model contains no mutexes.
//
// Inputs:
//   - The Model is created by Fit; a zero-valued Model is not fitted.
//
// Returns:
//   - Getter matrices are detached; Transform outputs never alias model or inputs.
//
// Errors:
//   - ErrNilInput from inference methods when the receiver is nil or unfitted.
//   - Other errors are specific to each operation and preserve dependency roots.
//
// Determinism:
//   - Original feature column order and the solver's sorted loading order are fixed.
//
// Complexity:
//   - Persistent storage O(d*k+d) float64 values, with k<=min(n-1,d).
//
// Notes:
//   - Concurrent inference is safe if callers do not mutate input matrices.
//   - The public FitResult.Model pointer may be reassigned by its owner; that does
//     not permit mutation of the private fields of an existing fitted Model.
//
// AI-Hints:
//   - Do not recompute test-batch moments during inference.
//   - Never expose loadings as a mutable view; call Components for a detached copy.
type Model struct {
	d        int       // Original feature count; never changes after fitting.
	k        int       // Retained latent dimension; never changes after fitting.
	means    []float64 // Training-only means, in original feature units, length d.
	scales   []float64 // Strictly positive training scales (or ones), length d.
	loadings []float64 // Row-major orthonormal loading matrix, d-by-k.
}

// FitResult publishes the immutable fitted model and detached spectral diagnostics.
// Its public slices belong to the caller and cannot change the private Model.
//
// Implementation:
//   - Stage 1: Fit checks all dimensions, covariance arithmetic, and eigenpairs.
//   - Stage 2: select k from the detected rank and freeze the model parameters.
//   - Stage 3: publish k eigenvalues, full-trace variance ratios, and train residual.
//
// Behavior highlights:
//   - Eigenvalues and ExplainedVarianceRatio have length k in descending order.
//   - Rank is the numerical covariance rank before component truncation.
//   - ReconstructionResidual is a nonnegative Frobenius norm in optimization units.
//   - No partial FitResult is returned on any error, including cancellation.
//
// Inputs:
//   - This struct is returned by Fit; it has no public constructor requirements.
//
// Returns:
//   - Its Model is a frozen trained projector; public diagnostics are detached.
//
// Errors:
//   - None from field inspection; Fit itself returns sentinel-classified errors.
//
// Determinism:
//   - Eigenpairs are descending, exact ties preserve solver order, then signs fix.
//
// Complexity:
//   - Result diagnostics O(k) memory plus Model O(d*k+d); residual O(ndk).
//
// Notes:
//   - Changing public slices cannot change future Transform/InverseTransform values.
//   - IsNil checks pointer presence, not successful training or Model validity.
//
// AI-Hints:
//   - Do not renormalize truncated variance ratios to sum to one.
//   - Do not interpret high variance retention as proof of forecasting quality.
type FitResult struct {
	// Model is a fitted immutable projection, nil only for caller-mutated results.
	// A failed Fit returns a nil result pointer, not a result with nil Model.
	Model *Model

	// Eigenvalues contains the k retained covariance eigenvalues in descending order.
	// Values are nonnegative and belong to the caller after Fit returns.
	Eigenvalues []float64

	// ExplainedVarianceRatio divides each retained lambda by the full accepted trace.
	// Length is k; the sum may be below one when positive eigenvalues are discarded.
	ExplainedVarianceRatio []float64

	// Rank is the complete detected numerical covariance rank before truncation.
	// It is at most min(n-1,d) for centered n-by-d training data.
	Rank int

	// ReconstructionResidual measures ||Y-Y*Vk*Vk^T||_F over all training rows.
	// Y is centered, and standardized iff the selected training policy requests it.
	ReconstructionResidual float64
}

// IsNil reports whether this FitResult pointer is absent, including typed nil.
// It implements core.Nilable without inspecting caller-mutable diagnostics.
//
// Implementation:
//   - Stage 1: compare the pointer with nil; do not dereference the receiver.
//   - Stage 2: return the boolean without inspecting Model or result fields.
//   - Stage 3: perform no mutation, allocation, or numerical computation.
//
// Behavior highlights:
//   - An allocated but empty FitResult is present, not a valid trained model.
//
// Inputs:
//   - r: possibly nil *FitResult, including when stored in core.Nilable.
//
// Returns:
//   - bool: true exactly when the receiver pointer is nil.
//
// Errors:
//   - None; a nil receiver never panics.
//
// Determinism:
//   - Pure pointer comparison independent of result contents.
//
// Complexity:
//   - Time O(1), space O(1); no heap allocation.
//
// Notes:
//   - Check Fit's error and Model separately when deciding if training succeeded.
//
// AI-Hints:
//   - Do not check len(Eigenvalues)==0 to infer a typed-nil result.
//   - Never use IsNil as a replacement for Fit error handling.
func (r *FitResult) IsNil() bool { return r == nil }

// The compile-time assertion prevents API drift from core.Nilable's signature.
var _ core.Nilable = (*FitResult)(nil)
