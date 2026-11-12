// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca verifies covariance symmetry, Jacobi convergence, eigenpair
// equations, positive semidefiniteness, numerical rank, and loading signs.
// Distinct certification stages prevent a loose solver tolerance from silently
// weakening the independent conditions for publishing a fitted PCA model.
package pca

import (
	"context"
	"math"
	"sort"

	"github.com/lvlath/go/matrix"
)

const (
	// orthogonalityRoundingFactor bounds accumulated dot-product roundoff in
	// binary64 loadings. It is a safety check, not a proof of solver convergence.
	orthogonalityRoundingFactor = 32

	// eigenResidualPivotFactor converts the verified Jacobi max-pivot threshold
	// into a conservative Euclidean residual budget for an eigenpair.
	eigenResidualPivotFactor = 4
)

// spectralData stores the certified spectrum in descending-eigenvalue order.
// vectors has private row-major d-by-d storage; rank counts strictly positive
// eigenvalues above the caller's relative covariance-eigenvalue threshold.
// normalizedSum is sum(lambda_i / maximum), avoiding an overflowing raw trace.
type spectralData struct {
	values        []float64
	vectors       []float64
	rank          int
	maximum       float64
	normalizedSum float64
}

// spectralPair keeps one Jacobi eigenvalue paired with its original column.
// The source index supplies a deterministic tie-break under equal eigenvalues;
// sorting never changes the eigenvalue-to-eigenvector correspondence.
type spectralPair struct {
	value float64
	index int
}

// covarianceSnapshot is a detached, validated representation of a symmetric C.
// values is row-major d-by-d; norm is the infinity norm of the covariance.
// The snapshot is used solely to verify the eigensolver's returned eigenpairs.
type covarianceSnapshot struct {
	values []float64
	norm   float64
}

// analyzeSpectrum validates every eigenpair before accepting a covariance rank.
// Its stage order keeps covariance corruption, nonconvergence, and numerical
// rank deficiency distinguishable rather than treating them as one error.
//
// Implementation:
//   - Stage 1: resolve the checked rotation budget; read finite, exactly
//     symmetric covariance and compute the infinity norm.
//   - Stage 2: run matrix.EigenSym exactly once, sort eigenpairs, canonicalize
//     loading signs, and reject negative eigenvalues outside PSD allowance.
//   - Stage 3: independently verify orthonormality/eigen-equation residuals,
//     then derive the full numerical rank and normalized spectral trace.
//
// Behavior highlights:
//   - EigenTolerance changes Jacobi's pivot cutoff, never the independent
//     eigenpair verification threshold beyond the default quality requirement.
//   - Only tiny negative eigenvalues within epsilon*d*lambda_max are clipped.
//
// Inputs:
//   - ctx: active context checked between all stages and during matrix scans.
//   - cov: caller-unmodified d-by-d sample covariance from matrix.Covariance.
//   - n,d: validated sample and feature counts; n>=2, d>=1.
//   - opts: validated rank, rotation, and relative convergence policies.
//
// Returns:
//   - *spectralData: detached complete certified eigenbasis or nil on failure.
//
// Errors:
//   - ErrZeroVariance: covariance has exactly zero norm.
//   - ErrNotConverged: exhausted Jacobi rotations, preserving matrix sentinel.
//   - ErrNumerical: asymmetry, invalid PSD, nonorthogonality, or large residual.
//   - ErrOverflow: norms, thresholds, or required counts cannot be represented.
//   - ErrRankDeficient: the prescribed numerical rank is zero.
//   - ctx.Err() and Matrix.At errors: cancellation or backend read failures.
//
// Determinism:
//   - Covariance entries are scanned row-major; Jacobi uses matrix pivot order.
//   - Equal eigenvalues keep the original solver column order; loading sign
//     pivots choose the lowest row index on equal absolute magnitudes.
//
// Complexity:
//   - Time O(rotations*d*d + d*d*d), Space O(d*d) beyond solver workspaces.
//   - Orthogonality and equation-residual checks each cost O(d*d*d).
//
// Notes:
//   - Sign canonicalization does not determine a unique repeated eigenspace
//     basis; cross-solver comparisons need projectors of complete eigenspaces.
//   - PSD allowance is strict and scale-aware, not a universal covariance
//     forward-error certificate for highly ill-conditioned data.
//
// AI-Hints:
//   - Never repair a corrupt covariance by symmetrizing it in this stage.
//   - Never promote matrix.EigenSym's tolerance into a license to certify
//     unrotated diagonal entries as correct eigenvalues.
func analyzeSpectrum(ctx context.Context, cov matrix.Matrix, n, d int, opts Options) (*spectralData, error) {
	rotations, err := checkedRotations(d, opts.MaxRotations)
	if err != nil {
		return nil, err
	}
	covariance, err := readSymmetricCovariance(ctx, cov, d)
	if err != nil {
		return nil, err
	}
	pairs, vectors, absoluteTol, err := solveCovarianceEigen(ctx, cov, d, opts.EigenTolerance, rotations, covariance.norm)
	if err != nil {
		return nil, err
	}

	// A sorting operation must carry eigenvectors with their eigenvalues.
	// Canonical sign selection affects only orientation, not the projector.
	spectrum, err := collectCanonicalSpectrum(ctx, vectors, pairs, d)
	if err != nil {
		return nil, err
	}
	allowance, err := enforceSpectrumPSD(spectrum, d)
	if err != nil {
		return nil, err
	}

	// Numerical certification uses the original covariance snapshot rather
	// than the Jacobi-mutated workspace, which is not exposed by the API.
	if err = verifySpectrum(ctx, covariance, spectrum, pairs, absoluteTol, allowance, d); err != nil {
		return nil, err
	}
	if err = determineSpectralRank(spectrum, n, d, opts.RankTolerance); err != nil {
		return nil, err
	}
	return spectrum, nil
}

// readSymmetricCovariance validates the eigensolver's mathematical input.
// It refuses nonfinite values, asymmetric entries, and arithmetic overflow
// without altering the supplied covariance or manufacturing symmetry.
//
// Implementation:
//   - Stage 1: allocate a detached d-by-d verification snapshot.
//   - Stage 2: scan all cells row-major, accumulating each absolute row sum.
//   - Stage 3: check exact mirrored equality and nonzero infinity norm.
//
// Behavior highlights:
//   - Exact symmetry is justified because matrix.Covariance mirrors each
//     upper-triangle result into the matching lower-triangle cell.
//
// Inputs:
//   - ctx: active context checked once per row.
//   - cov: read-only d-by-d covariance matrix; d: validated dimension.
//
// Returns:
//   - *covarianceSnapshot: detached finite values and finite infinity norm.
//
// Errors:
//   - ErrOverflow: unrepresentable matrix element or absolute row sum.
//   - ErrNumerical: asymmetry between corresponding upper/lower cells.
//   - ErrZeroVariance: the matrix has zero norm and thus no useful spectrum.
//   - ctx.Err() or Matrix.At error: cancellation or backend read failure.
//
// Determinism:
//   - Rows and columns ascend; the first observed symmetry defect wins.
//
// Complexity:
//   - Time O(d*d), Space O(d*d) for the independent snapshot.
//
// Notes:
//   - This is a validation boundary, not an alternative covariance kernel.
//
// AI-Hints:
//   - Do not replace strict mirrored equality with matrix.Symmetrize;
//     a success result with broken symmetry must remain a numerical failure.
func readSymmetricCovariance(ctx context.Context, cov matrix.Matrix, d int) (*covarianceSnapshot, error) {
	snapshot := &covarianceSnapshot{values: make([]float64, d*d)}
	for row := 0; row < d; row++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		var rowSum float64
		for col := 0; col < d; col++ {
			value, err := cov.At(row, col)
			if err != nil {
				return nil, err
			}
			if !finite(value) {
				return nil, ErrOverflow
			}
			snapshot.values[row*d+col] = value
			rowSum += math.Abs(value)
			if !finite(rowSum) {
				return nil, ErrOverflow
			}
		}
		if rowSum > snapshot.norm {
			snapshot.norm = rowSum
		}
	}
	if snapshot.norm == 0 {
		return nil, ErrZeroVariance
	}
	for row := 0; row < d; row++ {
		for col := row + 1; col < d; col++ {
			if snapshot.values[row*d+col] != snapshot.values[col*d+row] {
				return nil, ErrNumerical
			}
		}
	}
	return snapshot, nil
}

// solveCovarianceEigen adapts relative PCA convergence to matrix's Jacobi API.
// It calls the existing EigenSym kernel exactly once and preserves both PCA
// and matrix's error identities if the rotation budget is exhausted.
//
// Implementation:
//   - Stage 1: resolve the relative zero default and multiply by the input norm.
//   - Stage 2: run matrix.EigenSym with an explicit absolute pivot threshold.
//   - Stage 3: validate returned shape/finiteness and stable-sort paired indices.
//
// Behavior highlights:
//   - An underflowed absolute pivot threshold is promoted to the smallest
//     positive float64; a nonfinite threshold fails, never disables checks.
//
// Inputs:
//   - ctx: active context, polled before and after the noninterruptible solver.
//   - cov: original read-only symmetric covariance, d-by-d.
//   - relativeTol: validated relative pivot policy (zero means default).
//   - rotations: checked positive pivot budget; norm: finite positive C-infinity norm.
//
// Returns:
//   - []spectralPair: eigenvalues in descending order, keeping source indices.
//   - matrix.Matrix: the original solver's d-by-d eigenvector columns.
//   - float64: absolute pivot tolerance actually used by the solver.
//
// Errors:
//   - ErrNotConverged: solver exhausted rotations; matrix cause retained.
//   - ErrNumerical: eigensolver returned malformed or nonfinite successful data.
//   - ErrOverflow: tolerance multiplication exceeds finite float64 range.
//   - ctx.Err(): cancellation before or immediately after the dependency call.
//
// Determinism:
//   - Stable descending sort preserves the solver index on equal eigenvalues.
//
// Complexity:
//   - Time O(rotations*d*d+d*log(d)), Space O(d*d) for matrix kernel and pairs.
//
// Notes:
//   - EigenSym lacks a context argument and cannot be interrupted mid-rotation.
//   - The accuracy of this result is verified independently after sorting.
//
// AI-Hints:
//   - Never send a raw relative tolerance to an absolute-threshold eigensolver.
//   - Do not silently retry with another solver or more rotations on failure.
func solveCovarianceEigen(ctx context.Context, cov matrix.Matrix, d int, relativeTol float64, rotations int, norm float64) ([]spectralPair, matrix.Matrix, float64, error) {
	if relativeTol == 0 {
		relativeTol = defaultEigenTolerance
	}
	absoluteTol := relativeTol * norm
	if !finite(absoluteTol) {
		return nil, nil, 0, ErrOverflow
	}
	if absoluteTol == 0 {
		absoluteTol = math.SmallestNonzeroFloat64
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, 0, err
	}
	values, vectors, err := matrix.EigenSym(cov, absoluteTol, rotations)
	if err != nil {
		return nil, nil, 0, wrapDependency(err)
	}
	if err = ctx.Err(); err != nil {
		return nil, nil, 0, err
	}
	if len(values) != d || typedNilMatrix(vectors) || vectors.Rows() != d || vectors.Cols() != d {
		return nil, nil, 0, ErrNumerical
	}
	pairs := make([]spectralPair, d)
	for column, value := range values {
		if !finite(value) {
			return nil, nil, 0, ErrNumerical
		}
		pairs[column] = spectralPair{value: value, index: column}
	}
	sort.SliceStable(pairs, func(i, j int) bool { return pairs[i].value > pairs[j].value })
	return pairs, vectors, absoluteTol, nil
}

// collectCanonicalSpectrum copies the sorted eigenvectors into a private basis.
// The maximum-absolute coordinate of each column is made positive; exact ties
// favor the smallest feature index without changing any eigenvector subspace.
//
// Implementation:
//   - Stage 1: allocate complete d-by-d row-major loading and value arrays.
//   - Stage 2: read each eigenvector in sorted-pair order and locate its sign pivot.
//   - Stage 3: flip every coordinate of the column when the pivot is negative.
//
// Behavior highlights:
//   - Zero eigenvectors and nonfinite loadings never reach the fitted model.
//   - The rule is deterministic within the current solver's chosen basis.
//
// Inputs:
//   - ctx: active context; vectors: read-only solver eigenvector matrix.
//   - pairs: eigenvalues with original solver columns, ordered descending.
//   - d: feature dimension and complete number of eigenvectors.
//
// Returns:
//   - *spectralData: privately stored sorted loadings and eigenvalues.
//
// Errors:
//   - ErrNumerical: zero-length or nonfinite eigenvector entries.
//   - ctx.Err() or Matrix.At error: cancellation or backend read failure.
//
// Determinism:
//   - Eigenvector columns follow pairs; coordinates ascend; equal absolute
//     coordinates keep the first index because the pivot comparison is strict.
//
// Complexity:
//   - Time O(d*d), Space O(d*d) for owned arrays.
//
// Notes:
//   - Repeated eigenvalues permit arbitrary rotations inside their eigenspace;
//     canonical signs alone do not make the cross-implementation basis unique.
//
// AI-Hints:
//   - Do not sort eigenvalues without moving the corresponding eigenvector.
//   - Canonicalize a sign only after the entire column has been checked.
func collectCanonicalSpectrum(ctx context.Context, vectors matrix.Matrix, pairs []spectralPair, d int) (*spectralData, error) {
	result := &spectralData{values: make([]float64, d), vectors: make([]float64, d*d)}
	for column, pair := range pairs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		result.values[column] = pair.value
		pivot := 0
		var largest float64
		for row := 0; row < d; row++ {
			value, err := vectors.At(row, pair.index)
			if err != nil {
				return nil, err
			}
			if !finite(value) {
				return nil, ErrNumerical
			}
			result.vectors[row*d+column] = value
			if magnitude := math.Abs(value); magnitude > largest {
				largest, pivot = magnitude, row
			}
		}
		if largest == 0 {
			return nil, ErrNumerical
		}
		if result.vectors[pivot*d+column] < 0 {
			for row := 0; row < d; row++ {
				result.vectors[row*d+column] = -result.vectors[row*d+column]
			}
		}
	}
	return result, nil
}

// enforceSpectrumPSD applies only the narrowly specified negative-roundoff rule.
// Eigenvalues materially below zero indicate a broken covariance/eigenpair
// invariant; they are not replaced with a plausible-looking zero spectrum.
//
// Implementation:
//   - Stage 1: require a strictly positive leading eigenvalue.
//   - Stage 2: calculate a checked epsilon*d*lambda_max rounding allowance.
//   - Stage 3: reject values below -allowance; clip only negative values inside it.
//
// Behavior highlights:
//   - All nonnegative eigenvalues, including small positive ones, are preserved.
//
// Inputs:
//   - spectrum: locally owned sorted values; d: covariance feature dimension.
//
// Returns:
//   - float64: finite PSD rounding allowance; the spectrum may be clipped in place.
//
// Errors:
//   - ErrNumerical: no positive leading mode or materially negative eigenvalue.
//   - ErrOverflow: allowance is not representable.
//
// Determinism:
//   - Ascending sorted eigenvalue index determines the first rejection.
//
// Complexity:
//   - Time O(d), Space O(1) beyond the owned spectrum.
//
// Notes:
//   - The allowance is a strict spectrum-relative safeguard, not a universal
//     certified backward-error bound for the covariance construction process.
//
// AI-Hints:
//   - Do not increase the clipping threshold to the user's EigenTolerance;
//     a loose Jacobi stopping policy must not weaken positive semidefiniteness.
func enforceSpectrumPSD(spectrum *spectralData, d int) (float64, error) {
	maximum := spectrum.values[0]
	if maximum <= 0 {
		return 0, ErrNumerical
	}
	allowance := machineEpsilon * float64(d) * maximum
	if !finite(allowance) {
		return 0, ErrOverflow
	}
	for index, value := range spectrum.values {
		if value < -allowance {
			return 0, ErrNumerical
		}
		if value < 0 {
			spectrum.values[index] = 0
		}
	}
	spectrum.maximum = spectrum.values[0]
	return allowance, nil
}

// verifySpectrum checks the returned basis independently of Jacobi's exit flag.
// A loose pivot tolerance may end a solver call early, but it never relaxes
// the model publication criterion beyond PCA's default eigenpair accuracy.
//
// Implementation:
//   - Stage 1: choose the stricter user/default absolute verification tolerance.
//   - Stage 2: build finite orthogonality and eigen-equation residual budgets.
//   - Stage 3: check pairwise loading orthogonality and every C*v=lambda*v equation.
//
// Behavior highlights:
//   - The covariance snapshot comes from the original matrix, not a mutable
//     diagonalization workspace; residuals use the un-clipped raw eigenvalues.
//
// Inputs:
//   - ctx: active context; covariance: original finite covariance snapshot.
//   - spectrum: canonical loadings; pairs: raw eigenvalues in sorted order.
//   - absoluteTol: actual Jacobi pivot threshold; allowance: strict PSD limit.
//   - d: original feature dimension.
//
// Returns:
//   - error: nil only when all d vectors satisfy both independent checks.
//
// Errors:
//   - ErrNumerical: inaccurate eigenvectors or nonfinite equation residuals.
//   - ErrOverflow: either verification threshold is unrepresentable.
//   - ctx.Err(): cancellation while scanning the complete eigensystem.
//
// Determinism:
//   - Orthogonality checks visit columns ascending, earlier columns first;
//     equation residuals read rows and contracted features ascending.
//
// Complexity:
//   - Time O(d*d*d), Space O(1) beyond the input snapshots.
//
// Notes:
//   - Successful eigendecomposition cannot be inferred from diagonals alone.
//
// AI-Hints:
//   - Do not reuse a caller's loose stopping tolerance for model certification.
//   - A threshold of +Inf would make every equation pass; reject it explicitly.
func verifySpectrum(ctx context.Context, covariance *covarianceSnapshot, spectrum *spectralData, pairs []spectralPair, absoluteTol, allowance float64, d int) error {
	verificationTol := math.Min(absoluteTol, defaultEigenTolerance*covariance.norm)
	if verificationTol == 0 {
		verificationTol = math.SmallestNonzeroFloat64
	}
	orthogonalityBound := orthogonalityRoundingFactor * machineEpsilon * float64(d)
	eigenResidualBound := eigenResidualPivotFactor*verificationTol*math.Sqrt(float64(d)) +
		orthogonalityRoundingFactor*allowance
	if !finite(orthogonalityBound) || !finite(eigenResidualBound) {
		return ErrOverflow
	}
	if err := verifyLoadingOrthogonality(ctx, spectrum.vectors, d, orthogonalityBound); err != nil {
		return err
	}
	return verifyEigenpairEquations(ctx, covariance.values, spectrum.vectors, pairs, d, eigenResidualBound)
}

// verifyLoadingOrthogonality checks V^T*V against the identity matrix.
// It detects collapsed, duplicated, or non-unit eigenvectors even when their
// eigenvalues appear plausible and the original covariance is symmetric.
//
// Implementation:
//   - Stage 1: accumulate each squared Euclidean column norm.
//   - Stage 2: compare the column against each previously accepted column.
//   - Stage 3: accept only finite deviations within the scale-aware bound.
//
// Behavior highlights:
//   - All comparisons use the unmodified canonical loading values.
//
// Inputs:
//   - ctx: active context; vectors: private row-major d-by-d loadings.
//   - d: feature count; bound: finite absolute orthogonality tolerance.
//
// Returns:
//   - error: nil if every V column is normalized and mutually orthogonal.
//
// Errors:
//   - ErrNumerical: failed unit norm or nonzero cross-column dot product.
//   - ctx.Err(): cancellation between column checks.
//
// Determinism:
//   - Columns ascend, previous columns ascend, features ascend in dot products.
//
// Complexity:
//   - Time O(d*d*d), Space O(1).
//
// Notes:
//   - This validates geometry, not the eigen-equations; both are necessary.
//
// AI-Hints:
//   - Never skip off-diagonal dot products because all norms are close to one.
func verifyLoadingOrthogonality(ctx context.Context, vectors []float64, d int, bound float64) error {
	for column := 0; column < d; column++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		var squaredNorm float64
		for row := 0; row < d; row++ {
			value := vectors[row*d+column]
			squaredNorm += value * value
		}
		if !finite(squaredNorm) || math.Abs(squaredNorm-1) > bound {
			return ErrNumerical
		}
		for previous := 0; previous < column; previous++ {
			var dot float64
			for row := 0; row < d; row++ {
				dot += vectors[row*d+column] * vectors[row*d+previous]
			}
			if math.Abs(dot) > bound {
				return ErrNumerical
			}
		}
	}
	return nil
}

// verifyEigenpairEquations checks C*v=lambda*v for every certified eigenpair.
// It catches premature Jacobi termination even when the returned loading basis
// happens to be orthonormal (for example, an unrotated identity eigenbasis).
//
// Implementation:
//   - Stage 1: contract the original C against each candidate loading column.
//   - Stage 2: compute the Euclidean residual against its raw eigenvalue.
//   - Stage 3: reject nonfinite or over-budget residuals before model publication.
//
// Behavior highlights:
//   - The check is independent of clipped PSD eigenvalues and rank selection.
//
// Inputs:
//   - ctx: active context; covariance: flat row-major original C.
//   - vectors: flat d-by-d loading matrix; pairs: raw sorted eigenvalue pairs.
//   - d: feature count; bound: finite absolute residual budget.
//
// Returns:
//   - error: nil when every eigenpair equation is sufficiently accurate.
//
// Errors:
//   - ErrNumerical: nonfinite or excessive residual (including overflow).
//   - ctx.Err(): cancellation between column equations.
//
// Determinism:
//   - Columns ascend, then rows, then the contracted covariance coordinates.
//
// Complexity:
//   - Time O(d*d*d), Space O(1).
//
// Notes:
//   - A residual pass alone does not prove orthogonality; the preceding check
//     supplies the missing geometric part of the eigen-decomposition contract.
//
// AI-Hints:
//   - Use raw eigenvalues in the equation check; PSD clipping is a later
//     modeling policy and must not change which equation EigenSym actually solved.
func verifyEigenpairEquations(ctx context.Context, covariance, vectors []float64, pairs []spectralPair, d int, bound float64) error {
	for column, pair := range pairs {
		if err := ctx.Err(); err != nil {
			return err
		}
		var residual float64
		for row := 0; row < d; row++ {
			var product float64
			for feature := 0; feature < d; feature++ {
				product += covariance[row*d+feature] * vectors[feature*d+column]
			}
			difference := product - pair.value*vectors[row*d+column]
			if !finite(difference) {
				return ErrNumerical
			}
			residual = math.Hypot(residual, difference)
		}
		if residual > bound {
			return ErrNumerical
		}
	}
	return nil
}

// determineSpectralRank applies the explicitly eigenvalue-relative rank policy.
// It computes the full numeric covariance rank before component truncation and
// accumulates a normalized trace over all accepted nonnegative eigenvalues.
//
// Implementation:
//   - Stage 1: resolve RankTolerance=0 to epsilon*max(n,d).
//   - Stage 2: count lambda>tau*lambda_max and sum lambda/lambda_max.
//   - Stage 3: enforce the centered-data rank ceiling and nonzero usable rank.
//
// Behavior highlights:
//   - Tiny accepted nonnegative eigenvalues still enter the variance denominator.
//   - The strict inequality means a mode exactly at threshold is not retained.
//
// Inputs:
//   - spectrum: locally owned certified, PSD-checked sorted spectrum.
//   - n,d: sample/feature counts; relativeTolerance: finite nonnegative option.
//
// Returns:
//   - error: nil if usable rank is within min(n-1,d), otherwise sentinel.
//
// Errors:
//   - ErrInvalidOption: default cutoff itself becomes nonfinite (defensive).
//   - ErrNumerical: rank exceeds centered-data mathematical ceiling.
//   - ErrOverflow: normalized trace exceeds finite float64 representation.
//   - ErrRankDeficient: no eigenvalue exceeds the requested rank threshold.
//
// Determinism:
//   - Eigenvalues are traversed in sorted descending order, including zeros.
//
// Complexity:
//   - Time O(d), Space O(1), excluding the existing spectrum.
//
// Notes:
//   - This rank policy is defined on covariance eigenvalues, not singular
//     values; replacing it with an SVD-based threshold changes the contract.
//
// AI-Hints:
//   - Keep ratio denominators based on the full nonnegative spectrum, not k.
//   - Never drop a small positive eigenvalue merely to force ratios to sum to 1.
func determineSpectralRank(spectrum *spectralData, n, d int, relativeTolerance float64) error {
	threshold := relativeTolerance
	if threshold == 0 {
		threshold = machineEpsilon * float64(max(n, d))
	}
	if !finite(threshold) {
		return ErrInvalidOption
	}
	for _, eigenvalue := range spectrum.values {
		if eigenvalue > threshold*spectrum.maximum {
			spectrum.rank++
		}
		spectrum.normalizedSum += eigenvalue / spectrum.maximum
	}
	if !finite(spectrum.normalizedSum) {
		return ErrOverflow
	}
	if spectrum.rank > min(n-1, d) {
		return ErrNumerical
	}
	if spectrum.rank == 0 {
		return ErrRankDeficient
	}
	return nil
}
