// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca owns the ordered training stages: optional population scaling,
// sample covariance, verified eigenvectors, and reconstruction diagnostics.
// This file publishes no model until all stages and their invariants succeed.
package pca

import (
	"context"
	"math"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/stats"
)

// trainingData carries one call's original-unit fitted preprocessing parameters.
// Only the training call may populate means and scales; the published Model takes
// ownership after the covariance and spectrum have passed their validations.
// prepared aliases the caller's input when Standardize=false and otherwise owns
// the detached output of stats.Standardizer.Transform.
type trainingData struct {
	prepared matrix.Matrix
	means    []float64
	scales   []float64
}

// fitKernel runs exactly one covariance-based PCA training pipeline.
// It separates statistical preparation, covariance construction, spectral
// certification, and result publication so their failure laws are auditable.
//
// Implementation:
//   - Stage 1: fit optional DDOF=0 scales from the training rows only.
//   - Stage 2: build sample covariance and certify its sorted eigensystem.
//   - Stage 3: select k, measure the original training reconstruction error,
//     and publish an independent Model plus caller-owned diagnostics.
//
// Behavior highlights:
//   - No recomputation of training moments is permitted during inference.
//   - Every error returns a nil result; no incomplete Model escapes.
//
// Inputs:
//   - ctx: already validated active context; cancellation is polled at stages.
//   - input: caller-owned n-by-d finite training data, untouched and unretained.
//   - n,d: validated observation/feature counts, n>=2 and d>=1.
//   - opts: validated value-copy PCA policy with an admitted solver.
//
// Returns:
//   - *FitResult: immutable model and detached training diagnostics, or nil.
//
// Errors:
//   - ErrZeroVariance: required standardization or covariance has no variance.
//   - ErrRankDeficient: requested k exceeds the certified numerical rank.
//   - ErrNumerical: an inconsistent dependency output or invalid eigensystem.
//   - ErrOverflow: an unrepresentable fitted parameter, score, or residual.
//   - ErrNotConverged, ctx.Err(), or wrapped matrix/stats errors as applicable.
//
// Determinism:
//   - All feature, observation, and loading loops use ascending indices.
//   - Eigenpair sorting and sign canonicalization are delegated to one stage.
//
// Complexity:
//   - Time O(n*d*d + rotations*d*d + d*d*d + n*d*k).
//   - Space O(n*d + d*d + d*k), including called matrix workspaces.
//
// Notes:
//   - The output model does not retain or mutate input or stats workspaces.
//   - Covariance eigenanalysis is numerically less robust than direct SVD for
//     badly conditioned features; this routine does not hide that limitation.
//
// AI-Hints:
//   - Do not replace this pipeline with a second implementation in api.go.
//   - A reduced residual must be measured in frozen optimization coordinates,
//     not from a newly standardized holdout or a clipped eigenvalue tail.
func fitKernel(ctx context.Context, input matrix.Matrix, n, d int, opts Options) (*FitResult, error) {
	// The preprocessing stage is responsible for the only fit-time estimation
	// of means/scales. Standardize=false avoids an unnecessary matrix copy.
	training, err := prepareTraining(ctx, input, n, d, opts.Standardize)
	if err != nil {
		return nil, err
	}

	// The covariance stage owns the exact sample n-1 denominator through the
	// canonical matrix implementation, including its overflow protection.
	covariance, err := trainingCovariance(ctx, training, d, opts.Standardize)
	if err != nil {
		return nil, err
	}

	// Certify all d eigenpairs before applying the user's k-component policy.
	// An inaccurate eigensystem cannot be rescued by retaining fewer columns.
	spectrum, err := analyzeSpectrum(ctx, covariance, n, d, opts)
	if err != nil {
		return nil, err
	}

	// This is the only result publication point; cancellation or a failed
	// training residual check never returns a partially trained model.
	return publishFitResult(ctx, input, training, spectrum, d, opts.Components)
}

// prepareTraining optionally fits population scales on the training observations.
// It never estimates parameters from later prediction or validation batches.
//
// Implementation:
//   - Stage 1: allocate unit scales for the unstandardized coordinate system.
//   - Stage 2: when requested, call stats.FitStandardizer with DDOF=0 and
//     RejectConstant and check every returned parameter before division.
//   - Stage 3: materialize one detached standardized matrix without changing x.
//
// Behavior highlights:
//   - An unstandardized input is borrowed, not cloned, until Fit returns.
//   - Exact-constant columns are rejected when Standardize=true.
//
// Inputs:
//   - ctx: active training context, checked before and after dependency work.
//   - input: caller-owned, finite n-by-d matrix.
//   - n,d: validated sizes; standardize selects the population scale policy.
//
// Returns:
//   - *trainingData: per-call scales, means, and working matrix, or nil.
//
// Errors:
//   - ErrZeroVariance: stats.RejectConstant refuses a constant feature.
//   - ErrNumerical: successful stats output is structurally inconsistent.
//   - Wrapped stats error or ctx.Err(): learning or transformation failure.
//
// Determinism:
//   - Unit scales are initialized by increasing feature index; stats maintains
//     its row-major scan and fixed feature-accumulator order.
//
// Complexity:
//   - Time O(n*d), Space O(n*d+d) with scaling; O(d) without scaling.
//
// Notes:
//   - Fitted means and scales are copied from the stats result and never aliased.
//   - Concurrent mutation of input during fitting remains unsupported.
//
// AI-Hints:
//   - Do not derive scales as sqrt(materialized sample variance): that
//     can overflow when a valid standard deviation is representable.
//   - Keep the population DDOF=0 convention distinct from sample covariance.
func prepareTraining(ctx context.Context, input matrix.Matrix, n, d int, standardize bool) (*trainingData, error) {
	training := &trainingData{prepared: input, scales: make([]float64, d)}
	for feature := range training.scales {
		training.scales[feature] = 1
	}
	if !standardize {
		return training, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	fit, err := stats.FitStandardizer(input, stats.ScaleOptions{
		DDOF: 0, ConstantPolicy: stats.RejectConstant,
	})
	if err != nil {
		return nil, normalizeStatsError(err)
	}
	if fit == nil || fit.Model == nil || len(fit.Means) != d || len(fit.Scales) != d {
		return nil, ErrNumerical
	}

	// Check the entire dependency artifact before using any returned scale;
	// otherwise a malformed successful result could be partially consumed.
	for feature := 0; feature < d; feature++ {
		if !finite(fit.Means[feature]) || !finite(fit.Scales[feature]) || fit.Scales[feature] <= 0 {
			return nil, ErrNumerical
		}
	}
	training.means = append([]float64(nil), fit.Means...)
	copy(training.scales, fit.Scales)

	if err := ctx.Err(); err != nil {
		return nil, err
	}
	training.prepared, err = fit.Model.Transform(input)
	if err != nil {
		return nil, normalizeStatsError(err)
	}
	if typedNilMatrix(training.prepared) || training.prepared.Rows() != n || training.prepared.Cols() != d {
		return nil, ErrNumerical
	}
	return training, nil
}

// trainingCovariance computes sample covariance using the prepared coordinates.
// The canonical matrix kernel, not PCA eigendecomposition, owns the covariance
// accumulation and its protection against intermediate overflow.
//
// Implementation:
//   - Stage 1: check cancellation before any covariance calculation.
//   - Stage 2: call matrix.Covariance and check its dimensions and means.
//   - Stage 3: retain the unstandardized means or verify frozen scaled means.
//
// Behavior highlights:
//   - Always uses n-1 in sample covariance, including standardized training.
//   - Does not symmetrize or clip the returned covariance matrix.
//
// Inputs:
//   - ctx: active context; training: per-fit preprocessing state.
//   - d: validated feature count; standardized selects the fitted origin.
//
// Returns:
//   - matrix.Matrix: detached d-by-d sample covariance or nil on failure.
//
// Errors:
//   - ErrNumerical: inconsistent dimensions or an absent dependency artifact.
//   - ErrOverflow: nonfinite returned means/scales or unrepresentable covariance.
//   - Wrapped matrix errors and ctx.Err(): dependency arithmetic/cancellation.
//
// Determinism:
//   - Feature validation scans ascending indices; covariance order is set by
//     matrix.Covariance and not reordered by this wrapper.
//
// Complexity:
//   - Time O(n*d*d), Space O(d*d+d) excluding matrix's own temporary state.
//
// Notes:
//   - The matrix package is responsible for extreme covariance formation;
//     eigenvalue clipping must never attempt to repair a corrupt covariance.
//
// AI-Hints:
//   - Do not use a raw centered Gram product to replace matrix.Covariance.
//   - Train-only means, not the mean of a transformed validation batch,
//     determine the affine origin of the published Model.
func trainingCovariance(ctx context.Context, training *trainingData, d int, standardized bool) (matrix.Matrix, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	covariance, covarianceMeans, err := matrix.Covariance(training.prepared)
	if err != nil {
		return nil, wrapDependency(err)
	}
	if typedNilMatrix(covariance) || covariance.Rows() != d || covariance.Cols() != d || len(covarianceMeans) != d {
		return nil, ErrNumerical
	}
	if !standardized {
		// The unstandardized fit deliberately borrows X for covariance, then
		// takes its own means copy before the caller can reuse X.
		training.means = append([]float64(nil), covarianceMeans...)
	}
	for feature := 0; feature < d; feature++ {
		if !finite(covarianceMeans[feature]) || !finite(training.means[feature]) ||
			!finite(training.scales[feature]) || training.scales[feature] <= 0 {
			return nil, ErrOverflow
		}
	}
	return covariance, nil
}

// publishFitResult freezes the selected eigenbasis and its training diagnostics.
// This is deliberately the last stage: no operation after return changes the
// fitted Model, and any failed post-check returns a nil result instead.
//
// Implementation:
//   - Stage 1: enforce k<=rank, including k=0 meaning detected full rank.
//   - Stage 2: copy the first k loading columns into private row-major storage
//     and measure training reconstruction residual in frozen coordinates.
//   - Stage 3: copy eigenvalues and publish full-spectrum variance ratios.
//
// Behavior highlights:
//   - Truncated variance ratios are not renormalized to sum to one.
//   - An excessive component request never pads the basis with zero eigenmodes.
//
// Inputs:
//   - ctx: active context; original: read-only training matrix.
//   - training: privately owned means/scales; spectrum: certified d-dimensional
//     eigenpairs; d: feature count; requested: k, or zero for numerical rank.
//
// Returns:
//   - *FitResult: detached diagnostics and immutable Model, or nil on error.
//
// Errors:
//   - ErrRankDeficient: requested positive k exceeds the full numeric rank.
//   - ErrZeroVariance: no usable components remain (defensive invariant).
//   - ErrOverflow and ctx.Err(): residual cannot be represented or was canceled.
//
// Determinism:
//   - Loadings are copied feature-major with ascending component index;
//     ratios retain the stable sorted eigenvalue order.
//
// Complexity:
//   - Time O(d*k+n*d*k), Space O(d*k+d+k) plus residual row workspace.
//
// Notes:
//   - Exported result fields may be modified by callers without mutating Model.
//   - The private Model does not reference any matrix or external fitted stats.
//
// AI-Hints:
//   - For k<rank, ReconstructionResidual must remain nonzero unless omitted
//     variance is numerically negligible; never fabricate a zero residual.
//   - Never publish before trainingResidual finishes successfully.
func publishFitResult(ctx context.Context, original matrix.Matrix, training *trainingData, spectrum *spectralData, d, requested int) (*FitResult, error) {
	components := requested
	if components == 0 {
		components = spectrum.rank
	}
	if components > spectrum.rank {
		return nil, ErrRankDeficient
	}
	if components <= 0 {
		return nil, ErrZeroVariance
	}

	loadings := make([]float64, d*components)
	for feature := 0; feature < d; feature++ {
		for component := 0; component < components; component++ {
			loadings[feature*components+component] = spectrum.vectors[feature*d+component]
		}
	}
	model := &Model{
		d: d, k: components,
		means: training.means, scales: training.scales, loadings: loadings,
	}
	residual, err := trainingResidual(ctx, original, model)
	if err != nil {
		return nil, err
	}

	result := &FitResult{
		Model:                  model,
		Eigenvalues:            append([]float64(nil), spectrum.values[:components]...),
		ExplainedVarianceRatio: make([]float64, components),
		Rank:                   spectrum.rank,
		ReconstructionResidual: residual,
	}
	for component := 0; component < components; component++ {
		// Divide in eigenvalue-relative coordinates to avoid an overflowing
		// materialized covariance trace when its ratios are representable.
		result.ExplainedVarianceRatio[component] =
			(spectrum.values[component] / spectrum.maximum) / spectrum.normalizedSum
	}
	return result, nil
}

// trainingResidual independently measures the fitted projection's Frobenius error.
// It reads original training rows, applies frozen preprocessing, and reconstructs
// each row using only the retained basis, without allocating a reconstructed matrix.
//
// Implementation:
//   - Stage 1: allocate one original-space row and one latent-score workspace.
//   - Stage 2: center/scale a row and accumulate its retained projections.
//   - Stage 3: reconstruct the row and reduce all residuals using math.Hypot.
//
// Behavior highlights:
//   - Hypot avoids squaring a large residual before accumulating its norm.
//   - Centering uses train means and scales, even for a standardized fit.
//
// Inputs:
//   - ctx: active context; input: read-only original training observations.
//   - model: complete privately owned fitted parameters, d features and k scores.
//
// Returns:
//   - float64: finite, nonnegative training Frobenius error in optimization units.
//
// Errors:
//   - ctx.Err(): cancellation between training rows.
//   - ErrOverflow: subtraction, projection, or residual is unrepresentable.
//   - Matrix.At errors: preserves the original backend failure identity.
//
// Determinism:
//   - Observation, feature, and component indices ascend; FMA preserves the
//     specified accumulation sequence on a fixed Go toolchain.
//
// Complexity:
//   - Time O(n*d*k), Space O(d+k) per training call.
//
// Notes:
//   - This measured residual is independent of the discarded eigenvalue sum.
//   - Inputs and model-owned parameter buffers are never modified.
//
// AI-Hints:
//   - Do not divide the result by n or reverse the standardization to original
//     units; both operations change the public ReconstructionResidual contract.
//   - A tail-eigenvalue identity is an oracle, not a replacement for this kernel.
func trainingResidual(ctx context.Context, input matrix.Matrix, model *Model) (float64, error) {
	var norm float64
	preparedRow := make([]float64, model.d)
	scores := make([]float64, model.k)
	for row := 0; row < input.Rows(); row++ {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		for component := range scores {
			scores[component] = 0
		}
		for feature := 0; feature < model.d; feature++ {
			value, err := input.At(row, feature)
			if err != nil {
				return 0, err
			}
			centered := value - model.means[feature]
			preparedRow[feature] = centered / model.scales[feature]
			if !finite(centered) {
				// The exact difference can exceed float64 even when both ratios
				// remain finite; divide the two operands before subtraction.
				preparedRow[feature] = value/model.scales[feature] - model.means[feature]/model.scales[feature]
			}
			if !finite(preparedRow[feature]) {
				return 0, ErrOverflow
			}
			for component := 0; component < model.k; component++ {
				scores[component] = math.FMA(preparedRow[feature], model.loadings[feature*model.k+component], scores[component])
				if !finite(scores[component]) {
					return 0, ErrOverflow
				}
			}
		}
		for feature := 0; feature < model.d; feature++ {
			var reconstructed float64
			for component := 0; component < model.k; component++ {
				reconstructed = math.FMA(scores[component], model.loadings[feature*model.k+component], reconstructed)
			}
			residual := preparedRow[feature] - reconstructed
			if !finite(residual) {
				return 0, ErrOverflow
			}
			norm = math.Hypot(norm, residual)
			if !finite(norm) {
				return 0, ErrOverflow
			}
		}
	}
	return norm, nil
}
