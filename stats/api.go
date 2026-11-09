// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats publishes the canonical statistical facades and explicit wrappers.
// This file applies public validation and delegates all mathematics to one kernel.
package stats

import "github.com/lvlath/go/matrix"

// Summarize computes the fitted central-moment snapshot of a nonempty slice.
// This canonical facade selects population (ddof=0) or sample (ddof=1)
// variance without retaining or modifying any elements of the input slice.
// Implementation:
//   - Stage 1: validate ddof before nil and empty input checks.
//   - Stage 2: delegate one ordered scan to the Welford summarizeKernel.
//   - Stage 3: return a completely detached Summary or nil on failure.
//
// Behavior highlights:
//   - Intermediate M2 may exceed MaxFloat64; only final results need fit.
//
// Inputs:
//   - values: caller-owned non-nil finite float64 slice, len>0.
//   - ddof: 0 for population, 1 for unbiased sample denominator.
//
// Returns: *Summary with finite materialized fields and exact Count/Min/Max.
// Errors:
//   - ErrInvalidOption: unsupported ddof, even when input is nil.
//   - ErrNilInput: nil slice; ErrEmptyInput: non-nil empty slice.
//   - ErrNonFinite: invalid observation.
//   - ErrInsufficientSamples: len(values)<=ddof.
//   - ErrOverflow: final mean/variance does not fit finite float64.
//
// Determinism: left-to-right input iteration with no reordering.
// Complexity: Time O(len(values)), Space O(1), excluding the returned summary.
// Notes: does not mutate values; no retained alias or accumulator pointer.
// AI-Hints: use Moments directly for streaming, without reimplementing a sum-of-squares.
func Summarize(values []float64, ddof int) (*Summary, error) {
	if !validDDOF(ddof) {
		return nil, ErrInvalidOption
	}
	if values == nil {
		return nil, ErrNilInput
	}
	if len(values) == 0 {
		return nil, ErrEmptyInput
	}
	return summarizeKernel(values, ddof)
}

// SummarizePopulation materializes population moments without a ddof argument.
// It is an explicit convenience wrapper for Summarize(values, 0), preserving
// the same input ownership, error identity, and numerical reduction order.
// Implementation: delegate directly to the canonical Summarize facade.
// Inputs: values is a non-nil, nonempty caller-owned float64 slice.
// Returns: *Summary with variance denominator Count.
// Errors: exactly the applicable Summarize error sentinels.
// Determinism: identical to Summarize(values, 0).
// Complexity: Time O(len(values)), Space O(1).
// Notes: the wrapper does not discard or modify any result fields.
// AI-Hints: use SummarizeSample instead when you need denominator Count-1.
func SummarizePopulation(values []float64) (*Summary, error) {
	return Summarize(values, 0)
}

// SummarizeSample materializes sample moments without a ddof argument.
// It is an explicit convenience wrapper for Summarize(values, 1), retaining
// exactly the same validation, finite arithmetic, and result ownership laws.
// Implementation: delegate directly to the canonical Summarize facade.
// Inputs: values is a non-nil, nonempty caller-owned float64 slice.
// Returns: *Summary with variance denominator Count-1.
// Errors: Summarize sentinels, including ErrInsufficientSamples for a singleton.
// Determinism: identical to Summarize(values, 1).
// Complexity: Time O(len(values)), Space O(1).
// Notes: use this method only when a sample denominator is meaningful.
// AI-Hints: the wrapper does not use N-1 for Mean, Min, Max, or Count.
func SummarizeSample(values []float64) (*Summary, error) {
	return Summarize(values, 1)
}

// FitStandardizer learns column-wise parameters from a training matrix only.
// Rows are observations and columns are features; the returned immutable model
// must be reused to transform heldout batches without leakage or refitting.
// Implementation:
//   - Stage 1: validate complete options, presence, nonempty shape, and size.
//   - Stage 2: delegate one row-major scan to the fitted scaling kernel.
//   - Stage 3: publish detached metadata and a model with private copies.
//
// Behavior highlights:
//   - DDOF defaults to 0; RejectConstant is the default constant policy.
//   - Scale uses sqrt of extended-range variance, including when variance>MaxFloat64.
//
// Inputs:
//   - x: read-only-for-call matrix.Matrix, rows>=1 and cols>=1.
//   - opts: DDOF=0/1 and one recognized ConstantPolicy, passed by value.
//
// Returns: *ScaleFitResult containing immutable Model and detached arrays.
// Errors:
//   - ErrInvalidOption: unsupported DDOF/policy, before reading x.
//   - ErrNilInput: nil or conventionally typed-nil Matrix.
//   - ErrShape: invalid dimensions or unrepresentable element count.
//   - ErrEmptyInput: zero rows or columns.
//   - ErrNonFinite: any non-finite training cell.
//   - ErrInsufficientSamples: rows<=DDOF after finite scan.
//   - ErrZeroVariance: exact-constant column under RejectConstant.
//   - ErrOverflow: fitted mean or scale cannot fit finite float64.
//   - Wrapped Matrix.At errors preserve their root sentinel.
//
// Determinism: every row in order, columns ascending within each row.
// Complexity: Time O(rows*cols), Space O(cols) for state and fit metadata.
// Notes: no model-to-result slice alias; fit does not mutate its input.
// AI-Hints:
//   - Never call FitStandardizer on test/production inference data.
//   - Pass *matrix.Dense for its O(1) At access; no unsafe private backing access.
func FitStandardizer(x matrix.Matrix, opts ScaleOptions) (*ScaleFitResult, error) {
	// Options must precede all matrix nil, shape, and value checks.
	if err := validateScaleOptions(opts); err != nil {
		return nil, err
	}
	rows, cols, err := matrixShape(x)
	if err != nil {
		return nil, err
	}
	if rows == 0 || cols == 0 {
		return nil, ErrEmptyInput
	}
	return fitStandardizerKernel(x, rows, cols, opts)
}

// FitPopulationStandardizer fits with DDOF=0 and an explicit constant policy.
// This reduces configuration boilerplate without silently changing the default
// policy or creating another fitting mathematics path.
// Implementation: call FitStandardizer with a complete ScaleOptions value.
// Inputs:
//   - x: training matrix, observations by rows and features by columns.
//   - policy: RejectConstant or UnitScaleConstant.
//
// Returns: *ScaleFitResult with detached arrays and immutable Model.
// Errors: exactly the applicable FitStandardizer error sentinels.
// Determinism: same row-major fit as FitStandardizer(x, ScaleOptions{DDOF:0,...}).
// Complexity: Time O(rows*cols), Space O(cols).
// Notes: the wrapper fixes only DDOF; callers must choose the constant policy.
// AI-Hints: use the fitted Model for every subsequent inference batch.
func FitPopulationStandardizer(x matrix.Matrix, policy ConstantPolicy) (*ScaleFitResult, error) {
	return FitStandardizer(x, ScaleOptions{DDOF: 0, ConstantPolicy: policy})
}

// FitSampleStandardizer fits with DDOF=1 and an explicit constant policy.
// The wrapper is intended for sample-standard-deviation conventions and
// requires at least two observations per feature.
// Implementation: call FitStandardizer with a complete ScaleOptions value.
// Inputs:
//   - x: training matrix with at least two rows and one feature.
//   - policy: RejectConstant or UnitScaleConstant.
//
// Returns: *ScaleFitResult with detached arrays and immutable Model.
// Errors: exactly the applicable FitStandardizer sentinels; singleton->ErrInsufficientSamples.
// Determinism: identical to FitStandardizer(x, ScaleOptions{DDOF:1,...}).
// Complexity: Time O(rows*cols), Space O(cols).
// Notes: never fits or transforms holdout data as an implicit side effect.
// AI-Hints: choose this only if the sample denominator matches your data model.
func FitSampleStandardizer(x matrix.Matrix, policy ConstantPolicy) (*ScaleFitResult, error) {
	return FitStandardizer(x, ScaleOptions{DDOF: 1, ConstantPolicy: policy})
}
