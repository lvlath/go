// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats fits independent scaled moments for matrix feature columns.
// This file owns training-time arithmetic and immutable parameter publication.
package stats

import "github.com/lvlath/go/matrix"

// fitStandardizerKernel learns one immutable mean and scale per input column.
// The public facade has already validated options and matrix shape; the kernel
// never mutates observations, even when their concrete type is *matrix.Dense.
// Implementation:
//   - Stage 1: stream rows in deterministic order into per-column Moments.
//   - Stage 2: ensure samples suffice, then classify exact-zero M2 per column.
//   - Stage 3: copy fitted parameters into the immutable model and public result.
//
// Behavior highlights:
//   - Stddev is sqrt(M2/(rows-DDOF)) directly in extended exponent arithmetic.
//   - The scale can be finite even if the un-squared variance exceeds float64.
//
// Inputs:
//   - input: read-only-for-call matrix, shape rows>0 and cols>0.
//   - rows, cols: validated matrix dimensions, with rows*cols fitting int.
//   - opts: validated DDOF and constant-column policy.
//
// Returns: *ScaleFitResult with detached arrays and immutable Model.
// Errors:
//   - ErrNonFinite: any training cell is NaN or ±Inf.
//   - ErrInsufficientSamples: rows<=opts.DDOF, after input finiteness checks.
//   - ErrZeroVariance: exact-constant column with RejectConstant policy.
//   - ErrOverflow: mean or positive fitted scale is not representable.
//   - Wrapped matrix.At error: retains the original underlying sentinel.
//
// Determinism:
//   - Row-major reads; feature finalization proceeds from column 0 upward.
//
// Complexity:
//   - Time O(rows*cols), Space O(cols), excluding returned metadata.
//
// Notes:
//   - A model is published only if EVERY feature passes all checks.
//   - Input matrices are not snapshotted; concurrent mutation is unsupported.
//
// AI-Hints:
//   - Never call Summary to get a float64 variance before taking sqrt;
//     it would wrongly reject a representable scale for huge variance.
//   - Do not use an epsilon to decide whether a feature is constant.
func fitStandardizerKernel(input matrix.Matrix, rows, cols int, opts ScaleOptions) (*ScaleFitResult, error) {
	// Stage 1: allocate one bounded-size numeric state per feature; there is
	// intentionally no matrix-wide copy and no per-sample rational allocation.
	accumulators := make([]Moments, cols)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			value, err := matrixCell(input, row, col)
			if err != nil {
				return nil, err
			}
			if !finite(value) {
				return nil, ErrNonFinite
			}
			if err = accumulators[col].Update(value); err != nil {
				return nil, err
			}
		}
	}

	// Stage 2: data-domain errors precede insufficient samples, and a fitted
	// scale needs a strictly positive denominator for the requested DDOF.
	if rows <= opts.DDOF {
		return nil, ErrInsufficientSamples
	}

	means := make([]float64, cols)
	scales := make([]float64, cols)
	constant := make([]bool, cols)
	// Every column accumulator has committed exactly one value per row.
	// Read the already-exact uint64 sample count instead of converting the
	// signed matrix dimension; rows>DDOF above makes sample subtraction safe.
	denominatorCount := accumulators[0].count
	if opts.DDOF == 1 {
		denominatorCount--
	}
	denominator := scaledFromCount(denominatorCount)
	for col := 0; col < cols; col++ {
		accumulator := &accumulators[col]
		mean, err := accumulator.exactMeanSum.meanFloat(accumulator.count)
		if err != nil {
			return nil, err
		}
		means[col] = mean

		// Testing internal exact zero prevents tiny, positive variances that
		// underflow on conversion to float64 from being called constant.
		if accumulator.m2.hi == 0 {
			if opts.ConstantPolicy == RejectConstant {
				return nil, ErrZeroVariance
			}
			constant[col] = true
			scales[col] = 1
			continue
		}

		// Taking sqrt before materialization supports large, representable
		// standard deviations even when M2 and variance both exceed float64.
		variance := accumulator.m2.div(denominator)
		scale, err := variance.sqrtFloat()
		if err != nil {
			return nil, err
		}
		scales[col] = scale
	}

	// Stage 3: the public metadata slices and model-owned slices must never
	// share backing arrays. Transform subsequently reads only private fields.
	model := &Standardizer{
		means:    append([]float64(nil), means...),
		scales:   append([]float64(nil), scales...),
		constant: append([]bool(nil), constant...),
	}
	return &ScaleFitResult{
		Model:    model,
		Means:    means,
		Scales:   scales,
		Constant: constant,
	}, nil
}
