// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix validates numeric factor options and snapshots matrix inputs.
// This file protects arithmetic, dimensional, and ownership invariants before kernels.
package matrix

import (
	"context"
	"fmt"
	"math"
)

const (
	// factorMachineEpsilon is the binary64 distance between 1 and the next float.
	factorMachineEpsilon = 0x1p-52

	// factorUnity is the exact condition number of an identity matrix.
	factorUnity = 1.0
)

// factorContextErr validates the cooperative cancellation boundary.
// It keeps nil contexts distinct from a context that has already been canceled.
// Implementation:
//   - Stage 1: Reject the absent context as an invalid option.
//   - Stage 2: Query ctx.Err() exactly once for an existing context.
//   - Stage 3: Return its original cancellation cause or nil.
//
// Behavior highlights:
//   - No context cancellation is swallowed or wrapped into an unrelated sentinel.
//
// Inputs:
//   - ctx must be non-nil for new numerical APIs.
//
// Returns:
//   - nil on success, ErrInvalidOption or ctx.Err() on failure.
//
// Errors:
//   - ErrInvalidOption, context.Canceled or context.DeadlineExceeded.
//
// Determinism:
//   - No global clock reading or background task is introduced here.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The caller must repeat checks at long-lived loop boundaries.
//
// AI-Hints:
//   - Do not replace nil context with context.Background silently.
func factorContextErr(ctx context.Context) error {
	if ctx == nil {
		return ErrInvalidOption
	}
	return ctx.Err()
}

// factorValidatedOptions validates both caller-owned numerical factor settings.
// Unbounded or nonfinite tolerances would make rank and condition decisions meaningless.
// Implementation:
//   - Stage 1: Check RankTolerance is finite and nonnegative.
//   - Stage 2: Check ConditionLimit is exactly zero or finite and at least one.
//   - Stage 3: Return without mutating the supplied FactorOptions value.
//
// Behavior highlights:
//   - Zero tolerance remains a delayed dimension-dependent default.
//
// Inputs:
//   - options contains two dimensionless binary64 policy values.
//
// Returns:
//   - nil when all fields are admissible.
//
// Errors:
//   - ErrInvalidOption for any invalid policy field.
//
// Determinism:
//   - Fixed RankTolerance then ConditionLimit validation order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Validation precedes Matrix nil and shape checks by public contract.
//
// AI-Hints:
//   - Never accept +Inf as an implicit disabling sentinel; only 0 disables condition rejection.
func factorValidatedOptions(options FactorOptions) error {
	if math.IsNaN(options.RankTolerance) || math.IsInf(options.RankTolerance, 0) || options.RankTolerance < 0 {
		return ErrInvalidOption
	}
	if math.IsNaN(options.ConditionLimit) || math.IsInf(options.ConditionLimit, 0) ||
		(options.ConditionLimit != 0 && options.ConditionLimit < factorUnity) {
		return ErrInvalidOption
	}
	return nil
}

// factorRankCutoff resolves the relative numerical rank threshold.
// A zero policy needs a predictable machine-epsilon cutoff that depends on shape.
// Implementation:
//   - Stage 1: Return an explicitly positive user tolerance unchanged.
//   - Stage 2: Otherwise choose the larger validated dimension.
//   - Stage 3: Multiply by binary64 machine epsilon and return the relative cutoff.
//
// Behavior highlights:
//   - This threshold is heuristic numerical rank, not exact algebraic rank.
//
// Inputs:
//   - options is validated; rows and cols are positive feasible dimensions.
//
// Returns:
//   - Finite dimensionless relative cutoff.
//
// Errors:
//   - None for validated dimensions.
//
// Determinism:
//   - The max dimension selection is deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Equality against a pivot-to-largest ratio counts as rank loss.
//
// AI-Hints:
//   - Do not multiply a cutoff by a tiny diagonal before comparing: ratio comparison avoids underflow.
func factorRankCutoff(options FactorOptions, rows, cols int) float64 {
	if options.RankTolerance > 0 {
		return options.RankTolerance
	}
	if rows < cols {
		rows = cols
	}
	return factorMachineEpsilon * float64(rows)
}

// factorCheckedElements verifies a positive matrix shape and safe int product.
// It prevents a wrapped element count from reaching make or a flat-slice offset.
// Implementation:
//   - Stage 1: Reject zero or negative rows or columns.
//   - Stage 2: Check rows <= MaxInt/cols before any multiplication.
//   - Stage 3: Return the exact row-major element count.
//
// Behavior highlights:
//   - This checks arithmetic size, not physical RAM availability.
//
// Inputs:
//   - rows and cols are signed requested matrix dimensions.
//
// Returns:
//   - Safe positive int rows*cols.
//
// Errors:
//   - ErrBadShape or ErrInvalidDimensions.
//
// Determinism:
//   - Dimension order is checked before multiplying.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Even valid element counts can exceed available process memory.
//
// AI-Hints:
//   - Never recover from OOM to disguise an unchecked allocation.
func factorCheckedElements(rows, cols int) (int, error) {
	if rows <= 0 || cols <= 0 {
		return 0, ErrBadShape
	}
	return matrixCheckedElementCount(rows, cols)
}

// factorWalkFinite validates every coordinate with one shared access kernel.
// It optionally copies values into caller-owned row-major destination storage;
// nil destination means validation-only and never allocates a full input copy.
// Implementation:
//   - Stage 1: detect *Dense by type assertion and validate flat-buffer length.
//   - Stage 2: scan all cells in row-major order, checking ctx and finiteness.
//   - Stage 3: optionally write the exact float64 source bits into destination.
//
// Behavior highlights: Generic Matrix.At errors preserve their original cause.
// Inputs: validated rows,cols; read-only Matrix; optional exact-sized destination.
// Returns: nil on complete validation, never a partially published matrix.
// Errors: ErrInvalidDimensions, ErrNaNInf, ctx.Err(), wrapped Matrix.At errors.
// Determinism: Increasing flat offsets for Dense, increasing i then j otherwise.
// Complexity: O(rows*cols) time; O(1) auxiliary space excluding destination.
// Notes: Caller mutation during the scan is unsupported even with a Dense fast path.
// AI-Hints: Do not use reflect or validate/copy using different numeric policies.
func factorWalkFinite(ctx context.Context, input Matrix, rows, cols int, destination []float64) error {
	if dense, ok := input.(*Dense); ok {
		if len(dense.data) != rows*cols {
			return ErrInvalidDimensions
		}
		for i, value := range dense.data {
			if i&8191 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return ErrNaNInf
			}
			if destination != nil {
				destination[i] = value
			}
		}
		return nil
	}
	for i := 0; i < rows; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		for j := 0; j < cols; j++ {
			value, readErr := input.At(i, j)
			if readErr != nil {
				return fmt.Errorf("matrix.At(%d,%d): %w", i, j, readErr)
			}
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return ErrNaNInf
			}
			if destination != nil {
				destination[i*cols+j] = value
			}
		}
	}
	return nil
}

// factorReadMatrix snapshots a validated shape into private row-major storage.
// It delegates all finite-value and backend-error rules to factorWalkFinite,
// so the optional validation-only path and the owning-copy path cannot drift.
// Implementation:
//   - Stage 1: validate context and nil/typed-nil matrix presence.
//   - Stage 2: validate dimensions and integer element-count arithmetic.
//   - Stage 3: allocate exactly one detached copy and fill it through shared validation.
//
// Behavior highlights: No input mutation or data-dependent iteration order.
// Inputs: ctx non-nil, caller-owned Matrix, expected positive rows and columns.
// Returns: Detached row-major []float64 on success; nil on any failure.
// Errors: ErrNilMatrix, ErrDimensionMismatch, ErrInvalidDimensions,
// ErrNaNInf, ctx.Err() and wrapped backend errors.
// Determinism: Fixed row-major coordinate order for both storage backends.
// Complexity: Time O(rows*cols), Space O(rows*cols) for the owned snapshot.
// AI-Hints: Retain one original snapshot for residual evaluation; never
// recompute residuals against an already overwritten packed factor buffer.
func factorReadMatrix(ctx context.Context, input Matrix, rows, cols int) ([]float64, error) {
	if err := factorContextErr(ctx); err != nil {
		return nil, err
	}
	if err := ValidateNotNil(input); err != nil {
		return nil, err
	}
	if input.Rows() != rows || input.Cols() != cols {
		return nil, ErrDimensionMismatch
	}
	elements, err := factorCheckedElements(rows, cols)
	if err != nil {
		return nil, err
	}
	data := make([]float64, elements)
	if err = factorWalkFinite(ctx, input, rows, cols, data); err != nil {
		return nil, err
	}
	return data, nil
}

// factorInputShape establishes presence and valid dimensions before data access.
// The caller applies factor-specific square/tall restrictions after this shared check.
// Implementation:
//   - Stage 1: Call ValidateNotNil including recognized typed-nil Matrix values.
//   - Stage 2: Read positive rows and cols once.
//   - Stage 3: Validate the product with factorCheckedElements.
//
// Behavior highlights:
//   - It does not read data or allocate a packed factor.
//
// Inputs:
//   - input is caller-owned and must not be concurrently mutated.
//
// Returns:
//   - Positive rows, cols, and nil error on valid input.
//
// Errors:
//   - ErrNilMatrix, ErrBadShape or ErrInvalidDimensions.
//
// Determinism:
//   - Presence checks strictly precede shape arithmetic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Generic typed nils must follow the established core.Nilable convention.
//
// AI-Hints:
//   - Do not infer shape from an unvalidated interface receiver.
func factorInputShape(input Matrix) (int, int, error) {
	if err := ValidateNotNil(input); err != nil {
		return 0, 0, err
	}
	rows, cols := input.Rows(), input.Cols()
	if _, err := factorCheckedElements(rows, cols); err != nil {
		return 0, 0, err
	}
	return rows, cols, nil
}

// factorScaleAndNormalize makes a private magnitude-scaled coefficient copy.
// The normalized matrix avoids an unnecessary overflow in elimination products.
// Implementation:
//   - Stage 1: Find maximum absolute finite entry.
//   - Stage 2: Reject an all-zero matrix with rank deficiency.
//   - Stage 3: Divide each original coefficient once into a detached output slice.
//
// Behavior highlights:
//   - No source values or signs are overwritten.
//
// Inputs:
//   - original is a finite caller-independent flattened coefficient snapshot.
//
// Returns:
//   - Normalized copy, positive original scale, or rank-deficient error.
//
// Errors:
//   - ErrRankDeficient only for an identically zero input.
//
// Determinism:
//   - Two fixed ascending flat-index scans.
//
// Complexity:
//   - Time O(rows*cols), Space O(rows*cols).
//
// Notes:
//   - Scaling does not change the exact solution of a consistently scaled RHS.
//
// AI-Hints:
//   - Never normalize columns independently inside LU: that would change pivot and rank semantics.
func factorScaleAndNormalize(original []float64) ([]float64, float64, error) {
	maximum := 0.0
	for _, value := range original {
		if magnitude := math.Abs(value); magnitude > maximum {
			maximum = magnitude
		}
	}
	if maximum == 0 {
		return nil, 0, ErrRankDeficient
	}
	packed := make([]float64, len(original))
	for i, value := range original {
		packed[i] = value / maximum
	}
	return packed, maximum, nil
}

// factorCheckFiniteOutput validates a required scalar output of a numerical kernel.
// It separates computed arithmetic overflow from invalid source observations.
// Implementation:
//   - Stage 1: Inspect NaN status.
//   - Stage 2: Inspect positive or negative infinity status.
//   - Stage 3: Return ErrOverflow when either check fails, nil otherwise.
//
// Behavior highlights:
//   - A finite subnormal or signed zero is not an overflow.
//
// Inputs:
//   - value is the computed binary64 scalar to publish.
//
// Returns:
//   - nil for finite values; otherwise ErrOverflow.
//
// Errors:
//   - ErrOverflow for an unrepresentable or invalid computed value.
//
// Determinism:
//   - Pure scalar check, independent of input-order traversal.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Input scans must use ErrNaNInf before arithmetic begins.
//
// AI-Hints:
//   - Never return an infinite ConditionEstimate as successful diagnostic output.
func factorCheckFiniteOutput(value float64) error {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return ErrOverflow
	}
	return nil
}
