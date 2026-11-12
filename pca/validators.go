// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca centralizes context, option, matrix-shape, finite-value, and
// checked-allocation gates. Neither training nor inference bypasses these laws.
package pca

import (
	"context"
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

const (
	// The default is a relative max-pivot tolerance, not an absolute threshold.
	defaultEigenTolerance = 1e-12
	// An omitted rotation limit allows 50 individual pivots per matrix element.
	defaultRotationMultiplier = 50
	// The binary64 unit roundoff reference used for numerical-rank policies.
	machineEpsilon = 0x1p-52
)

// checkContext verifies an operation context before inspecting caller data.
// Nil and already-canceled contexts have different, stable error classifications.
//
// Implementation:
//   - Stage 1: reject a nil context without dereferencing it.
//   - Stage 2: query ctx.Err() exactly once.
//   - Stage 3: return the cancellation reason or nil; do not allocate workspace.
//
// Behavior highlights:
//   - Nil context is a caller option error, not context.Canceled.
//
// Inputs:
//   - ctx: possibly nil context.Context; no ownership transfer.
//
// Returns:
//   - error: nil for an active context, otherwise the exact failure category.
//
// Errors:
//   - ErrInvalidOption: ctx is nil; ctx.Err(): caller canceled/deadline expired.
//
// Determinism:
//   - Nil check always precedes cancellation-status inspection.
//
// Complexity:
//   - Time O(1), Space O(1), no heap allocation by this helper.
//
// Notes:
//   - The matrix eigensolver lacks an in-kernel context hook; the caller
//     must also check cancellation between long-running algorithmic stages.
//
// AI-Hints:
//   - Do not call ctx.Err() before testing ctx == nil.
//   - Do not convert context cancellation into ErrInvalidOption.
func checkContext(ctx context.Context) error {
	if ctx == nil {
		return ErrInvalidOption
	}
	return ctx.Err()
}

// checkOptions validates the entire PCA configuration before accessing input.
// Solver admission is explicit and precedes all other user-supplied option fields.
//
// Implementation:
//   - Stage 1: reject the reserved, unimplemented ThinSVD solver.
//   - Stage 2: enforce the sole implemented solver, nonnegative component and
//     rotation counts, and finite nonnegative rank/eigen tolerances.
//   - Stage 3: return success without modifying or resolving the caller's value.
//
// Behavior highlights:
//   - The zero Options value selects CovarianceEigen with documented defaults.
//   - An unavailable solver never silently falls back to another algorithm.
//
// Inputs:
//   - opts: value-copy training policy; no shared mutable references.
//
// Returns:
//   - error: nil only for the admitted covariance-based configuration.
//
// Errors:
//   - ErrUnsupportedSolver: ThinSVD, even if other fields are invalid.
//   - ErrInvalidOption: unknown solver or invalid numeric/count options.
//
// Determinism:
//   - Enum check precedes scalar checks; no option is applied incrementally.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Zero tolerance and rotation defaults are resolved only when the
//     covariance dimension and norm become known in the numerical stages.
//
// AI-Hints:
//   - Never interpret a user-supplied NaN tolerance as an omitted default.
//   - Do not turn invalid k or unsupported SVD requests into a valid k silently.
func checkOptions(opts Options) error {
	if opts.Solver == ThinSVD {
		return ErrUnsupportedSolver
	}
	if opts.Solver != CovarianceEigen || opts.Components < 0 || opts.MaxRotations < 0 ||
		!finite(opts.RankTolerance) || opts.RankTolerance < 0 ||
		!finite(opts.EigenTolerance) || opts.EigenTolerance < 0 {
		return ErrInvalidOption
	}
	return nil
}

// finite reports whether a binary64 value belongs to PCA's finite domain.
// It distinguishes invalid caller observations from finite arithmetic overflow.
//
// Implementation:
//   - Stage 1: test for NaN using math.IsNaN.
//   - Stage 2: reject both infinities using math.IsInf with sign zero.
//   - Stage 3: accept all remaining values, including signed zero and subnormals.
//
// Behavior highlights:
//   - There is no arbitrary magnitude threshold or automatic zero clipping.
//
// Inputs:
//   - v: any float64 operand or computed result.
//
// Returns:
//   - bool: true exactly when v is neither NaN nor infinite.
//
// Errors:
//   - None; call sites choose ErrNonFinite or ErrOverflow by value provenance.
//
// Determinism:
//   - Pure comparison without state or allocation.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Finite does not establish accuracy, positive variance, or good conditioning.
//
// AI-Hints:
//   - Do not use finite to replace independent eigenpair or PSD verification.
func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// typedNilMatrix tests recognized Matrix nil receivers without reflection.
// The operation avoids accidental method calls on absent Dense/View pointers.
//
// Implementation:
//   - Stage 1: detect a nil interface before any type-specific access.
//   - Stage 2: inspect known *matrix.Dense and *matrix.View pointer values.
//   - Stage 3: consult a backend's explicit IsNil contract if one exists.
//
// Behavior highlights:
//   - An arbitrary Matrix implementation without Nilable follows its own
//     receiver semantics; Go cannot inspect its hidden pointer through reflect.
//
// Inputs:
//   - x: possibly typed-nil matrix.Matrix supplied by a caller or dependency.
//
// Returns:
//   - bool: true for a missing interface or a recognized nil backend.
//
// Errors:
//   - None; only safe type assertions and nil comparisons are used.
//
// Determinism:
//   - Type switch precedence is fixed; there is no dynamic registry.
//
// Complexity:
//   - Time O(1), Space O(1) for conventional backend IsNil methods.
//
// Notes:
//   - User-defined Nilable implementations must make IsNil nil-receiver safe.
//
// AI-Hints:
//   - Do not import reflect or recover from typed-nil panics.
//   - Check this guard before any Matrix.Rows or Matrix.Cols invocation.
func typedNilMatrix(x matrix.Matrix) bool {
	if x == nil {
		return true
	}
	switch m := x.(type) {
	case *matrix.Dense:
		return m == nil
	case *matrix.View:
		return m == nil
	case interface{ IsNil() bool }:
		return m.IsNil()
	default:
		return false
	}
}

// checkedFloatElements bounds a contiguous rows-by-cols float64 allocation.
// It rejects signed product and byte-count overflow before multiplying sizes.
//
// Implementation:
//   - Stage 1: reject negative dimensions as an invalid shape.
//   - Stage 2: compare rows with MaxInt/cols before element multiplication.
//   - Stage 3: bound element count by MaxInt/8 before returning it.
//
// Behavior highlights:
//   - Zero dimensions are arithmetically valid here; training shape rules
//     decide whether a zero extent is meaningful for a given operation.
//
// Inputs:
//   - rows, cols: signed nonnegative matrix extents.
//
// Returns:
//   - int: safe element count for float64 storage, without allocation.
//
// Errors:
//   - ErrShape: negative extent; ErrOverflow: element or byte-count overflow.
//
// Determinism:
//   - Fixed validation order independent of the configured matrix backend.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This protects arithmetic, not physical allocation against system OOM.
//   - It uses architecture-dependent MaxInt, not unsafe unsigned conversions.
//
// AI-Hints:
//   - Never compute rows*cols before checking the division bound.
//   - Do not catch runtime OOM with recover; validate feasible shape budgets.
func checkedFloatElements(rows, cols int) (int, error) {
	// math.MaxInt is architecture-aware and avoids unchecked uint-to-int casts.
	const maxInt = math.MaxInt
	if rows < 0 || cols < 0 {
		return 0, ErrShape
	}
	if cols != 0 && rows > maxInt/cols {
		return 0, ErrOverflow
	}
	count := rows * cols
	if count > maxInt/8 {
		return 0, ErrOverflow
	}
	return count, nil
}

// checkedShape validates matrix presence, shape, and storage feasibility.
// Training calls require at least two rows; inference admits zero-row batches
// when the downstream fitted feature/score width agrees with the model.
//
// Implementation:
//   - Stage 1: classify nil and conventionally typed-nil matrix receivers.
//   - Stage 2: read dimensions once and reject negative/empty/too-short shape.
//   - Stage 3: verify a representable row-major float64 element count.
//
// Behavior highlights:
//   - A zero column count is always ErrEmptyInput; zero rows are admitted
//     only for calls passing minimumRows=0.
//
// Inputs:
//   - x: caller-owned read-only matrix backend; minimumRows: required samples.
//
// Returns:
//   - (rows, cols, nil): dimension pair on success; zeros and error otherwise.
//
// Errors:
//   - ErrNilInput, ErrEmptyInput, ErrShape, ErrOverflow by precedence above.
//
// Determinism:
//   - Nil, signed shape, empty extent, sample sufficiency, size checks in order.
//
// Complexity:
//   - Time O(1), Space O(1) for O(1) Matrix.Rows/Cols implementations.
//
// Notes:
//   - The caller must additionally match cols to Model.d or Model.k.
//   - No data element is read and no matrix allocation occurs here.
//
// AI-Hints:
//   - Distinguish valid inference 0-by-k from invalid training 0-by-d.
//   - Do not treat typed-nil matrices as legitimate empty matrices.
func checkedShape(x matrix.Matrix, minimumRows int) (int, int, error) {
	if typedNilMatrix(x) {
		return 0, 0, ErrNilInput
	}
	r, c := x.Rows(), x.Cols()
	if r < 0 || c < 0 {
		return 0, 0, ErrShape
	}
	if r == 0 && minimumRows > 0 || c == 0 {
		return 0, 0, ErrEmptyInput
	}
	if r < minimumRows {
		return 0, 0, ErrShape
	}
	if _, err := checkedFloatElements(r, c); err != nil {
		return 0, 0, err
	}
	return r, c, nil
}

// scanFinite establishes the caller-observation numeric domain before computation.
// It reads every cell in a stable order so a NaN/Inf outranks domain or rank
// failures on otherwise structurally valid training input.
//
// Implementation:
//   - Stage 1: check cancellation at the beginning of each observation row.
//   - Stage 2: read cells in increasing feature index and classify values.
//   - Stage 3: return the first backend or nonfinite error, or nil on success.
//
// Behavior highlights:
//   - No sanitization, clamping, or substitution occurs.
//   - Backend At failures retain their original error identities.
//
// Inputs:
//   - ctx: active context; x: caller-owned read-only matrix.
//   - rows, cols: previously checked dimensions with row-major storage capacity.
//
// Returns:
//   - error: first invalid cell/reading/cancellation or nil if all finite.
//
// Errors:
//   - ErrNonFinite: caller-provided NaN or infinity.
//   - ctx.Err() or Matrix.At error: cancellation or read failure.
//
// Determinism:
//   - Rows ascend and each row's features ascend; the first invalid cell wins.
//
// Complexity:
//   - Time O(rows*cols), Space O(1), assuming O(1) Matrix.At.
//
// Notes:
//   - The matrix must remain stable during later preparation and computation;
//     otherwise the initial finite scan is not a snapshot guarantee.
//
// AI-Hints:
//   - Do not replace this with a clone: Matrix.Clone cannot return an error.
//   - Never recast invalid source values as arithmetic ErrOverflow.
func scanFinite(ctx context.Context, x matrix.Matrix, rows, cols int) error {
	for i := 0; i < rows; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		for j := 0; j < cols; j++ {
			v, err := x.At(i, j)
			if err != nil {
				return err
			}
			if !finite(v) {
				return ErrNonFinite
			}
		}
	}
	return nil
}

// wrapDependency retains underlying matrix errors while adding PCA categories.
// Callers can inspect either the package-level failure or its source sentinel
// through errors.Is, without parsing diagnostic strings.
//
// Implementation:
//   - Stage 1: return a nil error unchanged.
//   - Stage 2: match matrix failures by a fixed precedence, especially
//     ErrMatrixEigenFailed before generic shape/numeric categories.
//   - Stage 3: use errors.Join for matching PCA+matrix classifications or
//     return an unknown dependency error unchanged.
//
// Behavior highlights:
//   - Jacobi budget exhaustion is ErrNotConverged, not ErrNumerical.
//   - A failed covariance allocation preserves matrix.ErrOverflow as well.
//
// Inputs:
//   - err: arbitrary nil, wrapped, or joined matrix dependency failure.
//
// Returns:
//   - error: unchanged root or joined PCA category plus original root.
//
// Errors:
//   - ErrNotConverged, ErrOverflow, ErrNonFinite, ErrNilInput, ErrShape,
//     ErrNumerical are added only for their corresponding matrix root errors.
//
// Determinism:
//   - First matching sentinel follows the fixed switch order, not text.
//
// Complexity:
//   - Time O(1) error-chain checks for bounded chains; Space O(1) wrapper size.
//
// Notes:
//   - Unknown errors are returned, not hidden behind a generic PCA sentinel.
//
// AI-Hints:
//   - Do not compare err==sentinel when a dependency may wrap its failures.
//   - Never discard matrix.ErrMatrixEigenFailed from the error chain.
func wrapDependency(err error) error {
	if err == nil {
		return nil
	}
	switch {
	case errors.Is(err, matrix.ErrMatrixEigenFailed):
		return errors.Join(ErrNotConverged, err)
	case errors.Is(err, matrix.ErrOverflow):
		return errors.Join(ErrOverflow, err)
	case errors.Is(err, matrix.ErrNaNInf):
		return errors.Join(ErrNonFinite, err)
	case errors.Is(err, matrix.ErrNilMatrix):
		return errors.Join(ErrNilInput, err)
	case errors.Is(err, matrix.ErrInvalidDimensions), errors.Is(err, matrix.ErrDimensionMismatch):
		return errors.Join(ErrShape, err)
	case errors.Is(err, matrix.ErrAsymmetry):
		return errors.Join(ErrNumerical, err)
	default:
		return err
	}
}

// checkedRotations resolves Jacobi's explicit individual-pivot budget.
// It guards the default 50*d*d calculation against signed integer overflow
// before a potentially large covariance or eigensolver workspace is created.
//
// Implementation:
//   - Stage 1: reject nonpositive dimensions and negative user limits.
//   - Stage 2: accept positive explicit rotation counts unchanged.
//   - Stage 3: for zero default, check d*d and multiplier arithmetic before use.
//
// Behavior highlights:
//   - The limit counts rotations, never sweeps or unlimited iterations.
//
// Inputs:
//   - d: feature dimension (>0); requested: nonnegative user rotation budget.
//
// Returns:
//   - int: positive safe rotation count, or zero with a classified error.
//
// Errors:
//   - ErrShape: invalid d; ErrInvalidOption: negative requested count.
//   - ErrOverflow: default d*d or 50*d*d exceeds MaxInt.
//
// Determinism:
//   - Positive user count always wins over defaults; no dynamic adaptation.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - No attempt is made to infer a larger limit from convergence failures.
//
// AI-Hints:
//   - Do not compute 50*d*d before testing both multiplication bounds.
//   - No hidden extra Jacobi sweep is permitted after budget exhaustion.
func checkedRotations(d, requested int) (int, error) {
	// The facade has already required d>=1; keep the arithmetic safe even if
	// an internal caller changes this order in a later refactor.
	if d <= 0 {
		return 0, ErrShape
	}
	if requested < 0 {
		return 0, ErrInvalidOption
	}
	if requested != 0 {
		return requested, nil
	}
	const maxI = math.MaxInt
	if d > maxI/d || d*d > maxI/defaultRotationMultiplier {
		return 0, ErrOverflow
	}
	return defaultRotationMultiplier * d * d, nil
}
