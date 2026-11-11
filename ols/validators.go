// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols enforces the input and numeric preconditions of its regression API.
// This file owns validation order and safe matrix interface handling.
package ols

import (
	"context"
	"math"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

const (
	// contextCheckRowInterval bounds the number of rows processed between
	// cancellation polls in loops under the package's direct control.
	contextCheckRowInterval = 128

	// interceptDesignColumns is the one explicit affine-basis column.
	// It is deliberately excluded when NoIntercept fixes the origin.
	interceptDesignColumns = 1
)

// finite classifies IEEE-754 values for the OLS input and result domains.
// It accepts signed zero and subnormal values while refusing NaN and both
// infinities without silently coercing observations into a different domain.
//
// Implementation:
//   - Stage 1: check the NaN predicate on the supplied binary64 value.
//   - Stage 2: reject either sign of infinity using the IEEE infinity predicate.
//   - Stage 3: report whether the value is representable as a finite float64.
//
// Behavior highlights:
//   - This is a classification primitive, not a tolerance or sanitation policy.
//
// Inputs:
//   - value: input, option, intermediate, or result float64 in its own units.
//
// Returns:
//   - bool: true for finite values; false for NaN, +Inf, and -Inf.
//
// Errors:
//   - None; the calling stage selects ErrInvalidOption, ErrNonFinite,
//     or ErrOverflow according to the provenance of the value.
//
// Determinism:
//   - Pure floating-point classification; no mutable state or traversal order.
//
// Complexity:
//   - Time O(1), Space O(1); no allocation.
//
// Notes:
//   - Finite operands can still overflow when later combined by arithmetic.
//
// AI-Hints:
//   - Never translate NaN/Inf into zero or reject subnormals by a magic epsilon.
func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// contextErr checks operation cancellation before examining inputs or options.
// It preserves the standard context cancellation/deadline identities rather
// than wrapping them in an unrelated OLS domain sentinel.
//
// Implementation:
//   - Stage 1: reject a missing context without accessing ctx.Err().
//   - Stage 2: read ctx.Err() exactly once for active-context classification.
//   - Stage 3: return nil or the original context error to the caller.
//
// Behavior highlights:
//   - A canceled context has precedence over invalid rank options and nil data.
//   - Higher-level loops recheck at bounded row or QR pivot boundaries.
//
// Inputs:
//   - ctx: caller-owned context; must be non-nil and active at each observed stage.
//
// Returns:
//   - error: nil for an active context, otherwise a classified cancellation.
//
// Errors:
//   - ErrInvalidOption: nil context.
//   - context.Canceled/context.DeadlineExceeded: ctx.Err() without reclassification.
//
// Determinism:
//   - A single context-state sample; cancellation timing belongs to the caller.
//
// Complexity:
//   - Time O(1), Space O(1); no goroutines or allocations are started.
//
// Notes:
//   - This helper is not a wait or timeout generator; the caller owns deadlines.
//
// AI-Hints:
//   - Do not dereference ctx before checking nil or hide ctx.Err() under ErrShape.
func contextErr(ctx context.Context) error {
	if ctx == nil {
		return ErrInvalidOption
	}
	return ctx.Err()
}

// nilMatrix detects absent matrices without reflection or unsafe pointer tricks.
// It recognizes the library's concrete owning/view types and the optional
// core.Nilable convention for custom implementations of matrix.Matrix.
//
// Implementation:
//   - Stage 1: reject a nil interface immediately.
//   - Stage 2: honor core.Nilable on custom matrix implementations.
//   - Stage 3: check supported concrete pointer receivers before method calls.
//
// Behavior highlights:
//   - A custom typed-nil matrix lacking core.Nilable cannot be detected generically
//     without reflection; such implementers must provide nil-safe Rows/Cols/At.
//
// Inputs: x is a possibly nil matrix.Matrix interface.
// Returns: true for a recognized nil backing object; false for present matrices.
// Errors: None; Fit/Predict convert true into ErrNilInput.
// Determinism: fixed type checks with no matrix element access.
// Complexity: Time O(1), Space O(1).
// Notes: The View branch requires matrix.View to implement matrix.Matrix via Clone.
// AI-Hints: Never reintroduce reflect.ValueOf(x).IsNil to extend this helper.
func nilMatrix(x matrix.Matrix) bool {
	if x == nil {
		return true
	}
	if n, ok := x.(core.Nilable); ok {
		return n.IsNil()
	}
	switch v := x.(type) {
	case *matrix.Dense:
		return v == nil
	case *matrix.View:
		return v == nil
	default:
		return false
	}
}

// checkedShape checks nonnegative dimensions and representable dense storage.
// It rejects element-count and float64-byte multiplication overflow before make.
//
// Implementation:
//   - Stage 1: reject negative row/column counts.
//   - Stage 2: permit zero-element shapes (used by empty prediction batches).
//   - Stage 3: compare columns against the maximum safe count divided by rows.
//
// Behavior highlights: no multiplication is executed until overflow is excluded.
// Inputs: rows and cols are signed matrix dimensions.
// Returns: true when rows*cols*sizeof(float64) fits int, false otherwise.
// Errors: None; callers classify invalid shape as ErrShape.
// Determinism: fixed integer comparisons, independent of data order.
// Complexity: Time O(1), Space O(1).
// Notes: Checked dimensions cannot protect against physical OOM for huge valid allocations.
// AI-Hints: Do not compute rows*cols first and then check whether it overflowed.
func checkedShape(rows, cols int) bool {
	if rows < 0 || cols < 0 {
		return false
	}
	if rows == 0 || cols == 0 {
		return true
	}
	const bytesPerFloat64 = 8
	const maxInt = math.MaxInt
	return cols <= (maxInt/bytesPerFloat64)/rows
}

// validateFit applies the complete public error precedence and fit-shape laws.
// No design storage or QR factor is allocated before validation succeeds.
//
// Implementation:
//   - Stage 1: validate context and RankTolerance before reading input methods.
//   - Stage 2: resolve nil/typed-nil, nonempty shape, y length, and n>=q.
//   - Stage 3: scan finite x and y entries in row-major order with cancellation polls.
//
// Behavior highlights:
//   - All observations are validated; missing or invalid rows are never dropped.
//   - An intercept consumes exactly one extra unknown, including at n=q.
//
// Inputs:
//   - ctx: active non-nil training context.
//   - x: read-only n-by-p matrix, including independently implemented Matrix types.
//   - y: read-only length-n vector; nil is distinct from zero length.
//   - opts: rank and intercept policies, validated before matrix access.
//
// Returns: n observations, p predictors, q unknowns, and nil error; or zero counts/error.
// Errors: ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrShape, ErrNonFinite,
//
//	ctx.Err(), and wrapped Matrix.At errors in that stage order.
//
// Determinism: row i increasing, feature j increasing, then y[i] for each row.
// Complexity: Time O(n*p), Space O(1).
// Notes: Concurrent mutation of x/y is unsupported throughout Fit, not just validation.
// AI-Hints: Keep options-before-nil ordering; a negative RankTolerance must win over nil x.
func validateFit(ctx context.Context, x matrix.Matrix, y []float64, opts Options) (int, int, int, error) {
	if err := contextErr(ctx); err != nil {
		return 0, 0, 0, err
	}
	if !finite(opts.RankTolerance) || opts.RankTolerance < 0 {
		return 0, 0, 0, ErrInvalidOption
	}
	if nilMatrix(x) || y == nil {
		return 0, 0, 0, ErrNilInput
	}

	// Validate sizes before accessing observations or allocating the design.
	n, p := x.Rows(), x.Cols()
	if !checkedShape(n, p) {
		return 0, 0, 0, ErrShape
	}
	if n == 0 || p == 0 || len(y) == 0 {
		return 0, 0, 0, ErrEmptyInput
	}
	if len(y) != n {
		return 0, 0, 0, ErrShape
	}
	q := p
	if !opts.NoIntercept {
		if p == math.MaxInt {
			return 0, 0, 0, ErrShape
		}
		q += interceptDesignColumns
	}
	if n < q || !checkedShape(n, q) {
		return 0, 0, 0, ErrShape
	}

	// Preserve the specified error ordering: shape errors outrank nonfinite data.
	for i := 0; i < n; i++ {
		if i%contextCheckRowInterval == 0 {
			if err := ctx.Err(); err != nil {
				return 0, 0, 0, err
			}
		}
		for j := 0; j < p; j++ {
			value, err := x.At(i, j)
			if err != nil {
				return 0, 0, 0, err
			}
			if !finite(value) {
				return 0, 0, 0, ErrNonFinite
			}
		}
		if !finite(y[i]) {
			return 0, 0, 0, ErrNonFinite
		}
	}
	return n, p, q, nil
}
