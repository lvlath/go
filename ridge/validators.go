// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge centralizes context, numeric, input, and allocation validation.
// These gates establish precedence before augmented-QR memory is allocated.
package ridge

import (
	"context"
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

const (
	// contextCheckInterval bounds row/feature work between cancellation polls.
	contextCheckInterval = 64
	// interceptColumns is the single unpenalized affine design column.
	interceptColumns = 1
	// bytesPerFloat64 is the storage width of an augmented dense matrix cell.
	bytesPerFloat64 = 8
)

// finite classifies float64 values without changing their numeric domain.
// Signed zeros and subnormals are finite; NaN and both infinities are invalid.
// Implementation:
//   - Stage 1: identify NaN.
//   - Stage 2: identify either infinity.
//   - Stage 3: return the combined finiteness predicate.
//
// Behavior highlights:
//   - This is classification only, not numeric sanitation.
//
// Inputs:
//   - value is a binary64 number from any validated domain.
//
// Returns:
//   - bool, true exactly when value is finite.
//
// Errors:
//   - None; the caller determines the error category by value provenance.
//
// Determinism:
//   - pure IEEE-754 classification, with no mutable state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Finite operands can still overflow during arithmetic.
//
// AI-Hints:
//   - Do not reject subnormals as effectively zero with a magic epsilon.
func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

// validateContext checks an operation's context before reading any input.
// It preserves cancellation identity, rather than replacing it with a ridge sentinel.
// Implementation:
//   - Stage 1: test for a missing context without calling a method on nil.
//   - Stage 2: read ctx.Err() for an already-canceled operation.
//   - Stage 3: return nil only while the observed context is active.
//
// Behavior highlights:
//   - Cancellation takes precedence over invalid Lambda and nil data.
//
// Inputs:
//   - ctx is a caller-owned context, required to be non-nil and active.
//
// Returns:
//   - nil or an error; no result is partially published.
//
// Errors:
//   - ErrInvalidOption for nil ctx; ctx.Err() for canceled/deadline contexts.
//
// Determinism:
//   - one context-state observation per call.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Long stages poll the same context at bounded row/column intervals.
//
// AI-Hints:
//   - A nil context is not an empty background context.
func validateContext(ctx context.Context) error {
	if ctx == nil {
		return ErrInvalidOption
	}
	return ctx.Err()
}

// validateOptions enforces explicit strictly positive regularization.
// It also validates the QR rank cutoff before any matrix method is invoked.
// Implementation:
//   - Stage 1: require finite Lambda strictly greater than zero.
//   - Stage 2: require finite nonnegative RankTolerance.
//   - Stage 3: accept the value-only basis policy without normalization.
//
// Behavior highlights:
//   - No zero-Lambda OLS fallback and no hidden tolerance replacement.
//
// Inputs:
//   - opts is the caller's value-copy ridge configuration.
//
// Returns:
//   - nil on valid options or ErrInvalidOption otherwise.
//
// Errors:
//   - ErrInvalidOption on either invalid numeric option.
//
// Determinism:
//   - fixed option validation order, with Lambda checked first.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - RankTolerance=0 delegates the matrix solver's automatic relative cutoff.
//
// AI-Hints:
//   - Never clamp a negative Lambda to zero or silently select another algorithm.
func validateOptions(opts Options) error {
	if !finite(opts.Lambda) || opts.Lambda <= 0 {
		return ErrInvalidOption
	}
	if !finite(opts.RankTolerance) || opts.RankTolerance < 0 {
		return ErrInvalidOption
	}
	return nil
}

// validateMatrixPresence checks an interface for nil and library typed-nil data.
// It retains the underlying matrix nil classification alongside ridge.ErrNilInput.
// Implementation:
//   - Stage 1: reject an absent interface without accessing methods.
//   - Stage 2: delegate core.Nilable detection to matrix.ValidateNotNil.
//   - Stage 3: wrap that dependency error with the ridge-level nil sentinel.
//
// Behavior highlights:
//   - No reflection or runtime pointer-introspection is used.
//
// Inputs:
//   - x is a possibly absent matrix.Matrix implementation.
//
// Returns:
//   - nil if present; otherwise a classified error.
//
// Errors:
//   - ErrNilInput plus matrix.ErrNilMatrix when the dependency detects absence.
//
// Determinism:
//   - fixed nil checks, O(1), no matrix element access.
//
// Complexity:
//   - Time O(1), Space O(1), excluding error wrapping.
//
// Notes:
//   - Custom typed-nil Matrix implementations should implement core.Nilable.
//
// AI-Hints:
//   - A custom Matrix without Nilable must guarantee nil-safe Rows and Cols.
func validateMatrixPresence(x matrix.Matrix) error {
	if x == nil {
		return ErrNilInput
	}
	if err := matrix.ValidateNotNil(x); err != nil {
		return errors.Join(ErrNilInput, err)
	}
	return nil
}

// validateTrainingShape validates shape and response ownership without data reads.
// It reserves numerical errors for value validation after all size checks pass.
// Implementation:
//   - Stage 1: reject nil/typed-nil x and nil y.
//   - Stage 2: require n,p>=1, a nonempty response, and len(y)=n.
//   - Stage 3: return the dimensions for checked augmented allocation.
//
// Behavior highlights:
//   - A wide original design is valid; no n>=p restriction applies.
//
// Inputs:
//   - x is a read-only n-by-p Matrix; y a read-only n-vector.
//
// Returns:
//   - positive n,p and nil, or zeros and the appropriate sentinel.
//
// Errors:
//   - ErrNilInput, ErrShape for negative/mismatched counts, ErrEmptyInput for zero size.
//
// Determinism:
//   - matrix presence precedes all dimension reads; counts read once.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Dense element-count validation occurs separately before any value scan.
//
// AI-Hints:
//   - Do not reject p>n: n+p >= p+1 for n>=1.
func validateTrainingShape(x matrix.Matrix, y []float64) (int, int, error) {
	if err := validateMatrixPresence(x); err != nil {
		return 0, 0, err
	}
	if y == nil {
		return 0, 0, ErrNilInput
	}
	n, p := x.Rows(), x.Cols()
	if n < 0 || p < 0 {
		return 0, 0, ErrShape
	}
	if n == 0 || p == 0 || len(y) == 0 {
		return 0, 0, ErrEmptyInput
	}
	if len(y) != n {
		return 0, 0, ErrShape
	}
	return n, p, nil
}

// checkAugmentedSize rejects dimensions that cannot fit in dense float64 storage.
// It guards both rows and columns before multiplication, including byte counts.
// Implementation:
//   - Stage 1: check n+p and optional p+1 against math.MaxInt.
//   - Stage 2: check (n+p)*q float64 elements via integer division.
//   - Stage 3: reject the impossible shape before reading an observation or allocating.
//
// Behavior highlights:
//   - The right-hand side (n+p)-by-1 is covered by the larger matrix bound.
//
// Inputs:
//   - n,p are positive validated counts; noIntercept selects q=p or p+1.
//
// Returns:
//   - nil for representable layout arithmetic; ErrOverflow otherwise.
//
// Errors:
//   - ErrOverflow for int or float64-byte element-count overflow.
//
// Determinism:
//   - fixed comparison order, without speculative multiplication.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This check cannot guarantee that a representable allocation fits RAM.
//
// AI-Hints:
//   - Checking (n+p)*q after multiplication is too late for signed int overflow.
func checkAugmentedSize(n, p int, noIntercept bool) error {
	if p > math.MaxInt-n {
		return ErrOverflow
	}
	q := p
	if !noIntercept {
		if p == math.MaxInt {
			return ErrOverflow
		}
		q += interceptColumns
	}
	rows := n + p
	if q > (math.MaxInt/bytesPerFloat64)/rows {
		return ErrOverflow
	}
	return nil
}

// validateTrainingValues scans the complete training domain for finite numbers.
// All input-value errors precede numerical work in training and optimization.
// Implementation:
//   - Stage 1: visit each row with cancellation polling.
//   - Stage 2: read all feature cells and then the corresponding y value.
//   - Stage 3: return only after every supplied observation is finite.
//
// Behavior highlights:
//   - Missing values are never silently imputed or dropped.
//
// Inputs:
//   - ctx active; x read-only n-by-p; y finite-length n-vector; shape prevalidated.
//
// Returns:
//   - nil for a fully valid data domain, otherwise its first error.
//
// Errors:
//   - ErrNonFinite for NaN/Inf, ctx.Err(), or Matrix.At error unchanged.
//
// Determinism:
//   - row i increasing, feature j increasing, then y[i].
//
// Complexity:
//   - Time O(n*p), Space O(1), no data copies.
//
// Notes:
//   - Caller mutation of x or y during the scan or fit is unsupported.
//
// AI-Hints:
//   - Do not fuse this with QR construction if error precedence must be stable.
func validateTrainingValues(ctx context.Context, x matrix.Matrix, y []float64, n, p int) error {
	for i := 0; i < n; i++ {
		if i%contextCheckInterval == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		for j := 0; j < p; j++ {
			if j%contextCheckInterval == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			value, err := x.At(i, j)
			if err != nil {
				return err
			}
			if !finite(value) {
				return ErrNonFinite
			}
		}
		if !finite(y[i]) {
			return ErrNonFinite
		}
	}
	return ctx.Err()
}
