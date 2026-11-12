// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pca owns stable error classifications for the fitted spectral API.
// Dependency failures retain their original sentinel identity through errors.Join.
package pca

import (
	"errors"

	"github.com/lvlath/go/stats"
)

var (
	// ErrNilInput identifies a missing Matrix or nil/unfitted Model receiver.
	// It is checked after context/options, before reading Matrix dimensions.
	ErrNilInput = errors.New("pca: nil input")

	// ErrEmptyInput identifies a zero feature extent or zero training rows.
	// Zero-row Transform/InverseTransform remains legal when width is correct.
	ErrEmptyInput = errors.New("pca: empty input")

	// ErrShape identifies an invalid matrix shape, inadequate training row count,
	// or a feature/score width inconsistent with the trained Model.
	ErrShape = errors.New("pca: invalid shape")

	// ErrInvalidOption identifies a nil context, invalid tolerance/count, or
	// unknown Solver enum; it precedes any access to caller-owned matrices.
	ErrInvalidOption = errors.New("pca: invalid option")

	// ErrNonFinite identifies NaN or either infinity in caller-provided input.
	// It is distinct from overflow in internal arithmetic of finite operands.
	ErrNonFinite = errors.New("pca: non-finite input")

	// ErrZeroVariance rejects all-constant training data, or any constant
	// feature when Standardize=true requires stats.RejectConstant.
	ErrZeroVariance = errors.New("pca: zero training variance")

	// ErrRankDeficient identifies an unattainable retained component count or
	// an empty numerical-rank selection under the explicit spectral tolerance.
	ErrRankDeficient = errors.New("pca: rank deficient")

	// ErrNotConverged identifies a Jacobi solver that exhausted its rotations.
	// A returned failure also retains matrix.ErrMatrixEigenFailed via errors.Is.
	ErrNotConverged = errors.New("pca: eigensolver did not converge")

	// ErrOverflow identifies unrepresentable integer allocation sizes or finite
	// numerical results; it does not silently convert failures into zeroes.
	ErrOverflow = errors.New("pca: arithmetic overflow")

	// ErrNumerical identifies invalid covariance PSD/symmetry, malformed fitted
	// dependency artifacts, or failed eigenpair residual and orthogonality checks.
	ErrNumerical = errors.New("pca: numerical invariant violation")

	// ErrUnsupportedSolver rejects direct thin SVD, which this package does not
	// implement; it never downgrades the request to covariance diagonalization.
	ErrUnsupportedSolver = errors.New("pca: unsupported solver")
)

// normalizeStatsError adds a PCA sentinel to an underlying fitted-statistics error.
// The original stats sentinel remains discoverable through errors.Is, including
// for a wrapped lower-level cause; unrecognized errors pass through unchanged.
//
// Implementation:
//   - Stage 1: determine the first matching stats error class in fixed order.
//   - Stage 2: join it with the corresponding public PCA classification.
//   - Stage 3: leave errors without a matching stats category unmodified.
//
// Inputs: err is an arbitrary dependency failure, possibly nil or wrapped.
// Returns: the original error or an errors.Join preserving both identities.
// Errors: it returns errors rather than raising new failures.
// Determinism: exact error-category order below, independent of error messages.
// Complexity: O(1) predicate checks and constant-size error wrapping.
// Notes: valid Fit has already scanned finite input before stats is invoked.
// AI-Hints: never compare dependency error strings or discard their original sentinels.
func normalizeStatsError(err error) error {
	switch {
	case errors.Is(err, stats.ErrZeroVariance):
		return errors.Join(ErrZeroVariance, err)
	case errors.Is(err, stats.ErrNonFinite):
		return errors.Join(ErrNonFinite, err)
	case errors.Is(err, stats.ErrOverflow):
		return errors.Join(ErrOverflow, err)
	case errors.Is(err, stats.ErrInvalidOption):
		return errors.Join(ErrInvalidOption, err)
	case errors.Is(err, stats.ErrShape), errors.Is(err, stats.ErrInsufficientSamples):
		return errors.Join(ErrShape, err)
	case errors.Is(err, stats.ErrNilInput):
		return errors.Join(ErrNilInput, err)
	case errors.Is(err, stats.ErrEmptyInput):
		return errors.Join(ErrEmptyInput, err)
	default:
		return err
	}
}
