// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats declares stable, package-owned error classifications.
// This file fixes the sentinel identities used across fitting and reductions.
package stats

import "errors"

var (
	// ErrNilInput reports a nil receiver, nil slice, or conventionally typed-nil matrix.
	// Matrix input checks also preserve matrix's original nil sentinel via wrapping.
	ErrNilInput = errors.New("stats: nil input")

	// ErrEmptyInput reports a non-nil empty slice or a training matrix with zero extent.
	// Inference with zero rows is supported and does not trigger this error.
	ErrEmptyInput = errors.New("stats: empty input")

	// ErrShape reports invalid dimensions, product overflow, or feature mismatch.
	// It is checked before reading matrix elements or publishing output matrices.
	ErrShape = errors.New("stats: shape mismatch")

	// ErrInvalidOption reports an unsupported DDOF or ConstantPolicy configuration.
	// The same sentinel rejects a non-nil accumulator's self-merge.
	ErrInvalidOption = errors.New("stats: invalid option")

	// ErrNonFinite reports NaN or either infinity in an observed or inferred value.
	// A rejected observation never changes a Moments accumulator's state.
	ErrNonFinite = errors.New("stats: non-finite value")

	// ErrInsufficientSamples reports Count<=DDOF or training rows<=DDOF.
	// Sample variance therefore rejects one observation but population variance permits it.
	ErrInsufficientSamples = errors.New("stats: insufficient samples")

	// ErrZeroVariance reports an exactly constant training feature under RejectConstant.
	// It is not triggered by arbitrary epsilon or rounded-to-zero variance alone.
	ErrZeroVariance = errors.New("stats: zero variance")

	// ErrOverflow reports unrepresentable final float64 statistics/scales/results
	// or accumulation counters exceeding MaxUint64; intermediate M2 is exempt.
	ErrOverflow = errors.New("stats: overflow")
)
