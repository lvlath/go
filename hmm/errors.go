// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm declares stable package-owned error sentinels for inference and fitting.
// This boundary separates input validation, mathematical impossibility, and numeric failure.
package hmm

import "errors"

var (
	// ErrNilInput identifies an absent model, prior, matrix, or required observation slice.
	// It is returned before inspecting shape or numeric domains of the missing input.
	ErrNilInput = errors.New("hmm: nil input")

	// ErrEmptyInput identifies an allocated but empty observation sequence or training collection.
	// A model requires at least one observation for each independent sequence.
	ErrEmptyInput = errors.New("hmm: empty input")

	// ErrShape identifies dimension mismatches, invalid matrix shapes, or an out-of-alphabet symbol.
	// In Gaussian calls, every observation row must have exactly D coordinates.
	ErrShape = errors.New("hmm: invalid shape")

	// ErrInvalidOption identifies a nil context, negative/nonfinite fitting option,
	// or an initial Gaussian variance below an explicitly required VarianceFloor.
	ErrInvalidOption = errors.New("hmm: invalid option")

	// ErrNonFinite identifies a NaN or infinite supplied parameter or observation.
	// Internal -Inf log probability for a true zero emission is not an input violation.
	ErrNonFinite = errors.New("hmm: nonfinite input")

	// ErrProbability identifies nonstochastic distributions or a nonpositive variance.
	// Zero categorical emission and zero transition probabilities remain valid domain values.
	ErrProbability = errors.New("hmm: invalid probability")

	// ErrImpossibleSequence identifies a categorical trajectory with zero evidence across all states.
	// It is distinct from overflow: a finite diagonal-Gaussian model has full real support.
	ErrImpossibleSequence = errors.New("hmm: impossible sequence")

	// ErrNotConverged indicates that EM exhausted MaxIterations without meeting the tolerance.
	// A last accepted immutable model and diagnostics may accompany this error.
	ErrNotConverged = errors.New("hmm: EM not converged")

	// ErrStateVersion identifies a mismatched, malformed, or unnormalized FilterState checkpoint.
	// A valid state belongs to exactly one canonical model configuration.
	ErrStateVersion = errors.New("hmm: invalid filter state")

	// ErrOverflow identifies arithmetic or checked-size overflow during inference or fitting.
	// It is not a synonym for an impossible sequence or a nonfinite input.
	ErrOverflow = errors.New("hmm: arithmetic overflow")

	// ErrNumerical identifies a failed EM monotonicity or smoothing normalization invariant.
	// A previously verified fit can be returned alongside the error for diagnostics.
	ErrNumerical = errors.New("hmm: numerical failure")
)
