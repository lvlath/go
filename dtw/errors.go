// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw centralizes sentinel errors for Dynamic Time Warping.
// Error strings are diagnostics; callers must classify failures with errors.Is.
package dtw

import "errors"

var (
	// ErrNilInput indicates that one or both scalar input sequence slices are nil.
	ErrNilInput = errors.New("dtw: nil input sequence")

	// ErrEmptyInput indicates an empty scalar sequence or an empty local-cost matrix domain.
	ErrEmptyInput = errors.New("dtw: input sequences must be non-empty")

	// ErrNilOptions indicates that a legacy *Options receiver is nil where nil is not accepted.
	ErrNilOptions = errors.New("dtw: nil options")

	// ErrNilOption indicates that a nil functional Option was passed to a canonical facade.
	ErrNilOption = errors.New("dtw: nil option")

	// ErrBadInput indicates an invalid option, adapter input, shape, or matrix operation context.
	ErrBadInput = errors.New("dtw: invalid input")

	// ErrInvalidWindow indicates an invalid Sakoe-Chiba window value.
	ErrInvalidWindow = errors.New("dtw: invalid window")

	// ErrInvalidPenalty indicates an invalid slope penalty.
	ErrInvalidPenalty = errors.New("dtw: invalid slope penalty")

	// ErrNaNInf indicates that NaN or Inf appeared where finite values are required.
	ErrNaNInf = errors.New("dtw: NaN or Inf encountered")

	// ErrOverflow indicates a reachable accumulated or local cost outside finite float64.
	// A terminal optimal overflow returns nil Result and is joined with ErrNaNInf
	// for compatibility with the existing family of numerical errors.
	// Overflow in a discarded transition does not by itself fail an alignment.
	ErrOverflow = errors.New("dtw: optimal cost exceeds finite float64 range")

	// ErrNoPath indicates that no admissible warping path exists under the active policy.
	ErrNoPath = errors.New("dtw: no admissible warping path")

	// ErrPathNeedsMatrix indicates that path reconstruction requires full accumulated-cost storage.
	ErrPathNeedsMatrix = errors.New("dtw: path tracking requires FullMatrix")

	// ErrPathNotTracked indicates that a caller requested a path from a result that did not track paths.
	ErrPathNotTracked = errors.New("dtw: path was not tracked")

	// ErrIncompletePath indicates that backtracking could not reconstruct a valid path to the origin.
	ErrIncompletePath = errors.New("dtw: path computation incomplete")

	// ErrNilResult indicates that a nil *Result receiver was used.
	ErrNilResult = errors.New("dtw: nil result")

	// ErrNilCostFunc indicates that the scalar local-cost callback is nil.
	ErrNilCostFunc = errors.New("dtw: nil cost function")

	// ErrNegativeCost indicates that a scalar, matrix, or precomputed local cost is negative.
	ErrNegativeCost = errors.New("dtw: negative local cost")
)
