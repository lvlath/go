// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File: errors.go
// Role: Package-owned error sentinels and their branchable numerical semantics.
// AI-Hint: Classify failures with errors.Is, not their text.

package sax

import "errors"

var (
	// ErrInvalidOption rejects invalid scalar policy, absent required custom
	// threshold identifiers, unsorted or non-finite custom threshold content,
	// and a segment count that exceeds the length of its source window.
	// It is also returned for a non-nil but uninitialized Encoder receiver.
	ErrInvalidOption = errors.New("sax: invalid option")

	// ErrNilInput reports an absent source []float64, Result, or Encoder pointer.
	// A typed-nil *Encoder receiver is rejected before any field dereference.
	ErrNilInput = errors.New("sax: nil input")

	// ErrEmptyInput distinguishes a present zero-length []float64 from nil.
	// It follows policy and nil validation and precedes the w>n shape gate.
	ErrEmptyInput = errors.New("sax: empty input")

	// ErrNonFinite rejects NaN or either infinity in source values and in
	// exposed Result numeric metadata/PAA that would corrupt comparisons.
	ErrNonFinite = errors.New("sax: non-finite observation")

	// ErrZeroVariance rejects exactly constant ZPopulation windows configured
	// with RejectConstant; Raw mode never raises it for constant observations.
	ErrZeroVariance = errors.New("sax: zero population variance")

	// ErrIncompatible rejects different length, segmentation, alphabet, policy
	// EncodingID, or normalization version before computing a symbolic distance.
	// A user-supplied BreakpointID is not used for this classification.
	ErrIncompatible = errors.New("sax: incompatible encoding")

	// ErrShape rejects wrong breakpoint counts, invalid Result slice lengths,
	// out-of-range symbol codes, or inconsistent public fit/diagnostic metadata.
	ErrShape = errors.New("sax: malformed shape")

	// ErrOverflow rejects an unrepresentable finite numerical result, including
	// stddev underflow on distinct observations or overflow in PAA/MINDIST.
	// If produced by paa.Transform, errors.Is also exposes paa.ErrOverflow.
	ErrOverflow = errors.New("sax: arithmetic overflow")
)
