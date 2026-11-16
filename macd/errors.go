// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd defines stable failure categories for its public batch and stream APIs.
// This file owns all sentinel instances used with errors.Is.
package macd

import "errors"

// Sentinel errors describe distinct caller-actionable failure categories.
// They contain no price values, snapshot contents, or other user-provided data.
//
// Behavior highlights:
//   - Input and option validation errors precede arithmetic work in Compute.
//   - Runtime arithmetic faults return ErrOverflow without partial state publication.
//   - Restore maps malformed and incompatible snapshots to ErrStateVersion.
//
// Determinism:
//   - Each category has one stable package-level error identity; use errors.Is.
//
// Complexity:
//   - Classification is O(1) for these unwrapped sentinel returns.
//
// AI-Hints:
//   - Never parse error strings or collapse a finite-domain fault into overflow.
var (
	// ErrNilInput means a nil close slice or nil Stream receiver was supplied.
	ErrNilInput = errors.New("macd: nil input")
	// ErrEmptyInput means Compute received an allocated but zero-length close slice.
	ErrEmptyInput = errors.New("macd: empty input")
	// ErrInvalidOption means a normalized period is invalid or warmup index would overflow int.
	ErrInvalidOption = errors.New("macd: invalid option")
	// ErrNonFinite means an input price was NaN or infinite before computation.
	ErrNonFinite = errors.New("macd: non-finite input")
	// ErrDomain means a finite input close was nonpositive.
	ErrDomain = errors.New("macd: input outside positive domain")
	// ErrOverflow means a required count or floating arithmetic result cannot be represented.
	ErrOverflow = errors.New("macd: arithmetic overflow")
	// ErrStateVersion means snapshot version, structure, phase, or recipient config is invalid.
	ErrStateVersion = errors.New("macd: incompatible or invalid state")
)
