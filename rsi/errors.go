// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rsi publishes stable sentinel error identities for Wilder RSI operations.
// This file keeps caller-input, arithmetic, and checkpoint-failure categories distinct.
package rsi

import "errors"

var (
	// ErrNilInput identifies a nil batch history or a nil Stream receiver.
	// It is distinct from an allocated empty price slice and from Stream{}.
	ErrNilInput = errors.New("rsi: nil input")

	// ErrEmptyInput identifies an allocated zero-length history passed to Compute.
	// A short, nonempty history is valid even when its entire mask remains false.
	ErrEmptyInput = errors.New("rsi: empty input")

	// ErrInvalidOption identifies an invalid smoothing period before price processing.
	// It includes negative periods and values for which Period+1 is unrepresentable.
	ErrInvalidOption = errors.New("rsi: invalid option")

	// ErrNonFinite identifies a NaN or infinite caller-supplied price.
	// These inputs are rejected before the finite-positive domain check.
	ErrNonFinite = errors.New("rsi: non-finite price")

	// ErrDomain identifies a finite but nonpositive caller-supplied price.
	// Neither zero nor negative prices are treated as missing bars or a default close.
	ErrDomain = errors.New("rsi: price must be positive")

	// ErrOverflow identifies an exhausted Count or unrepresentable numerical result.
	// It is never replaced with a manufactured RSI boundary value by clamping.
	ErrOverflow = errors.New("rsi: arithmetic overflow")

	// ErrStateVersion identifies a foreign, malformed, or numerically inconsistent checkpoint.
	// It also classifies Push/Restore calls on an uninitialized zero-value Stream.
	ErrStateVersion = errors.New("rsi: invalid state version or structure")
)
