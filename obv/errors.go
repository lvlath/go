// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package obv defines stable error identities for input, arithmetic and state failures.
// This file is the only source of exported sentinel errors for the package.
package obv

import "errors"

// Each error is a stable branching category; classify with errors.Is, never message text.
var (
	// ErrNilInput reports a missing batch column, nil slice, or nil stream mutator.
	ErrNilInput = errors.New("obv: nil input")

	// ErrEmptyInput reports a present but zero-length batch or candle slice.
	ErrEmptyInput = errors.New("obv: empty input")

	// ErrShape reports different OHLCV column lengths after presence validation.
	ErrShape = errors.New("obv: inconsistent OHLCV shape")

	// ErrNonFinite reports an OHLCV field containing NaN or either infinity.
	ErrNonFinite = errors.New("obv: nonfinite input")

	// ErrDomain reports nonpositive OHLC, negative volume, or invalid candle geometry.
	ErrDomain = errors.New("obv: invalid OHLCV domain")

	// ErrOverflow reports exhausted Count or a nonrepresentable compensated total.
	ErrOverflow = errors.New("obv: accumulator or count overflow")

	// ErrStateVersion reports incompatible snapshot ABI, fingerprint or numeric coordinates.
	ErrStateVersion = errors.New("obv: incompatible or invalid snapshot")
)
