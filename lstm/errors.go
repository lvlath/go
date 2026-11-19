// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm defines stable sentinel errors for invalid inputs and arithmetic failure.
// This file is the sole owner of exported error identities; callers use errors.Is.
package lstm

import "errors"

var (
	// ErrNilInput identifies a missing mandatory model, matrix, state or nonempty tensor.
	// An empty slice is valid only when the checked required element count is zero.
	ErrNilInput = errors.New("lstm: nil input")

	// ErrShape identifies malformed geometry or an overflowing element-count product.
	// Geometry is checked before mask lengths and numerical sample validation.
	ErrShape = errors.New("lstm: incompatible shape")

	// ErrInvalidOption identifies a nil context or an invalid per-batch lengths mask.
	// It is returned before starting recurrent computation or allocating its outputs.
	ErrInvalidOption = errors.New("lstm: invalid option")

	// ErrNonFinite identifies NaN or infinity supplied by the caller.
	// All input values are scanned, including masked padding and both state matrices.
	ErrNonFinite = errors.New("lstm: non-finite input")

	// ErrStateVersion identifies an initial State whose ModelID differs from Model.ID.
	// The mismatch is detected before inspecting initial-state matrix storage.
	ErrStateVersion = errors.New("lstm: state model mismatch")

	// ErrOverflow identifies an unrepresentable calculated affine, cell or hidden value.
	// This is never silently reclassified as a saturated activation or input-data error.
	ErrOverflow = errors.New("lstm: arithmetic overflow")
)
