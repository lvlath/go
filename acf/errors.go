// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package acf

// errors.go declares stable input, numeric-domain, and arithmetic sentinels.
// Use errors.Is for classification; error wording is diagnostic, not protocol.

import "errors"

// ErrNilInput reports a nil sample slice after valid options are established.
// It is distinct from a non-nil, allocated but empty slice.
var ErrNilInput = errors.New("acf: nil input")

// ErrEmptyInput reports a non-nil zero-length observation window.
// At least one finite observation is required even when maxLag is zero.
var ErrEmptyInput = errors.New("acf: empty input")

// ErrInvalidOption reports an unknown denominator or an invalid maximum lag.
// Negative lag and unknown enum are checked before inspecting the sample slice;
// maxLag >= n is rejected after the nil/empty presence gates.
var ErrInvalidOption = errors.New("acf: invalid option")

// ErrNonFinite reports a NaN or positive/negative infinity in the input data.
// Every observation is checked, including samples not used by nonzero-lag pairs.
var ErrNonFinite = errors.New("acf: non-finite observation")

// ErrZeroVariance reports an undefined normalized second-moment ratio.
// Centered constants and entirely zero uncentered samples have S0 == 0;
// a nonzero uncentered constant is well-defined and must not trigger this error.
var ErrZeroVariance = errors.New("acf: zero variance or second moment")

// ErrOverflow reports a required public result that cannot be represented by float64.
// Intermediate unscaled sums are not required by the algorithm and must not
// cause this error when a scaled computation of the ratio remains possible.
var ErrOverflow = errors.New("acf: arithmetic overflow")
