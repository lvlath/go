// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd validates input closes independently of signed intermediate state.
// This file separates public price-domain errors from arithmetic finiteness checks.
package macd

import "math"

// finite checks the arithmetic domain shared by prices, MACD and stored EMA means.
// It deliberately imposes no positivity requirement on signed indicator values.
// Time O(1), Space O(1). Internal helper; callers must validate price domain separately.
func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// validateClose checks the strictly positive finite domain of a market close.
// It preserves the distinct ErrNonFinite and ErrDomain classifications.
//
// Implementation:
//   - Reject NaN and both infinities before any domain comparison.
//   - Reject zero, negative zero, and all other nonpositive finite values.
//   - Accept every finite strictly positive binary64 price, including subnormals.
//
// Time O(1), Space O(1). This helper must not be used for signed Line/Signal/Histogram.
func validateClose(close float64) error {
	if !finite(close) {
		return ErrNonFinite
	}
	if close <= 0 {
		return ErrDomain
	}
	return nil
}
