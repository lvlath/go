// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 katalvlaran

package series

import "math"

// lagTransform validates even data positions that never become valid outputs.
func lagTransform(values []float64, lag int) (*TransformResult, error) {
	if lag < 0 {
		return nil, ErrInvalidOption
	}
	if err := validateValues(values); err != nil {
		return nil, err
	}
	result := newTransform(len(values))
	for i := lag; i < len(values); i++ {
		result.Values[i] = values[i-lag]
		result.Valid[i] = true
	}
	return result, nil
}

// differenceTransform checks each arithmetic result before publishing it.
func differenceTransform(values []float64, lag int) (*TransformResult, error) {
	if lag < 1 {
		return nil, ErrInvalidOption
	}
	if err := validateValues(values); err != nil {
		return nil, err
	}
	result := newTransform(len(values))
	for i := lag; i < len(values); i++ {
		v := values[i] - values[i-lag]
		if !finiteResult(v) {
			return nil, ErrOverflow
		}
		result.Values[i], result.Valid[i] = v, true
	}
	return result, nil
}

// simpleReturnsTransform validates all used denominators before computing.
// In the near-equal finite case, computing a relative difference preserves
// small returns that direct division followed by subtracting one loses.
func simpleReturnsTransform(values []float64) (*TransformResult, error) {
	if err := validateValues(values); err != nil {
		return nil, err
	}
	for i := 1; i < len(values); i++ {
		if values[i-1] == 0 {
			return nil, ErrDomain
		}
	}
	result := newTransform(len(values))
	for i := 1; i < len(values); i++ {
		prev, curr := values[i-1], values[i]
		// The subtraction can overflow although the quotient is finite.
		// Therefore, take the stable path only when the subtraction is finite
		// and its magnitude is smaller than half of the denominator.
		delta := curr - prev
		var v float64
		if finiteResult(delta) && math.Abs(delta) < math.Abs(prev)*0.5 {
			v = delta / prev
		} else {
			v = curr/prev - 1
		}
		if !finiteResult(v) {
			return nil, ErrOverflow
		}
		result.Values[i], result.Valid[i] = v, true
	}
	return result, nil
}

// logReturnsTransform avoids forming ratios that could overflow or underflow.
func logReturnsTransform(values []float64) (*TransformResult, error) {
	if err := validateValues(values); err != nil {
		return nil, err
	}
	for _, v := range values {
		if v <= 0 {
			return nil, ErrDomain
		}
	}
	result := newTransform(len(values))
	for i := 1; i < len(values); i++ {
		v := math.Log(values[i]) - math.Log(values[i-1])
		if !finiteResult(v) {
			return nil, ErrOverflow
		}
		result.Values[i], result.Valid[i] = v, true
	}
	return result, nil
}
