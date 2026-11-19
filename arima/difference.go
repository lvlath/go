// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima implements causal first/second differences and anchored integration.
// This file preserves the historical level and slope needed to recover future levels.
package arima

// difference applies the d-th causal difference without changing the input levels.
// It evaluates each difference as adjacent subtraction, retaining the correct
// observation alignment for a later forecast from the original physical origin.
//
// Implementation:
//   - Stage 1: detach the source sample into an owned working array, including d=0.
//   - Stage 2: for each requested order, subtract adjacent values in time order.
//   - Stage 3: reject nonfinite computed differences or publish the shorter array.
//
// Behavior highlights:
//   - For d=2, two first-difference passes avoid forming 2*x[t] prematurely.
//   - Each pass decreases the number of available samples by exactly one.
//
// Inputs:
//   - levels: validated finite, nonempty original values, owned by the caller.
//   - order: validated differencing degree 0, 1 or 2.
//
// Returns:
//   - []float64: owned d-th differences, length len(levels)-order.
//
// Errors:
//   - ErrOverflow: subtraction of finite values gives a nonfinite result.
//
// Determinism:
//   - Difference orders and sample indices are visited in ascending order.
//
// Complexity:
//   - Time O((d+1)*N), Space O(N), including the initial detached copy.
//
// Notes:
//   - The inverse level origin is not encoded in the resulting slice.
//
// AI-Hints:
//   - Preserve original-level anchors separately; do not recover them from
//     the d-th differences alone or replace the stable passes with 2*x[t].
func difference(levels []float64, order int) ([]float64, error) {
	current := append([]float64(nil), levels...)
	for step := 0; step < order; step++ {
		next := make([]float64, len(current)-1)
		for index := 1; index < len(current); index++ {
			value := current[index] - current[index-1]
			if !finite(value) {
				return nil, ErrOverflow
			}
			next[index-1] = value
		}
		current = next
	}
	return current, nil
}

// anchors is the local mutable integration state of one Forecast call.
// level=x[n-1] and first=x[n-1]-x[n-2] when d=2; neither value aliases input.
type anchors struct {
	level float64 // Last known or predicted original level.
	first float64 // Last first difference, updated before each d=2 level integration.
}

// lastAnchors captures the original-coordinate state required for inverse differencing.
// The first observation after the forecast origin is reconstructed from the
// latest actual level and, when d=2, the latest actual first difference.
//
// Implementation:
//   - Stage 1: read the final observed level from the validated input history.
//   - Stage 2: for d=2, calculate the final adjacent first difference once.
//   - Stage 3: reject overflow or return local scalar integration state.
//
// Behavior highlights:
//   - The final first difference is x[n-1]-x[n-2], not x[n-2]-x[n-3].
//
// Inputs:
//   - history: validated finite original levels, length >=d+1.
//   - order: validated degree in [0,2].
//
// Returns:
//   - anchors: value-only state; it has no alias to caller memory.
//
// Errors:
//   - ErrOverflow: required slope cannot be represented as finite float64.
//
// Determinism:
//   - Reads only the final two original sample positions, in fixed order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A distinct Forecast call reconstructs a distinct local origin.
//
// AI-Hints:
//   - Never initialize the slope to zero for d=2 or derive it from the
//     final second difference; both would translate the entire level forecast.
func lastAnchors(history []float64, order int) (anchors, error) {
	current := anchors{level: history[len(history)-1]}
	if order == 2 {
		current.first = history[len(history)-1] - history[len(history)-2]
		if !finite(current.first) {
			return anchors{}, ErrOverflow
		}
	}
	return current, nil
}

// integrate advances one forecast level from the next predicted d-th difference.
// Its mutable receiver belongs exclusively to one Forecast call and is
// updated only through the exact causal inverse-differencing recurrence.
//
// Implementation:
//   - Stage 1: choose direct publication (d=0) or an anchored integration mode.
//   - Stage 2: add the prediction to the level (d=1), or to the stored slope
//     and then the level (d=2), preserving the correct update order.
//   - Stage 3: reject overflow or publish the next physical level.
//
// Behavior highlights:
//   - For d=2, the slope update precedes the level update at every horizon.
//
// Inputs:
//   - state: private, per-call scalar anchors; not shared across goroutines.
//   - prediction: finite next conditional mean in d-th-difference units.
//   - order: validated differencing degree 0, 1 or 2.
//
// Returns:
//   - float64: next original-coordinate level, finite on success.
//
// Errors:
//   - ErrOverflow: intermediate slope or reconstructed level is nonfinite.
//
// Determinism:
//   - Executes one fixed ordered sequence per differencing mode.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - On failure, the caller discards the private anchors and partial forecast.
//
// AI-Hints:
//   - Do not compute future levels solely by cumulatively summing prediction
//     values from zero; retaining both historical anchors is essential for d=2.
func (state *anchors) integrate(prediction float64, order int) (float64, error) {
	switch order {
	case 1:
		state.level += prediction
	case 2:
		state.first += prediction
		if !finite(state.first) {
			return 0, ErrOverflow
		}
		state.level += state.first
	default:
		state.level = prediction
	}
	if !finite(state.level) {
		return 0, ErrOverflow
	}
	return state.level, nil
}
