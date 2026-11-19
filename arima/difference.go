// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima implements causal first/second differences and anchored integration.
// This file preserves the historical level and slope needed to recover future levels.
package arima

// difference applies exactly d sequential causal first differences to level samples.
// Each step computes y[t]=x[t+1]-x[t] in increasing index order and produces
// a newly owned slice; the input remains unchanged even for d=0.
// Time O((d+1)*N), Space O(N), with caller-validated d in [0,2] and N>=d+1.
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

// lastAnchors extracts origin coordinates from original, not differenced, history.
// Keeping both values for d=2 prevents a spurious level shift when future
// acceleration is accumulated. Time O(1), Space O(1), after history validation.
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

// integrate maps the next predicted d-th difference to its original level.
// d=1 accumulates against the last known level; d=2 first accumulates the
// historical slope and then the level; d=0 returns the difference-space mean.
// Time O(1), Space O(1). State is local and discarded on any error.
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
