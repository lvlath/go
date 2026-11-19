// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch implements the O(T) causal filter and O(1) posterior transition.
// Public validation and documentation are centralized in api.go.
package gjrgarch

import "context"

// initialStateKernel records the first return with the caller's h[0].
// Validation is complete before entering this pure constructor.
func (m *Model) initialStateKernel(observedReturn, variance float64) (State, error) {
	residual := observedReturn - m.p.Mean
	if !finite(residual) {
		return State{}, ErrOverflow
	}
	return State{ModelID: m.id, LastResidual: residual, LastVariance: variance, Steps: 1}, nil
}

// stepKernel advances the variance using the previously stored residual.
// The caller's state value is copied, so failures cannot alter its contents.
func (m *Model) stepKernel(state State, observedReturn float64) (State, error) {
	// Only the PRIOR innovation contributes to the next variance.
	next, err := nextVariance(m.p, state.LastResidual, state.LastVariance)
	if err != nil {
		return State{}, err
	}
	// Compute and validate the new innovation only after h[t+1] is fixed.
	residual := observedReturn - m.p.Mean
	if !finite(residual) {
		return State{}, ErrOverflow
	}
	state.LastVariance = next
	state.LastResidual = residual
	state.Steps++
	return state, nil
}

// filterKernel publishes h[0] through h[T-1] in the original observation order.
// It uses the same Step kernel as external stateful callers, with atomic output.
func (m *Model) filterKernel(ctx context.Context, returns []float64, initialVariance float64) ([]float64, error) {
	// Establish a valid posterior even for a singleton series. Without this
	// check, a representable return minus a finite Mean can overflow silently
	// when there is no second observation to trigger the Step kernel.
	state, err := m.initialStateKernel(returns[0], initialVariance)
	if err != nil {
		return nil, err
	}
	// Publish a detached slice only after the first posterior is valid.
	variances := make([]float64, len(returns))
	variances[0] = initialVariance
	if len(returns) == 1 {
		return variances, nil
	}
	for i := 1; i < len(returns); i++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		state, err = m.stepKernel(state, returns[i])
		if err != nil {
			return nil, err
		}
		variances[i] = state.LastVariance
	}
	return variances, nil
}
