// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch implements the O(T) causal filter and O(1) posterior transition.
// Public validation and documentation are centralized in api.go.
package gjrgarch

import "context"

// initialStateKernel records the first return with the caller's h[0].
// Validation is complete before entering this pure constructor.
//
// Implementation:
//   - Stage 1: Subtract the model mean from the first observed return.
//   - Stage 2: Copy it with supplied h[0] and ID.
//
// Behavior highlights:
//   - Validates subtraction even for singleton series, preventing silent last-residual overflow.
//
// Inputs:
//   - observedReturn: validated first finite value; variance: validated positive finite h[0].
//
// Returns:
//   - State: first posterior with Steps=1 or zero State on error.
//
// Errors:
//   - ErrOverflow if observedReturn-Mean is not finite.
//
// Determinism:
//   - Always records the same first observation before any later transition.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Caller facade must validate model, variance and finite input first.
//
// AI-Hints:
//   - Do not apply the first epsilon shock to h[0]; that would violate causality.
func (m *Model) initialStateKernel(observedReturn, variance float64) (State, error) {
	residual := observedReturn - m.p.Mean
	if !finite(residual) {
		return State{}, ErrOverflow
	}
	return State{ModelID: m.id, LastResidual: residual, LastVariance: variance, Steps: 1}, nil
}

// stepKernel advances the variance using the previously stored residual.
// The caller's state value is copied, so failures cannot alter its contents.
//
// Implementation:
//   - Stage 1: Compute next variance from prior State.
//   - Stage 2: Then calculate new residual.
//   - Stage 3: Publish incremented State.
//
// Behavior highlights:
//   - A subtraction failure discards the new state without mutating prior caller value.
//
// Inputs:
//   - state: validated prior posterior; observedReturn: validated next finite return.
//
// Returns:
//   - State: new posterior or zero on arithmetic failure.
//
// Errors:
//   - ErrOverflow on recursion or new residual subtraction.
//
// Determinism:
//   - Uses prior epsilon in the first stage and next epsilon only in the final stage.
//
// Complexity:
//   - Time O(1), Space O(1), no successful heap allocation.
//
// Notes:
//   - Public Step checks state identity and uint64 overflow before calling this kernel.
//
// AI-Hints:
//   - Do not reorder the residual calculation ahead of the variance transition.
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
//
// Implementation:
//   - Stage 1: Create first state.
//   - Stage 2: Allocate detached h sequence.
//   - Stage 3: Advance using stepKernel for each following return.
//
// Behavior highlights:
//   - A singleton still validates its first residual; no partial slice is returned on errors.
//
// Inputs:
//   - ctx: validated context; returns: nonempty validated read-only data; initialVariance: h[0].
//
// Returns:
//   - Detached []float64 aligned to returns, or nil on failure.
//
// Errors:
//   - ErrOverflow for arithmetic; ctx.Err() for mid-loop cancellation.
//
// Determinism:
//   - Observation index strictly increases; next output derives only from prior residual.
//
// Complexity:
//   - Time O(T), Space O(T) including output allocation.
//
// Notes:
//   - No caller or model slice is retained; caller must not mutate returns concurrently.
//
// AI-Hints:
//   - Do not optimize by folding return[t] directly into h[t].
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
