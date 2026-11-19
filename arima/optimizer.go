// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima implements bounded, deterministic local likelihood optimization.
// This file owns L-BFGS history, search-direction construction, and Armijo acceptance.
package arima

import (
	"context"
	"errors"
	"math"
)

const (
	lbfgsHistorySize = 8     // At most eight secant corrections are retained.
	lbfgsLineTrials  = 20    // The first step is one; each failed trial halves it.
	lbfgsArmijoSlope = 1e-4  // Standard sufficient-decrease coefficient.
	lbfgsCurvature   = 1e-14 // Relative positivity threshold for secant updates.
)

// lbfgsCorrection stores one accepted displacement and gradient difference.
// Its rho reciprocal is positive and finite; every slice is owned by the state.
type lbfgsCorrection struct {
	step, change []float64
	rho          float64
}

// lbfgsState owns the last valid candidate and deterministic optimizer history.
// Only accept mutates it, so a failed search cannot corrupt a publishable result.
type lbfgsState struct {
	coordinates []float64
	gradient    []float64
	parameters  Parameters
	objective   float64
	objectives  []float64
	corrections []lbfgsCorrection
	iterations  int
}

// lbfgsTrial contains a complete, not-yet-published line-search candidate.
// Its buffers are independent of the currently accepted optimizer state.
type lbfgsTrial struct {
	coordinates []float64
	gradient    []float64
	parameters  Parameters
	objective   float64
}

// converged checks a dimensionless gradient-to-objective scale.
// The division avoids overflowing the product tolerance*(1+abs(objective)).
// Time O(p+q+drift), Space O(1).
func (state *lbfgsState) converged(tolerance float64) bool {
	gradientNorm := norm(state.gradient)
	return finite(gradientNorm) && gradientNorm/(1+math.Abs(state.objective)) <= tolerance
}

// direction applies the L-BFGS two-loop recursion in fixed correction order.
// It reconstructs an approximate inverse-Hessian-gradient product using at most
// eight positive-curvature displacement pairs and rejects non-descent proposals.
//
// Implementation:
//   - Stage 1: apply newest-to-oldest inverse secant updates to a copied gradient.
//   - Stage 2: scale by the most recent secant curvature and apply forward updates.
//   - Stage 3: negate the direction or fall back to a steepest descent direction.
//
// Behavior highlights:
//   - A fallback changes only the direction, never the accepted model coefficients.
//
// Inputs:
//   - state: immutable accepted gradient and bounded secant history during this call.
//
// Returns:
//   - search: owned direction; slope: finite directional derivative less than zero.
//
// Errors:
//   - ErrOverflow if neither bounded L-BFGS nor steepest descent yields a finite descent slope.
//
// Determinism:
//   - Reverse and then forward corrections follow acceptance chronology exactly.
//
// Complexity:
//   - Time O(M*k), Space O(k+M), M<=8 and k=p+q+drift.
//
// Notes:
//   - Invalid curvature pairs are filtered at accept, not silently repaired here.
//
// AI-Hints:
//   - Do not interpret inverse-Hessian scaling as a model-parameter regularizer.
func (state *lbfgsState) direction() ([]float64, float64, error) {
	search := append([]float64(nil), state.gradient...)
	alpha := make([]float64, len(state.corrections))

	for i := len(state.corrections) - 1; i >= 0; i-- {
		correction := state.corrections[i]
		alpha[i] = correction.rho * dot(correction.step, search)
		for k := range search {
			search[k] -= alpha[i] * correction.change[k]
		}
	}

	if len(state.corrections) != 0 {
		last := state.corrections[len(state.corrections)-1]
		gamma := dot(last.step, last.change) / dot(last.change, last.change)
		for k := range search {
			search[k] *= gamma
		}
	}

	for i, correction := range state.corrections {
		beta := correction.rho * dot(correction.change, search)
		for k := range search {
			search[k] += correction.step[k] * (alpha[i] - beta)
		}
	}
	for k := range search {
		search[k] = -search[k]
	}

	slope := dot(state.gradient, search)
	if !finite(slope) || slope >= 0 {
		for k := range search {
			search[k] = -state.gradient[k]
		}
		slope = -dot(state.gradient, state.gradient)
	}
	if !finite(slope) || slope >= 0 {
		return nil, 0, ErrOverflow
	}
	return search, slope, nil
}

// lineSearch tries at most twenty deterministic Armijo steps without state mutation.
// It never accepts a candidate merely because a floating-point addition rounds
// away a proposed step; accepted candidates require strict objective decrease.
//
// Implementation:
//   - Stage 1: construct an L-BFGS/steepest-descent direction and negative slope.
//   - Stage 2: try 1, 1/2, 1/4, ... scales; reject out-of-domain numeric candidates.
//   - Stage 3: return the first sufficient-decrease trial, or explicit nonconvergence.
//
// Behavior highlights:
//   - Unit-root rounding, singular variance and numeric overflow backtrack without clipping.
//
// Inputs:
//   - ctx: active context; series: read-only differenced training sample.
//   - opts: fixed solver policy; state: latest immutable accepted candidate.
//
// Returns:
//   - lbfgsTrial: fully owned unpublished candidate; zero value on failure.
//
// Errors:
//   - ErrNotConverged: line trials exhausted with no acceptable positive progress.
//   - ctx.Err(): canceled search; unexpected evaluation errors propagate unchanged.
//
// Determinism:
//   - Trial steps halve in fixed order, with a strict first-acceptable tie policy.
//
// Complexity:
//   - Time O(M*k+20*N*(p+q+1)^2), Space O(N*(p+q+1)+k+M).
//
// Notes:
//   - Only accept may mutate an optimizer state; failed trials are never observable.
//
// AI-Hints:
//   - Do not call accept on a rejected trial or replace failures with unit-root clipping.
func (state *lbfgsState) lineSearch(ctx context.Context, series []float64, opts Options) (lbfgsTrial, error) {
	direction, slope, err := state.direction()
	if err != nil {
		return lbfgsTrial{}, err
	}

	for attempt, step := 0, 1.0; attempt < lbfgsLineTrials; attempt, step = attempt+1, step*0.5 {
		if err := ctx.Err(); err != nil {
			return lbfgsTrial{}, err
		}
		candidate := make([]float64, len(state.coordinates))
		changed := false
		for k, coordinate := range state.coordinates {
			candidate[k] = coordinate + step*direction[k]
			changed = changed || candidate[k] != coordinate
		}
		if !changed { // A rounded-away step cannot constitute optimizer progress.
			continue
		}
		value, parameters, gradient, evalErr := evaluate(ctx, series, candidate, opts)
		if evalErr != nil {
			if errors.Is(evalErr, ErrNonStationary) || errors.Is(evalErr, ErrNonInvertible) ||
				errors.Is(evalErr, ErrSingular) || errors.Is(evalErr, ErrOverflow) {
				continue
			}
			return lbfgsTrial{}, evalErr
		}
		if value < state.objective && value <= state.objective+lbfgsArmijoSlope*step*slope {
			return lbfgsTrial{candidate, gradient, parameters, value}, nil
		}
	}
	return lbfgsTrial{}, ErrNotConverged
}

// accept publishes an Armijo trial into the optimizer's private mutable state.
// It updates the secant history only for positive finite curvature and always
// preserves the chronological objective trajectory of accepted model coefficients.
//
// Implementation:
//   - Stage 1: compute coordinate and gradient displacements from the accepted state.
//   - Stage 2: retain up to eight finite positive-curvature secant corrections.
//   - Stage 3: replace the accepted candidate and append exactly one objective.
//
// Behavior highlights:
//   - Curvature rejection does not reject a statistically valid Armijo step.
//
// Inputs:
//   - state: owned optimizer workspace; next: complete accepted trial, detached.
//
// Returns:
//   - No value; mutates only private optimizer memory and accepted-step counters.
//
// Errors:
//   - None: numeric trial validity was established before this call.
//
// Determinism:
//   - Oldest secant correction is discarded first at the eight-element boundary.
//
// Complexity:
//   - Time O(k), Space O(k) for one new pair; retained history O(8*k).
//
// Notes:
//   - Invalid or nearly singular curvature never changes the accepted likelihood.
//
// AI-Hints:
//   - Never count rejected line-search evaluations as optimizer iterations.
func (state *lbfgsState) accept(next lbfgsTrial) {
	step := make([]float64, len(state.coordinates))
	change := make([]float64, len(state.coordinates))
	for k := range step {
		step[k] = next.coordinates[k] - state.coordinates[k]
		change[k] = next.gradient[k] - state.gradient[k]
	}

	curvature := dot(step, change)
	threshold := lbfgsCurvature * norm(step) * norm(change)
	if finite(curvature) && curvature > threshold {
		rho := 1 / curvature
		if finite(rho) {
			if len(state.corrections) == lbfgsHistorySize {
				// Discard precisely the oldest pair; the memory size is bounded.
				copy(state.corrections, state.corrections[1:])
				state.corrections = state.corrections[:lbfgsHistorySize-1]
			}
			state.corrections = append(state.corrections, lbfgsCorrection{step, change, rho})
		}
	}

	state.coordinates = next.coordinates
	state.gradient = next.gradient
	state.parameters = next.parameters
	state.objective = next.objective
	state.objectives = append(state.objectives, next.objective)
	state.iterations++ // Accepted steps, not evaluations or rejected trials.
}

// publish freezes the latest accepted model and detaches objective history.
// It returns precisely the last valid candidate, with a convergence flag that
// preserves the caller-visible distinction between successful and partial fits.
//
// Implementation:
//   - Stage 1: validate and copy physical coefficients using New.
//   - Stage 2: detach the accepted objective history and attach EffectiveN.
//   - Stage 3: publish a finite log likelihood and explicit convergence state.
//
// Behavior highlights:
//   - The initial objective counts as element zero, not an accepted iteration.
//
// Inputs:
//   - sampleSize: differenced sample length; order: fixed AR and MA burn-in.
//   - converged: optimizer's actual scaled-gradient termination outcome.
//
// Returns:
//   - *FitResult: immutable model with detached diagnostics; nil if publication fails.
//
// Errors:
//   - The New parameter-domain sentinel if the final candidate is invalid.
//
// Determinism:
//   - Objective[i] is the objective after exactly i accepted updates.
//
// Complexity:
//   - Time O(I+p+q), Space O(I+p+q).
//
// Notes:
//   - ErrNotConverged is attached by fitKernel, not by this constructor.
//
// AI-Hints:
//   - Do not infer convergence merely from the presence of a Model pointer.
func (state *lbfgsState) publish(sampleSize int, order Order, converged bool) (*FitResult, error) {
	model, err := New(state.parameters)
	if err != nil {
		return nil, err
	}
	return &FitResult{
		Model: model,
		Diagnostics: Diagnostics{
			Iterations: state.iterations,
			Converged:  converged,
			Objective:  append([]float64(nil), state.objectives...),
			EffectiveN: sampleSize - maxN(order.AR, order.MA),
		},
		LogLikelihood: -state.objective,
	}, nil
}
