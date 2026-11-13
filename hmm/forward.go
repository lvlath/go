// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm implements causal forward inference and immutable streaming state advancement.
// This boundary processes observations in temporal order without consulting future samples.
package hmm

import (
	"context"
	"math"

	"github.com/lvlath/go/matrix"
)

// step performs one causal HMM update using a prepared log-emission vector.
// Its returned checkpoint is detached; all existing posterior and likelihood state remain unchanged.
//
// Implementation:
//   - Stage 1: Validate an incoming checkpoint and allocate a new K-vector.
//   - Stage 2: Use log Prior on the first step or predecessor LogSumExp on subsequent steps.
//   - Stage 3: Reject impossible/overflowed evidence, normalize log posterior, and publish a new state.
//
// Behavior highlights:
//   - A zero transition has log mass -Inf and never creates an impossible-to-possible path.
//
// Inputs:
//   - c: immutable chain; prev: nil or read-only validated checkpoint; emission: K log scores at current t.
//
// Returns:
//   - *FilterState: Steps advanced, ModelID preserved, normalized log probabilities and cumulative evidence.
//
// Errors:
//   - ErrStateVersion, ErrOverflow, ErrImpossibleSequence.
//
// Determinism:
//   - Successor state is outer loop; predecessor state is inner loop, both ascending.
//
// Complexity:
//   - Time O(K^2), Space O(K) for the owned next posterior.
//
// Notes:
//   - The previous slice is never modified even if a later calculation fails.
//
// AI-Hints:
//   - Never substitute a smoothed gamma vector for prev: that introduces look-ahead.
func step(c *chain, prev *FilterState, emission []float64) (*FilterState, error) {
	if err := validState(c, prev); err != nil {
		return nil, err
	}
	next := make([]float64, c.k)
	// Initialization uses the prior exactly once. A later step predicts from
	// the previous normalized posterior, never from a smoothed distribution.
	if prev == nil {
		for j := range next {
			next[j] = logProbability(c.prior[j]) + emission[j]
		}
	} else {
		for j := range next {
			value := math.Inf(-1)
			for i := 0; i < c.k; i++ {
				value = LogSumExp(value, prev.LogPosterior[i]+logProbability(c.transition[i*c.k+j]))
			}
			next[j] = value + emission[j]
		}
	}
	// The normalizer is the evidence contributed by the current observation.
	// When every state has zero support, subtracting it would produce NaN.
	norm := LogSumExpSlice(next)
	if math.IsInf(norm, -1) {
		return nil, ErrImpossibleSequence
	}
	if !finite(norm) {
		return nil, ErrOverflow
	}
	likelihood := norm
	steps := uint64(1)
	if prev != nil {
		steps = prev.Steps + 1
		likelihood += prev.LogLikelihood
	}
	if !finite(likelihood) {
		return nil, ErrOverflow
	}
	// Subtract the common evidence increment to obtain a normalized log
	// posterior; returning a fresh slice prevents mutation of a saved snapshot.
	for j := range next {
		next[j] -= norm
	}
	return &FilterState{ModelID: c.id, Steps: steps, LogPosterior: next, LogLikelihood: likelihood}, nil
}

// FilterStep advances a categorical causal filter by exactly one observed symbol.
// Its posterior conditions only on the previous prefix and the supplied symbol.
// Implementation:
//   - Stage 1: Validate context, receiver, symbol domain, and external checkpoint.
//   - Stage 2: Evaluate per-state emission masses and normalize the log forward step.
//   - Stage 3: Return a new detached checkpoint, leaving prev unchanged on every path.
//
// Behavior highlights:
//   - prev=nil starts from Prior; impossible support returns ErrImpossibleSequence.
//
// Inputs:
//   - ctx: non-nil, uncanceled context; prev: optional immutable checkpoint; x: symbol in [0,A).
//
// Returns:
//   - *FilterState: Steps incremented by one, normalized log posterior, cumulative evidence.
//
// Errors:
//   - ErrInvalidOption/ctx.Err, ErrNilInput, ErrShape, ErrStateVersion, ErrOverflow, ErrImpossibleSequence.
//
// Determinism:
//   - Fixed successor-major and predecessor-ascending traversal.
//
// Complexity:
//   - Time O(K^2); space O(K), including the returned detached state.
//
// Notes:
//   - The receiver is immutable; the previous state is never modified.
//
// AI-Hints:
//   - Use snapshots from this exact model; swapping ModelID or replaying future data invalidates causality.
func (m *CategoricalModel) FilterStep(ctx context.Context, prev *FilterState, x uint32) (*FilterState, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if m == nil {
		return nil, ErrNilInput
	}
	if uint64(x) >= uint64(m.alphabet) {
		return nil, ErrShape
	}
	if err := validState(&m.chain, prev); err != nil {
		return nil, err
	}
	em := make([]float64, m.chain.k)
	if err := m.emissions(x, em); err != nil {
		return nil, err
	}
	return step(&m.chain, prev, em)
}

// FilterStep advances a diagonal-Gaussian causal filter using one feature vector.
// It consumes no observation later than x and never modifies prev or x.
// Implementation:
//   - Stage 1: Validate context, model, D finite features, and external checkpoint.
//   - Stage 2: Evaluate K diagonal-Gaussian log densities and the causal log recursion.
//   - Stage 3: Return a new detached normalized state with cumulative log evidence.
//
// Behavior highlights:
//   - Gaussian log density may be positive; this is not a probability-domain failure.
//
// Inputs:
//   - ctx: required; prev: optional validated checkpoint; x: D finite features, read-only.
//
// Returns:
//   - *FilterState: Steps advanced by one, finite accumulated evidence on success.
//
// Errors:
//   - ErrInvalidOption/ctx.Err, ErrNilInput, ErrShape, ErrNonFinite, ErrStateVersion, ErrOverflow.
//
// Determinism:
//   - States and features are evaluated in ascending index order.
//
// Complexity:
//   - Time O(K^2+K*D); space O(K) for emissions and returned posterior.
//
// Notes:
//   - The feature scaling policy is caller-owned and must stay fixed across updates.
//
// AI-Hints:
//   - Never impute missing features with arbitrary zeros; only finite complete vectors are accepted.
func (m *GaussianModel) FilterStep(ctx context.Context, prev *FilterState, x []float64) (*FilterState, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if m == nil || x == nil {
		return nil, ErrNilInput
	}
	if len(x) != m.dimension {
		return nil, ErrShape
	}
	for _, v := range x {
		if !finite(v) {
			return nil, ErrNonFinite
		}
	}
	if err := validState(&m.chain, prev); err != nil {
		return nil, err
	}
	em := make([]float64, m.chain.k)
	if err := m.emissions(x, em); err != nil {
		return nil, err
	}
	return step(&m.chain, prev, em)
}

// filterBatch consumes a complete sequence through the same causal step kernel as streaming.
// It optionally materializes chronological state posteriors without changing their evidence.
//
// Implementation:
//   - Stage 1: Allocate only a K-element emission buffer, plus an optional checked T-by-K output.
//   - Stage 2: Poll context, emit current log scores and advance immutable FilterState step by step.
//   - Stage 3: Publish detached final checkpoint, optional matrix, and identical cumulative likelihood scalars.
//
// Behavior highlights:
//   - History=false retains only O(K) live state even for a very long sequence.
//
// Inputs:
//   - ctx: context; c: immutable chain; t: sequence length; history: retention flag; emit: log-emission callback.
//
// Returns:
//   - *FilterResult: Last.Steps=T; Posterior is nil unless history is requested.
//
// Errors:
//   - ctx.Err(), ErrOverflow, ErrImpossibleSequence or emission/matrix materialization errors.
//
// Determinism:
//   - Time ascending, with the step kernel's successor/predecessor loop order.
//
// Complexity:
//   - Time O(T*K^2) plus emissions, Space O(K) live or O(T*K) with history.
//
// Notes:
//   - New states replace local prev references and never mutate caller-held snapshots.
//
// AI-Hints:
//   - The causal history row t must never be calculated using future-conditioned gamma.
func filterBatch(ctx context.Context, c *chain, t int, history bool, emit func(int, []float64) error) (*FilterResult, error) {
	var prev *FilterState
	var flat []float64
	if history {
		if c.k > 0 && t > int(^uint(0)>>1)/c.k {
			return nil, ErrOverflow
		}
		flat = make([]float64, t*c.k)
	}
	em := make([]float64, c.k)
	// Reuse the exact streaming kernel in chronological order so batch and
	// FilterStep replay share not only mathematics but floating evaluation order.
	for i := 0; i < t; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := emit(i, em); err != nil {
			return nil, err
		}
		next, err := step(c, prev, em)
		if err != nil {
			return nil, err
		}
		prev = next
		if history {
			for j, v := range next.LogPosterior {
				flat[i*c.k+j] = math.Exp(v)
			}
		}
	}
	if prev == nil {
		return nil, ErrEmptyInput
	}
	out := &FilterResult{Last: *prev, LogLikelihood: prev.LogLikelihood}
	if history {
		var err error
		out.Posterior, err = denseFromFlat(t, c.k, flat)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

// Filter processes a categorical sequence with strictly causal log-domain recursion.
// Its row t estimates P(z_t|x_0:t), never the retrospective smoothed posterior.
// Implementation:
//   - Stage 1: Validate context, model, and every categorical observation index.
//   - Stage 2: Update normalized log posteriors and cumulative evidence in time order.
//   - Stage 3: Publish detached Last and optional probability history.
//
// Behavior highlights:
//   - ReturnHistory=false avoids any T-by-K posterior allocation.
//
// Inputs:
//   - ctx: non-nil context; x: nonempty read-only symbols in [0,A); opts: history policy.
//
// Returns:
//   - *FilterResult: Last.Steps=T; Posterior is nil unless history is requested.
//
// Errors:
//   - ErrInvalidOption/ctx.Err, ErrNilInput, ErrEmptyInput, ErrShape, ErrImpossibleSequence, ErrOverflow.
//
// Determinism:
//   - Fixed chronological observation order and ascending state indices.
//
// Complexity:
//   - Time O(T*K^2); live workspace O(K), optional history O(T*K).
//
// Notes:
//   - Input symbols and model parameters are not mutated; returned data is detached.
//
// AI-Hints:
//   - If you need retrospective attribution rather than a real-time signal, call Smooth.
func (m *CategoricalModel) Filter(ctx context.Context, x []uint32, opts FilterOptions) (*FilterResult, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if m == nil {
		return nil, ErrNilInput
	}
	if err := validateSymbols(x, m.alphabet); err != nil {
		return nil, err
	}
	return filterBatch(ctx, &m.chain, len(x), opts.ReturnHistory, func(i int, e []float64) error { return m.emissions(x[i], e) })
}

// Filter processes finite Gaussian observations through a causal log-domain recursion.
// Each posterior row conditions only on the current observation and its past.
// Implementation:
//   - Stage 1: Validate context, model, matrix presence, nonempty shape, and D columns.
//   - Stage 2: Read one feature row at a time and update log-domain state posteriors.
//   - Stage 3: Return a detached final state and optional probability history.
//
// Behavior highlights:
//   - Without history, no full observation or posterior matrix is materialized.
//
// Inputs:
//   - ctx: required; x: T-by-D finite matrix, read-only; opts: history policy.
//
// Returns:
//   - *FilterResult: final causal state and accumulated marginal log density.
//
// Errors:
//   - ErrInvalidOption/ctx.Err, ErrNilInput, ErrEmptyInput, ErrShape, ErrNonFinite, ErrOverflow.
//
// Determinism:
//   - Chronological row order, then ascending state and feature indices.
//
// Complexity:
//   - Time O(T*(K^2+K*D)); workspace O(K+D), history O(T*K).
//
// Notes:
//   - The caller must not mutate x concurrently; matrix.At errors are propagated.
//
// AI-Hints:
//   - Prefer *matrix.Dense for fast At access; full T-by-D copying is intentionally avoided.
func (m *GaussianModel) Filter(ctx context.Context, x matrix.Matrix, opts FilterOptions) (*FilterResult, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if m == nil {
		return nil, ErrNilInput
	}
	if isNilMatrix(x) {
		return nil, ErrNilInput
	}
	if x.Rows() == 0 {
		return nil, ErrEmptyInput
	}
	if x.Cols() != m.dimension {
		return nil, ErrShape
	}
	// Read one observation at a time to keep no-history memory independent of T.
	row := make([]float64, m.dimension)
	return filterBatch(ctx, &m.chain, x.Rows(), opts.ReturnHistory, func(i int, e []float64) error {
		for d := range row {
			v, err := x.At(i, d)
			if err != nil {
				return err
			}
			if !finite(v) {
				return ErrNonFinite
			}
			row[d] = v
		}
		return m.emissions(row, e)
	})
}
