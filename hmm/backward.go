// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm implements the log-domain forward-backward inference kernel.
// This boundary separates causal evidence, retrospective smoothing, and EM pair counts.
package hmm

import (
	"context"
	"math"

	"github.com/lvlath/go/matrix"
)

// posteriorWorkspace owns the per-sequence messages required by forward-backward inference.
// EM reuses the same gamma/beta/alpha evidence without coupling independent sequences.
//
// Implementation:
//   - Stage 1: Retain per-state emission log likelihoods.
//   - Stage 2: Retain normalized forward and scaled backward messages plus gamma.
//   - Stage 3: Publish only requested result matrices or EM sufficient statistics.
//
// Behavior highlights:
//   - emissions, alpha, beta, gamma each have T*K elements; logLikelihood is scalar.
//
// Inputs:
//   - Fields are private call-owned buffers with state-major fastest dimension.
//
// Returns:
//   - Internal workspace pointer, not an exported user-owned artifact.
//
// Errors:
//   - Kernels report ErrOverflow, ErrImpossibleSequence, ErrNumerical before construction.
//
// Determinism:
//   - Messages use chronological forward and reverse chronological backward order.
//
// Complexity:
//   - Storage O(T*K); construction time O(T*K^2) plus emission evaluation.
//
// Notes:
//   - No model retains the workspace; distinct calls do not share mutable tables.
//
// AI-Hints:
//   - Do not allocate xi eagerly for Smooth: transition pairs are needed only by EM.
type posteriorWorkspace struct {
	emissions     []float64 // T*K log b_j(x_t), written once for the sequence.
	alpha         []float64 // T*K normalized causal log P(z_t=j | x_0:t).
	beta          []float64 // T*K scaled future-conditioned backward log messages.
	gamma         []float64 // T*K retrospective P(z_t=j | x_0:T-1).
	logLikelihood float64   // Sum of normalized forward log evidence increments.
}

// checkedTableSize validates row-major matrix lengths before allocating tables.
// It catches invalid dimensions and integer multiplication overflow, not OS memory exhaustion.
//
// Implementation:
//   - Stage 1: Reject negative rows and columns below one.
//   - Stage 2: Check rows <= MaxInt/columns before multiplying.
//   - Stage 3: Return the representable flattened cell count.
//
// Behavior highlights:
//   - A zero row count is valid and yields zero cells.
//
// Inputs:
//   - rows: row count; columns: strictly positive cell count per row.
//
// Returns:
//   - int: rows*columns if representable.
//
// Errors:
//   - ErrOverflow for invalid dimensions or unrepresentable multiplication.
//
// Determinism:
//   - Pure integer comparisons; no allocation or random state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Caller still decides the memory budget before allocating the returned size.
//
// AI-Hints:
//   - Use this at every T*K and (T-1)*K*K allocation boundary.
func checkedTableSize(rows, columns int) (int, error) {
	if rows < 0 || columns < 1 || rows > int(^uint(0)>>1)/columns {
		return 0, ErrOverflow
	}
	return rows * columns, nil
}

// collectEmissions evaluates per-state log emission scores for every observation.
// The resulting T-by-K table is shared only among the forward/backward kernels of this call.
//
// Implementation:
//   - Stage 1: Compute the checked T*K buffer size.
//   - Stage 2: Poll context and invoke the emission callback once per chronological row.
//   - Stage 3: Return only a completely filled private row-major table.
//
// Behavior highlights:
//   - Callback errors are propagated without rewriting their sentinel classification.
//
// Inputs:
//   - ctx: active context; c: immutable chain; t: observation count; emit: writes K log scores to dst.
//
// Returns:
//   - []float64: T*K log emission values in temporal and state order.
//
// Errors:
//   - ErrOverflow for size failure; ctx.Err() or the emission callback error.
//
// Determinism:
//   - Time ascends by one; state traversal is the callback's fixed policy.
//
// Complexity:
//   - Time O(T*K) buffer preparation plus callback cost, Space O(T*K).
//
// Notes:
//   - The callback must never retain or mutate dst after returning.
//
// AI-Hints:
//   - A categorical zero emission should be -Inf, not an arbitrarily small log epsilon.
func collectEmissions(ctx context.Context, c *chain, t int, emit func(int, []float64) error) ([]float64, error) {
	count, err := checkedTableSize(t, c.k)
	if err != nil {
		return nil, err
	}
	values := make([]float64, count)
	// Construct the entire emission table once; both inference directions must
	// use identical observed evidence rather than recalculating changing inputs.
	for row := 0; row < t; row++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		if err = emit(row, values[row*c.k:(row+1)*c.k]); err != nil {
			return nil, err
		}
	}
	return values, nil
}

// forwardMessages computes normalized causal log posteriors and total sequence evidence.
// Each alpha[t,j] conditions only on observations up to t despite later backward use.
//
// Implementation:
//   - Stage 1: Initialize from log Prior and the first emission row.
//   - Stage 2: For later rows use LogSumExp over predecessor alpha and log transitions.
//   - Stage 3: Normalize every row and accumulate the normalization into logLikelihood.
//
// Behavior highlights:
//   - All-impossible rows are classified before subtracting -Inf, preventing NaN.
//
// Inputs:
//   - ctx: active context; c: immutable transition chain; t: positive sequence length; emissions: T*K log scores.
//
// Returns:
//   - []float64 alpha, logLikelihood, error: normalized log posterior history and marginal evidence.
//
// Errors:
//   - ctx.Err(), ErrImpossibleSequence, or ErrOverflow on unrepresentable normalization/evidence.
//
// Determinism:
//   - Time, successor, predecessor indices increase in that precise nesting order.
//
// Complexity:
//   - Time O(T*K^2), Space O(T*K) for returned alpha history.
//
// Notes:
//   - The emissions slice is read-only, while alpha is an owning workspace.
//
// AI-Hints:
//   - Never reuse backward beta in this recurrence; doing so leaks future information.
func forwardMessages(ctx context.Context, c *chain, t int, emissions []float64) ([]float64, float64, error) {
	k := c.k
	alpha := make([]float64, len(emissions))
	var total float64
	// Each normalized row represents a complete causal prefix. Future beta
	// messages are unavailable in this pass by construction.
	for rowIndex := 0; rowIndex < t; rowIndex++ {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		row := alpha[rowIndex*k : (rowIndex+1)*k]
		for state := 0; state < k; state++ {
			if rowIndex == 0 {
				row[state] = logProbability(c.prior[state]) + emissions[state]
				continue
			}
			value := math.Inf(-1)
			for predecessor := 0; predecessor < k; predecessor++ {
				value = LogSumExp(value, alpha[(rowIndex-1)*k+predecessor]+logProbability(c.transition[predecessor*k+state]))
			}
			row[state] = value + emissions[rowIndex*k+state]
		}
		// This scalar equals log P(x_t | x_0:t-1), so adding it to the
		// running total restores full marginal evidence without raw products.
		normalization := LogSumExpSlice(row)
		if math.IsInf(normalization, -1) {
			return nil, 0, ErrImpossibleSequence
		}
		if !finite(normalization) {
			return nil, 0, ErrOverflow
		}
		total += normalization
		if !finite(total) {
			return nil, 0, ErrOverflow
		}
		for state := range row {
			row[state] -= normalization
		}
	}
	return alpha, total, nil
}

// backwardMessages computes future-conditioned log beta messages and smoothed gamma.
// Each beta row is shifted by a common constant to stabilize long backward recursions.
//
// Implementation:
//   - Stage 1: Initialize terminal beta entries to zero.
//   - Stage 2: Traverse backward, summing next emissions, transitions, and next beta with LogSumExp.
//   - Stage 3: Normalize alpha+beta and exponentiate once to produce gamma probabilities.
//
// Behavior highlights:
//   - A constant shift of all beta[t,*] cancels in both gamma and xi normalization.
//
// Inputs:
//   - ctx: active context; c: transition chain; t: sequence length; emissions, alpha: T*K read-only messages.
//
// Returns:
//   - beta, gamma, error: T*K backward log messages and smoothed normalized probabilities.
//
// Errors:
//   - ctx.Err() or ErrNumerical when a required posterior normalization is nonfinite.
//
// Determinism:
//   - Times decrease from T-1; predecessor and successor indices increase.
//
// Complexity:
//   - Time O(T*K^2), Space O(T*K) for beta and gamma.
//
// Notes:
//   - Gamma conditions on the whole sequence: it must never be used as online filter output.
//
// AI-Hints:
//   - Backward scaling affects absolute beta values; use only normalized posterior ratios.
func backwardMessages(ctx context.Context, c *chain, t int, emissions, alpha []float64) ([]float64, []float64, error) {
	k := c.k
	beta := make([]float64, len(alpha))
	gamma := make([]float64, len(alpha))
	// Reverse chronological order is intentional: beta[t] summarizes evidence
	// strictly after time t, while alpha[t] already contains the causal prefix.
	for rowIndex := t - 1; rowIndex >= 0; rowIndex-- {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		if rowIndex < t-1 {
			for predecessor := 0; predecessor < k; predecessor++ {
				value := math.Inf(-1)
				for successor := 0; successor < k; successor++ {
					value = LogSumExp(value, logProbability(c.transition[predecessor*k+successor])+emissions[(rowIndex+1)*k+successor]+beta[(rowIndex+1)*k+successor])
				}
				beta[rowIndex*k+predecessor] = value
			}
			// A state-independent shift prevents backward message growth over a
			// long record; normalized gamma and xi are invariant under this shift.
			scale := logPosteriorPairNormalization(alpha[rowIndex*k:(rowIndex+1)*k], beta[rowIndex*k:(rowIndex+1)*k])
			if !finite(scale) {
				return nil, nil, ErrNumerical
			}
			for state := 0; state < k; state++ {
				beta[rowIndex*k+state] -= scale
			}
		}
		scale := logPosteriorPairNormalization(alpha[rowIndex*k:(rowIndex+1)*k], beta[rowIndex*k:(rowIndex+1)*k])
		if !finite(scale) {
			return nil, nil, ErrNumerical
		}
		for state := 0; state < k; state++ {
			gamma[rowIndex*k+state] = math.Exp(alpha[rowIndex*k+state] + beta[rowIndex*k+state] - scale)
		}
	}
	return beta, gamma, nil
}

// logPosteriorPairNormalization computes log sum_i exp(alpha_i+beta_i) without an extra slice.
// It supplies the per-row normalization shared by smoothed gamma and beta scaling.
//
// Implementation:
//   - Stage 1: Initialize an impossible aggregate at -Inf.
//   - Stage 2: Add log-message pairs in ascending state order with LogSumExp.
//   - Stage 3: Return the common log normalization.
//
// Behavior highlights:
//   - All-impossible pairs return -Inf, leaving error classification to callers.
//
// Inputs:
//   - alpha, beta: equal-length read-only log-message rows; private kernel invariant.
//
// Returns:
//   - float64: log normalizer for the row, possibly -Inf.
//
// Errors:
//   - No direct error; callers return ErrNumerical for nonfinite normalization.
//
// Determinism:
//   - Ascending state index determines the binary64 summation order.
//
// Complexity:
//   - Time O(K), Space O(1).
//
// Notes:
//   - No new allocations or mutation of either input row.
//
// AI-Hints:
//   - Use this instead of summing raw exponentials of potentially tiny unnormalized scores.
func logPosteriorPairNormalization(alpha, beta []float64) float64 {
	total := math.Inf(-1)
	for state := range alpha {
		total = LogSumExp(total, alpha[state]+beta[state])
	}
	return total
}

// transitionExpectations computes smoothed transition-pair responsibilities xi[t,i,j].
// Only times 0..T-2 contribute; no transition links independent training sequences.
//
// Implementation:
//   - Stage 1: Return nil for singleton sequences, which have zero internal transition slots.
//   - Stage 2: Allocate checked (T-1)*K*K log scores and compute each pair mass.
//   - Stage 3: Normalize each within-time pair table and exponentiate to probabilities.
//
// Behavior highlights:
//   - Returned xi is EM-only; ordinary Smooth has no reason to allocate this table.
//
// Inputs:
//   - ctx: active context; c: chain; t: sequence length; work: corresponding normalized messages.
//
// Returns:
//   - []float64: contiguous (T-1)*K*K responsibilities, nil for T<=1.
//
// Errors:
//   - ctx.Err(), ErrOverflow for allocation dimensions, ErrNumerical for invalid pair mass.
//
// Determinism:
//   - Time then source state then successor state; all accumulation is deterministic.
//
// Complexity:
//   - Time O((T-1)*K^2), Space O((T-1)*K^2).
//
// Notes:
//   - xi rows sum to one, but are not independent of per-state gamma expectations.
//
// AI-Hints:
//   - Never concatenate multiple sequences before applying this within-sequence kernel.
func transitionExpectations(ctx context.Context, c *chain, t int, work *posteriorWorkspace) ([]float64, error) {
	if t <= 1 {
		return nil, nil
	}
	square, err := checkedTableSize(c.k, c.k)
	if err != nil {
		return nil, err
	}
	count, err := checkedTableSize(t-1, square)
	if err != nil {
		return nil, err
	}
	xi := make([]float64, count)
	// The upper bound excludes the final observation: it has no successor
	// inside this sequence, and therefore cannot generate an outgoing xi pair.
	for rowIndex := 0; rowIndex < t-1; rowIndex++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		normalization := math.Inf(-1)
		for predecessor := 0; predecessor < c.k; predecessor++ {
			for successor := 0; successor < c.k; successor++ {
				value := work.alpha[rowIndex*c.k+predecessor] + logProbability(c.transition[predecessor*c.k+successor]) + work.emissions[(rowIndex+1)*c.k+successor] + work.beta[(rowIndex+1)*c.k+successor]
				xi[rowIndex*square+predecessor*c.k+successor] = value
				normalization = LogSumExp(normalization, value)
			}
		}
		// Zero support or an unrepresentable pair normalizer invalidates the
		// E-step; publishing NaN responsibilities would poison subsequent M-steps.
		if !finite(normalization) {
			return nil, ErrNumerical
		}
		for cell := 0; cell < square; cell++ {
			xi[rowIndex*square+cell] = math.Exp(xi[rowIndex*square+cell] - normalization)
		}
	}
	return xi, nil
}

// buildPosteriorWorkspace assembles the shared forward-backward message state for one sequence.
// It performs no EM-specific xi allocation, allowing smoothing memory to stay O(T*K).
//
// Implementation:
//   - Stage 1: Collect all validated per-state emission log scores.
//   - Stage 2: Compute normalized forward alpha and reverse beta/gamma.
//   - Stage 3: Return a per-call workspace with the full marginal log likelihood.
//
// Behavior highlights:
//   - Every error discards incomplete intermediate tables; no partial workspace is published.
//
// Inputs:
//   - ctx: active context; c: immutable chain; t: positive length; emit: deterministic log-emission callback.
//
// Returns:
//   - *posteriorWorkspace: private owned T*K buffers and finite sequence evidence.
//
// Errors:
//   - Propagates context, emission, ErrImpossibleSequence, ErrOverflow, ErrNumerical.
//
// Determinism:
//   - One chronological forward pass and one reverse-time pass.
//
// Complexity:
//   - Time O(T*K^2) plus emissions, Space O(T*K).
//
// Notes:
//   - No returned array aliases the immutable model or caller observation matrix.
//
// AI-Hints:
//   - Use this kernel in Smooth; obtain xi separately only when fitting parameters.
func buildPosteriorWorkspace(ctx context.Context, c *chain, t int, emit func(int, []float64) error) (*posteriorWorkspace, error) {
	emissions, err := collectEmissions(ctx, c, t, emit)
	if err != nil {
		return nil, err
	}
	alpha, likelihood, err := forwardMessages(ctx, c, t, emissions)
	if err != nil {
		return nil, err
	}
	beta, gamma, err := backwardMessages(ctx, c, t, emissions, alpha)
	if err != nil {
		return nil, err
	}
	return &posteriorWorkspace{emissions: emissions, alpha: alpha, beta: beta, gamma: gamma, logLikelihood: likelihood}, nil
}

// posteriorTables returns the EM sufficient-statistic tables for a single sequence.
// Its xi table never contains a pair that crosses a training-sequence boundary.
//
// Implementation:
//   - Stage 1: Build normalized emissions, alpha, beta and gamma for the sequence.
//   - Stage 2: Compute the within-sequence xi transition posteriors.
//   - Stage 3: Return owned alpha, gamma, xi and sequence log likelihood.
//
// Behavior highlights:
//   - The caller adds gamma[0,*] once per independent sequence to the prior counts.
//
// Inputs:
//   - ctx: active context; c: immutable chain; t: positive length; emit: log-emission callback.
//
// Returns:
//   - alpha and gamma: T*K; xi: (T-1)*K*K; logLikelihood: marginal log evidence.
//
// Errors:
//   - Any forward/backward error or ErrNumerical on pair normalization.
//
// Determinism:
//   - Sequence order is caller-controlled; this helper uses fixed row/state order.
//
// Complexity:
//   - Time O(T*K^2) plus emissions, Space O(T*K + (T-1)*K^2).
//
// Notes:
//   - All tables are detached per call and may be discarded after accumulating EM counts.
//
// AI-Hints:
//   - Do not use xi as an unconditional transition frequency; it is conditioned on X and theta.
func posteriorTables(ctx context.Context, c *chain, t int, emit func(int, []float64) error) ([]float64, []float64, []float64, float64, error) {
	work, err := buildPosteriorWorkspace(ctx, c, t, emit)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	xi, err := transitionExpectations(ctx, c, t, work)
	if err != nil {
		return nil, nil, nil, 0, err
	}
	return work.alpha, work.gamma, xi, work.logLikelihood, nil
}

// smooth constructs the complete retrospective hidden-state posterior for one sequence.
// It deliberately differs from Filter because beta uses later observations.
//
// Implementation:
//   - Stage 1: Compute the shared emission, forward, and backward tables.
//   - Stage 2: Select gamma rather than alpha as the published posterior.
//   - Stage 3: Materialize an independent T-by-K matrix with total marginal evidence.
//
// Behavior highlights:
//   - No (T-1)*K*K xi history is constructed during inference.
//
// Inputs:
//   - ctx: active context; c: immutable chain; t: positive observations; emit: log-emission callback.
//
// Returns:
//   - *SmoothResult: caller-owned gamma matrix and finite marginal logLikelihood.
//
// Errors:
//   - Propagates context, emission, matrix allocation errors and numerical sentinels.
//
// Determinism:
//   - Forward and backward passes use deterministic state traversal.
//
// Complexity:
//   - Time O(T*K^2) plus emissions, Space O(T*K) excluding Gaussian input copy.
//
// Notes:
//   - The returned matrix does not alias workspace or model parameters.
//
// AI-Hints:
//   - Do not report gamma[t] as P(z_t | x_0:t); that causal quantity is alpha[t].
func smooth(ctx context.Context, c *chain, t int, emit func(int, []float64) error) (*SmoothResult, error) {
	work, err := buildPosteriorWorkspace(ctx, c, t, emit)
	if err != nil {
		return nil, err
	}
	result, err := denseFromFlat(t, c.k, work.gamma)
	if err != nil {
		return nil, err
	}
	return &SmoothResult{Posterior: result, LogLikelihood: work.logLikelihood}, nil
}

// Smooth computes retrospective categorical posterior probabilities over a whole sequence.
// Unlike Filter, the posterior at time t may depend on all later observations.
// Implementation:
//   - Stage 1: Validate context, receiver, and every categorical observation.
//   - Stage 2: Compute log emissions, normalized forward messages, then backward messages.
//   - Stage 3: Materialize detached gamma probabilities without EM transition tables.
//
// Behavior highlights:
//   - The posterior row at each time is normalized and conditions on the whole sequence.
//
// Inputs:
//   - ctx: required; x: nonempty, read-only symbols in the model alphabet.
//
// Returns:
//   - *SmoothResult: detached T-by-K probabilities and marginal log likelihood.
//
// Errors:
//   - ErrInvalidOption/ctx.Err, ErrNilInput, ErrEmptyInput, ErrShape, ErrImpossibleSequence, ErrOverflow, ErrNumerical.
//
// Determinism:
//   - Fixed temporal forward order, reverse-time backward order, ascending state indices.
//
// Complexity:
//   - Time O(T*K^2); space O(T*K), including the returned matrix.
//
// Notes:
//   - The model and input remain untouched; all returned matrices are caller-owned.
//
// AI-Hints:
//   - Do not use these future-conditioned probabilities to evaluate live alert causality.
func (m *CategoricalModel) Smooth(ctx context.Context, x []uint32) (*SmoothResult, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if m == nil {
		return nil, ErrNilInput
	}
	if err := validateSymbols(x, m.alphabet); err != nil {
		return nil, err
	}
	return smooth(ctx, &m.chain, len(x), func(i int, e []float64) error { return m.emissions(x[i], e) })
}

// Smooth computes retrospective diagonal-Gaussian hidden-state posteriors.
// Every returned state distribution conditions on the complete observed trajectory.
// Implementation:
//   - Stage 1: Validate and detach the finite T-by-D observation matrix.
//   - Stage 2: Compute log Gaussian emissions and the normalized forward-backward tables.
//   - Stage 3: Return a detached T-by-K matrix and full marginal log density.
//
// Behavior highlights:
//   - EM-only transition expectations are not allocated during smoothing.
//
// Inputs:
//   - ctx: required; x: nonempty T-by-D finite observations, read-only.
//
// Returns:
//   - *SmoothResult: probabilities conditioned on all T observations.
//
// Errors:
//   - ErrInvalidOption/ctx.Err, ErrNilInput, ErrEmptyInput, ErrShape, ErrNonFinite, ErrOverflow, ErrNumerical.
//
// Determinism:
//   - Fixed state/feature order and reverse-time backward recursion.
//
// Complexity:
//   - Time O(T*(K^2+K*D)), space O(T*K+T*D).
//
// Notes:
//   - This operation intentionally depends on future observations, unlike Filter.
//
// AI-Hints:
//   - Use it for postmortem regime attribution and offline annotation, not prospective decisions.
func (m *GaussianModel) Smooth(ctx context.Context, x matrix.Matrix) (*SmoothResult, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if m == nil {
		return nil, ErrNilInput
	}
	t, flat, err := validateObservations(x, m.dimension)
	if err != nil {
		return nil, err
	}
	return smooth(ctx, &m.chain, t, func(i int, e []float64) error { return m.emissions(flat[i*m.dimension:(i+1)*m.dimension], e) })
}
