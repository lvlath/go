// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm estimates categorical HMM parameters using independent-sequence EM.
// This boundary accumulates prior, transition, and emission responsibilities within each sequence.
package hmm

import (
	"context"
	"fmt"
)

// categoricalCounts stores additive expected sufficient statistics for categorical Baum-Welch.
// Its prior and transition counts explicitly distinguish sequence starts from within-sequence steps.
//
// Implementation:
//   - Stage 1: Allocate zeroed arrays for K initial masses, K*K transitions and K*A symbols.
//   - Stage 2: Accumulate gamma/xi in fixed sequence and time order.
//   - Stage 3: Pass the completed statistics to the constrained MAP M-step.
//
// Behavior highlights:
//   - occupancy includes every observed time; prior includes exactly one gamma row per sequence.
//
// Inputs:
//   - prior[K], trans[K*K], emission[K*A], occupancy[K] are EM-only floating counts; ll is total log evidence.
//
// Returns:
//   - A single-call workspace value for one E-step across all supplied sequences.
//
// Errors:
//   - Categorical expectation returns an error instead of exposing incomplete counts.
//
// Determinism:
//   - Arrays are always accumulated in declared sequence/time/state order.
//
// Complexity:
//   - Storage O(K*K+K*A), excluding the per-sequence posterior tables.
//
// Notes:
//   - Counts are not persistent model state and are not safe for concurrent mutation.
//
// AI-Hints:
//   - Do not confuse zero outgoing transitions from a singleton sequence with zero state occupancy.
type categoricalCounts struct {
	prior     []float64 // K gamma[0,j] contributions, one start per independent sequence.
	trans     []float64 // K*K within-sequence xi transition-pair responsibilities.
	emission  []float64 // K*A weighted categorical observation counts.
	occupancy []float64 // K expected visits over all time steps of all sequences.
	ll        float64   // Sum of per-sequence marginal log likelihoods.
}

// categoricalExpectation evaluates the categorical EM E-step on independent observation sequences.
// It accumulates gamma state occupancy and xi transitions without joining session endpoints.
//
// Implementation:
//   - Stage 1: Allocate K/K*K/K*A count buffers for the current immutable model.
//   - Stage 2: Run posteriorTables separately for each sequence; add its gamma and internal xi cells.
//   - Stage 3: Accumulate the sequence evidence and reject unrepresentable total log likelihood.
//
// Behavior highlights:
//   - Every sequence contributes one initial gamma; only its T_s-1 adjacent pairs contribute xi.
//
// Inputs:
//   - ctx: active context; m: immutable categorical model; sequences: validated nonempty symbol slices.
//
// Returns:
//   - categoricalCounts: expected initial/transitional/emission mass and aggregate log evidence.
//
// Errors:
//   - ctx.Err(), ErrImpossibleSequence, ErrOverflow, ErrNumerical; sequence index wraps failures.
//
// Determinism:
//   - Sequences outermost, time then states, then transition flat cells ascending.
//
// Complexity:
//   - Time O(sum_s T_s*K^2), Space O(max_s T_s*K^2+K*K+K*A).
//
// Notes:
//   - Transient posterior arrays are discarded at each sequence boundary.
//
// AI-Hints:
//   - Reset the prior for each sequence; concatenation fabricates spurious hidden transitions.
func categoricalExpectation(ctx context.Context, m *CategoricalModel, sequences [][]uint32) (categoricalCounts, error) {
	k, a := m.chain.k, m.alphabet
	out := categoricalCounts{prior: make([]float64, k), trans: make([]float64, k*k), emission: make([]float64, k*a), occupancy: make([]float64, k)}
	// Run a fresh posterior recursion for each session; carrying alpha
	// across these boundaries would invent transitions that never occurred.
	for s, x := range sequences {
		_, gamma, xi, ll, err := posteriorTables(ctx, &m.chain, len(x), func(i int, e []float64) error { return m.emissions(x[i], e) })
		if err != nil {
			return out, fmt.Errorf("sequence %d: %w", s, err)
		}
		out.ll += ll
		for i := 0; i < len(x); i++ {
			for j := 0; j < k; j++ {
				g := gamma[i*k+j]
				out.occupancy[j] += g
				out.emission[j*a+int(x[i])] += g
				// Every independent session contributes one prior responsibility,
				// regardless of what hidden state the previous session ended in.
				if i == 0 {
					out.prior[j] += g
				}
			}
		}
		// xi has exactly T_s-1 slices; a singleton session adds no
		// transitions but still contributes prior, occupancy and emissions.
		for i := 0; i < len(x)-1; i++ {
			for v := 0; v < k*k; v++ {
				out.trans[v] += xi[i*k*k+v]
			}
		}
	}
	if !finite(out.ll) {
		return out, ErrOverflow
	}
	return out, nil
}

// newCategoricalFromFlat constructs a fully validated categorical candidate from M-step slices.
// It converts private row-major buffers to the existing public constructor contract.
//
// Implementation:
//   - Stage 1: Allocate a fresh K-by-K transition matrix.
//   - Stage 2: Allocate a K-by-A emission matrix and copy every M-step cell.
//   - Stage 3: Call NewCategorical to validate, copy, normalize and hash the accepted model.
//
// Behavior highlights:
//   - A candidate gets a content-derived new ModelID; no mutable existing model is patched.
//
// Inputs:
//   - k: states; a: symbols; prior[K], trans[K*K], em[K*A]: private candidate arrays.
//
// Returns:
//   - *CategoricalModel: fully immutable candidate or nil with an error.
//
// Errors:
//   - Matrix allocation/Set errors or constructor ErrProbability/ErrNonFinite/ErrShape.
//
// Determinism:
//   - Dense row-major storage and constructor validation in fixed ascending order.
//
// Complexity:
//   - Time O(K^2+K*A), Space O(K^2+K*A).
//
// Notes:
//   - The extra conversion is intentional until the public matrix constructor boundary changes.
//
// AI-Hints:
//   - Never publish partially normalized raw M-step slices as a model.
func newCategoricalFromFlat(k, a int, prior, trans, em []float64) (*CategoricalModel, error) {
	t, err := denseFromFlat(k, k, trans)
	if err != nil {
		return nil, err
	}
	e, err := denseFromFlat(k, a, em)
	if err != nil {
		return nil, err
	}
	return NewCategorical(CategoricalSpec{Common: Common{Prior: prior, Transition: t}, Emission: e})
}

// FitCategorical estimates categorical HMM parameters with deterministic Baum-Welch EM.
// Each input sequence starts afresh from Prior; no inter-session transition is learned.
// Implementation:
//   - Stage 1: Validate context/options, supplied initial model, and every symbol sequence.
//   - Stage 2: Accumulate per-sequence gamma/xi and perform Dirichlet-penalized M-steps.
//   - Stage 3: Accept monotone objective improvements or return the last valid diagnostics.
//
// Behavior highlights:
//   - Pseudocount=0 means no penalty; dead rows keep previous values when unobserved.
//
// Inputs:
//   - ctx: required; sequences: independent nonempty categorical series; initial: required seed.
//   - opts: iteration limit, tolerance, and nonnegative Dirichlet pseudocount.
//
// Returns:
//   - *CategoricalFitResult: immutable last accepted model with penalized objective history.
//
// Errors:
//   - Input sentinels, ctx.Err, ErrImpossibleSequence; ErrNotConverged/ErrNumerical may accompany a result.
//
// Determinism:
//   - Input sequence order, time order, state order, and emission order remain fixed.
//
// Complexity:
//   - Time O(iterations*sum(T_s)*K^2); memory O(max(T_s)*K^2+K*A).
//
// Notes:
//   - No hidden parameter randomization; initial probabilities and zeroes are respected.
//
// AI-Hints:
//   - Always inspect both returned error and Diagnostics.Converged before deploying a model.
func FitCategorical(ctx context.Context, sequences [][]uint32, initial CategoricalSpec, opts FitOptions) (*CategoricalFitResult, error) {
	opts, err := checkFitInput(ctx, opts)
	if err != nil {
		return nil, err
	}
	model, err := NewCategorical(initial)
	if err != nil {
		return nil, err
	}
	if sequences == nil {
		return nil, ErrNilInput
	}
	if len(sequences) == 0 {
		return nil, ErrEmptyInput
	}
	for _, x := range sequences {
		if err = validateSymbols(x, model.alphabet); err != nil {
			return nil, err
		}
	}
	counts, err := categoricalExpectation(ctx, model, sequences)
	if err != nil {
		return nil, err
	}
	oldScore := counts.ll + simplexPenalty(opts.Pseudocount, model.chain.prior, model.chain.transition, model.emission)
	out := &CategoricalFitResult{Model: model, Diagnostics: Diagnostics{Objective: []float64{oldScore}, Occupancy: counts.occupancy}}
	for iteration := 0; iteration < opts.MaxIterations; iteration++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		prior := updateSimplex(counts.prior, model.chain.prior, 1, model.chain.k, opts.Pseudocount)
		trans := updateSimplex(counts.trans, model.chain.transition, model.chain.k, model.chain.k, opts.Pseudocount)
		emission := updateSimplex(counts.emission, model.emission, model.chain.k, model.alphabet, opts.Pseudocount)
		candidate, err := newCategoricalFromFlat(model.chain.k, model.alphabet, prior, trans, emission)
		if err != nil {
			return out, fmt.Errorf("M-step: %w", err)
		}
		// Evaluate the candidate against the *whole independent-sequence*
		// data likelihood; monotonicity is tested only after adding the MAP penalty.
		newCounts, err := categoricalExpectation(ctx, candidate, sequences)
		if err != nil {
			return nil, err
		}
		score := newCounts.ll + simplexPenalty(opts.Pseudocount, candidate.chain.prior, candidate.chain.transition, candidate.emission)
		done, err := monotonic(oldScore, score, opts.Tolerance)
		if err != nil {
			return out, err
		}
		out.Model = candidate
		out.Diagnostics.Iterations++
		out.Diagnostics.Objective = append(out.Diagnostics.Objective, score)
		out.Diagnostics.Occupancy = newCounts.occupancy
		if done {
			out.Diagnostics.Converged = true
			return out, nil
		}
		model = candidate
		counts = newCounts
		oldScore = score
	}
	return out, ErrNotConverged
}

// emissions writes categorical log emission probabilities for a single observed symbol.
// The observation selects one column shared by all hidden states.
//
// Implementation:
//   - Stage 1: Reject symbol indexes outside the fixed alphabet.
//   - Stage 2: Read the corresponding emission cell from each state in ascending order.
//   - Stage 3: Write log probabilities, preserving exact zero as negative infinity.
//
// Behavior highlights:
//   - A valid but universally impossible symbol is classified by the forward kernel, not this lookup.
//
// Inputs:
//   - m: immutable categorical model; x: symbol in [0,A); dst: caller-owned workspace of K cells.
//
// Returns:
//   - error: nil with fully filled dst, or ErrShape for invalid symbol.
//
// Errors:
//   - ErrShape for out-of-alphabet symbol.
//
// Determinism:
//   - States visited from 0 to K-1; input symbol never changes.
//
// Complexity:
//   - Time O(K), Space O(1) beyond dst.
//
// Notes:
//   - The caller must allocate dst with len>=K; this is a private kernel precondition.
//
// AI-Hints:
//   - Do not take a Gaussian feature value and cast it into x without a defined categorical encoder.
func (m *CategoricalModel) emissions(x uint32, dst []float64) error {
	// Both uint32 symbols and positive int alphabet sizes fit in int64 on all Go targets.
	// A signed comparison avoids wrapping an invalid negative alphabet into uint64.
	if int64(x) >= int64(m.alphabet) {
		return ErrShape
	}
	for k := 0; k < m.chain.k; k++ {
		dst[k] = logProbability(m.emission[k*m.alphabet+int(x)])
	}
	return nil
}
