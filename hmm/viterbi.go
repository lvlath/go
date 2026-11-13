// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm implements global maximum-joint-probability path decoding.
// This boundary uses deterministic lowest-index predecessor and terminal-state tie breaking.
package hmm

import (
	"context"
	"math"

	"github.com/lvlath/go/matrix"
)

// decode finds the single highest joint-probability hidden-state trajectory with Viterbi.
// It is a max-product dynamic program in logarithms, not a per-time marginal argmax.
//
// Implementation:
//   - Stage 1: Check the T*K predecessor table size and allocate two score rows.
//   - Stage 2: For each observed time, maximize predecessor log joint plus log transition and emission.
//   - Stage 3: Choose the lowest-index optimal terminal state and trace deterministic backpointers.
//
// Behavior highlights:
//   - Strict greater-than comparisons preserve the first predecessor in exact ties.
//
// Inputs:
//   - ctx: active context; c: chain; t: positive sequence length; emit: fills K log-emission values.
//
// Returns:
//   - *DecodeResult: caller-owned length-T state path and maximum joint log score.
//
// Errors:
//   - ctx.Err(), ErrOverflow, ErrImpossibleSequence, or emission callback error.
//
// Determinism:
//   - Time ascending; destination then predecessor ascending; final state ascending.
//
// Complexity:
//   - Time O(T*K^2) plus emissions, Space O(T*K) backpointers and O(K) score rows.
//
// Notes:
//   - Backpointers are private and the returned states do not alias the workspace.
//
// AI-Hints:
//   - Avoid choosing argmax of Smooth.Posterior rows: those may violate legal transitions.
func decode(ctx context.Context, c *chain, t int, emit func(int, []float64) error) (*DecodeResult, error) {
	k := c.k
	if t > int(^uint(0)>>1)/k {
		return nil, ErrOverflow
	}
	pointers := make([]int, t*k)
	previous := make([]float64, k)
	current := make([]float64, k)
	em := make([]float64, k)
	// Maximum *joint* scores advance forward; retaining only two score rows
	// keeps scoring memory O(K) while pointers retain the optimal ancestors.
	for i := 0; i < t; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := emit(i, em); err != nil {
			return nil, err
		}
		for j := 0; j < k; j++ {
			if i == 0 {
				current[j] = logProbability(c.prior[j]) + em[j]
				continue
			}
			best := math.Inf(-1)
			arg := 0
			// Strict greater-than preserves the earliest (smallest-index)
			// predecessor on exact ties; >= would break deterministic witness order.
			for p := 0; p < k; p++ {
				v := previous[p] + logProbability(c.transition[p*k+j])
				if v > best {
					best = v
					arg = p
				}
			}
			current[j] = best + em[j]
			pointers[i*k+j] = arg
		}
		previous, current = current, previous
		if math.IsInf(LogSumExpSlice(previous), -1) {
			return nil, ErrImpossibleSequence
		}
	}
	// The terminal state is selected by the same strict-improvement rule,
	// resolving equal final scores toward the smallest index.
	best := math.Inf(-1)
	last := 0
	for j, v := range previous {
		if v > best {
			best = v
			last = j
		}
	}
	if !finite(best) {
		return nil, ErrOverflow
	}
	path := make([]int, t)
	path[t-1] = last
	for i := t - 1; i > 0; i-- {
		path[i-1] = pointers[i*k+path[i]]
	}
	return &DecodeResult{States: path, LogJointProbability: best}, nil
}

// Decode computes the globally most likely categorical hidden-state path with Viterbi.
// The returned states jointly maximize P(z_0:T-1,x_0:T-1), not marginal posterior states.
// Implementation:
//   - Stage 1: Validate context/model and each observed categorical symbol.
//   - Stage 2: Compute maximum log joint scores and predecessor backpointers.
//   - Stage 3: Choose the lowest-index optimal terminal state and backtrack the path.
//
// Behavior highlights:
//   - Exact ties use the lowest predecessor index, then lowest final-state index.
//
// Inputs:
//   - ctx: required; x: nonempty categorical observation sequence, read-only.
//
// Returns:
//   - *DecodeResult: detached length-T path and log joint path probability.
//
// Errors:
//   - ErrInvalidOption/ctx.Err, ErrNilInput, ErrEmptyInput, ErrShape, ErrImpossibleSequence, ErrOverflow.
//
// Determinism:
//   - Fixed forward state order, ascending predecessor comparisons, reverse backtracking.
//
// Complexity:
//   - Time O(T*K^2), space O(T*K) for backpointers and the path.
//
// Notes:
//   - The path is global and therefore not prefix-invariant like causal Filter.
//
// AI-Hints:
//   - Never approximate this operation by argmax of individual Smooth posterior rows.
func (m *CategoricalModel) Decode(ctx context.Context, x []uint32) (*DecodeResult, error) {
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	if m == nil {
		return nil, ErrNilInput
	}
	if err := validateSymbols(x, m.alphabet); err != nil {
		return nil, err
	}
	return decode(ctx, &m.chain, len(x), func(i int, e []float64) error { return m.emissions(x[i], e) })
}

// Decode computes the maximum joint hidden-state trajectory of Gaussian observations.
// It uses diagonal-Gaussian emission log densities and deterministic Viterbi ties.
// Implementation:
//   - Stage 1: Validate context and detach finite T-by-D observations.
//   - Stage 2: Evaluate state emission scores and maximum log transition paths.
//   - Stage 3: Backtrack a single globally optimal path of length T.
//
// Behavior highlights:
//   - Lowest predecessor and terminal state indices resolve exact score ties.
//
// Inputs:
//   - ctx: required; x: nonempty T-by-D finite matrix, read-only.
//
// Returns:
//   - *DecodeResult: state indices and maximum log joint density.
//
// Errors:
//   - ErrInvalidOption/ctx.Err, ErrNilInput, ErrEmptyInput, ErrShape, ErrNonFinite, ErrOverflow.
//
// Determinism:
//   - Time, feature, state, and predecessor orders are fixed.
//
// Complexity:
//   - Time O(T*(K^2+K*D)); space O(T*K+T*D).
//
// Notes:
//   - The decoded path can change when future observations are appended.
//
// AI-Hints:
//   - For online, future-free decisions use FilterStep rather than Decode.
func (m *GaussianModel) Decode(ctx context.Context, x matrix.Matrix) (*DecodeResult, error) {
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
	return decode(ctx, &m.chain, t, func(i int, e []float64) error { return m.emissions(flat[i*m.dimension:(i+1)*m.dimension], e) })
}
