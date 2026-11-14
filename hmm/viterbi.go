// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm implements global maximum-joint-probability path decoding.
// This boundary uses deterministic lowest-index predecessor and terminal-state tie breaking.
package hmm

import (
	"context"
	"math"
	"math/big"
	"strconv"

	"github.com/lvlath/go/matrix"
)

// pathScore preserves structural support separately from the numerical path cost.
// Its extended representation is allocated only after ordinary float64 addition overflows.
//
// Implementation:
//   - Stage 1: Track whether a path exists through the reachable bit.
//   - Stage 2: Keep representable path scores as ordinary float64 values.
//   - Stage 3: Retain exceptional finite-input sums in a higher-range big.Float.
//
// Behavior highlights:
//   - A cost below -MaxFloat64 is not an impossible structural path.
//
// Inputs:
//   - score: compact value; wide: exceptional value; reachable: supported path flag.
//
// Returns:
//   - One transient Viterbi candidate; never a published model or result.
//
// Errors:
//   - None; callers classify unrepresentable final values as ErrOverflow.
//
// Determinism:
//   - Stable tie comparisons follow increasing predecessor index.
//
// Complexity:
//   - O(1) storage normally; exceptional values allocate O(precision) words.
//
// Notes:
//   - The exceptional path uses 256-bit precision to preserve scale beyond float64.
//
// AI-Hints:
//   - Never encode arithmetic overflow as a structural -Inf sentinel.
type pathScore struct {
	score     float64
	wide      *big.Float
	reachable bool
}

const pathScorePrecision = 256 // Fractional precision for exceptional Viterbi sums only.

// extendPathScore adds one finite log probability without losing an existing path.
// A nonrepresentable intermediate remains comparable and can later reenter float64 range.
//
// Implementation:
//   - Stage 1: Return unreachable input unchanged, and reject no values implicitly.
//   - Stage 2: Use fast float64 addition for ordinary representable candidates.
//   - Stage 3: Use 256-bit exponent-extended arithmetic when needed.
//
// Behavior highlights:
//   - The caller must skip log(0) arcs before calling this function.
//
// Inputs:
//   - current: supported score; increment: finite log emission or transition score.
//
// Returns:
//   - pathScore: independently owned accumulated candidate.
//
// Errors:
//   - None; final publication verifies representability.
//
// Determinism:
//   - Fixed nearest-even precision and left-to-right addition order.
//
// Complexity:
//   - O(1) fast path; O(pathScorePrecision) exceptional arithmetic.
//
// Notes:
//   - The original candidate is immutable and remains safe for competing successors.
//
// AI-Hints:
//   - Do not use a single float64 -Inf as both zero transition mass and overflow.
func extendPathScore(current pathScore, increment float64) pathScore {
	if !current.reachable {
		return current
	}
	if current.wide == nil {
		sum := current.score + increment
		if finite(sum) {
			return pathScore{score: sum, reachable: true}
		}
		current.wide = new(big.Float).SetPrec(pathScorePrecision).SetFloat64(current.score)
	} else {
		current.wide = new(big.Float).SetPrec(pathScorePrecision).Set(current.wide)
	}
	step := new(big.Float).SetPrec(pathScorePrecision).SetFloat64(increment)
	current.wide.Add(current.wide, step)
	return current
}

// greaterPathScore compares two supported Viterbi candidates without overflowing.
// The comparison is strict to preserve lowest-index predecessor tie-breaking.
//
// Implementation:
//   - Stage 1: Handle structural presence before comparing score magnitudes.
//   - Stage 2: Compare two ordinary float64 values directly.
//   - Stage 3: Promote only mixed exceptional comparisons into high-range arithmetic.
//
// Behavior highlights:
//   - A finite candidate always outranks a negative-overflowed candidate.
//
// Inputs:
//   - candidate, best: read-only scores for two competing paths to the same state.
//
// Returns:
//   - bool: true exactly when candidate has a strictly higher joint log score.
//
// Errors:
//   - None; only supported finite-input candidates reach this helper.
//
// Determinism:
//   - Equal scores never replace an earlier predecessor.
//
// Complexity:
//   - O(1) normally; O(pathScorePrecision) for exceptional comparison.
//
// Notes:
//   - A later positive Gaussian emission can recover a path with an exceptional prefix.
//
// AI-Hints:
//   - Avoid comparing -Inf sentinels: consult reachable first.
func greaterPathScore(candidate, best pathScore) bool {
	if !candidate.reachable {
		return false
	}
	if !best.reachable {
		return true
	}
	if candidate.wide == nil && best.wide == nil {
		return candidate.score > best.score
	}
	left := candidate.wide
	if left == nil {
		left = new(big.Float).SetPrec(pathScorePrecision).SetFloat64(candidate.score)
	}
	right := best.wide
	if right == nil {
		right = new(big.Float).SetPrec(pathScorePrecision).SetFloat64(best.score)
	}
	return left.Cmp(right) > 0
}

// decode finds the single globally maximum joint hidden-state trajectory by Viterbi.
// It distinguishes an absent path from a supported path whose score overflows float64.
//
// Implementation:
//   - Stage 1: Check the byte footprint of backpointers before any emission access.
//   - Stage 2: Maximize predecessor-supported scores, retaining rare wide sums.
//   - Stage 3: Reject an unrepresentable optimal score or backtrack the detached path.
//
// Behavior highlights:
//   - Losing overflowed candidates do not fail a finite winning path.
//   - Strict comparison selects the earliest predecessor, then earliest terminal state.
//
// Inputs:
//   - ctx: active context; c: validated immutable chain; t: positive observation count.
//   - emit: fills K log emission scores for each chronological observation.
//
// Returns:
//   - *DecodeResult: detached T-state witness and finite maximum log joint score.
//
// Errors:
//   - ErrOverflow on byte limits or an unrepresentable optimal supported score.
//   - ErrImpossibleSequence only when all paths are structurally unsupported.
//   - ctx.Err() or errors from the emission callback.
//
// Determinism:
//   - Time, successor, then predecessor order; strict greater-than comparisons.
//
// Complexity:
//   - Time O(T*K^2) normally; Space O(T*K) backpointers and O(K) scores.
//   - Exceptionally wide-score operations add fixed-precision work on overflowed paths.
//
// Notes:
//   - No model or input mutation; no partial result on overflow or cancellation.
//
// AI-Hints:
//   - A negative-infinite arithmetic score is not proof that all transition probabilities are zero.
//   - Do not replace this decoder with independent argmax of smoothed marginals.
func decode(ctx context.Context, c *chain, t int, emit func(int, []float64) error) (*DecodeResult, error) {
	k := c.k
	count, err := checkedAllocation(t, k, strconv.IntSize/8)
	if err != nil {
		return nil, err
	}
	pointers := make([]int, count)
	previous := make([]pathScore, k)
	current := make([]pathScore, k)
	em := make([]float64, k)

	for i := 0; i < t; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := emit(i, em); err != nil {
			return nil, err
		}
		anySupported := false
		for j := 0; j < k; j++ {
			current[j] = pathScore{}
			if math.IsInf(em[j], -1) {
				continue
			}
			if !finite(em[j]) {
				return nil, ErrOverflow
			}
			if i == 0 {
				prior := logProbability(c.prior[j])
				if !math.IsInf(prior, -1) {
					current[j] = extendPathScore(pathScore{score: prior, reachable: true}, em[j])
					anySupported = true
				}
				continue
			}
			best := pathScore{}
			arg := 0
			for p := 0; p < k; p++ {
				transition := logProbability(c.transition[p*k+j])
				if !previous[p].reachable || math.IsInf(transition, -1) {
					continue
				}
				candidate := extendPathScore(previous[p], transition)
				if greaterPathScore(candidate, best) {
					best = candidate
					arg = p
				}
			}
			if best.reachable {
				current[j] = extendPathScore(best, em[j])
				pointers[i*k+j] = arg
				anySupported = true
			}
		}
		if !anySupported {
			return nil, ErrImpossibleSequence
		}
		previous, current = current, previous
	}

	best := pathScore{}
	last := 0
	for j, candidate := range previous {
		if greaterPathScore(candidate, best) {
			best = candidate
			last = j
		}
	}
	score := best.score
	if best.wide != nil {
		score, _ = best.wide.Float64()
	}
	if !finite(score) {
		return nil, ErrOverflow
	}
	path := make([]int, t)
	path[t-1] = last
	for i := t - 1; i > 0; i-- {
		path[i-1] = pointers[i*k+path[i]]
	}
	return &DecodeResult{States: path, LogJointProbability: score}, nil
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
