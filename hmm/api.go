// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm provides validated HMM constructors and transparent convenience facades.
// This boundary normalizes copied parameters, assigns canonical identities, and delegates inference.
package hmm

import (
	"context"

	"github.com/lvlath/go/matrix"
)

// NewCategorical constructs an immutable categorical HMM from explicit distributions.
// It copies and canonicalizes the stochastic parameters before deriving ModelID.
// Implementation:
//   - Stage 1: Validate presence, chain shape, finite values, and stochastic rows.
//   - Stage 2: Copy K-by-A emissions and normalize only permitted row-sum residuals.
//   - Stage 3: Hash canonical parameters and return an independent model.
//
// Behavior highlights:
//   - Exact zero probabilities remain zero; no pseudocount or RNG is applied.
//
// Inputs:
//   - spec: K states, A>=1 symbols, K-by-K transitions and K-by-A emissions.
//
// Returns:
//   - *CategoricalModel: immutable copied parameters with stable canonical ModelID.
//
// Errors:
//   - ErrNilInput, ErrShape, ErrNonFinite, ErrProbability, ErrOverflow; wrapped matrix errors.
//
// Determinism:
//   - Row-major matrix traversal, exact normalized float64 hash encoding.
//
// Complexity:
//   - Time O(K^2+K*A), space O(K^2+K*A) for private parameters.
//
// Notes:
//   - Caller inputs remain owned by the caller; concurrent mutation during construction is unsupported.
//
// AI-Hints:
//   - Column indices are symbolic categories, not arbitrary numeric feature values.
func NewCategorical(spec CategoricalSpec) (*CategoricalModel, error) {
	if spec.Common.Prior == nil || isNilMatrix(spec.Common.Transition) || isNilMatrix(spec.Emission) {
		return nil, ErrNilInput
	}
	c, err := validateCommon(spec.Common)
	if err != nil {
		return nil, err
	}
	a := spec.Emission.Cols()
	if a < 1 {
		return nil, ErrShape
	}
	e, err := readMatrix(spec.Emission, c.k, a)
	if err != nil {
		return nil, err
	}
	if err = normalizeRows(e, c.k, a); err != nil {
		return nil, err
	}
	c.id = hashParameters("categorical", c.k, a, c.prior, c.transition, e)
	return &CategoricalModel{chain: c, alphabet: a, emission: e}, nil
}

// NewGaussian constructs an immutable HMM with diagonal-Gaussian emissions.
// Every state's D means and strictly positive variances are independently copied.
// Implementation:
//   - Stage 1: Validate presence, stochastic chain, and K-by-D parameter shapes.
//   - Stage 2: Copy finite means and finite strictly positive variances.
//   - Stage 3: Hash canonical parameters and publish a detached immutable model.
//
// Behavior highlights:
//   - No floor is imposed by this constructor; FitGaussian enforces its explicit floor.
//
// Inputs:
//   - spec: K states, D>=1 features, K-by-K transition and K-by-D emission parameters.
//
// Returns:
//   - *GaussianModel: privately owned model with a stable identity.
//
// Errors:
//   - ErrNilInput, ErrShape, ErrNonFinite, ErrProbability, ErrOverflow; matrix errors.
//
// Determinism:
//   - Row-major chain, then state-major feature ordering in identity generation.
//
// Complexity:
//   - Time O(K^2+K*D), space O(K^2+K*D) for private parameters.
//
// Notes:
//   - Variance units equal feature units squared; no full covariance is supported.
//
// AI-Hints:
//   - FitGaussian rejects an initial variance below VarianceFloor instead of silently projecting it.
func NewGaussian(spec GaussianSpec) (*GaussianModel, error) {
	if spec.Common.Prior == nil || isNilMatrix(spec.Common.Transition) || isNilMatrix(spec.Means) || isNilMatrix(spec.Variances) {
		return nil, ErrNilInput
	}
	c, err := validateCommon(spec.Common)
	if err != nil {
		return nil, err
	}
	d := spec.Means.Cols()
	if d < 1 {
		return nil, ErrShape
	}
	means, err := readMatrix(spec.Means, c.k, d)
	if err != nil {
		return nil, err
	}
	variances, err := readMatrix(spec.Variances, c.k, d)
	if err != nil {
		return nil, err
	}
	for _, v := range variances {
		if v <= 0 {
			return nil, ErrProbability
		}
	}
	c.id = hashParameters("diagonal-gaussian", c.k, d, c.prior, c.transition, means, variances)
	return &GaussianModel{chain: c, dimension: d, means: means, variances: variances}, nil
}

// denseFromFlat publishes an independent matrix from a validated private row-major slice.
// It is the common output and candidate-construction bridge into matrix.Matrix.
//
// Implementation:
//   - Stage 1: Allocate a zero matrix with the requested shape via matrix.NewZeros.
//   - Stage 2: Copy each row/column from the flat buffer through checked Set.
//   - Stage 3: Return only the fully constructed matrix.
//
// Behavior highlights:
//   - All inputs must already satisfy len(v)=rows*cols and finite-domain requirements.
//
// Inputs:
//   - rows, cols: validated dimensions; v: immutable read-only private numeric data.
//
// Returns:
//   - matrix.Matrix, error: detached owning matrix on success.
//
// Errors:
//   - Wrapped matrix constructor or Set errors; callers validate dimensions before entry.
//
// Determinism:
//   - Rows then columns increase; no map iteration.
//
// Complexity:
//   - Time O(rows*cols), Space O(rows*cols).
//
// Notes:
//   - The returned matrix never aliases v or model internals.
//
// AI-Hints:
//   - Do not hand this helper unvalidated arbitrary user slices; shape is an internal invariant.
func denseFromFlat(rows, cols int, v []float64) (matrix.Matrix, error) {
	m, err := matrix.NewZeros(rows, cols)
	if err != nil {
		return nil, err
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if err = m.Set(i, j, v[i*cols+j]); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}

// ModelID returns the canonical content identity of the categorical model.
// Its SHA-256 digest includes the algorithm revision and normalized parameter values.
// Implementation:
//   - Stage 1: Reject an absent receiver without dereferencing it.
//   - Stage 2: Read the identity established by the constructor.
//   - Stage 3: Return an immutable string, without exposing model parameters.
//
// Behavior highlights:
//   - Identical canonical categorical models have identical identifiers.
//
// Inputs:
//   - m: immutable categorical model; no concurrent mutation is supported.
//
// Returns:
//   - string: model identity; empty for a nil receiver.
//
// Errors:
//   - None; use the empty string to recognize an absent receiver.
//
// Determinism:
//   - The constructor hashes parameters in fixed row-major order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - An identity is not a durable state serialization or an authorization token.
//
// AI-Hints:
//   - Do not compare hidden-state labels across models solely by matching their IDs.
func (m *CategoricalModel) ModelID() string {
	if m == nil {
		return ""
	}
	return m.chain.id
}

// ModelID returns the content identity of the diagonal-Gaussian model.
// The identity includes all normalized chain parameters, means, and variances.
// Implementation:
//   - Stage 1: Check for an absent receiver.
//   - Stage 2: Read the constructor-established canonical digest.
//   - Stage 3: Return the immutable string without exposing storage.
//
// Behavior highlights:
//   - Gaussian and categorical models cannot share an identity by construction.
//
// Inputs:
//   - m: immutable Gaussian model.
//
// Returns:
//   - string: model identity; empty for a nil receiver.
//
// Errors:
//   - None.
//
// Determinism:
//   - Fixed row-major parameter ordering and IEEE-754 bit encoding.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A matching identity authenticates parameter equality, not data provenance.
//
// AI-Hints:
//   - Retain this value alongside FilterState to detect accidental model mixing.
func (m *GaussianModel) ModelID() string {
	if m == nil {
		return ""
	}
	return m.chain.id
}

// FilterDefault computes categorical causal evidence without allocating posterior history.
// It delegates to Filter with the documented zero-value FilterOptions policy.
// Implementation:
//   - Stage 1: Forward context and observations without changing validation order.
//   - Stage 2: Delegate to the unique categorical forward kernel.
//   - Stage 3: Return the final detached state and marginal log likelihood.
//
// Behavior highlights:
//   - Posterior is nil; Last is available for resuming a stream.
//
// Inputs:
//   - ctx: non-nil context; x: nonempty categorical symbol sequence.
//
// Returns:
//   - *FilterResult: detached final state; nil on failure.
//
// Errors:
//   - Same errors and precedence as Filter.
//
// Determinism:
//   - Observation order and model state indices are fixed.
//
// Complexity:
//   - Time O(T*K^2); workspace O(K).
//
// Notes:
//   - Never conditions on future observations.
//
// AI-Hints:
//   - Prefer FilterStep for individual incoming events; this wrapper processes a whole batch.
func (m *CategoricalModel) FilterDefault(ctx context.Context, x []uint32) (*FilterResult, error) {
	return m.Filter(ctx, x, FilterOptions{})
}

// FilterDefault computes Gaussian causal evidence without allocating posterior history.
// It preserves the scalar per-row streaming behavior of the canonical Filter method.
// Implementation:
//   - Stage 1: Forward context and matrix without copying the observation sequence.
//   - Stage 2: Delegate to Filter with the zero-value options.
//   - Stage 3: Return a detached last state and likelihood.
//
// Behavior highlights:
//   - The result does not contain a T-by-K matrix.
//
// Inputs:
//   - ctx: non-nil context; x: nonempty T-by-D finite observation matrix.
//
// Returns:
//   - *FilterResult: final causal state or nil on failure.
//
// Errors:
//   - Same sentinel and precedence as Filter.
//
// Determinism:
//   - Rows are evaluated in ascending time order.
//
// Complexity:
//   - Time O(T*(K^2+K*D)); workspace O(K+D).
//
// Notes:
//   - Caller must not concurrently modify the observation matrix.
//
// AI-Hints:
//   - Use returned Last for checkpointing; external state validation happens in FilterStep.
func (m *GaussianModel) FilterDefault(ctx context.Context, x matrix.Matrix) (*FilterResult, error) {
	return m.Filter(ctx, x, FilterOptions{})
}

// FilterWithHistory computes all categorical causal posterior distributions.
// Unlike Smooth, earlier rows never depend on observations appended in the future.
// Implementation:
//   - Stage 1: Preserve the context and original symbols.
//   - Stage 2: Delegate to Filter with ReturnHistory enabled.
//   - Stage 3: Return a detached T-by-K posterior matrix.
//
// Behavior highlights:
//   - Each returned row contains normalized probabilities rather than log values.
//
// Inputs:
//   - ctx: non-nil context; x: ordered categorical observations.
//
// Returns:
//   - *FilterResult: history and final state, or nil on failure.
//
// Errors:
//   - Same sentinel and precedence as Filter.
//
// Determinism:
//   - Identical inputs produce identical rows in the same order.
//
// Complexity:
//   - Time O(T*K^2); space O(T*K) for the returned matrix.
//
// Notes:
//   - History is caller-owned and may be mutated without affecting the model.
//
// AI-Hints:
//   - Do not use Smooth when computing online trading or monitoring decisions.
func (m *CategoricalModel) FilterWithHistory(ctx context.Context, x []uint32) (*FilterResult, error) {
	return m.Filter(ctx, x, FilterOptions{ReturnHistory: true})
}

// FilterWithHistory computes every causal Gaussian posterior and the final evidence.
// It deliberately retains a T-by-K probability history for inspection or plotting.
// Implementation:
//   - Stage 1: Forward the observations without changing their ownership.
//   - Stage 2: Delegate to Filter with history enabled.
//   - Stage 3: Return a detached matrix plus the final state.
//
// Behavior highlights:
//   - Each row depends only on that row and preceding observations.
//
// Inputs:
//   - ctx: non-nil context; x: T-by-D finite Gaussian observations.
//
// Returns:
//   - *FilterResult: detached probability history, or nil on failure.
//
// Errors:
//   - Same sentinel and precedence as Filter.
//
// Determinism:
//   - Fixed temporal order and floating-point evaluation order.
//
// Complexity:
//   - Time O(T*(K^2+K*D)); space O(T*K+D).
//
// Notes:
//   - Use FilterDefault to avoid the history allocation.
//
// AI-Hints:
//   - This method is causal; do not interpret its rows as globally smoothed states.
func (m *GaussianModel) FilterWithHistory(ctx context.Context, x matrix.Matrix) (*FilterResult, error) {
	return m.Filter(ctx, x, FilterOptions{ReturnHistory: true})
}
