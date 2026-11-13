// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm fits diagonal-Gaussian HMMs using constrained deterministic EM.
// This boundary keeps distinct sequence boundaries and projects estimated variances onto the floor.
package hmm

import (
	"context"
	"fmt"
	"math"

	"github.com/lvlath/go/matrix"
)

// gaussianCounts stores posterior-weighted sufficient statistics for diagonal-Gaussian EM.
// Its means and variances arrays accumulate centered first and second moments around current means.
//
// Implementation:
//   - Stage 1: Create independent prior, transition, occupancy and K-by-D moment arrays.
//   - Stage 2: Sum gamma-weighted centered feature contributions and xi within each sequence.
//   - Stage 3: Use the totals for M-step means and projected diagonal variances.
//
// Behavior highlights:
//   - A dead state has zero occupancy and leaves its previous Gaussian emission parameters unchanged.
//
// Inputs:
//   - prior[K], trans[K*K], occupancy[K], means[K*D], variances[K*D], and ll: E-step fields.
//
// Returns:
//   - One ephemeral EM counts value, never retained by a published model.
//
// Errors:
//   - ErrOverflow from expectation if totals become nonfinite.
//
// Determinism:
//   - Order is sequence, time, state, feature; no worker reduction reordering.
//
// Complexity:
//   - Storage O(K*K+K*D), excluding the per-sequence posterior tables.
//
// Notes:
//   - The moment arrays are not normalized until the constrained M-step.
//
// AI-Hints:
//   - Do not use the public rounded means to infer an unbiased variance estimator.
type gaussianCounts struct {
	prior     []float64 // K gamma at sequence starts; a new prior draw for every run.
	trans     []float64 // K*K xi sums over adjacent steps within each run only.
	occupancy []float64 // K gamma weights summed over every observation.
	means     []float64 // K*D gamma-weighted residual sums around current state means.
	variances []float64 // K*D gamma-weighted squared residual sums around current state means.
	ll        float64   // Sum of independent sequence marginal log likelihoods.
}

// gaussianExpectation performs a diagonal-Gaussian E-step on independently bounded series.
// It weighs each observed feature by gamma and each within-sequence transition by xi.
//
// Implementation:
//   - Stage 1: Allocate per-state transition, occupancy and centered moment accumulators.
//   - Stage 2: Build one posterior table per series and accumulate residuals relative to current model means.
//   - Stage 3: Reject nonfinite aggregate evidence/moments before returning the sufficient statistics.
//
// Behavior highlights:
//   - Temporal boundaries are respected even when two series happen to have consecutive timestamps.
//
// Inputs:
//   - ctx: context; m: immutable model; observations: detached flattened T_s*D matrices; lengths: T_s per series.
//
// Returns:
//   - gaussianCounts: prior counts, transition counts, occupancies and centered first/second moments.
//
// Errors:
//   - ctx.Err(), ErrOverflow, ErrNumerical or sequence-wrapped emission/forward-backward errors.
//
// Determinism:
//   - Fixed sequence/time/state/feature ordering determines floating accumulation.
//
// Complexity:
//   - Time O(sum_s T_s*(K^2+K*D)), Space O(max_s T_s*K^2+K*K+K*D).
//
// Notes:
//   - Extreme offsets may exceed binary64 centered moment representability; no silent rescaling.
//
// AI-Hints:
//   - Unlike a continuous transition density, xi still represents a discrete K-by-K state pair.
func gaussianExpectation(ctx context.Context, m *GaussianModel, observations [][]float64, lengths []int) (gaussianCounts, error) {
	k, d := m.chain.k, m.dimension
	out := gaussianCounts{prior: make([]float64, k), trans: make([]float64, k*k), occupancy: make([]float64, k), means: make([]float64, k*d), variances: make([]float64, k*d)}
	// Weighted centered first and second moments relative to the current means.
	// Independent machine runs are scored separately. Each new run begins at
	// Prior; no boundary transition is included in the accumulated xi table.
	for s, flat := range observations {
		t := lengths[s]
		_, gamma, xi, ll, err := posteriorTables(ctx, &m.chain, t, func(i int, e []float64) error { return m.emissions(flat[i*d:(i+1)*d], e) })
		if err != nil {
			return out, fmt.Errorf("sequence %d: %w", s, err)
		}
		out.ll += ll
		for i := 0; i < t; i++ {
			for j := 0; j < k; j++ {
				w := gamma[i*k+j]
				out.occupancy[j] += w
				if i == 0 {
					out.prior[j] += w
				}
				// Use differences from the *previous* state mean to reduce raw
				// second-moment cancellation; the M-step corrects for mean movement.
				for feature := 0; feature < d; feature++ {
					p := j*d + feature
					delta := flat[i*d+feature] - m.means[p]
					if !finite(delta) {
						return out, ErrOverflow
					}
					out.means[p] += w * delta
					out.variances[p] += w * delta * delta
				}
			}
		}
		for i := 0; i < t-1; i++ {
			for v := 0; v < k*k; v++ {
				out.trans[v] += xi[i*k*k+v]
			}
		}
	}
	if !finite(out.ll) {
		return out, ErrOverflow
	}
	for _, v := range out.means {
		if !finite(v) {
			return out, ErrOverflow
		}
	}
	for _, v := range out.variances {
		if !finite(v) {
			return out, ErrOverflow
		}
	}
	return out, nil
}

// newGaussianFromFlat reconstructs an immutable diagonal-Gaussian candidate from M-step arrays.
// It retains the public NewGaussian validation and canonical ModelID path.
//
// Implementation:
//   - Stage 1: Materialize a detached K-by-K transition matrix.
//   - Stage 2: Materialize detached K-by-D means and variance matrices.
//   - Stage 3: Call NewGaussian to validate positivity, copy parameters and hash the model.
//
// Behavior highlights:
//   - No covariance off-diagonal cells or implicit jitter are invented.
//
// Inputs:
//   - k: states; d: features; prior[K], trans[K*K], means[K*D], variances[K*D]: private buffers.
//
// Returns:
//   - *GaussianModel: complete candidate model or nil with a classified error.
//
// Errors:
//   - Matrix allocation/Set errors or constructor shape, finite and variance errors.
//
// Determinism:
//   - Every matrix is written row-major in a deterministic order.
//
// Complexity:
//   - Time O(K^2+K*D), Space O(K^2+K*D).
//
// Notes:
//   - The caller separately checks the Fit-specific VarianceFloor domain.
//
// AI-Hints:
//   - Avoid publishing a candidate with zero or negative variance to make an EM step seem successful.
func newGaussianFromFlat(k, d int, prior, trans, means, variances []float64) (*GaussianModel, error) {
	t, err := denseFromFlat(k, k, trans)
	if err != nil {
		return nil, err
	}
	mm, err := denseFromFlat(k, d, means)
	if err != nil {
		return nil, err
	}
	vv, err := denseFromFlat(k, d, variances)
	if err != nil {
		return nil, err
	}
	return NewGaussian(GaussianSpec{Common: Common{Prior: prior, Transition: t}, Means: mm, Variances: vv})
}

// FitGaussian estimates a diagonal-Gaussian HMM using deterministic constrained EM.
// Its variance floor is a lower bound in squared feature units, not an added epsilon.
// Implementation:
//   - Stage 1: Validate context/options, initial variance domain, and independent matrices.
//   - Stage 2: Compute responsibilities and centered weighted moments for each state/feature.
//   - Stage 3: Project variance estimates onto the floor and accept monotone objectives.
//
// Behavior highlights:
//   - Zero-occupancy states retain their old Gaussian emissions; no random reseeding occurs.
//
// Inputs:
//   - ctx: required; sequences: independent T_s-by-D finite matrices; initial: required model.
//   - opts: iteration limit, tolerance, prior/transition pseudocount, and VarianceFloor.
//
// Returns:
//   - *GaussianFitResult: final accepted model and penalized diagnostics.
//
// Errors:
//   - Input errors, ErrInvalidOption for a below-floor seed; ErrNotConverged/ErrNumerical may accompany a result.
//
// Determinism:
//   - Fixed sequence/time/state/feature accumulation order.
//
// Complexity:
//   - Time O(iterations*sum(T_s)*(K^2+K*D)); memory O(sum(T_s)*D+max(T_s)*K^2+K*D).
//
// Notes:
//   - Gaussian means/variances are unpenalized; Dirichlet penalty covers only Prior/Transition.
//
// AI-Hints:
//   - Keep feature scaling unchanged across all training sequences and later inference.
func FitGaussian(ctx context.Context, sequences []matrix.Matrix, initial GaussianSpec, opts FitOptions) (*GaussianFitResult, error) {
	opts, err := checkFitInput(ctx, opts)
	if err != nil {
		return nil, err
	}
	model, err := NewGaussian(initial)
	if err != nil {
		return nil, err
	}
	for _, v := range model.variances {
		if v < opts.VarianceFloor {
			return nil, ErrInvalidOption
		}
	}
	if sequences == nil {
		return nil, ErrNilInput
	}
	if len(sequences) == 0 {
		return nil, ErrEmptyInput
	}
	observations := make([][]float64, len(sequences))
	lengths := make([]int, len(sequences))
	for i, x := range sequences {
		lengths[i], observations[i], err = validateObservations(x, model.dimension)
		if err != nil {
			return nil, err
		}
	}
	counts, err := gaussianExpectation(ctx, model, observations, lengths)
	if err != nil {
		return nil, err
	}
	oldScore := counts.ll + simplexPenalty(opts.Pseudocount, model.chain.prior, model.chain.transition)
	out := &GaussianFitResult{Model: model, Diagnostics: Diagnostics{Objective: []float64{oldScore}, Occupancy: counts.occupancy}}
	for iteration := 0; iteration < opts.MaxIterations; iteration++ {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		k, d := model.chain.k, model.dimension
		prior := updateSimplex(counts.prior, model.chain.prior, 1, k, opts.Pseudocount)
		trans := updateSimplex(counts.trans, model.chain.transition, k, k, opts.Pseudocount)
		means := append([]float64(nil), model.means...)
		variances := append([]float64(nil), model.variances...)
		// The state dimension never changes. A state with exactly zero posterior
		// occupancy retains its old Gaussian emissions rather than being reseeded.
		for j := 0; j < k; j++ {
			if counts.occupancy[j] == 0 {
				continue
			}
			for feature := 0; feature < d; feature++ {
				p := j*d + feature
				shift := counts.means[p] / counts.occupancy[j]
				means[p] = model.means[p] + shift
				// E[(x-oldMu)^2] - (newMu-oldMu)^2 is the variance around
				// the new weighted mean. Project onto the explicit feasible floor.
				variance := counts.variances[p]/counts.occupancy[j] - shift*shift
				if !finite(means[p]) || !finite(variance) {
					return out, ErrOverflow
				}
				variances[p] = math.Max(variance, opts.VarianceFloor)
			}
		}
		candidate, err := newGaussianFromFlat(k, d, prior, trans, means, variances)
		if err != nil {
			return out, fmt.Errorf("M-step: %w", err)
		}
		newCounts, err := gaussianExpectation(ctx, candidate, observations, lengths)
		if err != nil {
			return nil, err
		}
		score := newCounts.ll + simplexPenalty(opts.Pseudocount, candidate.chain.prior, candidate.chain.transition)
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

// emissions evaluates independent-coordinate Gaussian log densities in each hidden state.
// It returns log PDF values, not categorical probabilities; a density may exceed one.
//
// Implementation:
//   - Stage 1: Reject invalid feature length or nonfinite feature data.
//   - Stage 2: For each state and feature compute a standardized residual and negative quadratic/log-normalizer.
//   - Stage 3: Check every derived term and total for representable binary64 log density.
//
// Behavior highlights:
//   - The negative quadratic evaluates (0.5*z)*z to avoid avoidable intermediate overflow.
//
// Inputs:
//   - m: immutable Gaussian model; x: D finite features; dst: caller-owned K log-score slots.
//
// Returns:
//   - error: nil with K log densities or a typed input/arithmetic failure.
//
// Errors:
//   - ErrShape, ErrNonFinite, ErrOverflow; no Gaussian observation is treated as impossible support.
//
// Determinism:
//   - State index increases outermost; feature index increases within a state.
//
// Complexity:
//   - Time O(K*D), Space O(1) beyond dst.
//
// Notes:
//   - Diagonal conditional independence is a model assumption; missing values are not inferred.
//
// AI-Hints:
//   - Use consistent feature units; Variances are in squared units, not standard deviations.
func (m *GaussianModel) emissions(x []float64, dst []float64) error {
	if len(x) != m.dimension {
		return ErrShape
	}
	for _, v := range x {
		if !finite(v) {
			return ErrNonFinite
		}
	}
	const log2pi = 1.8378770664093453 // Natural logarithm of 2*pi (Gaussian density normalizer).
	for k := 0; k < m.chain.k; k++ {
		var value float64
		for d := 0; d < m.dimension; d++ {
			v := m.variances[k*m.dimension+d]
			// Divide before squaring to avoid overflow for moderate standardized residuals.
			delta := x[d] - m.means[k*m.dimension+d]
			if !finite(delta) {
				return ErrOverflow
			}
			z := delta / math.Sqrt(v)
			if !finite(z) {
				return ErrOverflow
			}
			// Scale before squaring: z*z may overflow although z^2/2 is representable.
			// The multiplication is intentionally ordered to preserve that domain.
			term := -0.5*(log2pi+math.Log(v)) - (0.5*z)*z
			if !finite(term) {
				return ErrOverflow
			}
			value += term
		}
		if !finite(value) {
			return ErrOverflow
		}
		dst[k] = value
	}
	return nil
}
