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
// Its mean and variance arrays hold finalized old-model-weighted central estimates.
//
// Implementation:
//   - Stage 1: Create independent prior, transition, occupancy and K-by-D moment arrays.
//   - Stage 2: Sum gamma-weighted scaled deviations and within-sequence xi.
//   - Stage 3: Recompute gamma for a second central-variance pass around new means.
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
//   - The expectation stage finalizes physical-unit means and variances before
//     the M-step applies the caller-selected VarianceFloor.
//
// AI-Hints:
//   - Do not use the public rounded means to infer an unbiased variance estimator.
type gaussianCounts struct {
	prior     []float64 // K gamma at sequence starts; a new prior draw for every run.
	trans     []float64 // K*K xi sums over adjacent steps within each run only.
	occupancy []float64 // K gamma weights summed over every observation.
	means     []float64 // K*D newly estimated weighted means in physical feature units.
	variances []float64 // K*D population variances in squared feature units, before floor projection.
	ll        float64   // Sum of independent sequence marginal log likelihoods.
}

// addCompensated incorporates a finite term into a compensated floating sum.
// It preserves low-order terms when a long EM batch contains unequal weights.
//
// Implementation:
//   - Stage 1: Form the new high-order sum.
//   - Stage 2: Recover the low-order rounding residual by Neumaier's rule.
//   - Stage 3: Store both parts for finalization after deterministic traversal.
//
// Behavior highlights:
//   - Compensation is applied only to locally scaled, representable terms.
//
// Inputs:
//   - sum, correction: distinct accumulators for the same state/feature cell.
//   - term: finite, scaled observation contribution.
//
// Returns:
//   - The two accumulator pointers are updated in place.
//
// Errors:
//   - None; callers check the final mean and variance for representability.
//
// Determinism:
//   - Evaluation order is exactly the order of supplied observations and states.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - No global state and no mutation of input observations.
//
// AI-Hints:
//   - Compensation does not replace scaling when a raw quadratic sum may overflow.
func addCompensated(sum, correction *float64, term float64) {
	updated := *sum + term
	if math.Abs(*sum) >= math.Abs(term) {
		*correction += (*sum - updated) + term
	} else {
		*correction += (term - updated) + *sum
	}
	*sum = updated
}

// stableScaledDifference computes (x-center)/scale without materializing an overflowing delta.
// It retains the precise direct subtraction whenever that difference is representable.
//
// Implementation:
//   - Stage 1: Try the physical-unit subtraction for best small-offset resolution.
//   - Stage 2: Divide first only when opposing finite values overflow subtraction.
//   - Stage 3: Return the scale-free difference for weighted mean/variance accumulation.
//
// Behavior highlights:
//   - The positive scale bounds x and center, so each scaled component is finite.
//
// Inputs:
//   - x, center: finite feature values; scale: positive max-magnitude feature scale.
//
// Returns:
//   - float64: dimensionless centered difference, preserving its sign.
//
// Errors:
//   - None for the validated input domain.
//
// Determinism:
//   - Branch depends only on IEEE-754 finiteness of the raw difference.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Scale is computed once over every independent training sequence, not per regime.
//
// AI-Hints:
//   - Never square an unscaled 1e154 residual before dividing by occupancy.
func stableScaledDifference(x, center, scale float64) float64 {
	delta := x - center
	if finite(delta) {
		return delta / scale
	}
	return x/scale - center/scale
}

// gaussianExpectation performs a deterministic two-pass weighted central-moment E-step.
// The first pass estimates means from old-model responsibilities; the second
// measures centered population variances around those NEW means without raw moments.
//
// Implementation:
//   - Stage 1: Find per-feature scales and accumulate prior/transition/gamma mass
//     and compensated anchored differences from the fixed old model.
//   - Stage 2: Finalize physical weighted means from anchored scaled differences.
//   - Stage 3: Recompute identical old-model gamma and accumulate normalized squared
//     deviations, publishing only representable variance estimates.
//
// Behavior highlights:
//   - The second pass never updates the model or its posterior weights.
//   - State labels, sequence boundaries, and transition counts retain old behavior.
//
// Inputs:
//   - ctx: active context; m: immutable old Gaussian model.
//   - observations: detached T_s*D matrices; lengths: matching positive lengths.
//
// Returns:
//   - gaussianCounts: prior, transition, occupancy, physical means, variances,
//     and sum of independent log likelihoods.
//
// Errors:
//   - ctx.Err(), ErrOverflow for unrepresentable means/variances, and inference errors.
//
// Determinism:
//   - Both passes scan sequence, time, state, feature in ascending order.
//
// Complexity:
//   - Time O(2*sum_s T_s*(K^2+K*D)); Space O(max_s(T_s*K^2)+K^2+K*D).
//
// Notes:
//   - Recomputing posteriors trades approximately twice the E-step work for
//     memory bounded by the longest sequence, avoiding a global gamma cache.
//
// AI-Hints:
//   - Population variance divides by sum(gamma), never by occupancy-1.
//   - Apply VarianceFloor only after the centered variance has been evaluated.
func gaussianExpectation(ctx context.Context, m *GaussianModel, observations [][]float64, lengths []int) (gaussianCounts, error) {
	k, d := m.chain.k, m.dimension
	out := gaussianCounts{
		prior: make([]float64, k), trans: make([]float64, k*k),
		occupancy: make([]float64, k), means: make([]float64, k*d),
		variances: make([]float64, k*d),
	}
	// Fixed per-feature scales retain a bounded coordinate system even for
	// values near MaxFloat64. A zero-only feature uses scale one.
	scales := make([]float64, d)
	for _, flat := range observations {
		for i, value := range flat {
			feature := i % d
			if magnitude := math.Abs(value); magnitude > scales[feature] {
				scales[feature] = magnitude
			}
		}
	}
	for feature := range scales {
		if scales[feature] == 0 {
			scales[feature] = 1
		}
	}

	anchors := make([]float64, k*d)
	anchored := make([]bool, k*d)
	meanCorrection := make([]float64, k*d)
	// E-step pass one: every sequence starts at Prior; xi is accumulated only
	// within that sequence, while gamma supplies weighted feature means.
	for sequence, flat := range observations {
		length := lengths[sequence]
		_, gamma, xi, ll, err := posteriorTables(ctx, &m.chain, length,
			func(i int, e []float64) error { return m.emissions(flat[i*d:(i+1)*d], e) })
		if err != nil {
			return out, fmt.Errorf("sequence %d: %w", sequence, err)
		}
		out.ll += ll
		for time := 0; time < length; time++ {
			for state := 0; state < k; state++ {
				weight := gamma[time*k+state]
				out.occupancy[state] += weight
				if time == 0 {
					out.prior[state] += weight
				}
				if weight == 0 {
					continue
				}
				for feature := 0; feature < d; feature++ {
					at := state*d + feature
					value := flat[time*d+feature]
					if !anchored[at] {
						anchors[at] = value
						anchored[at] = true
					}
					delta := stableScaledDifference(value, anchors[at], scales[feature])
					addCompensated(&out.means[at], &meanCorrection[at], weight*delta)
				}
			}
		}
		for time := 0; time < length-1; time++ {
			for pair := 0; pair < k*k; pair++ {
				out.trans[pair] += xi[time*k*k+pair]
			}
		}
	}
	if !finite(out.ll) {
		return out, ErrOverflow
	}

	// Convert the normalized anchored mean into physical feature coordinates.
	// The normalized fallback prevents an overflowing offset from obscuring a
	// finite convex weighted mean of finite source observations.
	for state := 0; state < k; state++ {
		if out.occupancy[state] == 0 {
			continue // Dead-state emissions remain unchanged in the M-step.
		}
		for feature := 0; feature < d; feature++ {
			at := state*d + feature
			scale := scales[feature]
			shift := (out.means[at] + meanCorrection[at]) / out.occupancy[state]
			offset := shift * scale
			mean := anchors[at] + offset
			if !finite(offset) || !finite(mean) {
				mean = (anchors[at]/scale + shift) * scale
			}
			if !finite(mean) {
				return out, ErrOverflow
			}
			out.means[at] = mean
		}
	}

	if err := gaussianCentralVariance(ctx, m, observations, lengths, scales, &out); err != nil {
		return out, err
	}

	return out, nil
}

// gaussianCentralVariance accumulates old-model-weighted central second moments.
// It evaluates population variances about the newly estimated means rather than
// subtracting the square of a large raw first moment from a raw second moment.
//
// Implementation:
//   - Stage 1: Reproduce posterior responsibilities under the unchanged old model.
//   - Stage 2: Accumulate compensated squared deviations in dimensionless coordinates.
//   - Stage 3: Divide by existing occupancy and restore representable feature-squared units.
//
// Behavior highlights:
//   - A raw sum of squares may be unrepresentable although its averaged variance is finite.
//   - Every sequence starts independently from the old model's Prior.
//
// Inputs:
//   - ctx: active context; m: unchanged model defining all responsibilities.
//   - observations, lengths: finite detached per-sequence buffers and positive row counts.
//   - scales[D]: positive feature scales calculated across all training sequences.
//   - out: caller-owned first-pass counts containing final means and occupancies;
//     its variance buffer is populated without modifying its other fields.
//
// Returns:
//   - error: nil only if every populated variance is finite and nonnegative.
//
// Errors:
//   - ctx.Err() and inference errors from replay; ErrOverflow if a required
//     population variance cannot be represented in feature-squared float64 units.
//
// Determinism:
//   - Sequence, time, state, feature; state occupancy and means are frozen.
//
// Complexity:
//   - Time O(sum_s T_s*(K^2+K*D)); Space O(max_s T_s*K+K*D),
//     excluding the caller-owned input and first-pass aggregate buffers.
//
// Notes:
//   - The M-step applies VarianceFloor only AFTER this helper finishes.
//   - No observations, prior state, or model parameters are mutated.
//
// AI-Hints:
//   - For a single state, this is population variance, never the n-1 estimator.
//   - Do not cache future-model gamma: the second pass must replay the old model.
func gaussianCentralVariance(ctx context.Context, m *GaussianModel, observations [][]float64, lengths []int, scales []float64, out *gaussianCounts) error {
	k, d := m.chain.k, m.dimension
	// E-step pass two: the OLD model deterministically reproduces gamma.
	// Squared deviations are normalized before summation, avoiding a raw
	// sum-of-squares overflow even when the final population variance fits.
	varianceCorrection := make([]float64, k*d)
	for sequence, flat := range observations {
		length := lengths[sequence]
		work, err := buildPosteriorWorkspace(ctx, &m.chain, length,
			func(i int, e []float64) error { return m.emissions(flat[i*d:(i+1)*d], e) })
		if err != nil {
			return fmt.Errorf("variance sequence %d: %w", sequence, err)
		}
		for time := 0; time < length; time++ {
			for state := 0; state < k; state++ {
				weight := work.gamma[time*k+state]
				if weight == 0 || out.occupancy[state] == 0 {
					continue
				}
				for feature := 0; feature < d; feature++ {
					at := state*d + feature
					delta := stableScaledDifference(flat[time*d+feature], out.means[at], scales[feature])
					term := (weight / out.occupancy[state]) * delta * delta
					addCompensated(&out.variances[at], &varianceCorrection[at], term)
				}
			}
		}
	}
	for state := 0; state < k; state++ {
		if out.occupancy[state] == 0 {
			continue
		}
		for feature := 0; feature < d; feature++ {
			at := state*d + feature
			normalizedVariance := out.variances[at] + varianceCorrection[at]
			if normalizedVariance < 0 || !finite(normalizedVariance) {
				return ErrOverflow
			}
			// Compute a standard deviation first to avoid multiplying an
			// enormous scale by itself before a small variance factor.
			standardDeviation := scales[feature] * math.Sqrt(normalizedVariance)
			variance := standardDeviation * standardDeviation
			if !finite(variance) {
				return ErrOverflow
			}
			out.variances[at] = variance
		}
	}
	return nil
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
//   - Stage 2: Reproduce old-model responsibilities and compute two-pass scaled central moments.
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
//   - Time O(iterations*2*sum(T_s)*(K^2+K*D)); memory O(sum(T_s)*D+max(T_s)*K^2+K*D).
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
				// The E-step already produced two-pass weighted central moments
				// in physical units. Never subtract raw second moments here.
				means[p] = counts.means[p]
				variance := counts.variances[p]
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
		// A one-ULP downward candidate is not published. When this is
		// also within the caller's convergence threshold, retain the last
		// accepted model as a numerically stationary EM solution.
		if objectiveRoundoffPlateau(oldScore, score, opts.Tolerance) {
			out.Diagnostics.Converged = true
			return out, nil
		}
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
//   - A representable raw residual is preferred; an overflowing difference is scaled first.
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
			// Favor direct subtraction when it is representable: it retains small
			// differences between large, nearby numbers in their physical units.
			// If opposing finite operands overflow the raw subtraction, divide
			// them separately by sigma before subtracting their scaled values.
			// The standardized residual may be finite even when x-mean is not.
			standardDeviation := math.Sqrt(v)
			delta := x[d] - m.means[k*m.dimension+d]
			z := 0.0
			if finite(delta) {
				z = delta / standardDeviation
			} else {
				z = x[d]/standardDeviation - m.means[k*m.dimension+d]/standardDeviation
			}
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
