// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package acf

// impl_direct.go owns the direct ACF lag scan and immutable result publication.
// Every lag reuses the same normalized deviation vector and zero-lag energy.

import "math"

// direct evaluates every requested lag through compensated scalar pair scans.
// It reuses one globally centered, scaled deviation vector for all S_k and
// therefore never needs to materialize a potentially overflowing raw S_0.
//
// Implementation:
//   - Stage 1: form globally centered or raw normalized deviations and fitted Mean.
//   - Stage 2: compute positive S_0, then scan lag k and all n-k pairs in order.
//   - Stage 3: normalize each sum, apply the explicit estimator, and publish owned slices.
//
// Behavior highlights:
//   - Values[0] is exactly 1, without a floating-point divide.
//   - LagAdjusted is computed after S_k/S_0 to prevent n*S_k intermediate overflow.
//   - Any arithmetic failure returns nil and never publishes partial coefficients.
//
// Inputs:
//   - values: previously validated, finite, nonempty, read-only observations.
//   - maxLag: validated inclusive maximum lag, less than len(values).
//   - opts: validated estimator and centering policy.
//
// Returns:
//   - *Result: complete detached coefficients, counts, mean, and estimator metadata.
//   - error: ErrZeroVariance or ErrOverflow; nil result on either failure.
//
// Determinism:
//   - Positive lags are traversed ascending, and each pair scan is ascending by endpoint.
//
// Complexity:
//   - Time O(n*(maxLag+1)), Space O(n+maxLag), including output arrays.
//
// Notes:
//   - Public callers must pass through Compute to enforce the validation contract.
//
// AI-Hints:
//   - Never recenter subsets or merge different S_0 values across lag computations.
//   - Exact PSD properties refer to real arithmetic, not to rounded binary64 matrices.
func direct(values []float64, maxLag int, opts Options) (*Result, error) {
	// The same deviations and fitted center are used at every lag. Keeping only
	// normalized values prevents squares of 1e154 or larger from overflowing.
	deviations, mean, err := normalizedDeviations(values, opts.NoCenter)
	if err != nil {
		return nil, err
	}

	// A positive zero-lag energy is required to define a normalized correlation.
	// The scale does not change the ratios, so the raw S_0 is never materialized.
	s0 := dotScaled(deviations, 0)
	if s0 == 0 {
		return nil, ErrZeroVariance
	}
	if math.IsInf(s0, 0) || math.IsNaN(s0) {
		return nil, ErrOverflow
	}

	// Allocate one private output artifact only after the domain has been checked.
	// Both slices are owned by the returned Result and cannot alias the input.
	n := len(values)
	result := &Result{
		Values:      make([]float64, maxLag+1),
		PairCounts:  make([]int, maxLag+1),
		Mean:        mean,
		Denominator: opts.Denominator,
	}
	result.Values[0] = 1
	result.PairCounts[0] = n

	// k=0 is already defined exactly. For k>0, n-k is strictly positive because
	// the facade enforces maxLag<n; no special-case missing pairs are introduced.
	for lag := 1; lag <= maxLag; lag++ {
		pairs := n - lag
		result.PairCounts[lag] = pairs

		// Complete the compensated cross-product reduction before dividing by S_0.
		// Dividing first keeps the adjusted multiplication in a bounded numeric regime.
		r := dotScaled(deviations, lag) / s0
		if opts.Denominator == LagAdjusted {
			r *= float64(n) / float64(pairs)
		}

		// An unrepresentable public coefficient is a classified arithmetic failure;
		// never clamp, silently replace, or return the partly populated artifact.
		if math.IsInf(r, 0) || math.IsNaN(r) {
			return nil, ErrOverflow
		}
		result.Values[lag] = r
	}
	return result, nil
}
