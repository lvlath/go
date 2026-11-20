// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix implements cyclic one-sided Jacobi rotations for economy SVD.
// This file contains the single iterative kernel; validation and publication are separate.
package matrix

import (
	"context"
	"math"
)

// svdCompute computes the decomposition using one canonical tall-oriented kernel.
// It consumes a private original-input snapshot and never forms A^T*A or A*A^T.
//
// Implementation:
//   - Stage 1: orient and normalize independently scaled working columns.
//   - Stage 2: orthogonalize column pairs in deterministic full Jacobi sweeps.
//   - Stage 3: sort, complete, canonicalize, and publish an economy result.
//
// Behavior highlights:
//   - Already orthogonal columns require zero completed sweeps.
//   - Nonconvergence publishes a diagnostic, finite result and ErrNotConverged.
//
// Inputs: finite detached input of shape m-by-n; validated finalized options.
// Returns: Detached factorization or last diagnostic state on budget exhaustion.
// Errors: ErrOverflow, ErrNotConverged, or ctx.Err() (which discards partial work).
// Determinism: First pair (0,1), then lexicographic (p,q), stable output ordering.
// Complexity: Time O((sweeps+1)*max(m,n)*k*k), Space O((m+n)*k), k=min(m,n).
// Notes: Each invocation owns all workspace; parallel reads of a stable input are safe.
// AI-Hints: Do not replace cyclic Jacobi with a normal-equations eigensolver.
func svdCompute(ctx context.Context, input []float64, m, n int, options SVDOptions) (*SVDResult, error) {
	working := svdNewWorkspace(input, m, n)
	iterations, converged, err := svdJacobiSweeps(ctx, &working, options)
	if err != nil {
		return nil, err
	}
	result, err := svdBuildResult(ctx, &working, m, n, iterations, converged)
	if err != nil {
		return nil, err
	}
	if !converged {
		return result, ErrNotConverged
	}
	return result, nil
}

// svdJacobiSweeps checks the *updated* maximum correlation after every full sweep.
// The initial check makes an already orthogonal matrix an exact zero-sweep case.
//
// Implementation:
//   - Stage 1: test every current column pair for initial convergence.
//   - Stage 2: apply a complete lexicographically ordered rotation sweep.
//   - Stage 3: remeasure post-sweep correlations and stop on convergence or stagnation.
//
// Inputs: active ctx, private working state, positive normalized options.
// Returns: completed sweep count, convergence flag, and cancellation error.
// Errors: ctx.Err(); any canceled operation returns nil at the public boundary.
// Determinism: p ascending, q>p ascending; no data-dependent pair permutation.
// Complexity: O(sweeps*rows*cols*cols) time, O(1) additional workspace.
// AI-Hints: Counting skipped rotations is not a valid convergence test:
// a late rotation can re-correlate a pair visited earlier in the same sweep.
func svdJacobiSweeps(ctx context.Context, state *svdWorkspace, options SVDOptions) (int, bool, error) {
	maximum, err := svdMaximumCorrelation(ctx, state)
	if err != nil {
		return 0, false, err
	}
	if maximum <= options.Tolerance {
		return 0, true, nil
	}

	for sweep := 1; sweep <= options.MaxIterations; sweep++ {
		rotated, sweepErr := svdJacobiSweep(ctx, state, options.Tolerance)
		if sweepErr != nil {
			return 0, false, sweepErr
		}
		maximum, sweepErr = svdMaximumCorrelation(ctx, state)
		if sweepErr != nil {
			return 0, false, sweepErr
		}
		if maximum <= options.Tolerance {
			return sweep, true, nil
		}
		if !rotated {
			// A rotation too small for binary64 cannot silently count as success.
			return sweep, false, nil
		}
	}
	return options.MaxIterations, false, nil
}

// svdJacobiSweep visits all unordered working-column pairs in fixed order.
// Each applied rotation updates both the data columns and the same two columns of V.
//
// Implementation:
//   - Stage 1: check cancellation at each pair boundary.
//   - Stage 2: measure normalized correlation and compute a bounded Jacobi tangent.
//   - Stage 3: rotate only when correlation exceeds the caller's tolerance.
//
// Inputs: active ctx, mutable private state, correlation tolerance (0,1).
// Returns: whether at least one representable rotation was applied.
// Errors: ctx.Err() only.
// Determinism: First dimension p ascending, second q=p+1..cols-1 ascending.
// Complexity: Time O(rows*cols*cols), Space O(1).
// AI-Hints: Never use the raw dot product to decide a rotation at huge input scales.
func svdJacobiSweep(ctx context.Context, state *svdWorkspace, tolerance float64) (bool, error) {
	rotated := false
	for p := 0; p < state.cols; p++ {
		for q := p + 1; q < state.cols; q++ {
			if err := ctx.Err(); err != nil {
				return false, err
			}
			correlation := svdNormalizedCorrelation(state, p, q)
			if math.Abs(correlation) <= tolerance {
				continue
			}
			tangent := svdRotationTangent(state, p, q, correlation)
			if tangent == 0 {
				// A nonzero residual correlation must survive into the final check.
				continue
			}
			svdRotateColumns(state, p, q, tangent)
			rotated = true
		}
	}
	return rotated, nil
}

// svdMaximumCorrelation reevaluates the global off-diagonal convergence metric.
// It deliberately uses the current state rather than pair decisions made earlier.
//
// Inputs: active context and the current independently scaled columns.
// Returns: Maximum absolute normalized correlation, or a context error.
// Errors: ctx.Err() at each pair boundary.
// Determinism: Lexicographic pair scan; ties retain the first maximum.
// Complexity: Time O(rows*cols*cols), Space O(1).
// AI-Hints: This is a transient pairwise dot calculation, not an A^T*A matrix.
func svdMaximumCorrelation(ctx context.Context, state *svdWorkspace) (float64, error) {
	maximum := 0.0
	for p := 0; p < state.cols; p++ {
		for q := p + 1; q < state.cols; q++ {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
			maximum = math.Max(maximum, math.Abs(svdNormalizedCorrelation(state, p, q)))
		}
	}
	return maximum, nil
}

// svdNormalizedCorrelation computes <Bp,Bq>/(||Bp||*||Bq||) without physical
// magnitudes, using Neumaier compensation for opposing products.
//
// Implementation:
//   - Stage 1: return zero for any exact-zero column.
//   - Stage 2: form products of separately normalized coordinate values.
//   - Stage 3: add the residual compensation and clamp harmless roundoff to [-1,1].
//
// Inputs: private workspace, valid distinct pair indices.
// Returns: Finite normalized correlation in [-1,1].
// Errors: None after finite input validation and bounded rotations.
// Determinism: All products and compensated additions follow increasing row order.
// Complexity: Time O(rows), Space O(1).
// AI-Hints: Do not sum physical Bp*Bq; that both overflows and erases tiny modes.
func svdNormalizedCorrelation(state *svdWorkspace, p, q int) float64 {
	normP, normQ := state.norms[p], state.norms[q]
	if normP == 0 || normQ == 0 {
		return 0
	}
	sum, compensation := 0.0, 0.0
	for row := 0; row < state.rows; row++ {
		base := row * state.cols
		term := (state.data[base+p] / normP) * (state.data[base+q] / normQ)
		updated := sum + term
		if math.Abs(sum) >= math.Abs(term) {
			compensation += (sum - updated) + term
		} else {
			compensation += (term - updated) + sum
		}
		sum = updated
	}
	correlation := sum + compensation
	if correlation > 1 {
		return 1
	}
	if correlation < -1 {
		return -1
	}
	return correlation
}

// svdRotationTangent computes a stable Jacobi tangent without squared physical norms.
// With r=min(||Bp||,||Bq||)/max(...) and rho the signed correlation, the
// magnitude is 2*r*|rho| / (1-r*r + hypot(1-r*r,2*r*|rho|)).
//
// Implementation:
//   - Stage 1: compare column norms via signed integer exponents.
//   - Stage 2: use the cancellation-aware small-r representation of Jacobi's formula.
//   - Stage 3: select the direction from the larger column and correlation sign.
//
// Inputs: nonzero columns and a nonzero normalized correlation.
// Returns: Tangent in [-1,1]; zero means an unrepresentable tiny rotation.
// Errors: None; stagnation is classified by the completed-sweep convergence check.
// Determinism: Exact magnitude equality selects positive tangent one.
// Complexity: O(1) arithmetic and auxiliary space.
// AI-Hints: Computing (beta-alpha)/(2*gamma) directly can overflow to Inf
// even when the correct rotation is tiny and the SVD is representable.
func svdRotationTangent(state *svdWorkspace, p, q int, correlation float64) float64 {
	ratio, firstLarger, equal := svdScaledMagnitudeRatio(
		state.norms[p], state.exponents[p], state.norms[q], state.exponents[q],
	)
	if equal {
		return 1
	}
	product := 2 * ratio * math.Abs(correlation)
	if product == 0 {
		return 0
	}
	difference := math.FMA(-ratio, ratio, 1)
	tangent := product / (difference + math.Hypot(difference, product))
	if firstLarger {
		return -math.Copysign(tangent, correlation)
	}
	return math.Copysign(tangent, correlation)
}

// svdRotateColumns applies an orthogonal pair rotation in independently scaled
// coordinates, without materializing the physical coefficient magnitudes.
//
// Implementation:
//   - Stage 1: determine separate output exponents from both contributing terms.
//   - Stage 2: combine both columns using a fused multiply-add per output coordinate.
//   - Stage 3: renormalize the resulting columns and apply the same rotation to V.
//
// Inputs: mutable workspace, distinct pair p<q, finite nonzero tangent.
// Returns: No value; all modifications remain private until final publication.
// Errors: None for valid working columns and bounded rotation parameters.
// Determinism: Increasing row order, then increasing right-basis coordinate order.
// Complexity: Time O(rows+cols), Space O(1).
// Notes: Unrepresentably small relative terms can still be rounded away.
// AI-Hints: A single common pair scale may erase the smaller entire column;
// each *output* column needs its own exponent based on its contributing terms.
func svdRotateColumns(state *svdWorkspace, p, q int, tangent float64) {
	cosine := 1 / math.Hypot(1, tangent)
	sine := cosine * tangent
	oldExpP, oldExpQ := state.exponents[p], state.exponents[q]
	_, expCos := math.Frexp(cosine)
	_, expSin := math.Frexp(math.Abs(sine))
	newExpP := max(oldExpP+expCos, oldExpQ+expSin)
	newExpQ := max(oldExpP+expSin, oldExpQ+expCos)

	// Each coefficient is scaled toward its output's largest contributing term.
	// The relative terms are bounded and FMA limits cancellation during mixing.
	pp := math.Ldexp(cosine, oldExpP-newExpP)
	pq := math.Ldexp(sine, oldExpQ-newExpP)
	qp := math.Ldexp(sine, oldExpP-newExpQ)
	qq := math.Ldexp(cosine, oldExpQ-newExpQ)
	for row := 0; row < state.rows; row++ {
		index := row * state.cols
		oldP, oldQ := state.data[index+p], state.data[index+q]
		state.data[index+p] = math.FMA(-pq, oldQ, pp*oldP)
		state.data[index+q] = math.FMA(qp, oldP, qq*oldQ)
	}

	// Normalize only after both outputs use the original pair; the columns do
	// not alias the input snapshot and neither normalization mutates the other.
	state.normalizeColumn(p, newExpP)
	state.normalizeColumn(q, newExpQ)
	for row := 0; row < state.cols; row++ {
		index := row * state.cols
		oldP, oldQ := state.right[index+p], state.right[index+q]
		state.right[index+p] = math.FMA(-sine, oldQ, cosine*oldP)
		state.right[index+q] = math.FMA(sine, oldP, cosine*oldQ)
	}
}
