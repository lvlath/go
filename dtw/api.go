// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw exposes public Dynamic Time Warping facades.
// Facades assemble policy, validate adapter-level inputs, and delegate to canonical kernels.
package dtw

import (
	"context"

	"github.com/lvlath/go/matrix"
)

// Align computes Dynamic Time Warping for two scalar sequences.
//
// What:
//   - Align is the canonical scalar DTW facade.
//   - It returns a Result artifact carrying distance, reachability, path state,
//     and optional matrix artifacts.
//
// Implementation:
//   - Stage 1: assemble and finalize functional options.
//   - Stage 2: validate scalar inputs and numeric policy inside compute.
//   - Stage 3: run the classified row-major recurrence, retaining the chosen
//     predecessor during forward DP when path tracking is enabled.
//   - Stage 4: distinguish an unreachable endpoint, an unrepresentable finite
//     optimum, and a finite exact result before reconstructing a path.
//   - Stage 5: publish detached result artifacts only under their requested policy.
//
// Behavior highlights:
//   - Window == -1 disables the Sakoe-Chiba band.
//   - Window >= 0 enforces |i-j| <= Window.
//   - Window == 0 is strict diagonal alignment.
//   - A non-path result has Distance=+Inf and Reachable=false only when the
//     endpoint is structurally unreachable; final numeric overflow is an error.
//   - Overflowed discarded branches do not invalidate a finite optimal path.
//   - WithReturnPath(true) automatically uses FullMatrix storage.
//   - WithReturnLocalCost(true) evaluates every published cell, including off-band
//     cells, exactly once; callbacks are pure, deterministic cost functions.
//
// Inputs:
//   - a: first scalar sequence; must be non-nil and non-empty.
//   - b: second scalar sequence; must be non-nil and non-empty.
//   - opts: explicit policy switches; nil options are rejected.
//
// Returns:
//   - *Result: canonical detached result with finite Distance on success, or
//     an unreachable result when the window blocks all endpoint-valid paths.
//   - An error from invalid input or an unrepresentable optimum publishes nil.
//
// Errors:
//   - ErrNilOption from option assembly.
//   - ErrInvalidWindow / ErrInvalidPenalty joined with ErrBadInput.
//   - ErrNilInput / ErrEmptyInput from sequence validation.
//   - ErrNaNInf from finite-input or local-cost validation.
//   - ErrNegativeCost when a scalar cost callback returns a negative value.
//   - ErrOverflow joined with ErrNaNInf when the selected optimal accumulated
//     cost cannot be represented as finite float64.
//   - User CostFunc failures propagate without changing their sentinel identity.
//   - ErrNoPath when path tracking is requested but no admissible path exists.
//
// Determinism:
//   - DP traversal is row-major: i ascending, then j ascending.
//   - Strictly smaller candidates replace predecessors; equal values keep
//     diagonal, then vertical/up, then horizontal/left priority.
//   - Backtracking follows recorded codes and never subtracts rounded costs.
//
// Complexity:
//   - Time O(n*m), where n=len(a), m=len(b), even with a narrow window.
//   - Space O(m) for distance-only rolling rows without local artifacts.
//   - Space O(n*m) when path, accumulated, or full local-cost output is requested.
//
// Notes:
//   - Align does not mutate input slices; all returned buffers are detached.
//   - Concurrent mutation of inputs is unsupported. State-dependent CostFunc
//     callbacks are not a valid mechanism for controlling evaluation order.
//   - An accumulated +Inf artifact cell alone cannot identify whether an
//     intermediate state is structurally absent or numerically overflowed.
//
// AI-Hints:
//   - Do not reintroduce pointer-based options into the canonical facade.
//   - Do not treat nil Result and unreachable Result as the same state.
//   - Never reconstruct a predecessor using D(i,j)-localCost(i,j) after rounding.
func Align(a, b []float64, opts ...Option) (*Result, error) {
	cfg, err := applyOptions(opts...)
	if err != nil {
		return nil, err
	}

	return compute(a, b, cfg)
}

// AlignCostMatrix computes DTW over a precomputed local-cost matrix.
//
// What:
//   - AlignCostMatrix treats local.Rows() as the number of steps in sequence A.
//   - It treats local.Cols() as the number of steps in sequence B.
//   - local.At(i,j) is the finite non-negative local cost for aligning A[i] with B[j].
//
// Implementation:
//   - Stage 1: assemble and finalize DTW options.
//   - Stage 2: validate the local-cost matrix shape and numeric contract.
//   - Stage 3: materialize local costs into a row-major []float64 for hot-loop purity.
//   - Stage 4: delegate to computeFromFlatLocal.
//   - Stage 5: optionally publish a complete detached local-cost Dense
//     artifact, including cells excluded from the admissible DTW band.
//
// Behavior highlights:
//   - This facade is the adapter for domain-specific or externally computed costs.
//   - SlopePenalty, Window, ReturnPath, and matrix artifacts work exactly as in Align.
//   - CostFunc is ignored because local costs are already supplied.
//   - The entire matrix is validated even when Window excludes some cells.
//   - Forward predecessors use strict diagonal/up/left ordering.
//
// Inputs:
//   - local: matrix.Matrix with rows>0, cols>0, finite non-negative values.
//   - opts: explicit DTW policy switches.
//
// Returns:
//   - *Result: canonical detached result artifact.
//
// Errors:
//   - ErrNilOption from option assembly.
//   - ErrBadInput joined with matrix validation/access errors.
//   - ErrEmptyInput when local has zero rows or zero columns.
//   - ErrNaNInf when any local cost is NaN or ±Inf.
//   - ErrNegativeCost when any local cost is negative.
//   - ErrOverflow joined with ErrNaNInf for an unrepresentable optimal sum.
//   - ErrNoPath when path tracking is requested but no admissible path exists.
//
// Determinism:
//   - Matrix values are read in row-major order.
//   - DP traversal is row-major; strict ties prefer diagonal, then up,
//     then left, and path reconstruction reads recorded predecessors.
//
// Complexity:
//   - Validation/materialization Time O(n*m), Space O(n*m).
//   - DP Time O(n*m), Space O(m) or O(n*m) depending on options.
//
// Notes:
//   - local is never mutated.
//   - Result.LocalCost is a complete detached Dense copy, not a window-masked
//     matrix, only when WithReturnLocalCost(true) is set.
//
// AI-Hints:
//   - Do not call CostFunc in this facade; the matrix already is the local-cost source.
//   - Do not keep aliases to caller-owned matrix storage in Result.
func AlignCostMatrix(local matrix.Matrix, opts ...Option) (*Result, error) {
	cfg, err := applyOptions(opts...)
	if err != nil {
		return nil, err
	}

	costs, rows, cols, err := materializeLocalCostMatrix(local)
	if err != nil {
		return nil, err
	}

	res, err := computeFromFlatLocal(costs, rows, cols, cfg)
	if res != nil && cfg.returnLocalCost {
		localCopy, cloneErr := denseCopyOfMatrix(local)
		if cloneErr != nil {
			return res, cloneErr
		}
		res.LocalCost = localCopy
	}

	return res, err
}

// AlignMatrix computes multivariate DTW using squared L2 row-to-row local costs.
//
// What:
//   - AlignMatrix treats each row of x and y as one time step.
//   - Columns are feature dimensions and must match.
//   - The local cost is squared Euclidean distance between rows.
//
// Implementation:
//   - Stage 1: assemble and finalize DTW options.
//   - Stage 2: validate finite rows and compute squared coordinate
//     differences through a scaled sum-of-squares accumulator.
//   - Stage 3: delegate to AlignCostMatrix-compatible flat local-cost DP.
//   - Stage 4: publish the local-cost artifact when requested.
//
// Behavior highlights:
//   - This is the canonical matrix-backed multivariate DTW facade.
//   - Every row-pair cost is a non-negative scaled sum of squared differences.
//   - It does not use the cancellation-prone norm/dot-product identity or
//     clamp negative intermediates as a substitute for stable arithmetic.
//   - True local squared-distance overflow is classified independently from
//     overflow in discarded accumulated transitions.
//
// Inputs:
//   - x: matrix where rows are time steps and columns are features.
//   - y: matrix where rows are time steps and columns are features.
//   - opts: explicit DTW policy switches.
//
// Returns:
//   - *Result: canonical detached result artifact.
//
// Errors:
//   - ErrBadInput joined with matrix errors for nil, shape, access, or operation failures.
//   - ErrNaNInf joined with matrix finite-validation failures.
//   - matrix.ErrDimensionMismatch is preserved when feature dimensions differ.
//   - ErrOverflow joined with ErrNaNInf for unrepresentable local squared L2
//     or a terminal optimal accumulated distance.
//   - ErrNoPath when path tracking is requested but no admissible path exists.
//
// Determinism:
//   - Matrix local costs are generated in row-major order.
//   - DP traversal and backtracking use the same laws as Align.
//
// Complexity:
//   - Local-cost construction is O(nx*ny*d) conceptually, where d is feature count.
//   - DP is O(nx*ny).
//   - Space O(nx*ny) for the complete local-cost matrix and flat copy;
//     optional accumulated/path tracking adds O(nx*ny).
//
// Notes:
//   - AlignMatrix uses squared L2 costs, not Euclidean L2 with sqrt.
//   - Every local pair is evaluated and checked, regardless of the DTW band.
//   - Use matrix preprocessing functions before calling AlignMatrix when normalization is needed.
//
// AI-Hints:
//   - Do not replace squared differences with x*x+y*y-2*x*y: cancellation
//     can destroy finite, small distances between large coordinates.
//   - Do not silently normalize inputs here; preprocessing belongs to callers.
func AlignMatrix(x, y matrix.Matrix, opts ...Option) (*Result, error) {
	cfg, err := applyOptions(opts...)
	if err != nil {
		return nil, err
	}

	local, err := localSquaredL2Matrix(x, y)
	if err != nil {
		return nil, err
	}

	costs, rows, cols, err := materializeLocalCostMatrix(local)
	if err != nil {
		return nil, err
	}

	res, err := computeFromFlatLocal(costs, rows, cols, cfg)
	if res != nil && cfg.returnLocalCost {
		res.LocalCost = local
	}

	return res, err
}

// AlignBounded computes canonical scalar DTW and classifies its optimal outcome
// against an inclusive non-negative finite limit or unbounded +Inf limit.
// It never disguises an unrepresentable final optimum as a cutoff or no-path result.
//
// Implementation:
//   - Stage 1: validate context, option assembly, scalar inputs, and limit.
//   - Stage 2: execute the shared classified recurrence with optional path tracking.
//   - Stage 3: publish Exact, AboveLimit, or Unreachable with distinct numeric fields.
//
// Behavior highlights:
//   - Exact is inclusive at distance==limit and retains the canonical path if requested.
//   - AboveLimit has NaN Distance and a finite LowerBound strictly greater
//     than a finite limit; it is the exact computed score, not a guessed value.
//   - Unreachable has two +Inf fields, independent of the legacy ErrNoPath rule.
//
// Inputs:
//   - ctx: non-nil context; cancellation is checked before and after the recurrence.
//   - a, b: borrowed read-only scalar series; concurrent mutation is unsupported.
//   - limit: accumulated-cost units; finite >=0 or +Inf.
//   - opts: ordinary DTW options; requested matrix artifacts are unsupported here.
//
// Returns:
//   - *BoundedResult: owned path/metadata on success; nil on cancellation/numeric failure.
//
// Errors:
//   - ErrInvalidOption: nil context, bad limit, or unavailable matrix artifacts.
//   - Legacy option/input errors preserve their established errors.Is identity.
//   - ErrOverflow joined with ErrNaNInf: unrepresentable terminal optimum.
//   - context.Canceled/DeadlineExceeded: canceled invocation, without a result.
//
// Determinism:
//   - The same row-major recurrence and strict diagonal/up/left policy as Align.
//
// Complexity:
//   - Time O(n*m); Space O(m) without a path and O(n*m) with path tracking.
//
// Notes:
//   - This exact-first bounded API does not promise early abandonment.
//   - It does not change Result or the legacy DTW signature.
//
// AI-Hints:
//   - Do not compare BoundAboveLimit.Distance as an exact score: it is NaN.
//   - Do not convert final overflow into an above-limit status.
func AlignBounded(ctx context.Context, a, b []float64, limit float64, opts ...Option) (*BoundedResult, error) {
	return alignBounded(ctx, a, b, limit, opts...)
}

// Search returns the globally exact top-K of fixed-length squared-scalar DTW windows.
// It may skip candidates only when a machine-certified Keogh lower bound proves that
// their finite score is strictly worse than the currently retained kth score.
//
// Implementation:
//   - Stage 1: validate context, options, data domain, and excluded start ranges.
//   - Stage 2: prepare optional normalized query and radius-derived envelope.
//   - Stage 3: score or certify admitted windows using independent worker buffers.
//   - Stage 4: sort completed hits by (distance,start,end) and publish status/counters.
//
// Behavior highlights:
//   - Radius=-1 is unconstrained, Radius=0 is diagonal-only; TopK/Stride/Workers>=1.
//   - ExcludedStarts contains half-open *start index* ranges, not time overlaps.
//   - Only Complete=true certifies global exact top-K; incomplete Hits rank
//     completed candidates only and must not be used as a global ranking.
//   - Optional pruning is strictly certified and preserves exactness
//     without heuristic epsilon comparisons or approximate shortlist publication.
//
// Inputs:
//   - ctx: non-nil cancellation context; query: nonempty finite borrowed template.
//   - data: finite borrowed series; a shorter series returns a complete empty result.
//   - opts: explicit SearchOptions; zero-value is invalid, budgets of zero are unlimited.
//
// Returns:
//   - *SearchResult: detached sorted hits and classified completion counters.
//   - Partial result and error after an interrupted/candidate-failed computation.
//
// Errors:
//   - ErrInvalidOption/ErrNilInput/ErrEmptyInput/ErrShape: validation failures.
//   - ErrNonFinite joined with ErrNaNInf: invalid input samples.
//   - ErrZeroVariance: constant query or a candidate under SearchZPopulation.
//   - ErrBudget or context cancellation: incomplete resource-governed
//     computation with partial SearchResult after candidate execution begins.
//   - ErrOverflow joined with ErrNaNInf: unrepresentable local or optimal score.
//
// Determinism:
//   - Candidate starts follow stride order; completed Hits sort by score,start,end.
//   - Complete Hits are independent of scheduling; pruning counters may differ.
//
// Complexity:
//   - Worst-case time O(C*L*b + C log K), b=min(L,2r+1) or L when unconstrained.
//   - Space O(R+K+W*b+W*L+L) in z-mode; raw mode omits W*L.
//     R is exclusion-range storage and W is the actual worker count.
//
// Notes:
//   - This is NOT open-boundary subsequence DTW, nor an approximate shortlist.
//   - SearchZPopulation normalizes query and each candidate separately.
//   - If an optional certificate exceeds the byte budget, exact scoring continues
//     without pruning; mandatory workspace exhaustion remains ErrBudget.
//
// AI-Hints:
//   - Never treat partial Hits as a certified global result.
//   - Never prune on lowerBound==cutoff or on an unverified rounded diagnostic bound.
func Search(ctx context.Context, query, data []float64, opts SearchOptions) (*SearchResult, error) {
	return searchFixedWindows(ctx, query, data, opts)
}

// LBKeoghSquared computes the squared deviation from caller-supplied finite envelopes.
// This public diagnostic reports a rounded float64 bound, not by itself a machine
// certificate for excluding an exact DTW candidate at a near-equal cutoff.
//
// Implementation:
//   - Stage 1: validate non-nil same-length slices and finite ordered bounds.
//   - Stage 2: accumulate outside-envelope squared deviations in index order.
//   - Stage 3: return a finite checked result or a classified numeric failure.
//
// Behavior highlights:
//   - Inside-envelope samples contribute zero; radius validity is caller-owned.
//
// Inputs:
//   - query: finite scalar samples; lower/upper: finite envelopes of equal length.
//
// Returns:
//   - float64: rounded nonnegative squared deviation; input slices are not mutated.
//
// Errors:
//   - ErrNilInput/ErrEmptyInput/ErrShape for invalid vector/interval dimensions.
//   - ErrNonFinite joined with ErrNaNInf for nonfinite samples.
//   - ErrOverflow joined with ErrNaNInf for unrepresentable squared accumulation.
//
// Determinism:
//   - Increasing-index computation using checked nonnegative float64 addition.
//
// Complexity:
//   - Time O(L), Space O(1).
//
// Notes:
//   - Search constructs its own radius-derived envelope and verifies a stronger
//     finite-domain/diagonal machine certificate before pruning.
//
// AI-Hints:
//   - A rounded public diagnostic is not a safe one-step cutoff comparison.
//   - Do not use it for penalty-modified, absolute-cost or unequal-length DTW.
func LBKeoghSquared(query, lower, upper []float64) (float64, error) {
	return lbKeoghSquared(query, lower, upper)
}

// DTW computes DTW distance through the legacy pointer-options API.
//
// What:
//   - DTW preserves backward compatibility with the previous public signature.
//   - New code should prefer Align and the Result artifact.
//
// Implementation:
//   - Stage 1: convert legacy *Options into finalized internal options.
//   - Stage 2: delegate to the same scalar kernel used by Align.
//   - Stage 3: project Result into the legacy typed-value return shape.
//
// Behavior highlights:
//   - nil legacy options mean DefaultOptions.
//   - ReturnPath still requires MemoryMode == FullMatrix in the legacy API.
//   - The wrapper does not introduce alternative mathematics.
//
// Inputs:
//   - a: first scalar sequence.
//   - b: second scalar sequence.
//   - legacy: legacy options pointer; nil means defaults.
//
// Returns:
//   - dist: Result.Distance.
//   - path: detached path only when path tracking is enabled.
//   - error: sentinel-classified failure.
//
// Errors:
//   - ErrPathNeedsMatrix for legacy ReturnPath without FullMatrix.
//   - ErrNilInput / ErrEmptyInput / ErrNaNInf from the shared kernel.
//   - ErrNoPath may be returned with dist=+Inf when path was requested.
//   - ErrOverflow joined with ErrNaNInf returns the legacy zero distance and
//     nil path with error; overflow is never a successful no-path result.
//
// Determinism:
//   - Same as Align because this wrapper delegates to the same kernel.
//
// Complexity:
//   - Same as Align for the finalized storage mode.
//
// Notes:
//   - Deprecated: use Align with WithXxx options and Result helper methods.
//   - Compatibility requires nil legacy options to remain accepted as defaults.
//
// AI-Hints:
//   - Do not add special-case math here; this wrapper must stay a projection of Align.
func DTW(a, b []float64, legacy *Options) (dist float64, path []Coord, err error) {
	cfg, err := optionsFromLegacy(legacy)
	if err != nil {
		return 0, nil, err
	}

	res, err := compute(a, b, cfg)
	if res == nil {
		return 0, nil, err
	}

	if res.PathTracked && res.Path != nil {
		path = make([]Coord, len(res.Path))
		copy(path, res.Path)
	}

	return res.Distance, path, err
}
