// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp implements 3-opt local search for symmetric TSP and restricted ATSP 3-opt*.
//
// The 3-opt kernel performs local search over 3-edge exchanges on a closed tour.
//
// Policies:
//   - First-improvement by default: apply the first strictly improving move.
//   - Best-improvement via Options.BestImprovement: scan the whole neighborhood and pick the best.
//   - Options.ThreeOptMaxMoves bounds accepted 3-opt moves independently from 2-opt.
//
// Neighborhood order:
//   - If Options.ShuffleNeighborhood is true, triples (i,j,k) are scanned in a randomized,
//     constraint-respecting cyclic order using rngFromSeed(opts.Seed). seed==0 means deterministic stream.
//   - If false, a canonical deterministic order is used.
//
// Symmetric vs asymmetric:
//   - Symmetric mode evaluates classic 3-opt over S1=T[i..j-1], S2=T[j..k-1]
//     with tail S3=T[k..n-1] fixed.
//     It evaluates seven non-identity reconnections in
//     {S1,rev(S1)}×{S2,rev(S2)} plus the segment-swap variants.
//     Δ = (a→first(X))+(last(X)→first(Y))+(last(Y)→f)
//     − [(a→b)+(c→d)+(e→f)],
//     where a=T[i−1], b=T[i], c=T[j−1], d=T[j], e=T[k−1], f=T[k].
//     Internal arcs cancel by symmetry.
//   - Asymmetric mode uses restricted 3-opt* without reversals.
//     With fixed tail S3, the orientation-preserving reconnection is the tail swap:
//     out = prefix + S2 + S1 + S3. Δ uses the same three boundary arcs.
//     Candidates that introduce +Inf are rejected.
//
// Contracts:
//   - dist is a complete final solver matrix validated through copyCompleteWeights.
//   - initTour is a closed Hamiltonian cycle.
//   - The kernel returns localSearchResult so facades can preserve timeout metadata.
//
// Complexity:
//   - Symmetric mode: O(iter*n³) candidate checks, O(n) working memory.
//   - Restricted asymmetric mode follows the implemented tail-swap neighborhood and keeps O(n) memory.
//
// AI-Hints:
//   - Do not describe asymmetric mode as full ATSP 3-opt.
//   - Do not reorder symmetricThreeOptMoves without updating deterministic tests.
//   - Do not return nil tour on ErrTimeLimit after a valid current tour exists.
package tsp

import (
	"errors"
	"math"
	"math/rand"
	"time"

	"github.com/lvlath/go/matrix"
)

// segmentID identifies one of the two movable middle segments in a 3-opt cut.
type segmentID uint8

const (
	// segmentS1 is the segment tour[i:j].
	segmentS1 segmentID = iota

	// segmentS2 is the segment tour[j:k].
	segmentS2
)

// orientedSegment selects a segment and orientation for explicit 3-opt assembly.
type orientedSegment struct {
	// id selects S1 or S2 from the current 3-opt cut.
	// It must be segmentS1 or segmentS2.
	id segmentID

	// reverse tells whether this segment is emitted in reverse order.
	// Reverse is valid only for symmetric 3-opt moves.
	reverse bool
}

// threeOptMove names one symmetric 3-opt reconnection case.
// The identity move is intentionally absent from symmetricThreeOptMoves.
type threeOptMove struct {
	// name is a stable diagnostic identifier for tests and table audits.
	// It must not be used as algorithm control flow.
	name string

	// x is the first emitted movable segment in the reconnection.
	// It may be S1 or S2 in either orientation.
	x orientedSegment

	// y is the second emitted movable segment in the reconnection.
	// It must use the other segment exactly once.
	y orientedSegment
}

var symmetricThreeOptMoves = [...]threeOptMove{
	{name: "reverse_s1", x: orientedSegment{id: segmentS1, reverse: true}, y: orientedSegment{id: segmentS2, reverse: false}},
	{name: "reverse_s2", x: orientedSegment{id: segmentS1, reverse: false}, y: orientedSegment{id: segmentS2, reverse: true}},
	{name: "reverse_both", x: orientedSegment{id: segmentS1, reverse: true}, y: orientedSegment{id: segmentS2, reverse: true}},
	{name: "swap_s1_s2", x: orientedSegment{id: segmentS2, reverse: false}, y: orientedSegment{id: segmentS1, reverse: false}},
	{name: "swap_s1_reverse", x: orientedSegment{id: segmentS2, reverse: false}, y: orientedSegment{id: segmentS1, reverse: true}},
	{name: "swap_s2_reverse", x: orientedSegment{id: segmentS2, reverse: true}, y: orientedSegment{id: segmentS1, reverse: false}},
	{name: "swap_reverse_both", x: orientedSegment{id: segmentS2, reverse: true}, y: orientedSegment{id: segmentS1, reverse: true}},
}

// defaultRNGSeed is the fixed “zero” seed used when callers pass seed==0.
// The value is arbitrary but stable to keep reproducible defaults.
const defaultRNGSeed int64 = 1

// threeOptSearch runs 3-opt local search and publishes canonical result metadata.
//
// Implementation:
//   - Stage 1: Force the internal algorithm identity to ThreeOptOnly.
//   - Stage 2: Run threeOptKernel with opts.BestImprovement.
//   - Stage 3: Publish localSearchResult as Result without IDs or metric-closure metadata.
//   - Stage 4: Preserve ErrTimeLimit when a valid current tour exists.
//
// Behavior highlights:
//   - Symmetric mode uses the full symmetric move table.
//   - Asymmetric mode uses restricted orientation-preserving 3-opt*.
//   - Never claims exactness or global optimality.
//
// Inputs:
//   - dist: complete final solver matrix.
//   - initTour: closed Hamiltonian cycle.
//   - opts: local-search policy.
//
// Returns:
//   - *Result: improved or timeout-current tour result.
//   - error: nil or sentinel-classified failure.
//
// Errors:
//   - ErrInvalidOptions.
//   - ErrNilTour, ErrInvalidTour.
//   - Matrix validation sentinels.
//   - ErrTimeLimit with non-nil result when timeout happens after a valid current tour exists.
//
// Determinism:
//   - Canonical or seeded neighborhood scan order is governed by threeOptKernel.
//
// Complexity:
//   - Symmetric mode: O(iterations*n^3), Space O(n).
//   - Restricted asymmetric mode: governed by the tail-swap scan, Space O(n).
//
// Notes:
//   - IDs and metric closure metadata are attached only by SolveMatrix.
//
// AI-Hints:
//   - Do not describe asymmetric mode as full ATSP 3-opt.
//   - Do not drop timeout results when local.hasTour() is true.
func threeOptSearch(dist matrix.Matrix, initTour []int, opts Options) (*Result, error) {
	solverOptions := opts
	solverOptions.Algo = ThreeOptOnly

	local, err := threeOptKernel(dist, initTour, solverOptions, solverOptions.BestImprovement)
	if err != nil {
		if errors.Is(err, ErrTimeLimit) && local.hasTour() {
			result := publishLocalSearchResult(local, nil, solverOptions, ThreeOptOnly, false)
			return result, ErrTimeLimit
		}

		return nil, err
	}
	if !local.hasTour() {
		return nil, ErrInvalidTour
	}

	return publishLocalSearchResult(local, nil, solverOptions, ThreeOptOnly, false), nil
}

// threeOptKernel runs 3-opt / restricted ATSP 3-opt* and returns structured local-search metadata.
//
// Implementation:
//   - Stage 1: Validate options, input tour, complete weights, and fixed-start tour invariants.
//   - Stage 2: Copy the working tour and compute baseline cost.
//   - Stage 3: Enumerate triples i<j<k in deterministic or seed-controlled cyclic order.
//   - Stage 4: Evaluate explicit symmetric move table or ATSP tail-swap candidate.
//   - Stage 5: Validate every accepted move and publish success or partial-timeout result.
//
// Behavior highlights:
//   - Symmetric mode evaluates exactly seven named non-identity reconnections.
//   - ATSP mode uses a restricted orientation-preserving tail-swap neighborhood.
//   - Timeout returns a valid current tour when one exists.
//
// Inputs:
//   - dist: complete final distance matrix.
//   - initTour: closed Hamiltonian tour starting and ending at opts.StartVertex.
//   - opts: local-search policy.
//   - bestImprovement: true to apply the best candidate per sweep.
//
// Returns:
//   - localSearchResult: finalized local result when a current tour exists.
//   - error: nil or sentinel-classified failure.
//
// Errors:
//   - ErrInvalidOptions, ErrNilTour, ErrInvalidTour, ErrDimensionMismatch.
//   - ErrNilDistanceMatrix, ErrNonSquare, ErrNaNInf, ErrNegativeWeight, ErrIncompleteGraph.
//   - ErrTimeLimit with non-empty localSearchResult when a valid current tour exists.
//
// Determinism:
//   - Without ShuffleNeighborhood, triples and moves are scanned in fixed table order.
//   - With ShuffleNeighborhood, cyclic offsets are seed-controlled.
//
// Complexity:
//   - Time O(iter*n^3), Space O(n) plus the shared O(n^2) weight buffer.
//   - Accepted symmetric and ATSP moves assemble O(n) output tours.
//
// Notes:
//   - iterations counts accepted moves, not candidate evaluations.
//   - The accepted-move ValidateTour postcheck is intentionally compiled into normal builds.
//
// AI-Hints:
//   - Do not replace the explicit move table with opaque X/Y arrays.
//   - Do not skip ValidateTour after accepted 3-opt moves.
//   - WARNING: Performance-critical kernel. 3-opt involves evaluating up to 7
//     different edge-reconnection scenarios inside nested loops. Keeping this logic
//     inlined inside a single function is mandatory to allow the compiler to optimize
//     registers and eliminate non-inlineable function call overhead.
//
// nolint:gocyclo
func threeOptKernel(dist matrix.Matrix, initTour []int, opts Options, bestImprovement bool) (localSearchResult, error) {
	// Tour shape & invariants (the dispatcher already validated matrix shape).
	if err := validateOptionsStandalone(opts); err != nil {
		return localSearchResult{}, err
	}
	if initTour == nil {
		return localSearchResult{}, ErrNilTour
	}
	if len(initTour) < 2 {
		return localSearchResult{}, ErrInvalidTour
	}

	n := len(initTour) - 1
	if n < 2 {
		return localSearchResult{}, ErrInvalidTour
	}
	weights, err := copyCompleteWeights(dist, opts.Symmetric)
	if err != nil {
		return localSearchResult{}, err
	}
	if weights.n != n {
		return localSearchResult{}, ErrDimensionMismatch
	}

	// Validate the cycle invariants: closure, unique vertices, fixed start.
	if err = ValidateTour(initTour, n, opts.StartVertex); err != nil {
		return localSearchResult{}, err
	}

	// Working copy and baseline tour cost (strict validation of current edges).
	cur := make([]int, n+1)
	copy(cur, initTour)              // keep caller’s slice immutable
	cost, err := TourCost(dist, cur) // verifies no NaN/+Inf on existing arcs
	if err != nil {
		return localSearchResult{}, err
	}

	// Policy knobs.
	eps := opts.Eps                   // accept only Δ < −eps (eps≥0 validated beforehand)
	maxMoves := opts.ThreeOptMaxMoves // 0 ⇒ unlimited number of accepted moves

	// RNG for randomized triple order: enabled only when ShuffleNeighborhood is set.
	var rng randLite // tiny shim interface with Intn(int)
	if opts.ShuffleNeighborhood {
		rng = rngFromSeed(opts.Seed)
	}

	// Soft deadline (cheap periodic checks; negligible overhead).
	var (
		useDeadline bool      // whether to enforce a time budget
		deadline    time.Time // absolute deadline if enabled
		steps       int       // Δ-evaluation counter for sparse checks
	)
	if opts.TimeLimit > 0 {
		useDeadline = true
		deadline = localSearchNow().Add(opts.TimeLimit)
	}
	// Check every 4096 Δ evaluations; this keeps the check overhead tiny.
	checkDeadline := func() bool {
		steps++
		if (steps & threeOptDeadlineCheckMask) != 0 { // every 4096 Δ-evals
			return false
		}

		return localSearchDeadlineExpired(useDeadline, deadline)
	}

	// Main improvement loop.
	accepted := 0
	var i, j int // matrix indices reused across loops
	for {
		found := false // did we discover an improving candidate in this sweep?

		// Best-improvement bookkeeping for a single sweep.
		bestDelta := 0.0            // most negative Δ seen so far
		var bestI, bestJ, bestK int // triple indices for the best move
		var bestMove threeOptMove   // symmetric move selected by best-improvement
		var bestIsATSPTailSwap bool // whether best move is the ATSP tail swap

		// Randomized cyclic offset for the outermost index i (optional when rng!=nil).
		offI := 0
		if rng != nil && n > 3 {
			offI = rng.Intn(maxi(1, n-3)) // safe even at minimal n
		}

		// Enumerate all triples 1≤i<j<k≤n−1 with optional cyclic offsets to reduce structure bias.
		var (
			k                        int     // k index
			ii, jj, kk               int     // loop counters
			spanJ, spanK, offJ, offK int     // per-level spans and offsets
			moveIndex                int     // symmetric move-table index
			feasible                 bool    // candidate feasibility under missing-edge policy
			delta                    float64 // candidate improvement (negative is good)
		)
		for ii = 0; ii < n-3; ii++ {
			i = 1 + ((ii + offI) % (n - 3)) // ensure i ∈ [1..n-3] with cyclic shift

			spanJ = (n - 2) - i // j ∈ (i..n-2] ⇒ span of length (n-2)-i
			if spanJ <= 0 {
				continue // no feasible j for this i
			}
			offJ = 0
			if rng != nil {
				offJ = rng.Intn(spanJ) // independent cyclic offset per i
			}

			for jj = 0; jj < spanJ; jj++ {
				j = i + 1 + ((jj + offJ) % spanJ) // j ∈ [i+1..n-2]

				spanK = (n - 1) - j // k ∈ (j..n-1] ⇒ span of length (n-1)-j
				if spanK <= 0 {
					continue // no feasible k for this (i,j)
				}
				offK = 0
				if rng != nil {
					offK = rng.Intn(spanK) // independent cyclic offset per (i,j)
				}

				for kk = 0; kk < spanK; kk++ {
					k = j + 1 + ((kk + offK) % spanK) // k ∈ [j+1..n-1]

					if opts.Symmetric {
						for moveIndex = 0; moveIndex < len(symmetricThreeOptMoves); moveIndex++ {
							if checkDeadline() {
								local, finishErr := finishLocalSearchCurrent(cur, cost, accepted, true, n, opts.StartVertex)
								if finishErr != nil {
									return localSearchResult{}, finishErr
								}

								return local, ErrTimeLimit
							}

							move := symmetricThreeOptMoves[moveIndex]
							if err = validateThreeOptMove(move); err != nil {
								return localSearchResult{}, err
							}

							delta, feasible = threeOptDeltaSymmetric(weights.at, cur, i, j, k, move)
							if !feasible || delta >= -eps {
								continue
							}

							if !bestImprovement {
								next := applyThreeOptSymmetric(cur, i, j, k, move)
								if err = ValidateTour(next, n, opts.StartVertex); err != nil {
									return localSearchResult{}, err
								}

								cur = next
								cost += delta
								accepted++
								found = true
							} else if delta < bestDelta {
								bestDelta = delta
								bestI, bestJ, bestK = i, j, k
								bestMove = move
								bestIsATSPTailSwap = false
								found = true
							}

							if found && !bestImprovement {
								break
							}
						}
					} else {
						if checkDeadline() {
							local, finishErr := finishLocalSearchCurrent(cur, cost, accepted, true, n, opts.StartVertex)
							if finishErr != nil {
								return localSearchResult{}, finishErr
							}

							return local, ErrTimeLimit
						}

						delta, feasible = threeOptDeltaATSPTailSwap(weights.at, cur, i, j, k)
						if !feasible || delta >= -eps {
							continue
						}

						if !bestImprovement {
							next := applyThreeOptATSPTailSwap(cur, i, j, k)
							if err = ValidateTour(next, n, opts.StartVertex); err != nil {
								return localSearchResult{}, err
							}

							cur = next
							cost += delta
							accepted++
							found = true
						} else if delta < bestDelta {
							bestDelta = delta
							bestI, bestJ, bestK = i, j, k
							bestIsATSPTailSwap = true
							found = true
						}
					}

					// Early exit for first-improvement policy; best-improvement keeps scanning.
					if found && !bestImprovement {
						break
					}
				}
				if found && !bestImprovement {
					break
				}
			}
			if found && !bestImprovement {
				break
			}
		}

		// Best-improvement: apply the remembered best move once per sweep.
		if bestImprovement && found {
			var next []int
			if bestIsATSPTailSwap {
				next = applyThreeOptATSPTailSwap(cur, bestI, bestJ, bestK)
			} else {
				next = applyThreeOptSymmetric(cur, bestI, bestJ, bestK, bestMove)
			}
			if err = ValidateTour(next, n, opts.StartVertex); err != nil {
				return localSearchResult{}, err
			}
			cur = next
			cost += bestDelta
			accepted++
		}

		// Termination guards.
		if !found {
			break // local optimum for the chosen neighborhood/policy
		}
		if maxMoves > 0 && accepted >= maxMoves {
			break // hit user-specified move cap
		}
	}

	return finishLocalSearchCurrent(cur, cost, accepted, false, n, opts.StartVertex)
}

// validateThreeOptMove verifies that a symmetric 3-opt move uses S1 and S2 exactly once.
// It protects the move table from invalid edits that would duplicate or drop a segment.
//
// Implementation:
//   - Stage 1: Reject moves that use the same segment twice.
//   - Stage 2: Reject unknown segment IDs.
//   - Stage 3: Return nil for a structurally valid table entry.
//
// Behavior highlights:
//   - No allocation.
//   - Designed for normal build checks and internal tests.
//
// Inputs:
//   - move: symmetric 3-opt move table entry.
//
// Returns:
//   - error: nil for valid move.
//
// Errors:
//   - ErrInvalidTour when the move duplicates one segment.
//   - ErrInvalidOptions when a segment ID is outside the known domain.
//
// Determinism:
//   - Pure structural check.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The identity move is absent by construction and should remain absent.
//
// AI-Hints:
//   - Do not add a new move without updating invariant and delta tests.
func validateThreeOptMove(move threeOptMove) error {
	if move.x.id == move.y.id {
		return ErrInvalidTour
	}
	if (move.x.id != segmentS1 && move.x.id != segmentS2) ||
		(move.y.id != segmentS1 && move.y.id != segmentS2) {
		return ErrInvalidOptions
	}

	return nil
}

// segmentEndpoints maps an oriented segment to its first and last boundary vertices.
// For S1 the forward endpoints are b,c; for S2 the forward endpoints are d,e.
//
// Implementation:
//   - Stage 1: Select the segment by ID.
//   - Stage 2: Swap endpoints when reverse=true.
//   - Stage 3: Return sentinel endpoints for invalid IDs; validation should prevent them.
//
// Behavior highlights:
//   - No allocation.
//   - Used by delta calculation only.
//
// Inputs:
//   - seg: oriented segment descriptor.
//   - b,c,d,e: cut boundary vertices.
//
// Returns:
//   - first: first vertex of the oriented segment.
//   - last: last vertex of the oriented segment.
//
// Errors:
//   - None. validateThreeOptMove owns structural errors.
//
// Determinism:
//   - Pure switch over segment ID.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Invalid endpoints are defensive only and should be unreachable after validation.
//
// AI-Hints:
//   - Do not interpret reverse as reversing the whole tour; it reverses only one segment.
func segmentEndpoints(seg orientedSegment, b, c, d, e int) (first int, last int) {
	switch seg.id {
	case segmentS1:
		if seg.reverse {
			return c, b
		}

		return b, c

	case segmentS2:
		if seg.reverse {
			return e, d
		}

		return d, e

	default:
		return -1, -1
	}
}

// threeOptDeltaSymmetric computes the boundary-edge delta for one symmetric 3-opt move.
// Internal segment edges cancel under symmetric distances, so only three removed and
// three added boundary edges are needed.
//
// Implementation:
//   - Stage 1: Read the six boundary vertices around cuts i,j,k.
//   - Stage 2: Resolve oriented endpoints for X and Y.
//   - Stage 3: Compute added-minus-removed boundary cost.
//   - Stage 4: Reject candidates that would introduce missing edges.
//
// Behavior highlights:
//   - Does not allocate.
//   - Does not mutate the tour.
//   - Returns feasible=false for +Inf candidate edges.
//
// Inputs:
//   - at: O(1) weight accessor.
//   - tour: closed Hamiltonian tour.
//   - i,j,k: cut indices with 1 <= i < j < k <= n-1.
//   - move: validated symmetric move descriptor.
//
// Returns:
//   - float64: candidate delta, negative means improvement.
//   - bool: false when the move would introduce a missing edge.
//
// Errors:
//   - None. Shape validation belongs to the caller and tests.
//
// Determinism:
//   - Fixed boundary reads and fixed arithmetic order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This formula is valid for symmetric TSP because reversed segment internals keep cost.
//   - ATSP uses a separate restricted tail-swap delta helper.
//
// AI-Hints:
//   - Do not reuse this symmetric delta for ATSP reversed segments.
func threeOptDeltaSymmetric(at func(int, int) float64, tour []int, i, j, k int, move threeOptMove) (float64, bool) {
	a, b := tour[i-1], tour[i]
	c, d := tour[j-1], tour[j]
	e, f := tour[k-1], tour[k]

	xFirst, xLast := segmentEndpoints(move.x, b, c, d, e)
	yFirst, yLast := segmentEndpoints(move.y, b, c, d, e)

	w1 := at(a, xFirst)
	w2 := at(xLast, yFirst)
	w3 := at(yLast, f)

	if math.IsInf(w1, 0) || math.IsInf(w2, 0) || math.IsInf(w3, 0) {
		return 0, false
	}

	removed := at(a, b) + at(c, d) + at(e, f)
	added := w1 + w2 + w3

	return added - removed, true
}

// threeOptDeltaATSPTailSwap computes the restricted ATSP 3-opt* tail-swap delta.
// The move keeps segment orientations and swaps S1/S2: P + S2 + S1 + S3.
//
// Implementation:
//   - Stage 1: Read the six boundary vertices around cuts i,j,k.
//   - Stage 2: Compute added arcs a->d, e->b, c->f.
//   - Stage 3: Subtract removed arcs a->b, c->d, e->f.
//   - Stage 4: Reject missing added arcs.
//
// Behavior highlights:
//   - Does not reverse directed arcs.
//   - Does not mutate the tour.
//   - No allocation.
//
// Inputs:
//   - at: O(1) weight accessor.
//   - tour: closed Hamiltonian tour.
//   - i,j,k: cut indices.
//
// Returns:
//   - float64: candidate delta.
//   - bool: false when an added arc is missing.
//
// Errors:
//   - None.
//
// Determinism:
//   - Fixed boundary reads and arithmetic order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is not full ATSP 3-opt; it is a restricted orientation-preserving 3-opt* move.
//
// AI-Hints:
//   - Do not call this “full ATSP 3-opt”.
func threeOptDeltaATSPTailSwap(at func(int, int) float64, tour []int, i, j, k int) (float64, bool) {
	a, b := tour[i-1], tour[i]
	c, d := tour[j-1], tour[j]
	e, f := tour[k-1], tour[k]

	w1 := at(a, d)
	w2 := at(e, b)
	w3 := at(c, f)

	if math.IsInf(w1, 0) || math.IsInf(w2, 0) || math.IsInf(w3, 0) {
		return 0, false
	}

	removed := at(a, b) + at(c, d) + at(e, f)
	added := w1 + w2 + w3

	return added - removed, true
}

// appendOrientedSegment appends S1 or S2 to out in the requested orientation.
// It is the only segment-emission helper used by symmetric 3-opt assembly.
//
// Implementation:
//   - Stage 1: Select S1 or S2 by segment ID.
//   - Stage 2: Append forward order directly or reverse order manually.
//   - Stage 3: Leave invalid segment IDs as no-op; validation prevents them.
//
// Behavior highlights:
//   - Mutates only the output slice pointed to by out.
//   - Does not allocate beyond append growth already reserved by the caller.
//
// Inputs:
//   - out: destination tour builder.
//   - s1: first middle segment.
//   - s2: second middle segment.
//   - seg: segment and orientation descriptor.
//
// Returns:
//   - None.
//
// Errors:
//   - None. validateThreeOptMove owns segment validity.
//
// Determinism:
//   - Fixed left-to-right or right-to-left append order.
//
// Complexity:
//   - Time O(len(segment)), Space amortized O(1) beyond out capacity.
//
// Notes:
//   - Caller preallocates n+1 capacity to avoid repeated growth.
//
// AI-Hints:
//   - Do not append S3 here; S3 is anchored by applyThreeOptSymmetric.
func appendOrientedSegment(out *[]int, s1 []int, s2 []int, seg orientedSegment) {
	var source []int

	switch seg.id {
	case segmentS1:
		source = s1
	case segmentS2:
		source = s2
	default:
		return
	}

	if !seg.reverse {
		*out = append(*out, source...)
		return
	}

	for index := len(source) - 1; index >= 0; index-- {
		*out = append(*out, source[index])
	}
}

// applyThreeOptSymmetric builds P + X + Y + S3 + start for a validated symmetric move.
// It preserves start position and uses S1/S2 exactly once according to the move table.
//
// Implementation:
//   - Stage 1: Slice tour into P, S1, S2, and S3.
//   - Stage 2: Append P, oriented X, oriented Y, and fixed tail S3.
//   - Stage 3: Close the tour with the original start.
//
// Behavior highlights:
//   - Returns a fresh slice.
//   - Does not mutate the input tour.
//   - Keeps S3 anchored as the tail.
//
// Inputs:
//   - tour: closed Hamiltonian tour.
//   - i,j,k: cut indices.
//   - move: validated symmetric move descriptor.
//
// Returns:
//   - []int: fresh closed candidate tour.
//
// Errors:
//   - None. The caller validates the result with ValidateTour.
//
// Determinism:
//   - Fixed append order from the named move.
//
// Complexity:
//   - Time O(n), Space O(n).
//
// Notes:
//   - Accepted moves must be postchecked with ValidateTour in normal builds.
//
// AI-Hints:
//   - Do not skip the postcheck after this helper returns.
func applyThreeOptSymmetric(tour []int, i int, j int, k int, move threeOptMove) []int {
	n := len(tour) - 1

	prefix := tour[:i]
	s1 := tour[i:j]
	s2 := tour[j:k]
	tail := tour[k:n]

	out := make([]int, 0, n+1)
	out = append(out, prefix...)

	appendOrientedSegment(&out, s1, s2, move.x)
	appendOrientedSegment(&out, s1, s2, move.y)

	out = append(out, tail...)
	out = append(out, tour[0])

	return out
}

// applyThreeOptATSPTailSwap builds P + S2 + S1 + S3 + start.
// It is a restricted orientation-preserving ATSP 3-opt* move, not full ATSP 3-opt.
//
// Implementation:
//   - Stage 1: Slice tour into P, S1, S2, and S3.
//   - Stage 2: Append P, S2, S1, S3 in forward orientation.
//   - Stage 3: Close with the original start.
//
// Behavior highlights:
//   - Returns a fresh slice.
//   - Does not reverse any directed segment.
//   - Does not mutate the input tour.
//
// Inputs:
//   - tour: closed Hamiltonian tour.
//   - i,j,k: cut indices.
//
// Returns:
//   - []int: fresh closed candidate tour.
//
// Errors:
//   - None. The caller validates the result with ValidateTour.
//
// Determinism:
//   - Fixed append order.
//
// Complexity:
//   - Time O(n), Space O(n).
//
// Notes:
//   - This helper name intentionally states “TailSwap” to avoid overclaiming full ATSP 3-opt.
//
// AI-Hints:
//   - Do not replace this with symmetric reversal logic for ATSP.
func applyThreeOptATSPTailSwap(tour []int, i int, j int, k int) []int {
	n := len(tour) - 1

	prefix := tour[:i]
	s1 := tour[i:j]
	s2 := tour[j:k]
	tail := tour[k:n]

	out := make([]int, 0, n+1)
	out = append(out, prefix...)
	out = append(out, s2...)
	out = append(out, s1...)
	out = append(out, tail...)
	out = append(out, tour[0])

	return out
}

// maxi returns the larger of two ints.
//
// Implementation:
//   - Stage 1: Compare a and b.
//   - Stage 2: Return a when a>=b, otherwise b.
//
// Behavior highlights:
//   - Allocation-free.
//   - Kept local to avoid importing generic helpers into hot local-search code.
//
// Inputs:
//   - a: first integer.
//   - b: second integer.
//
// Returns:
//   - int: maximum value.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure comparison.
//
// Complexity:
//   - Time O(1), Space O(1).
func maxi(a, b int) int {
	if a > b {
		return a
	}

	return b
}

// randLite is the minimal RNG interface required by randomized local-search moves.
// It is satisfied by *rand.Rand and by deterministic test doubles that implement Intn.
//
// Implementation:
//   - Single-method interface: Intn(int) int.
//
// Behavior highlights:
//   - Keeps local-search code decoupled from concrete RNG types.
//   - Enables deterministic tests.
//   - Avoids package-level random state.
//
// Notes:
//   - Instances are created by rngFromSeed.
type randLite interface {
	Intn(n int) int
}

// rngFromSeed returns a deterministic *rand.Rand for local-search perturbation choices.
// seed==0 maps to defaultRNGSeed so the zero Options value remains reproducible.
//
// Implementation:
//   - Stage 1: Replace zero seed with defaultRNGSeed.
//   - Stage 2: Create a new rand.Source.
//   - Stage 3: Return rand.New(source).
//
// Behavior highlights:
//   - Does not use global math/rand state.
//   - Deterministic for the same seed.
//   - Allocation is limited to RNG construction.
//
// Inputs:
//   - seed: caller-provided seed; zero means package default.
//
// Returns:
//   - *rand.Rand: deterministic RNG stream.
//
// Errors:
//   - None.
//
// Determinism:
//   - Same effective seed produces the same sequence.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Do not use rand.Seed or package-level rand.Intn here.
//   - This is not used for security-sensitive operations or cryptography.
//
// nolint:gosec // G404: weak random provider is required for mathematical simulations
func rngFromSeed(seed int64) *rand.Rand {
	if seed == 0 {
		seed = defaultRNGSeed
	}

	return rand.New(rand.NewSource(seed))
}
