// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw coordinates deterministic exact top-K ranking across independent workspaces.
// This file keeps per-worker data isolated and accounts for completed versus interrupted work.
package dtw

import (
	"container/heap"
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
)

// candidateOutcome is a single fully evaluated result or a classified stopped candidate.
// An errored outcome is never eligible for the exact top-K ranking.
type candidateOutcome struct {
	start    int
	distance float64
	err      error
}

// worstSearchHeap retains the K smallest exact hits with the worst hit at its root.
// The heap uses descending (distance,start,end) order internally and is private.
type worstSearchHeap []SearchHit

// Len reports the number of retained exact hits in the coordinator's private heap.
// It intentionally excludes unfinished and lower-bound-only candidates.
//
// Implementation:
//   - Stage 1: inspect the retained backing slice length.
//   - Stage 2: return its count without additional allocation.
//
// Behavior highlights:
//   - The heap's size never exceeds validated TopK.
//
// Inputs:
//   - h: private worst-first heap receiver.
//
// Returns:
//   - int: exact count of retained finite scores.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure slice-length lookup.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Workers never mutate this heap; only the coordinator owns it.
//
// AI-Hints:
//   - Do not count in-flight candidates as retained exact scores.
func (h worstSearchHeap) Len() int { return len(h) }

// Less compares two retained hits in reverse canonical rank order.
// This max-heap ordering keeps the worst selected hit at index zero for replacement.
//
// Implementation:
//   - Stage 1: retrieve the indexed exact hits.
//   - Stage 2: invert searchHitLess to place larger distance/start/end first.
//
// Behavior highlights:
//   - A tie on score is broken by start, then end, without epsilon.
//
// Inputs:
//   - i, j: valid private heap indices.
//
// Returns:
//   - bool: true when h[i] ranks worse than h[j].
//
// Errors:
//   - None; heap.Interface controls index validity.
//
// Determinism:
//   - Exact lexicographic comparison, unaffected by arrival order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The heap relation must remain the inverse of final ascending sorting.
//
// AI-Hints:
//   - Never use a tolerance that can violate strict heap ordering.
func (h worstSearchHeap) Less(i, j int) bool { return searchHitLess(h[j], h[i]) }

// Swap exchanges two retained hit values at valid heap indices.
// It preserves each hit's complete window and squared objective together.
//
// Implementation:
//   - Stage 1: index two independently owned heap entries.
//   - Stage 2: exchange the SearchHit values in one assignment.
//
// Behavior highlights:
//   - No input buffer or worker workspace is modified.
//
// Inputs:
//   - i, j: valid private heap indices.
//
// Returns:
//   - None; receiver storage is updated in place.
//
// Errors:
//   - None; heap.Interface controls index validity.
//
// Determinism:
//   - Pure swap, preserving the comparator's total rank.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Only the coordinator goroutine invokes this mutator.
//
// AI-Hints:
//   - Keep Window and SquaredDistance together when swapping.
func (h worstSearchHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

// Push appends one completed exact SearchHit to the coordinator-owned heap.
// container/heap calls this method as one step of an ordered insertion.
//
// Implementation:
//   - Stage 1: accept a SearchHit from the type-safe coordinator.
//   - Stage 2: append its value to the private backing slice.
//
// Behavior highlights:
//   - Only completed finite exact hits are ever inserted.
//
// Inputs:
//   - value: SearchHit boxed as any for heap.Interface compatibility.
//
// Returns:
//   - None; backing slice length increases by one.
//
// Errors:
//   - None under the private type precondition.
//
// Determinism:
//   - Insertion ordering is restored by container/heap using Less.
//
// Complexity:
//   - Amortized O(1) append; heap.Push as a whole is O(log K).
//
// Notes:
//   - Capacity is preallocated to TopK so normal insertions do not resize.
//
// AI-Hints:
//   - Never pass an arbitrary value type from outside the private coordinator.
func (h *worstSearchHeap) Push(value any) { *h = append(*h, value.(SearchHit)) }

// Pop removes and returns the last private heap element after heap-order rearrangement.
// container/heap controls the rearrangement; this method performs only slice removal.
//
// Implementation:
//   - Stage 1: locate the final element of the nonempty heap backing slice.
//   - Stage 2: shorten the slice and return that SearchHit value.
//
// Behavior highlights:
//   - No worker scratch, caller input, or incomplete candidate state is published.
//
// Inputs:
//   - h: nonempty coordinator-owned heap receiver.
//
// Returns:
//   - any: the removed SearchHit boxed for heap.Interface.
//
// Errors:
//   - None under heap.Interface's nonempty-call precondition.
//
// Determinism:
//   - Removes exactly the element positioned by the heap's comparator.
//
// Complexity:
//   - Time O(1) slice operation; heap.Pop as a whole is O(log K).
//
// Notes:
//   - No memory shrink or reallocation is performed during removal.
//
// AI-Hints:
//   - Do not call this method directly on an empty heap.
func (h *worstSearchHeap) Pop() any {
	old := *h
	last := len(old) - 1
	value := old[last]
	*h = old[:last]
	return value
}

// searchHitLess compares two exact hits in canonical ascending rank order.
// Its strict float64 comparison is deterministic because all published costs are finite.
//
// Implementation:
//   - Stage 1: compare exact squared scores.
//   - Stage 2: resolve ties by lower half-open start index.
//   - Stage 3: resolve any remaining tie by lower exclusive end index.
//
// Behavior highlights:
//   - No tolerance or approximate comparison changes ranking.
//
// Inputs:
//   - a, b: completed exact finite hit values.
//
// Returns:
//   - bool: true exactly when a ranks ahead of b.
//
// Errors:
//   - None after scorer validation.
//
// Determinism:
//   - Lexicographic tuple (SquaredDistance,Start,End).
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This relation must be used identically for heap replacement and final sorting.
//
// AI-Hints:
//   - Never prune at lowerBound==worst exact distance, because an earlier start may win.
//   - Do not mix normalized and raw scores in a single ranking.
func searchHitLess(a, b SearchHit) bool {
	if a.SquaredDistance != b.SquaredDistance {
		return a.SquaredDistance < b.SquaredDistance
	}
	if a.Window.Start != b.Window.Start {
		return a.Window.Start < b.Window.Start
	}
	return a.Window.End < b.Window.End
}

// runSearchWorkers evaluates admitted windows with private scratch and publishes exact hits.
// Workers may finish candidates in any order, but the coordinator's total-order heap
// produces the same complete ranking for any positive worker count.
//
// Implementation:
//   - Stage 1: launch a bounded worker pool, each with two band rows and optional z buffer.
//   - Stage 2: claim stride-grid positions atomically; skip excluded starts without evaluation.
//   - Stage 3: score every claimed admitted candidate, sending completed/error outcomes.
//   - Stage 4: drain all worker outcomes even after a failure, then sort the transferred heap.
//
// Behavior highlights:
//   - No O(N) candidate queue is built; positions are computed directly from a shared ordinal.
//   - A candidate runtime failure stops further dispatch and returns a non-global partial result.
//   - Every evaluated DP cell reserves a shared credit before computing its local cost.
//
// Inputs:
//   - ctx: user-owned cancellation source; result: unpublished SearchResult under construction.
//   - plan: validated copied exclusion policy/dimensions; query/data: read-only finite samples.
//   - opts: validated worker/normalization/budget policy.
//
// Returns:
//   - *SearchResult: complete exact ranking or incomplete completed-only ranking.
//   - error: first relevant runtime/budget/context error with sentinel identity preserved.
//
// Errors:
//   - ErrBudget, ErrZeroVariance, ErrOverflow/ErrNaNInf, or context error after work starts.
//
// Determinism:
//   - Full rankings do not depend on completion order; one-worker partial prefix is stable.
//
// Complexity:
//   - Work O(C*L*b + C log K), aggregate private storage O(W*b + W*L + K).
//
// Notes:
//   - Partial rankings are scheduling-dependent when Workers>1, as explicitly documented.
//   - Worker scratch, input pointers, and result channel storage never escape SearchResult.
//
// AI-Hints:
//   - Always drain the result channel after cancellation to avoid goroutine leaks.
//   - Never promote an errored or interrupted candidate into CandidatesCompleted.
func runSearchWorkers(
	ctx context.Context,
	result *SearchResult,
	plan searchPlan,
	query, data []float64,
	opts SearchOptions,
) (*SearchResult, error) {
	workerContext, cancel := context.WithCancel(ctx)
	defer cancel()

	// Atomic positions avoid O(number-of-candidates) dispatch storage. Admitted
	// starts are a subset of the increasing stride progression after exclusions.
	var nextPosition atomic.Uint64
	var visited atomic.Uint64
	outcomes := make(chan candidateOutcome, plan.workers)
	var workers sync.WaitGroup
	workers.Add(plan.workers)

	for worker := 0; worker < plan.workers; worker++ {
		go func() {
			defer workers.Done()
			prev := make([]dpCell, plan.width)
			curr := make([]dpCell, plan.width)
			var normalizedCandidate []float64
			if opts.Normalization == SearchZPopulation {
				normalizedCandidate = make([]float64, plan.length)
			}

			for {
				if workerContext.Err() != nil {
					return
				}
				ordinal := nextPosition.Add(1) - 1
				if ordinal >= plan.positions {
					return
				}
				start := int(ordinal) * plan.stride
				if excludedStart(plan.excluded, start) {
					continue
				}

				// Each worker owns its normalized buffer and classified rows. In raw
				// mode the complete window aliases borrowed, immutable caller input.
				candidate := data[start : start+plan.length]
				var err error
				if normalizedCandidate != nil {
					if err = normalizeZPopulationInto(normalizedCandidate, candidate); err == nil {
						candidate = normalizedCandidate
					}
				}
				var distance float64
				if err == nil {
					distance, err = scoreSquaredBand(
						workerContext, query, candidate, plan.radius,
						prev, curr, &visited, opts.MaxCells,
					)
				}
				if err != nil {
					err = fmt.Errorf("dtw: candidate start=%d: %w", start, err)
				}
				// Sending even after cancellation allows the coordinator to account
				// for every started task and to release all workers without a leak.
				outcomes <- candidateOutcome{start: start, distance: distance, err: err}
				if err != nil {
					return
				}
			}
		}()
	}
	go func() {
		workers.Wait()
		close(outcomes)
	}()

	best := make(worstSearchHeap, 0, plan.hitCap)
	var firstError error
	for candidate := range outcomes {
		if candidate.err != nil {
			// Preserve the original runtime/budget classification even when
			// cancellation later interrupts other in-flight candidates.
			if firstError == nil || (errors.Is(firstError, context.Canceled) &&
				!errors.Is(candidate.err, context.Canceled)) {
				firstError = candidate.err
			}
			cancel()
			continue
		}
		result.CandidatesCompleted++
		hit := SearchHit{
			Window:          IndexRange{Start: candidate.start, End: candidate.start + plan.length},
			SquaredDistance: candidate.distance,
		}
		if len(best) < plan.hitCap {
			heap.Push(&best, hit)
		} else if searchHitLess(hit, best[0]) {
			// Strict total ordering makes heap replacement scheduling-independent.
			best[0] = hit
			heap.Fix(&best, 0)
		}
	}
	result.CellsVisited = visited.Load()

	// The heap backing buffer becomes caller-owned; no additional full-size
	// copy is needed and no worker retains an alias to this slice.
	sort.Slice(best, func(i, j int) bool { return searchHitLess(best[i], best[j]) })
	result.Hits = []SearchHit(best)
	if firstError != nil {
		return result, firstError
	}
	if result.CandidatesCompleted == result.CandidatesTotal {
		result.Complete = true
		return result, nil
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	// Every admitted start must yield a completed/error outcome. An unmet
	// invariant must not be presented as a successfully completed ranking.
	return result, ErrBadInput
}
