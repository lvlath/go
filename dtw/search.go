// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw implements an exact, deterministic fixed-length subsequence search kernel.
// This file validates the search domain and prepares certified worker execution.
package dtw

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"sort"
	"unsafe"
)

// searchPlan is a validated read-only description of one search invocation.
// It owns its copied exclusion ranges and keeps no mutable per-worker state.
type searchPlan struct {
	length    int             // Complete candidate and query length in scalar samples.
	stride    int             // Positive candidate start increment.
	radius    int             // -1 or nonnegative admissible half-band width.
	positions uint64          // Raw stride-aligned starts before exclusions.
	admitted  uint64          // Exact number of starts after exclusions.
	workers   int             // Actual bounded worker count, at most admitted.
	hitCap    int             // Retained exact top-K count, at most admitted.
	width     int             // Per-worker classified band row width.
	excluded  []IndexRange    // Detached sorted/coalesced excluded starts.
	envelope  *searchEnvelope // Optional immutable machine-pruning certificate.
}

// move to api.gp and discribe!
func Search(ctx context.Context, query, data []float64, opts SearchOptions) (*SearchResult, error) {
	return searchFixedWindows(ctx, query, data, opts)
}

// searchFixedWindows finds the globally exact top-K fixed-length squared-DTW windows in a scalar series.
// Candidate starts are 0,Stride,... with complete query-length coverage; excluded *starts*
// are removed before scoring. This is not an open-boundary subsequence-DTW algorithm.
//
// Implementation:
//   - Stage 1: validate context, options, input shapes, every sample, and normalization domain.
//   - Stage 2: copy/coalesce exclusions, account for all explicit buffers, and normalize query.
//   - Stage 3: evaluate or soundly prune admitted windows using the band-only recurrence.
//   - Stage 4: retain the best K fully evaluated hits, sort by (distance,start,end), and publish.
//
// Behavior highlights:
//   - Complete results are exact top-K; optional pruning requires machine certificates.
//   - Constant z-normalized query is an input error (nil result); constant candidate yields partial+error.
//   - Workers share only atomic cutoff and checked cell counters; scratch is per-worker.
//   - MaxCells includes all evaluated cells, including interrupted/inflight candidate work.
//
// Inputs:
//   - ctx: non-nil context, checked before validation and cooperatively during each candidate.
//   - query: nonempty finite scalar template, borrowed read-only.
//   - data: finite scalar series, borrowed read-only; shorter-than-query data yields no hits.
//   - opts: explicit SearchOptions; TopK, Stride, and Workers must each be positive.
//
// Returns:
//   - *SearchResult: detached, sorted exact hits; Complete=true certifies global ranking.
//   - After compute begins, ErrBudget/context/runtime numeric failure returns partial+error.
//
// Errors:
//   - ErrInvalidOption: nil context or invalid policy; ErrNilInput/ErrEmptyInput: invalid slices.
//   - ErrShape: malformed excluded-start intervals; ErrNonFinite joined with ErrNaNInf: samples.
//   - ErrZeroVariance: exactly constant query or candidate under SearchZPopulation.
//   - ErrBudget: working-buffer preflight failure or exhausted global cell credits.
//   - ErrOverflow joined with ErrNaNInf: squared local or accumulated numeric overflow.
//
// Determinism:
//   - Starts use increasing stride multiples; final ranking is distance/start/end ascending.
//   - Complete rankings are identical across worker counts; partial rankings may vary by scheduling.
//
// Complexity:
//   - For C admitted windows, L=len(query), b=min(L,2*Radius+1), O(C*L*b + C*log K).
//   - Radius=-1 uses b=L. Z normalization adds O(C*L) time.
//   - Memory O(R + K + W*b + W*L + L) in z-mode, O(R + K + W*b + L) in raw mode.
//
// Notes:
//   - All costs are squared scalar local costs with zero slope penalty; generic CostFunc is excluded.
//   - Certificates require a finite diagonal, finite local domain, and strictly greater machine LB_Keogh.
//   - MaxWorkingBytes covers explicitly allocated buffers, not runtime/goroutine stack metadata.
//
// AI-Hints:
//   - A non-complete SearchResult is not certified global top-K, even when Hits is nonempty.
//   - Do not use an uncertified floating lower bound for pruning near cutoff equality.
func searchFixedWindows(ctx context.Context, query, data []float64, opts SearchOptions) (*SearchResult, error) {
	if ctx == nil {
		return nil, ErrInvalidOption
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if opts.Radius < -1 || opts.TopK <= 0 || opts.Stride <= 0 || opts.Workers <= 0 ||
		(opts.Normalization != SearchRaw && opts.Normalization != SearchZPopulation) {
		return nil, ErrInvalidOption
	}
	if query == nil || data == nil {
		return nil, ErrNilInput
	}
	if len(query) == 0 {
		return nil, ErrEmptyInput
	}

	// Shape validation precedes scanning potentially expensive/non-finite input.
	// Copy the caller's exclusions: sort/coalescing must never mutate their slice.
	positions := uint64(0)
	if len(data) >= len(query) {
		positions = uint64((len(data)-len(query))/opts.Stride) + 1
	} else if len(opts.ExcludedStarts) != 0 {
		return nil, ErrShape
	}
	startDomain := 0
	if len(data) >= len(query) {
		startDomain = len(data) - len(query) + 1
	}
	for _, exclusion := range opts.ExcludedStarts {
		if exclusion.Start < 0 || exclusion.Start > exclusion.End || exclusion.End > startDomain {
			return nil, ErrShape
		}
	}

	for index, sample := range query {
		if math.IsNaN(sample) || math.IsInf(sample, 0) {
			return nil, fmt.Errorf("dtw: query index %d: %w", index, errors.Join(ErrNonFinite, ErrNaNInf))
		}
	}
	for index, sample := range data {
		if math.IsNaN(sample) || math.IsInf(sample, 0) {
			return nil, fmt.Errorf("dtw: data index %d: %w", index, errors.Join(ErrNonFinite, ErrNaNInf))
		}
	}
	if opts.Normalization == SearchZPopulation && isExactlyConstant(query) {
		return nil, fmt.Errorf("dtw: query: %w", ErrZeroVariance)
	}

	// Check the exclusion-copy size before copying any caller-owned range slice.
	exclusionBytes, overflow := checkedBytes(uint64(len(opts.ExcludedStarts)), uint64(unsafe.Sizeof(IndexRange{})))
	if overflow {
		return nil, errors.Join(ErrOverflow, ErrNaNInf)
	}
	if opts.MaxWorkingBytes > 0 && exclusionBytes > opts.MaxWorkingBytes {
		return nil, ErrBudget
	}

	exclusions := append([]IndexRange(nil), opts.ExcludedStarts...)
	sort.Slice(exclusions, func(i, j int) bool {
		if exclusions[i].Start != exclusions[j].Start {
			return exclusions[i].Start < exclusions[j].Start
		}
		return exclusions[i].End < exclusions[j].End
	})
	exclusions = coalesceExclusions(exclusions)

	// Count the admissible arithmetic progression exactly, without materializing
	// every candidate start. A zero-length exclusion removes no starts.
	admitted := positions
	for _, exclusion := range exclusions {
		removed := countStrideBefore(exclusion.End, opts.Stride) -
			countStrideBefore(exclusion.Start, opts.Stride)
		admitted -= uint64(removed)
	}
	result := &SearchResult{CandidatesTotal: admitted, Hits: make([]SearchHit, 0)}
	if admitted == 0 {
		result.Complete = true
		return result, nil
	}

	workers := opts.Workers
	if uint64(workers) > admitted {
		workers = int(admitted)
	}
	hitCap := opts.TopK
	if uint64(hitCap) > admitted {
		hitCap = int(admitted)
	}
	plan := searchPlan{
		length: len(query), stride: opts.Stride, radius: opts.Radius,
		positions: positions, admitted: admitted, workers: workers,
		hitCap: hitCap, width: bandWidth(len(query), opts.Radius),
		excluded: exclusions,
	}

	baseBytes, err := estimateSearchBytes(plan, opts.Normalization, exclusionBytes, opts.MaxWorkingBytes)
	if err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}

	// Raw inputs remain borrowed slices; z-mode owns a separate query buffer.
	preparedQuery := query
	if opts.Normalization == SearchZPopulation {
		preparedQuery = make([]float64, len(query))
		if err = normalizeZPopulationInto(preparedQuery, query); err != nil {
			return nil, fmt.Errorf("dtw: query: %w", err)
		}
	}
	// The optional envelope is prepared only after normalization, ensuring the
	// lower bound is measured in exactly the same units as the squared scorer.
	// A tight budget disables this optimization, not the exact alignment.
	plan.envelope = prepareSearchEnvelope(preparedQuery, opts.Radius, baseBytes, opts.MaxWorkingBytes)
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return runSearchWorkers(ctx, result, plan, preparedQuery, data, opts)
}

// checkedBytes multiplies two nonnegative sizes while detecting uint64 overflow.
// It is used for explicit backing-buffer accounting before allocation.
//
// Implementation:
//   - Stage 1: use 128-bit multiplication via bits.Mul64.
//   - Stage 2: return a low word and report whether any high bits were lost.
//
// Behavior highlights:
//   - No intermediate signed-int multiplication is performed.
//
// Inputs:
//   - count, size: nonnegative element count and bytes per element.
//
// Returns:
//   - uint64: product when representable; bool: true on overflow.
//
// Errors:
//   - None; caller maps overflow to ErrOverflow.
//
// Determinism:
//   - Exact integer arithmetic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Allocation still requires an executable-machine int-sized address space.
//
// AI-Hints:
//   - Never cast uint64 to int before checking the upper bound.
//   - Budget exhaustion is distinct from arithmetic size overflow.
func checkedBytes(count, size uint64) (uint64, bool) {
	hi, lo := bits.Mul64(count, size)
	return lo, hi != 0
}

// estimateSearchBytes accounts for all explicit backing buffers needed by exact workers.
// It rejects size overflows and a configured byte budget before the first candidate starts.
//
// Implementation:
//   - Stage 1: count exclusions, the top-K heap, and the worker result channel.
//   - Stage 2: include two classified rows and optional normalized vectors per worker.
//   - Stage 3: add with carry detection; compare the checked total to MaxWorkingBytes.
//
// Behavior highlights:
//   - The borrowed query/data arrays are not charged because Search does not copy them.
//
// Inputs:
//   - plan: validated dimensions; normalization: chosen policy; exclusions: copy byte count.
//   - limit: zero denotes unlimited explicitly allocated working bytes.
//
// Returns:
//   - uint64: required accounted backing bytes; error on size overflow/limit.
//
// Errors:
//   - ErrOverflow joined with ErrNaNInf: uint64 buffer-size arithmetic overflow.
//   - ErrBudget: minimum planned workspace exceeds configured limit.
//
// Determinism:
//   - Integer-only calculation independent of worker scheduling.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Runtime allocator headers, channels' internal headers, stacks, and scheduling are excluded.
//
// AI-Hints:
//   - Charge each worker its own scratch; a shared global scratch would race.
//   - Do not allocate an O(C) candidate-start list without adding it to this estimate.
func estimateSearchBytes(plan searchPlan, normalization SearchNormalization, exclusions, limit uint64) (uint64, error) {
	total := exclusions
	add := func(count, elementSize uint64) bool {
		bytes, overflow := checkedBytes(count, elementSize)
		if overflow {
			return false
		}
		var carry uint64
		total, carry = bits.Add64(total, bytes, 0)
		return carry == 0
	}

	// Never pre-multiply width*elementSize: either operand can be near
	// MaxInt even though the resulting byte product is not representable.
	rowsTotal, overflow := checkedBytes(uint64(plan.workers), 2)
	if overflow {
		return 0, errors.Join(ErrOverflow, ErrNaNInf)
	}
	rowElements, overflow := checkedBytes(rowsTotal, uint64(plan.width))
	if overflow {
		return 0, errors.Join(ErrOverflow, ErrNaNInf)
	}
	if !add(uint64(plan.hitCap), uint64(unsafe.Sizeof(SearchHit{}))) ||
		!add(uint64(plan.workers), uint64(unsafe.Sizeof(candidateOutcome{}))) ||
		!add(rowElements, uint64(unsafe.Sizeof(dpCell{}))) {
		return 0, errors.Join(ErrOverflow, ErrNaNInf)
	}
	if normalization == SearchZPopulation {
		vectorElements, vectorOverflow := checkedBytes(uint64(plan.workers)+1, uint64(plan.length))
		if vectorOverflow || !add(vectorElements, uint64(unsafe.Sizeof(float64(0)))) {
			return 0, errors.Join(ErrOverflow, ErrNaNInf)
		}
	}
	if limit > 0 && total > limit {
		return 0, ErrBudget
	}
	if total > uint64(maxIntValue) {
		return 0, errors.Join(ErrOverflow, ErrNaNInf)
	}
	return total, nil
}

// countStrideBefore counts multiples of a positive stride in [0,end).
// It allows exact excluded-start counts without enumerating a long dataset.
//
// Implementation:
//   - Stage 1: treat an empty domain as zero candidates.
//   - Stage 2: perform ceiling division with a remainder test, avoiding end+stride overflow.
//
// Returns:
//   - int: number of starts in the half-open prefix.
//
// Inputs:
//   - end: nonnegative exclusive bound; stride: positive integer.
//
// Errors:
//   - None after options validation.
//
// Determinism:
//   - Exact integer arithmetic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Ranges are first validated against [0,N-L+1).
//
// AI-Hints:
//   - Half-open [0,end) includes zero when end is positive.
//   - Avoid (end+stride-1)/stride: it may overflow int.
func countStrideBefore(end, stride int) int {
	if end == 0 {
		return 0
	}
	return 1 + (end-1)/stride
}

// coalesceExclusions merges sorted nonempty half-open start ranges in-place.
// The input is already a detached copy, so caller-owned ranges remain untouched.
//
// Implementation:
//   - Stage 1: ignore empty ranges without changing their validation outcome.
//   - Stage 2: merge overlapping or adjacent ranges in sorted order.
//   - Stage 3: return the compact prefix for indexed admission checks.
//
// Returns:
//   - []IndexRange: merged view of a privately owned array.
//
// Inputs:
//   - ranges: mutable detached copy, sorted Start then End.
//
// Errors:
//   - None after shape validation.
//
// Determinism:
//   - Canonical ascending range sequence.
//
// Complexity:
//   - Time O(R), Space O(1) extra.
//
// Notes:
//   - Sorting is performed once by Search.
//
// AI-Hints:
//   - Do not mutate ExcludedStarts directly.
//   - Adjacent ranges coalesce because the union has no missing integer start.
func coalesceExclusions(ranges []IndexRange) []IndexRange {
	write := 0
	for _, next := range ranges {
		if next.Start == next.End {
			continue
		}
		if write != 0 && next.Start <= ranges[write-1].End {
			if next.End > ranges[write-1].End {
				ranges[write-1].End = next.End
			}
			continue
		}
		ranges[write] = next
		write++
	}
	return ranges[:write]
}

// excludedStart reports whether one admitted arithmetic-progression start is excluded.
// It uses binary search over normalized half-open ranges instead of linear scans per worker.
//
// Implementation:
//   - Stage 1: find the first interval whose End exceeds start.
//   - Stage 2: test whether that interval's Start is <=start.
//
// Returns:
//   - bool: true if start is excluded.
//
// Inputs:
//   - ranges: sorted disjoint half-open intervals; start: nonnegative candidate start.
//
// Errors:
//   - None after preprocessing.
//
// Determinism:
//   - Pure binary search.
//
// Complexity:
//   - Time O(log R), Space O(1).
//
// Notes:
//   - Empty exclusions never reach this function.
//
// AI-Hints:
//   - Start is excluded only when Start<=start<End.
//   - Do not apply interval overlap tests to a hit's full sample window.
func excludedStart(ranges []IndexRange, start int) bool {
	at := sort.Search(len(ranges), func(i int) bool { return ranges[i].End > start })
	return at < len(ranges) && start >= ranges[at].Start
}

// isExactlyConstant detects an all-equal finite sequence without a floating tolerance.
// Constant vectors cannot be population z-normalized because their variance is zero.
//
// Implementation:
//   - Stage 1: take the first validated finite value as reference.
//   - Stage 2: stop at the first non-equal binary64 value.
//
// Returns:
//   - bool: true for lengths <=1 or exactly equal samples.
//
// Inputs:
//   - values: finite scalar series, never mutated.
//
// Errors:
//   - None.
//
// Determinism:
//   - Increasing-index exact comparisons.
//
// Complexity:
//   - Time O(L), Space O(1).
//
// Notes:
//   - No epsilon-based conversion of nearly constant signals to constants.
//
// AI-Hints:
//   - Do not classify a tiny-but-nonzero variance as zero by arbitrary tolerance.
//   - Nonfinite checks must precede this predicate.
func isExactlyConstant(values []float64) bool {
	if len(values) < 2 {
		return true
	}
	for _, value := range values[1:] {
		if value != values[0] {
			return false
		}
	}
	return true
}

// normalizeZPopulationInto performs stable population z-normalization on one complete window.
// It first shifts by a reference sample and scales differences, reducing catastrophic
// cancellation and intermediate square overflow without introducing a variance epsilon.
//
// Implementation:
//   - Stage 1: detect exact constants; find finite differences from the first value.
//   - Stage 2: when a difference overflows, rescale before subtracting large coordinates.
//   - Stage 3: form bounded centered differences, compute Welford population variance,
//     and convert every sample to a finite z-score in its caller-provided buffer.
//
// Behavior highlights:
//   - Population variance divides M2 by L, not by L-1.
//   - Normalization never mutates the borrowed window.
//
// Inputs:
//   - dst: a dedicated len(values) scratch buffer; values: finite nonempty float64 samples.
//
// Returns:
//   - error: nil with finite z values; destination is unspecified after an error.
//
// Errors:
//   - ErrShape: wrong destination size; ErrZeroVariance: exactly constant window.
//   - ErrNonFinite joined with ErrNaNInf: nonrepresentable normalized arithmetic.
//
// Determinism:
//   - Fixed forward-index Welford traversal and forward publication.
//
// Complexity:
//   - Time O(L), Space O(1) extra beyond dst's O(L) buffer.
//
// Notes:
//   - This is a floating binary64 normalization, not a symbolic exact-real transform.
//
// AI-Hints:
//   - Do not use naive sum(x*x) or sum(x)^2 to estimate variance at large magnitudes.
//   - Do not silently discard constant candidate windows; propagate their start-indexed error.
func normalizeZPopulationInto(dst, values []float64) error {
	if len(dst) != len(values) || len(values) == 0 {
		return ErrShape
	}
	if isExactlyConstant(values) {
		return ErrZeroVariance
	}

	anchor := values[0]
	maxAbsInput, maxAbsDelta := 0.0, 0.0
	rescale := false
	for _, value := range values {
		if absolute := math.Abs(value); absolute > maxAbsInput {
			maxAbsInput = absolute
		}
		difference := value - anchor
		if math.IsInf(difference, 0) {
			rescale = true
		} else if absolute := math.Abs(difference); absolute > maxAbsDelta {
			maxAbsDelta = absolute
		}
	}
	if rescale {
		maxAbsDelta = 0
		for _, value := range values {
			difference := value/maxAbsInput - anchor/maxAbsInput
			if absolute := math.Abs(difference); absolute > maxAbsDelta {
				maxAbsDelta = absolute
			}
		}
	}
	if maxAbsDelta == 0 {
		return errors.Join(ErrNonFinite, ErrNaNInf)
	}

	mean, sumSquaredDeviations := 0.0, 0.0
	for index, value := range values {
		difference := value - anchor
		if rescale {
			difference = value/maxAbsInput - anchor/maxAbsInput
		}
		scaled := difference / maxAbsDelta
		dst[index] = scaled
		change := scaled - mean
		mean += change / float64(index+1)
		sumSquaredDeviations += change * (scaled - mean)
	}
	standardDeviation := math.Sqrt(sumSquaredDeviations / float64(len(values)))
	if standardDeviation <= 0 || math.IsNaN(standardDeviation) || math.IsInf(standardDeviation, 0) {
		return errors.Join(ErrNonFinite, ErrNaNInf)
	}
	for index, scaled := range dst {
		normalized := (scaled - mean) / standardDeviation
		if math.IsNaN(normalized) || math.IsInf(normalized, 0) {
			return errors.Join(ErrNonFinite, ErrNaNInf)
		}
		dst[index] = normalized
	}
	return nil
}
