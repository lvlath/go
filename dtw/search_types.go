// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw defines explicit bounded-alignment outcomes and deterministic
// fixed-length subsequence search results. This file owns the public data contracts.
package dtw

// BoundStatus classifies a bounded alignment without confusing a cutoff with an exact distance.
// A successful alignment is either exact, certified above the limit, or structurally unreachable.
//
// Implementation:
//   - Stage 1: classify the terminal state of a validated alignment.
//   - Stage 2: compare its finite distance with the caller's inclusive limit.
//   - Stage 3: publish the corresponding BoundedResult representation.
//
// Behavior highlights:
//   - BoundAboveLimit is not an approximate distance; the exact value is intentionally hidden.
//
// Inputs:
//   - Used as the Status field of BoundedResult.
//
// Returns:
//   - One of the named constant classifications.
//
// Errors:
//   - Invalid input/overflow is returned as error, never encoded as a status.
//
// Determinism:
//   - Equal distance and limit always yield BoundExact.
//
// Complexity:
//   - Time O(1), Space O(1) for classification.
//
// Notes:
//   - A future early-abandonment implementation must preserve this status protocol.
//
// AI-Hints:
//   - Never publish an unverified finite distance for BoundAboveLimit.
//   - Keep BoundUnreachable distinct from ErrOverflow.
type BoundStatus uint8

const (
	// BoundExact means that Distance and LowerBound are the same finite exact DTW score.
	BoundExact BoundStatus = iota
	// BoundAboveLimit means that Distance is NaN and LowerBound strictly exceeds the limit.
	BoundAboveLimit
	// BoundUnreachable means no admissible path exists; both distance fields are +Inf.
	BoundUnreachable
)

// BoundedResult describes the outcome of one inclusive-limit DTW alignment.
// It does not change the existing Result layout or leak unrequested local/DP artifacts.
//
// Implementation:
//   - Stage 1: resolve the exact/numeric/unreachable final state.
//   - Stage 2: derive the bound classification against the supplied limit.
//   - Stage 3: detach the optional optimal path and report visited admitted cells.
//
// Behavior highlights:
//   - Exact: Distance=LowerBound is finite; AboveLimit: Distance=NaN, LowerBound>limit.
//   - Unreachable: Distance=LowerBound=+Inf; overflow/cancellation yields nil+error.
//
// Inputs:
//   - Produced by AlignBounded; the limit uses accumulated DTW cost units.
//
// Returns:
//   - Caller-owned Path, no shared workspace or mutable input aliases.
//
// Errors:
//   - Errors are returned separately; no usable result is published on numeric failure.
//
// Determinism:
//   - Path coordinates follow the same strict predecessor policy as Align.
//
// Complexity:
//   - Space O(n+m) for Path when requested, O(1) otherwise, excluding the compute workspace.
//
// Notes:
//   - CellsVisited counts admitted local-cost cells computed by the current full kernel.
//
// AI-Hints:
//   - NaN in BoundAboveLimit is an intentionally unavailable exact objective, not bad input.
//   - Do not assume Path is available unless Status==BoundExact and tracking was requested.
type BoundedResult struct {
	// Status reports the exact, certified above-limit, or unreachable outcome.
	Status BoundStatus
	// Distance is finite only for BoundExact, NaN for BoundAboveLimit, or +Inf for BoundUnreachable.
	Distance float64
	// LowerBound equals the exact finite distance for BoundExact, exceeds limit for
	// BoundAboveLimit, and is +Inf for BoundUnreachable.
	LowerBound float64
	// Path is the detached canonical witness only for a path-requested BoundExact result.
	Path Path
	// CellsVisited counts every admissible local cell computed by the full legacy recurrence.
	CellsVisited uint64
}

// SearchNormalization chooses the explicit scalar preprocessing policy for exact window search.
// It fixes whether candidate and query samples are compared in raw or population z-score units.
//
// Implementation:
//   - Stage 1: validate the chosen named mode.
//   - Stage 2: normalize query and each candidate independently when requested.
//   - Stage 3: score the corresponding squared-difference DTW problem.
//
// Behavior highlights:
//   - SearchRaw never changes input values; SearchZPopulation rejects constant windows.
//
// Inputs:
//   - Used through SearchOptions.Normalization.
//
// Returns:
//   - The policy selected by the caller; zero denotes the raw mode.
//
// Errors:
//   - Unsupported values yield ErrInvalidOption before computation.
//
// Determinism:
//   - Population z-score uses fixed input order and float64 arithmetic.
//
// Complexity:
//   - Normalization costs O(L) time and O(L) candidate storage when enabled.
//
// Notes:
//   - Search's zero-value SearchOptions is invalid even though this enum has a zero value.
//
// AI-Hints:
//   - Never silently skip a constant candidate in z-mode.
//   - Do not normalize across query and data together; each window has its own moments.
type SearchNormalization uint8

const (
	// SearchRaw compares original finite scalar samples by squared local differences.
	SearchRaw SearchNormalization = iota
	// SearchZPopulation normalizes each complete window using population standard deviation.
	SearchZPopulation
)

// IndexRange denotes a half-open [Start,End) zero-based interval.
// Exclusion ranges apply to candidate *starts*, while hit ranges cover data samples.
//
// Implementation:
//   - Stage 1: validate non-negative ordered endpoints inside the relevant domain.
//   - Stage 2: sort and coalesce exclusion ranges without editing caller storage.
//   - Stage 3: test candidate start membership or publish a matched window.
//
// Behavior highlights:
//   - Empty [s,s) exclusions are legal and exclude no starts.
//
// Inputs:
//   - Start, End: element indices; End is not included.
//
// Returns:
//   - A value-only range, with no pointer aliasing.
//
// Errors:
//   - Invalid exclusion endpoints return ErrShape.
//
// Determinism:
//   - Exclusions are sorted by Start then End before coalescing.
//
// Complexity:
//   - O(1) storage per range.
//
// Notes:
//   - For query length L and data length N, valid start-domain is [0,N-L+1).
//
// AI-Hints:
//   - Do not confuse exclusion starts with overlapping sample-time intervals.
//   - Preserve half-open semantics for both exclusions and returned hit windows.
type IndexRange struct {
	// Start is the included zero-based start position of this range.
	Start int
	// End is the excluded zero-based end position of this range.
	End int
}

// SearchOptions configures exact fixed-length scalar squared-DTW window search.
// Zero-value SearchOptions is intentionally invalid; use DefaultSearchOptions.
//
// Implementation:
//   - Stage 1: validate Radius, TopK, Stride, Workers, normalization, and range endpoints.
//   - Stage 2: copy/coalesce excluded start ranges and compute a checked workspace budget.
//   - Stage 3: score or soundly prune admitted starts with independent worker workspaces.
//
// Behavior highlights:
//   - Radius=-1 is unconstrained; Radius=0 permits only diagonal matches.
//   - MaxCells/MaxWorkingBytes=0 disables the corresponding limit; workers own private rows.
//
// Inputs:
//   - TopK, Stride, Workers: positive integers; ExcludedStarts is borrowed, not mutated.
//   - MaxCells: upper bound of actually evaluated DP cells across all candidates.
//   - MaxWorkingBytes: upper bound of explicitly accounted working-buffer bytes.
//
// Returns:
//   - Settings used by Search; not retained as aliases inside SearchResult.
//
// Errors:
//   - ErrInvalidOption for unsupported scalar policy values; ErrShape for bad exclusions.
//
// Determinism:
//   - Candidate starts increase from zero by Stride before exclusion.
//
// Complexity:
//   - Exclusion preprocessing O(R log R) time, O(R) space for R ranges.
//
// Notes:
//   - Worker stacks, scheduler bookkeeping, and allocator metadata are not covered by the byte limit.
//
// AI-Hints:
//   - Do not treat Workers=0, TopK=0, or Stride=0 as default values.
//   - Avoid copying the entire data array or holding an unbounded candidate queue.
type SearchOptions struct {
	// Radius is -1 for no constraint, or a nonnegative Sakoe-Chiba half-width.
	Radius int
	// TopK is the positive maximum number of exact matches retained.
	TopK int
	// Stride is the positive increment between consecutive candidate starts.
	Stride int
	// Workers is the positive number of independent scoring workspaces requested.
	Workers int
	// Normalization is the raw or independently computed population-z scalar policy.
	Normalization SearchNormalization
	// ExcludedStarts holds half-open ranges of candidate starts and is copied before sorting.
	ExcludedStarts []IndexRange
	// MaxCells is a hard global count of computed DP cells; zero disables the limit.
	MaxCells uint64
	// MaxWorkingBytes limits explicitly accounted backing-buffer bytes; zero is unlimited.
	MaxWorkingBytes uint64
}

// SearchHit stores one completely scored fixed-length candidate window.
// SquaredDistance uses accumulated squared scalar local-cost units, never absolute L1 units.
//
// Implementation:
//   - Stage 1: finish a valid DTW candidate against the normalized-or-raw query.
//   - Stage 2: record its half-open data window and exact accumulated score.
//   - Stage 3: rank hits ascending by (SquaredDistance,Window.Start,Window.End).
//
// Behavior highlights:
//   - Only finite exact scores are published; errors and incomplete windows are never hits.
//
// Inputs:
//   - Window: [start,start+len(query)); SquaredDistance: finite >=0.
//
// Returns:
//   - A copyable value with no input aliasing.
//
// Errors:
//   - Numeric failures are returned by Search, not stored in SearchHit.
//
// Determinism:
//   - Equal scores prefer the smaller window start then end.
//
// Complexity:
//   - O(1) per hit; ranking K hits takes O(K log K).
//
// Notes:
//   - SearchHit is a new struct; it does not extend the legacy DTW Result.
//
// AI-Hints:
//   - Do not mix search results from different Normalization or Radius policies.
//   - Do not replace exact scores with lower bounds in Hits.
type SearchHit struct {
	// Window is the exact half-open [start,start+len(query)) interval of the candidate.
	Window IndexRange
	// SquaredDistance is the finite exact accumulated squared-local-cost DTW objective.
	SquaredDistance float64
}

// SearchResult reports completed candidate work and a deterministic exact top-K ranking.
// Complete=false means Hits rank only fully scored candidates, not the global search domain.
//
// Implementation:
//   - Stage 1: count admitted candidate starts after exclusions.
//   - Stage 2: collect exact completed candidates and any certified pruning decisions.
//   - Stage 3: sort detached Hits by score/start/end and publish completion counters.
//
// Behavior highlights:
//   - Complete=true guarantees CandidatesCompleted+CandidatesPruned==CandidatesTotal.
//   - Machine-certified envelope exclusions count toward CandidatesPruned; no uncertified score is skipped.
//
// Inputs:
//   - Produced by Search; all counts are exact uint64 values.
//
// Returns:
//   - Caller-owned Hits slice (no aliases to input data or worker workspaces).
//
// Errors:
//   - Search can return partial SearchResult plus ErrBudget, context error, or candidate error.
//
// Determinism:
//   - Complete rankings are worker-scheduling independent; partial rankings need not be.
//
// Complexity:
//   - Hits stores O(min(TopK,CandidatesTotal)) values.
//
// Notes:
//   - Partial Hits must not be presented as certified global top-K.
//
// AI-Hints:
//   - Inspect Complete before making domain-wide ranking decisions.
//   - CellsVisited includes work on interrupted candidates, not just completed ones.
type SearchResult struct {
	// Hits contains the detached sorted best K completed exact candidates.
	Hits []SearchHit
	// Complete means the Hits are a globally exact ranking over all admitted starts.
	Complete bool
	// CandidatesTotal counts all stride-aligned starts after exclusion.
	CandidatesTotal uint64
	// CandidatesCompleted counts fully computed finite exact candidate distances.
	CandidatesCompleted uint64
	// CandidatesPruned counts machine-certified strict-above-cutoff exclusions, not completed hits.
	CandidatesPruned uint64
	// CellsVisited counts all evaluated DP cells, including on interrupted candidates.
	CellsVisited uint64
}

// DefaultSearchOptions constructs the explicit default exact-window search policy.
// It returns values rather than shared mutable global configuration.
//
// Implementation:
//   - Stage 1: choose unconstrained radius, single best hit, stride one, one worker.
//   - Stage 2: leave normalization raw and exclusion ranges empty.
//   - Stage 3: disable both optional resource budgets.
//
// Behavior highlights:
//   - Window starts are enumerated at each possible position by default.
//
// Inputs:
//   - None.
//
// Returns:
//   - SearchOptions: independent caller-owned option value.
//
// Errors:
//   - None.
//
// Determinism:
//   - Always returns identical scalar defaults.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - SearchOptions{} is deliberately not the same policy.
//
// AI-Hints:
//   - Use this function before overriding only the needed search settings.
//   - In particular, Radius=0 explicitly requests a strict diagonal path.
func DefaultSearchOptions() SearchOptions {
	return SearchOptions{
		Radius:        -1,
		TopK:          1,
		Stride:        1,
		Workers:       1,
		Normalization: SearchRaw,
	}
}
