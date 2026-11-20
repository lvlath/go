// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax defines the public data carriers for an experimental DTW retrieval harness.
// This file fixes identity, ownership, result semantics and the supported stress families.
package tfsax

// CandidateID identifies a corpus trajectory independently of its slice position.
// Numeric IDs are unique within one frozen candidate universe and break distance ties.
//
// Implementation:
//   - Validation rejects repeated IDs rather than silently overwriting a candidate.
//   - Retrieval sorts equal-distance candidates by increasing numeric ID.
//
// Behavior highlights:
//   - Zero is a valid ID; IDs are not required to be consecutive or monotonically assigned.
//
// Inputs:
//   - Assigned by corpus owner; zero is legal and duplicate IDs are rejected.
//
// Returns:
//   - Identifies selected windows in RetrievalResult.Candidates without exposing their backing slices.
//
// Errors:
//   - ValidateCorpus and experiment membership validators return ErrDuplicateID on repeated identifiers.
//
// Determinism:
//   - ID comparison is unsigned numeric comparison, never input or map iteration order.
//
// Complexity:
//   - Comparisons use O(1) time and space.
//
// Notes:
//   - An ID identifies an immutable-for-the-call window; it is not an array offset.
//
// AI-Hints:
//   - Never use iteration order as a tie-breaker or reinterpret IDs as row indices.
type CandidateID uint64

// Query requests exact-DTW nearest neighbors for a caller-owned scalar window.
// Values is read-only during a search, and TopK specifies the maximum result size.
//
// Implementation:
//   - The facade validates a non-nil finite sequence and positive TopK.
//   - The kernel evaluates a shared candidate universe with one DTW configuration.
//
// Behavior highlights:
//   - TopK may exceed the reachable corpus size; the result is then shorter.
//
// Inputs:
//   - Values: non-nil, non-empty finite samples in corpus units, owned by the caller.
//   - TopK: number of requested neighbors, >=1.
//
// Returns:
//   - Search methods publish detached CandidateID slices, not slices of Values.
//
// Errors:
//   - ErrInvalidInput for an empty window, invalid dimensions or TopK.
//   - ErrNonFinite for NaN or infinite samples.
//
// Determinism:
//   - Query order never changes candidate tie precedence.
//
// Complexity:
//   - Validation O(len(Values)) time and O(1) additional memory.
//
// Notes:
//   - Concurrent mutation of Values while a search reads it is unsupported.
//
// AI-Hints:
//   - Do not use TopK as a shortlist size: the two control different operations.
type Query struct {
	Values []float64 // Scalar samples, read-only for the duration of an operation.
	TopK   int       // Positive requested count; an exact result can be shorter.
}

// SearchDiagnostics separates work counters from mathematically justified completeness.
// Exact means no reachable exact TopK candidate was lost; it does not mean the run finished.
//
// Implementation:
//   - Examined counts full DTW refinement calls, including structurally unreachable outcomes.
//   - Pruned counts candidates excluded before DTW by a heuristic or certified filter.
//
// Behavior highlights:
//   - Exact is true for complete full scans and must stay false for heuristic shortlists.
//   - RepresentationVersion identifies the scoring policy, not an algorithmic proof.
//
// Inputs:
//   - Published by exact or heuristic retrieval kernels; callers must not fabricate their proof flag.
//
// Returns:
//   - The fields belong to the detached RetrievalResult artifact.
//
// Errors:
//   - A failed search returns nil instead of partially meaningful counters.
//
// Determinism:
//   - Counters depend on the fixed shortlist and corpus, not timing or scheduling.
//
// Complexity:
//   - Storage O(1) excluding the version string.
//
// Notes:
//   - Pruned does not imply mathematically safe exclusion.
//
// AI-Hints:
//   - Never set Exact=true based solely on empirical recall or a JSON approval flag.
type SearchDiagnostics struct {
	Examined              int    // Number of full DTW evaluations actually attempted.
	Pruned                int    // Universe members not passed to full DTW.
	Exact                 bool   // Certified mathematical equivalence to exact TopK.
	RepresentationVersion string // Version of exact or heuristic representation.
}

// RetrievalResult publishes ordered, caller-owned nearest-neighbor IDs.
// Complete describes successful completion of the requested procedure; Exact is independent.
//
// Implementation:
//   - The search sorts reachable DTW distances, then selects up to TopK IDs.
//   - It publishes a new Candidates allocation without retaining output ownership.
//
// Behavior highlights:
//   - An empty Candidates slice is allowed when no candidate is reachable.
//   - No usable partial result is returned on validation, cancellation or numeric failure.
//
// Inputs:
//   - Produced by successful search methods; TopK is interpreted relative to the complete frozen corpus.
//
// Returns:
//   - Candidates is detached; caller mutation cannot change corpus or future searches.
//
// Errors:
//   - Error results publish nil; an empty successful Candidates slice is not itself an error.
//
// Determinism:
//   - Candidates order is ascending (exact DTW distance, numeric CandidateID).
//
// Complexity:
//   - Result storage is O(min(TopK, reachable candidates)).
//
// Notes:
//   - A zero-value non-nil result is not evidence that a search has run.
//
// AI-Hints:
//   - Check Complete and Diagnostics.Exact separately; neither implies the other.
type RetrievalResult struct {
	Candidates  []CandidateID     // Detached TopK candidate identifiers.
	Complete    bool              // Requested procedure terminated successfully.
	Diagnostics SearchDiagnostics // Actual DTW evaluations and exactness policy.
}

// IsNil reports only whether the RetrievalResult receiver is a nil pointer.
// A non-nil zero-value result remains distinguishable from an absent result.
//
// Implementation:
//   - Compare the pointer with nil without dereferencing it.
//
// Behavior highlights:
//   - Safe through an interface holding a typed nil *RetrievalResult.
//
// Inputs:
//   - r: possibly typed-nil *RetrievalResult receiver; safe without dereference.
//
// Returns:
//   - bool: true exactly when r == nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pointer identity has no dependence on algorithm state.
//
// Complexity:
//   - Time O(1), space O(1).
//
// Notes:
//   - Conforms to the core.Nilable method shape without requiring reflection.
//
// AI-Hints:
//   - Do not equate len(Candidates)==0 with a nil receiver or a failed scan.
func (r *RetrievalResult) IsNil() bool { return r == nil }

// Candidate couples a stable ID to one read-only scalar corpus window.
// A successful search does not alter Values or retain aliases in its result.
//
// Implementation:
//   - Corpus validators require uniform nonempty finite Values and distinct IDs before scoring.
//
// Behavior highlights:
//   - CandidateIDs must be unique; all candidate lengths in a corpus must agree.
//
// Inputs:
//   - ID: identifier preserved across ordering and serialization.
//   - Values: non-nil finite sequence, read-only while consumed.
//
// Returns:
//   - PreparedSAXIndex makes a deep copy; one-shot scans borrow input during the call.
//
// Errors:
//   - Invalid or duplicate candidate values/IDs are reported by corpus validation before search.
//
// Determinism:
//   - Original corpus index controls scan order; CandidateID alone resolves score ties.
//
// Complexity:
//   - The carrier itself is O(1) space; the referenced samples occupy O(n).
//
// Notes:
//   - One-shot scans borrow caller data for their call; PrepareSAXIndex privately clones it.
//
// AI-Hints:
//   - Changing Values after computing CorpusHash invalidates the frozen universe.
type Candidate struct {
	ID     CandidateID // Stable unique identity, not the corpus position.
	Values []float64   // Caller-owned scalar window.
}

// NamedQuery binds an immutable experiment query identity to its source family.
// The ID allows replay and per-family reporting independently of input order.
//
// Implementation:
//   - Experiment validation sorts named queries by ID and validates each finite window before measurement.
//
// Behavior highlights:
//   - Duplicate query IDs are rejected before executing any branch.
//
// Inputs:
//   - ID: unique stable query identifier, including zero.
//   - Query: caller-owned read-only query window and TopK.
//   - Family: one declared experiment stress regime.
//
// Returns:
//   - Observation.QueryID preserves the stable ID and Observation.Family preserves the declared stress regime.
//
// Errors:
//   - Repeated IDs yield ErrDuplicateID; unrecognized families or bad shapes are rejected.
//
// Determinism:
//   - Experiment runs publish observations in increasing query-ID order.
//
// Complexity:
//   - O(1) carrier space beyond the query's samples.
//
// Notes:
//   - ID and Family are audit metadata; the Query.Values samples must come from a separately tracked source.
//
// AI-Hints:
//   - Never use a timing-dependent generated ID or map order for query identity.
type NamedQuery struct {
	ID     uint64 // Stable experiment ID.
	Query  Query  // Query inputs in original physical units.
	Family Family // Predeclared synthetic or audit regime.
}

// Family identifies one deterministic synthetic stress regime.
// Each family must be reported separately rather than hidden in global means.
//
// Implementation:
//   - GenerateFamily maps each named family to one deterministic RNG stream domain and synthesis law.
//
// Behavior highlights:
//   - The value is a stable research label, not a learned class or diagnostic prediction.
//
// Inputs:
//   - Use the exported Family constants; arbitrary names are rejected at generation/experiment admission.
//
// Returns:
//   - Carries the declared synthetic stress regime on Candidate queries and observations.
//
// Errors:
//   - Unknown family values produce ErrInvalidOption from generation or ErrInvalidInput from experiment validation.
//
// Determinism:
//   - String labels are stable serialized identifiers; samples use independent RNG paths.
//
// Complexity:
//   - O(1) comparison and labeling cost.
//
// Notes:
//   - Families use dimensionless illustrative series and cannot substantiate empirical real-world claims.
//
// AI-Hints:
//   - Report the least favorable family; do not optimize to a pooled average.
type Family string

const (
	// FamilyConstant generates exact constant windows and abrupt level changes.
	// Repeated constant values deliberately exercise tie-breaking and degeneracy.
	FamilyConstant Family = "constant-step"
	// FamilyHeavyTail adds rare large perturbations to a harmonic baseline.
	// Independent RNG draws test ranking sensitivity to isolated high-amplitude outliers.
	FamilyHeavyTail Family = "heavy-tail"
	// FamilyWarp applies smooth monotone remapping of sample time.
	// The transform fixes endpoints while changing local speed and adding small noise.
	FamilyWarp Family = "local-warp"
	// FamilySlope creates piecewise-linear trend breaks.
	// The slope and reversal are fixed per generated trajectory, not fitted from evaluation.
	FamilySlope Family = "slope-break"
	// FamilyMotif repeats a cyclic waveform under nuisance noise.
	// Per-path cyclic phase shifts and independent draws challenge ordinary SAX ranking.
	FamilyMotif Family = "motif-distractor"
	// FamilyPhase shifts a harmonic waveform in time.
	// Phase is derived from the candidate ID, making this a deterministic grid stressor.
	FamilyPhase Family = "phase-shift"
	// FamilyAffine applies positive global scaling and offset.
	// Its deterministic ID-based coefficients probe normalization sensitivity.
	FamilyAffine Family = "affine"
	// FamilyPlateau inserts a non-informative level segment in a time series.
	// The segment is exactly flat while surrounding samples vary independently.
	FamilyPlateau Family = "plateau"
)

// MetricValue encodes a numerical measurement or an explicit not-applicable state.
// Applicable=false means the numeric Value must not be interpreted as a zero score.
//
// Implementation:
//   - Metric functions compute Value only when a nonempty exact truth denominator exists.
//
// Behavior highlights:
//   - RecallAtK and MissedBest return NA when exact truth contains no reachable ID.
//
// Inputs:
//   - Published by RecallAtK or MissedBest from strict CandidateID comparisons.
//
// Returns:
//   - Value in [0,1] only when Applicable is true for those metrics.
//
// Errors:
//   - No error is returned; Applicable=false encodes an undefined measurement, not a zero.
//
// Determinism:
//   - Uses strict ID membership, independent of iteration order.
//
// Complexity:
//   - Time O(1) to inspect a metric.
//
// Notes:
//   - Do not divide by a zero truth count or replace NA with an invented perfect recall.
//
// AI-Hints:
//   - Never convert NA to 1, 0, or a hidden denominator in report aggregates.
type MetricValue struct {
	Value      float64 `json:"value"`      // Defined measurement; undefined when Applicable=false.
	Applicable bool    `json:"applicable"` // True only when the target denominator is meaningful.
}

// Observation records one query/branch pairing and its independent measurements.
// Timings are observations, not deterministic identities or a mathematical speed guarantee.
//
// Implementation:
//   - Search runners attach the actual branch result, per-query ground-truth metrics and timing metadata.
//
// Behavior highlights:
//   - Available is the number of reachable corpus members for exact scan.
//   - Shortlist is the requested full-refinement count, not always TopK.
//
// Inputs:
//   - Branch: a named existing implementation, never an imaginary TFSAX encoder.
//
// Returns:
//   - Result.Candidates is detached; callers own the published observation slice.
//
// Errors:
//   - The runner rejects a failed query without releasing partial observation batches.
//
// Determinism:
//   - QueryID and result order are stable; DurationNS can vary between executions.
//
// Complexity:
//   - O(1) fields plus a detached result slice.
//
// Notes:
//   - DurationNS is nonreproducible; output identity hashes should exclude performance measurements.
//
// AI-Hints:
//   - Never incorporate DurationNS into deterministic experiment identity hashes.
type Observation struct {
	QueryID    uint64          `json:"query_id"`    // Stable per-query identifier.
	Family     Family          `json:"family"`      // Predeclared stress family.
	Branch     string          `json:"branch"`      // Executed comparator name.
	Result     RetrievalResult `json:"result"`      // Detached retrieval artifact.
	Recall     MetricValue     `json:"recall"`      // Exact-ID recall relative to exact TopK.
	MissedBest MetricValue     `json:"missed_best"` // Whether exact best is missing from this result.
	Available  int             `json:"available"`   // Reachable exact corpus candidates.
	Shortlist  int             `json:"shortlist"`   // DTW refinement budget for this arm.
	DurationNS int64           `json:"duration_ns"` // Elapsed nanoseconds; inherently noisy.
	IndexBytes uint64          `json:"index_bytes"` // Defined representation payload bytes, not RSS.
}

// MeasurementSummary contains nearest-rank latency percentiles.
// Quantiles are computed from explicit nonnegative duration observations.
//
// Implementation:
//   - LatencyQuantiles copies and orders nonnegative samples before selecting nearest ranks.
//
// Behavior highlights:
//   - P50, P95 and P99 are all measured in nanoseconds.
//
// Inputs:
//   - Produced from at least one nonnegative duration observation in nanoseconds.
//
// Returns:
//   - The three percentile fields in integer nanoseconds with no interpolation.
//
// Errors:
//   - LatencyQuantiles returns ErrInvalidInput for zero samples or a negative observation.
//
// Determinism:
//   - Sorted copy and nearest-rank indexing; no interpolation or input mutation.
//
// Complexity:
//   - O(1) result size.
//
// Notes:
//   - Measurements from different hardware or warmup regimes should not be pooled blindly.
//
// AI-Hints:
//   - Compare separately measured warm/cold workloads; quantiles alone imply no speedup.
type MeasurementSummary struct {
	P50 int64 // Nearest-rank median duration in nanoseconds.
	P95 int64 // Nearest-rank 95th percentile in nanoseconds.
	P99 int64 // Nearest-rank 99th percentile in nanoseconds.
}
