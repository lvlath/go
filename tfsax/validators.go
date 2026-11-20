// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax validates experiments, temporal splitting and numeric input boundaries.
// This file keeps all temporal and resource invariants separate from retrieval kernels.
package tfsax

import (
	"errors"
	"math"
)

const (
	maxSeriesLength     = 4096    // Hard cap on samples in a generated window or protocol.
	maxExperimentItems  = 100000  // Hard cap on candidates or queries, not a timing estimate.
	minProtocolLength   = 2       // At least two samples are required by the research protocol.
	maxGeneratorSamples = 1 << 25 // At most 33,554,432 generated scalar samples per call.
)

// Interval is a half-open, zero-based interval in raw-series sample coordinates.
// Start is included, End is excluded; windows and labels use the same time axis.
//
// Implementation:
//   - Validators compare half-open integer endpoints without converting to wall-clock durations.
//
// Behavior highlights:
//   - Start >= 0 and End > Start are required for any declared split or window.
//
// Inputs:
//   - Start, End: raw sample indices, not rounded timestamps or window IDs.
//
// Returns:
//   - Used as raw-time boundaries and full input support witnesses.
//
// Errors:
//   - Invalid nonpositive width or negative start yields ErrInvalidOption from validation.
//
// Determinism:
//   - Pure exact integer interval comparisons.
//
// Complexity:
//   - O(1) time and space.
//
// Notes:
//   - Use raw sample indices on a validated uniform grid, never arbitrary external timestamp labels.
//
// AI-Hints:
//   - Convert timestamps to one consistent integer grid before validation.
type Interval struct {
	Start int // First included raw sample index.
	End   int // First excluded raw sample index.
}

// SplitPlan commits to raw Train/Calibration/Test ranges before window slicing.
// FitIDs identifies every candidate used to fit thresholds, scaling or other parameters.
//
// Implementation:
//   - ValidateSplit checks gap geometry and disjoint ID sets.
//   - ValidateWindowRecords optionally checks per-window source support and label ends.
//
// Behavior highlights:
//   - Membership arrays may be empty, but no ID may occur in multiple splits.
//
// Inputs:
//   - Train, Calibration, Test: ordered raw intervals before sliding windows are created.
//   - FitIDs: unique IDs strictly contained in TrainIDs, never held-out IDs.
//
// Returns:
//   - The validator retains no slices, maps or prepared configuration.
//
// Errors:
//   - ErrLeakage for insufficient separation or invalid fitted provenance.
//   - ErrDuplicateID for repeated identities inside any membership list.
//
// Determinism:
//   - Classification is independent of the order of membership IDs.
//
// Complexity:
//   - Validation O(number of membership IDs) time and space.
//
// Notes:
//   - The value is read-only to validators; callers retain ownership and must freeze it before slicing windows.
//
// AI-Hints:
//   - Membership alone cannot prove support containment; also call ValidateWindowRecords.
type SplitPlan struct {
	Train, Calibration, Test                  Interval      // Raw-time ranges assigned before window construction.
	FitIDs, TrainIDs, CalibrationIDs, TestIDs []CandidateID // Unique IDs committed to each split or fitting.
}

// WindowSplit is the declared origin of one sliced sequence.
// It is checked against both the membership ID and raw-time support evidence.
//
// Implementation:
//   - ValidateWindowRecords compares the label against committed membership and raw-time parent partition.
//
// Behavior highlights:
//   - Only SplitTrain, SplitCalibration and SplitTest are accepted as record origins.
//
// Inputs:
//   - Supplied as WindowRecord.Split for an original raw-series support interval.
//
// Returns:
//   - A stable partition identity, not a fitted model or label value.
//
// Errors:
//   - Unknown labels cause ErrInvalidOption; mismatched committed labels cause ErrLeakage.
//
// Determinism:
//   - String values are stable serialization labels.
//
// Complexity:
//   - O(1) comparison and storage.
//
// Notes:
//   - A split name cannot authenticate actual source timestamps by itself.
//
// AI-Hints:
//   - A split label alone does not authorize a held-out ID for training.
type WindowSplit string

const (
	// SplitTrain identifies windows eligible to fit scaling and thresholds.
	SplitTrain WindowSplit = "train"
	// SplitCalibration identifies holdout windows excluded from parameter fitting.
	SplitCalibration WindowSplit = "calibration"
	// SplitTest identifies final-evaluation windows excluded from fitting and tuning.
	SplitTest WindowSplit = "test"
)

// WindowRecord proves raw sample support and label horizon for one window ID.
// Records are validated against a frozen SplitPlan before any data fitting.
//
// Implementation:
//   - ValidateWindowRecords checks ID uniqueness, exact support width, partition membership and LabelEnd.
//
// Behavior highlights:
//   - Support.End-Support.Start must equal the declared window width.
//   - LabelEnd equals Support.End+horizon and must remain in the assigned split.
//
// Inputs:
//   - ID: unique member of the corresponding split's ID list.
//   - Split: exact train, calibration or test assignment.
//   - Support: raw half-open interval containing every feature input sample.
//   - LabelEnd: first excluded label sample, or Support.End when horizon==0.
//
// Returns:
//   - One evidence record per declared train/calibration/test candidate ID.
//
// Errors:
//   - ErrLeakage when a witness crosses its assigned raw interval or contradicts membership.
//
// Determinism:
//   - Each input record is checked once; missing and duplicated records are errors.
//
// Complexity:
//   - O(1) value storage, validation O(number of records).
//
// Notes:
//   - The record does not copy samples or authenticate an upstream loader that lies about support.
//
// AI-Hints:
//   - Do not record just the window start; full support and label end are required.
type WindowRecord struct {
	ID       CandidateID // Stable identity of the materialized window.
	Split    WindowSplit // Declared data partition of the window.
	Support  Interval    // Full input support, inclusive start and exclusive end.
	LabelEnd int         // First sample after optional outcome/label support.
}

func validInterval(x Interval) bool { return x.Start >= 0 && x.End > x.Start }

// separated tests integer raw-time separation without unsigned underflow.
// The caller already validated gap >=0; no subtraction occurs for overlapping inputs.
func separated(left, right Interval, gap int) bool {
	return left.End <= right.Start && right.Start-left.End >= gap
}

// membership rejects identity collisions without depending on map iteration order.
// It owns a fresh set and leaves the supplied ID slice unchanged.
func membership(ids []CandidateID) (map[CandidateID]struct{}, error) {
	m := make(map[CandidateID]struct{}, len(ids))
	for _, id := range ids {
		if _, ok := m[id]; ok {
			return nil, ErrDuplicateID
		}
		m[id] = struct{}{}
	}
	return m, nil
}

// ValidateSplit enforces raw-time gap geometry and disjoint train-only fitting.
// It rejects leakage before any window materialization or learned preprocessing.
//
// Implementation:
//   - Stage 1: verify half-open raw intervals and overflow-safe gap parameters.
//   - Stage 2: enforce purge + horizon + embargo between consecutive partitions.
//   - Stage 3: reject duplicate/cross-split IDs and fit IDs outside TrainIDs.
//
// Behavior highlights:
//   - Purge must cover at least the full input window width.
//   - Horizon and embargo are additional protected raw-sample distances.
//
// Inputs:
//   - p: read-only split boundaries and ID provenance.
//   - window: feature support width in raw samples, >=1.
//   - horizon: label lookahead beyond input support, >=0.
//   - purge: raw-sample gap budget, >=window.
//   - embargo: additional raw-sample exclusion after purge and horizon, >=0.
//
// Returns:
//   - nil only for a geometrically isolated membership plan; no owned state retained.
//
// Errors:
//   - ErrInvalidOption for malformed intervals, gap policy or integer overflow.
//   - ErrLeakage for insufficient separation or invalid fit/cross-split membership.
//   - ErrDuplicateID for repeats within any list.
//
// Determinism:
//   - Checks membership sets without publishing their iteration order.
//
// Complexity:
//   - Time O(T+C+E+F), space O(T+C+E+F), measured in IDs.
//
// Notes:
//   - Without WindowRecords this validates declarations, not the physical origin of data.
//
// AI-Hints:
//   - Split raw history before making windows; never assign overlapping windows retroactively.
func ValidateSplit(p SplitPlan, window, horizon, purge, embargo int) error {
	if window < 1 || horizon < 0 || purge < window || embargo < 0 ||
		!validInterval(p.Train) || !validInterval(p.Calibration) || !validInterval(p.Test) {
		return ErrInvalidOption
	}

	// Compute the gap only after checking every addition for integer overflow.
	maxInt := int(^uint(0) >> 1)
	if purge > maxInt-horizon || purge+horizon > maxInt-embargo {
		return ErrInvalidOption
	}
	gap := purge + horizon + embargo
	if !separated(p.Train, p.Calibration, gap) || !separated(p.Calibration, p.Test, gap) {
		return ErrLeakage
	}

	train, err := membership(p.TrainIDs)
	if err != nil {
		return err
	}
	calibration, err := membership(p.CalibrationIDs)
	if err != nil {
		return err
	}
	test, err := membership(p.TestIDs)
	if err != nil {
		return err
	}
	for id := range train {
		if _, collision := calibration[id]; collision {
			return ErrLeakage
		}
		if _, collision := test[id]; collision {
			return ErrLeakage
		}
	}
	for id := range calibration {
		if _, collision := test[id]; collision {
			return ErrLeakage
		}
	}
	fit, err := membership(p.FitIDs)
	if err != nil {
		return err
	}
	for id := range fit {
		if _, trained := train[id]; !trained {
			return ErrLeakage
		}
	}
	return nil
}

// ValidateWindowRecords verifies that each declared membership has one physical witness.
// It prevents a window or its label support from crossing a raw temporal partition.
//
// Implementation:
//   - Stage 1: validate the split plan and collect all committed ID memberships.
//   - Stage 2: check each record's split, raw support width and exact label endpoint.
//   - Stage 3: require one and only one physical witness per declared member.
//
// Behavior highlights:
//   - A test ID carrying train-labelled samples is rejected even when strings look valid.
//
// Inputs:
//   - plan: raw split and fit provenance; read-only.
//   - records: full source evidence; one record per declared train/calibration/test ID.
//   - window, horizon, purge, embargo: exactly the ValidateSplit gap policy.
//
// Returns:
//   - nil only for complete validated evidence; no aliases are retained.
//
// Errors:
//   - ErrInvalidOption for malformed policy or record geometry.
//   - ErrInvalidInput for a missing or unlisted physical witness.
//   - ErrDuplicateID for multiple records of one window ID.
//   - ErrLeakage for origin, label-horizon or membership contradictions.
//
// Determinism:
//   - Record checks are performed in input order; ID set enumeration is not observable.
//
// Complexity:
//   - Time O(T+C+E+F+W), space O(T+C+E+W) for membership and records.
//
// Notes:
//   - Does not inspect values: raw data importers must bind the recorded support to the samples.
//
// AI-Hints:
//   - Require this audit for learned breakpoints even if ValidateSplit alone succeeds.
func ValidateWindowRecords(plan SplitPlan, records []WindowRecord, window, horizon, purge, embargo int) error {
	if err := ValidateSplit(plan, window, horizon, purge, embargo); err != nil {
		return err
	}
	declared := make(map[CandidateID]WindowSplit, len(plan.TrainIDs)+len(plan.CalibrationIDs)+len(plan.TestIDs))
	for _, id := range plan.TrainIDs {
		declared[id] = SplitTrain
	}
	for _, id := range plan.CalibrationIDs {
		declared[id] = SplitCalibration
	}
	for _, id := range plan.TestIDs {
		declared[id] = SplitTest
	}
	if len(records) != len(declared) {
		return ErrInvalidInput
	}
	seen := make(map[CandidateID]struct{}, len(records))
	maxInt := int(^uint(0) >> 1)
	for _, record := range records {
		if _, repeated := seen[record.ID]; repeated {
			return ErrDuplicateID
		}
		seen[record.ID] = struct{}{}
		expectedSplit, listed := declared[record.ID]
		if !listed {
			return ErrInvalidInput
		}
		if record.Split != expectedSplit {
			return ErrLeakage
		}
		if !validInterval(record.Support) || record.Support.End-record.Support.Start != window {
			return ErrInvalidOption
		}
		if record.Support.End > maxInt-horizon || record.LabelEnd != record.Support.End+horizon {
			return ErrInvalidOption
		}
		var parent Interval
		switch record.Split {
		case SplitTrain:
			parent = plan.Train
		case SplitCalibration:
			parent = plan.Calibration
		case SplitTest:
			parent = plan.Test
		default:
			return ErrInvalidOption
		}
		if record.Support.Start < parent.Start || record.LabelEnd > parent.End {
			return ErrLeakage
		}
	}
	return nil
}

// ValidateProtocol checks declared experiment dimensions and budget before execution.
// It rejects malformed policy or impossible cell counts without allocating a dataset.
//
// Implementation:
//   - Stage 1: check version, counts, dimensions, window and temporal gap parameters.
//   - Stage 2: verify finite precommitted thresholds and valid metric ranges.
//   - Stage 3: compute bounded work using division-before-multiplication guards.
//
// Behavior highlights:
//   - MaxCells covers Q * CorpusSize * Length^2 full-DTW DP cells.
//
// Inputs:
//   - p: read-only versioned Protocol.
//
// Returns:
//   - nil on valid declared shape and bounded total search work.
//
// Errors:
//   - ErrInvalidOption for invalid version, dimensions, policy or thresholds.
//   - ErrBudget if full exact-baseline work exceeds MaxCells.
//
// Determinism:
//   - Pure integer and finite-float comparisons; no dataset allocation.
//
// Complexity:
//   - O(1) time and space.
//
// Notes:
//   - Budgeting does not guarantee that the Go runtime can satisfy every allocation.
//
// AI-Hints:
//   - Include QueryCount in cell budgeting; a single-query guard misses full-run cost.
func ValidateProtocol(p Protocol) error {
	if p.Schema != protocolSchema || p.Length < minProtocolLength || p.Length > maxSeriesLength ||
		p.CorpusSize < 1 || p.CorpusSize > maxExperimentItems ||
		p.QueryCount < 1 || p.QueryCount > maxExperimentItems || p.TopK < 1 ||
		p.Shortlist < 1 || p.Shortlist > p.CorpusSize || p.Window < -1 ||
		p.Purge < p.Length || p.Embargo < 0 || p.Horizon < 0 || p.MaxCells == 0 {
		return ErrInvalidOption
	}
	maxInt := int(^uint(0) >> 1)
	if p.Purge > maxInt-p.Horizon || p.Purge+p.Horizon > maxInt-p.Embargo {
		return ErrInvalidOption
	}
	t := p.Thresholds
	if !finite(t.RecallMin) || !finite(t.MissedBestMax) ||
		!finite(t.SpeedupMin) || !finite(t.MemoryRatioMax) ||
		t.RecallMin < 0 || t.RecallMin > 1 ||
		t.MissedBestMax < 0 || t.MissedBestMax > 1 ||
		t.SpeedupMin < 1 || t.MemoryRatioMax < 1 {
		return ErrInvalidOption
	}
	// Division-before-multiplication avoids integer wraparound while accounting
	// for every query in the full exact-scan reference workload.
	if !withinCellBudget(p.MaxCells, p.QueryCount, p.CorpusSize, p.Length, p.Length) {
		return ErrBudget
	}
	return nil
}

// withinCellBudget checks a product of strictly positive dimensions against a limit.
// A nonpositive factor cannot describe a valid DP workload and is always rejected.
//
// Implementation:
//   - Stage 1: reject zero budget or nonpositive factors before signed conversion.
//   - Stage 2: divide the remaining budget by each factor in stable caller order.
//   - Stage 3: return true iff the full mathematical product fits the original limit.
//
// Inputs:
//   - budget: nonnegative DP cell allowance, uint64.
//   - factors: dimension counts, positive platform ints, read-only for this operation.
//
// Returns:
//   - bool: true only when product(factors) <= budget without any intermediate overflow.
//
// Errors:
//   - None; invalid factors produce false rather than an unsafe conversion.
//
// Determinism:
//   - Fixed factor order, integer arithmetic only; no allocations or floating-point rounding.
//
// Complexity:
//   - Time O(len(factors)), Space O(1) beyond the variadic caller arguments.
//
// Notes:
//   - This helper accounts for algorithmic work, not OS allocation success.
//
// AI-Hints:
//   - A product must never be computed first and compared afterward: it can already overflow.
//   - Keep the nonpositive guard in this function when updating G115-safe conversions.
func withinCellBudget(budget uint64, factors ...int) bool {
	if budget == 0 {
		return false
	}
	for _, factor := range factors {
		if factor <= 0 {
			return false
		}
		// A positive int has no more than 63 significant bits on supported Go targets.
		unsignedFactor := uint64(factor)
		if unsignedFactor > budget {
			return false
		}
		budget /= unsignedFactor
	}
	return true
}

// finite checks the algorithm's finite-float input domain without reflection.
// The check is shared by corpus, score, metric and query validators.
func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

// validateCorpusShape performs one deterministic validation pass without hashing.
// Its split from CorpusHash avoids repeated corpus scans and validation recursion.
func validateCorpusShape(candidates []Candidate) error {
	if len(candidates) == 0 || len(candidates[0].Values) == 0 {
		return ErrInvalidInput
	}
	seen := make(map[CandidateID]struct{}, len(candidates))
	length := len(candidates[0].Values)
	for _, candidate := range candidates {
		if _, duplicate := seen[candidate.ID]; duplicate {
			return ErrDuplicateID
		}
		seen[candidate.ID] = struct{}{}
		if candidate.Values == nil || len(candidate.Values) != length {
			return ErrInvalidInput
		}
		for _, sample := range candidate.Values {
			if !finite(sample) {
				return ErrNonFinite
			}
		}
	}
	return nil
}

// ValidateCorpus rejects malformed, duplicated or tampered candidate universes.
// An optional SHA-256 identity commits to order, IDs, lengths and float64 bits.
//
// Implementation:
//   - Stage 1: validate non-empty uniform finite sequences and unique IDs.
//   - Stage 2: when requested, hash values in input order and compare digests.
//   - Stage 3: return without retaining or changing caller-owned samples.
//
// Behavior highlights:
//   - A hash mismatch is invalid data and is separately classed as provenance failure.
//
// Inputs:
//   - candidates: read-only ordered universe, every Values length identical and positive.
//   - expectedHash: empty to omit provenance comparison; otherwise canonical hash text.
//
// Returns:
//   - nil for validated input; no result artifact or partial model is published.
//
// Errors:
//   - ErrInvalidInput for empty, jagged or hash-mismatched corpus.
//   - ErrNonFinite for any NaN or infinite sample.
//   - ErrDuplicateID for any repeated CandidateID.
//   - ErrLeakage joined with ErrInvalidInput for a changed frozen digest.
//
// Determinism:
//   - Validation is in corpus order, then sample order; hashing is likewise ordered.
//
// Complexity:
//   - Time O(N*L), space O(N) for ID uniqueness; fixed-hash space O(1).
//
// Notes:
//   - Concurrent mutation of any candidate or Values during validation is unsupported.
//
// AI-Hints:
//   - Digest authentication requires trusted external storage; SHA alone does not prove history.
func ValidateCorpus(candidates []Candidate, expectedHash string) error {
	if err := validateCorpusShape(candidates); err != nil {
		return err
	}
	if expectedHash == "" {
		return nil
	}
	hash := corpusHashValidated(candidates)
	if hash != expectedHash {
		return errors.Join(ErrInvalidInput, ErrLeakage)
	}
	return nil
}

// ValidateQuery checks sample shape, finite numbers and requested TopK.
// It does not normalize samples or silently change the retrieval metric.
//
// Implementation:
//   - Stage 1: validate TopK and exact sample-count compatibility.
//   - Stage 2: scan samples for NaN and ±Inf in left-to-right order.
//
// Behavior highlights:
//   - TopK may be greater than the number of corpus members.
//
// Inputs:
//   - q: read-only caller-owned query and TopK.
//   - length: exact required sample count; must be positive.
//
// Returns:
//   - nil on success, error on failure; no buffers retained.
//
// Errors:
//   - ErrInvalidInput for nil, empty, mismatched length or nonpositive TopK.
//   - ErrNonFinite for any invalid numeric sample.
//
// Determinism:
//   - First invalid sample in ascending index order determines classification.
//
// Complexity:
//   - O(length) time and O(1) additional space.
//
// Notes:
//   - All accepted values remain in their original physical units.
//
// AI-Hints:
//   - Normalize only under a separately declared policy shared by all comparison arms.
func ValidateQuery(q Query, length int) error {
	if length < 1 || q.TopK < 1 || q.Values == nil || len(q.Values) != length {
		return ErrInvalidInput
	}
	for _, value := range q.Values {
		if !finite(value) {
			return ErrNonFinite
		}
	}
	return nil
}
