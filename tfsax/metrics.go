// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax defines strict-ID retrieval quality and deterministic sample quantiles.
// This file keeps empty-oracle NA states distinct from successful zero error rates.
package tfsax

import "sort"

// RecallAtK computes the set fraction of reachable exact TopK IDs recovered by a result.
// It returns NA when there are no exact reachable target IDs to use as denominator.
//
// Implementation:
//   - Stage 1: build a unique exact-truth set from the provided target prefix.
//   - Stage 2: count unique returned IDs also present in exact truth.
//   - Stage 3: divide by unique truth cardinality or publish NA for empty truth.
//
// Behavior highlights:
//   - CandidateIDs are compared strictly; distance tolerance is a separate metric.
//   - Duplicate IDs in malformed external lists cannot inflate recall or the denominator.
//
// Inputs:
//   - got: candidate IDs returned by approximate search, read-only.
//   - want: exact reachable TopK IDs, read-only; must correspond to the same query.
//
// Returns:
//   - MetricValue in [0,1] with Applicable=true, or Applicable=false for empty truth.
//
// Errors:
//   - None; empty truth uses NA instead of an error or a fabricated perfect recall.
//
// Determinism:
//   - Membership checks depend only on ID sets; map iteration order is never published.
//
// Complexity:
//   - O(len(got)+len(want)) time and space.
//
// Notes:
//   - This metric does not excuse a different answer at a tied cutoff; strict ID policy applies.
//
// AI-Hints:
//   - Never evaluate recall against the heuristic shortlist instead of exact TopK.
func RecallAtK(got, want []CandidateID) MetricValue {
	if len(want) == 0 {
		return MetricValue{}
	}
	truth := make(map[CandidateID]struct{}, len(want))
	for _, id := range want {
		truth[id] = struct{}{}
	}
	observed := make(map[CandidateID]struct{}, len(got))
	hits := 0
	for _, id := range got {
		if _, already := observed[id]; already {
			continue
		}
		observed[id] = struct{}{}
		if _, present := truth[id]; present {
			hits++
		}
	}
	return MetricValue{Value: float64(hits) / float64(len(truth)), Applicable: true}
}

// MissedBest indicates whether the exact oracle's deterministic first ID is absent.
// It is a binary miss event rather than a recall proportion across multiple candidates.
//
// Implementation:
//   - Stage 1: return NA if the exact oracle has no reachable candidate.
//   - Stage 2: scan result IDs for want[0], the canonical exact best.
//   - Stage 3: return zero if present and one if absent.
//
// Behavior highlights:
//   - Exact-best identity respects deterministic smallest-ID tie-breaking.
//
// Inputs:
//   - got: read-only approximate retrieved IDs.
//   - want: read-only exact TopK with the best candidate first.
//
// Returns:
//   - Binary MetricValue with Applicable=true, or NA on empty exact target.
//
// Errors:
//   - None.
//
// Determinism:
//   - Checks IDs in result order; no floating-point approximation enters this metric.
//
// Complexity:
//   - O(len(got)) time and O(1) extra memory.
//
// Notes:
//   - Missed-best aggregate is the mean of applicable binary events per declared family.
//
// AI-Hints:
//   - Do not identify best by an approximate distance or an arbitrary tied candidate.
func MissedBest(got, want []CandidateID) MetricValue {
	if len(want) == 0 {
		return MetricValue{}
	}
	for _, id := range got {
		if id == want[0] {
			return MetricValue{Applicable: true}
		}
	}
	return MetricValue{Value: 1, Applicable: true}
}

// LatencyQuantiles calculates nearest-rank p50/p95/p99 from nonnegative observations.
// It never sorts, mutates or retains the caller-owned duration slice.
//
// Implementation:
//   - Stage 1: reject an empty sample or any negative duration.
//   - Stage 2: sort an owned copy in ascending nanosecond order.
//   - Stage 3: select ceil(p*n/100)-1 with division-before-multiplication arithmetic.
//
// Behavior highlights:
//   - No interpolation or benchmark postprocessing changes the sample values.
//
// Inputs:
//   - samples: nonempty read-only integer nanoseconds, all >=0.
//
// Returns:
//   - MeasurementSummary containing nearest-rank p50, p95 and p99.
//
// Errors:
//   - ErrInvalidInput for empty samples or a negative duration.
//
// Determinism:
//   - Results depend on sorted numeric values, never arrival timing order.
//
// Complexity:
//   - O(n log n) time, O(n) memory for a detached copy.
//
// Notes:
//   - A quantile estimates only the supplied experiment workload; include toolchain and hardware.
//
// AI-Hints:
//   - Avoid percentile*n multiplication directly: it can overflow int for enormous slices.
func LatencyQuantiles(samples []int64) (MeasurementSummary, error) {
	if len(samples) == 0 {
		return MeasurementSummary{}, ErrInvalidInput
	}
	ordered := append([]int64(nil), samples...)
	for _, sample := range ordered {
		if sample < 0 {
			return MeasurementSummary{}, ErrInvalidInput
		}
	}
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	percentile := func(percent int) int64 {
		// floor(n/100)*p + ceil((n%100)*p/100) computes ceil(n*p/100)
		// without ever multiplying an unbounded n by 99.
		n := len(ordered)
		rank := (n/100)*percent + ((n%100)*percent+99)/100
		return ordered[rank-1]
	}
	return MeasurementSummary{P50: percentile(50), P95: percentile(95), P99: percentile(99)}, nil
}

// FamilySummary reports worst-case strict recall and missed-best rate per branch/family.
// No pooled average can hide a family where the heuristic fails repeatedly.
//
// Implementation:
//   - Aggregate observations by their frozen (Family,Branch), select the minimum applicable recall and summarize measured durations.
//
// Behavior highlights:
//   - WorstRecall is NA if every exact target is empty for the group.
//   - MissedBestRate is NA without applicable exact-first IDs.
//   - HasLatency is false for semantic-only observations carrying no measured duration.
//
// Inputs:
//   - Published by SummarizeByFamily from complete, valid observations only.
//
// Returns:
//   - A detached summary value with independent string/scalar fields.
//
// Errors:
//   - Malformed input is rejected by SummarizeByFamily before summary publication.
//
// Determinism:
//   - Groups are published in ascending (family,branch) lexical order.
//
// Complexity:
//   - O(1) output fields per branch/family.
//
// Notes:
//   - WorstRecall and MissedBestRate may independently be NA; HasLatency identifies a genuinely timed group.
//
// AI-Hints:
//   - Inspect worst-case families, not only the pooled mean or median.
type FamilySummary struct {
	Family         Family             // Declared synthetic stress regime.
	Branch         string             // Existing measured retrieval arm.
	Queries        int                // Number of query observations in this group.
	WorstRecall    MetricValue        // Smallest applicable strict-ID recall in the group.
	MissedBestRate MetricValue        // Mean of applicable binary missed-best events.
	HasLatency     bool               // True only when this group contains timed samples.
	Latency        MeasurementSummary // Nanosecond quantiles of measured samples only.
}

// SummarizeByFamily aggregates valid observations without pooling away weak regimes.
// It intentionally reports NA for families lacking reachable exact-target denominators.
//
// Implementation:
//   - Stage 1: validate observation metrics and group by (Family, Branch).
//   - Stage 2: accumulate the worst applicable recall, binary missed-best and timings.
//   - Stage 3: publish sorted detached summaries with explicit NA/timing flags.
//
// Behavior highlights:
//   - Observations with DurationNS=0 are treated as semantic-only, not measured latency.
//
// Inputs:
//   - observations: read-only flat per-query measurements with complete results.
//
// Returns:
//   - []FamilySummary sorted lexically by family then branch, or nil on error.
//
// Errors:
//   - ErrInvalidInput for empty, malformed, incomplete or invalid metric payloads.
//
// Determinism:
//   - Stable lexical group ordering; no map iteration order enters published output.
//
// Complexity:
//   - O(n log n) time for grouping/quantiles/sorting; O(n) additional memory.
//
// Notes:
//   - Does not evaluate gate passage; absent TFSAX results cannot be synthesized.
//
// AI-Hints:
//   - Do not substitute zero for NA or merge different family labels post hoc.
func SummarizeByFamily(observations []Observation) ([]FamilySummary, error) {
	if len(observations) == 0 {
		return nil, ErrInvalidInput
	}
	type key struct {
		family Family
		branch string
	}
	type accumulator struct {
		queries    int
		worst      MetricValue
		misses     float64
		applicable int
		durations  []int64
	}
	groups := make(map[key]*accumulator)
	for _, observation := range observations {
		if !observation.Result.Complete || !knownFamily(observation.Family) ||
			observation.Branch == "" || observation.DurationNS < 0 ||
			!validMetric(observation.Recall) || !validMetric(observation.MissedBest) ||
			observation.Recall.Applicable != observation.MissedBest.Applicable {
			return nil, ErrInvalidInput
		}
		entryKey := key{observation.Family, observation.Branch}
		entry := groups[entryKey]
		if entry == nil {
			entry = &accumulator{}
			groups[entryKey] = entry
		}
		entry.queries++
		if observation.Recall.Applicable {
			if !entry.worst.Applicable || observation.Recall.Value < entry.worst.Value {
				entry.worst = observation.Recall
			}
			entry.misses += observation.MissedBest.Value
			entry.applicable++
		}
		if observation.DurationNS > 0 {
			entry.durations = append(entry.durations, observation.DurationNS)
		}
	}
	keys := make([]key, 0, len(groups))
	for groupKey := range groups {
		keys = append(keys, groupKey)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].family == keys[j].family {
			return keys[i].branch < keys[j].branch
		}
		return keys[i].family < keys[j].family
	})
	summaries := make([]FamilySummary, 0, len(keys))
	for _, groupKey := range keys {
		entry := groups[groupKey]
		summary := FamilySummary{Family: groupKey.family, Branch: groupKey.branch,
			Queries: entry.queries, WorstRecall: entry.worst}
		if entry.applicable > 0 {
			summary.MissedBestRate = MetricValue{
				Applicable: true, Value: entry.misses / float64(entry.applicable),
			}
		}
		if len(entry.durations) > 0 {
			var err error
			summary.Latency, err = LatencyQuantiles(entry.durations)
			if err != nil {
				return nil, err
			}
			summary.HasLatency = true
		}
		summaries = append(summaries, summary)
	}
	return summaries, nil
}

// validMetric accepts only explicit NA or finite numeric proportions in [0,1].
// Its predicate cannot accidentally legalize a NaN in JSON-report artifacts.
func validMetric(value MetricValue) bool {
	if !value.Applicable {
		return value.Value == 0
	}
	return finite(value.Value) && value.Value >= 0 && value.Value <= 1
}
