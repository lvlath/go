// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax_test demonstrates risk-aware workflows with deterministic realistic fixtures.
// All signals are synthetic engineering scenarios, never claims about historical market data.
package tfsax_test

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/tfsax"
)

// ExampleRunAvailableComparison_exchangeMinuteMotifs demonstrates a financial
// pattern-retrieval audit over synthetic minute-candle return trajectories.
//
// Scenario:
//   - A surveillance team collects a fixed corpus of thirty-two volatility motifs.
//   - Two independent replay queries represent candidate trading-session patterns.
//   - The team wants to quantify how much standard SAX ranking misses versus DTW Top5.
//
// Process:
//   - Freeze a versioned exact-DTW policy and SHA-256 corpus identity.
//   - Execute only the real exact and SAX branches with a prepared SAX index.
//   - Inspect deterministic leading IDs and keep the heuristic proof flag false.
//
// Operational warning:
//   - These are generated waveforms, not historical exchange measurements.
//   - An approximate result must not serve as an exact exclusion in financial controls.
func ExampleRunAvailableComparison_exchangeMinuteMotifs() {
	ctx := context.Background()
	corpus, err := tfsax.GenerateFamily(tfsax.FamilyMotif, 701, 32, 32)
	if err != nil {
		fmt.Println("generator failed")
		return
	}
	// Query windows come from a separately seeded generator, not from the
	// indexed corpus: this avoids artificial self-match recall inflation.
	independent, err := tfsax.GenerateFamily(tfsax.FamilyMotif, 1701, 2, 32)
	if err != nil {
		fmt.Println("query generator failed")
		return
	}
	digest, err := tfsax.CorpusHash(corpus)
	if err != nil {
		fmt.Println("hash failed")
		return
	}
	protocol := tfsax.DefaultProtocol()
	protocol.CorpusSize, protocol.QueryCount = 32, 2
	protocol.Length, protocol.Purge = 32, 32
	protocol.TopK, protocol.Shortlist = 5, 16
	queries := []tfsax.NamedQuery{
		{ID: 702, Family: tfsax.FamilyMotif, Query: tfsax.Query{Values: independent[1].Values, TopK: 5}},
		{ID: 701, Family: tfsax.FamilyMotif, Query: tfsax.Query{Values: independent[0].Values, TopK: 5}},
	}
	experiment := tfsax.Experiment{Protocol: protocol, Definition: tfsax.UnresolvedDefinition(),
		Corpus: corpus, CorpusDigest: digest, Queries: queries}
	comparison, err := tfsax.RunAvailableComparison(ctx, experiment,
		tfsax.ComparisonOptions{SAXSegments: 8, SAXAlphabet: 8, WarmupRuns: 1})
	if err != nil {
		fmt.Println("comparison failed")
		return
	}
	exact := comparison.Observations[0]
	saxRanking := comparison.Observations[1]
	fmt.Println(len(comparison.Observations), exact.QueryID,
		exact.Result.Diagnostics.Examined, exact.Recall.Value == 1,
		saxRanking.Recall.Applicable, saxRanking.Result.Diagnostics.Exact)
	// Output: 4 701 32 true true false
}

// ExampleScanExact_vibrationSignatureAudit demonstrates deterministic maintenance
// triage using a measured-like rotating-machine vibration waveform fixture.
//
// Scenario:
//   - Twelve inspection windows cover a machine under different load conditions.
//   - The reference signature includes ninety-six samples across eight gear cycles.
//   - Exact DTW is used when missing a match could hide an expensive bearing defect.
//
// Process:
//   - Build finite synthetic sensor windows with different amplitude and drift.
//   - Insert one identical registered reference window under stable CandidateID 42.
//   - Retrieve exact Top4 and inspect the auditable first identity.
func ExampleScanExact_vibrationSignatureAudit() {
	const length = 96
	template := make([]float64, length)
	for t := range template {
		cycle := math.Sin(2*math.Pi*float64(t)/12) + 0.22*math.Sin(2*math.Pi*float64(t)/4)
		template[t] = cycle + 0.08*math.Cos(2*math.Pi*float64(t)/length)
	}
	corpus := make([]tfsax.Candidate, 0, 12)
	for sensor := 0; sensor < 12; sensor++ {
		values := make([]float64, length)
		for sample := range values {
			values[sample] = template[sample]*(0.75+0.06*float64(sensor)) + float64(sensor)*0.015
		}
		if sensor == 7 {
			copy(values, template)
		}
		corpus = append(corpus, tfsax.Candidate{ID: tfsax.CandidateID(35 + sensor), Values: values})
	}
	result, err := tfsax.ScanExact(context.Background(), corpus, tfsax.Query{Values: template, TopK: 4}, 5)
	if err != nil {
		fmt.Println("exact scan failed")
		return
	}
	fmt.Println(result.Candidates[0], len(result.Candidates), result.Diagnostics.Exact)
	// Output: 42 4 true
}

// ExampleScanSAXWithOptions_networkLatencyTriage models a multi-service latency
// incident in which a fast symbolic shortlist is useful but cannot prove safety.
//
// Scenario:
//   - Twenty services each have forty-eight sampled latency measurements.
//   - Service 217 has the reference burst family; a separate probe adds its own
//     deterministic measurement ripple rather than copying a corpus window.
//   - Operations may rank alerts quickly but must still check full DTW before closure.
//   - This controlled ID witness is a functional audit, not unbiased holdout recall.
//
// Process:
//   - Run a specified eight-segment SAX Top8 ranking with DTW refinement.
//   - Check that excluded services remain explicitly uncertified.
//   - Fall back to exact scan to obtain the ground-truth critical alert identity.
func ExampleScanSAXWithOptions_networkLatencyTriage() {
	const sampleCount = 48
	corpus := make([]tfsax.Candidate, 0, 20)
	for service := 0; service < 20; service++ {
		latency := make([]float64, sampleCount)
		for sample := range latency {
			burst := 0.0
			if sample >= 20 && sample < 28 {
				burst = 90 + 3*float64(service%5)
			}
			latency[sample] = 20 + float64(service)*0.4 +
				3*math.Sin(float64(sample)/6) + burst
		}
		corpus = append(corpus, tfsax.Candidate{ID: tfsax.CandidateID(200 + service), Values: latency})
	}
	// Construct a separately allocated synthetic probe with its own deterministic
	// measurement ripple. It follows the designated reference service's model
	// but is not a byte-for-byte copy of the indexed candidate's signal.
	probe := make([]float64, sampleCount)
	for sample := range probe {
		burst := 0.0
		if sample >= 20 && sample < 28 {
			burst = 90 + 3*float64(17%5)
		}
		probe[sample] = 20 + 17*0.4 + 3*math.Sin(float64(sample)/6) +
			burst + 0.02*math.Cos(float64(sample)/3)
	}
	query := tfsax.Query{Values: probe, TopK: 3}
	shortlist, err := tfsax.ScanSAXWithOptions(context.Background(), corpus, query,
		tfsax.SAXSearchOptions{Window: 4, Shortlist: 8, Segments: 8, Alphabet: 8})
	if err != nil {
		fmt.Println("ranking failed")
		return
	}
	groundTruth, err := tfsax.ScanExact(context.Background(), corpus, query, 4)
	if err != nil {
		fmt.Println("exact audit failed")
		return
	}
	fmt.Println(shortlist.Diagnostics.Examined, shortlist.Diagnostics.Pruned,
		!shortlist.Diagnostics.Exact, groundTruth.Candidates[0])
	// Output: 8 12 true 217
}

// ExampleValidateWindowRecords_energyDemandLeakageAudit demonstrates a protected
// forecast-backtest split for hourly smart-grid load measurements.
//
// Scenario:
//   - Forty-eight-hour feature windows predict the following eight-hour horizon.
//   - Train, calibration and test occupy disjoint blocks of raw chronological data.
//   - A mislabeled test window accidentally enters the fitted-normalization set.
//
// Process:
//   - Validate full support and label endpoints before any training transform.
//   - Confirm that a test ID in FitIDs is rejected rather than silently ignored.
func ExampleValidateWindowRecords_energyDemandLeakageAudit() {
	plan := tfsax.SplitPlan{
		Train:       tfsax.Interval{Start: 0, End: 400},
		Calibration: tfsax.Interval{Start: 460, End: 600},
		Test:        tfsax.Interval{Start: 660, End: 800},
		TrainIDs:    []tfsax.CandidateID{11}, CalibrationIDs: []tfsax.CandidateID{22},
		TestIDs: []tfsax.CandidateID{33}, FitIDs: []tfsax.CandidateID{11},
	}
	records := []tfsax.WindowRecord{
		{ID: 11, Split: tfsax.SplitTrain, Support: tfsax.Interval{Start: 100, End: 148}, LabelEnd: 156},
		{ID: 22, Split: tfsax.SplitCalibration, Support: tfsax.Interval{Start: 480, End: 528}, LabelEnd: 536},
		{ID: 33, Split: tfsax.SplitTest, Support: tfsax.Interval{Start: 700, End: 748}, LabelEnd: 756},
	}
	valid := tfsax.ValidateWindowRecords(plan, records, 48, 8, 48, 4) == nil
	plan.FitIDs = append(plan.FitIDs, 33)
	blocked := errors.Is(tfsax.ValidateWindowRecords(plan, records, 48, 8, 48, 4), tfsax.ErrLeakage)
	fmt.Println(valid, blocked)
	// Output: true true
}

// ExampleLBKeoghAbsolute_alignmentCounterexample verifies that diagonal
// pointwise costs cannot be advertised as arbitrary-DTW lower bounds.
//
// Scenario:
//   - Two binary event traces represent the same event with one repeated sample.
//   - Unrestricted time warping aligns them at zero cost despite a pointwise mismatch.
//   - A naive squared pointwise score would therefore incorrectly discard a true match.
//
// Process:
//   - Evaluate the exact oracle, a genuine diagnostic LB_Keogh envelope,
//     and the explicit refusal to run uncertified DTW pruning.
func ExampleLBKeoghAbsolute_alignmentCounterexample() {
	query := []float64{0, 0, 1, 1}
	candidate := []float64{0, 1, 1, 1}
	distance, _, err := tfsax.ExactDTW(query, candidate, -1)
	if err != nil {
		fmt.Println("oracle failed")
		return
	}
	naivePointwiseSquared := 1.0
	bound, err := tfsax.LBKeoghAbsolute(query, candidate, 3)
	if err != nil {
		fmt.Println("envelope failed")
		return
	}
	_, gateErr := tfsax.ScanLBConservative(context.Background(), nil, tfsax.Query{}, 3)
	fmt.Println(naivePointwiseSquared > distance, bound <= distance,
		errors.Is(gateErr, tfsax.ErrUncertifiedBound))
	// Output: true true true
}

// ExampleSummarizeByFamily_worstCaseGate demonstrates family-local reporting
// for a monitoring fleet with plateau and phase-shift disturbances.
//
// Scenario:
//   - An anomaly detector compares candidate windows under two stress families.
//   - A pooled mean would conceal a poor recall under the less favorable family.
//   - The harness therefore groups exact and approximate arms independently.
//
// Process:
//   - Freeze the candidate universe and run the two implemented branches.
//   - Report the number of groups and distinguish genuine recall from NA.
func ExampleSummarizeByFamily_worstCaseGate() {
	corpus, err := tfsax.GenerateFamily(tfsax.FamilyPlateau, 81, 12, 24)
	if err != nil {
		fmt.Println("generator failed")
		return
	}
	plateauQueries, err := tfsax.GenerateFamily(tfsax.FamilyPlateau, 1081, 1, 24)
	if err != nil {
		fmt.Println("plateau query failed")
		return
	}
	phaseQueries, err := tfsax.GenerateFamily(tfsax.FamilyPhase, 2081, 1, 24)
	if err != nil {
		fmt.Println("phase query failed")
		return
	}
	digest, err := tfsax.CorpusHash(corpus)
	if err != nil {
		fmt.Println("hash failed")
		return
	}
	p := tfsax.DefaultProtocol()
	p.CorpusSize, p.QueryCount, p.Length, p.Purge = 12, 2, 24, 24
	p.TopK, p.Shortlist = 2, 6
	experiment := tfsax.Experiment{Protocol: p, Definition: tfsax.UnresolvedDefinition(),
		Corpus: corpus, CorpusDigest: digest,
		Queries: []tfsax.NamedQuery{
			{ID: 1, Family: tfsax.FamilyPlateau, Query: tfsax.Query{Values: plateauQueries[0].Values, TopK: 2}},
			{ID: 2, Family: tfsax.FamilyPhase, Query: tfsax.Query{Values: phaseQueries[0].Values, TopK: 2}},
		},
	}
	comparison, err := tfsax.RunAvailableComparison(context.Background(), experiment,
		tfsax.ComparisonOptions{SAXSegments: 6, SAXAlphabet: 8})
	if err != nil {
		fmt.Println("comparison failed")
		return
	}
	summaries, err := tfsax.SummarizeByFamily(comparison.Observations)
	if err != nil {
		fmt.Println("summary failed")
		return
	}
	fmt.Println(len(summaries), summaries[0].WorstRecall.Applicable,
		summaries[len(summaries)-1].MissedBestRate.Applicable)
	// Output: 4 true true
}

// ExamplePrepareSAXIndex_reusablePumpFaultTriage demonstrates a detached,
// reusable SAX shortlist for repeated pump-vibration investigations.
//
// Scenario:
//   - A water-treatment facility retains sixty 48-sample waveform windows.
//   - Each new vibration trace may suggest a transient, but a heuristic match
//     must be independently confirmed before an operational alarm is issued.
//   - The example uses dimensionless synthetic stress fixtures, not sensor logs.
//
// Implementation:
//   - Stage 1: generate an indexed corpus and two independently seeded queries.
//   - Stage 2: prepare SAX once, then submit each query for exact-DTW refinement.
//   - Stage 3: demonstrate that later mutation of the original corpus does not
//     change the prepared index identity or its precomputed scoring state.
//
// Behavior highlights:
//   - An index is immutable after construction and safe for repeated read-only searches.
//   - Neither search establishes global exactness when the shortlist excludes candidates.
//
// Inputs:
//   - Fixed fixture seeds, a 48-sample window, a 12-candidate shortlist and TopK=4.
//
// Returns:
//   - Only deterministic contract diagnostics are printed; no alarm policy is inferred.
//
// Errors:
//   - Setup/search failures are reported without interpreting missing results as matches.
//
// Determinism:
//   - Prepared words are scored in corpus order; MINDIST and DTW ties use numeric ID.
//
// Complexity:
//   - Index setup O(N*(L+w log a)); each query O(N*w+N log N+M*L^2).
//
// Notes:
//   - Original caller input may be changed after PrepareSAXIndex returns;
//     do not mutate query values while a Search call is reading them.
//
// AI-Hints:
//   - Compare query latency only after index construction, and keep index
//     memory measurement separate from the ID+symbol IndexBytes estimate.
func ExamplePrepareSAXIndex_reusablePumpFaultTriage() {
	corpus, err := tfsax.GenerateFamily(tfsax.FamilyHeavyTail, 803, 60, 48)
	if err != nil {
		fmt.Println("corpus generation failed")
		return
	}
	querySet, err := tfsax.GenerateFamily(tfsax.FamilyHeavyTail, 1803, 2, 48)
	if err != nil {
		fmt.Println("query generation failed")
		return
	}

	index, err := tfsax.PrepareSAXIndex(corpus, 8, 8)
	if err != nil {
		fmt.Println("index preparation failed")
		return
	}
	frozenIdentity := index.CorpusDigest()

	// Modify the original source *after* the prepared index has detached it.
	// This simulates reusing a mutable ingestion buffer across measurement cycles.
	corpus[0].Values[0] += 100

	completed := 0
	uncertified := true
	for _, queryCandidate := range querySet {
		selected, searchErr := index.Search(context.Background(),
			tfsax.Query{Values: queryCandidate.Values, TopK: 4}, 5, 12)
		if searchErr != nil {
			fmt.Println("search failed")
			return
		}
		if selected.Complete {
			completed++
		}
		uncertified = uncertified && !selected.Diagnostics.Exact &&
			selected.Diagnostics.Examined == 12 && selected.Diagnostics.Pruned == 48
	}
	fmt.Printf("queries=%d index-detached=%t shortlist-uncertified=%t\n",
		completed, index.CorpusDigest() == frozenIdentity, uncertified)
	// Output: queries=2 index-detached=true shortlist-uncertified=true
}

// ExampleRunExactReference_independentDatacenterLoadAudits demonstrates a
// multi-query exact reference without relying on a JSON configuration file.
//
// Scenario:
//   - A datacenter analysis team compares sixty-four-sample load-shift histories.
//   - Three independently generated shift patterns are evaluated against a frozen
//     corpus to establish the true candidates for later filtering experiments.
//
// Implementation:
//   - Stage 1: freeze an exact-work protocol, an unresolved research definition
//     and an input-order corpus checksum using ordinary Go values.
//   - Stage 2: run every query through the full exact absolute-cost DTW oracle.
//   - Stage 3: confirm each published result is complete and mathematically exact.
//
// Behavior highlights:
//   - An exact-only reference needs neither an experimental TFSAX definition nor
//     an on-disk protocol document, but it cannot grant a four-arm research result.
//
// Inputs:
//   - Three stable query IDs and independent fixed seeds; lengths are in samples.
//
// Returns:
//   - Three detached observations ranked by query ID, without duration fabrication.
//
// Errors:
//   - Any validation or DTW failure aborts without a partial observation batch.
//
// Determinism:
//   - Query IDs are sorted numerically; candidates tie-break on numeric CandidateID.
//
// Complexity:
//   - O(Q*(N*L^2+N log N)) time, O(N+Q*TopK) retained observation/work memory.
//
// Notes:
//   - The fixture is synthetic; actual datacenter time-grid checking belongs
//     to the ingestion layer before the experiment is constructed.
//
// AI-Hints:
//   - Do not interpret RecallAtK(exact, exact)==1 as validation of a heuristic;
//     it simply defines the reference target for a later independent arm.
func ExampleRunExactReference_independentDatacenterLoadAudits() {
	corpus, err := tfsax.GenerateFamily(tfsax.FamilySlope, 410, 24, 64)
	if err != nil {
		fmt.Println("corpus generation failed")
		return
	}
	independent, err := tfsax.GenerateFamily(tfsax.FamilySlope, 1410, 3, 64)
	if err != nil {
		fmt.Println("query generation failed")
		return
	}
	digest, err := tfsax.CorpusHash(corpus)
	if err != nil {
		fmt.Println("checksum failed")
		return
	}

	policy := tfsax.DefaultProtocol()
	policy.Length, policy.Purge = 64, 64
	policy.CorpusSize, policy.QueryCount = 24, 3
	policy.TopK, policy.Shortlist = 4, 12
	experiment := tfsax.Experiment{
		Protocol: policy, Definition: tfsax.UnresolvedDefinition(),
		Corpus: corpus, CorpusDigest: digest,
		Queries: []tfsax.NamedQuery{
			{ID: 6103, Family: tfsax.FamilySlope,
				Query: tfsax.Query{Values: independent[2].Values, TopK: 4}},
			{ID: 6101, Family: tfsax.FamilySlope,
				Query: tfsax.Query{Values: independent[0].Values, TopK: 4}},
			{ID: 6102, Family: tfsax.FamilySlope,
				Query: tfsax.Query{Values: independent[1].Values, TopK: 4}},
		},
	}
	results, err := tfsax.RunExactReference(context.Background(), experiment)
	if err != nil {
		fmt.Println("exact reference failed")
		return
	}
	everyExact := true
	everyApplicable := true
	for _, record := range results {
		everyExact = everyExact && record.Result.Complete && record.Result.Diagnostics.Exact
		everyApplicable = everyApplicable && record.Recall.Applicable && record.Recall.Value == 1
	}
	fmt.Printf("queries=%d first-query=%d exact=%t oracle-recall=%t\n",
		len(results), results[0].QueryID, everyExact, everyApplicable)
	// Output: queries=3 first-query=6101 exact=true oracle-recall=true
}
