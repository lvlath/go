// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax_test tests an honest two-arm comparison without fabricated TFSAX evidence.
// Test assertions defend both exact-DTW identity and conservative heuristic diagnostics.
package tfsax_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/tfsax"
)

func comparisonFixture(t *testing.T) tfsax.Experiment {
	t.Helper()
	corpus, err := tfsax.GenerateFamily(tfsax.FamilyMotif, 19, 12, 24)
	if err != nil {
		t.Fatal(err)
	}
	// A separate seed keeps evaluation queries out of the indexed corpus.
	independent, err := tfsax.GenerateFamily(tfsax.FamilyMotif, 1019, 3, 24)
	if err != nil {
		t.Fatal(err)
	}
	digest, err := tfsax.CorpusHash(corpus)
	if err != nil {
		t.Fatal(err)
	}
	protocol := tfsax.DefaultProtocol()
	protocol.Length, protocol.Purge = 24, 24
	protocol.CorpusSize, protocol.QueryCount = 12, 3
	protocol.TopK, protocol.Shortlist = 3, 8
	return tfsax.Experiment{Protocol: protocol, Definition: tfsax.UnresolvedDefinition(),
		Corpus: corpus, CorpusDigest: digest,
		Queries: []tfsax.NamedQuery{
			{ID: 303, Query: tfsax.Query{Values: independent[2].Values, TopK: 3}, Family: tfsax.FamilyMotif},
			{ID: 301, Query: tfsax.Query{Values: independent[0].Values, TopK: 3}, Family: tfsax.FamilyMotif},
			{ID: 302, Query: tfsax.Query{Values: independent[1].Values, TopK: 3}, Family: tfsax.FamilyMotif},
		}}
}

func TestComparator_ExactlyTwoHonestBranches(t *testing.T) {
	experiment := comparisonFixture(t)
	options := tfsax.ComparisonOptions{SAXSegments: 6, SAXAlphabet: 8, WarmupRuns: 1}
	result, err := tfsax.RunAvailableComparison(context.Background(), experiment, options)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Complete || len(result.Observations) != 6 || result.SAXIndexBytes != 12*(6*2+8) {
		t.Fatalf("wrong comparator envelope %+v", result)
	}
	queryDigest, err := tfsax.QueriesHash(experiment.Queries)
	if err != nil || result.QueriesDigest != queryDigest || result.CorpusDigest != experiment.CorpusDigest {
		t.Fatalf("frozen comparison identity lost: %+v, query hash error=%v", result, err)
	}
	for queryIndex := 0; queryIndex < 3; queryIndex++ {
		exact := result.Observations[2*queryIndex]
		sax := result.Observations[2*queryIndex+1]
		if exact.QueryID != uint64(301+queryIndex) || sax.QueryID != exact.QueryID {
			t.Fatalf("unstable query order: %+v %+v", exact, sax)
		}
		if exact.Branch != "exact" || sax.Branch != "sax-ranking" ||
			!exact.Result.Diagnostics.Exact || sax.Result.Diagnostics.Exact ||
			!exact.Result.Complete || !sax.Result.Complete {
			t.Fatalf("invented or missing branch: %+v %+v", exact, sax)
		}
		if exact.Result.Diagnostics.Examined != 12 || exact.Result.Diagnostics.Pruned != 0 ||
			sax.Result.Diagnostics.Examined != 8 || sax.Result.Diagnostics.Pruned != 4 ||
			exact.Available != 12 || sax.Available != 12 {
			t.Fatalf("incorrect work accounting: %+v %+v", exact, sax)
		}
		if !exact.Recall.Applicable || exact.Recall.Value != 1 ||
			!sax.Recall.Applicable || sax.Recall.Value < 0 || sax.Recall.Value > 1 ||
			!sax.MissedBest.Applicable {
			t.Fatalf("invalid strict metrics: %+v %+v", exact, sax)
		}
		if exact.DurationNS < 0 || sax.DurationNS < 0 || sax.IndexBytes != result.SAXIndexBytes {
			t.Fatalf("invalid measurement semantics: %+v %+v", exact, sax)
		}
	}
	// Caller mutation of one result cannot change subsequent scans or the corpus.
	result.Observations[0].Result.Candidates[0] = 900
	second, err := tfsax.RunAvailableComparison(context.Background(), experiment, options)
	if err != nil {
		t.Fatal(err)
	}
	if second.Observations[0].Result.Candidates[0] == 900 {
		t.Fatal("nested result ownership violated")
	}
	var nilResult *tfsax.ComparisonResult
	if !nilResult.IsNil() || (&tfsax.ComparisonResult{}).IsNil() {
		t.Fatal("nilable comparison result violated")
	}
}

func TestComparator_ValidationAndResearchGate(t *testing.T) {
	experiment := comparisonFixture(t)
	options := tfsax.ComparisonOptions{SAXSegments: 6, SAXAlphabet: 8}
	_, err := tfsax.RunAvailableComparison(nil, experiment, options)
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = tfsax.RunAvailableComparison(ctx, experiment, options)
	mustErrorIs(t, err, context.Canceled)
	_, err = tfsax.RunAvailableComparison(context.Background(), experiment, tfsax.ComparisonOptions{})
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	badOptions := options
	badOptions.WarmupRuns = 5
	_, err = tfsax.RunAvailableComparison(context.Background(), experiment, badOptions)
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	bad := experiment
	bad.CorpusDigest = ""
	_, err = tfsax.RunAvailableComparison(context.Background(), bad, options)
	mustErrorIs(t, err, tfsax.ErrInvalidInput)
	bad = experiment
	bad.Queries = append([]tfsax.NamedQuery(nil), experiment.Queries...)
	bad.Queries[0].ID = bad.Queries[1].ID
	_, err = tfsax.RunAvailableComparison(context.Background(), bad, options)
	mustErrorIs(t, err, tfsax.ErrDuplicateID)
	bad = experiment
	bad.Queries = append([]tfsax.NamedQuery(nil), experiment.Queries...)
	bad.Queries[0].Query.TopK = 1
	_, err = tfsax.RunAvailableComparison(context.Background(), bad, options)
	mustErrorIs(t, err, tfsax.ErrInvalidInput)
	bad = experiment
	bad.Definition.Approved = true // Incomplete metadata cannot authorize an algorithm.
	_, err = tfsax.RunAvailableComparison(context.Background(), bad, options)
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	_, err = tfsax.Run(context.Background(), experiment)
	if !errors.Is(err, tfsax.ErrUndefinedRepresentation) || !errors.Is(err, tfsax.ErrUncertifiedBound) {
		t.Fatalf("full gate admitted incomplete research: %v", err)
	}
}

func TestComparator_FamilyWorstCaseAndMetricNA(t *testing.T) {
	experiment := comparisonFixture(t)
	result, err := tfsax.RunAvailableComparison(context.Background(), experiment,
		tfsax.ComparisonOptions{SAXSegments: 6, SAXAlphabet: 8})
	if err != nil {
		t.Fatal(err)
	}
	summaries, err := tfsax.SummarizeByFamily(result.Observations)
	if err != nil {
		t.Fatal(err)
	}
	if len(summaries) != 2 || summaries[0].Branch != "exact" || summaries[1].Branch != "sax-ranking" {
		t.Fatalf("not independently grouped: %+v", summaries)
	}
	if summaries[0].Queries != 3 || !summaries[0].WorstRecall.Applicable ||
		summaries[0].WorstRecall.Value != 1 || summaries[0].MissedBestRate.Value != 0 {
		t.Fatalf("exact reference lost truth: %+v", summaries[0])
	}
	if summaries[1].WorstRecall.Value < 0 || summaries[1].WorstRecall.Value > 1 {
		t.Fatalf("invalid worst-family metric: %+v", summaries[1])
	}
	na := tfsax.Observation{Family: tfsax.FamilyConstant, Branch: "exact", Result: tfsax.RetrievalResult{Complete: true}}
	s, err := tfsax.SummarizeByFamily([]tfsax.Observation{na})
	if err != nil || len(s) != 1 || s[0].WorstRecall.Applicable || s[0].MissedBestRate.Applicable || s[0].HasLatency {
		t.Fatalf("empty exact target must be NA, no fabricated timing: %v %+v", err, s)
	}
	na.Recall = tfsax.MetricValue{Applicable: true, Value: math.NaN()}
	_, err = tfsax.SummarizeByFamily([]tfsax.Observation{na})
	mustErrorIs(t, err, tfsax.ErrInvalidInput)
	_, err = tfsax.SummarizeByFamily(nil)
	mustErrorIs(t, err, tfsax.ErrInvalidInput)
}

func TestComparator_DefaultWrappersAndWarmupLimits(t *testing.T) {
	experiment := comparisonFixture(t)
	defaults := tfsax.DefaultComparisonOptions()
	if defaults.SAXSegments != 8 || defaults.SAXAlphabet != 8 || defaults.WarmupRuns != 0 {
		t.Fatalf("unexpected disclosed SAX defaults: %+v", defaults)
	}
	result, err := tfsax.RunAvailableDefault(context.Background(), experiment)
	if err != nil || result == nil || !result.Complete || len(result.Observations) != 6 {
		t.Fatalf("default comparison failed: %+v %v", result, err)
	}
	options := tfsax.DefaultSAXSearchOptions(2, 4)
	if options.Window != 2 || options.Shortlist != 4 || options.Segments != 8 || options.Alphabet != 8 {
		t.Fatalf("unexpected search defaults: %+v", options)
	}
	_, err = tfsax.ScanSAXWithOptions(context.Background(), experiment.Corpus,
		tfsax.Query{Values: experiment.Corpus[2].Values, TopK: 2}, options)
	if err != nil {
		t.Fatal(err)
	}
}

// TestComparator_BudgetCoversBothArmsAndWarmup anchors the actual DP-work limit.
// The protocol may admit a single exact scan while a timed two-arm run must fail.
func TestComparator_BudgetCoversBothArmsAndWarmup(t *testing.T) {
	experiment := comparisonFixture(t)
	const exactCells = 3 * 12 * 24 * 24
	experiment.Protocol.MaxCells = exactCells
	if err := tfsax.ValidateProtocol(experiment.Protocol); err != nil {
		t.Fatal(err)
	}
	result, err := tfsax.RunAvailableComparison(context.Background(), experiment,
		tfsax.ComparisonOptions{SAXSegments: 6, SAXAlphabet: 8})
	if result != nil || !errors.Is(err, tfsax.ErrBudget) {
		t.Fatalf("second DTW branch was not budgeted: %+v / %v", result, err)
	}
	// One exact scan plus eight exact-refined candidates per query is the two-arm
	// baseline. The boundary is inclusive for integer arithmetic.
	experiment.Protocol.MaxCells = 3 * (12 + 8) * 24 * 24
	if _, err := tfsax.RunAvailableComparison(context.Background(), experiment,
		tfsax.ComparisonOptions{SAXSegments: 6, SAXAlphabet: 8}); err != nil {
		t.Fatalf("exact two-arm budget must be sufficient: %v", err)
	}
	_, err = tfsax.RunAvailableComparison(context.Background(), experiment,
		tfsax.ComparisonOptions{SAXSegments: 6, SAXAlphabet: 8, WarmupRuns: 1})
	mustErrorIs(t, err, tfsax.ErrBudget)
}
