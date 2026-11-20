// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
package tfsax_test

import (
	"context"
	"errors"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/lvlath/go/tfsax"
)

func TestEightFamiliesAndHashes(t *testing.T) {
	families := []tfsax.Family{tfsax.FamilyConstant, tfsax.FamilyHeavyTail, tfsax.FamilyWarp, tfsax.FamilySlope, tfsax.FamilyMotif, tfsax.FamilyPhase, tfsax.FamilyAffine, tfsax.FamilyPlateau}
	hashes := make(map[string]bool)
	for _, f := range families {
		c, err := tfsax.GenerateFamily(f, 27, 4, 16)
		if err != nil {
			t.Fatal(f, err)
		}
		hash, err := tfsax.CorpusHash(c)
		if err != nil {
			t.Fatal(err)
		}
		if hashes[hash] {
			t.Fatalf("family hash collision %s", f)
		}
		hashes[hash] = true
	}
	if _, err := tfsax.GenerateFamily("unrecognized", 1, 4, 16); !errors.Is(err, tfsax.ErrInvalidOption) {
		t.Fatal(err)
	}
	if _, err := tfsax.GenerateFamily(tfsax.FamilyPhase, 0, 100001, 1); !errors.Is(err, tfsax.ErrBudget) {
		t.Fatal(err)
	}
}
func TestProtocolValidationMatrix(t *testing.T) {
	good := tfsax.DefaultProtocol()
	if err := tfsax.ValidateProtocol(good); err != nil {
		t.Fatal(err)
	}
	bad := good
	bad.Length = 0
	mustErrorIs(t, tfsax.ValidateProtocol(bad), tfsax.ErrInvalidOption)
	bad = good
	bad.MaxCells = 1
	mustErrorIs(t, tfsax.ValidateProtocol(bad), tfsax.ErrBudget)
	bad = good
	bad.Thresholds.RecallMin = math.NaN()
	mustErrorIs(t, tfsax.ValidateProtocol(bad), tfsax.ErrInvalidOption)
	bad = good
	bad.Window = -2
	mustErrorIs(t, tfsax.ValidateProtocol(bad), tfsax.ErrInvalidOption)
	bad = good
	bad.Shortlist = 0
	mustErrorIs(t, tfsax.ValidateProtocol(bad), tfsax.ErrInvalidOption)
}
func TestQueryAndCorpusInvalidity(t *testing.T) {
	c := []tfsax.Candidate{{ID: 1, Values: []float64{1, 2}}}
	if err := tfsax.ValidateCorpus(c, ""); err != nil {
		t.Fatal(err)
	}
	if err := tfsax.ValidateQuery(tfsax.Query{Values: []float64{1, 2}, TopK: 1}, 2); err != nil {
		t.Fatal(err)
	}
	c[0].Values[0] = math.Inf(1)
	mustErrorIs(t, tfsax.ValidateCorpus(c, ""), tfsax.ErrNonFinite)
	mustErrorIs(t, tfsax.ValidateQuery(tfsax.Query{Values: []float64{math.NaN(), 1}, TopK: 1}, 2), tfsax.ErrNonFinite)
	mustErrorIs(t, tfsax.ValidateQuery(tfsax.Query{Values: []float64{1, 2}, TopK: 0}, 2), tfsax.ErrInvalidInput)
	mustErrorIs(t, tfsax.ValidateCorpus(nil, ""), tfsax.ErrInvalidInput)
}
func TestReferenceAndHeuristicMismatch(t *testing.T) {
	c := []tfsax.Candidate{{ID: 1, Values: []float64{0, 0, 0, 0}}, {ID: 4, Values: []float64{2, 2, 2, 2}}}
	q := tfsax.Query{Values: []float64{0, 0, 0, 0}, TopK: 5}
	r, err := tfsax.ScanExact(context.Background(), c, q, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Candidates) != 2 || r.Diagnostics.Examined != 2 {
		t.Fatal(r)
	}
	noScore := func([]float64, []float64) (float64, error) { return 0, nil }
	a, err := tfsax.RefineShortlist(context.Background(), c, q, 0, 1, "rank-1", noScore)
	if err != nil {
		t.Fatal(err)
	}
	if a.Diagnostics.Pruned != 1 || a.Diagnostics.Exact || len(a.Candidates) != 1 {
		t.Fatal(a)
	}
	rankNaN := func([]float64, []float64) (float64, error) { return math.NaN(), nil }
	_, err = tfsax.RefineShortlist(context.Background(), c, q, 0, 1, "rank-1", rankNaN)
	mustErrorIs(t, err, tfsax.ErrNonFinite)
	_, err = tfsax.RefineShortlist(context.Background(), c, q, 0, 1, "rank-1", nil)
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	_, err = tfsax.ScanExact(nil, c, q, 0)
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = tfsax.ScanExact(ctx, c, q, 0)
	mustErrorIs(t, err, context.Canceled)
}
func TestMetricsAndLatency(t *testing.T) {
	a := tfsax.RecallAtK([]tfsax.CandidateID{1, 1, 2}, []tfsax.CandidateID{1, 2, 3})
	if !a.Applicable || math.Abs(a.Value-2.0/3) > 1e-12 {
		t.Fatal(a)
	}
	b := tfsax.MissedBest([]tfsax.CandidateID{3}, []tfsax.CandidateID{2, 3})
	if !b.Applicable || b.Value != 1 {
		t.Fatal(b)
	}
	latency, err := tfsax.LatencyQuantiles([]int64{90, 10, 30, 20, 40})
	if err != nil {
		t.Fatal(err)
	}
	if latency.P50 != 30 || latency.P95 != 90 || latency.P99 != 90 {
		t.Fatal(latency)
	}
	_, err = tfsax.LatencyQuantiles([]int64{-1})
	mustErrorIs(t, err, tfsax.ErrInvalidInput)
	_, err = tfsax.LatencyQuantiles(nil)
	mustErrorIs(t, err, tfsax.ErrInvalidInput)
}
func TestExactReferenceExperimentAndReports(t *testing.T) {
	p := tfsax.DefaultProtocol()
	p.Length = 8
	p.CorpusSize = 4
	p.QueryCount = 1
	p.Shortlist = 2
	p.TopK = 2
	p.Purge = 8
	c, err := tfsax.GenerateFamily(tfsax.FamilyPhase, 32, 4, 8)
	if err != nil {
		t.Fatal(err)
	}
	hash, err := tfsax.CorpusHash(c)
	if err != nil {
		t.Fatal(err)
	}
	e := tfsax.Experiment{Protocol: p, Definition: tfsax.UnresolvedDefinition(), Corpus: c, CorpusDigest: hash, Queries: []tfsax.NamedQuery{{ID: 1, Family: tfsax.FamilyPhase, Query: tfsax.Query{Values: c[0].Values, TopK: 2}}}}
	obs, err := tfsax.RunExactReference(context.Background(), e)
	if err != nil {
		t.Fatal(err)
	}
	if len(obs) != 1 || !obs[0].Result.Diagnostics.Exact {
		t.Fatal(obs)
	}
	if _, err = tfsax.Run(context.Background(), e); !errors.Is(err, tfsax.ErrUndefinedRepresentation) {
		t.Fatalf("definition gate missing: %v", err)
	}
	path := t.TempDir() + "/raw.csv"
	if err = tfsax.WriteCSV(path, obs); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "query_id,family,branch") {
		t.Fatal("CSV schema invalid")
	}
	d, err := tfsax.DigestJSON(p)
	if err != nil || len(d) != 64 {
		t.Fatal(d, err)
	}
	if err = tfsax.WriteJSON(t.TempDir()+"/protocol.json", p); err != nil {
		t.Fatal(err)
	}
	neg := tfsax.EvaluateGate(tfsax.Definition{Approved: true}, obs)
	if neg.Passed || neg.Reason == "" {
		t.Fatal(neg)
	}
	if err = tfsax.WriteNegativeReport(t.TempDir()+"/negative.md", neg); err != nil {
		t.Fatal(err)
	}
	neg.Passed = true
	mustErrorIs(t, tfsax.WriteNegativeReport(t.TempDir()+"/bad.md", neg), tfsax.ErrInvalidOption)
}
func TestSAXAdapterAndGate(t *testing.T) {
	c, err := tfsax.GenerateFamily(tfsax.FamilyAffine, 42, 8, 16)
	if err != nil {
		t.Fatal(err)
	}
	q := tfsax.Query{Values: c[0].Values, TopK: 2}
	r, err := tfsax.ScanSAX(context.Background(), c, q, 4, 4, 4, 8)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Complete || r.Diagnostics.Exact || r.Diagnostics.Pruned != 4 {
		t.Fatal(r)
	}
	_, err = tfsax.ScanSAX(context.Background(), c, q, 4, 4, 0, 8)
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
}
