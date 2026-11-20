// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
package tfsax_test

import (
	"context"
	"errors"
	"testing"

	"github.com/lvlath/go/tfsax"
)

func TestTF01TiesByCandidateID(t *testing.T) {
	corpus := []tfsax.Candidate{{ID: 7, Values: []float64{0, 1, 2, 3}}, {ID: 2, Values: []float64{0, 1, 2, 3}}}
	res, err := tfsax.ScanExact(context.Background(), corpus, tfsax.Query{Values: []float64{0, 1, 2, 3}, TopK: 2}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Candidates) != 2 || res.Candidates[0] != 2 || res.Candidates[1] != 7 || !res.Diagnostics.Exact || !res.Complete {
		t.Fatalf("bad deterministic tie: %+v", res)
	}
	res.Candidates[0] = 9
	if corpus[1].ID != 2 {
		t.Fatal("candidate alias leaked")
	}
}
func TestTF02MissingBest(t *testing.T) {
	corpus := []tfsax.Candidate{{ID: 1, Values: []float64{0, 0, 0, 0}}, {ID: 2, Values: []float64{9, 9, 9, 9}}}
	q := tfsax.Query{Values: []float64{0, 0, 0, 0}, TopK: 1}
	rank := func(_, candidate []float64) (float64, error) {
		if candidate[0] > 0 {
			return 0, nil
		}
		return 1, nil
	}
	approximate, err := tfsax.RefineShortlist(context.Background(), []tfsax.Candidate{corpus[1], corpus[0]}, q, 0, 1, "intentionally-bad", rank)
	if err != nil {
		t.Fatal(err)
	}
	exact, err := tfsax.ScanExact(context.Background(), corpus, q, 0)
	if err != nil {
		t.Fatal(err)
	}
	m := tfsax.MissedBest(approximate.Candidates, exact.Candidates)
	if !m.Applicable || m.Value != 1 || approximate.Diagnostics.Exact || !approximate.Complete {
		t.Fatalf("false negative hidden: %+v / %+v", approximate, m)
	}
}
func TestTF03Leakage(t *testing.T) {
	p := tfsax.SplitPlan{Train: tfsax.Interval{Start: 0, End: 100}, Calibration: tfsax.Interval{Start: 140, End: 200}, Test: tfsax.Interval{Start: 240, End: 300}, TrainIDs: []tfsax.CandidateID{1}, TestIDs: []tfsax.CandidateID{5}, FitIDs: []tfsax.CandidateID{1, 5}}
	if !errors.Is(tfsax.ValidateSplit(p, 32, 4, 32, 0), tfsax.ErrLeakage) {
		t.Fatal("test ID admitted to fit")
	}
	p.FitIDs = []tfsax.CandidateID{1}
	if err := tfsax.ValidateSplit(p, 32, 4, 32, 0); err != nil {
		t.Fatal(err)
	}
}
func TestTF05AllUnreachable(t *testing.T) {
	c := []tfsax.Candidate{{ID: 2, Values: []float64{1, 2, 3, 4}}}
	r, err := tfsax.ScanExact(context.Background(), c, tfsax.Query{Values: []float64{1, 2, 3, 4}, TopK: 1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Equal-length diagonal band never creates structural unreachability.
	// Test NA independently to avoid faking an unreachable scalar DTW case.
	if !r.Diagnostics.Exact || !r.Complete {
		t.Fatal("exact full scan must terminate")
	}
	if tfsax.RecallAtK(nil, nil).Applicable || tfsax.MissedBest(nil, nil).Applicable {
		t.Fatal("empty oracle truth must be NA")
	}
}
func TestTF06DuplicateAndDigest(t *testing.T) {
	c := []tfsax.Candidate{{ID: 4, Values: []float64{1, 2}}, {ID: 4, Values: []float64{3, 4}}}
	if !errors.Is(tfsax.ValidateCorpus(c, ""), tfsax.ErrDuplicateID) {
		t.Fatal("duplicate accepted")
	}
	c[1].ID = 5
	hash, err := tfsax.CorpusHash(c)
	if err != nil {
		t.Fatal(err)
	}
	c[1].Values[0] = 10
	if !errors.Is(tfsax.ValidateCorpus(c, hash), tfsax.ErrInvalidInput) {
		t.Fatal("tampering not rejected")
	}
}
func TestTF07Determinism(t *testing.T) {
	x, err := tfsax.GenerateFamily(tfsax.FamilyPhase, 31, 10, 16)
	if err != nil {
		t.Fatal(err)
	}
	y, err := tfsax.GenerateFamily(tfsax.FamilyPhase, 31, 10, 16)
	if err != nil {
		t.Fatal(err)
	}
	a, err := tfsax.CorpusHash(x)
	if err != nil {
		t.Fatal(err)
	}
	b, err := tfsax.CorpusHash(y)
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Fatal("nonreproducible series")
	}
	q := tfsax.Query{Values: x[3].Values, TopK: 3}
	rx, err := tfsax.ScanExact(context.Background(), x, q, 2)
	if err != nil {
		t.Fatal(err)
	}
	ry, err := tfsax.ScanExact(context.Background(), y, q, 2)
	if err != nil {
		t.Fatal(err)
	}
	for i, id := range rx.Candidates {
		if id != ry.Candidates[i] {
			t.Fatal("nonreproducible results")
		}
	}
}
func TestTF08NegativeGate(t *testing.T) {
	r := tfsax.EvaluateGate(tfsax.UnresolvedDefinition(), nil)
	if r.Passed || r.CompleteComparison || r.DefinitionApproved {
		t.Fatalf("false positive gate %+v", r)
	}
	if err := tfsax.WriteNegativeReport(t.TempDir()+"/report.md", r); err != nil {
		t.Fatal(err)
	}
}
func TestResult_IsNilInterface(t *testing.T) {
	var r *tfsax.RetrievalResult
	var nilable interface{ IsNil() bool } = r
	if !nilable.IsNil() {
		t.Fatal("nil pointer not recognized")
	}
	r = &tfsax.RetrievalResult{}
	if r.IsNil() {
		t.Fatal("zero-valued object called nil")
	}
}
func TestMissingDefinitionStopsExperiment(t *testing.T) {
	if _, err := tfsax.Run(context.Background(), tfsax.Experiment{}); !errors.Is(err, tfsax.ErrInvalidOption) {
		t.Fatalf("bad protocol must reject: %v", err)
	}
}

// TestPlateauFamily_IndependentProfiles verifies a flat stress segment without
// turning the full dataset into identical self-matching trajectories.
func TestPlateauFamily_IndependentProfiles(t *testing.T) {
	first, err := tfsax.GenerateFamily(tfsax.FamilyPlateau, 29, 3, 48)
	if err != nil {
		t.Fatal(err)
	}
	second, err := tfsax.GenerateFamily(tfsax.FamilyPlateau, 30, 3, 48)
	if err != nil {
		t.Fatal(err)
	}
	firstHash, err := tfsax.CorpusHash(first)
	if err != nil {
		t.Fatal(err)
	}
	secondHash, err := tfsax.CorpusHash(second)
	if err != nil || firstHash == secondHash {
		t.Fatalf("plateau windows ignored independent seed or failed hashing: %v", err)
	}
	for _, candidate := range first {
		longestFlatRun := 1
		flatRun := 1
		for i := 1; i < len(candidate.Values); i++ {
			if candidate.Values[i] == candidate.Values[i-1] {
				flatRun++
				if flatRun > longestFlatRun {
					longestFlatRun = flatRun
				}
			} else {
				flatRun = 1
			}
		}
		if longestFlatRun < 48/3 {
			t.Fatalf("plateau family lost its exact constant support: id=%d", candidate.ID)
		}
	}
}
