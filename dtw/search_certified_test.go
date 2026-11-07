// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test verifies that machine-certified Keogh pruning preserves exact rankings.
// Tests compare against independent path enumeration and cover runtime error preservation.
package dtw_test

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/dtw"
)

// TestSearch_CertifiedPrunesWithoutChangingTopK confirms that a strong envelope
// saves DP cells while preserving the globally best window under several radii.
func TestSearch_CertifiedPrunesWithoutChangingTopK(t *testing.T) {
	query := []float64{0, 1, 2, 3}
	data := []float64{0, 1, 2, 3, 100, 101, 102, 103, 104, 105}
	for _, radius := range []int{-1, 0, 1, 2} {
		opts := dtw.DefaultSearchOptions()
		opts.Radius = radius
		result, err := dtw.Search(context.Background(), query, data, opts)
		if err != nil || result == nil || !result.Complete {
			t.Fatalf("radius=%d: result=%+v err=%v", radius, result, err)
		}
		assertExactHits(t, result.Hits, exactSearchOracle(query, data, radius, 1, 1))
		if result.CandidatesPruned == 0 || result.CandidatesCompleted+result.CandidatesPruned != result.CandidatesTotal {
			t.Fatalf("radius=%d: certificate or accounting missing: %+v", radius, result)
		}
		if result.CellsVisited >= uint64(len(data)-len(query)+1)*uint64(len(query))*uint64(len(query)) {
			t.Fatalf("radius=%d: excessive DP visits: %+v", radius, result)
		}
	}
}

// TestSearch_CertifiedNextafterMatchesIndependentOracle probes adjacent float64
// inputs around zero and unit-scale values without using any fuzzy path tolerance.
func TestSearch_CertifiedNextafterMatchesIndependentOracle(t *testing.T) {
	values := []float64{
		0, math.SmallestNonzeroFloat64,
		math.Nextafter(1, 0), 1, math.Nextafter(1, 2),
		math.Nextafter(2, 1), 2, math.Nextafter(2, 3),
	}
	rng := rand.New(rand.NewSource(20261106))
	for length := 1; length <= 5; length++ {
		for trial := 0; trial < 80; trial++ {
			query := make([]float64, length)
			data := make([]float64, length+5)
			for i := range query {
				query[i] = values[rng.Intn(len(values))]
			}
			for i := range data {
				data[i] = values[rng.Intn(len(values))]
			}
			for _, radius := range []int{-1, 0, 1, 2} {
				opts := dtw.DefaultSearchOptions()
				opts.Radius, opts.TopK = radius, 2
				result, err := dtw.Search(context.Background(), query, data, opts)
				if err != nil || result == nil || !result.Complete {
					t.Fatalf("L=%d trial=%d radius=%d result=%+v err=%v", length, trial, radius, result, err)
				}
				assertExactHits(t, result.Hits, exactSearchOracle(query, data, radius, 1, opts.TopK))
				if result.CandidatesCompleted+result.CandidatesPruned != result.CandidatesTotal {
					t.Fatalf("accounting drift: %+v", result)
				}
			}
		}
	}
}

// TestSearch_CertifiedNeverHidesLocalOverflow ensures a large Keogh deviation
// cannot replace the scorer's ErrOverflow with a successful pruning outcome.
func TestSearch_CertifiedNeverHidesLocalOverflow(t *testing.T) {
	opts := dtw.DefaultSearchOptions()
	opts.Radius = 0
	result, err := dtw.Search(context.Background(), []float64{0, 0}, []float64{0, 0, 1e200}, opts)
	if !errors.Is(err, dtw.ErrOverflow) || !errors.Is(err, dtw.ErrNaNInf) {
		t.Fatalf("expected overflow family; result=%+v err=%v", result, err)
	}
	if result == nil || result.Complete || result.CandidatesCompleted != 1 || result.CandidatesPruned != 0 {
		t.Fatalf("local overflow cannot be silently pruned: %+v", result)
	}
}

// TestSearch_CertifiedNeverHidesFinalOverflow verifies that a provably overflowing
// diagonal disables pruning until the full recurrence classifies the final optimum.
func TestSearch_CertifiedNeverHidesFinalOverflow(t *testing.T) {
	opts := dtw.DefaultSearchOptions()
	opts.Radius = 0
	result, err := dtw.Search(context.Background(), []float64{0, 0}, []float64{0, 0, 1e154, 1e154}, opts)
	if !errors.Is(err, dtw.ErrOverflow) || result == nil || result.Complete {
		t.Fatalf("expected final overflowing candidate: result=%+v err=%v", result, err)
	}
	if result.CandidatesCompleted+result.CandidatesPruned >= result.CandidatesTotal {
		t.Fatalf("numeric error suppressed by pruning: %+v", result)
	}
}

// TestSearch_CertifiedWorkerCountRetainsGlobalRanking tests exactness under
// concurrent cutoff publication; performance counters may vary with scheduling.
func TestSearch_CertifiedWorkerCountRetainsGlobalRanking(t *testing.T) {
	query := []float64{0, 1, 2, 3, 4}
	data := []float64{0, 1, 2, 3, 4, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 9, 9, 9, 9}
	for _, workers := range []int{1, 2, 4} {
		for repeat := 0; repeat < 8; repeat++ {
			opts := dtw.DefaultSearchOptions()
			opts.Workers, opts.Radius, opts.TopK = workers, 2, 3
			result, err := dtw.Search(context.Background(), query, data, opts)
			if err != nil || !result.Complete {
				t.Fatalf("workers=%d repeat=%d result=%+v err=%v", workers, repeat, result, err)
			}
			assertExactHits(t, result.Hits, exactSearchOracle(query, data, 2, 1, 3))
			if result.CandidatesCompleted+result.CandidatesPruned != result.CandidatesTotal {
				t.Fatalf("workers=%d inconsistent accounting: %+v", workers, result)
			}
		}
	}
}

// TestSearch_CertifiedOptionalWorkspaceDoesNotBreakBudget checks that a hard
// byte limit continues to admit the existing exact scorer when an optional
// envelope would consume more memory than the caller allowed.
func TestSearch_CertifiedOptionalWorkspaceDoesNotBreakBudget(t *testing.T) {
	query := []float64{0, 1, 2, 3}
	data := []float64{0, 1, 2, 3, 100, 101, 102, 103}
	opts := dtw.DefaultSearchOptions()
	opts.Radius = 0
	opts.MaxWorkingBytes = 160
	result, err := dtw.Search(context.Background(), query, data, opts)
	if err != nil || result == nil || !result.Complete {
		t.Fatalf("exact scorer must respect old bounded workspace: result=%+v err=%v", result, err)
	}
	if result.CandidatesPruned != 0 || result.CandidatesCompleted != result.CandidatesTotal {
		t.Fatalf("optional certification must not exhaust mandatory budget: %+v", result)
	}
	assertExactHits(t, result.Hits, exactSearchOracle(query, data, 0, 1, 1))
}

// TestSearch_CertifiedAdversarialVsUnpruned checks extreme, subnormal, and
// adjacent-binary64 inputs against the same scorer with optional bounds disabled.
// This differential test protects numeric errors as well as complete rankings.
func TestSearch_CertifiedAdversarialVsUnpruned(t *testing.T) {
	values := []float64{0, math.SmallestNonzeroFloat64, math.Nextafter(1, 0), 1,
		1e-150, 1e150, 1e154, -1e154, 1e200}
	rng := rand.New(rand.NewSource(2026110619))
	for length := 4; length <= 6; length++ {
		for trial := 0; trial < 150; trial++ {
			query := make([]float64, length)
			data := make([]float64, length+3)
			for i := range query {
				query[i] = values[rng.Intn(len(values))]
			}
			for i := range data {
				data[i] = values[rng.Intn(len(values))]
			}
			baselineOpts := dtw.DefaultSearchOptions()
			baselineOpts.Radius = 0
			// The mandatory rows and heap fit; the optional envelope does not.
			baselineOpts.MaxWorkingBytes = 180
			baseline, baselineErr := dtw.Search(context.Background(), query, data, baselineOpts)
			acceleratedOpts := baselineOpts
			acceleratedOpts.MaxWorkingBytes = 0
			accelerated, acceleratedErr := dtw.Search(context.Background(), query, data, acceleratedOpts)
			if errors.Is(baselineErr, dtw.ErrOverflow) != errors.Is(acceleratedErr, dtw.ErrOverflow) ||
				(baselineErr == nil) != (acceleratedErr == nil) {
				t.Fatalf("length=%d trial=%d numeric identity changed: base=%v new=%v", length, trial, baselineErr, acceleratedErr)
			}
			if baseline == nil || accelerated == nil || baseline.Complete != accelerated.Complete {
				t.Fatalf("length=%d trial=%d completion changed: base=%+v new=%+v", length, trial, baseline, accelerated)
			}
			assertExactHits(t, accelerated.Hits, baseline.Hits)
			if baseline.CandidatesPruned != 0 {
				t.Fatalf("optional bound unexpectedly enabled under tight budget: %+v", baseline)
			}
			if accelerated.CandidatesCompleted+accelerated.CandidatesPruned !=
				baseline.CandidatesCompleted+baseline.CandidatesPruned {
				t.Fatalf("length=%d trial=%d candidate classification changed", length, trial)
			}
			if accelerated.CellsVisited > baseline.CellsVisited {
				t.Fatalf("certification increased DP work: old=%d new=%d", baseline.CellsVisited, accelerated.CellsVisited)
			}
		}
	}
}

// TestSearch_CertifiedSingleWorkerAccountingStable verifies that a single
// worker observes every committed cutoff before claiming the next window.
// The acknowledgement removes scheduler-dependent prefetch from Workers=1.
func TestSearch_CertifiedSingleWorkerAccountingStable(t *testing.T) {
	query := []float64{0, 1, 2, 3}
	data := []float64{0, 1, 2, 3, 100, 101, 102, 103, 104, 105}
	opts := dtw.DefaultSearchOptions()
	opts.Radius, opts.Workers = 0, 1
	for repeat := 0; repeat < 50; repeat++ {
		result, err := dtw.Search(context.Background(), query, data, opts)
		if err != nil || result == nil || !result.Complete ||
			result.CandidatesCompleted != 1 || result.CandidatesPruned != 6 || result.CellsVisited != 4 {
			t.Fatalf("repeat=%d: nondeterministic single-worker accounting: result=%+v err=%v", repeat, result, err)
		}
	}
}
