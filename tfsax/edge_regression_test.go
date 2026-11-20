// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax_test anchors numerical, cancellation and corpus-identity regressions.
// Cases use public facades and an independent small-space dynamic-programming oracle.
package tfsax_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/tfsax"
)

func TestExactWrappers_RealDistancePolicyAndNil(t *testing.T) {
	corpus := []tfsax.Candidate{
		{ID: 77, Values: []float64{0, 1, 1, 1}},
		{ID: 22, Values: []float64{0, 0, 1, 1}},
	}
	query := tfsax.Query{Values: []float64{0, 0, 1, 1}, TopK: 2}
	unbounded, err := tfsax.ScanExactUnbounded(context.Background(), corpus, query)
	if err != nil {
		t.Fatal(err)
	}
	diagonal, err := tfsax.ScanExactDiagonal(context.Background(), corpus, query)
	if err != nil {
		t.Fatal(err)
	}
	mustIDs(t, unbounded.Candidates, []tfsax.CandidateID{22, 77})
	mustIDs(t, diagonal.Candidates, []tfsax.CandidateID{22, 77})
	if !unbounded.Diagnostics.Exact || !diagonal.Diagnostics.Exact {
		t.Fatal("exact wrappers lost proof status")
	}
	mustNilState(t, (*tfsax.RetrievalResult)(nil), true, "typed nil retrieval")
	mustNilState(t, &tfsax.RetrievalResult{}, false, "non-nil empty retrieval")
}

func TestReferenceDTW_PreservesRootAndBoundarySentinels(t *testing.T) {
	_, _, err := tfsax.ExactDTW(nil, []float64{0}, 0)
	mustErrorIs(t, err, tfsax.ErrInvalidInput)
	mustErrorIs(t, err, dtw.ErrNilInput)
	_, _, err = tfsax.ExactDTW([]float64{0}, []float64{math.NaN()}, 0)
	mustErrorIs(t, err, tfsax.ErrNonFinite)
	mustErrorIs(t, err, dtw.ErrNaNInf)
	_, _, err = tfsax.ExactDTW([]float64{0}, []float64{0}, -2)
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	mustErrorIs(t, err, dtw.ErrInvalidWindow)
	_, _, err = tfsax.ExactDTW([]float64{math.MaxFloat64}, []float64{-math.MaxFloat64}, 0)
	mustErrorIs(t, err, tfsax.ErrOverflow)
	mustErrorIs(t, err, dtw.ErrOverflow)
}

// TestReferenceDTW_SeparatesLocalAndAccumulatedOverflow checks numeric error
// categories independently from the traversal tie-break and retrieval layers.
// Finite inputs can overflow either at local subtraction or during DP summation;
// invalid samples must never acquire an arithmetic-overflow classification.
func TestReferenceDTW_SeparatesLocalAndAccumulatedOverflow(t *testing.T) {
	maximum := math.MaxFloat64
	quarter := maximum / 4
	cases := []struct {
		name         string
		a, b         []float64
		window       int
		wantDistance float64
		wantErr      error
		wantDTWErr   error
	}{
		{
			name: "finite-local-cost-overflow",
			a:    []float64{maximum}, b: []float64{-maximum}, window: 0,
			wantErr: tfsax.ErrOverflow, wantDTWErr: dtw.ErrOverflow,
		},
		{
			name: "finite-accumulated-cost-overflow",
			a:    []float64{quarter, quarter, quarter},
			b:    []float64{-quarter, -quarter, -quarter}, window: 0,
			wantErr: tfsax.ErrOverflow, wantDTWErr: dtw.ErrOverflow,
		},
		{
			name: "nonfinite-input",
			a:    []float64{1}, b: []float64{math.Inf(1)}, window: 0,
			wantErr: tfsax.ErrNonFinite, wantDTWErr: dtw.ErrNaNInf,
		},
		{
			name: "representable-difference",
			a:    []float64{quarter}, b: []float64{-quarter}, window: 0,
			wantDistance: maximum / 2,
		},
		{
			name: "unvisited-off-band-overflow",
			a:    []float64{maximum, 0, 0},
			b:    []float64{maximum, 0, -maximum}, window: 0,
			wantDistance: maximum,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			distance, reachable, err := tfsax.ExactDTW(tc.a, tc.b, tc.window)
			if tc.wantErr == nil {
				if err != nil || !reachable || distance != tc.wantDistance {
					t.Fatalf("finite alignment got distance=%g reachable=%t error=%v, want distance=%g", distance, reachable, err, tc.wantDistance)
				}
				return
			}
			if distance != 0 || reachable {
				t.Fatalf("failed alignment published distance=%g reachable=%t", distance, reachable)
			}
			mustErrorIs(t, err, tc.wantErr)
			mustErrorIs(t, err, tc.wantDTWErr)
			if errors.Is(tc.wantErr, tfsax.ErrOverflow) {
				// Upstream DTW uses this companion classification for numeric
				// overflow; callers must still distinguish invalid inputs.
				mustErrorIs(t, err, dtw.ErrNaNInf)
			}
			if errors.Is(tc.wantErr, tfsax.ErrOverflow) && errors.Is(err, tfsax.ErrNonFinite) {
				t.Fatalf("finite arithmetic overflow misclassified as invalid input: %v", err)
			}
			if errors.Is(tc.wantErr, tfsax.ErrNonFinite) && errors.Is(err, tfsax.ErrOverflow) {
				t.Fatalf("invalid input misclassified as arithmetic overflow: %v", err)
			}
		})
	}
}

// TestReferenceDTW_PreservesCanonicalFiniteDistances demonstrates that the
// overflow classification adapter does not alter the accepted DTW mathematics.
// Exact binary64 distances and reachability must match upstream dtw.Align for
// different admissible bands, unequal lengths and subnormal input differences.
func TestReferenceDTW_PreservesCanonicalFiniteDistances(t *testing.T) {
	tests := []struct {
		name   string
		a, b   []float64
		window int
	}{
		{
			name: "bounded-timing-drift",
			a:    []float64{0, 1, 4, 3, -2, 0},
			b:    []float64{0, 0, 1, 4, 3, -2}, window: 2,
		},
		{
			name: "unrestricted-timing-drift",
			a:    []float64{0, 1, 4, 3, -2, 0},
			b:    []float64{0, 0, 1, 4, 3, -2}, window: -1,
		},
		{
			name: "diagonal-only",
			a:    []float64{0, 1, 4, 3, -2, 0},
			b:    []float64{0, 0, 1, 4, 3, -2}, window: 0,
		},
		{
			name: "unequal-length-with-path",
			a:    []float64{1, 2, 3, 4, 5},
			b:    []float64{1, 2, 4}, window: 2,
		},
		{
			name: "unequal-length-no-path",
			a:    []float64{1, 2, 3, 4, 5},
			b:    []float64{1, 2, 4}, window: 0,
		},
		{
			name: "signed-zero-and-subnormals",
			a:    []float64{math.Copysign(0, -1), math.SmallestNonzeroFloat64},
			b:    []float64{0, -math.SmallestNonzeroFloat64}, window: 0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			upstream, err := dtw.Align(tc.a, tc.b, dtw.WithWindow(tc.window))
			if err != nil || upstream == nil {
				t.Fatalf("upstream oracle failure: %v", err)
			}
			got, reachable, err := tfsax.ExactDTW(tc.a, tc.b, tc.window)
			if err != nil || upstream.Reachable != reachable ||
				math.Float64bits(upstream.Distance) != math.Float64bits(got) {
				t.Fatalf("adapter changed DTW: upstream=(%g,%t) adapter=(%g,%t) err=%v",
					upstream.Distance, upstream.Reachable, got, reachable, err)
			}
		})
	}
}

func TestRankerCallbackAndCancellation_NoPartial(t *testing.T) {
	corpus := []tfsax.Candidate{
		{ID: 7, Values: []float64{0, 0, 0, 0}},
		{ID: 2, Values: []float64{0, 0, 0, 0}},
		{ID: 9, Values: []float64{2, 2, 2, 2}},
	}
	query := tfsax.Query{Values: []float64{0, 0, 0, 0}, TopK: 1}
	rank := func(_, _ []float64) (float64, error) { return 0, nil }
	shortlist, err := tfsax.RefineShortlist(context.Background(), corpus, query, 0, 2, "constant-score", rank)
	if err != nil {
		t.Fatal(err)
	}
	mustIDs(t, shortlist.Candidates, []tfsax.CandidateID{2})
	if shortlist.Diagnostics.Exact || shortlist.Diagnostics.Pruned != 1 {
		t.Fatal("uncertified shortlist cannot be exact")
	}
	rankError := errors.New("rank callback rejected input")
	result, err := tfsax.RefineShortlist(context.Background(), corpus, query, 0, 2, "callback-error",
		func(_, _ []float64) (float64, error) { return 0, rankError })
	if result != nil || !errors.Is(err, rankError) {
		t.Fatalf("callback error lost or partial result: %+v %v", result, err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	result, err = tfsax.RefineShortlist(ctx, corpus, query, 0, 2, "cancel-after-first-score",
		func(_, _ []float64) (float64, error) {
			calls++
			if calls == 1 {
				cancel()
			}
			return 0, nil
		})
	if result != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled rank must not publish partial result: %+v %v", result, err)
	}
	_, err = tfsax.ScanLBConservative(nil, nil, tfsax.Query{}, 0)
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	_, err = tfsax.ScanLBConservative(ctx, nil, tfsax.Query{}, 0)
	mustErrorIs(t, err, context.Canceled)
}

func TestCorpusHash_OrderAndSignedZeroIdentity(t *testing.T) {
	positive := []tfsax.Candidate{
		{ID: 2, Values: []float64{0, 1}},
		{ID: 1, Values: []float64{1, 0}},
	}
	hashPositive, err := tfsax.CorpusHash(positive)
	if err != nil {
		t.Fatal(err)
	}
	negative := []tfsax.Candidate{
		{ID: 2, Values: []float64{math.Copysign(0, -1), 1}},
		{ID: 1, Values: []float64{1, 0}},
	}
	hashNegative, err := tfsax.CorpusHash(negative)
	if err != nil {
		t.Fatal(err)
	}
	if hashNegative == hashPositive {
		t.Fatal("signed zero bytes were silently canonicalized")
	}
	reordered := []tfsax.Candidate{positive[1], positive[0]}
	hashReordered, err := tfsax.CorpusHash(reordered)
	if err != nil || hashReordered == hashPositive {
		t.Fatalf("original candidate ordering not committed: %s %v", hashReordered, err)
	}
	_, err = tfsax.CorpusHash([]tfsax.Candidate{{ID: 1, Values: nil}})
	mustErrorIs(t, err, tfsax.ErrInvalidInput)
}

// independentSmallDTW is an intentionally separate test-only oracle with full DP storage.
// It does not call the package's retrieval kernel or upstream dtw.Align.
func independentSmallDTW(a, b []float64, radius int) float64 {
	rows, cols := len(a), len(b)
	dp := make([][]float64, rows+1)
	for i := range dp {
		dp[i] = make([]float64, cols+1)
		for j := range dp[i] {
			dp[i][j] = math.Inf(1)
		}
	}
	dp[0][0] = 0
	for i := 1; i <= rows; i++ {
		for j := 1; j <= cols; j++ {
			if radius >= 0 && (i-j > radius || j-i > radius) {
				continue
			}
			local := math.Abs(a[i-1] - b[j-1])
			dp[i][j] = local + math.Min(dp[i-1][j-1], math.Min(dp[i-1][j], dp[i][j-1]))
		}
	}
	return dp[rows][cols]
}

func TestLBKeogh_ExhaustiveLengthFiveBinaryAndIndependentOracle(t *testing.T) {
	const length = 5
	for leftCode := 0; leftCode < 1<<length; leftCode++ {
		for rightCode := 0; rightCode < 1<<length; rightCode++ {
			left, right := make([]float64, length), make([]float64, length)
			for sample := 0; sample < length; sample++ {
				if leftCode&(1<<sample) != 0 {
					left[sample] = 1
				}
				if rightCode&(1<<sample) != 0 {
					right[sample] = 1
				}
			}
			for radius := 0; radius < length; radius++ {
				bound, err := tfsax.LBKeoghAbsolute(left, right, radius)
				if err != nil {
					t.Fatal(err)
				}
				upstream, reachable, err := tfsax.ExactDTW(left, right, radius)
				if err != nil || !reachable {
					t.Fatalf("upstream exact failure %v", err)
				}
				independent := independentSmallDTW(left, right, radius)
				if upstream != independent || bound < 0 || bound > upstream {
					t.Fatalf("n=5 counterexample left=%v right=%v radius=%d lower=%g upstream=%g independent=%g",
						left, right, radius, bound, upstream, independent)
				}
			}
		}
	}
}

func TestLBKeogh_InvalidShapeAndOverflow(t *testing.T) {
	_, err := tfsax.LBKeoghAbsolute(nil, []float64{0}, 0)
	mustErrorIs(t, err, tfsax.ErrInvalidInput)
	_, err = tfsax.LBKeoghAbsolute([]float64{0}, []float64{0}, 1)
	mustErrorIs(t, err, tfsax.ErrInvalidOption)
	_, err = tfsax.LBKeoghAbsolute([]float64{math.NaN()}, []float64{0}, 0)
	mustErrorIs(t, err, tfsax.ErrNonFinite)
	_, err = tfsax.LBKeoghAbsolute([]float64{math.MaxFloat64}, []float64{-math.MaxFloat64}, 0)
	mustErrorIs(t, err, tfsax.ErrOverflow)
}
