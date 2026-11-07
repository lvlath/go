// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test verifies exact search ranking against independently enumerated DTW paths.
// The oracle does not import private production recurrence helpers or rely on tolerance.
package dtw_test

import (
	"context"
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/lvlath/go/dtw"
)

// enumerateAllSearchPaths computes a tiny squared-DTW reference by enumerating
// every legal monotone path. Small integer-valued inputs keep each summed path
// exact in binary64, so the comparison is not vulnerable to rounding reassociation.
func enumerateAllSearchPaths(query, candidate []float64, radius int) float64 {
	length := len(query)
	best := math.Inf(1)
	var enumerate func(i, j int, accumulated float64)
	enumerate = func(i, j int, accumulated float64) {
		if radius >= 0 && absOracleIndex(i-j) > radius {
			return
		}
		difference := query[i] - candidate[j]
		next := accumulated + difference*difference
		if i == length-1 && j == length-1 {
			if next < best {
				best = next
			}
			return
		}
		if i+1 < length && j+1 < length {
			enumerate(i+1, j+1, next)
		}
		if i+1 < length {
			enumerate(i+1, j, next)
		}
		if j+1 < length {
			enumerate(i, j+1, next)
		}
	}
	enumerate(0, 0, 0)
	return best
}

// absOracleIndex returns a nonnegative difference of two bounded tiny indices.
func absOracleIndex(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// exactSearchOracle ranks all possible fixed-length starts by a separate
// combinatorial path enumeration and exact lexicographic tuple comparison.
func exactSearchOracle(query, data []float64, radius, stride, topK int) []dtw.SearchHit {
	var hits []dtw.SearchHit
	for start := 0; start+len(query) <= len(data); start += stride {
		hits = append(hits, dtw.SearchHit{
			Window:          dtw.IndexRange{Start: start, End: start + len(query)},
			SquaredDistance: enumerateAllSearchPaths(query, data[start:start+len(query)], radius),
		})
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].SquaredDistance != hits[j].SquaredDistance {
			return hits[i].SquaredDistance < hits[j].SquaredDistance
		}
		if hits[i].Window.Start != hits[j].Window.Start {
			return hits[i].Window.Start < hits[j].Window.Start
		}
		return hits[i].Window.End < hits[j].Window.End
	})
	if len(hits) > topK {
		hits = hits[:topK]
	}
	return hits
}

// assertExactHits checks every finite score and coordinate exactly for tiny integer fixtures.
func assertExactHits(t *testing.T, got, want []dtw.SearchHit) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("hits=%v; want=%v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("hits[%d]=%+v; want=%+v; all=%v", i, got[i], want[i], got)
		}
	}
}

// TestSearch_ExactTopKAgainstExhaustive validates every selected hit against
// all legal monotone paths, independent of the DP implementation and tie heuristics.
func TestSearch_ExactTopKAgainstExhaustive(t *testing.T) {
	rng := rand.New(rand.NewSource(20261106))
	for length := 1; length <= 6; length++ {
		for trial := 0; trial < 20; trial++ {
			query := make([]float64, length)
			data := make([]float64, length+4)
			for i := range query {
				query[i] = float64(rng.Intn(7) - 3)
			}
			for i := range data {
				data[i] = float64(rng.Intn(7) - 3)
			}
			for _, radius := range []int{-1, 0, 1, 2} {
				for _, stride := range []int{1, 2} {
					opts := dtw.DefaultSearchOptions()
					opts.Radius, opts.Stride, opts.TopK = radius, stride, 3
					result, err := dtw.Search(context.Background(), query, data, opts)
					if err != nil || result == nil || !result.Complete {
						t.Fatalf("L=%d trial=%d radius=%d stride=%d: result=%+v err=%v", length, trial, radius, stride, result, err)
					}
					assertExactHits(t, result.Hits, exactSearchOracle(query, data, radius, stride, opts.TopK))
					if result.CandidatesCompleted+result.CandidatesPruned != result.CandidatesTotal {
						t.Fatalf("inconsistent exhaustive accounting: %+v", result)
					}
				}
			}
		}
	}
}
