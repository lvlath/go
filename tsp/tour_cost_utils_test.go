// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test validates tour utilities and cost routines in lvlath/tsp.
// Contract: strict sentinels, deterministic outcomes, table-driven structure.
package tsp_test

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

//
// Helpers
//

func clone2D(a [][]float64) [][]float64 {
	cp := make([][]float64, len(a))
	for i := range a {
		cp[i] = append([]float64(nil), a[i]...)
	}
	return cp
}

func withEdge(a [][]float64, i, j int, w float64) matrix.Matrix {
	cp := clone2D(a)
	cp[i][j] = w
	cp[j][i] = w // symmetric for these tests
	return testDense{a: cp}
}

func round1e9(x float64) int64 { return int64(math.Round(x * 1e9)) }

//
// 1) Validation - tsp.ValidateTour
//

func TestValidateTour_InvalidLength_Duplicates_OOB(t *testing.T) {
	const n = 4
	const start = 0

	t.Run("length != n → ErrInvalidTour", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			tour := []int{0, 1, 2} // len=3, expect n=4
			err := tsp.ValidateTour(tour, n, start)
			if !errors.Is(err, tsp.ErrInvalidTour) {
				t.Fatalf("want ErrInvalidTour, got %v", err)
			}
		})
	})

	t.Run("duplicates → ErrInvalidTour", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			tour := []int{0, 1, 1, 3}
			err := tsp.ValidateTour(tour, n, start)
			if !errors.Is(err, tsp.ErrInvalidTour) {
				t.Fatalf("want ErrInvalidTour, got %v", err)
			}
		})
	})

	t.Run("out-of-range → ErrInvalidTour", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			tour := []int{0, 1, 2, 9}
			err := tsp.ValidateTour(tour, n, start)
			if !errors.Is(err, tsp.ErrInvalidTour) {
				t.Fatalf("want ErrInvalidTour, got %v", err)
			}
		})
	})
}

//
// 2) Validation - MakeTourFromPermutation, RotateTourToStart
//

func TestMakeTourFromPermutation_StartMissing_ErrDimensionMismatch(t *testing.T) {
	const n = 5
	perm := []int{1, 2, 3, 4} // start=0 is absent

	_, err := tsp.MakeTourFromPermutation(perm, n, 0)
	if !errors.Is(err, tsp.ErrDimensionMismatch) {
		t.Fatalf("want ErrDimensionMismatch, got %v", err)
	}
}

func TestRotateTourToStart_StartNotFound_ErrDimensionMismatch(t *testing.T) {
	tour := []int{3, 4, 5, 6}
	_, err := tsp.RotateTourToStart(tour, 2) // 2 is not in tour
	if !errors.Is(err, tsp.ErrDimensionMismatch) {
		t.Fatalf("want ErrDimensionMismatch, got %v", err)
	}
}

//
// 3) Validation - TourCost error paths (+Inf/−Inf, negative, NaN)
//    IMPORTANT: tsp.TourCost (в текущей реализации) суммирует РЕБРА ОТКРЫТОГО ПУТИ,
//    т.е. пары (tour[i], tour[i+1]) без автоматического "замыкания" на старт.
//    Поэтому проверяемые ребра должны лежать на этих парах.
//

func TestTourCost_StrictSentinels_OnBadEdges(t *testing.T) {
	base := [][]float64{
		{0, 1, 2},
		{1, 0, 3},
		{2, 3, 0},
	}

	t.Run("+Inf edge → ErrIncompleteGraph", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			// Edge used by path: (0,1)
			m := withEdge(base, 0, 1, math.Inf(+1))
			tour := []int{0, 1, 2}
			_, err := tsp.TourCost(m, tour)
			if !errors.Is(err, tsp.ErrIncompleteGraph) {
				t.Fatalf("want ErrIncompleteGraph, got %v", err)
			}
		})
	})

	t.Run("−Inf edge → ErrIncompleteGraph", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			// Edge used by path: (1,2)
			m := withEdge(base, 1, 2, math.Inf(-1))
			tour := []int{0, 1, 2}
			_, err := tsp.TourCost(m, tour)
			if !errors.Is(err, tsp.ErrIncompleteGraph) {
				t.Fatalf("want ErrIncompleteGraph, got %v", err)
			}
		})
	})

	t.Run("negative edge → ErrNegativeWeight", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			// Put negative on an EDGE USED BY PATH (1,2), not on (0,2).
			m := withEdge(base, 1, 2, -5)
			tour := []int{0, 1, 2}
			_, err := tsp.TourCost(m, tour)
			if !errors.Is(err, tsp.ErrNegativeWeight) {
				t.Fatalf("want ErrNegativeWeight, got %v", err)
			}
		})
	})

	t.Run("NaN edge → ErrDimensionMismatch", func(t *testing.T) {
		Repeat(t, 3, func(t *testing.T) {
			// Place NaN on an EDGE USED BY PATH (0,1), not on (0,2).
			m := withEdge(base, 0, 1, math.NaN())
			tour := []int{0, 1, 2}
			_, err := tsp.TourCost(m, tour)
			if !errors.Is(err, tsp.ErrNaNInf) {
				t.Fatalf("want ErrNaNInf, got %v", err)
			}
		})
	})
}

//
// 4) Medium - CanonicalizeOrientationInPlace
//     If tour[1] > tour[n-1], reverse segment [1..n-1] in-place.
//

func TestCanonicalizeOrientationInPlace(t *testing.T) {
	t.Run("mirrors [1..n-1] when left-neighbor > right", func(t *testing.T) {
		// IMPORTANT: vertices must be in-range 0..n-1; use 4, not 5.
		tour := []int{0, 4, 1, 2, 3, 0} // tour[1]=4 > tour[n-1]=3 ⇒ mirror [1..4]
		want := []int{0, 3, 2, 1, 4, 0}

		if err := tsp.CanonicalizeOrientationInPlace(tour); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(tour, want) {
			t.Fatalf("canonicalize mismatch:\n got:  %v\n want: %v", tour, want)
		}
	})

	t.Run("keeps orientation when left-neighbor ≤ right", func(t *testing.T) {
		tour := []int{0, 1, 2, 3, 4, 0} // 1 ≤ 4 ⇒ no change
		want := append([]int(nil), tour...)

		if err := tsp.CanonicalizeOrientationInPlace(tour); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(tour, want) {
			t.Fatalf("unexpected change:\n got:  %v\n want: %v", tour, want)
		}
	})
}

//
// 5) Medium - ShortcutEulerianToHamiltonian
//

func TestShortcutEulerianToHamiltonian(t *testing.T) {
	// Eulerian walk over {0,1,2,3} with repeats; shortcut removes revisits.
	euler := []int{0, 1, 2, 1, 3, 0}

	h, err := tsp.ShortcutEulerianToHamiltonian(euler, 4, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Current implementation returns a CLOSED cycle (n+1 with return to start).
	// Accept either clockwise or mirrored orientation; both end at 0.
	wantA := []int{0, 1, 2, 3, 0}
	wantB := []int{0, 3, 2, 1, 0}

	if !reflect.DeepEqual(h, wantA) && !reflect.DeepEqual(h, wantB) {
		t.Fatalf("shortcut result mismatch:\n got:  %v\n want: %v or %v", h, wantA, wantB)
	}
}

//
// 6) Special - identical cost across different Matrix implementations
//

func TestTourCost_IdenticalAcrossImplementations(t *testing.T) {
	tour := []int{0, 1, 2, 3} // open path cost
	a := [][]float64{
		{0, 1, 9, 4},
		{1, 0, 5, 6},
		{9, 5, 0, 2},
		{4, 6, 2, 0},
	}

	m1 := testDense{a: clone2D(a)}
	m2 := altDense{a: clone2D(a)}

	c1, err1 := tsp.TourCost(m1, tour)
	c2, err2 := tsp.TourCost(m2, tour)

	if err1 != nil || err2 != nil {
		t.Fatalf("unexpected errors: err1=%v err2=%v", err1, err2)
	}
	if round1e9(c1) != round1e9(c2) {
		t.Fatalf("cost mismatch across impls: c1=%.12f c2=%.12f", c1, c2)
	}
}
