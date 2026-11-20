// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
package tfsax_test

import (
	"context"
	"errors"
	"testing"

	"github.com/lvlath/go/tfsax"
)

func TestTF04CounterexampleToDiagonalEuclideanBound(t *testing.T) {
	// For a diagonal-only DTW with absolute local cost, squared Euclidean is not
	// a lower bound. One counterexample invalidates unrestricted certification.
	a := []float64{0, 0, 0, 0}
	b := []float64{3, 0, 0, 0}
	d, reachable, err := tfsax.ExactDTW(a, b, 0)
	if err != nil {
		t.Fatal(err)
	}
	badProposedBound := 9.0
	if !reachable || !(badProposedBound > d) {
		t.Fatalf("counterexample lost: bound=%v DTW=%v", badProposedBound, d)
	}
	if _, err = tfsax.ScanLBConservative(context.Background(), nil, tfsax.Query{}, 0); !errors.Is(err, tfsax.ErrUncertifiedBound) {
		t.Fatal("uncertified pruning unexpectedly enabled")
	}
}
func TestLBKeoghFiniteSmallSpace(t *testing.T) {
	digits := []float64{-1, 0, 1}
	for a := 0; a < 81; a++ {
		for b := 0; b < 81; b++ {
			x := make([]float64, 4)
			y := make([]float64, 4)
			aa, bb := a, b
			for i := 0; i < 4; i++ {
				x[i] = digits[aa%3]
				y[i] = digits[bb%3]
				aa /= 3
				bb /= 3
			}
			for radius := 0; radius < 4; radius++ {
				lower, err := tfsax.LBKeoghAbsolute(x, y, radius)
				if err != nil {
					t.Fatal(err)
				}
				exact, reachable, err := tfsax.ExactDTW(x, y, radius)
				if err != nil {
					t.Fatal(err)
				}
				if !reachable || lower > exact+1e-12 {
					t.Fatalf("counterexample found x=%v y=%v r=%d LB=%v DTW=%v", x, y, radius, lower, exact)
				}
			}
		}
	}
}
