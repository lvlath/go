// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
package hmm_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/hmm"
)

// enumeratePaths is deliberately independent of the forward/backward/Viterbi kernels.
func enumeratePaths(prior []float64, transition, emission [][]float64, symbols []uint32) (float64, []float64, []int, float64) {
	k, t := len(prior), len(symbols)
	mass := make([]float64, t*k)
	path := make([]int, t)
	bestPath := make([]int, t)
	var total, best float64
	best = -1
	var visit func(int, float64)
	visit = func(depth int, p float64) {
		if depth == t {
			total += p
			for i, z := range path {
				mass[i*k+z] += p
			}
			if p > best {
				best = p
				copy(bestPath, path)
			}
			return
		}
		for state := 0; state < k; state++ {
			path[depth] = state
			step := emission[state][int(symbols[depth])]
			if depth == 0 {
				step *= prior[state]
			} else {
				step *= transition[path[depth-1]][state]
			}
			visit(depth+1, p*step)
		}
	}
	visit(0, 1)
	for i := range mass {
		mass[i] /= total
	}
	return math.Log(total), mass, bestPath, math.Log(best)
}
func TestForward_TinyPathEnumeration(t *testing.T) {
	prior := []float64{.2, .3, .5}
	transition := [][]float64{{.7, .2, .1}, {.1, .7, .2}, {.3, .1, .6}}
	emission := [][]float64{{.8, .2}, {.3, .7}, {.6, .4}}
	x := []uint32{0, 1, 0, 0, 1}
	m, e := hmm.NewCategorical(hmm.CategoricalSpec{Common: hmm.Common{Prior: prior, Transition: dense(t, transition)}, Emission: dense(t, emission)})
	if e != nil {
		t.Fatal(e)
	}
	expected, marginal, expectedPath, expectedJoint := enumeratePaths(prior, transition, emission, x)
	f, e := m.Filter(context.Background(), x, hmm.FilterOptions{ReturnHistory: true})
	if e != nil {
		t.Fatal(e)
	}
	approx(t, f.LogLikelihood, expected)
	// Filter history is intentionally NOT compared with smoothed marginals.
	s, e := m.Smooth(context.Background(), x)
	if e != nil {
		t.Fatal(e)
	}
	approx(t, s.LogLikelihood, expected)
	for i := range x {
		for j := 0; j < len(prior); j++ {
			v, e := s.Posterior.At(i, j)
			if e != nil {
				t.Fatal(e)
			}
			approx(t, v, marginal[i*len(prior)+j])
		}
	}
	d, e := m.Decode(context.Background(), x)
	if e != nil {
		t.Fatal(e)
	}
	approx(t, d.LogJointProbability, expectedJoint)
	for i, v := range d.States {
		if v != expectedPath[i] {
			t.Fatalf("Viterbi at %d: got %d expected %d", i, v, expectedPath[i])
		}
	}
}
