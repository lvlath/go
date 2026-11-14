// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm_test independently enumerates exact rational categorical paths.
// The oracle does not call production forward, backward, or Viterbi routines.
package hmm_test

import (
	"context"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/hmm"
)

// auditRational creates an exact fraction representing one binary64 input.
// The closed-form tiny oracle uses no approximate probability products.
func auditRational(x float64) *big.Rat { return new(big.Rat).SetFloat64(x) }

// TestAuditCategoricalExactPathOracle compares every causal prefix, every
// smoothed posterior row, and the maximum joint path with exact K^T enumeration.
func TestAuditCategoricalExactPathOracle(t *testing.T) {
	prior := []float64{.25, .75}
	transition := [][]float64{{.75, .25}, {.5, .5}}
	emission := [][]float64{{.75, .25}, {.25, .75}}
	symbols := []uint32{0, 1, 1, 0}
	model, err := hmm.NewCategorical(hmm.CategoricalSpec{
		Common: hmm.Common{Prior: prior, Transition: dense(t, transition)}, Emission: dense(t, emission),
	})
	if err != nil {
		t.Fatal(err)
	}
	const k = 2
	pathCount := 1 << len(symbols)
	totals := make([]*big.Rat, len(symbols))
	smoothMass := make([][]*big.Rat, len(symbols))
	for time := range symbols {
		totals[time] = new(big.Rat)
		smoothMass[time] = []*big.Rat{new(big.Rat), new(big.Rat)}
	}
	bestWeight := new(big.Rat)
	bestPath := make([]int, len(symbols))
	for mask := 0; mask < pathCount; mask++ {
		path := make([]int, len(symbols))
		for i := range path {
			path[i] = (mask >> (len(path) - i - 1)) & 1
		}
		total := new(big.Rat).SetInt64(1)
		for time, state := range path {
			if time == 0 {
				total.Mul(total, auditRational(prior[state]))
			} else {
				total.Mul(total, auditRational(transition[path[time-1]][state]))
			}
			total.Mul(total, auditRational(emission[state][symbols[time]]))
		}
		if total.Cmp(bestWeight) > 0 {
			bestWeight.Set(total)
			copy(bestPath, path)
		}
		for time, state := range path {
			smoothMass[time][state].Add(smoothMass[time][state], total)
		}
	}
	// Enumerate K^(t+1) prefix paths independently of the full-path oracle.
	for time := range symbols {
		for mask := 0; mask < (1 << (time + 1)); mask++ {
			prefix := new(big.Rat).SetInt64(1)
			previous := -1
			for i := 0; i <= time; i++ {
				state := (mask >> (time - i)) & 1
				if i == 0 {
					prefix.Mul(prefix, auditRational(prior[state]))
				} else {
					prefix.Mul(prefix, auditRational(transition[previous][state]))
				}
				prefix.Mul(prefix, auditRational(emission[state][symbols[i]]))
				previous = state
			}
			totals[time].Add(totals[time], prefix)
		}
	}
	totalFloat, _ := totals[len(symbols)-1].Float64()
	full, err := model.Filter(context.Background(), symbols, hmm.FilterOptions{ReturnHistory: true})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(full.LogLikelihood-math.Log(totalFloat)) > 1e-13 {
		t.Fatalf("marginal got=%g want=%g", full.LogLikelihood, math.Log(totalFloat))
	}
	for time := range symbols {
		prefix, err := model.Filter(context.Background(), symbols[:time+1], hmm.FilterOptions{ReturnHistory: true})
		if err != nil {
			t.Fatal(err)
		}
		for state := 0; state < k; state++ {
			// Re-enumerate only paths ending in this particular state.
			subset := new(big.Rat)
			for mask := 0; mask < (1 << (time + 1)); mask++ {
				if mask&1 != state {
					continue
				}
				weight := new(big.Rat).SetInt64(1)
				previous := -1
				for i := 0; i <= time; i++ {
					current := (mask >> (time - i)) & 1
					if i == 0 {
						weight.Mul(weight, auditRational(prior[current]))
					} else {
						weight.Mul(weight, auditRational(transition[previous][current]))
					}
					weight.Mul(weight, auditRational(emission[current][symbols[i]]))
					previous = current
				}
				subset.Add(subset, weight)
			}
			probability, _ := new(big.Rat).Quo(subset, totals[time]).Float64()
			actual, e := prefix.Posterior.At(time, state)
			if e != nil {
				t.Fatal(e)
			}
			history, e := full.Posterior.At(time, state)
			if e != nil {
				t.Fatal(e)
			}
			if math.Abs(actual-probability) > 1e-13 || math.Abs(history-probability) > 1e-13 {
				t.Fatalf("alpha[%d,%d] got=%g history=%g want=%g", time, state, actual, history, probability)
			}
		}
	}
	smoother, err := model.Smooth(context.Background(), symbols)
	if err != nil {
		t.Fatal(err)
	}
	for time := range symbols {
		for state := 0; state < k; state++ {
			want, _ := new(big.Rat).Quo(smoothMass[time][state], totals[len(symbols)-1]).Float64()
			got, e := smoother.Posterior.At(time, state)
			if e != nil {
				t.Fatal(e)
			}
			if math.Abs(got-want) > 1e-13 {
				t.Fatalf("gamma[%d,%d] got=%g want=%g", time, state, got, want)
			}
		}
	}
	decoded, err := model.Decode(context.Background(), symbols)
	if err != nil {
		t.Fatal(err)
	}
	bestFloat, _ := bestWeight.Float64()
	if math.Abs(decoded.LogJointProbability-math.Log(bestFloat)) > 1e-13 {
		t.Fatalf("path log=%g want=%g", decoded.LogJointProbability, math.Log(bestFloat))
	}
	for i, want := range bestPath {
		if decoded.States[i] != want {
			t.Fatalf("decoded path=%v oracle=%v", decoded.States, bestPath)
		}
	}
}
