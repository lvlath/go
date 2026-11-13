// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm_test verifies categorical domain boundaries and canonical model identities.
// These fixtures protect public FilterState compatibility across changes to internal encoding.
package hmm_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/hmm"
)

// TestModelID_CanonicalDimensionsRemainCompatible pins the exact serialized
// SHA-256 input format for positive state and emission dimensions. The expected
// digests were independently derived from the versioned, big-endian canonical
// byte layout, not by hashing the returned identity a second time.
//
// A change in either digest invalidates previously saved FilterState values;
// therefore a refactor of dimension encoding must not silently change it.
func TestModelID_CanonicalDimensionsRemainCompatible(t *testing.T) {
	common := hmm.Common{
		Prior:      []float64{0.75, 0.25},
		Transition: dense(t, [][]float64{{0.75, 0.25}, {0.25, 0.75}}),
	}

	categoricalD, err := hmm.NewCategorical(hmm.CategoricalSpec{
		Common:   common,
		Emission: dense(t, [][]float64{{0.125, 0.875}, {0.5, 0.5}}),
	})
	if err != nil {
		t.Fatal(err)
	}
	const categoricalID = "30bf73ebf49b471df1862c637815a7a457ca2abff9d829c2235d180506a24495"
	if got := categoricalD.ModelID(); got != categoricalID {
		t.Fatalf("categorical canonical ID changed: got %s, want %s", got, categoricalID)
	}

	gaussian, err := hmm.NewGaussian(hmm.GaussianSpec{
		Common:    common,
		Means:     dense(t, [][]float64{{0}, {1}}),
		Variances: dense(t, [][]float64{{1}, {4}}),
	})
	if err != nil {
		t.Fatal(err)
	}
	const gaussianID = "2bc98dba6924af92971c685c7f8bb731d6c491c69e6ca807ce0e4fce8a238d68"
	if got := gaussian.ModelID(); got != gaussianID {
		t.Fatalf("Gaussian canonical ID changed: got %s, want %s", got, gaussianID)
	}

	// A state checkpoint from one categorical allocation can be resumed with
	// a newly allocated but bitwise-identical model, without sharing memory.
	copyModel, err := hmm.NewCategorical(hmm.CategoricalSpec{
		Common:   common,
		Emission: dense(t, [][]float64{{0.125, 0.875}, {0.5, 0.5}}),
	})
	if err != nil {
		t.Fatal(err)
	}
	state, err := categoricalD.FilterStep(context.Background(), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	resumed, err := copyModel.FilterStep(context.Background(), state, 1)
	if err != nil || resumed == nil || resumed.ModelID != categoricalID || resumed.Steps != 2 {
		t.Fatalf("canonical checkpoint rejected: result=%+v, error=%v", resumed, err)
	}
}

// TestCategoricalSymbols_WideUnsignedBoundary verifies that an observation
// remains unsigned while its alphabet is a positive Go int. In particular,
// MaxUint32 must be rejected on both 32-bit and 64-bit platforms instead of
// wrapping through an int cast or leaking into emission array indexing.
func TestCategoricalSymbols_WideUnsignedBoundary(t *testing.T) {
	spec := hmm.CategoricalSpec{
		Common: hmm.Common{
			Prior:      []float64{1},
			Transition: dense(t, [][]float64{{1}}),
		},
		Emission: dense(t, [][]float64{{0.25, 0.75}}),
	}
	model, err := hmm.NewCategorical(spec)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	// In-range observations must keep their categorical probabilities; a
	// domain guard is not permitted to modify emission likelihoods.
	for _, tc := range []struct {
		symbol uint32
		mass   float64
	}{{0, 0.25}, {1, 0.75}} {
		state, stepErr := model.FilterStep(ctx, nil, tc.symbol)
		if stepErr != nil || state == nil {
			t.Fatalf("valid symbol %d: state=%+v error=%v", tc.symbol, state, stepErr)
		}
		if got, want := state.LogLikelihood, math.Log(tc.mass); math.Abs(got-want) > 1e-14 {
			t.Fatalf("symbol %d: log evidence got %.17g, want %.17g", tc.symbol, got, want)
		}
	}

	// All public categorical operations must report the same shape error,
	// and must not return partial results or index an emission row.
	for _, invalid := range []uint32{2, math.MaxUint32} {
		if result, callErr := model.FilterStep(ctx, nil, invalid); result != nil || !errors.Is(callErr, hmm.ErrShape) {
			t.Fatalf("FilterStep(%d): result=%+v error=%v", invalid, result, callErr)
		}
		if result, callErr := model.Filter(ctx, []uint32{0, invalid}, hmm.FilterOptions{}); result != nil || !errors.Is(callErr, hmm.ErrShape) {
			t.Fatalf("Filter(%d): result=%+v error=%v", invalid, result, callErr)
		}
		if result, callErr := model.Smooth(ctx, []uint32{0, invalid}); result != nil || !errors.Is(callErr, hmm.ErrShape) {
			t.Fatalf("Smooth(%d): result=%+v error=%v", invalid, result, callErr)
		}
		if result, callErr := model.Decode(ctx, []uint32{0, invalid}); result != nil || !errors.Is(callErr, hmm.ErrShape) {
			t.Fatalf("Decode(%d): result=%+v error=%v", invalid, result, callErr)
		}
		if result, callErr := hmm.FitCategorical(ctx, [][]uint32{{0, invalid}}, spec, hmm.FitOptions{}); result != nil || !errors.Is(callErr, hmm.ErrShape) {
			t.Fatalf("FitCategorical(%d): result=%+v error=%v", invalid, result, callErr)
		}
	}
}
