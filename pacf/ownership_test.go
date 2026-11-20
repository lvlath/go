// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf_test verifies detached results, nilability, and immutable-input sharing.
// All tests call only the public pacf API without reflection or private hooks.
package pacf_test

import (
	"context"
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/pacf"
)

// TestResult_CloneDetached verifies deep ownership of both result slices,
// including nil and allocated zero-value receivers and nil/empty slice identity.
func TestResult_CloneDetached(t *testing.T) {
	var absent *pacf.Result
	if absent.Clone() != nil || !absent.IsNil() {
		t.Fatal("nil receiver must remain nil without panic")
	}

	zero := &pacf.Result{}
	zeroCopy := zero.Clone()
	if zeroCopy == nil || zeroCopy.IsNil() || zeroCopy.Values != nil || zeroCopy.PredictionVariance != nil {
		t.Fatal("allocated zero-value Result must clone as a non-nil zero-value Result")
	}

	empty := &pacf.Result{Values: make([]float64, 0), PredictionVariance: make([]float64, 0)}
	emptyCopy := empty.Clone()
	if emptyCopy == nil || emptyCopy.Values == nil || emptyCopy.PredictionVariance == nil {
		t.Fatal("Clone must preserve non-nil empty slice identity")
	}

	computed, err := pacf.ComputeCentered(context.Background(), []float64{1, 3, 2, 7, 4, 5, 8, 6}, 4)
	if err != nil {
		t.Fatal(err)
	}
	cloned := computed.Clone()
	if cloned == nil || cloned == computed || cloned.Count != computed.Count {
		t.Fatal("Clone must return independent pointer with matching scalar metadata")
	}
	mustSlices(t, cloned.Values, computed.Values, 0)
	mustSlices(t, cloned.PredictionVariance, computed.PredictionVariance, 0)

	originalPACF := computed.Values[1]
	originalVariance := computed.PredictionVariance[1]
	cloned.Values[1] = -0.99
	cloned.PredictionVariance[1] = 999
	if computed.Values[1] != originalPACF || computed.PredictionVariance[1] != originalVariance {
		t.Fatal("Clone retained an alias of caller-owned result memory")
	}
	computed.Values[2] = 777
	if cloned.Values[2] == 777 {
		t.Fatal("Result.Clone must detach in both mutation directions")
	}
}

// TestResult_NilableStructuralInterface asserts compatibility with the
// library's pointer-based nil-state protocol without package coupling.
func TestResult_NilableStructuralInterface(t *testing.T) {
	var absent *pacf.Result
	var nilable interface{ IsNil() bool } = absent
	if !nilable.IsNil() {
		t.Fatal("typed nil inside interface must be recognized")
	}
	var present interface{ IsNil() bool } = &pacf.Result{}
	if present.IsNil() {
		t.Fatal("allocated empty Result is present")
	}
}

// TestPACF_ConcurrentCallsProduceIdenticalResults repeats full profiles with
// shared immutable input and checks every coefficient, not just the sample count.
func TestPACF_ConcurrentCallsProduceIdenticalResults(t *testing.T) {
	const workers = 20
	samples := make([]float64, 2048)
	for i := range samples {
		samples[i] = math.Sin(0.19*float64(i)) + 0.45*math.Cos(0.07*float64(i))
	}
	want, err := pacf.ComputeCentered(context.Background(), samples, 25)
	if err != nil {
		t.Fatal(err)
	}

	problems := make(chan string, workers)
	var group sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		group.Add(1)
		go func() {
			defer group.Done()
			got, computeErr := pacf.ComputeCentered(context.Background(), samples, 25)
			if computeErr != nil {
				problems <- computeErr.Error()
				return
			}
			for i := range got.Values {
				if math.Float64bits(got.Values[i]) != math.Float64bits(want.Values[i]) ||
					math.Float64bits(got.PredictionVariance[i]) != math.Float64bits(want.PredictionVariance[i]) {
					problems <- "non-deterministic profile"
					return
				}
			}
		}()
	}
	group.Wait()
	close(problems)
	for problem := range problems {
		t.Error(problem)
	}
}
