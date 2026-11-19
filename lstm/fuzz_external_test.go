// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm_test fuzzes public import and inference with bounded, arbitrary
// binary64 payloads, including nonfinite words and negative-zero encodings.
package lstm_test

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/lstm"
)

// FuzzImportAndInferencePayload tests that accepted payloads remain finite and
// rejected inputs never produce a usable Result or panic from package code.
// Geometry is deliberately bounded; a fuzzer must never allocate by untrusted sizes.
func FuzzImportAndInferencePayload(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0})
	f.Add([]byte{255, 255, 255, 255, 255, 255, 239, 127}) // MaxFloat64.
	f.Add([]byte{0, 0, 0, 0, 0, 0, 240, 127})             // +Inf.
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 128})               // Negative zero.

	f.Fuzz(func(t *testing.T, payload []byte) {
		if len(payload) < 8 || len(payload) > 64 {
			return // Bound fuzz execution and keep each input's semantic width known.
		}
		value := math.Float64frombits(binary.LittleEndian.Uint64(payload[:8]))
		weights := lstm.Weights{
			Input:     &adapterMatrix{rows: 4, cols: 1, values: []float64{value, 0, .1, 0}},
			Recurrent: mustMatrix(t, 4, 1, nil),
			Bias:      []float64{0, 0, 0, 0}, Hidden: 1,
		}
		model, err := lstm.Import(weights)
		if !finiteFixture(value) {
			if model != nil {
				t.Fatal("nonfinite model was published")
			}
			mustErrorIs(t, err, lstm.ErrNonFinite)
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		result, err := model.InferSequence(context.Background(), lstm.Tensor3{
			Data: []float64{2}, Batch: 1, Time: 1, Features: 1,
		}, nil, nil)
		if err != nil {
			if result != nil {
				t.Fatal("error published a partial result")
			}
			if !errors.Is(err, lstm.ErrOverflow) {
				t.Fatalf("unexpected arithmetic error: %v", err)
			}
			return
		}
		if result == nil || len(result.Sequence.Data) != 1 {
			t.Fatal("finite input failed to produce a complete output")
		}
		if !finiteFixture(result.Sequence.Data[0]) {
			t.Fatal("successful inference returned nonfinite output")
		}
	})
}

func finiteFixture(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
