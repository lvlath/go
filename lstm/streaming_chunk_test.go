// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm_test checks bitwise equivalence of two streaming chunks
// against a full deterministic evaluation with identical local time order.
package lstm_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/lstm"
)

func TestLSTM07_BitwiseStreaming(t *testing.T) {
	m := fixture(t, 1, 1, []float64{.2, -.3, .5, .1})
	full, e := m.Infer(context.Background(), lstm.Tensor3{Data: []float64{.3, -.1, .9, .2}, Batch: 1, Time: 4, Features: 1}, nil, nil, lstm.Options{ReturnSequence: true})
	if e != nil {
		t.Fatal(e)
	}
	first, e := m.Infer(context.Background(), lstm.Tensor3{Data: []float64{.3, -.1}, Batch: 1, Time: 2, Features: 1}, nil, nil, lstm.Options{ReturnSequence: true})
	if e != nil {
		t.Fatal(e)
	}
	second, e := m.Infer(context.Background(), lstm.Tensor3{Data: []float64{.9, .2}, Batch: 1, Time: 2, Features: 1}, nil, &first.Final, lstm.Options{ReturnSequence: true})
	if e != nil {
		t.Fatal(e)
	}
	for i, v := range full.Sequence.Data {
		var got float64
		if i < 2 {
			got = first.Sequence.Data[i]
		} else {
			got = second.Sequence.Data[i-2]
		}
		if math.Float64bits(v) != math.Float64bits(got) {
			t.Fatalf("mismatch step %d", i)
		}
	}
	if math.Float64bits(at(t, full.Final.Cell, 0, 0)) != math.Float64bits(at(t, second.Final.Cell, 0, 0)) {
		t.Fatal("cell not bitwise equal")
	}
}
