// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm_test validates variable-duration telemetry masks, strict zero
// outputs and finite checks on every supplied padded input coordinate.
package lstm_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/lstm"
)

func TestPaddingLengthsAndNonfiniteTail(t *testing.T) {
	m := fixture(t, 1, 1, []float64{0, 0, math.Atanh(.5), 0})
	data := make([]float64, 4*30)
	r, e := m.Infer(context.Background(), lstm.Tensor3{Data: data, Batch: 4, Time: 30, Features: 1}, []int{10, 25, 15, 30}, nil, lstm.Options{ReturnSequence: true})
	if e != nil {
		t.Fatal(e)
	}
	for b, L := range []int{10, 25, 15, 30} {
		for j := L; j < 30; j++ {
			if r.Sequence.Data[b*30+j] != 0 {
				t.Fatalf("nonzero padding batch %d", b)
			}
		}
	}
	data[11] = math.Inf(1)
	r, e = m.Infer(context.Background(), lstm.Tensor3{Data: data, Batch: 4, Time: 30, Features: 1}, []int{10, 25, 15, 30}, nil, lstm.Options{})
	if r != nil {
		t.Fatal("partial result")
	}
	mustErrorIs(t, e, lstm.ErrNonFinite)
}
