// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm_test checks concurrent read-only model inference and bounded
// fuzzing of local sequence lengths without relying on scheduling outcomes.
package lstm_test

import (
	"context"
	"sync"
	"testing"

	"github.com/lvlath/go/lstm"
)

func TestConcurrentImmutableInfer(t *testing.T) {
	m := fixture(t, 1, 1, []float64{.1, .2, .3, .4})
	input := lstm.Tensor3{Data: make([]float64, 128), Batch: 1, Time: 128, Features: 1}
	var wg sync.WaitGroup
	fail := make(chan error, 16)
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, e := m.Infer(context.Background(), input, nil, nil, lstm.Options{})
			if e == nil && r == nil {
				fail <- lstm.ErrShape
				return
			}
			if e != nil {
				fail <- e
			}
		}()
	}
	wg.Wait()
	close(fail)
	for e := range fail {
		t.Fatal(e)
	}
}
func FuzzInferenceLengths(f *testing.F) {
	f.Add(uint8(0), uint8(0))
	f.Add(uint8(2), uint8(1))
	f.Add(uint8(10), uint8(9))
	f.Fuzz(func(t *testing.T, time, length uint8) {
		m := fixture(t, 1, 1, nil)
		T := int(time) % 32
		L := int(length) % 33
		data := make([]float64, T)
		r, e := m.Infer(context.Background(), lstm.Tensor3{Data: data, Batch: 1, Time: T, Features: 1}, []int{L}, nil, lstm.Options{ReturnSequence: true})
		if L > T {
			if r != nil {
				t.Fatal("partial result")
			}
			mustErrorIs(t, e, lstm.ErrInvalidOption)
			return
		}
		if e != nil {
			t.Fatal(e)
		}
		if len(r.Sequence.Data) != T {
			t.Fatalf("unexpected sequence length")
		}
		for j := L; j < T; j++ {
			if r.Sequence.Data[j] != 0 {
				t.Fatal("nonzero padding")
			}
		}
	})
}
