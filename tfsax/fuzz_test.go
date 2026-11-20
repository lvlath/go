// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax_test fuzzes the small finite-input oracle and symbol-search boundaries.
// Fuzzing never exercises a production TFSAX encoder or assumes an unproven DTW bound.
package tfsax_test

import (
	"context"
	"testing"

	"github.com/lvlath/go/tfsax"
)

// FuzzAbsoluteDTWAndLBKeogh checks a separately implemented full-matrix recurrence.
// Bounded integer-valued windows make exact equality meaningful without rounding tolerance.
func FuzzAbsoluteDTWAndLBKeogh(f *testing.F) {
	f.Add([]byte{0, 1, 1, 0, 1, 0, 1, 0, 2})
	f.Add([]byte{2, 2, 2, 2, 0, 0, 0, 0, 0})
	f.Add([]byte{3, 0, 2, 1, 2, 1, 0, 2, 1, 1, 0})
	f.Fuzz(func(t *testing.T, seed []byte) {
		if len(seed) < 9 || len(seed) > 32 {
			return
		}
		length := 4
		if seed[0]%2 == 1 && len(seed) >= 11 {
			length = 5
		}
		left, right := make([]float64, length), make([]float64, length)
		for i := 0; i < length; i++ {
			left[i] = float64(seed[1+i] % 3)
			right[i] = float64(seed[1+length+i] % 3)
		}
		radius := int(seed[0]) % length
		distance, reachable, err := tfsax.ExactDTW(left, right, radius)
		if err != nil || !reachable {
			t.Fatalf("finite equal-length banded DTW must be reachable: %v", err)
		}
		expected := independentSmallDTW(left, right, radius)
		lower, err := tfsax.LBKeoghAbsolute(left, right, radius)
		if err != nil || distance != expected || lower < 0 || lower > distance {
			t.Fatalf("independent oracle mismatch left=%v right=%v radius=%d exact=%g expected=%g bound=%g err=%v",
				left, right, radius, distance, expected, lower, err)
		}
		corpus := []tfsax.Candidate{{ID: 9, Values: left}, {ID: 2, Values: right}}
		result, err := tfsax.ScanExact(context.Background(), corpus,
			tfsax.Query{Values: left, TopK: 1}, radius)
		if err != nil || result == nil || !result.Complete || !result.Diagnostics.Exact ||
			len(result.Candidates) != 1 || result.Candidates[0] != 9 {
			// The exact zero-distance self-window always ranks first unless another
			// independent identical window has a numerically smaller ID.
			if err == nil && result != nil && len(result.Candidates) == 1 &&
				result.Candidates[0] == 2 && expected == 0 {
				return
			}
			t.Fatalf("exact search contract failed: %+v %v", result, err)
		}
	})
}
