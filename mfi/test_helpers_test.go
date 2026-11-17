// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External test helpers: errors.Is, independent arithmetic and explicit comparisons.
package mfi_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/mfi"
)

func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("error = %v, want errors.Is(%v)", err, target)
	}
}
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %v want %v", op, got, want)
	}
}
func mustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d want %d", op, got, want)
	}
}
func isNilLike(v any) bool {
	if v == nil {
		return true
	}
	n, ok := v.(interface{ IsNil() bool })
	return ok && n.IsNil()
}
func mustNilState(t *testing.T, v any, want bool, op string) {
	t.Helper()
	mustEqualBool(t, isNilLike(v), want, op)
}
func mustNear(t *testing.T, got, want float64) {
	t.Helper()
	if !finiteTest(got) || math.Abs(got-want) > 1e-10*math.Max(1, math.Abs(want)) {
		t.Fatalf("got %.17g want %.17g", got, want)
	}
}
func finiteTest(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func sampleBars(prices, volume []float64) mfi.Bars {
	o, h, l, c, v := make([]float64, len(prices)), make([]float64, len(prices)), make([]float64, len(prices)), make([]float64, len(prices)), make([]float64, len(prices))
	for i, p := range prices {
		o[i], h[i], l[i], c[i], v[i] = p, p, p, p, volume[i]
	}
	return mfi.Bars{Open: o, High: h, Low: l, Close: c, Volume: v}
}
func sampleBar(price, volume float64) mfi.Bar {
	return mfi.Bar{Open: price, High: price, Low: price, Close: price, Volume: volume}
}
func sliceBars(b mfi.Bars, i int) mfi.Bar {
	return mfi.Bar{Open: b.Open[i], High: b.High[i], Low: b.Low[i], Close: b.Close[i], Volume: b.Volume[i]}
}

// oracle recomputes the entire window of exact fixture contributions from scratch.
func oracle(b mfi.Bars, period int) ([]float64, []bool) {
	n := len(b.Close)
	values, valid := make([]float64, n), make([]bool, n)
	tp := make([]float64, n)
	for i := range tp {
		tp[i] = (b.High[i] + b.Low[i] + b.Close[i]) / 3
	}
	for i := period; i < n; i++ {
		var pos, neg float64
		for j := i - period + 1; j <= i; j++ {
			f := tp[j] * b.Volume[j]
			if tp[j] > tp[j-1] {
				pos += f
			} else if tp[j] < tp[j-1] {
				neg += f
			}
		}
		valid[i] = true
		switch {
		case pos == 0 && neg == 0:
			values[i] = 50
		case neg == 0:
			values[i] = 100
		case pos == 0:
			values[i] = 0
		default:
			values[i] = 100 * pos / (pos + neg)
		}
	}
	return values, valid
}

// mustSnapshotBits checks complete checkpoint identity, including correction bits.
// It is deliberately explicit rather than reflecting across exported carriers.
func mustSnapshotBits(t *testing.T, got, want mfi.Snapshot) {
	t.Helper()
	if got.Algorithm != want.Algorithm || got.Config != want.Config || got.Count != want.Count ||
		len(got.FloatState) != len(want.FloatState) || len(got.IntState) != len(want.IntState) {
		t.Fatalf("snapshot header mismatch: got count=%d want=%d", got.Count, want.Count)
	}
	for i := range got.FloatState {
		if math.Float64bits(got.FloatState[i]) != math.Float64bits(want.FloatState[i]) {
			t.Fatalf("snapshot float[%d] changed: got %g want %g", i, got.FloatState[i], want.FloatState[i])
		}
	}
	for i := range got.IntState {
		if got.IntState[i] != want.IntState[i] {
			t.Fatalf("snapshot int[%d] changed: got %d want %d", i, got.IntState[i], want.IntState[i])
		}
	}
}
