// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test provides contract-focused tests for the public DTW API.
// Helpers in this file intentionally use only the Go standard library.
package dtw_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/dtw"
)

func mustNoError(t *testing.T, err error, op string) {
	t.Helper()

	if err != nil {
		t.Fatalf("%s: unexpected error: %v", op, err)
	}
}

func mustErrorIs(t *testing.T, err, target error, op string) {
	t.Helper()

	if !errors.Is(err, target) {
		t.Fatalf("%s: errors.Is(err, %v)=false; got %v", op, target, err)
	}
}

func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()

	if got != want {
		t.Fatalf("%s: got %t, want %t", op, got, want)
	}
}

func mustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()

	if got != want {
		t.Fatalf("%s: got %d, want %d", op, got, want)
	}
}

func mustFloatEqual(t *testing.T, got, want, epsilon float64, op string) {
	t.Helper()

	if math.IsInf(got, 0) || math.IsInf(want, 0) {
		if got != want {
			t.Fatalf("%s: got %v, want %v", op, got, want)
		}

		return
	}

	if math.IsNaN(got) || math.IsNaN(want) {
		t.Fatalf("%s: NaN comparison is invalid: got=%v want=%v", op, got, want)
	}

	if math.Abs(got-want) > epsilon {
		t.Fatalf("%s: got %.12g, want %.12g, epsilon %.12g", op, got, want, epsilon)
	}
}

func mustFinite(t *testing.T, value float64, op string) {
	t.Helper()

	if math.IsNaN(value) || math.IsInf(value, 0) {
		t.Fatalf("%s: got non-finite value %v", op, value)
	}
}

func mustInf(t *testing.T, value float64, op string) {
	t.Helper()

	if !math.IsInf(value, 1) {
		t.Fatalf("%s: got %v, want +Inf", op, value)
	}
}

func mustPathValid(t *testing.T, path dtw.Path, n, m int, op string) {
	t.Helper()

	if len(path) == 0 {
		t.Fatalf("%s: empty path", op)
	}

	if path[0] != (dtw.Coord{I: 0, J: 0}) {
		t.Fatalf("%s: first coord got %+v, want {I:0 J:0}", op, path[0])
	}

	last := path[len(path)-1]
	if last.I != n-1 || last.J != m-1 {
		t.Fatalf("%s: last coord got %+v, want {I:%d J:%d}", op, last, n-1, m-1)
	}

	for idx := 1; idx < len(path); idx++ {
		prev := path[idx-1]
		curr := path[idx]
		di := curr.I - prev.I
		dj := curr.J - prev.J

		if di < 0 || dj < 0 {
			t.Fatalf("%s: non-monotone step %d: %+v -> %+v", op, idx, prev, curr)
		}

		if !((di == 1 && dj == 1) || (di == 1 && dj == 0) || (di == 0 && dj == 1)) {
			t.Fatalf("%s: invalid step %d: di=%d dj=%d path=%+v", op, idx, di, dj, path)
		}
	}
}

func mustDistanceMatchesPath(
	t *testing.T,
	a, b []float64,
	path dtw.Path,
	penalty float64,
	want float64,
	op string,
) {
	t.Helper()

	total := 0.0

	for idx, coord := range path {
		total += math.Abs(a[coord.I] - b[coord.J])

		if idx == 0 {
			continue
		}

		prev := path[idx-1]
		di := coord.I - prev.I
		dj := coord.J - prev.J

		if (di == 1 && dj == 0) || (di == 0 && dj == 1) {
			total += penalty
		}
	}

	mustFloatEqual(t, total, want, 1e-9, op)
}

func mustPathEqual(t *testing.T, got, want dtw.Path, op string) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s: path length got %d, want %d; got=%v want=%v", op, len(got), len(want), got, want)
	}

	for idx := range want {
		if got[idx] != want[idx] {
			t.Fatalf("%s: path[%d] got %+v, want %+v; got=%v want=%v", op, idx, got[idx], want[idx], got, want)
		}
	}
}

func mustMatrixShape(t *testing.T, got interface {
	Rows() int
	Cols() int
}, rows, cols int, op string) {
	t.Helper()

	if got == nil {
		t.Fatalf("%s: got nil matrix", op)
	}

	if got.Rows() != rows || got.Cols() != cols {
		t.Fatalf("%s: shape got %dx%d, want %dx%d", op, got.Rows(), got.Cols(), rows, cols)
	}
}
