// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test contains test helpers
//
// Purpose:
//   - Provide small, deterministic test fixtures and utilities for builders/kernels.
//   - Keep all data finite and well-formed to avoid numeric-policy interference.

package matrix_test

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/matrix"
)

// Number of vertices used in table-driven graphs.
const V = 8

// Number of edges in an undirected complete graph K_V.
const EComplete = V * (V - 1) / 2

// hide WRAPS any Matrix to hide its concrete type from type assertions.
// Implementation:
//   - Stage 1: Embed matrix.Matrix to forward all methods.
//   - Stage 2: Use hide{X} in tests to force non-*Dense (fallback) paths.
//
// Behavior highlights:
//   - Prevents "*Dense" fast-path via type switch in code under test.
//
// Inputs:
//   - matrix.Matrix: any implementation.
//
// Returns:
//   - hide: wrapper that still satisfies Matrix but masks concrete type.
//
// Errors:
//   - None.
//
// Determinism:
//   - N/A (wrapper only).
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Useful to assert fast-path == fallback bitwise (or via AllClose).
//
// AI-Hints:
//   - Prefer wrapping ONLY the operand you want to de-opt; keep the other one *Dense to isolate path differences.
type hide struct{ matrix.Matrix }

// MustDense ALLOCATES an r×c *Dense or fails the test (fatal on error).
// Implementation:
//   - Stage 1: Call matrix.NewDense(r,c).
//   - Stage 2: t.Fatalf on error to abort the test early.
//
// Behavior highlights:
//   - Concise boilerplate reduction in tests.
//
// Inputs:
//   - r,c: matrix shape.
//
// Returns:
//   - *matrix.Dense allocated with zeroed data.
//
// Errors:
//   - Fatal test failure if allocation fails.
//
// Determinism:
//   - Deterministic zero-initialized buffer.
//
// Complexity:
//   - Time O(r*c) zeroing by runtime, Space O(r*c).
//
// Notes:
//   - Prefer MustDense when subsequent steps assume non-nil Dense.
//
// AI-Hints:
//   - When you need non-zero data, pair with RandomFill or manual Set.
func MustDense(t *testing.T, r, c int) *matrix.Dense {
	t.Helper()
	m, err := matrix.NewDense(r, c)
	if err != nil {
		t.Fatalf("NewDense(%d,%d): %v", r, c, err)
	}

	return m
}

// IdentityDense RETURNS an n×n identity Matrix (main diagonal = 1, else 0).
// Implementation:
//   - Stage 1: matrix.NewIdentity(n).
//   - Stage 2: t.Fatalf on error.
//
// Behavior highlights:
//   - Compact identity builder without exposing internal loops.
//
// Inputs:
//   - n: matrix size (n≥0).
//
// Returns:
//   - matrix.Matrix (likely *Dense) containing I_n.
//
// Errors:
//   - Fatal test failure if allocation fails.
//
// Determinism:
//   - Deterministic pattern (no RNG).
//
// Complexity:
//   - Time O(n^2) (initialization), Space O(n^2).
//
// Notes:
//   - Use in algebra/graph tests to assert neutral operations.
//
// AI-Hints:
//   - Great as a baseline for perturbations and property tests.
func IdentityDense(t *testing.T, n int) matrix.Matrix {
	t.Helper()
	m, err := matrix.NewIdentity(n)
	if err != nil {
		t.Fatalf("NewIdentity(%d): %v", n, err)
	}

	return m
}

// MustAt READS m[i,j] or fails the test.
// Implementation:
//   - Stage 1: Call m.At(i,j).
//   - Stage 2: t.Fatalf on error, return value otherwise.
//
// Behavior highlights:
//   - Clear failure site on bounds/impl errors.
//
// Inputs:
//   - m,i,j.
//
// Returns:
//   - float64 value.
//
// Errors:
//   - Fatal test failure on At error.
//
// Determinism:
//   - N/A.
//
// Complexity:
//   - O(1) per call.
//
// Notes:
//   - Pair with CompareExact/Close.
//
// AI-Hints:
//   - Safe for fallback paths where At may allocate internally.
func MustAt(t *testing.T, m matrix.Matrix, i, j int) float64 {
	t.Helper()
	v, err := m.At(i, j)
	if err != nil {
		t.Fatalf("At(%d,%d): %v", i, j, err)
	}

	return v
}

// MustSet WRITES a finite scalar v into m[i,j] using Set() policy.
// Non-finite values are rejected here intentionally to keep most tests "clean".
//
// Implementation:
//   - Stage 1: Reject NaN/±Inf at helper level (fail fast with guidance).
//   - Stage 2: Call m.Set(i,j,v) and fatal on error.
//
// Behavior highlights:
//   - Ensures helpers do not accidentally fight numeric-policy.
//
// Inputs:
//   - m: target matrix.
//   - i,j: coordinates.
//   - v: value (must be finite).
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal if v is NaN/±Inf or Set fails.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - For constructing matrices with NaN/±Inf in storage, allocate with
//     WithNoValidateNaNInf and then use MustFillRowMajor, or use NewFilledDenseRaw.
//
// AI-Hints:
//   - Keep sanitizer tests explicit: build dirty inputs via Fill, then call sanitizer.
func MustSet(t *testing.T, m matrix.Matrix, i, j int, v float64) {
	t.Helper()

	if math.IsNaN(v) || math.IsInf(v, 0) {
		t.Fatalf("MustSet refuses non-finite v=%v; use MustFillRowMajor (raw ingest) for NaN/±Inf fixtures", v)
	}

	if err := m.Set(i, j, v); err != nil {
		t.Fatalf("Set(%d,%d,%v): %v", i, j, v, err)
	}
}

// NewFilledDense BUILDS r×c *Dense from a row-major flat slice.
// Implementation:
//   - Stage 1: Validate len(vals)==r*c.
//   - Stage 2: Allocate Dense and Set(i,j, vals[i*c+j]).
//
// Behavior highlights:
//   - Deterministic fixture creation with explicit values.
//
// Inputs:
//   - r,c: shape; vals: row-major data of length r*c.
//
// Returns:
//   - *matrix.Dense with copied values.
//
// Errors:
//   - Fatal test failure if lengths mismatch or Set fails.
//
// Determinism:
//   - Deterministic fill order.
//
// Complexity:
//   - Time O(r*c), Space O(r*c).
//
// Notes:
//   - Prefer for small exact-equality tests.
//
// AI-Hints:
//   - Use with CompareExact for integer-like matrices.
func NewFilledDense(t *testing.T, r, c int, vals []float64) *matrix.Dense {
	t.Helper()

	if len(vals) != r*c {
		t.Fatalf("NewFilledDense: want %d values, got %d", r*c, len(vals))
	}

	// Guard against accidental NaN/Inf in "clean" fixtures.
	for idx, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("NewFilledDense: vals[%d]=%v is non-finite; use MustFillRowMajor/NewFilledDenseRaw for dirty fixtures", idx, v)
		}
	}

	d := MustDense(t, r, c)

	var i, j int // loop iterators
	// Keep Set() path here intentionally (policy-respecting).
	for i = 0; i < r; i++ {
		for j = 0; j < c; j++ {
			MustSet(t, d, i, j, vals[i*c+j])
		}
	}

	return d
}

// MustFillRowMajor fills matrix storage from a row-major slice through Fill.
//
// Important:
//   - This bypasses per-cell Set calls.
//   - It does NOT bypass the matrix numeric policy.
//   - If the target Dense has validation enabled, NaN/±Inf will still be rejected.
//   - For dirty fixtures, allocate the Dense with WithNoValidateNaNInf first.
//
// Implementation:
//   - Stage 1: Assert the matrix implements Fill([]float64) error.
//   - Stage 2: Call Fill with the provided row-major data.
//
// Behavior highlights:
//   - Deterministic bulk ingestion.
//   - Policy remains owned by the concrete Matrix.
//
// Inputs:
//   - m: target matrix.
//   - vals: row-major values, length must match Rows*Cols.
//
// Errors:
//   - Fatal test failure if Fill is unavailable or returns an error.
func MustFillRowMajor(t *testing.T, m matrix.Matrix, vals []float64) {
	t.Helper()

	f, ok := m.(interface {
		Fill([]float64) error
	})
	if !ok {
		t.Fatalf("matrix does not support Fill([]float64)")
	}

	if err := f.Fill(vals); err != nil {
		t.Fatalf("Fill(row-major): %v", err)
	}
}

// RandFilledDense RETURNS a new r×c Dense filled with deterministic U(-1,1).
// Implementation:
//   - Stage 1: Allocate Dense.
//   - Stage 2: Fill via seeded RNG, row-major.
//
// Behavior highlights:
//   - One-liner to allocate+fill.
//
// Inputs:
//   - r,c: shape; seed: RNG seed.
//
// Returns:
//   - matrix.Matrix (Dense) populated with random values.
//
// Errors:
//   - Fatal test failure if allocation/Set fails.
//
// Determinism:
//   - Deterministic per seed.
//
// Complexity:
//   - Time O(r*c), Space O(r*c).
//
// Notes:
//   - Prefer for medium-sized randomized tests.
//
// AI-Hints:
//   - Use identical seeds across fast vs fallback to isolate path differences.
func RandFilledDense(t *testing.T, r, c int, seed int64) matrix.Matrix {
	t.Helper()
	m, err := matrix.NewDense(r, c)
	if err != nil {
		t.Fatalf("NewDense(%d,%d): %v", r, c, err)
	}
	rng := rand.New(rand.NewSource(seed))
	var i, j int
	for i = 0; i < r; i++ {
		for j = 0; j < c; j++ {
			if err = m.Set(i, j, rng.Float64()*2-1); err != nil {
				t.Fatalf("Set(%d,%d): %v", i, j, err)
			}
		}
	}
	return m
}

// RandomFill FILLS a Matrix with deterministic U(-1,1) values by seed.
// Implementation:
//   - Stage 1: rng := rand.New(rand.NewSource(seed)).
//   - Stage 2: For each cell, Set(i,j, rng.Float64()*2-1).
//
// Behavior highlights:
//   - Reproducible randomness for property tests.
//
// Inputs:
//   - m: target Matrix; seed: RNG seed.
//
// Returns:
//   - None (mutates m).
//
// Errors:
//   - Fatal test failure if Set returns error.
//
// Determinism:
//   - Deterministic for a fixed seed.
//
// Complexity:
//   - Time O(r*c), Space O(1) extra.
//
// Notes:
//   - Keeps values finite to avoid NaN/Inf policy interference.
//
// AI-Hints:
//   - Sweep multiple seeds in table-driven tests to increase coverage.
func RandomFill(t *testing.T, m matrix.Matrix, seed int64) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	r, c := m.Rows(), m.Cols()
	var (
		i, j int     // loop iterators
		v    float64 // random value
		err  error
	)
	for i = 0; i < r; i++ {
		for j = 0; j < c; j++ {
			v = rng.Float64()*2 - 1 // 0*2-1=-1 || 1*2-1=1
			if err = m.Set(i, j, v); err != nil {
				t.Fatalf("Set RandomFill(%d,%d): %v", i, j, err)
			}
		}
	}
}

// CompareExact ASSERTS strict equality between matrix and 2D literal.
// Implementation:
//   - Stage 1: Shape checks.
//   - Stage 2: Iterate and compare with == (no tolerances).
//
// Behavior highlights:
//   - Fails with exact mismatch location.
//
// Inputs:
//   - want: [][]float64 expected; m: Matrix under test.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on size/value mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(r*c), Space O(1).
//
// Notes:
//   - Use only for integer-like or carefully crafted small matrices.
//
// AI-Hints:
//   - For floats use CompareClose instead.
func CompareExact(t *testing.T, want [][]float64, m matrix.Matrix) {
	t.Helper()

	if m == nil {
		t.Fatalf("CompareExact: nil matrix")
	}

	r, c := m.Rows(), m.Cols()
	if len(want) != r {
		t.Fatalf("CompareExact: Rows = %d; want %d", r, len(want))
	}

	var i, j int // loop iterators
	var v float64
	for i = 0; i < r; i++ {
		if len(want[i]) != c {
			t.Fatalf("CompareExact: Cols[%d] = %d; want %d", i, c, len(want[i]))
		}
		for j = 0; j < c; j++ {
			if v = MustAt(t, m, i, j); v != want[i][j] {
				t.Fatalf("m[%d,%d]=%v; want %v", i, j, v, want[i][j])
			}
		}
	}
}

// CompareClose ASSERTS AllClose(a,b) under (rtol, atol).
// Implementation:
//   - Stage 1: matrix.AllClose(a,b, rtol, atol).
//   - Stage 2: t.Fatalf if false or if AllClose returns error.
//
// Behavior highlights:
//   - Encapsulates numeric tolerance logic used across tests.
//
// Inputs:
//   - a,b: matrices; rtol,atol: tolerances.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on mismatch or AllClose error.
//
// Determinism:
//   - Deterministic for fixed inputs.
//
// Complexity:
//   - Time O(r*c), Space O(1).
//
// Notes:
//   - Consider constants e.g. RtolTiny=1e-12, AtolTiny=1e-12 for reuse.
//
// AI-Hints:
//   - Use (0,0) for pure equality when numbers are exact.
func CompareClose(t *testing.T, a, b matrix.Matrix, rtol, atol float64) {
	t.Helper()
	ok, err := matrix.AllClose(a, b, rtol, atol)
	if err != nil {
		t.Fatalf("AllClose err: %v", err)
	}
	if !ok {
		t.Fatalf("AllClose=false (rtol=%g, atol=%g)", rtol, atol)
	}
}

// sliceClose ASSERTS |a[i]-b[i]| ≤ atol + rtol*|b[i]| element-wise.
// Implementation:
//   - Stage 1: Length check.
//   - Stage 2: Iterate with tolerance formula (as in AllClose).
//
// Behavior highlights:
//   - Aligns with matrix.AllClose policy for 1D slices.
//
// Inputs:
//   - a,b: slices; rtol,atol: tolerances.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - Useful for means/stds snapshots.
//
// AI-Hints:
//   - Keep tolerances consistent with CompareClose to avoid split-brain.
func sliceClose(t *testing.T, a, b []float64, rtol, atol float64) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("slice lengths: %d vs %d", len(a), len(b))
	}
	var diff, absb float64
	for i := range a {
		diff = a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		absb = b[i]
		if absb < 0 {
			absb = -absb
		}
		if diff > (atol + rtol*absb) {
			t.Fatalf("sliceClose idx=%d: got=%g want=%g (rtol=%g atol=%g)", i, a[i], b[i], rtol, atol)
		}
	}
}

// AlmostEqualSlice reports whether all elements are within absolute delta.
//
// Policy:
//   - Different lengths => false.
//   - NaN never matches.
//   - Infinities match only by exact sign.
//   - Negative delta is normalized to abs(delta).
//
// Complexity:
//   - Time O(n), Space O(1).
//
// AI-Hints:
//   - Use sliceClose when you want fatal diagnostics.
//   - Use AlmostEqualSlice for boolean table predicates.
func AlmostEqualSlice(got, want []float64, delta float64) bool {
	if len(got) != len(want) {
		return false
	}
	if delta < 0 {
		delta = -delta
	}

	for i := 0; i < len(got); i++ {
		if math.IsNaN(got[i]) || math.IsNaN(want[i]) {
			return false
		}
		if math.IsInf(got[i], 0) || math.IsInf(want[i], 0) {
			if got[i] != want[i] {
				return false
			}
			continue
		}
		if math.Abs(got[i]-want[i]) > delta {
			return false
		}
	}

	return true
}

// AssertErrorIs WRAPS errors.Is with consistent failure text.
// Implementation:
//   - Stage 1: if !errors.Is(err, target) → t.Fatalf.
//
// Behavior highlights:
//   - Reduces repeated boilerplate for sentinel checks.
//
// Inputs:
//   - err, target.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if not matching.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - O(depth) for errors.Is chain.
//
// Notes:
//   - Prefer for ErrNilMatrix, ErrDimensionMismatch checks.
//
// AI-Hints:
//   - Combine with table-driven tests for coverage.
func AssertErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("want %v; got %v", target, err)
	}
}

// InDelta reports whether got and want differ by at most delta.
//
// Policy:
//   - Negative delta is normalized to abs(delta).
//   - NaN never matches.
//   - Infinities match only by exact sign.
//
// Returns:
//   - true if got and want are within delta.
//   - false otherwise.
//
// Complexity:
//   - O(1).
//
// AI-Hints:
//   - Prefer this helper for scalar assertions.
//   - Do not use inverted boolean semantics in test helpers.
func InDelta(t *testing.T, got, want, delta float64) bool {
	t.Helper()

	if delta < 0 {
		delta = -delta
	}
	// NaN is never acceptable in comparisons: force mismatch.
	if math.IsNaN(got) || math.IsNaN(want) {
		return false
	}
	// Infinities must match exactly by sign.
	if math.IsInf(got, 0) || math.IsInf(want, 0) {
		return got == want
	}

	// Standard absolute delta check.
	return math.Abs(got-want) <= delta
}

// ---------- bench helpers () ----------

func mustDense(b *testing.B, r, c int) *matrix.Dense {
	d, err := matrix.NewZeros(r, c) // fast path alloc + zero
	if err != nil {
		b.Fatalf("NewZeros(%d,%d): %v", r, c, err)
	}

	return d
}

func fillDenseRand(b *testing.B, d *matrix.Dense, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	rows, cols := d.Rows(), d.Cols()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if err := d.Set(i, j, rng.Float64()*2-1); err != nil { // [-1,1]
				b.Fatalf("Set(%d,%d): %v", i, j, err)
			}
		}
	}
}

func onesVec(n int) []float64 {
	v := make([]float64, n)
	for i := 0; i < n; i++ {
		v[i] = 1
	}

	return v
}
