// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test provides one home for shared external test fixtures,
// numerical assertions, independent rational oracles, and benchmark setup.
// Every helper uses public Matrix operations, not reflection or privileged kernels.

package matrix_test

import (
	"context"
	"errors"
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/lvlath/go/core"
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

// ---------- Shared statistics, factorization, oracle, and benchmark helpers ----------

// statsTestDense creates a fully owning Dense input and fails immediately on
// fixture errors. A separate constructor is used for zero-column matrices.
func statsTestDense(t testing.TB, values [][]float64) *matrix.Dense {
	t.Helper()
	if len(values) == 0 {
		t.Fatal("use NewDense explicitly for a matrix with no rows")
	}
	cols := len(values[0])
	out, err := matrix.NewDense(len(values), cols)
	if err != nil {
		t.Fatalf("NewDense: %v", err)
	}
	for i, row := range values {
		if len(row) != cols {
			t.Fatalf("row %d has width %d, expected %d", i, len(row), cols)
		}
		for j, value := range row {
			if err = out.Set(i, j, value); err != nil {
				t.Fatalf("Set(%d,%d): %v", i, j, err)
			}
		}
	}
	return out
}

// statsTestValue inspects a published result using Matrix.At, not private
// storage or reflection. An unexpected accessor error always fails the test.
func statsTestValue(t testing.TB, input matrix.Matrix, i, j int) float64 {
	t.Helper()
	got, err := input.At(i, j)
	if err != nil {
		t.Fatalf("At(%d,%d): %v", i, j, err)
	}
	return got
}

// statsTestClose uses an absolute-plus-relative error budget. Non-finite
// results are never accepted as close to finite expected values.
func statsTestClose(t testing.TB, got, want, absTol, relTol float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > absTol+relTol*math.Abs(want) {
		t.Fatalf("got %.17g, want %.17g (absolute %.3g, relative %.3g)", got, want, absTol, relTol)
	}
}

// statsTestFallback intentionally hides *Dense from the implementation.
// Its At method executes the generic Matrix path; it can also inject a
// caller-owned error, proving that reductions do not discard its identity.
type statsTestFallback struct {
	matrix.Matrix
	failure error
	failRow int
	failCol int
}

func (m *statsTestFallback) At(i, j int) (float64, error) {
	if m.failure != nil && i == m.failRow && j == m.failCol {
		return 0, m.failure
	}
	return m.Matrix.At(i, j)
}

func (m *statsTestFallback) Clone() matrix.Matrix {
	return &statsTestFallback{Matrix: m.Matrix.Clone(), failure: m.failure, failRow: m.failRow, failCol: m.failCol}
}

// errStatisticsShapeAccess is returned if a kernel touches a synthetic matrix
// before rejecting an impossible shape. Shape tests verify that this error is
// never observed; no test fixture needs a panic to detect early access.
var errStatisticsShapeAccess = errors.New("matrix test: invalid synthetic data access")

// statsTestShapeOnly proves that shape validation executes before any At call
// or quadratic covariance allocation. The test never creates huge backing data.
type statsTestShapeOnly struct{ rows, cols int }

func (m statsTestShapeOnly) Rows() int { return m.rows }
func (m statsTestShapeOnly) Cols() int { return m.cols }
func (m statsTestShapeOnly) At(_, _ int) (float64, error) {
	return 0, errStatisticsShapeAccess
}
func (m statsTestShapeOnly) Set(_, _ int, _ float64) error { return errStatisticsShapeAccess }
func (m statsTestShapeOnly) Clone() matrix.Matrix          { return m }

// statisticsOracleMoments computes the exact rational mean and sample
// covariance of float64 observations, interpreting each source float64 as
// an exact binary fraction. It deliberately never invokes a matrix routine.
// Complexity: O(n*d*d) rational operations for small verification fixtures.
func statisticsOracleMoments(observations [][]float64) ([]*big.Rat, [][]*big.Rat) {
	rows, cols := len(observations), len(observations[0])
	means := make([]*big.Rat, cols)
	for j := range means {
		means[j] = new(big.Rat)
		for i := 0; i < rows; i++ {
			means[j].Add(means[j], new(big.Rat).SetFloat64(observations[i][j]))
		}
		means[j].Quo(means[j], big.NewRat(int64(rows), 1))
	}
	moments := make([][]*big.Rat, cols)
	for j := 0; j < cols; j++ {
		moments[j] = make([]*big.Rat, cols)
		for k := 0; k < cols; k++ {
			value := new(big.Rat)
			for i := 0; i < rows; i++ {
				left := new(big.Rat).Sub(new(big.Rat).SetFloat64(observations[i][j]), means[j])
				right := new(big.Rat).Sub(new(big.Rat).SetFloat64(observations[i][k]), means[k])
				value.Add(value, new(big.Rat).Mul(left, right))
			}
			moments[j][k] = value.Quo(value, big.NewRat(int64(rows-1), 1))
		}
	}
	return means, moments
}

// statisticsOracleFloat rounds an exact value once at the float64 output
// boundary. The actual oracle reductions retain full rational precision.
func statisticsOracleFloat(value *big.Rat) float64 {
	out, _ := new(big.Float).SetPrec(statisticsOraclePrecision).SetRat(value).Float64()
	return out
}

// statisticsOracleCorrelation computes a Pearson coefficient independently
// from exact rational moments, with a high-precision square root. A constant
// column receives correlation zero, matching the package's explicit policy.
func statisticsOracleCorrelation(cov [][]*big.Rat, j, k int) float64 {
	if cov[j][j].Sign() == 0 || cov[k][k].Sign() == 0 {
		return 0
	}
	numerator := new(big.Float).SetPrec(statisticsOraclePrecision).SetRat(cov[j][k])
	left := new(big.Float).SetPrec(statisticsOraclePrecision).SetRat(cov[j][j])
	right := new(big.Float).SetPrec(statisticsOraclePrecision).SetRat(cov[k][k])
	denominator := new(big.Float).SetPrec(statisticsOraclePrecision).Sqrt(new(big.Float).SetPrec(statisticsOraclePrecision).Mul(left, right))
	result, _ := new(big.Float).SetPrec(statisticsOraclePrecision).Quo(numerator, denominator).Float64()
	return result
}

// factorFixture builds a caller-owned finite Dense from a nonempty rectangular
// fixture; unexpected construction failures invalidate the test setup itself.
// Complexity: O(rows*cols) time and memory.
func factorFixture(t testing.TB, values [][]float64) *matrix.Dense {
	t.Helper()
	return statsTestDense(t, values)
}

// factorCell observes a Matrix using its public accessor, preserving the
// black-box testing boundary and failing clearly if its shape is inconsistent.
func factorCell(t testing.TB, input matrix.Matrix, row, col int) float64 {
	t.Helper()
	return statsTestValue(t, input, row, col)
}

// factorNear checks the absolute-plus-relative error budget against a known
// mathematical solution. It rejects all non-finite values before comparison.
func factorNear(t testing.TB, got, want, absolute, relative float64) {
	t.Helper()
	statsTestClose(t, got, want, absolute, relative)
}

// factorErrorIs classifies an error through errors.Is, never by matching text.
func factorErrorIs(t testing.TB, err, sentinel error) {
	t.Helper()
	if !errors.Is(err, sentinel) {
		t.Fatalf("error = %v; expected errors.Is(..., %v)", err, sentinel)
	}
}

// factorOracleRationalMatrix copies finite binary64 entries into exact rational
// values: every observation is interpreted as the represented binary fraction,
// not the decimal number printed by a Go formatter.
func factorOracleRationalMatrix(source [][]float64) [][]*big.Rat {
	out := make([][]*big.Rat, len(source))
	for i, row := range source {
		out[i] = make([]*big.Rat, len(row))
		for j, value := range row {
			out[i][j] = new(big.Rat).SetFloat64(value)
		}
	}
	return out
}

// factorOracleRationalSolve independently performs exact Gauss-Jordan
// elimination on an augmented rational matrix. The test oracle's pivot policy
// is irrelevant to the production LU algorithm and cannot mask its failures.
// Complexity: O(n^3) exact rational operations and O(n^2) storage.
func factorOracleRationalSolve(a [][]*big.Rat, b []*big.Rat) ([]*big.Rat, bool) {
	n := len(a)
	augmented := make([][]*big.Rat, n)
	for i := 0; i < n; i++ {
		augmented[i] = make([]*big.Rat, n+1)
		for j := 0; j < n; j++ {
			augmented[i][j] = new(big.Rat).Set(a[i][j])
		}
		augmented[i][n] = new(big.Rat).Set(b[i])
	}
	for k := 0; k < n; k++ {
		pivot := -1
		for i := k; i < n; i++ {
			if augmented[i][k].Sign() != 0 {
				pivot = i
				break
			}
		}
		if pivot == -1 {
			return nil, false
		}
		augmented[k], augmented[pivot] = augmented[pivot], augmented[k]
		divisor := new(big.Rat).Set(augmented[k][k])
		for j := k; j <= n; j++ {
			augmented[k][j].Quo(augmented[k][j], divisor)
		}
		for i := 0; i < n; i++ {
			if i == k {
				continue
			}
			multiplier := new(big.Rat).Set(augmented[i][k])
			for j := k; j <= n; j++ {
				term := new(big.Rat).Mul(multiplier, augmented[k][j])
				augmented[i][j].Sub(augmented[i][j], term)
			}
		}
	}
	solution := make([]*big.Rat, n)
	for i := range solution {
		solution[i] = new(big.Rat).Set(augmented[i][n])
	}
	return solution, true
}

// factorOracleLeastSquares constructs the exact rational normal-equation
// system only for the independent test oracle. The production QR solver must
// use Householder reflections instead; this deliberately different route
// prevents two implementations from agreeing on the same numerical mistake.
func factorOracleLeastSquares(a [][]float64, b []float64) ([]*big.Rat, bool) {
	m, n := len(a), len(a[0])
	gram := make([][]*big.Rat, n)
	rhs := make([]*big.Rat, n)
	for j := 0; j < n; j++ {
		gram[j] = make([]*big.Rat, n)
		rhs[j] = new(big.Rat)
		for k := 0; k < n; k++ {
			gram[j][k] = new(big.Rat)
			for i := 0; i < m; i++ {
				term := new(big.Rat).Mul(new(big.Rat).SetFloat64(a[i][j]), new(big.Rat).SetFloat64(a[i][k]))
				gram[j][k].Add(gram[j][k], term)
			}
		}
		for i := 0; i < m; i++ {
			term := new(big.Rat).Mul(new(big.Rat).SetFloat64(a[i][j]), new(big.Rat).SetFloat64(b[i]))
			rhs[j].Add(rhs[j], term)
		}
	}
	return factorOracleRationalSolve(gram, rhs)
}

// factorOracleFailingMatrix deliberately injects an accessor error while
// wrapping a real Dense. It hides the concrete *Dense fast path from callers.
type factorOracleFailingMatrix struct {
	matrix.Matrix
	fault error
}

func (m *factorOracleFailingMatrix) At(i, j int) (float64, error) {
	if i == 1 && j == 0 {
		return 0, m.fault
	}
	return m.Matrix.At(i, j)
}

func (m *factorOracleFailingMatrix) Clone() matrix.Matrix {
	return &factorOracleFailingMatrix{Matrix: m.Matrix.Clone(), fault: m.fault}
}

// factorBenchmarkDense constructs a deterministic well-conditioned tall design.
// A positive diagonal block guarantees full column rank for each requested
// shape, while lower rows exercise real Householder reflector application.
func factorBenchmarkDense(b *testing.B, rows, cols int) *matrix.Dense {
	b.Helper()
	input, err := matrix.NewDense(rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			coefficient := float64((i*19+j*47+13)%101-50) / float64(cols)
			if i == j {
				coefficient += float64(cols)
			}
			if err = input.Set(i, j, coefficient); err != nil {
				b.Fatal(err)
			}
		}
	}
	return input
}

// statsBenchFixture constructs deterministic, finite observations outside
// the timed section. Column offsets avoid artificial all-zero outputs while
// row-dependent terms expose the O(rows*cols*cols) covariance regime.
func statsBenchFixture(b *testing.B, rows, cols int) *matrix.Dense {
	b.Helper()
	input, err := matrix.NewDense(rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			value := float64((i*7919+j*1237+97)%2029-1014) / 64
			if err = input.Set(i, j, value); err != nil {
				b.Fatal(err)
			}
		}
	}
	return input
}

// matrixMustOptions assembles public matrix policy for external contract tests.
// The helper preserves the real option application order and never accesses
// private Options fields or a privileged test-only production export.
// Implementation:
//   - Stage 1: mark the helper frame for accurate caller diagnostics.
//   - Stage 2: apply the public NewMatrixOptions resolver without shortcuts.
//   - Stage 3: return its finalized value only on success; fixture errors fail.
//
// Behavior highlights: tests compare complete public Options values or execute
// an affected public operation rather than constructing a test-only snapshot.
// Inputs: t is the owning testing goroutine; options are applied left-to-right.
// Returns: finalized Options owned by this test, never a private pointer.
// Errors: any failed option resolution aborts the owning test via t.Fatalf.
// Determinism: matches the production resolver's documented last-writer policy.
// Complexity: O(len(options)) time, O(1) auxiliary space.
// Notes: it must never be called from an unowned worker goroutine.
// AI-Hints: adding a production Options getter only to simplify a test violates
// the public-surface separation; verify observable behavior through facades.
func matrixMustOptions(t *testing.T, options ...matrix.Option) matrix.Options {
	t.Helper()
	result, err := matrix.NewMatrixOptions(options...)
	if err != nil {
		t.Fatalf("NewMatrixOptions(%d options): %v", len(options), err)
	}
	return result
}

const (
	// coverageRelativeTolerance permits binary64 rounding in small numerical oracles.
	coverageRelativeTolerance = 2e-12
	// coverageEigenTolerance is an absolute Jacobi threshold for moderate test matrices.
	coverageEigenTolerance = 1e-13
	// coverageEigenRotations bounds each small eigensolver fixture.
	coverageEigenRotations = 100
	// coverageResidualPrecision exceeds binary64 precision for the residual norm oracle.
	coverageResidualPrecision = 256
	// coverageCheckpointLimit prevents an accidental unbounded replay suite.
	coverageCheckpointLimit = 1024
)

// errCoverageBackend is independent of matrix sentinels and must survive wrapping.
var errCoverageBackend = errors.New("matrix test: external backend unavailable")

// coverageAccessFault rejects one coordinate after a prescribed number of successful accesses.
// A zero-value rule is disabled; its counter belongs to one private backend instance.
// Implementation:
//   - Match row and column before consuming the local access counter.
//   - Reject matching accesses after the configured number of successful reads or writes.
//
// Behavior highlights: Unrelated coordinates never consume the fault budget.
// Inputs: row, col identify a valid fixture cell; skip is a nonnegative count.
// Returns: A deterministic error rule, not a replacement numeric value.
// Errors: errCoverageBackend is emitted only by the adapter's At or Set.
// Determinism: Fixed coordinate and per-instance access count.
// Complexity: Time O(1), space O(1) per access.
// Notes: These fixtures are single-owner; do not share counters across goroutines.
// AI-Hints: A delayed fault distinguishes an initial validation scan from a later computation.
type coverageAccessFault struct {
	enabled bool // The zero value permits every access.
	row     int  // Zero-based target row.
	col     int  // Zero-based target column.
	skip    int  // Successful matching accesses before failure.
	seen    int  // Matching accesses observed by this instance.
}

// coverageFaultAt creates a coordinate-specific error rule.
// The first skip matching accesses succeed; later matching accesses fail.
// Implementation: Store a fresh enabled rule with its counter at zero.
// Behavior highlights: skip=0 rejects the first access.
// Inputs: row, col and skip are nonnegative test-fixture coordinates/counts.
// Returns: An independently owned coverageAccessFault.
// Errors: None; the rule is applied by the backend methods.
// Determinism: Pure value construction.
// Complexity: Time O(1), space O(1).
// Notes: Do not use it to fabricate impossible Dense.Set failures.
// AI-Hints: Use separate clone rules when a kernel clones its operand.
func coverageFaultAt(row, col, skip int) coverageAccessFault {
	return coverageAccessFault{enabled: true, row: row, col: col, skip: skip}
}

// rejects advances only a matching enabled rule and reports backend refusal.
// The method does not inspect or modify a matrix cell.
// Implementation: Check coordinates, compare the counter, then advance it.
// Behavior highlights: Once the allowed accesses are exhausted, refusal persists.
// Inputs: row and col are the attempted coordinates.
// Returns: True if this access must fail.
// Errors: None directly.
// Determinism: Single-owner counter in call order.
// Complexity: Time O(1), space O(1).
// Notes: The counter records successful matching accesses only.
// AI-Hints: Do not configure a global call count when a coordinate identifies the stage.
func (f *coverageAccessFault) rejects(row, col int) bool {
	if !f.enabled || f.row != row || f.col != col {
		return false
	}
	if f.seen >= f.skip {
		return true
	}
	f.seen++
	return false
}

// coverageMatrix exercises the Matrix interface without the concrete Dense fast path.
// Its Clone preserves generic dispatch and independently owned storage, unlike a thin embedding.
// Implementation:
//   - Forward shape reads to the stable underlying Matrix.
//   - Inject a read or write failure only at a declared access boundary.
//   - Clone the underlying storage and initialize independent clone access rules.
//
// Behavior highlights: Successful access retains the underlying Matrix semantics.
// Inputs: Matrix is a valid non-nil fixture; rules are optional value fields.
// Returns: Matrix-compatible read/write and detached clone behavior.
// Errors: Underlying errors or errCoverageBackend.
// Determinism: No wall clock, global state, random faults, reflect or unsafe access.
// Complexity: At/Set O(1) for Dense; Clone O(rows*cols) for Dense.
// Notes: Tests never mutate this adapter concurrently.
// AI-Hints: Use cloneRead/cloneWrite to test Eigen after successful input validation.
type coverageMatrix struct {
	matrix.Matrix
	read       coverageAccessFault // Input read policy.
	write      coverageAccessFault // Input write policy.
	cloneRead  coverageAccessFault // Read policy installed on a detached clone.
	cloneWrite coverageAccessFault // Write policy installed on a detached clone.
}

var _ matrix.Matrix = (*coverageMatrix)(nil)
var _ core.Nilable = (*matrix.SolveResult)(nil)
var _ core.Nilable = (*matrix.SVDResult)(nil)

// At reads one cell unless its configured external backend rejects the access.
// Successful reads delegate exactly to the underlying Matrix.
// Implementation: Consult the local read rule, then forward the coordinate.
// Behavior highlights: A refusal does not publish a substitute observation.
// Inputs: i, j are zero-based coordinates.
// Returns: The underlying value or zero with an error.
// Errors: errCoverageBackend or the original Matrix.At error.
// Determinism: Local coordinate-specific access counter.
// Complexity: Time O(1) plus underlying access, space O(1).
// Notes: No source cell is modified.
// AI-Hints: Assert errors.Is against the injected sentinel, not its text.
func (m *coverageMatrix) At(i, j int) (float64, error) {
	if m.read.rejects(i, j) {
		return 0, errCoverageBackend
	}
	return m.Matrix.At(i, j)
}

// Set assigns a cell unless the external backend refuses the write.
// Refused writes leave the selected cell unchanged.
// Implementation: Consult the local write rule before delegating.
// Behavior highlights: Earlier successful writes are not rolled back here.
// Inputs: i, j are coordinates and v is the proposed value.
// Returns: nil on success, otherwise an error.
// Errors: errCoverageBackend or the original Matrix.Set error.
// Determinism: Local coordinate-specific counter.
// Complexity: Time O(1) plus underlying access, space O(1).
// Notes: This models an in-place storage failure, not an output Dense failure.
// AI-Hints: Use it for FloydWarshall or cloned Eigen workspaces.
func (m *coverageMatrix) Set(i, j int, v float64) error {
	if m.write.rejects(i, j) {
		return errCoverageBackend
	}
	return m.Matrix.Set(i, j, v)
}

// Clone creates a detached generic backend with fresh access rules.
// The clone remains a Matrix implementation rather than becoming a bare Dense.
// Implementation: Clone storage, install clone-specific rule copies, return wrapper.
// Behavior highlights: Input validation and workspace faults have independent counters.
// Inputs: A valid receiver with stable source storage.
// Returns: An independent Matrix with the same shape and numeric policy.
// Errors: None from the Matrix.Clone contract.
// Determinism: Underlying clone order plus value-copy rules.
// Complexity: Time O(rows*cols), space O(rows*cols) for Dense.
// Notes: The source's read and write counters are not inherited.
// AI-Hints: A wrapper whose Clone returns Dense cannot exercise generic Eigen rotations.
func (m *coverageMatrix) Clone() matrix.Matrix {
	return &coverageMatrix{Matrix: m.Matrix.Clone(), read: m.cloneRead, write: m.cloneWrite,
		cloneRead: m.cloneRead, cloneWrite: m.cloneWrite}
}

// coverageRawDense constructs an explicitly permissive ingestion fixture.
// It permits NaN and signed infinities only to test finite-domain boundaries.
// Implementation: Allocate with disabled write validation, then fill every cell.
// Behavior highlights: All constructor and Fill errors are checked.
// Inputs: rows, cols and row-major values; length must equal rows*cols.
// Returns: An owning Dense with the requested raw contents.
// Errors: Fixture construction errors fail the current test.
// Determinism: Row-major Fill order.
// Complexity: Time O(rows*cols), space O(rows*cols).
// Notes: Never use this helper for clean numeric oracles.
// AI-Hints: Raw storage permission does not authorize nonfinite factor or distance inputs.
func coverageRawDense(t *testing.T, rows, cols int, values []float64) *matrix.Dense {
	t.Helper()
	dense, err := matrix.NewPreparedDense(rows, cols, matrix.WithNoValidateNaNInf())
	if err != nil {
		t.Fatal(err)
	}
	MustFillRowMajor(t, dense, values)
	return dense
}

// coverageDistanceDense constructs a matrix with the distance-specific +Inf policy.
// It preserves absent paths without allowing NaN or negative infinity.
// Implementation: Allocate with WithAllowInfDistances and fill the checked fixture.
// Behavior highlights: Distances remain in physical units.
// Inputs: n is the square order; values are row-major finite or +Inf entries.
// Returns: A detached owning Dense.
// Errors: Unexpected fixture errors terminate the test.
// Determinism: Fixed storage order.
// Complexity: Time O(n^2), space O(n^2).
// Notes: Diagonal/domain validation is deliberately left to the public operation.
// AI-Hints: Do not replace +Inf with a large finite stand-in.
func coverageDistanceDense(t *testing.T, n int, values []float64) *matrix.Dense {
	t.Helper()
	dense, err := matrix.NewPreparedDense(n, n, matrix.WithAllowInfDistances())
	if err != nil {
		t.Fatal(err)
	}
	MustFillRowMajor(t, dense, values)
	return dense
}

// coverageAssertMatrixError checks both error identity and result publication.
// Allocation-producing matrix operations must not expose partially written output on failure.
// Implementation: Check errors.Is, then require a nil Matrix interface.
// Behavior highlights: Error text is never part of the assertion.
// Inputs: got is the published Matrix, err the call error, want the sentinel.
// Returns: No value; terminates the subtest on mismatch.
// Errors: Test failure only.
// Determinism: Pure assertion of the supplied result pair.
// Complexity: Time O(error chain length), space O(1).
// Notes: This helper is not suitable for in-place or documented partial-result APIs.
// AI-Hints: A nil result is independent of whether the source was mutated.
func coverageAssertMatrixError(t *testing.T, got matrix.Matrix, err, want error) {
	t.Helper()
	AssertErrorIs(t, err, want)
	if got != nil {
		t.Fatal("failed allocating operation published a matrix")
	}
}

// coverageContext deterministically delivers real cancellation at a cooperative poll.
// Err always delegates to a cancelable context, so Done and Err remain consistent.
// Implementation: Count sequential Err calls and cancel at the selected checkpoint.
// Behavior highlights: No sleeping, goroutine scheduling assumption or clock dependence.
// Inputs: cancelAt is a positive poll index, or zero for a counting-only run.
// Returns: A context suitable for one sequential public operation.
// Errors: The original context.Canceled sentinel after the chosen checkpoint.
// Determinism: Same input and same public call produce the same poll sequence.
// Complexity: Time O(1) per poll, space O(1).
// Notes: Not for concurrent use; replay is intentionally confined to small fixtures.
// AI-Hints: Discover the poll count dynamically instead of pinning private line numbers.
type coverageContext struct {
	context.Context
	cancel   context.CancelFunc // Closes the delegated context's Done channel.
	cancelAt int                // One-based checkpoint; zero disables cancellation.
	polls    int                // Number of observed sequential Err calls.
}

// Err performs a deterministic checkpoint before reading the delegated context.
// Once canceled, the underlying context retains its normal cancellation law.
// Implementation: Advance the poll counter, optionally cancel, then delegate.
// Behavior highlights: The real context owns the terminal error state.
// Inputs: A single-owner coverageContext.
// Returns: nil before cancellation, context.Canceled afterward.
// Errors: No synthetic error or wall-clock timeout.
// Determinism: Counted cooperative checkpoints.
// Complexity: Time O(1), space O(1).
// Notes: Do not call this test context from multiple goroutines.
// AI-Hints: Pass it only through public APIs, never through private kernels.
func (c *coverageContext) Err() error {
	c.polls++
	if c.cancelAt > 0 && c.polls == c.cancelAt {
		c.cancel()
	}
	return c.Context.Err()
}

// coverageReplayCancellation checks every observed cancellation boundary of a public call.
// The baseline discovers the actual poll count, so the test does not encode implementation offsets.
// Implementation:
//   - Run one successful operation with a counting context.
//   - Repeat from a fresh context, canceling at each observed checkpoint.
//   - Require context.Canceled and no published result at each interruption.
//
// Behavior highlights: A later failure cannot be mistaken for the desired cancellation.
// Inputs: invoke returns whether a result was published and its original error.
// Returns: No value; any violated cancellation contract fails the test.
// Errors: Test failures for invalid baseline, unexpected error or partial publication.
// Determinism: Sequential replay, no asynchronous timing.
// Complexity: O(P*T) time for P observed polls and one call cost T; O(call workspace) live space.
// Notes: Only small fixed inputs and immutable reusable factors belong here.
// AI-Hints: This tests cancellation semantics, not a mandatory number of polls.
func coverageReplayCancellation(t *testing.T, invoke func(context.Context) (bool, error)) {
	t.Helper()
	base, cancel := context.WithCancel(context.Background())
	counted := &coverageContext{Context: base, cancel: cancel}
	published, err := invoke(counted)
	cancel()
	if err != nil || !published {
		t.Fatalf("uncanceled baseline: published=%t err=%v", published, err)
	}
	if counted.polls == 0 || counted.polls > coverageCheckpointLimit {
		t.Fatalf("unexpected checkpoint count: %d", counted.polls)
	}
	for checkpoint := 1; checkpoint <= counted.polls; checkpoint++ {
		ctx, stop := context.WithCancel(context.Background())
		probe := &coverageContext{Context: ctx, cancel: stop, cancelAt: checkpoint}
		published, err = invoke(probe)
		stop()
		if published || !errors.Is(err, context.Canceled) {
			t.Fatalf("checkpoint %d/%d: published=%t err=%v", checkpoint, counted.polls, published, err)
		}
	}
}

// coverageResidualOracle calculates the original-coordinate Euclidean residual independently.
// Exact binary64 rational products and sums avoid cancellation and premature overflow.
// Implementation:
//   - Read every finite source coefficient and returned solution coordinate.
//   - Accumulate each residual and its square as an exact rational.
//   - Evaluate the square root once with a high-precision floating conversion.
//
// Behavior highlights: The oracle uses neither matrix.Product nor production residual helpers.
// Inputs: a is the accepted coefficient matrix, x the solution, b the RHS, column the RHS index.
// Returns: The rounded high-precision residual norm.
// Errors: Unexpected nonfinite or unreadable values fail the test.
// Determinism: Fixed ascending row and unknown order.
// Complexity: O(rows*cols) rational operations; precision-dependent arithmetic cost.
// Notes: Use the accepted symmetric midpoint for tolerated Cholesky asymmetry.
// AI-Hints: A small residual does not itself certify the accuracy of ill-conditioned coefficients.
func coverageResidualOracle(t *testing.T, a, x, b matrix.Matrix, column int) float64 {
	t.Helper()
	squares := new(big.Rat)
	for row := 0; row < a.Rows(); row++ {
		residual := new(big.Rat)
		for j := 0; j < a.Cols(); j++ {
			av := MustAt(t, a, row, j)
			xv := MustAt(t, x, j, column)
			if math.IsNaN(xv) || math.IsInf(xv, 0) || math.IsNaN(av) || math.IsInf(av, 0) {
				t.Fatal("nonfinite coefficient in residual oracle")
			}
			term := new(big.Rat).Mul(new(big.Rat).SetFloat64(av), new(big.Rat).SetFloat64(xv))
			residual.Add(residual, term)
		}
		bv := MustAt(t, b, row, column)
		if math.IsNaN(bv) || math.IsInf(bv, 0) {
			t.Fatal("nonfinite RHS in residual oracle")
		}
		residual.Sub(residual, new(big.Rat).SetFloat64(bv))
		squares.Add(squares, new(big.Rat).Mul(residual, residual))
	}
	norm := new(big.Float).SetPrec(coverageResidualPrecision).SetRat(squares)
	norm.Sqrt(norm)
	value, _ := norm.Float64()
	return value
}
