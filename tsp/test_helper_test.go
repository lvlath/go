// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp_test provides lightweight testing helpers shared across *_test.go
// files in this package. The helpers are intentionally minimal, stdlib-only,
// and avoid duplicating functionality that already lives in focused test files.
package tsp_test

import (
	"errors"
	"math"
	"math/rand"
	"slices"
	"testing"
	"time"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/tsp"
)

// -----------------------------------------------------------------------------
// Constants - single source of truth for test knobs
// -----------------------------------------------------------------------------

const (
	// epsTiny matches tsp.DefaultEps (1e-12): strict threshold to accept improvements.
	// We keep a local alias to make the test code intention explicit and decouple
	// style from production defaults if they ever change.
	epsTiny = 1e-12

	// epsLoose is a relaxed tolerance for occasional noisy geometric comparisons.
	epsLoose = 1e-3

	// seedDet is a deterministic seed for RNG-based components (when applicable).
	seedDet = int64(0)

	// startV is the canonical start vertex used across tests for normalization.
	startV = 0

	// timeTiny is a tiny wall-clock budget used to exercise deadline behavior.
	timeTiny = 1 * time.Millisecond

	// radiusN120 is the default instance size for circle-based time-budget tests.
	radiusN120 = 120
)

// -----------------------------------------------------------------------------
// Minimal matrix implementations for tests (square, bounds-checked, with Clone).
// Both types satisfy matrix.Matrix. testDense is the default; altDense is used
// to verify identical behavior across independent implementations.
// -----------------------------------------------------------------------------

// testDense is a simple dense matrix with bounds-checked At/Set and deep Clone.
type testDense struct{ a [][]float64 }

var _ matrix.Matrix = testDense{}

func (m testDense) Rows() int { return len(m.a) }
func (m testDense) Cols() int {
	if len(m.a) == 0 {
		return 0
	}

	return len(m.a[0])
}
func (m testDense) At(i, j int) (float64, error) {
	if i < 0 || i >= m.Rows() || j < 0 || j >= m.Cols() {
		return 0, matrix.ErrIndexOutOfBounds
	}

	return m.a[i][j], nil
}
func (m testDense) Set(i, j int, v float64) error {
	if i < 0 || i >= m.Rows() || j < 0 || j >= m.Cols() {
		return matrix.ErrIndexOutOfBounds
	}
	m.a[i][j] = v

	return nil
}
func (m testDense) Clone() matrix.Matrix {
	cp := make([][]float64, len(m.a))
	var i int
	for i = range m.a {
		cp[i] = append([]float64(nil), m.a[i]...)
	}

	return testDense{a: cp}
}

// altDense is a second, independent implementation to assert identical outcomes.
type altDense struct{ a [][]float64 }

var _ matrix.Matrix = altDense{}

func (m altDense) Rows() int { return len(m.a) }
func (m altDense) Cols() int {
	if len(m.a) == 0 {
		return 0
	}

	return len(m.a[0])
}
func (m altDense) At(i, j int) (float64, error) {
	if i < 0 || i >= m.Rows() || j < 0 || j >= m.Cols() {
		return 0, matrix.ErrIndexOutOfBounds
	}

	return m.a[i][j], nil
}
func (m altDense) Set(i, j int, v float64) error {
	if i < 0 || i >= m.Rows() || j < 0 || j >= m.Cols() {
		return matrix.ErrIndexOutOfBounds
	}
	m.a[i][j] = v

	return nil
}
func (m altDense) Clone() matrix.Matrix {
	cp := make([][]float64, len(m.a))
	var i int
	for i = range m.a {
		cp[i] = append([]float64(nil), m.a[i]...)
	}

	return altDense{a: cp}
}

func denseFromRows(rows [][]float64) matrix.Matrix {
	cp := make([][]float64, len(rows))
	for i := range rows {
		cp[i] = append([]float64(nil), rows[i]...)
	}
	return testDense{a: cp}
}

func denseWithNaN(n int) matrix.Matrix {
	rows := completeUnitRows(n)
	if n > 1 {
		rows[0][1] = math.NaN()
		rows[1][0] = math.NaN()
	}
	return testDense{a: rows}
}

func denseWithInf(n int, i int, j int) matrix.Matrix {
	rows := completeUnitRows(n)
	rows[i][j] = math.Inf(1)
	rows[j][i] = math.Inf(1)
	return testDense{a: rows}
}

func denseWithNegative(n int, i int, j int) matrix.Matrix {
	rows := completeUnitRows(n)
	rows[i][j] = -1
	rows[j][i] = -1
	return testDense{a: rows}
}

func denseNonZeroDiagonal(n int) matrix.Matrix {
	rows := completeUnitRows(n)
	rows[0][0] = 1
	return testDense{a: rows}
}

func completeUnitRows(n int) [][]float64 {
	rows := make([][]float64, n)
	for i := 0; i < n; i++ {
		rows[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			rows[i][j] = 1
		}
	}
	return rows
}

// -----------------------------------------------------------------------------
// Generic helpers (repeaters, assertions, numeric closeness)
// -----------------------------------------------------------------------------

// Repeat runs fn N times. Useful for determinism/stability checks.
func Repeat(t *testing.T, n int, fn func(t *testing.T)) {
	t.Helper()
	var i int // loop iterator
	for i = 0; i < n; i++ {
		fn(t)
	}
}

func mustNoError(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// mustErrIs asserts that err matches target using errors.Is.
// Intended for strict sentinels (ErrDimensionMismatch, ErrTimeLimit, ...).
func mustErrIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("want %v, got %v", target, err)
	}
}

func mustEqualInt(t *testing.T, got int, want int, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d, want %d", op, got, want)
	}
}

// mustEqualInts asserts exact equality of two integer slices (length & values).
// Prefer slices.Equal over reflect.DeepEqual for slices of basic types.
func mustEqualInts(t *testing.T, got, want []int) {
	t.Helper()
	if !slices.Equal(got, want) {
		t.Fatalf("mismatch:\n got:  %v\n want: %v", got, want)
	}
}

// floatsClose checks relative/absolute closeness of two float64 values.
// Policy: first try bitwise equality (covers +0/-0; excludes NaN),
// then absolute tolerance, then relative tolerance for larger magnitudes.
func floatsClose(a, b, rel, abs float64) bool {
	if a == b {
		// Bitwise equal (covers +0/-0, excludes NaN comparisons).
		return true
	}
	diff := math.Abs(a - b)
	if diff <= abs {
		// Absolute tolerance covers common rounding noise.
		return true
	}
	den := math.Max(math.Abs(a), math.Abs(b))

	// Relative tolerance guards against proportional error on large values.
	return diff <= rel*den
}

// mustFloatClose asserts closeness of two float64 values under rel/abs tolerances.
// The failure message includes both tolerances to simplify CI flaky analysis.
func mustFloatClose(t *testing.T, got, want, rel, abs float64) {
	t.Helper()
	if !floatsClose(got, want, rel, abs) {
		t.Fatalf("float mismatch: got=%.17g want=%.17g (rel=%.1e abs=%.1e)", got, want, rel, abs)
	}
}

func mustEqualFloat(t *testing.T, got float64, want float64, eps float64, op string) {
	t.Helper()
	if math.IsNaN(got) || math.IsNaN(want) {
		t.Fatalf("%s: NaN comparison got=%v want=%v", op, got, want)
	}
	if math.Abs(got-want) > eps {
		t.Fatalf("%s: got %.17g, want %.17g, eps %.3g", op, got, want, eps)
	}
}

// mustEqualStrings asserts exact equality of two string slices with a domain-specific
// operation label. It avoids reflect.DeepEqual so result-projection failures are easier
// to diagnose.
func mustEqualStrings(t *testing.T, got []string, want []string, op string) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s: len mismatch: got=%d want=%d; got=%v want=%v", op, len(got), len(want), got, want)
	}

	var index int
	for index = 0; index < len(got); index++ {
		if got[index] != want[index] {
			t.Fatalf("%s: value mismatch at %d: got=%q want=%q; got=%v want=%v",
				op, index, got[index], want[index], got, want)
		}
	}
}

func mustAdjEqual(t *testing.T, got [][]int, want [][]int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("adj rows: got %d, want %d", len(got), len(want))
	}
	for row := range got {
		if !slices.Equal(got[row], want[row]) {
			t.Fatalf("adj[%d]: got %v, want %v", row, got[row], want[row])
		}
	}
}

// mustValidResult verifies the public Result contract shared by solver tests.
// It checks shape, metadata, finite cost, algorithm identity, and Hamiltonian tour validity.
//
// Implementation:
//   - Stage 1: Reject nil result.
//   - Stage 2: Check public algorithm metadata.
//   - Stage 3: Validate the closed Hamiltonian tour shape through tsp.ValidateTour.
//   - Stage 4: Reject NaN, negative, and infinite costs.
//   - Stage 5: Verify optional vertex IDs when they are present.
//
// Behavior highlights:
//   - Does not require IDs because direct matrix tests may intentionally omit them.
//   - Does not require Exact/Optimal because approximation and timeout policies differ.
//   - Delegates tour-shape rules to production validation instead of duplicating them.
//
// Inputs:
//   - result: returned public solver result.
//   - n: expected number of vertices.
//   - start: expected start/end vertex.
//   - algorithm: expected result.Algorithm.
//
// Returns:
//   - None. Fails the test on violation.
//
// Errors:
//   - Test failure on nil result, invalid metadata, invalid tour, invalid cost, or malformed IDs.
//
// Determinism:
//   - Pure validation with fixed field checks.
//
// Complexity:
//   - Time O(n), Space O(n) inside tsp.ValidateTour.
//
// Notes:
//   - Use specialized helpers for Exact/Optimal/TimedOut assertions when a test targets those policies.
//
// AI-Hints:
//   - Do not check ApproximationRatio here; GreedyMatch and exact algorithms intentionally differ.
//   - Do not require IDs unless the test explicitly passed IDs into SolveMatrix/SolveGraph.
func mustValidResult(t *testing.T, result *tsp.Result, n int, start int, algorithm tsp.Algorithm) {
	t.Helper()

	if result == nil {
		t.Fatalf("nil Result")
	}
	if result.Algorithm != algorithm {
		t.Fatalf("algorithm mismatch: got %v, want %v", result.Algorithm, algorithm)
	}
	if math.IsNaN(result.Cost) || math.IsInf(result.Cost, 0) || result.Cost < 0 {
		t.Fatalf("invalid result cost: %.17g", result.Cost)
	}
	if err := tsp.ValidateTour(result.Tour, n, start); err != nil {
		t.Fatalf("invalid result tour: %v; tour=%v", err, result.Tour)
	}

	if len(result.IDs) != 0 && len(result.IDs) != n {
		t.Fatalf("invalid IDs length: got %d, want 0 or %d", len(result.IDs), n)
	}
}

// -----------------------------------------------------------------------------
// Geometric generators (Euclidean symmetric / asymmetric)
// -----------------------------------------------------------------------------

// euclid builds a symmetric metric from 2D points with zero diagonal.
func euclid(pts [][2]float64) matrix.Matrix {
	n := len(pts)
	a := make([][]float64, n)
	// Pre-allocate row slices.
	var i, j int
	for i = 0; i < n; i++ {
		a[i] = make([]float64, n)
	}

	// Fill upper triangle with Euclidean distances, mirror to lower triangle.
	var dx, dy, d float64
	for i = 0; i < n; i++ {
		for j = i; j < n; j++ {
			if i == j {
				a[i][j] = 0
				continue // keep exact zeros on the diagonal
			}
			dx = pts[i][0] - pts[j][0]
			dy = pts[i][1] - pts[j][1]
			d = math.Hypot(dx, dy) // stable sqrt(dx*dx+dy*dy)
			a[i][j] = d
			a[j][i] = d
		}
	}

	return testDense{a: a} // defined in tour_cost_utils_test.go
}

// euclidAsym builds a directed (asymmetric) matrix: Euclidean distances + bias.
// For bias>0 it ensures D(i,j) != D(j,i) while retaining a metric-like shape.
func euclidAsym(pts [][2]float64, bias float64) matrix.Matrix {
	n := len(pts)
	a := make([][]float64, n)

	// Pre-allocate row slices.
	var i, j int
	for i = 0; i < n; i++ {
		a[i] = make([]float64, n)
	}

	// Fill full matrix with directional penalty on one orientation.
	var dx, dy, d float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			if i == j {
				a[i][j] = 0
				continue // exact zeros on diagonal
			}
			dx = pts[i][0] - pts[j][0]
			dy = pts[i][1] - pts[j][1]
			d = math.Hypot(dx, dy) // Euclidean distance
			// Small directional penalty to break symmetry.
			if i < j {
				a[i][j] = d // plain distance on one direction
			} else {
				a[i][j] = d + bias // penalized distance on the opposite direction
			}
		}
	}

	return testDense{a: a} // defined in tour_cost_utils_test.go
}

// normalizeClosedToOpen rotates to start=0 and strips the closing vertex.
// (We reuse rotateToStart0 / normalizeOpenCycle from other *_test.go in the same package.)
func normalizeClosedToOpen(t *testing.T, tour []int) []int {
	t.Helper()
	rot := rotateToStart0(t, tour)  // rotate to put 0 in front (closed or open is fine)
	open := normalizeOpenCycle(rot) // return open cycle of length n

	return open
}

// normalizeOpenCycle returns an open tour (length n) if the input is a closed
// cycle (length n+1 with tour[0]==tour[n]); otherwise returns the input as-is.
func normalizeOpenCycle(tour []int) []int {
	if len(tour) >= 2 && tour[0] == tour[len(tour)-1] {
		return tour[:len(tour)-1]
	}

	return tour
}

// rotateToStart0 normalizes a tour so that it starts at 0 (open tour of length n).
// Accepts either open (n) or closed (n+1) input tours and always returns open.
func rotateToStart0(t *testing.T, tour []int) []int {
	t.Helper()
	// Rotate first (works for both forms), then normalize to open.
	rot, err := tsp.RotateTourToStart(tour, 0)
	if err != nil {
		t.Fatalf("RotateTourToStart failed: %v", err)
	}

	return normalizeOpenCycle(rot)
}

// edgesCount returns the number of undirected edges encoded in an adjacency list.
// For an undirected multigraph, |E| = (sum_v deg(v)) / 2 = (sum_v |adj[v]|) / 2.
// Time: O(n).
func edgesCount(adj [][]int) int {
	var sum int                    // sum of row lengths
	var i int                      // loop iterator
	for i = 0; i < len(adj); i++ { // walk rows
		sum += len(adj[i]) // accumulate degree(u)
	}

	return sum / 2 // divide by 2 (each edge counted twice)
}

// doubleAdj duplicates every undirected edge in an adjacency list in-place style.
// Input 'adj' is a *simple* undirected graph where every u–v appears once in adj[u]
// and once in adj[v]. Output represents a multigraph with two parallel edges per pair.
// Complexity: O(E) memory/time where E is the number of undirected edges.
func doubleAdj(adj [][]int) [][]int {
	// Allocate result with the same number of rows as input.
	n := len(adj)             // number of vertices
	var cp = make([][]int, n) // output adjacency (multigraph)
	var u int                 // row index
	for u = 0; u < n; u++ {   // iterate rows
		// Pre-allocate capacity for doubled degree to avoid re-allocations.
		var row = make([]int, 0, 2*len(adj[u])) // capacity = 2×deg(u)
		// Append original neighbors.
		row = append(row, adj[u]...) // first copy
		// Append duplicate neighbors (parallel edges).
		row = append(row, adj[u]...) // second copy
		// Assign the built row into the copy.
		cp[u] = row // store multigraph row
	}

	// Return the multigraph adjacency.
	return cp
}

func cloneAdj(adj [][]int) [][]int {
	cp := make([][]int, len(adj))
	for i := range adj {
		cp[i] = append([]int(nil), adj[i]...)
	}
	return cp
}

func seededSymmetricComplete(n int, seed int64) matrix.Matrix {
	rng := rand.New(rand.NewSource(seed))
	rows := make([][]float64, n)

	for i := 0; i < n; i++ {
		rows[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			weight := 1 + float64(rng.Intn(10_000))/100
			rows[i][j] = weight
			rows[j][i] = weight
		}
	}

	return testDense{a: rows}
}
