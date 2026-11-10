// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package matrix_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

// fillInfOffDiagZeroDiag INITIALIZES a distance-matrix fixture:
//   - diagonal = 0
//   - off-diagonal = +Inf
//
// This uses a row-major bulk fill to allow +Inf fixtures without MustSet.
func fillInfOffDiagZeroDiag(t *testing.T, d *matrix.Dense) {
	t.Helper()

	n := d.Rows()
	if n != d.Cols() {
		t.Fatalf("fixture matrix must be square, got %dx%d", d.Rows(), d.Cols())
	}

	inf := math.Inf(1)
	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		base := i * n
		for j := 0; j < n; j++ {
			if i == j {
				data[base+j] = 0.0
			} else {
				data[base+j] = inf
			}
		}
	}

	if err := d.Fill(data); err != nil {
		t.Fatalf("Fill(row-major): %v", err)
	}
}

// ---------- 5. FloydWarshall ----------

func TestFloydWarshall_Errors(t *testing.T) {
	t.Parallel()

	var err error

	// nil → ErrNilMatrix
	err = matrix.FloydWarshall(nil)
	AssertErrorIs(t, err, matrix.ErrNilMatrix)

	// non-square → ErrDimensionMismatch
	ns, _ := matrix.NewDense(3, 4)
	err = matrix.FloydWarshall(ns)

	//AssertErrorIs(t, err, matrix.ErrDimensionMismatch)
	AssertErrorIs(t, err, matrix.ErrNonSquare)
}

// Classic CLRS example (5×5, directed, with negative edges but no negative cycles).
// Expected distance matrix:
// [ [ 0, 1, -3, 2, -4],
//
//	[ 3, 0, -4, 1, -1],
//	[ 7, 4,  0, 5,  3],
//	[ 2,-1, -5, 0, -2],
//	[ 8, 5,  1, 6,  0] ]
func TestFloydWarshall_CLRS_5x5_FastPath_Correctness(t *testing.T) {
	t.Parallel()

	const n = 5
	var (
		i, j int
		err  error
	)

	A, _ := matrix.NewPreparedDense(n, n, matrix.WithAllowInfDistances())

	// init ∞ off-diagonal, 0 on diagonal via raw row-major fill
	fillInfOffDiagZeroDiag(t, A)
	// edges (u→v = w)
	MustSet(t, A, 0, 1, 3)
	MustSet(t, A, 0, 2, 8)
	MustSet(t, A, 0, 4, -4)
	MustSet(t, A, 1, 3, 1)
	MustSet(t, A, 1, 4, 7)
	MustSet(t, A, 2, 1, 4)
	MustSet(t, A, 3, 0, 2)
	MustSet(t, A, 3, 2, -5)
	MustSet(t, A, 4, 3, 6)

	if err = matrix.FloydWarshall(A); err != nil {
		t.Fatalf("FloydWarshall(%v): %v", A, err)
	}

	exp := [][]float64{
		{0, 1, -3, 2, -4},
		{3, 0, -4, 1, -1},
		{7, 4, 0, 5, 3},
		{2, -1, -5, 0, -2},
		{8, 5, 1, 6, 0},
	}
	var got float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			got, _ = A.At(i, j)
			if got != exp[i][j] {
				t.Fatalf("dist[%d,%d]=%v; want %v", i, j, got, exp[i][j])
			}
		}
	}
}

// The same CLRS graph, but forced interface fallback via wrapper.
// Result must match the fast-path one element-by-element.
func TestFloydWarshall_CLRS_5x5_Fallback_MatchesFast(t *testing.T) {
	t.Parallel()

	const n = 5
	var (
		i, j int
		err  error
	)

	makeCLRS := func() matrix.Matrix {
		M, _ := matrix.NewPreparedDense(n, n, matrix.WithAllowInfDistances())
		fillInfOffDiagZeroDiag(t, M)
		// edges
		_ = M.Set(0, 1, 3)
		_ = M.Set(0, 2, 8)
		_ = M.Set(0, 4, -4)
		_ = M.Set(1, 3, 1)
		_ = M.Set(1, 4, 7)
		_ = M.Set(2, 1, 4)
		_ = M.Set(3, 0, 2)
		_ = M.Set(3, 2, -5)
		_ = M.Set(4, 3, 6)
		return M
	}

	fast := makeCLRS()       // *Dense
	slow := hide{makeCLRS()} // wrapped → fallback
	if err = matrix.FloydWarshall(fast); err != nil {
		t.Fatalf("matrix.FloydWarshall(fast): %v", err)
	}
	if err = matrix.FloydWarshall(slow); err != nil {
		t.Fatalf("matrix.FloydWarshall(slow): %v", err)
	}

	var a, b float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			a, _ = fast.At(i, j)
			b, _ = slow.At(i, j)
			if b != a {
				t.Fatalf("mismatch at dist[%d,%d]=%v; want %v", i, j, b, a)
			}
		}
	}
}

// Unreachable nodes remain at +Inf; diagonal zeros; triangle inequality holds;
// and running FW again on the computed distance matrix does not change it (idempotent).
func TestFloydWarshall_Unreachable_Properties_And_Idempotent(t *testing.T) {
	t.Parallel()

	const n = 6
	var i, j, k int
	var err error

	D, _ := matrix.NewPreparedDense(n, n, matrix.WithAllowInfDistances())

	inf := math.Inf(1)
	fillInfOffDiagZeroDiag(t, D)

	// Build an undirected component on {0,1,2} and a directed chain {3 -> 4} ; node 5 isolated.
	// Undirected edges (symmetric weights):
	MustSet(t, D, 0, 1, 2)
	MustSet(t, D, 1, 0, 2)
	MustSet(t, D, 1, 2, 3)
	MustSet(t, D, 2, 1, 3)
	MustSet(t, D, 0, 2, 10)
	MustSet(t, D, 2, 0, 10)
	// Directed chain 3→4 (weight 7); 4 has no outgoing edges back.
	MustSet(t, D, 3, 4, 7)

	if err = matrix.FloydWarshall(D); err != nil {
		t.Fatalf("matrix.FloydWarshall(D): %v", err)
	}

	// 1) diagonal zeros
	var v float64
	for i = 0; i < n; i++ {
		v, _ = D.At(i, i)
		if v != 0.0 {
			t.Fatalf("diagonal must be zero at [%d,%d]=%v; want %v", i, i, v, 0.0)
		}
	}

	// 2) unreachable pairs stay +Inf (from {0,1,2} or {3,4} to 5; and from 5 to others)
	var v1, v2 float64
	for i = 0; i < n; i++ {
		if i == 5 {
			continue
		}
		v1, _ = D.At(i, 5)
		v2, _ = D.At(5, i)
		if v1 != inf {
			t.Fatalf("expect unreachable i→5, i=%d, got=%v; want %v", i, v1, inf)
		}
		if v2 != inf {
			t.Fatalf("expect unreachable 5→i, i=%d, got=%v; want %v", i, v2, inf)
		}
	}

	// 3) triangle inequality: d[i,j] ≤ d[i,k] + d[k,j] for all i,j,k with finite paths
	var ij, ik, kj float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			ij, _ = D.At(i, j)
			for k = 0; k < n; k++ {
				ik, _ = D.At(i, k)
				kj, _ = D.At(k, j)
				if ik == inf || kj == inf {
					continue
				}

				if ij > ik+kj {
					t.Fatalf("triangle inequality violated for (%d,%d,%d)", i, j, k)
				}
			}
		}
	}

	// 4) idempotent: running FW again on the distance matrix must not change it
	before := D.Clone()
	if err = matrix.FloydWarshall(D); err != nil {
		t.Fatalf("matrix.FloydWarshall(D): %v", err)
	}
	var a, b float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			a, _ = before.At(i, j)
			b, _ = D.At(i, j)
			if a != b {
				t.Fatalf("idempotency mismatch at [%d,%d]=%v; want %v", i, j, a, b)
			}
		}
	}
}

// Negative cycle sanity: if a negative cycle exists and is reachable from i,
// Floyd-Warshall must classify the matrix as invalid by returning ErrNegativeCycle.
// The matrix is mutated in-place before classification, so the negative diagonal
// can still be inspected as a diagnostic witness.
func TestFloydWarshall_NegativeCycle_DiagonalNegative(t *testing.T) {
	t.Parallel()

	const n = 4 // 0-1-2 - negative cycle; 3 - isolated
	var i int
	var err error

	G, _ := matrix.NewPreparedDense(n, n, matrix.WithAllowInfDistances())
	fillInfOffDiagZeroDiag(t, G)

	// Negative cycle: 0→1 (1), 1→2 (-1), 2→0 (-1) => total -1
	MustSet(t, G, 0, 1, 1)
	MustSet(t, G, 1, 2, -1)
	MustSet(t, G, 2, 0, -1)

	err = matrix.FloydWarshall(G)
	if !errors.Is(err, matrix.ErrNegativeCycle) {
		t.Fatalf("FloydWarshall negative cycle: got %v, want ErrNegativeCycle", err)
	}

	// Nodes 0..2 are in the negative cycle: diagonals should be negative
	// after the in-place relaxation phase.
	for i = 0; i < 3; i++ {
		d, atErr := G.At(i, i)
		if atErr != nil {
			t.Fatalf("At(%d,%d): %v", i, i, atErr)
		}
		if d >= 0 {
			t.Fatalf("expected negative diagonal at node %d due to negative cycle; got %v", i, d)
		}
	}

	// Node 3 is isolated and should keep zero diagonal.
	d, atErr := G.At(3, 3)
	if atErr != nil {
		t.Fatalf("At(3,3): %v", atErr)
	}
	if d != 0 {
		t.Fatalf("isolated node must keep zero diagonal [3,3]=%v; want 0", d)
	}
}

// TestInitDistancesInPlace_NegativeSelfLoopPreservesDiagonal FIXES the contract:
// a negative self-loop weight on the diagonal MUST remain negative after initialization.
func TestInitDistancesInPlace_NegativeSelfLoopPreservesDiagonal(t *testing.T) {
	// Stage 1: Allocate a 2×2 distance matrix container.
	d, err := matrix.NewPreparedDense(2, 2, matrix.WithAllowInfDistances())
	if err != nil {
		t.Fatalf("NewDense(2,2): %v", err)
	}

	// Stage 2: Raw-ingest values:
	//   - diag[0,0] is a negative self-loop and MUST remain negative.
	//   - diag[1,1] is zero (baseline).
	//   - off-diagonal [0,1] is +Inf meaning "no direct edge".
	//   - off-diagonal [1,0] is a finite edge weight.
	vals := []float64{
		-2, math.Inf(1),
		5, 0,
	}
	if err = d.Fill(vals); err != nil {
		t.Fatalf("Fill(row-major): %v", err)
	}

	// Stage 3: Run the initializer under test.
	if err = matrix.InitDistancesInPlace(d); err != nil {
		t.Fatalf("InitDistancesInPlace: %v", err)
	}

	// Stage 4: Assert the negative diagonal is preserved (not overwritten to 0).
	got00, err := d.At(0, 0)
	if err != nil {
		t.Fatalf("At(0,0): %v", err)
	}
	if got00 != -2 {
		t.Fatalf("diag[0,0]=%v; want %v (negative self-loop must be preserved)", got00, -2)
	}

	// Optional sanity: ensure +Inf sentinel off-diagonal is still +Inf (no accidental clobber).
	got01, err := d.At(0, 1)
	if err != nil {
		t.Fatalf("At(0,1): %v", err)
	}
	if !math.IsInf(got01, 1) {
		t.Fatalf("m[0,1]=%v; want +Inf (no-path sentinel must remain +Inf)", got01)
	}
}

func TestFloydWarshall_ClosesPositiveInfNoPathDistance(t *testing.T) {
	t.Parallel()

	d, err := matrix.NewPreparedDense(3, 3, matrix.WithAllowInfDistances())
	if err != nil {
		t.Fatalf("NewPreparedDense: %v", err)
	}

	values := [][]float64{
		{0, 2, math.Inf(+1)},
		{math.Inf(+1), 0, 3},
		{math.Inf(+1), math.Inf(+1), 0},
	}

	for i := range values {
		for j := range values[i] {
			if err = d.Set(i, j, values[i][j]); err != nil {
				t.Fatalf("Set(%d,%d): %v", i, j, err)
			}
		}
	}

	if err = matrix.FloydWarshall(d); err != nil {
		t.Fatalf("FloydWarshall: %v", err)
	}

	got, err := d.At(0, 2)
	if err != nil {
		t.Fatalf("At(0,2): %v", err)
	}
	if got != 5 {
		t.Fatalf("distance[0,2]: got %v, want 5", got)
	}
}

func TestFloydWarshall_RejectsNaNDespiteDisabledDenseValidation(t *testing.T) {
	t.Parallel()

	d, err := matrix.NewPreparedDense(
		2,
		2,
		matrix.WithNoValidateNaNInf(),
		matrix.WithAllowInfDistances(),
	)
	if err != nil {
		t.Fatalf("NewPreparedDense: %v", err)
	}

	if err = d.Set(0, 0, 0); err != nil {
		t.Fatalf("Set(0,0): %v", err)
	}
	if err = d.Set(0, 1, math.NaN()); err != nil {
		t.Fatalf("Set(0,1,NaN): %v", err)
	}
	if err = d.Set(1, 0, 1); err != nil {
		t.Fatalf("Set(1,0): %v", err)
	}
	if err = d.Set(1, 1, 0); err != nil {
		t.Fatalf("Set(1,1): %v", err)
	}

	err = matrix.FloydWarshall(d)
	if !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("FloydWarshall(NaN): got %v, want ErrNaNInf", err)
	}
}
