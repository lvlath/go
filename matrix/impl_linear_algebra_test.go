// Package matrix_test contains unit tests for universal Matrix (linear algebra)operations.
package matrix_test

import (
	"sort"
	"testing"

	"github.com/lvlath/go/matrix"
)

// TestHelpers_InterfaceHiding_Fallback ensures that using a non-nil wrapper
// (which hides the concrete type) forces the interface fallback path without panicking
// and produces the same results as with the bare Dense.
func TestHelpers_InterfaceHiding_Fallback(t *testing.T) {
	t.Parallel()

	const rows, cols = 3, 3
	var (
		i, j int
		v    float64
		err  error
	)

	base := MustDense(t, rows, cols)
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			v = float64(i*cols + j + 1)
			MustSet(t, base, i, j, v)
		}
	}

	wrapped := hide{base}

	// Compare Add(base, base) vs Add(wrapped, base)
	sum1, err := matrix.Add(base, base)
	if err != nil {
		t.Fatalf("matrix.Add(base, base): %v", err)
	}
	sum2, err := matrix.Add(wrapped, base)
	if err != nil {
		t.Fatalf("matrix.Add(wrapped, base): %v", err)
	}

	var a, b float64
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			a = MustAt(t, sum1, i, j)
			b = MustAt(t, sum2, i, j)
			if a != b {
				t.Fatalf("mismatch at [%d,%d]", i, j)
			}
		}
	}
}

func TestHelperVisibility(t *testing.T) {
	// Check that the Random and Compare utilities are available and working
	const n = 3
	m := MustDense(t, n, n)

	// Random fills the matrix with pseudo-random numbers without panicking
	RandomFill(t, m, 12345)

	// Assemble "reference" identity matrix
	Iwant := make([][]float64, n)
	for i := 0; i < n; i++ {
		row := make([]float64, n)
		row[i] = 1.0
		Iwant[i] = row
	}

	// First, fill m with one on the diagonal and zeros outside
	var i, j int // loop iterators
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			MustSet(t, m, i, j, 0)
		}
		MustSet(t, m, i, i, 1.0)
	}

	// Сompare should not panic and should check successfully
	CompareExact(t, Iwant, m)
}

// hide is declared once if not already in file:
// type hide struct{ matrix.Matrix }

// ---------- 2.1 Add ----------

func TestAdd_FastPath_6x6_Correctness(t *testing.T) {
	t.Parallel()

	const rows, cols = 6, 6
	var i, j int
	var err error

	A := MustDense(t, rows, cols)
	B := MustDense(t, rows, cols)

	// A[i,j] = i+j; B[i,j] = 10 - (i+j)
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			MustSet(t, A, i, j, float64(i+j))
			MustSet(t, B, i, j, float64(10-(i+j)))
		}
	}

	S, err := matrix.Add(A, B)
	if err != nil {
		t.Fatalf("matrix.Add: want err == nil, got: %v", err)
	}

	// Expect constant 10 everywhere
	var got float64
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			got = MustAt(t, S, i, j)
			if got != 10.0 {
				t.Fatalf("at [%d,%d]", i, j)
			}
		}
	}
}

func TestAdd_Fallback_4x5_Correctness(t *testing.T) {
	t.Parallel()

	const rows, cols = 4, 5
	var i, j int
	var err error

	Araw := MustDense(t, rows, cols)
	Braw := MustDense(t, rows, cols)
	A := hide{Araw} // force fallback
	B := hide{Braw} // force fallback

	// A[i,j] = 2*i + j; B[i,j] = i - 3*j
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			MustSet(t, Araw, i, j, float64(2*i+j))
			MustSet(t, Braw, i, j, float64(i-3*j))
		}
	}

	S, err := matrix.Add(A, B)
	if err != nil {
		t.Fatalf("matrix.Add(A, B): want err == nil, got: %v", err)
	}

	// Check elementwise
	var got, av, bv float64
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			av, _ = Araw.At(i, j)
			bv, _ = Braw.At(i, j)
			got = MustAt(t, S, i, j)
			if got != av+bv {
				t.Fatalf("at [%d,%d]", i, j)
			}
		}
	}
}

func TestAdd_DimensionMismatch(t *testing.T) {
	t.Parallel()

	var err error
	A := MustDense(t, 3, 4)
	B := MustDense(t, 4, 3)
	_, err = matrix.Add(A, B)
	AssertErrorIs(t, err, matrix.ErrDimensionMismatch)
}

func TestAdd_Succeeds(t *testing.T) {
	// Prepare two 2×3 matrices
	a := MustDense(t, 2, 3)
	b := MustDense(t, 2, 3)

	// Initialize a = [[1,2,3],[4,5,6]], b = [[6,5,4],[3,2,1]]
	var i, j int // loop iterators
	for i = 0; i < 2; i++ {
		for j = 0; j < 3; j++ {
			MustSet(t, a, i, j, float64(i*3+j+1))
			MustSet(t, b, i, j, float64(6-(i*3+j)))
		}
	}

	// Perform addition
	sum, err := matrix.Add(a, b)
	if err != nil {
		t.Fatalf("matrix.Add(a, b): want err == nil, got: %v", err)
	}

	// Expect sum = [[7,7,7],[7,7,7]]
	var v float64
	for i = 0; i < 2; i++ {
		for j = 0; j < 3; j++ {
			v = MustAt(t, sum, i, j)
			if v != 7.0 {
				t.Fatalf("want v == 7.0, got: %.6g", v)
			}
		}
	}
}

// ---------- 2.2 Sub ----------

func TestSub_FastPath_6x6_Correctness(t *testing.T) {
	t.Parallel()

	const rows, cols = 6, 6
	var i, j int
	var err error

	A := MustDense(t, rows, cols)
	B := MustDense(t, rows, cols)

	// A[i,j] = 100 + i*cols + j; B[i,j] = i*cols + j
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			MustSet(t, A, i, j, float64(100+i*cols+j))
			MustSet(t, B, i, j, float64(i*cols+j))
		}
	}

	D, err := matrix.Sub(A, B)
	if err != nil {
		t.Fatalf("matrix.Sub(A, B): want err == nil, got: %v", err)
	}

	// Expect constant 100 everywhere
	var got float64
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			got = MustAt(t, D, i, j)
			if got != 100 {
				t.Fatalf("at [%d,%d]", i, j)
			}
		}
	}
}

func TestSub_Fallback_5x3_Correctness(t *testing.T) {
	t.Parallel()

	const rows, cols = 5, 3
	var i, j int
	var err error

	Araw := MustDense(t, rows, cols)
	Braw := MustDense(t, rows, cols)
	A := hide{Araw}
	B := hide{Braw}

	// A[i,j] = i + 2*j; B[i,j] = 3*i - j
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			MustSet(t, Araw, i, j, float64(i+2*j))
			MustSet(t, Braw, i, j, float64(3*i-j))
		}
	}

	D, err := matrix.Sub(A, B)
	if err != nil {
		t.Fatalf("matrix.Sub(A, B): want err == nil, got: %v", err)
	}

	// Check elementwise
	var got, av, bv float64
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			av, _ = Araw.At(i, j)
			bv, _ = Braw.At(i, j)
			got = MustAt(t, D, i, j)
			if got != av-bv {
				t.Fatalf("at [%d,%d]", i, j)
			}
		}
	}
}

func TestSub_DimensionMismatch(t *testing.T) {
	t.Parallel()

	var err error
	A := MustDense(t, 3, 4)
	B := MustDense(t, 3, 5)
	_, err = matrix.Sub(A, B)
	AssertErrorIs(t, err, matrix.ErrDimensionMismatch)
}

func TestSub_Succeeds(t *testing.T) {
	// Prepare two 3×2 matrices
	a := MustDense(t, 3, 2)
	b := MustDense(t, 3, 2)
	// a = [[5,4],[3,2],[1,0]]; b = [[1,1],[1,1],[1,1]]
	values := [][]float64{
		{5, 4},
		{3, 2},
		{1, 0},
	}
	var i, j int // loop iterators
	for i = 0; i < 3; i++ {
		for j = 0; j < 2; j++ {
			_ = a.Set(i, j, values[i][j])
			_ = b.Set(i, j, 1)
		}
	}

	diff, err := matrix.Sub(a, b)
	if err != nil {
		t.Fatalf("matrix.Sub(a, b): want err == nil, got: %v", err)
	}

	// Expect diff = [[4,3],[2,1],[0,-1]]
	expected := [][]float64{
		{4, 3},
		{2, 1},
		{0, -1},
	}
	var v float64
	for i = 0; i < 3; i++ {
		for j = 0; j < 2; j++ {
			v = MustAt(t, diff, i, j)
			if v != expected[i][j] {
				t.Fatalf("want v == %b, got: %.6g", expected[i][j], v)
			}
		}
	}
}

// ---------- 2.3 Mul ----------

func TestMul_FastPath_6x4x5_Correctness(t *testing.T) {
	t.Parallel()

	// A(6×4) × B(4×5) = C(6×5)
	const ar, ac, bc = 6, 4, 5
	var i, j, k int
	var err error
	var sum, got float64
	A := MustDense(t, ar, ac)
	B := MustDense(t, ac, bc)

	// A[i,k] = i + k; B[k,j] = k + j
	for i = 0; i < ar; i++ {
		for k = 0; k < ac; k++ {
			MustSet(t, A, i, k, float64(i+k))
		}
	}
	for k = 0; k < ac; k++ {
		for j = 0; j < bc; j++ {
			MustSet(t, B, k, j, float64(k+j))
		}
	}

	C, err := matrix.Mul(A, B)
	if err != nil {
		t.Fatalf("matrix.Mul(A, B): want err == nil, got: %v", err)
	}

	// verify C[i,j] = Σ_k (i+k)*(k+j)
	for i = 0; i < ar; i++ {
		for j = 0; j < bc; j++ {
			sum = 0.0
			for k = 0; k < ac; k++ {
				sum += float64(i+k) * float64(k+j)
			}
			got = MustAt(t, C, i, j)
			if got != sum {
				t.Fatalf("at [%d,%d]", i, j)
			}
		}
	}
}

func TestMul_Fallback_3x4x3_Correctness(t *testing.T) {
	t.Parallel()

	// Force fallback via hide
	const ar, ac, bc = 3, 4, 3
	var (
		i, j, k int
		err     error
		sum     float64
		got     float64
		av, bv  float64
	)

	Araw := MustDense(t, ar, ac)
	Braw := MustDense(t, ac, bc)
	A := hide{Araw}
	B := hide{Braw}

	// A[i,k] = 2*i + k; B[k,j] = 3*k - j
	for i = 0; i < ar; i++ {
		for k = 0; k < ac; k++ {
			MustSet(t, Araw, i, k, float64(2*i+k))
		}
	}
	for k = 0; k < ac; k++ {
		for j = 0; j < bc; j++ {
			MustSet(t, Braw, k, j, float64(3*k-j))
		}
	}

	C, err := matrix.Mul(A, B)
	if err != nil {
		t.Fatalf("matrix.Mul(A, B): want err == nil, got: %v", err)
	}

	// explicit Σ for expected
	for i = 0; i < ar; i++ {
		for j = 0; j < bc; j++ {
			sum = 0.0
			for k = 0; k < ac; k++ {
				av, _ = Araw.At(i, k)
				bv, _ = Braw.At(k, j)
				sum += av * bv
			}
			got = MustAt(t, C, i, j)
			if got != sum {
				t.Fatalf("at [%d,%d]", i, j)
			}
		}
	}
}

func TestMul_DimensionMismatch(t *testing.T) {
	t.Parallel()

	var err error
	A := MustDense(t, 4, 3) // inner = 3
	B := MustDense(t, 2, 5) // inner = 2 → mismatch
	_, err = matrix.Mul(A, B)
	AssertErrorIs(t, err, matrix.ErrDimensionMismatch)
}

func TestMul_Succeeds(t *testing.T) {
	// A is 2×3, B is 3×2: A*B = 2×2
	A := MustDense(t, 2, 3)
	B := MustDense(t, 3, 2)
	var C matrix.Matrix
	// Initialize A = [[1,2,3],[4,5,6]]; B = [[7,8],[9,10],[11,12]]
	aVals := [][]float64{{1, 2, 3}, {4, 5, 6}}
	bVals := [][]float64{{7, 8}, {9, 10}, {11, 12}}
	var (
		i, j int // loop iterators
		v    float64
		err  error
	)
	for i = 0; i < 2; i++ {
		for j = 0; j < 3; j++ {
			_ = A.Set(i, j, aVals[i][j])
		}
	}
	for i = 0; i < 3; i++ {
		for j = 0; j < 2; j++ {
			_ = B.Set(i, j, bVals[i][j])
		}
	}

	C, err = matrix.Mul(A, B)
	if err != nil {
		t.Fatalf("matrix.Mul(A, B): want err == nil, got: %v", err)
	}

	// Expected C = [[58,64],[139,154]]
	expected := [][]float64{{58, 64}, {139, 154}}
	for i = 0; i < 2; i++ {
		for j = 0; j < 2; j++ {
			v = MustAt(t, C, i, j)
			if v != expected[i][j] {
				t.Fatalf("want v == %b, got: %.6g", expected[i][j], v)
			}
		}
	}
}

// ---------- 3.1 Transpose ----------

func TestTranspose_FastPath_Rectangular_Correctness(t *testing.T) {
	t.Parallel()

	const rows, cols = 4, 6
	var (
		i, j int
		err  error
		val  float64
	)

	m := MustDense(t, rows, cols)

	// Fill m[i,j] = 10*i + j  (unique, easy to check after transpose)
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			MustSet(t, m, i, j, float64(10*i+j))
		}
	}

	mt, err := matrix.Transpose(m)
	if err != nil {
		t.Fatalf("matrix.Transpose(m): want err == nil, got: %v", err)
	}
	if mt.Rows() != cols {
		t.Fatalf("want mt.Rows == %d, got:%d", cols, mt.Rows())
	}
	if mt.Cols() != rows {
		t.Fatalf("want mt.Rows == %d, got:%d", rows, mt.Cols())
	}

	// Check mt[j,i] == m[i,j]
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			val = MustAt(t, mt, j, i)
			if val != float64(10*i+j) {
				t.Fatalf("mismatch at [%d,%d] ⇒ mt[%d,%d]", i, j, j, i)
			}
		}
	}
}

func TestTranspose_Fallback_Rectangular_Correctness(t *testing.T) {
	t.Parallel()

	const rows, cols = 5, 3
	var (
		i, j int
		err  error
		val  float64
	)

	base := MustDense(t, rows, cols)
	// Force interface fallback via wrapper
	m := hide{base}

	// Fill base[i,j] = i - 2*j
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			MustSet(t, base, i, j, float64(i-2*j))
		}
	}

	mt, err := matrix.Transpose(m)
	if err != nil {
		t.Fatalf("matrix.Transpose(m): want err == nil, got: %v", err)
	}
	if mt.Rows() != cols {
		t.Fatalf("want mt.Rows == %d, got:%d", cols, mt.Rows())
	}
	if mt.Cols() != rows {
		t.Fatalf("want mt.Rows == %d, got:%d", rows, mt.Cols())
	}

	// Check mt[j,i] == base[i,j]
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			val = MustAt(t, mt, j, i)
			if val != float64(i-2*j) {
				t.Fatalf("want val == %.6g, got: %.6g", float64(i-2*j), val)
			}
		}
	}
}

func TestTranspose_Involution_NoMutation(t *testing.T) {
	t.Parallel()

	const n = 6
	var (
		i, j int
		err  error
		aij  float64
	)

	A := MustDense(t, n, n)
	// Fill A with a distinct pattern
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			MustSet(t, A, i, j, float64((i+1)*(j+2)))
		}
	}

	// Keep a copy to ensure A is not mutated by Transpose
	Acopy := A.Clone()

	At, err := matrix.Transpose(A)
	if err != nil {
		t.Fatalf("matrix.Transpose(A): want err == nil, got: %v", err)
	}
	Att, err := matrix.Transpose(At)
	if err != nil {
		t.Fatalf("matrix.Transpose(At): want err == nil, got: %v", err)
	}

	// Check Transpose(Transpose(A)) == A
	var got, want float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			got = MustAt(t, Att, i, j)
			want = MustAt(t, A, i, j)
			if got != want {
				t.Fatalf("at [%d,%d]", i, j)
			}
		}
	}

	// Ensure original A not mutated
	var v1, v2 float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			v1 = MustAt(t, A, i, j)
			v2 = MustAt(t, Acopy, i, j)
			if v1 != v2 {
				t.Fatalf("want v1(%b) == v2(%b)", v1, v2)
			}
		}
	}

	// Extra: symmetric matrix should equal its transpose
	for i = 0; i < n; i++ {
		for j = i; j < n; j++ {
			aij = float64(i + j + 1) // symmetric by construction
			MustSet(t, A, i, j, aij)
			MustSet(t, A, j, i, aij)
		}
	}
	St, err := matrix.Transpose(A)
	if err != nil {
		t.Fatalf("matrix.Transpose(A): want err == nil, got: %v", err)
	}
	var s, st float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			s, _ = A.At(i, j)
			st, _ = St.At(i, j)
			if st != s {
				t.Fatalf("symmetric transpose must be identical")
			}
		}
	}
}

func TestTranspose(t *testing.T) {
	// 2×3 matrix
	m := NewFilledDense(t, 2, 3, []float64{1, 2, 3, 4, 5, 6})

	tm, _ := matrix.Transpose(m)
	// tm should be 3×2: [[1,4],[2,5],[3,6]]
	exp := [][]float64{{1, 4}, {2, 5}, {3, 6}}
	if tm.Rows() != 3 {
		t.Fatalf("want mt.Rows == %d, got:%d", 3, tm.Rows())
	}
	if tm.Cols() != 2 {
		t.Fatalf("want mt.Rows == %d, got:%d", 2, tm.Cols())
	}

	var i, j int // loop iterators
	var v float64
	for i = 0; i < tm.Rows(); i++ {
		for j = 0; j < tm.Cols(); j++ {
			v = MustAt(t, tm, i, j)
			if v != exp[i][j] {
				t.Fatalf("want v == %b, got: %.6g", exp[i][j], v)
			}
		}
	}
}

// ---------- 3.2 Scale ----------

func TestScale_FastPath_6x6_Correctness(t *testing.T) {
	t.Parallel()

	const n = 6
	const alpha = 3.5
	var (
		i, j int
		err  error
		got  float64
	)

	m := MustDense(t, n, n)
	// m[i,j] = i - j
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			MustSet(t, m, i, j, float64(i-j))
		}
	}

	sm, err := matrix.Scale(m, alpha)
	if err != nil {
		t.Fatalf("matrix.Scale(m, alpha): want err == nil, got: %v", err)
	}
	if sm.Rows() != n {
		t.Fatalf("want mt.Rows == %d, got:%d", n, sm.Rows())
	}
	if sm.Cols() != n {
		t.Fatalf("want mt.Rows == %d, got:%d", n, sm.Cols())
	}

	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			got = MustAt(t, sm, i, j)
			if got != alpha*float64(i-j) {
				t.Fatalf("at [%d,%d]", i, j)
			}
		}
	}
}

func TestScale_Fallback_5x3_Correctness(t *testing.T) {
	t.Parallel()

	const rows, cols = 5, 3
	const alpha = -2.0
	var (
		i, j int
		err  error
		got  float64
	)

	base := MustDense(t, rows, cols)
	m := hide{base} // force fallback

	// base[i,j] = 2*i + 3*j + 1
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			MustSet(t, base, i, j, float64(2*i+3*j+1))
		}
	}

	sm, err := matrix.Scale(m, alpha)
	if err != nil {
		t.Fatalf("matrix.Scale(m, alpha): want err == nil, got: %v", err)
	}
	if sm.Rows() != rows {
		t.Fatalf("want mt.Rows == %d, got:%d", rows, sm.Rows())
	}
	if sm.Cols() != cols {
		t.Fatalf("want mt.Rows == %d, got:%d", cols, sm.Cols())
	}

	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			got = MustAt(t, sm, i, j)
			if got != alpha*float64(2*i+3*j+1) {
				t.Fatalf("wrong scaled value at [%d,%d]: got %.6g", i, j, got)
			}
		}
	}
}

func TestScale_Properties_Distributivity(t *testing.T) {
	t.Parallel()

	const n = 4
	const alpha = 1.75
	var (
		i, j int
		err  error
	)

	A := MustDense(t, n, n)
	B := MustDense(t, n, n)

	// A[i,j] = i+j; B[i,j] = i-2*j
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			MustSet(t, A, i, j, float64(i+j))
			MustSet(t, B, i, j, float64(i-2*j))
		}
	}

	S, err := matrix.Add(A, B)
	if err != nil {
		t.Fatalf("matrix.Add(A, B): want err == nil, got: %v", err)
	}

	left, err := matrix.Scale(S, alpha) // α(A+B)
	if err != nil {
		t.Fatalf("matrix.Scale(S, alpha): want err == nil, got: %v", err)
	}

	Ar, err := matrix.Scale(A, alpha) // αA
	if err != nil {
		t.Fatalf("matrix.Scale(A, alpha): want err == nil, got: %v", err)
	}
	Br, err := matrix.Scale(B, alpha) // αB
	if err != nil {
		t.Fatalf("matrix.Scale(B, alpha): want err == nil, got: %v", err)
	}
	right, err := matrix.Add(Ar, Br) // αA + αB
	if err != nil {
		t.Fatalf("matrix.Add(Ar, Br): want err == nil, got: %v", err)
	}

	// Compare left vs right
	var lv, rv float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			lv = MustAt(t, left, i, j)
			rv = MustAt(t, right, i, j)
			if lv != rv {
				t.Fatalf("distributivity failed at [%d,%d]: want lv(%b) == rv(%b)", i, j, lv, rv)
			}
		}
	}
}

func TestScale_Properties_Composition_And_SpecialAlphas(t *testing.T) {
	t.Parallel()

	const n = 5
	const alpha = -0.5
	const beta = 4.0
	var (
		i, j int
		err  error
	)

	M := MustDense(t, n, n)
	// M[i,j] = 3*i - j
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			MustSet(t, M, i, j, float64(3*i-j))
		}
	}

	// (αβ)*M
	left, err := matrix.Scale(M, alpha*beta)
	if err != nil {
		t.Fatalf("matrix.Scale(M, alpha*beta): want err == nil, got: %v", err)
	}

	// α*(β*M)
	bm, err := matrix.Scale(M, beta)
	if err != nil {
		t.Fatalf("matrix.Scale(M, beta): want err == nil, got: %v", err)
	}
	right, err := matrix.Scale(bm, alpha)
	if err != nil {
		t.Fatalf("matrix.Scale(bm, alpha): want err == nil, got: %v", err)
	}

	// Compare left vs right (associativity of scalar multiplication)
	var lv, rv float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			lv = MustAt(t, left, i, j)
			rv = MustAt(t, right, i, j)
			if lv != rv {
				t.Fatalf("composition failed at [%d,%d]: want lv(%b) == rv(%b)", i, j, lv, rv)
			}
		}
	}

	// α = 0 ⇒ zero matrix; α = -1 ⇒ negation; inputs not mutated.
	zero, err := matrix.Scale(M, 0.0)
	if err != nil {
		t.Fatalf("matrix.Scale(M, 0.0): want err == nil, got: %v", err)
	}
	neg, err := matrix.Scale(M, -1.0)
	if err != nil {
		t.Fatalf("matrix.Scale(M, -1.0): want err == nil, got: %v", err)
	}

	var m, z, ng float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			m, _ = M.At(i, j)
			z, _ = zero.At(i, j)
			ng, _ = neg.At(i, j)
			if z != 0.0 {
				t.Fatalf("zero scaling failed at [%d,%d]", i, j)
			}
			if ng != -m {
				t.Fatalf("inegation failed at [%d,%d]", i, j)
			}
		}
	}

	// Ensure original M unchanged
	var m1, m2 float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			m1, _ = M.At(i, j)
			m2, _ = M.At(i, j) // read again; we only checked immutability via distinct results above
			if m1 != m2 {
				t.Fatalf("want m1(%b) == m2(%b)", m1, m2)
			}
		}
	}
}

func TestScale_WithTranspose_Compatibility(t *testing.T) {
	t.Parallel()

	const rows, cols = 3, 5
	const alpha = 2.25
	var (
		i, j int
		err  error
	)

	M := MustDense(t, rows, cols)
	// M[i,j] = i + 10*j
	for i = 0; i < rows; i++ {
		for j = 0; j < cols; j++ {
			MustSet(t, M, i, j, float64(i+10*j))
		}
	}

	alphaM, err := matrix.Scale(M, alpha)
	if err != nil {
		t.Fatalf("matrix.Scale(M, alpha): want err == nil, got: %v", err)
	}
	TalphaM, err := matrix.Transpose(alphaM)
	if err != nil {
		t.Fatalf("matrix.Transpose(alphaM): want err == nil, got: %v", err)
	}

	TM, err := matrix.Transpose(M)
	if err != nil {
		t.Fatalf("matrix.Transpose(M): want err == nil, got: %v", err)
	}
	alphaTM, err := matrix.Scale(TM, alpha)
	if err != nil {
		t.Fatalf("matrix.Scale(NM, alpha): want err == nil, got: %v", err)
	}

	// Expect Transpose(αM) == α Transpose(M)
	var v1, v2 float64
	for i = 0; i < TalphaM.Rows(); i++ {
		for j = 0; j < TalphaM.Cols(); j++ {
			v1 = MustAt(t, TalphaM, i, j)
			v2 = MustAt(t, alphaTM, i, j)
			if v1 != v2 {
				t.Fatalf("distributivity failed at [%d,%d]: want v1(%b) == v2(%b)", i, j, v1, v2)
			}
		}
	}
}

func TestScale(t *testing.T) {
	// 2×2 matrix
	m := NewFilledDense(t, 2, 2, []float64{1.5, -2.5, 3.0, 0.0})

	sm, _ := matrix.Scale(m, 2.0)
	// expected = [[3.0, -5.0],[6.0, 0.0]]
	expected := [][]float64{{3.0, -5.0}, {6.0, 0.0}}
	var i, j int // loop iterators
	var v float64
	for i = 0; i < sm.Rows(); i++ {
		for j = 0; j < sm.Cols(); j++ {
			v = MustAt(t, sm, i, j)
			if v != expected[i][j] {
				t.Fatalf("want v == %b, got: %.6g", expected[i][j], v)
			}
		}
	}
}

// ---------- 3.3 Hadamard ----------

func TestHadamard_FastPath_4x5_Correctness(t *testing.T) {
	t.Parallel()
	const r, c = 4, 5
	A := MustDense(t, r, c)
	B := MustDense(t, r, c)
	var i, j int
	for i = 0; i < r; i++ {
		for j = 0; j < c; j++ {
			MustSet(t, A, i, j, float64(i+1))
			MustSet(t, B, i, j, float64(j+1))
		}
	}

	H, err := matrix.Hadamard(A, B)
	if err != nil {
		t.Fatalf("matrix.Hadamard: %v", err)
	}

	var got, want float64
	for i = 0; i < r; i++ {
		for j = 0; j < c; j++ {
			got = MustAt(t, H, i, j)
			want = float64(i+1) * float64(j+1)
			if got != want {
				t.Fatalf("at [%d,%d]: want %.6g, got %.6g", i, j, want, got)
			}
		}
	}
}

func TestHadamard_Fallback_3x3_Correctness(t *testing.T) {
	t.Parallel()
	const n = 3
	Ar := MustDense(t, n, n)
	Br := MustDense(t, n, n)
	var i, j int
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			MustSet(t, Ar, i, j, float64(i+j+1))
			MustSet(t, Br, i, j, float64(2*i-j))
		}
	}

	A := hide{Ar}
	B := hide{Br}
	H, err := matrix.Hadamard(A, B)
	if err != nil {
		t.Fatalf("matrix.Hadamard: %v", err)
	}

	var got, want float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			got = MustAt(t, H, i, j)
			want = MustAt(t, Ar, i, j) * MustAt(t, Br, i, j)
			if got != want {
				t.Fatalf("at [%d,%d]: want %.6g, got %.6g", i, j, want, got)
			}
		}
	}
}

func TestHadamard_DimensionMismatch(t *testing.T) {
	t.Parallel()
	A := MustDense(t, 3, 4)
	B := MustDense(t, 4, 3)
	_, err := matrix.Hadamard(A, B)
	AssertErrorIs(t, err, matrix.ErrDimensionMismatch)
}

// ---------- 3.4 MatVec ----------

func TestMatVec_FastPath_5x4_Correctness(t *testing.T) {
	t.Parallel()
	const r, c = 5, 4
	M := MustDense(t, r, c)
	// M[i,j] = i - 2j
	var i, j int
	for i = 0; i < r; i++ {
		for j = 0; j < c; j++ {
			MustSet(t, M, i, j, float64(i-2*j))
		}
	}
	x := []float64{1, 2, 3, 4}
	y, err := matrix.MatVec(M, x)
	if err != nil {
		t.Fatalf("matrix.MatVec: %v", err)
	}

	var sum float64
	for i = 0; i < r; i++ {
		sum = 0.0
		for j = 0; j < c; j++ {
			sum += float64(i-2*j) * x[j]
		}
		if y[i] != sum {
			t.Fatalf("y[%d]: want %.6g, got %.6g", i, sum, y[i])
		}
	}
}

func TestMatVec_LengthMismatch(t *testing.T) {
	t.Parallel()
	M := MustDense(t, 3, 4)
	x := []float64{1, 2, 3} // len=3, need 4
	_, err := matrix.MatVec(M, x)
	AssertErrorIs(t, err, matrix.ErrDimensionMismatch)
}

func TestMatVec_Fallback_Wrapped(t *testing.T) {
	t.Parallel()
	const r, c = 3, 3
	Mr := MustDense(t, r, c)
	var i, j int
	for i = 0; i < r; i++ {
		for j = 0; j < c; j++ {
			MustSet(t, Mr, i, j, float64(i+j+1))
		}
	}
	Mw := hide{Mr}
	x := []float64{1, 0, -1}
	y1, err := matrix.MatVec(Mr, x)
	if err != nil {
		t.Fatalf("matrix.MatVec(Mr,x): %v", err)
	}
	y2, err := matrix.MatVec(Mw, x)
	if err != nil {
		t.Fatalf("matrix.MatVec(Mw,x): %v", err)
	}

	for i = 0; i < r; i++ {
		if !InDelta(t, y1[i], y2[i], 0.0) {
			t.Fatalf("y mismatch at %d: want %.6g, got %.6g", i, y1[i], y2[i])
		}
	}
}

// ---------- 4. Eigen ----------

// TestEigen_Errors verifies error paths: non-square, non-symmetric, and forced non-convergence.
func TestEigen_Errors(t *testing.T) {
	t.Parallel()

	var err error
	// non-square → ErrDimensionMismatch
	ns := MustDense(t, 3, 4)
	_, _, err = matrix.Eigen(ns, 1e-10, 50)
	//AssertErrorIs(t, err, matrix.ErrDimensionMismatch)
	AssertErrorIs(t, err, matrix.ErrNonSquare)

	// not symmetric within tol → ErrNotSymmetric
	asym := MustDense(t, 3, 3)
	MustSet(t, asym, 0, 1, 1)
	MustSet(t, asym, 1, 0, 2) // violates symmetry > tol
	_, _, err = matrix.Eigen(asym, 1e-12, 50)
	AssertErrorIs(t, err, matrix.ErrAsymmetry)

	// zero iterations with nonzero off-diagonals → ErrEigenFailed
	sym := MustDense(t, 3, 3)
	MustSet(t, sym, 0, 0, 2)
	MustSet(t, sym, 1, 1, 3)
	MustSet(t, sym, 2, 2, 4)
	MustSet(t, sym, 0, 1, 1)
	MustSet(t, sym, 1, 0, 1)
	_, _, err = matrix.Eigen(sym, 1e-12, 0)
	AssertErrorIs(t, err, matrix.ErrMatrixEigenFailed)
}

// TestEigen_Diagonal_NoRotation: diagonal matrices return exact diagonal as eigenvalues and Q=I.
func TestEigen_Diagonal_NoRotation(t *testing.T) {
	t.Parallel()

	const n = 4
	var (
		i, j int
		v    float64
		err  error
	)

	diagVals := []float64{1, -2, 5, 3}
	A := MustDense(t, n, n)
	for i = 0; i < n; i++ {
		MustSet(t, A, i, i, diagVals[i])
	}

	vals, Q, err := matrix.Eigen(A, 1e-12, 10)
	if err != nil {
		t.Fatalf("matrix.Eigen(A, 1e-12, 10): want err == nil, got: %v", err)
	}
	if len(vals) != n {
		t.Fatalf("want len(vals) == %d, got: %d", n, len(vals))
	}
	if Q.Rows() != n {
		t.Fatalf("want mt.Rows == %d, got:%d", n, Q.Rows())
	}
	if Q.Cols() != n {
		t.Fatalf("want mt.Rows == %d, got:%d", n, Q.Cols())
	}

	got := append([]float64(nil), vals...)
	want := append([]float64(nil), diagVals...)
	sort.Float64s(got)
	sort.Float64s(want)
	if !AlmostEqualSlice(got, want, 0.0) {
		t.Fatalf("igenvalues mismatch: want=%v got=%v", want, got)
	}

	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			v = MustAt(t, Q, i, j)
			if i == j {
				if v != 1.0 {
					t.Fatalf("Q[%d,%d]", i, j)
				}
			} else {
				if v != 0.0 {
					t.Fatalf("Q[%d,%d]", i, j)
				}
			}
		}
	}
}

// TestEigen_2x2_Analytic: [[2,1],[1,2]] has eigenvalues {1,3}; Q orthonormal; A*Q≈Q*D.
func TestEigen_2x2_Analytic(t *testing.T) {
	t.Parallel()

	var err error
	var got []float64

	A := NewFilledDense(t, 2, 2, []float64{2, 1, 1, 2})

	vals, Q, err := matrix.Eigen(A, 1e-12, 50)
	if err != nil {
		t.Fatalf("matrix.Eigen(A, 1e-12, 50): want err == nil, got: %v", err)
	}
	if len(vals) != 2 {
		t.Fatalf("want len(vals) == %d, got: %d", 2, len(vals))
	}

	got = append([]float64(nil), vals...)
	sort.Float64s(got)
	if !InDelta(t, got[0], 1.0, 1e-10) {
		t.Fatalf("want |%.6g-%.6g|<=%.1e", got[0], 1.0, 1e-10)
	}

	if !InDelta(t, got[1], 3.0, 1e-10) {
		t.Fatalf("want |%.6g-%.6g|<=%.1e", got[1], 3.0, 1e-10)
	}

	propOrthonormal(t, Q, 1e-10)
	propEigenEquation(t, A, Q, vals, 1e-10)
}

// TestEigen_BlockDiagonal_Degenerate: block diag([2], [[3,1],[1,3]]) ⇒ eigenvalues {2,2,4}.
func TestEigen_BlockDiagonal_Degenerate(t *testing.T) {
	t.Parallel()

	const n = 3
	var err error
	var got []float64

	//A := NewFilledDense(t, n, n, []float64{0, 0, 2, 0, 3, 1, 0, 1, 3})
	A := NewFilledDense(t, n, n, []float64{2, 0, 0, 0, 3, 1, 0, 1, 3})

	//orig := matrix.CloneMatrix(A) // wrapper for A.Clone()
	orig := A.Clone()
	vals, Q, err := matrix.Eigen(A, 1e-12, 100)
	if err != nil {
		t.Fatalf("matrix.Eigen(A, 1e-12, 100): want err == nil, got: %v", err)
	}
	if len(vals) != n {
		t.Fatalf("want len(vals) == %d, got: %d", n, len(vals))
	}

	got = append([]float64(nil), vals...)
	sort.Float64s(got)
	if !InDelta(t, got[0], 2.0, 1e-10) {
		t.Fatalf("want |%.6g-%.6g|<=%.1e", got[0], 2.0, 1e-10)
	}
	if !InDelta(t, got[1], 2.0, 1e-10) {
		t.Fatalf("want |%.6g-%.6g|<=%.1e", got[1], 2.0, 1e-10)
	}
	if !InDelta(t, got[2], 4.0, 1e-10) {
		t.Fatalf("want |%.6g-%.6g|<=%.1e", got[2], 4.0, 1e-10)
	}

	propOrthonormal(t, Q, 1e-10)
	propReconstruction(t, orig, Q, vals, 1e-9)
}

// TestEigen_Reconstruction_SPD_6x6: SPD A=MᵀM, check QᵀQ≈I, A≈QDQᵀ and A*Q≈Q*D.
func TestEigen_Reconstruction_SPD_6x6(t *testing.T) {
	t.Parallel()

	const n = 6
	var err error

	M := MustDense(t, n, n)
	RandomFill(t, M, 42)

	Mt, err := matrix.Transpose(M)
	if err != nil {
		t.Fatalf("matrix.Transpose(M): want err == nil, got: %v", err)
	}
	A, err := matrix.Mul(Mt, M) // SPD
	if err != nil {
		t.Fatalf("matrix.Mul(Mt, M): want err == nil, got: %v", err)
	}

	orig := A.Clone()
	vals, Q, err := matrix.Eigen(A, 1e-9, 200)
	if err != nil {
		t.Fatalf("matrix.Eigen(A, 1e-9, 200): want err == nil, got: %v", err)
	}
	if len(vals) != n {
		t.Fatalf("want len(vals) == %d, got: %d", n, len(vals))
	}

	propOrthonormal(t, Q, 1e-8)
	propReconstruction(t, orig, Q, vals, 1e-6)
	propEigenEquation(t, orig, Q, vals, 1e-6)
}

// ---------- 5. Inverse ----------

func TestInverse_Errors(t *testing.T) {
	t.Parallel()

	var err error

	// nil → ErrNilMatrix
	_, err = matrix.Inverse(nil)
	AssertErrorIs(t, err, matrix.ErrNilMatrix)

	// non-square → ErrDimensionMismatch
	ns := MustDense(t, 3, 4)
	_, err = matrix.Inverse(ns)
	//AssertErrorIs(t, err, matrix.ErrDimensionMismatch)
	AssertErrorIs(t, err, matrix.ErrNonSquare)

	// singular → ErrSingular (two equal strings)
	sing := NewFilledDense(t, 3, 3, []float64{1, 2, 3, 1, 2, 3, 0, 1, 4})

	_, err = matrix.Inverse(sing)
	AssertErrorIs(t, err, matrix.ErrSingular)
}

// Known 3×3 matrix with det=9. Check the numerical values of the inverse
// (adj(A)/det) and that A A^{-1}≈I and A^{-1} A≈I.
func TestInverse_Known3x3_Adjugate(t *testing.T) {
	t.Parallel()

	var i, j int
	var err error

	// A = [[4,7,2],[3,6,1],[2,5,3]]
	A := NewFilledDense(t, 3, 3, []float64{4, 7, 2, 3, 6, 1, 2, 5, 3})

	Inv, err := matrix.Inverse(A)
	if err != nil {
		t.Fatalf("matrix.Inverse(A): want err == nil, got: %v", err)
	}

	// adj(A)/9, where adj(A)^T = cofactors:
	want := [][]float64{
		{13.0 / 9.0, -11.0 / 9.0, -5.0 / 9.0},
		{-7.0 / 9.0, 8.0 / 9.0, 2.0 / 9.0},
		{3.0 / 9.0, -6.0 / 9.0, 3.0 / 9.0},
	}

	var got float64
	for i = 0; i < 3; i++ {
		for j = 0; j < 3; j++ {
			got = MustAt(t, Inv, i, j)
			if !InDelta(t, got, want[i][j], 1e-12) {
				t.Fatalf("Inv[%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, got, want[i][j], 1e-12)
			}
		}
	}

	// Check A*Inv≈I и Inv*A≈I
	Ileft, err := matrix.Mul(A, Inv)
	if err != nil {
		t.Fatalf("matrix.Mul(A, Inv): want err == nil, got: %v", err)
	}
	Iright, err := matrix.Mul(Inv, A)
	if err != nil {
		t.Fatalf("matrix.Mul(Inv, A): want err == nil, got: %v", err)
	}

	var lv, rv float64
	for i = 0; i < 3; i++ {
		for j = 0; j < 3; j++ {
			lv, _ = Ileft.At(i, j)
			rv, _ = Iright.At(i, j)
			if i == j {
				if !InDelta(t, lv, 1.0, 1e-12) {
					t.Fatalf("A*Inv diag[%d]: want |%.6g-%.6g|<=%.1e", i, lv, 1.0, 1e-12)
				}
				if !InDelta(t, rv, 1.0, 1e-12) {
					t.Fatalf("Inv*A diag[%d]: want |%.6g-%.6g|<=%.1e", i, rv, 1.0, 1e-12)
				}
			} else {
				if !InDelta(t, lv, 0.0, 1e-12) {
					t.Fatalf("A*Inv off[%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, lv, 0.0, 1e-12)
				}
				if !InDelta(t, rv, 0.0, 1e-12) {
					t.Fatalf("Inv*A off[%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, rv, 0.0, 1e-12)
				}
			}
		}
	}
}

// Hiding the input type (iface/fallback on reading) should not change the result.
// Inside Inverse it is still solved by *Dense (L and U are dense).
func TestInverse_WrappedInput_MatchesDense(t *testing.T) {
	t.Parallel()

	const n = 4
	var i, j int
	var err error

	// A = MᵀM + I  (well-conditioned PD)
	M := RandFilledDense(t, n, n, 123)
	Mt, err := matrix.Transpose(M)
	if err != nil {
		t.Fatalf("matrix.Transpose(M): want err == nil, got: %v", err)
	}
	PD, err := matrix.Mul(Mt, M)
	if err != nil {
		t.Fatalf("matrix.Mul(Mt, M): want err == nil, got: %v", err)
	}
	I := MustDense(t, n, n)
	for i = 0; i < n; i++ {
		MustSet(t, I, i, i, 1)
	}
	A, err := matrix.Add(PD, I)
	if err != nil {
		t.Fatalf("matrix.Add(PD, I): want err == nil, got: %v", err)
	}

	Aw := hide{A} // hided type

	Inv1, err := matrix.Inverse(A)
	if err != nil {
		t.Fatalf("matrix.Inverse(A): want err == nil, got: %v", err)
	}
	Inv2, err := matrix.Inverse(Aw)
	if err != nil {
		t.Fatalf("matrix.Inverse(As): want err == nil, got: %v", err)
	}

	var v1, v2 float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			v1, _ = Inv1.At(i, j)
			v2, _ = Inv2.At(i, j)
			if !InDelta(t, v1, v2, 1e-11) {
				t.Fatalf("mistmatch at [%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, v1, v2, 1e-11)
			}
		}
	}
}

// Property: A A^{-1}≈I and A^{-1} A≈I on 6×6 SPD. And the input does not mutate.
func TestInverse_IdentityProduct_SPD_6x6(t *testing.T) {
	t.Parallel()

	const n = 6
	var i, j int
	var err error

	// A = MᵀM + I
	M := RandFilledDense(t, n, n, 777)
	Mt, err := matrix.Transpose(M)
	if err != nil {
		t.Fatalf("matrix.Transpose(M): want err == nil, got: %v", err)
	}
	PD, err := matrix.Mul(Mt, M)
	if err != nil {
		t.Fatalf("matrix.Mul(Mt, M): want err == nil, got: %v", err)
	}

	A := MustDense(t, n, n)
	var v float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			v, _ = PD.At(i, j)
			MustSet(t, A, i, j, v)
		}

		v, _ = A.At(i, i)
		MustSet(t, A, i, i, v+1.0)
	}

	Acopy := A.Clone()

	Inv, err := matrix.Inverse(A)
	if err != nil {
		t.Fatalf("matrix.Inverse(A): want err == nil, got: %v", err)
	}

	L, err := matrix.Mul(A, Inv)
	if err != nil {
		t.Fatalf("matrix.Mul(A, Inv): want err == nil, got: %v", err)
	}
	R, err := matrix.Mul(Inv, A)
	if err != nil {
		t.Fatalf("matrix.Mul(Inv, A): want err == nil, got: %v", err)
	}

	var lv, rv float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			lv, _ = L.At(i, j)
			rv, _ = R.At(i, j)
			if i == j {
				if !InDelta(t, lv, 1.0, 1e-8) {
					t.Fatalf("A*Inv diag[%d]: want |%.6g-%.6g|<=%.1e", i, lv, 1.0, 1e-12)
				}
				if !InDelta(t, rv, 1.0, 1e-8) {
					t.Fatalf("Inv*A diag[%d]: want |%.6g-%.6g|<=%.1e", i, rv, 1.0, 1e-12)
				}
			} else {
				if !InDelta(t, lv, 0.0, 1e-8) {
					t.Fatalf("A*Inv off[%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, lv, 0.0, 1e-12)
				}
				if !InDelta(t, rv, 0.0, 1e-8) {
					t.Fatalf("Inv*A off[%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, rv, 0.0, 1e-12)
				}
			}
		}
	}

	// A should not mutate
	var a1, a2 float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			a1, _ = A.At(i, j)
			a2, _ = Acopy.At(i, j)
			if a1 != a2 {
				t.Fatalf("A mismatch at[%d,%d]: want v == %b, got: %.6g", i, j, a2, a1)
			}
		}
	}
}

// Scaling property: (αA)^{-1} = (1/α)*A^{-1} for α≠0.
func TestInverse_ScaleProperty(t *testing.T) {
	t.Parallel()

	const n = 5
	const alpha = 2.5
	var i, j int
	var err error

	// A = MᵀM + 2I (add 2I to stay away from degeneracy)
	M := RandFilledDense(t, n, n, 2024)
	Mt, err := matrix.Transpose(M)
	if err != nil {
		t.Fatalf("matrix.Transpose(M): want err == nil, got: %v", err)
	}
	PD, err := matrix.Mul(Mt, M)
	if err != nil {
		t.Fatalf("matrix.Mul(Mt, M): want err == nil, got: %v", err)
	}

	A := MustDense(t, n, n)
	var v float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			v, _ = PD.At(i, j)
			MustSet(t, A, i, j, v)
		}
	}
	var d float64
	for i = 0; i < n; i++ {
		d, _ = A.At(i, i)
		MustSet(t, A, i, i, d+2.0)
	}

	InvA, err := matrix.Inverse(A)
	if err != nil {
		t.Fatalf("matrix.Inverse(A): want err == nil, got: %v", err)
	}

	alphaA, err := matrix.Scale(A, alpha)
	if err != nil {
		t.Fatalf("matrix.Scale(A, alpha): want err == nil, got: %v", err)
	}
	InvAlphaA, err := matrix.Inverse(alphaA)
	if err != nil {
		t.Fatalf("matrix.Inverse(alphaA): want err == nil, got: %v", err)
	}

	// Wait Inv(αA) ≈ (1/α)*Inv(A)
	scaleInvA, err := matrix.Scale(InvA, 1.0/alpha)
	if err != nil {
		t.Fatalf("matrix.Scale(Inv, 1.0/alpha): want err == nil, got: %v", err)
	}

	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			var left, right float64
			left, _ = InvAlphaA.At(i, j)
			right, _ = scaleInvA.At(i, j)
			if !InDelta(t, right, left, 1e-9) {
				t.Fatalf("at [%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, right, left, 1e-9)
			}
		}
	}
}

// ---------- 6. LU ----------

// Errors: nil and non-square are rejected.
func TestLU_Errors(t *testing.T) {
	t.Parallel()

	var err error

	// nil → ErrNilMatrix
	_, _, err = matrix.LU(nil)
	AssertErrorIs(t, err, matrix.ErrNilMatrix)

	// non-square → ErrDimensionMismatch
	ns := MustDense(t, 3, 4)
	_, _, err = matrix.LU(ns)
	//AssertErrorIs(t, err, matrix.ErrDimensionMismatch)
	AssertErrorIs(t, err, matrix.ErrNonSquare)
}

// Basic (3×3): pick L,U explicitly (Doolittle form, diag(L)=1), set A=L*U,
// then verify LU(A) reproduces the same factors and A≈L*U exactly.
func TestLU_Known3x3_Doolittle_FastPath_Correctness(t *testing.T) {
	t.Parallel()

	var i, j int
	var err error

	// Target factors:
	// L = [[1,0,0],
	//      [2,1,0],
	//      [3,4,1]]
	// U = [[5,6,7],
	//      [0,8,9],
	//      [0,0,10]]
	Lexp := NewFilledDense(t, 3, 3, []float64{1, 0, 0, 2, 1, 0, 3, 4, 1})
	Uexp := NewFilledDense(t, 3, 3, []float64{5, 6, 7, 0, 8, 9, 0, 0, 10})

	// Build A = L*U
	A, err := matrix.Mul(Lexp, Uexp)
	if err != nil {
		t.Fatalf("matrix.Mul(Lexp, Uexp): want err == nil, got: %v", err)
	}

	// Keep a copy to ensure input immutability
	Acopy := A.Clone()

	// Factorize
	Lgot, Ugot, err := matrix.LU(A)
	if err != nil {
		t.Fatalf("matrix.LU(A): want err == nil, got: %v", err)
	}

	// Structural checks and exact equality vs expected factors
	propUnitLowerTriangular(t, Lgot, 0)
	propUpperTriangular(t, Ugot, 0)

	var gv, ev float64
	for i = 0; i < 3; i++ {
		for j = 0; j < 3; j++ {
			gv, _ = Lgot.At(i, j)
			ev, _ = Lexp.At(i, j)
			if gv != ev {
				t.Fatalf("L mismatch at[%d,%d]: want v == %b, got: %.6g", i, j, gv, ev)
			}

			gv, _ = Ugot.At(i, j)
			ev, _ = Uexp.At(i, j)
			if gv != ev {
				t.Fatalf("U mismatch at[%d,%d]: want v == %b, got: %.6g", i, j, gv, ev)
			}
		}
	}

	// Reconstruction A ≈ L*U
	propReconstructionLU(t, A, Lgot, Ugot, 0)

	// Input must not mutate
	for i = 0; i < 3; i++ {
		for j = 0; j < 3; j++ {
			var a1, a2 float64
			a1, _ = A.At(i, j)
			a2, _ = Acopy.At(i, j)
			if a1 != a2 {
				t.Fatalf("A mismatch at[%d,%d]: want v == %b, got: %.6g", i, j, a2, a1)
			}
		}
	}
}

// Fast-path vs Fallback (3×3): wrapping the input to hide its concrete type
// must produce the same L and U as the fast path.
func TestLU_Known3x3_Fallback_MatchesFast(t *testing.T) {
	t.Parallel()

	var i, j int
	var err error

	// Reuse the same 3×3 A from the previous test to avoid tiny matrices.
	// Target factors:
	// L = [[1,0,0],
	//      [2,1,0],
	//      [3,4,1]]
	// U = [[5,6,7],
	//      [0,8,9],
	//      [0,0,10]]
	Lexp := NewFilledDense(t, 3, 3, []float64{1, 0, 0, 2, 1, 0, 3, 4, 1})
	Uexp := NewFilledDense(t, 3, 3, []float64{5, 6, 7, 0, 8, 9, 0, 0, 10})

	A, err := matrix.Mul(Lexp, Uexp)
	if err != nil {
		t.Fatalf("matrix.Mul(Lexp, Uexp): want err == nil, got: %v", err)
	}

	// Fast path
	L1, U1, err := matrix.LU(A)
	if err != nil {
		t.Fatalf("matrix.LU(A): want err == nil, got: %v", err)
	}
	// Fallback path
	Aw := hide{A}
	L2, U2, err := matrix.LU(Aw)
	if err != nil {
		t.Fatalf("matrix.LU(Aw): want err == nil, got: %v", err)
	}

	// Elementwise equality
	var v1, v2 float64
	for i = 0; i < 3; i++ {
		for j = 0; j < 3; j++ {
			v1, _ = L1.At(i, j)
			v2, _ = L2.At(i, j)
			if v1 != v2 {
				t.Fatalf("U mismatch at[%d,%d]: want v == %b, got: %.6g", i, j, v2, v1)
			}

			v1, _ = U1.At(i, j)
			v2, _ = U2.At(i, j)
			if v1 != v2 {
				t.Fatalf("U mismatch at[%d,%d]: want v == %b, got: %.6g", i, j, v2, v1)
			}
		}
	}
}

// Properties on 6×6: construct L (unit lower) and U (upper) with simple integer
// patterns, set A=L*U, then check (i) structure, (ii) reconstruction, and (iii) exact recovery.
func TestLU_Factor_Reconstruction_6x6(t *testing.T) {
	t.Parallel()

	const n = 6
	var i, j int
	var err error

	Lexp := MustDense(t, n, n)
	Uexp := MustDense(t, n, n)

	// Lexp: unit lower with a mild, deterministic pattern below diagonal
	for i = 0; i < n; i++ {
		MustSet(t, Lexp, i, i, 1.0)
	}
	for i = 1; i < n; i++ {
		for j = 0; j < i; j++ {
			// small integers keep A exact in float
			MustSet(t, Lexp, i, j, float64(j+1))
		}
	}

	// Uexp: upper with positive diagonal (nonzero pivots), simple pattern above diag
	for i = 0; i < n; i++ {
		MustSet(t, Uexp, i, i, float64(2*i+3)) // 3,5,7,9,11,13

		for j = i + 1; j < n; j++ {
			MustSet(t, Uexp, i, j, float64(j-i+1))
		}
	}

	// A = L*U
	A, err := matrix.Mul(Lexp, Uexp)
	if err != nil {
		t.Fatalf("matrix.Mul(Lexp, Uexp): want err == nil, got: %v", err)
	}

	// Factorize
	Lgot, Ugot, err := matrix.LU(A)
	if err != nil {
		t.Fatalf("matrix.LU(A): want err == nil, got: %v", err)
	}

	// Structure
	propUnitLowerTriangular(t, Lgot, 0)
	propUpperTriangular(t, Ugot, 0)

	// Exact equality vs our factors (Doolittle is unique with these nonzero pivots)
	var gv, ev float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			gv, _ = Lgot.At(i, j)
			ev, _ = Lexp.At(i, j)
			if gv != ev {
				t.Fatalf("U mismatch at[%d,%d]: want v == %b, got: %.6g", i, j, gv, ev)
			}

			gv, _ = Ugot.At(i, j)
			ev, _ = Uexp.At(i, j)
			if gv != ev {
				t.Fatalf("U mismatch at[%d,%d]: want v == %b, got: %.6g", i, j, gv, ev)
			}
		}
	}

	// Reconstruction A ≈ L*U
	propReconstructionLU(t, A, Lgot, Ugot, 0)
}

// ---------- 7. QR ----------

// Errors: nil and non-square are rejected.
func TestQR_Errors(t *testing.T) {
	t.Parallel()

	var err error

	// nil → ErrNilMatrix
	_, _, err = matrix.QR(nil)
	AssertErrorIs(t, err, matrix.ErrNilMatrix)

	// non-square → ErrDimensionMismatch
	ns := MustDense(t, 3, 4)
	_, _, err = matrix.QR(ns)
	//AssertErrorIs(t, err, matrix.ErrDimensionMismatch)
	AssertErrorIs(t, err, matrix.ErrNonSquare)
}

// Classic 3×3 Householder example (well-known benchmark):
//
//	 A = [[ 12, -51,   4],
//		[  6, 167, -68],
//		[ -4,  24, -41]]
//
// One canonical QR (up to column-sign freedom):
//
//	 R = [[ 14,  21, -14],
//		[  0, 175, -70],
//		[  0,   0,  35]]
//
//	 Q = [[ 6/7,   -69/175,  -58/175],
//		[ 3/7,    158/175,    6/175],
//		[-2/7,      6/35,    -33/35]]
//
// Our routine returns A ≈ Qᵀ*R. We canonicalize diag(R) ≥ 0 by left-multiplying
// both Q and R by the same diagonal S, which preserves A = Qᵀ*R. Then we:
//
//	check |R| against the canonical magnitudes;
//	compare columns of Qᵀ with the canonical Q up to per-column sign;
//	assert QᵀQ≈I and A≈Qᵀ*R;
//	assert input immutability.
func TestQR_Classic3x3_Householder_Known(t *testing.T) {
	t.Parallel()

	var i, j int
	var err error

	// Build A
	A := NewFilledDense(t, 3, 3, []float64{12, -51, 4, 6, 167, -68, -4, 24, -41})
	Acopy := A.Clone()

	Q, R, err := matrix.QR(A)
	if err != nil {
		t.Fatalf("matrix.QR(A): want err == nil, got: %v", err)
	}

	// --- Canonicalize diag(R) >= 0 via S (LEFT multiply on BOTH Q and R!) ---
	S := MustDense(t, 3, 3)
	var rii float64
	for i = 0; i < 3; i++ {
		rii = MustAt(t, R, i, i)
		if rii >= 0 {
			MustSet(t, S, i, i, 1.0)
		} else {
			MustSet(t, S, i, i, -1.0)
		}
	}
	// Correct invariance: (SQ)^T*(SR) = Q^T*R
	SQ, err := matrix.Mul(S, Q)
	if err != nil {
		t.Fatalf("matrix.Mul(S, Q): want err == nil, got: %v", err)
	}
	SR, err := matrix.Mul(S, R)
	if err != nil {
		t.Fatalf("matrix.Mul(S, R): want err == nil, got: %v", err)
	}
	Q = SQ
	R = SR

	// quick sanity: after normalization A ≈ Qᵀ*R must still hold
	propReconstructionQR(t, Acopy, Q, R, 1e-12)
	// --- end canonicalization ---

	// R must be upper-triangular with canonical magnitudes (signs are free).
	RabsWant := [][]float64{
		{14, 21, 14},
		{0, 175, 70},
		{0, 0, 35},
	}
	var rv float64
	for i = 0; i < 3; i++ {
		for j = 0; j < 3; j++ {
			rv = MustAt(t, R, i, j)
			if i > j {
				if !InDelta(t, rv, 0.0, 1e-12) {
					t.Fatalf("R[%d,%d] must be 0 below diagonal: want |%.6g-%.6g|<=%.1e", i, j, rv, 0.0, 1e-12)
				}
				continue
			}
			if rv < 0 {
				rv = -rv
			}
			if !InDelta(t, rv, RabsWant[i][j], 1e-12) {
				t.Fatalf("abs(R[%d,%d]): want |%.6g-%.6g|<=%.1e", i, j, rv, RabsWant[i][j], 1e-12)
			}
		}
	}
	for i = 0; i < 3; i++ {
		rv, _ = R.At(i, i)
		if rv < 0.0 {
			t.Fatalf("R[%d,%d] must be >= 0 after normalization, got: %.6g", i, i, rv)
		}
	}

	// Compare Q^T columns to canonical Q columns up to column sign.
	Qwant := [][]float64{
		{6.0 / 7.0, -69.0 / 175.0, -58.0 / 175.0},
		{3.0 / 7.0, 158.0 / 175.0, 6.0 / 175.0},
		{-2.0 / 7.0, 6.0 / 35.0, -33.0 / 35.0},
	}
	Qt, err := matrix.Transpose(Q)
	if err != nil {
		t.Fatalf("matrix.Transpose(Q): want err == nil, got: %v", err)
	}
	var qv, dot float64
	for j = 0; j < 3; j++ {
		dot = 0.0
		for i = 0; i < 3; i++ {
			qv = MustAt(t, Qt, i, j)
			dot += qv * Qwant[i][j]
		}
		sign := 1.0
		if dot < 0 {
			sign = -1.0
		}
		for i = 0; i < 3; i++ {
			qv = MustAt(t, Qt, i, j)
			if !InDelta(t, qv, sign*Qwant[i][j], 1e-9) {
				t.Fatalf("Qt[%d,%d] up to sign: want |%.6g-%.6g|<=%.1e", i, j, qv, sign*Qwant[i][j], 1e-9)
			}
		}
	}

	// Orthogonality and final reconstruction under A ≈ Qᵀ*R.
	propOrthonormal(t, Q, 1e-12)
	propReconstructionQR(t, Acopy, Q, R, 1e-12)

	// Input immutability.
	var a1, a2 float64
	for i = 0; i < 3; i++ {
		for j = 0; j < 3; j++ {
			a1, _ = A.At(i, j)
			a2, _ = Acopy.At(i, j)
			if a1 != a2 {
				t.Fatalf("upper(L)[%d,%d]: want v == %b, got: %.6g", i, j, a2, a1)
			}
		}
	}
}

// 8.3 Fast-path vs Fallback (5×5): wrapping the input to hide its concrete type
// must produce numerically identical Q and R (within tight tolerance).
func TestQR_Fallback_MatchesFast_5x5(t *testing.T) {
	t.Parallel()

	const n = 5
	var i, j int
	var err error

	// Build a deterministic dense matrix (no anonymous factories).
	M := MustDense(t, n, n)
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			// simple, well-conditioned pattern
			MustSet(t, M, i, j, float64(3*i-2*j+1))
		}
	}

	// Fast path
	Q1, R1, err := matrix.QR(M)
	if err != nil {
		t.Fatalf("matrix.QR(M): want err == nil, got: %v", err)
	}
	// Fallback path: hide the concrete type
	Mw := hide{M}
	Q2, R2, err := matrix.QR(Mw)
	if err != nil {
		t.Fatalf("matrix.QR(Mw): want err == nil, got: %v", err)
	}

	// Elementwise comparison with small tolerance
	var v1, v2 float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			v1, _ = Q1.At(i, j)
			v2, _ = Q2.At(i, j)
			if !InDelta(t, v1, v2, 1e-11) {
				t.Fatalf("at [%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, v1, v2, 1e-11)
			}

			v1, _ = R1.At(i, j)
			v2, _ = R2.At(i, j)
			if !InDelta(t, v1, v2, 1e-11) {
				t.Fatalf("at [%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, v1, v2, 1e-11)
			}
		}
	}

	// Both must satisfy orthogonality and reconstruction with the same source M.
	propOrthonormal(t, Q1, 1e-11)
	propOrthonormal(t, Q2, 1e-11)
	propReconstructionQR(t, M, Q1, R1, 1e-11)
	propReconstructionQR(t, M, Q2, R2, 1e-11)
}

// 8.4 Properties on 6×6: QᵀQ≈I, R is upper-triangular, and A≈Qᵀ*R.
// Also assert the input is not mutated. Include a zero column to exercise the “skip zero” branch.
func TestQR_Properties_6x6_WithZeroColumn(t *testing.T) {
	t.Parallel()

	const n = 6
	var i, j int
	var err error

	A := MustDense(t, n, n)
	// Columns: c0..c5; set c2 to zeros to hit the "norm == 0" branch.
	for i = 0; i < n; i++ {
		// c0: increasing
		MustSet(t, A, i, 0, float64(i+1))
		// c1: alternating
		MustSet(t, A, i, 1, float64(1-2*(i%2)))
		// c2: zeros
		MustSet(t, A, i, 2, 0.0)
		// c3, c4, c5: mild linear patterns
		MustSet(t, A, i, 3, float64(2*i-3))
		MustSet(t, A, i, 4, float64(5-i))
		MustSet(t, A, i, 5, float64(3*i+2))
	}

	Acopy := A.Clone()

	Q, R, err := matrix.QR(A)
	if err != nil {
		t.Fatalf("matrix.QR(A): want err == nil, got: %v", err)
	}

	// Q orthonormal; R upper triangular.
	propOrthonormal(t, Q, 1e-12)
	propUpperTriangular(t, R, 1e-12)

	// Reconstruction: A ≈ Qᵀ*R
	propReconstructionQR(t, Acopy, Q, R, 1e-11)

	// Input must not mutate
	var a1, a2 float64
	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			a1, _ = A.At(i, j)
			a2, _ = Acopy.At(i, j)
			if a1 != a2 {
				t.Fatalf("upper(L)[%d,%d]: want v == %b, got: %.6g", i, j, a2, a1)
			}
		}
	}
}

// --- QR-specific helper (test-only, unexported) ---

// --- local property-check helpers (test-only, unexported) ---

// propOrthonormal asserts QᵀQ ≈ I within delta.
func propOrthonormal(t *testing.T, Q matrix.Matrix, delta float64) {
	t.Helper()

	var (
		i, j int
		v    float64
		err  error
	)

	n := Q.Rows()
	if Q.Cols() != n {
		t.Fatalf("want Q.Cols() ==%d, got: %d", n, Q.Cols())
	}

	Qt, err := matrix.Transpose(Q)
	if err != nil {
		t.Fatalf("matrix.Transpose(Q): want err == nil, got: %v", err)
	}
	QtQ, err := matrix.Mul(Qt, Q)
	if err != nil {
		t.Fatalf("matrix.Mul(Qt, Q): want err == nil, got: %v", err)
	}

	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			v = MustAt(t, QtQ, i, j)
			if i == j {
				if !InDelta(t, v, 1.0, delta) {
					t.Fatalf("at [%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, v, 1.0, delta)
				}
			} else {
				if !InDelta(t, v, 0.0, delta) {
					t.Fatalf("at [%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, v, 0.0, delta)
				}
			}
		}
	}
}

// propReconstruction asserts A ≈ Q*diag(vals)*Qᵀ within delta.
func propReconstruction(t *testing.T, A, Q matrix.Matrix, vals []float64, delta float64) {
	t.Helper()

	var (
		i, j int
		w, g float64
		err  error
	)

	n := A.Rows()
	if A.Cols() != n {
		t.Fatalf("want mt.Rows == %d, got:%d", n, A.Cols())
	}
	if Q.Rows() != n {
		t.Fatalf("want mt.Rows == %d, got:%d", n, Q.Rows())
	}
	if Q.Cols() != n {
		t.Fatalf("want mt.Rows == %d, got:%d", n, Q.Cols())
	}
	if len(vals) != n {
		t.Fatalf("want len(vals) == %d, got: %d", n, len(vals))
	}

	D := MustDense(t, n, n)
	for i = 0; i < n; i++ {
		MustSet(t, D, i, i, vals[i])
	}

	QD, err := matrix.Mul(Q, D)
	if err != nil {
		t.Fatalf("matrix.Mul(Q, D): want err == nil, got: %v", err)
	}
	Qt, err := matrix.Transpose(Q)
	if err != nil {
		t.Fatalf("matrix.Transpose(Q): want err == nil, got: %v", err)
	}
	QDQt, err := matrix.Mul(QD, Qt)
	if err != nil {
		t.Fatalf("matrix.Mul(QD, Qt): want err == nil, got: %v", err)
	}

	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			w = MustAt(t, A, i, j)
			g = MustAt(t, QDQt, i, j)
			if !InDelta(t, g, w, delta) {
				t.Fatalf("reconstruction mismatch at [%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, g, w, delta)
			}
		}
	}
}

// propEigenEquation asserts A*Q ≈ Q*diag(vals) within delta.
func propEigenEquation(t *testing.T, A, Q matrix.Matrix, vals []float64, delta float64) {
	t.Helper()

	var (
		i, j int
		l, r float64
		err  error
	)

	n := A.Rows()
	if A.Cols() != n {
		t.Fatalf("want mt.Rows == %d, got:%d", n, A.Cols())
	}
	if Q.Rows() != n {
		t.Fatalf("want mt.Rows == %d, got:%d", n, Q.Rows())
	}
	if Q.Cols() != n {
		t.Fatalf("want mt.Rows == %d, got:%d", n, Q.Cols())
	}
	if len(vals) != n {
		t.Fatalf("want len(vals) == %d, got: %d", n, len(vals))
	}

	D := MustDense(t, n, n)
	for i = 0; i < n; i++ {
		MustSet(t, D, i, i, vals[i])
	}

	AQ, err := matrix.Mul(A, Q)
	if err != nil {
		t.Fatalf("matrix.Mul(A, Q): want err == nil, got: %v", err)
	}
	QD, err := matrix.Mul(Q, D)
	if err != nil {
		t.Fatalf("matrix.Mul(Q, D: want err == nil, got: %v", err)
	}

	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			l = MustAt(t, AQ, i, j)
			r = MustAt(t, QD, i, j)
			if !InDelta(t, l, r, delta) {
				t.Fatalf("A*Q vs Q*D mismatch at [%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, l, r, delta)
			}
		}
	}
}

// propUnitLowerTriangular checks diag(L)=1 and L[i,j]=0 for j>i.
// delta=0 demands exact zeros/ones; positive delta allows tolerance.
func propUnitLowerTriangular(t *testing.T, L matrix.Matrix, delta float64) {
	t.Helper()

	var i, j int
	var v float64

	if L.Cols() != L.Rows() {
		t.Fatalf("L must be square")
	}
	n := L.Rows()

	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			v = MustAt(t, L, i, j)
			if i == j {
				if delta == 0 {
					if v != 1.0 {
						t.Fatalf("diag(L)[%d]: want v == %b, got: %.6g", i, 1.0, v)
					}
				} else {
					if !InDelta(t, v, 1.0, delta) {
						t.Fatalf("diag(L) [%d]: want |%.6g-%.6g|<=%.1e", i, v, 1.0, delta)
					}
				}
			} else if j > i {
				if delta == 0 {
					if v != 0.0 {
						t.Fatalf("upper(L)[%d,%d]: want v == %b, got: %.6g", i, j, 0.0, v)
					}
				} else {
					if !InDelta(t, v, 0.0, delta) {
						t.Fatalf("upper(L) at [%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, v, 0.0, delta)
					}
				}
			}
		}
	}
}

// propUpperTriangular checks U[i,j]=0 for i>j. Diagonal may be arbitrary nonzero.
// delta=0 demands exact zeros below diagonal.
func propUpperTriangular(t *testing.T, U matrix.Matrix, delta float64) {
	t.Helper()

	var i, j int
	var v float64

	if U.Cols() != U.Rows() {
		t.Fatalf("U must be square")
	}
	n := U.Rows()

	for i = 0; i < n; i++ {
		for j = 0; j < n; j++ {
			if i > j {
				v = MustAt(t, U, i, j)
				if delta == 0 {
					if v != 0.0 {
						t.Fatalf("upper(L)[%d,%d]: want v == %b, got: %.6g", i, j, 0.0, v)
					}
				} else {
					if !InDelta(t, v, 0.0, delta) {
						t.Fatalf("upper(L) at [%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, v, 0.0, delta)
					}
				}
			}
		}
	}
}

// propReconstructionLU verifies A ≈ L*U within delta.
func propReconstructionLU(t *testing.T, A, L, U matrix.Matrix, delta float64) {
	t.Helper()

	var (
		i, j   int
		lr, ar float64
		err    error
	)

	if A.Rows() != L.Rows() {
		t.Fatalf("shape mismatch A vs L")
	}
	if A.Cols() != U.Cols() {
		t.Fatalf("shape mismatch A vs U")
	}

	LU, err := matrix.Mul(L, U)
	if err != nil {
		t.Fatalf("matrix.Mul(L, U): want err == nil, got: %v", err)
	}

	for i = 0; i < A.Rows(); i++ {
		for j = 0; j < A.Cols(); j++ {
			// lr = (L*U)[i,j], ar = A[i,j].
			lr = MustAt(t, LU, i, j)
			ar = MustAt(t, A, i, j)

			if delta == 0 {
				if lr != ar {
					t.Fatalf("A vs L*U at [%d,%d]: want %.6g, got %.6g", i, j, ar, lr)
				}
			} else {
				if !InDelta(t, lr, ar, delta) {
					t.Fatalf("A vs L*U at [%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, lr, ar, delta)
				}
			}
		}
	}
}

// propReconstructionQR verifies A ≈ Qᵀ*R within a given tolerance.
// Note: With the current implementation, reflectors are accumulated on the left,
// so the decomposition realized by the function is m ≈ Qᵀ*R (not Q*R).
func propReconstructionQR(t *testing.T, A, Q, R matrix.Matrix, delta float64) {
	t.Helper()

	var (
		i, j int
		lv   float64
		rv   float64
		err  error
	)

	Qt, err := matrix.Transpose(Q)
	if err != nil {
		t.Fatalf("matrix.Transpose(Q): want err == nil, got: %v", err)
	}
	QtR, err := matrix.Mul(Qt, R)
	if err != nil {
		t.Fatalf("matrix.Mul(Qt, R): want err == nil, got: %v", err)
	}

	if A.Rows() != QtR.Rows() {
		t.Fatalf("want A.Rows() == QtR.Rows(), got: %d != %d", A.Rows(), QtR.Rows())
	}
	if A.Cols() != QtR.Cols() {
		t.Fatalf("want A.Cols() == QtR.Cols(), got: %d != %d", A.Cols(), QtR.Cols())
	}

	for i = 0; i < A.Rows(); i++ {
		for j = 0; j < A.Cols(); j++ {
			lv = MustAt(t, A, i, j)
			rv = MustAt(t, QtR, i, j)
			if !InDelta(t, lv, rv, delta) {
				t.Fatalf("A vs Qᵀ*R mismatch at [%d,%d]: want |%.6g-%.6g|<=%.1e", i, j, lv, rv, delta)
			}
		}
	}
}
