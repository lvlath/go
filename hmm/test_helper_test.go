package hmm_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/hmm"
	"github.com/lvlath/go/matrix"
)

func dense(t testing.TB, values [][]float64) matrix.Matrix {
	t.Helper()
	m, e := matrix.NewZeros(len(values), len(values[0]))
	if e != nil {
		t.Fatal(e)
	}
	for i, r := range values {
		for j, v := range r {
			if e = m.Set(i, j, v); e != nil {
				t.Fatal(e)
			}
		}
	}
	return m
}
func categorical(t testing.TB) *hmm.CategoricalModel {
	t.Helper()
	m, e := hmm.NewCategorical(hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{.5, .5}, Transition: dense(t, [][]float64{{1, 0}, {0, 1}})}, Emission: dense(t, [][]float64{{.9, .1}, {.1, .9}})})
	if e != nil {
		t.Fatal(e)
	}
	return m
}
func approx(t testing.TB, a, b float64) {
	t.Helper()
	if math.Abs(a-b) > 1e-10*math.Max(1, math.Abs(b)) {
		t.Fatalf("got %.17g expected %.17g", a, b)
	}
}
func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }
