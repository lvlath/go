// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// External acceptance, regressions and adversarial properties.
// AI-Hints: Never weaken fractional overlap or finite extreme-case tests to fit a kernel.

package paa_test

import (
	"math"
	"math/big"
	"math/rand"
	"sync"
	"testing"

	"github.com/lvlath/go/paa"
)

// TestTransform_FractionalFiveByTwo The fractional five-to-two fixture proves every source cell participates.
// It protects exact unsimplified rational boundaries from floor-block changes.
func TestTransform_FractionalFiveByTwo(t *testing.T) {
	input := []float64{0, 2, 4, 6, 8}
	got, err := paa.Transform(input, 2)
	if err != nil {
		t.Fatal(err)
	}
	assertExactSegments(t, got, 5, 2)
	mustClose(t, got.Values[0], 1.6)
	mustClose(t, got.Values[1], 6.4)
	if got.Segments[0] != (paa.Segment{LeftNumerator: 0, RightNumerator: 5, Denominator: 2}) || got.Segments[1] != (paa.Segment{LeftNumerator: 5, RightNumerator: 10, Denominator: 2}) {
		t.Fatal("fractional boundaries changed")
	}
}

// TestTransform_WeightedMeanConservation Weighted conservation compares actual rounded output integrals in rational arithmetic.
// It also checks each independently computed segment mean against exact overlaps.
func TestTransform_WeightedMeanConservation(t *testing.T) {
	rng := rand.New(rand.NewSource(721))
	for n := 1; n <= 20; n++ {
		for trial := 0; trial < 12; trial++ {
			values := make([]float64, n)
			for i := range values {
				values[i] = float64(rng.Intn(20001)-10000) / 16
			}
			for w := 1; w <= n; w++ {
				got, err := paa.Transform(values, w)
				if err != nil {
					t.Fatalf("n=%d w=%d: %v", n, w, err)
				}
				assertExactSegments(t, got, n, w)
				oracle := oraclePAA(t, values, w)
				for j := 0; j < w; j++ {
					assertOracle(t, got.Values[j], oracle[j])
				}
				// Conservation is checked in a rational domain on the ACTUAL rounded
				// output, not with a potentially overflowing float64 reduction.
				inputIntegral := new(big.Rat)
				outputIntegral := new(big.Rat)
				length := new(big.Rat).SetFrac64(int64(n), int64(w))
				for _, v := range values {
					inputIntegral.Add(inputIntegral, new(big.Rat).SetFloat64(v))
				}
				for _, v := range got.Values {
					part := new(big.Rat).Mul(new(big.Rat).SetFloat64(v), length)
					outputIntegral.Add(outputIntegral, part)
				}
				diff := new(big.Rat).Sub(inputIntegral, outputIntegral)
				diff.Abs(diff)
				budget := new(big.Rat).SetFloat64(3e-13)
				absMass := new(big.Rat)
				for _, v := range values {
					absMass.Add(absMass, new(big.Rat).Abs(new(big.Rat).SetFloat64(v)))
				}
				budget.Mul(budget, absMass)
				if diff.Cmp(budget) > 0 {
					t.Fatalf("n=%d w=%d conservation discrepancy: %v > %v", n, w, diff, budget)
				}
			}
		}
	}
}

// TestTransform_IdentityAndOneSegment Identity and global-mean endpoints use opposite partition shapes.
// This prevents accidental truncation or aliasing in degenerate dimensions.
func TestTransform_IdentityAndOneSegment(t *testing.T) {
	values := []float64{1, 2, 3, -4, 9}
	identity, err := paa.Transform(values, len(values))
	if err != nil {
		t.Fatal(err)
	}
	assertExactSegments(t, identity, len(values), len(values))
	for i := range values {
		if identity.Values[i] != values[i] {
			t.Fatalf("identity[%d] = %v, want %v", i, identity.Values[i], values[i])
		}
	}
	one, err := paa.Transform([]float64{1, 2, 3}, 1)
	if err != nil {
		t.Fatal(err)
	}
	mustClose(t, one.Values[0], 2)
	assertExactSegments(t, one, 3, 1)
}

// TestTransform_FiniteExtremeScale Extreme but finite binary64 samples must never overflow on intermediate sums.
// Independent rational means distinguish harmless cancellation from a bad result.
func TestTransform_FiniteExtremeScale(t *testing.T) {
	cases := []struct {
		name  string
		input []float64
		w     int
		want  []float64
	}{
		{"largeConstant", []float64{1e308, 1e308}, 1, []float64{1e308}},
		{"maxConstant", []float64{math.MaxFloat64, math.MaxFloat64}, 1, []float64{math.MaxFloat64}},
		{"cancelMax", []float64{math.MaxFloat64, -math.MaxFloat64}, 1, []float64{0}},
		{"cancelFractional", []float64{math.MaxFloat64, -math.MaxFloat64, 0}, 2, []float64{math.MaxFloat64 / 3, -math.MaxFloat64 / 3}},
		{"negativeConstant", []float64{-1e308, -1e308}, 1, []float64{-1e308}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := paa.Transform(tc.input, tc.w)
			if err != nil {
				t.Fatal(err)
			}
			for j := range tc.want {
				assertOracle(t, got.Values[j], oraclePAA(t, tc.input, tc.w)[j])
			}
			for j := range tc.want {
				mustClose(t, got.Values[j], tc.want[j])
			}
		})
	}
}

// TestTransform_ConvexHullProperty Every mean lies between the min and max of its actual contributors.
// This property holds even for fractional boundaries and large signed values.
func TestTransform_ConvexHullProperty(t *testing.T) {
	rng := rand.New(rand.NewSource(478))
	for n := 1; n <= 100; n++ {
		values := make([]float64, n)
		for i := range values {
			values[i] = (rng.Float64()*2 - 1) * math.MaxFloat64
		}
		for w := 1; w <= n; w++ {
			got, err := paa.Transform(values, w)
			if err != nil {
				t.Fatal(err)
			}
			for j := 0; j < w; j++ {
				lo, hi := math.Inf(1), math.Inf(-1)
				for i, x := range values {
					if i*w >= (j+1)*n || (i+1)*w <= j*n {
						continue
					}
					if x < lo {
						lo = x
					}
					if x > hi {
						hi = x
					}
				}
				if got.Values[j] < lo || got.Values[j] > hi {
					t.Fatalf("n=%d w=%d j=%d: %v outside [%v,%v]", n, w, j, got.Values[j], lo, hi)
				}
			}
		}
	}
}

// TestTransform_DetachedOutputs Published means and rational cells are caller-owned, detached artifacts.
// Mutation of outputs, prior results or the input after return cannot alias them.
func TestTransform_DetachedOutputs(t *testing.T) {
	in := []float64{0, 2, 4, 6, 8}
	first, err := paa.Transform(in, 2)
	if err != nil {
		t.Fatal(err)
	}
	first.Values[0] = 99
	first.Segments[0].LeftNumerator = 77
	second, err := paa.Transform(in, 2)
	if err != nil {
		t.Fatal(err)
	}
	mustClose(t, second.Values[0], 1.6)
	if second.Segments[0].LeftNumerator != 0 {
		t.Fatal("result shared geometry")
	}
	in[0] = 100
	mustClose(t, second.Values[0], 1.6)
	if first.Values[0] == in[0] {
		t.Fatal("input/output shared storage")
	}
	identity, err := paa.Transform(in, len(in))
	if err != nil {
		t.Fatal(err)
	}
	identity.Values[0] = -12
	if in[0] != 100 {
		t.Fatal("identity aliases input")
	}
}

// TestTransform_ErrorPrecedence Invalid count, nil/empty source, bad shape and nonfinite values are ordered.
// Failures are classified through errors.Is and never publish a partial Result.
func TestTransform_ErrorPrecedence(t *testing.T) {
	cases := []struct {
		name   string
		values []float64
		w      int
		want   error
	}{
		{"optionBeforeNil", nil, 0, paa.ErrInvalidOption},
		{"negativeBeforeNil", nil, -1, paa.ErrInvalidOption},
		{"nil", nil, 1, paa.ErrNilInput},
		{"empty", []float64{}, 1, paa.ErrEmptyInput},
		{"emptyBeforeLargeOption", []float64{}, 2, paa.ErrEmptyInput},
		{"largeOptionBeforeNonfinite", []float64{math.NaN()}, 2, paa.ErrInvalidOption},
		{"nonfiniteNaN", []float64{math.NaN()}, 1, paa.ErrNonFinite},
		{"nonfinitePositiveInf", []float64{math.Inf(1)}, 1, paa.ErrNonFinite},
		{"nonfiniteNegativeInf", []float64{math.Inf(-1)}, 1, paa.ErrNonFinite},
		{"nonfiniteTail", []float64{1, 2, math.NaN()}, 1, paa.ErrNonFinite},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := paa.Transform(tc.values, tc.w)
			mustErrorIs(t, err, tc.want)
			if got != nil {
				t.Fatal("partial result published")
			}
		})
	}
}

// TestTransform_IntegerOverflowProtection The reachable oversized-count gate fails before indexing or allocation.
// Genuine n*w overflow needs an infeasibly large allocated valid Go slice.
func TestTransform_IntegerOverflowProtection(t *testing.T) {
	// An actual n*w int64 overflow with w<=n requires at least billions
	// of addressable float64 elements; Go cannot construct this valid fixture
	// without allocating prohibitive memory or fabricating an unsafe slice.
	// We verify the externally reachable upper-bound gate instead.
	got, err := paa.Transform([]float64{1, 2}, math.MaxInt)
	mustErrorIs(t, err, paa.ErrInvalidOption)
	if got != nil {
		t.Fatal("unexpected result")
	}
}

// TestTransform_ConcurrentIndependentCalls Independent transforms share no mutable hidden accumulator workspace.
// A stable source may be read by multiple goroutines without result aliasing.
func TestTransform_ConcurrentIndependentCalls(t *testing.T) {
	values := []float64{0, 2, 4, 6, 8, 10, 12, 14, 16}
	const workers = 16
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for k := 0; k < workers; k++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, err := paa.Transform(values, 4)
			if err == nil && (len(r.Values) != 4 || r.Segments[0].Denominator != 4) {
				err = paa.ErrOverflow
			}
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
}

// TestTransform_ExponentStressAgainstRationalOracle probes changing segment
// scales, severe sign cancellation, subnormal samples, and unbalanced overlaps.
// The rational oracle independently computes exact binary-float arithmetic.
func TestTransform_ExponentStressAgainstRationalOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(20261110))
	for n := 2; n <= 20; n++ {
		for trial := 0; trial < 30; trial++ {
			values := make([]float64, n)
			for i := range values {
				switch rng.Intn(6) {
				case 0:
					values[i] = math.MaxFloat64
				case 1:
					values[i] = -math.MaxFloat64
				case 2:
					values[i] = math.SmallestNonzeroFloat64
				case 3:
					values[i] = -math.SmallestNonzeroFloat64
				default:
					values[i] = math.Ldexp(rng.Float64()*2-1, rng.Intn(2046)-1022)
				}
			}
			for w := 1; w <= n; w++ {
				got, err := paa.Transform(values, w)
				if err != nil {
					t.Fatalf("n=%d w=%d trial=%d: %v", n, w, trial, err)
				}
				want := oraclePAA(t, values, w)
				for j, v := range got.Values {
					assertOracle(t, v, want[j])
				}
			}
		}
	}
}

// TestTransform_ExtremeCancellationResidue is a targeted regression for an
// incorrectly zeroed tiny tail after exact cancellation of enormous operands.
func TestTransform_ExtremeCancellationResidue(t *testing.T) {
	tiny := math.SmallestNonzeroFloat64
	values := []float64{math.MaxFloat64, -math.MaxFloat64, tiny, tiny, tiny}
	got, err := paa.Transform(values, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got.Values[0] != tiny {
		t.Fatalf("extreme cancellation lost representable residue: got %g want %g", got.Values[0], tiny)
	}
	assertOracle(t, got.Values[0], oraclePAA(t, values, 1)[0])
}

// TestTransform_FinalRoundingTiesToEven verifies IEEE-754 subnormal half-way
// cases rather than accepting a merely-close floating-point approximation.
func TestTransform_FinalRoundingTiesToEven(t *testing.T) {
	tiny := math.SmallestNonzeroFloat64
	cases := []struct {
		in   []float64
		want float64
	}{
		{[]float64{0, tiny}, 0},
		{[]float64{tiny, 2 * tiny}, 2 * tiny},
		{[]float64{2 * tiny, 3 * tiny}, 2 * tiny},
		{[]float64{3 * tiny, 4 * tiny}, 4 * tiny},
	}
	for k, tc := range cases {
		got, err := paa.Transform(tc.in, 1)
		if err != nil {
			t.Fatalf("fixture %d: %v", k, err)
		}
		if got.Values[0] != tc.want {
			t.Fatalf("fixture %d: got %g want %g", k, got.Values[0], tc.want)
		}
	}
}
