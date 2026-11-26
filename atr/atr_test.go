// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr_test protects candle-domain, exact-rounding, and owning batch contracts.
// AI-Hints: compare binary64 bits, not an epsilon that hides incorrect recurrence.
package atr_test

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/lvlath/go/atr"
)

// TestATR_SeedAndStepGolden anchors TR=[2,4,2], seed=3, then ATR=2.5 for period two.
// hand-calculated first-bar, gap, seed, and first mature step.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - hand-calculated first-bar, gap, seed, and first mature step.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Do not seed from an invented previous close.
func TestATR_SeedAndStepGolden(t *testing.T) {
	r := compute(t, []float64{10, 13, 12}, []float64{8, 11, 10}, []float64{9, 12, 11}, 2)
	mustResult(t, r, &atr.Result{TR: []float64{2, 4, 2}, ATR: []float64{0, 3, 2.5}, Valid: []bool{false, true, true}})
}

// TestATR_PeriodOneIdentity requires ATR and TR to have identical bits at every period-one index.
// finite varied candles, ready masks, and canonical zero outputs.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - finite varied candles, ready masks, and canonical zero outputs.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Do not route P1 through a rounded affine approximation.
func TestATR_PeriodOneIdentity(t *testing.T) {
	h, l, c := marketBars(257)
	negativeZero := math.Copysign(0, -1)
	h[0], l[0], c[0] = negativeZero, 0, negativeZero
	r := compute(t, h, l, c, 1)
	for i := range r.TR {
		mustFloatBits(t, r.ATR[i], r.TR[i])
		if !r.Valid[i] {
			t.Fatalf("P=1 masked index %d", i)
		}
	}
	mustFloatBits(t, r.TR[0], 0)
}

// TestATR_FlatSeriesZero distinguishes a valid zero range from unfinished warmup.
// constant candles with false then true readiness.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - constant candles with false then true readiness.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Never use ATR!=0 as the readiness predicate.
func TestATR_FlatSeriesZero(t *testing.T) {
	r := compute(t, []float64{5, 5, 5}, []float64{5, 5, 5}, []float64{5, 5, 5}, 2)
	mustResult(t, r, &atr.Result{TR: []float64{0, 0, 0}, ATR: []float64{0, 0, 0}, Valid: []bool{false, true, true}})
}

// TestATR_GapFromClose measures gaps from the last accepted close.
// upward and downward movement beyond the candle interval.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - upward and downward movement beyond the candle interval.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - A current open or close is not the previous close.
func TestATR_GapFromClose(t *testing.T) {
	r := compute(t, []float64{10, 21, 6}, []float64{8, 20, 5}, []float64{9, 20, 5}, 2)
	mustResult(t, r, &atr.Result{TR: []float64{2, 12, 15}, ATR: []float64{0, 7, 11}, Valid: []bool{false, true, true}})
}

// TestATR_SignedPrices admits finite negative and zero-crossing prices.
// signed interval geometry and positive range magnitudes.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - signed interval geometry and positive range magnitudes.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Do not copy a positive-price restriction from another indicator.
func TestATR_SignedPrices(t *testing.T) {
	r := compute(t, []float64{-1, 1}, []float64{-3, -2}, []float64{-2, 0}, 2)
	mustResult(t, r, &atr.Result{TR: []float64{2, 3}, ATR: []float64{0, 2.5}, Valid: []bool{false, true}})
}

// TestATR_RangeOverflow rejects nonrepresentable required ranges and preserves state.
// first and subsequent range overflows with nil batch results.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - first and subsequent range overflows with nil batch results.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - A finite possible average cannot excuse an infinite TR output.
func TestATR_RangeOverflow(t *testing.T) {
	r, err := atr.Compute(context.Background(), []float64{math.MaxFloat64}, []float64{-math.MaxFloat64}, []float64{0}, atr.Options{Period: 14})
	mustErrorIs(t, err, atr.ErrOverflow)
	if r != nil {
		t.Fatal("overflow published partial result")
	}
	s := newStream(t, 14)
	before := s.Snapshot()
	v, err := s.Push(context.Background(), math.MaxFloat64, -math.MaxFloat64, 0)
	mustErrorIs(t, err, atr.ErrOverflow)
	mustValue(t, v, atr.Value{})
	mustSnapshot(t, s.Snapshot(), before)
	// A zero-width candle can still overflow through its gap from the last close.
	push(t, s, candle{-math.MaxFloat64, -math.MaxFloat64, -math.MaxFloat64})
	before = s.Snapshot()
	v, err = s.Push(context.Background(), math.MaxFloat64, math.MaxFloat64, math.MaxFloat64)
	mustErrorIs(t, err, atr.ErrOverflow)
	mustValue(t, v, atr.Value{})
	mustSnapshot(t, s.Snapshot(), before)
}

// TestATR_WrongBarsErrBar rejects each finite candle geometry violation.
// inverted high/low and close outside either interval endpoint.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - inverted high/low and close outside either interval endpoint.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Never repair the candle or advance the stream on rejection.
func TestATR_WrongBarsErrBar(t *testing.T) {
	for i, bar := range []candle{{1, 2, 1.5}, {3, 1, 0}, {3, 1, 4}, {-3, -1, -2}, {math.MaxFloat64, -math.MaxFloat64, math.Inf(1)}} {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			want := atr.ErrBar
			if math.IsInf(bar.close, 0) {
				want = atr.ErrNonFinite
			}
			r, err := atr.Compute(context.Background(), []float64{bar.high}, []float64{bar.low}, []float64{bar.close}, atr.Options{Period: 2})
			mustErrorIs(t, err, want)
			if r != nil {
				t.Fatal("invalid candle published result")
			}
		})
	}
}

// TestATR_ValidationPrecedenceAndEmpty anchors context, option, shape, data, and empty-input precedence.
// competing invalid inputs and owning nonnil empty results.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - competing invalid inputs and owning nonnil empty results.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Changing scan order changes observable error classification.
func TestATR_ValidationPrecedenceAndEmpty(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	cases := []struct {
		name    string
		ctx     context.Context
		h, l, c []float64
		period  int
		want    error
	}{
		{"context-nil", nil, nil, []float64{1}, nil, 0, atr.ErrNilContext},
		{"context-canceled", canceled, nil, []float64{1}, nil, 0, context.Canceled},
		{"options", context.Background(), nil, []float64{1}, nil, 0, atr.ErrInvalidOption},
		{"nil-high", context.Background(), nil, []float64{1}, []float64{1}, 2, atr.ErrNilInput},
		{"nil-low", context.Background(), []float64{1}, nil, []float64{1}, 2, atr.ErrNilInput},
		{"nil-close", context.Background(), []float64{1}, []float64{1}, nil, 2, atr.ErrNilInput},
		{"shape-low", context.Background(), []float64{math.NaN()}, []float64{}, []float64{0}, 2, atr.ErrShape},
		{"shape-close", context.Background(), []float64{1}, []float64{1}, []float64{}, 2, atr.ErrShape},
		{"nan-high", context.Background(), []float64{math.NaN()}, []float64{2}, []float64{0}, 2, atr.ErrNonFinite},
		{"inf-low", context.Background(), []float64{1}, []float64{math.Inf(-1)}, []float64{0}, 2, atr.ErrNonFinite},
		{"inf-close", context.Background(), []float64{1}, []float64{0}, []float64{math.Inf(1)}, 2, atr.ErrNonFinite},
		{"indexed-geometry-first", context.Background(), []float64{1, math.NaN()}, []float64{2, 0}, []float64{1, 0}, 2, atr.ErrBar},
		{"domain-before-arithmetic", context.Background(), []float64{math.MaxFloat64, 1}, []float64{-math.MaxFloat64, 2}, []float64{0, 1}, 2, atr.ErrBar},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, err := atr.Compute(tc.ctx, tc.h, tc.l, tc.c, atr.Options{Period: tc.period})
			mustErrorIs(t, err, tc.want)
			if r != nil {
				t.Fatal("validation published a result")
			}
		})
	}
	for _, slices := range [][3][]float64{{nil, nil, nil}, {[]float64{}, nil, []float64{}}, {[]float64{}, []float64{}, []float64{}}} {
		r := compute(t, slices[0], slices[1], slices[2], 14)
		if r.TR == nil || r.ATR == nil || r.Valid == nil || len(r.TR) != 0 || len(r.ATR) != 0 || len(r.Valid) != 0 {
			t.Fatal("empty input contract")
		}
	}
	for _, period := range []int{-1, 0} {
		s, err := atr.New(atr.Options{Period: period})
		mustErrorIs(t, err, atr.ErrInvalidOption)
		if s != nil {
			t.Fatal("invalid New returned stream")
		}
	}
}

// TestATR_ShortHistoryOwnershipAndPrefixes protects warmup measurements, output ownership, and prefix causality.
// short histories, input/output mutations, and complete prefixes.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - short histories, input/output mutations, and complete prefixes.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Full validation is not permission to use a future close numerically.
func TestATR_ShortHistoryOwnershipAndPrefixes(t *testing.T) {
	h, l, c := marketBars(40)
	hCopy, lCopy, cCopy := append([]float64(nil), h...), append([]float64(nil), l...), append([]float64(nil), c...)
	short := compute(t, h[:3], l[:3], c[:3], 14)
	for i := range short.TR {
		if short.Valid[i] {
			t.Fatal("premature readiness")
		}
		mustFloatBits(t, short.ATR[i], 0)
		if short.TR[i] <= 0 {
			t.Fatal("warmup TR erased")
		}
	}
	full := compute(t, h, l, c, 14)
	for n := 0; n <= len(h); n++ {
		prefix := compute(t, h[:n], l[:n], c[:n], 14)
		mustResult(t, prefix, &atr.Result{TR: full.TR[:n], ATR: full.ATR[:n], Valid: full.Valid[:n]})
	}
	for i := range h {
		mustFloatBits(t, h[i], hCopy[i])
		mustFloatBits(t, l[i], lCopy[i])
		mustFloatBits(t, c[i], cCopy[i])
	}
	expected := rationalHistory(h, l, c, 14)
	h[0], l[0], c[0] = 0, 0, 0
	mustResult(t, full, expected)
	full.TR[1], full.ATR[2], full.Valid[3] = -7, -8, true
	mustFloatBits(t, h[1], hCopy[1])
	mustFloatBits(t, l[2], lCopy[2])
	mustFloatBits(t, c[3], cCopy[3])
	// Even P=1 must return distinct TR and ATR arrays.
	one := compute(t, hCopy, lCopy, cCopy, 1)
	saved := one.ATR[0]
	one.TR[0] = -1
	mustFloatBits(t, one.ATR[0], saved)
}

// TestResult_IsNilInterface checks pointer presence through a typed nilable interface.
// typed nil, allocated empty Result, and populated Result.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - typed nil, allocated empty Result, and populated Result.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - IsNil must not dereference the receiver or inspect readiness.
func TestResult_IsNilInterface(t *testing.T) {
	var absent *atr.Result
	var wrapped nilable = absent
	if !wrapped.IsNil() {
		t.Fatal("typed nil is not nilable")
	}
	for _, r := range []*atr.Result{{}, compute(t, nil, nil, nil, 14), compute(t, []float64{1}, []float64{0}, []float64{0}, 14)} {
		wrapped = r
		if wrapped.IsNil() {
			t.Fatal("allocated result reported nil")
		}
	}
}

// TestATR_CancellationAcrossBothPasses cancels deterministically at validation and calculation boundaries.
// logical context observations across both long scans and publication.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - logical context observations across both long scans and publication.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Do not replace logical cancellation with timing-dependent sleeps.
func TestATR_CancellationAcrossBothPasses(t *testing.T) {
	h, l, c := marketBars(2049)
	// Initial=1; validation=2,3,4; preallocation=5; computation=6,7,8; publication=9.
	for at := 1; at <= 9; at++ {
		t.Run(fmt.Sprint(at), func(t *testing.T) {
			ctx := logicalCancellation(t, at)
			r, err := atr.Compute(ctx, h, l, c, atr.Options{Period: 14})
			mustErrorIs(t, err, context.Canceled)
			if r != nil {
				t.Fatal("cancellation published result")
			}
		})
	}
	for _, at := range []int{2, 3} {
		r, err := atr.Compute(logicalCancellation(t, at), nil, nil, nil, atr.Options{Period: 2})
		mustErrorIs(t, err, context.Canceled)
		if r != nil {
			t.Fatal("empty publication ignored cancellation")
		}
	}
}

// TestATR_RationalOracle checks complete histories against independently rounded rational arithmetic.
// multiple periods and reproducible finite sequences.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - multiple periods and reproducible finite sequences.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - The oracle must round TR before seeding or stepping ATR.
func TestATR_RationalOracle(t *testing.T) {
	h, l, c := marketBars(300)
	for _, period := range []int{1, 2, 3, 14, 64, 512} {
		mustResult(t, compute(t, h, l, c, period), rationalHistory(h, l, c, period))
	}
	rng := rand.New(rand.NewSource(3202))
	// Widely separated exponents, signs, subnormals, and exact binary endpoints.
	for scenario := 0; scenario < 40; scenario++ {
		h, l, c = make([]float64, 30), make([]float64, 30), make([]float64, 30)
		for i := range h {
			a := math.Ldexp(rng.Float64(), rng.Intn(2047)-1074)
			b := math.Ldexp(rng.Float64(), rng.Intn(2047)-1074)
			if rng.Intn(2) == 0 {
				a = -a
			}
			if rng.Intn(2) == 0 {
				b = -b
			}
			l[i], h[i] = math.Min(a, b), math.Max(a, b)
			if i%2 == 0 {
				c[i] = l[i]
			} else {
				c[i] = h[i]
			}
		}
		period := 1 + rng.Intn(17)
		mustResult(t, compute(t, h, l, c, period), rationalHistory(h, l, c, period))
	}
}

// TestATR_ExactRoundingAndFiniteConvexBounds protects subnormal ties and finite convex-combination bounds.
// extreme finite magnitudes, ties, and representable averages.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - extreme finite magnitudes, ties, and representable averages.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - No epsilon clamp may conceal a rounding or overflow defect.
func TestATR_ExactRoundingAndFiniteConvexBounds(t *testing.T) {
	unit := math.SmallestNonzeroFloat64
	smallestNormal := math.Float64frombits(1 << 52)
	cases := []struct {
		name         string
		previous, tr float64
		period       int
	}{
		{"half-unit-even-zero", unit, 0, 2},
		{"half-unit-even-two", 3 * unit, 0, 2},
		{"normal-boundary", math.Nextafter(smallestNormal, 0), smallestNormal, 2},
		{"normal-carry", math.Nextafter(2, 0), 2, 2},
		{"no-float-numerator", math.MaxFloat64, math.MaxFloat64, 14},
		{"max-to-unit", math.MaxFloat64, unit, 3},
		{"unit-to-max", unit, math.MaxFloat64, 64},
		{"exact-zero", 0, 0, 512},
		{"max-period", math.Nextafter(1, 2), unit, int(^uint(0) >> 1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := matureState(t, tc.period, tc.previous)
			got := push(t, s, candle{tc.tr, 0, 0})
			mustValue(t, got, atr.Value{TR: tc.tr, ATR: rationalStep(tc.previous, tc.tr, tc.period), Valid: true})
		})
	}
	for _, values := range [][]float64{{unit, 0}, {unit, 2 * unit}, {math.MaxFloat64, math.MaxFloat64}, {math.Nextafter(2, 0), 2}} {
		zeros := make([]float64, len(values))
		mustResult(t, compute(t, values, zeros, zeros, 2), rationalHistory(values, zeros, zeros, 2))
	}
	for _, period := range []int{3, 14, 64, 512} {
		h, l, c := make([]float64, period+1), make([]float64, period+1), make([]float64, period+1)
		for i := range h {
			h[i] = math.MaxFloat64
		}
		r := compute(t, h, l, c, period)
		mustFloatBits(t, r.ATR[period-1], math.MaxFloat64)
		mustFloatBits(t, r.ATR[period], math.MaxFloat64)
	}
}

// TestATR_RangeRoundingOverflowThreshold distinguishes finite rounding shoulders from infinite TR.
// binary64 values around the maximum finite rounding boundary.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - binary64 values around the maximum finite rounding boundary.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Exact magnitude above MaxFloat64 alone does not imply rounded overflow.
func TestATR_RangeRoundingOverflowThreshold(t *testing.T) {
	// Slightly above MaxFloat64 can still round to MaxFloat64; the required
	// rounded output, not exact real representability, governs overflow.
	r := compute(t, []float64{math.MaxFloat64}, []float64{-math.Ldexp(1, 969)}, []float64{0}, 1)
	mustFloatBits(t, r.TR[0], math.MaxFloat64)
	r, err := atr.Compute(context.Background(), []float64{math.MaxFloat64}, []float64{-math.Ldexp(1, 970)}, []float64{0}, atr.Options{Period: 1})
	mustErrorIs(t, err, atr.ErrOverflow)
	if r != nil {
		t.Fatal("overflow tie published result")
	}
}

// TestATR_RationalAffineBitSweep compares mature steps over a reproducible binary64 bit sweep.
// independent rational numerator and integer denominator.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - independent rational numerator and integer denominator.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Preserve every integer period bit; do not convert weights to float64.
func TestATR_RationalAffineBitSweep(t *testing.T) {
	rng := rand.New(rand.NewSource(107432))
	maximumInt := int(^uint(0) >> 1)
	for i := 0; i < 4096; i++ {
		previous := math.Float64frombits(rng.Uint64() & 0x7fefffffffffffff)
		tr := math.Float64frombits(rng.Uint64() & 0x7fefffffffffffff)
		period := int(rng.Uint64() & uint64(maximumInt))
		if period < 1 {
			period = 1
		}
		if i%4 == 0 {
			period = []int{1, 2, 3, 14, 64, 512, maximumInt}[i%7]
		}
		s := matureState(t, period, previous)
		value := push(t, s, candle{tr, 0, 0})
		want := rationalStep(previous, tr, period)
		if math.Float64bits(value.ATR) != math.Float64bits(want) {
			t.Fatalf("case %d P=%d previous=%x tr=%x: got=%x want=%x", i, period, math.Float64bits(previous), math.Float64bits(tr), math.Float64bits(value.ATR), math.Float64bits(want))
		}
	}
}

// TestATR_RationalSeedBitSweep compares first-ready seed values over varied binary64 magnitudes.
// exact rational sums of already-rounded ranges.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - exact rational sums of already-rounded ranges.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - A running floating mean is not an exact seed accumulator.
func TestATR_RationalSeedBitSweep(t *testing.T) {
	rng := rand.New(rand.NewSource(32021074))
	for trial := 0; trial < 80; trial++ {
		period := 2 + rng.Intn(35)
		h, l, c := make([]float64, period+2), make([]float64, period+2), make([]float64, period+2)
		for i := range h {
			h[i] = math.Float64frombits(rng.Uint64() & 0x7fefffffffffffff)
		}
		mustResult(t, compute(t, h, l, c, period), rationalHistory(h, l, c, period))
	}
}

// TestATR_EachNonFinitePriceAndExtremeGeometry checks every price channel and extreme invalid interval.
// NaN, infinities, and finite invalid geometry before arithmetic.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - NaN, infinities, and finite invalid geometry before arithmetic.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Validate finiteness before comparisons involving NaN.
func TestATR_EachNonFinitePriceAndExtremeGeometry(t *testing.T) {
	for channel := 0; channel < 3; channel++ {
		for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			prices := [3]float64{3, 1, 2}
			prices[channel] = bad
			r, err := atr.Compute(context.Background(), []float64{prices[0]}, []float64{prices[1]}, []float64{prices[2]}, atr.Options{Period: 2})
			mustErrorIs(t, err, atr.ErrNonFinite)
			if r != nil {
				t.Fatal("nonfinite price published result")
			}
		}
	}
	r, err := atr.Compute(context.Background(), []float64{-math.MaxFloat64}, []float64{math.MaxFloat64}, []float64{0}, atr.Options{Period: 2})
	mustErrorIs(t, err, atr.ErrBar)
	if r != nil {
		t.Fatal("inverted extreme bar published result")
	}
}

// TestATR_WilderNotRoundedAlphaOrDeltaUpdate anchors the one-ULP distinction from rounded-alpha or delta updates.
// the exact period-three one-third/two-thirds rounding witness.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - the exact period-three one-third/two-thirds rounding witness.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Do not change a bitwise assertion into an approximate comparison.
func TestATR_WilderNotRoundedAlphaOrDeltaUpdate(t *testing.T) {
	stream := matureState(t, 3, 1)
	got := push(t, stream, candle{0, 0, 0})
	mustFloatBits(t, got.ATR, math.Float64frombits(0x3fe5555555555555))
	mustFloatBits(t, got.ATR, rationalStep(1, 0, 3))
	high := []float64{1 << 53, 1, 1}
	zeros := []float64{0, 0, 0}
	mustResult(t, compute(t, high, zeros, zeros, 3), rationalHistory(high, zeros, zeros, 3))
}
