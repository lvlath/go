// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External numeric regressions compare the public API with independent arithmetic.
// They preserve discrete error identities and reject panics at allocation limits.
package series_test

import (
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/series"
)

// TestValidate_UnsignedDeltaCrossesZero verifies the exact signed-int64 boundary.
// A mathematically positive interval may be unrepresentable even when its later
// endpoint is zero; the oracle uses unbounded signed integer subtraction.
func TestValidate_UnsignedDeltaCrossesZero(t *testing.T) {
	const cadence = int64(1)
	tests := []struct {
		name     string
		previous int64
		current  int64
		want     error
	}{
		{name: "min to zero overflow", previous: math.MinInt64, current: 0, want: series.ErrOverflow},
		{name: "minus one to max overflow", previous: -1, current: math.MaxInt64, want: series.ErrOverflow},
		{name: "min to negative one boundary", previous: math.MinInt64, current: -1},
		{name: "minus one to max minus one boundary", previous: -1, current: math.MaxInt64 - 1},
		{name: "duplicate is order first", previous: math.MinInt64, current: math.MinInt64, want: series.ErrOrder},
		{name: "reverse across zero", previous: 0, current: math.MinInt64, want: series.ErrOrder},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Use the mathematical integer delta, not the production uint64 method.
			span := new(big.Int).Sub(big.NewInt(tc.current), big.NewInt(tc.previous))
			if tc.want == nil && (span.Sign() <= 0 || !span.IsInt64()) {
				t.Fatalf("bad oracle fixture: unrepresentable span %s", span.String())
			}

			for _, opts := range []series.GridOptions{{}, {CadenceNS: cadence, AllowGaps: true}} {
				got, err := series.Validate([]int64{tc.previous, tc.current}, []float64{1, 2}, opts)
				mustErrorIs(t, err, tc.want)
				if tc.want != nil {
					if got != nil {
						t.Fatalf("error must have nil result: %+v", got)
					}
					continue
				}
				if got == nil || got.Samples != 2 {
					t.Fatalf("successful interval returned %+v", got)
				}
				// An allowed positive-cadence gap is described but never filled.
				if opts.CadenceNS > 0 {
					wantMissing := span.Int64() - 1
					assertGaps(t, got.Gaps, []series.Gap{{AfterIndex: 0, MissingCount: wantMissing}})
				} else {
					assertGaps(t, got.Gaps, []series.Gap{})
				}
			}
		})
	}
}

// TestLogReturns_AdjacentLargeFloatsPreserveSignal protects tiny changes.
// The reference uses exact binary64 rational inputs and evaluates the first
// terms of ln(1+r), where |r| is approximately one machine epsilon.
func TestLogReturns_AdjacentLargeFloatsPreserveSignal(t *testing.T) {
	bases := []float64{1e300, 1e-300}
	for _, previous := range bases {
		for _, direction := range []float64{math.Inf(-1), math.Inf(1)} {
			current := math.Nextafter(previous, direction)
			result, err := series.LogReturns([]float64{previous, current})
			if err != nil || result == nil {
				t.Fatalf("previous=%g current=%g error=%v result=%+v", previous, current, err, result)
			}
			if !result.Valid[1] || !finiteLogReturn(result.Values[1]) {
				t.Fatalf("nonfinite/invalid log result: %+v", result)
			}

			// Construct r=(current-previous)/previous using exact input rationals.
			previousRat := new(big.Rat).SetFloat64(previous)
			currentRat := new(big.Rat).SetFloat64(current)
			relative := new(big.Rat).Quo(new(big.Rat).Sub(currentRat, previousRat), previousRat)
			// ln(1+r) = r - r^2/2 + O(r^3); the omitted term is < 1e-46
			// for these adjacent high-magnitude IEEE-754 numbers.
			square := new(big.Rat).Mul(relative, relative)
			square.Quo(square, big.NewRat(2, 1))
			want, _ := new(big.Rat).Sub(relative, square).Float64()
			if want == 0 || (want < 0) != (result.Values[1] < 0) {
				t.Fatalf("incorrect sign: got=%g want=%g", result.Values[1], want)
			}
			if math.Abs(result.Values[1]-want) > 4e-30 {
				t.Fatalf("tiny log return lost accuracy: got=%g want=%g", result.Values[1], want)
			}
		}
	}
}

// TestWindowRanges_RejectsUnrepresentableByteSize protects the allocation gate.
// A range count may fit int even when two-int Range storage cannot; panicking
// here is not the specified arithmetic-error protocol.
func TestWindowRanges_RejectsUnrepresentableByteSize(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("WindowRanges panicked instead of ErrOverflow: %v", recovered)
		}
	}()

	result, err := series.WindowRanges(math.MaxInt, 1, 1)
	mustErrorIs(t, err, series.ErrOverflow)
	if result != nil {
		t.Fatalf("ErrOverflow must return nil ranges, got length=%d", len(result))
	}

	// The same huge length remains legal if only one interval is emitted.
	ranges, err := series.WindowRanges(math.MaxInt, math.MaxInt, 1)
	mustErrorIs(t, err, nil)
	assertRanges(t, ranges, []series.Range{{Start: 0, End: math.MaxInt}})
}

// finiteLogReturn is a test-only predicate for the required finite result domain.
// It deliberately does not call a production helper or inspect private state.
func finiteLogReturn(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// TestSimpleReturns_ClosePairAndOppositeSign verifies both numeric branches.
// An adjacent huge pair retains its nonzero return, while opposite-sign
// MaxFloat64 values produce a finite -2 despite overflowing subtraction.
func TestSimpleReturns_ClosePairAndOppositeSign(t *testing.T) {
	previous := 1e300
	current := math.Nextafter(previous, math.Inf(1))
	closeResult, err := series.SimpleReturns([]float64{previous, current})
	mustErrorIs(t, err, nil)

	// Use an exact rational oracle rather than quotient-minus-one in float64.
	previousRat := new(big.Rat).SetFloat64(previous)
	currentRat := new(big.Rat).SetFloat64(current)
	want, _ := new(big.Rat).Quo(new(big.Rat).Sub(currentRat, previousRat), previousRat).Float64()
	if closeResult == nil || !closeResult.Valid[1] || math.Abs(closeResult.Values[1]-want) > 4e-30 {
		t.Fatalf("close return = %+v, want %g", closeResult, want)
	}

	// delta=-MaxFloat64-MaxFloat64 would overflow, but curr/prev-1 = -2.
	oppositeResult, err := series.SimpleReturns([]float64{math.MaxFloat64, -math.MaxFloat64})
	mustErrorIs(t, err, nil)
	assertTransform(t, oppositeResult, []float64{0, -2}, []bool{false, true})
}
