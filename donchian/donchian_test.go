// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian_test tests golden vectors, causality, and ownership through the public API.
// This file uses only exported production operations and preserves explicit fixture ownership.
// AI-Hints: Keep mask checks and signed-zero bit checks separate from approximate numerical comparisons.
package donchian_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/lvlath/go/donchian"
)

// TestDonchian_IncludeCurrentGolden anchors the published include-current window and midpoint vector.
// The first ready row is P-1, and the current low participates in the next range.
//
// Implementation:
//   - Stage 1: Supply the three fixed candles with Period=2 and current-bar inclusion.
//   - Stage 2: Call the public canonical facade once.
//   - Stage 3: Compare all channels with the literal F,T,T mask and the two inclusive boundary triples.
//
// Behavior highlights:
//   - Compare all channels with the literal F,T,T mask and the two inclusive boundary triples.
//
// Inputs:
//   - literal P=2 high/low arrays and independently specified payloads.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - The literal golden artifact fixes every bit and observation position.
//
// Complexity:
//   - O(n) work for the fixed golden vector.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Do not replace the literal expectations with values produced by Compute.
func TestDonchian_IncludeCurrentGolden(t *testing.T) {
	got := mustCompute(t, []float64{2, 4, 3}, []float64{1, 2, 0}, donchian.Options{Period: 2})
	assertResult(t, got, &donchian.Result{
		Upper: []float64{0, 4, 4}, Middle: []float64{0, 2.5, 2},
		Lower: []float64{0, 1, 0}, Valid: []bool{false, true, true},
	})
}

// TestDonchian_ExcludeCurrentGolden anchors the published previous-window golden vector.
// The current low must not affect the prior-range midpoint or readiness.
//
// Implementation:
//   - Stage 1: Supply the same three candles with Period=2 and current-bar exclusion.
//   - Stage 2: Call the public canonical facade once.
//   - Stage 3: Compare the F,F,T mask and the last prior-window triple with literal expectations.
//
// Behavior highlights:
//   - Compare the F,F,T mask and the last prior-window triple with literal expectations.
//
// Inputs:
//   - literal P=2 arrays and an explicit false,false,true mask.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - The literal expectation distinguishes the prior low of 1 from the current low of 0.
//
// Complexity:
//   - O(n) work for the fixed golden vector.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Do not shift only the mask while leaving inclusive boundary values in place.
func TestDonchian_ExcludeCurrentGolden(t *testing.T) {
	got := mustCompute(t, []float64{2, 4, 3}, []float64{1, 2, 0},
		donchian.Options{Period: 2, ExcludeCurrent: true})
	assertResult(t, got, &donchian.Result{
		Upper: []float64{0, 0, 4}, Middle: []float64{0, 0, 2.5},
		Lower: []float64{0, 0, 1}, Valid: []bool{false, false, true},
	})
}

// TestDonchian_PeriodOne checks current versus one-observation-delayed channels.
// Negative prices and ready zero values exercise the degenerate window without a special kernel.
//
// Implementation:
//   - Stage 1: Build positive, negative, and zero single-candle ranges.
//   - Stage 2: Compute Period=1 in both modes.
//   - Stage 3: Compare current versus one-row-delayed bounds and readiness with the direct oracle.
//
// Behavior highlights:
//   - Compare current versus one-row-delayed bounds and readiness with the direct oracle.
//
// Inputs:
//   - signed finite prices and both modes with Period=1.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Include precedes exclude; the ordered input contains no random values.
//
// Complexity:
//   - O(n) direct-window comparison.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Period one does not make the first exclude-current row ready.
func TestDonchian_PeriodOne(t *testing.T) {
	high, low := []float64{2, 4, -1, 0}, []float64{1, 3, -5, 0}
	for _, exclude := range []bool{false, true} {
		opts := donchian.Options{Period: 1, ExcludeCurrent: exclude}
		assertResult(t, mustCompute(t, high, low, opts), exactOracle(high, low, opts))
	}
}

// TestDonchian_FutureSpikeCausality checks future-spike isolation and every history prefix.
// Earlier rows must agree whether computed alone or as part of a longer admitted history.
//
// Implementation:
//   - Stage 1: Compute an original history and clone it before introducing a future two-sided spike.
//   - Stage 2: Recompute the changed history and every prefix of the original history in each mode.
//   - Stage 3: Check unchanged past rows, unchanged current exclude bounds, and complete prefix equality.
//
// Behavior highlights:
//   - Check unchanged past rows, unchanged current exclude bounds, and complete prefix equality.
//
// Inputs:
//   - a procedural history, an extreme future spike, and all prefixes.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - The spike location and every prefix length are fixed; all comparisons preserve index order.
//
// Complexity:
//   - O(n*n+n*P) test work across every prefix.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Do not assume that checking only the final row proves prefix causality.
func TestDonchian_FutureSpikeCausality(t *testing.T) {
	high, low := syntheticBars(73)
	const boundary = 40
	for _, exclude := range []bool{false, true} {
		opts := donchian.Options{Period: 14, ExcludeCurrent: exclude}
		original := mustCompute(t, high, low, opts)
		h, l := append([]float64(nil), high...), append([]float64(nil), low...)
		h[boundary+1], l[boundary+1] = math.MaxFloat64, -math.MaxFloat64
		changed := mustCompute(t, h, l, opts)
		prefix := mustCompute(t, high[:boundary+1], low[:boundary+1], opts)
		assertPrefix(t, original, prefix)
		assertPrefix(t, changed, prefix)
		if exclude {
			assertFloats(t, []float64{changed.Upper[boundary+1], changed.Lower[boundary+1]},
				[]float64{original.Upper[boundary+1], original.Lower[boundary+1]})
		}
		for n := 0; n <= len(high); n++ {
			assertPrefix(t, original, mustCompute(t, high[:n], low[:n], opts))
		}
	}
}

// TestDonchian_CurrentBarExcluded separates same-bar contamination from future-data dependence.
// An extreme current bar changes inclusion but leaves its excluded window unchanged.
//
// Implementation:
//   - Stage 1: Compute the previous-two-bar channel before changing the final candle.
//   - Stage 2: Replace the final high/low with opposite maximal finite values.
//   - Stage 3: Require the exclude artifact unchanged and inclusive bounds equal to the current spike.
//
// Behavior highlights:
//   - Require the exclude artifact unchanged and inclusive bounds equal to the current spike.
//
// Inputs:
//   - a three-bar history with a replaced final high and low.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Both calls use the same revised input, differing only in the explicit window policy.
//
// Complexity:
//   - O(n) work on the fixed fixture.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Run both modes so a no-op kernel cannot satisfy the causality assertion.
func TestDonchian_CurrentBarExcluded(t *testing.T) {
	high, low := []float64{2, 4, 3}, []float64{1, 2, 0}
	opts := donchian.Options{Period: 2, ExcludeCurrent: true}
	before := mustCompute(t, high, low, opts)
	high[2], low[2] = math.MaxFloat64, -math.MaxFloat64
	assertResult(t, mustCompute(t, high, low, opts), before)
	included := mustCompute(t, high, low, donchian.Options{Period: 2})
	if included.Upper[2] != math.MaxFloat64 || included.Lower[2] != -math.MaxFloat64 {
		t.Fatal("include-current failed to incorporate the current spike")
	}
}

// TestDonchian_LargeMidpointNoOverflow checks equal boundaries at both finite magnitude extremes.
// A same-sign overflowing sum must not destroy a representable boundary mean.
//
// Implementation:
//   - Stage 1: Choose positive and negative MaxFloat64 as equal high/low pairs.
//   - Stage 2: Compute one-bar channels through the public API.
//   - Stage 3: Require each midpoint to retain the exact finite operand bits.
//
// Behavior highlights:
//   - Require each midpoint to retain the exact finite operand bits.
//
// Inputs:
//   - positive and negative MaxFloat64 identity cases.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - The positive and negative maximal pairs are exercised in declared order.
//
// Complexity:
//   - O(1) fixed-case work.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Do not change these expected means to infinity or permit an overflow error.
func TestDonchian_LargeMidpointNoOverflow(t *testing.T) {
	for _, value := range []float64{math.MaxFloat64, -math.MaxFloat64} {
		got := mustCompute(t, []float64{value}, []float64{value}, donchian.Options{Period: 1})
		assertFloats(t, got.Middle, []float64{value})
	}
}

// TestDonchian_OppositeMidpoint checks exact cancellation of opposite maximal boundaries.
// The result must be positive zero without forming an overflowing difference.
//
// Implementation:
//   - Stage 1: Choose high=MaxFloat64 and low=-MaxFloat64.
//   - Stage 2: Compute the single complete window.
//   - Stage 3: Require the midpoint bit pattern to be positive zero.
//
// Behavior highlights:
//   - Require the midpoint bit pattern to be positive zero.
//
// Inputs:
//   - MaxFloat64 and its negation.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - The result is checked by bits rather than an epsilon or sign-insensitive zero comparison.
//
// Complexity:
//   - O(1) fixed-case work.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Numeric equality alone cannot distinguish the required zero sign.
func TestDonchian_OppositeMidpoint(t *testing.T) {
	got := mustCompute(t, []float64{math.MaxFloat64}, []float64{-math.MaxFloat64}, donchian.Options{Period: 1})
	assertFloats(t, got.Middle, []float64{0})
}

// TestDonchian_EqualExtremaEviction protects latest-index ties through repeated expiration.
// Signed zeros make otherwise value-equal index selection publicly observable.
//
// Implementation:
//   - Stage 1: Build repeated extrema followed by smaller ranges and later new extrema.
//   - Stage 2: Compute several periods in both modes, then repeat with alternating zero signs.
//   - Stage 3: Compare every eviction and latest-equal choice with direct window scans.
//
// Behavior highlights:
//   - Compare every eviction and latest-equal choice with direct window scans.
//
// Inputs:
//   - repeated extrema, multiple periods, both modes, and alternating zero signs.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Numerically equal signed zeros make the latest-index choice publicly observable.
//
// Complexity:
//   - O(n*P) independent direct-window comparison.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Do not weaken bit comparisons to numerical equality on signed-zero fixtures.
func TestDonchian_EqualExtremaEviction(t *testing.T) {
	high := []float64{5, 5, 4, 3, 3, 2, 7, 7, 6, 1, 1, 0}
	low := []float64{-5, -5, -4, -3, -3, -2, -7, -7, -6, -1, -1, 0}
	for _, exclude := range []bool{false, true} {
		for _, period := range []int{1, 2, 3, 5} {
			opts := donchian.Options{Period: period, ExcludeCurrent: exclude}
			assertResult(t, mustCompute(t, high, low, opts), exactOracle(high, low, opts))
		}
	}
	// Signed zeros make the latest-index tie rule observable through public output.
	negativeZero := math.Copysign(0, -1)
	high, low = []float64{0, negativeZero, 0, negativeZero}, []float64{negativeZero, 0, negativeZero, 0}
	for _, exclude := range []bool{false, true} {
		opts := donchian.Options{Period: 2, ExcludeCurrent: exclude}
		assertResult(t, mustCompute(t, high, low, opts), exactOracle(high, low, opts))
	}
}

// TestDonchian_ResourceBoundaryMaxInt checks wholly masked output without period-sized storage.
// A maximal positive int remains admissible on a two-observation input.
//
// Implementation:
//   - Stage 1: Supply two finite candles with the largest positive int period.
//   - Stage 2: Check each mode against an entirely masked oracle and measure repeated allocations.
//   - Stage 3: Require successful zero-payload output and a bounded number of allocated objects.
//
// Behavior highlights:
//   - Require successful zero-payload output and a bounded number of allocated objects.
//
// Inputs:
//   - both modes, Period=MaxInt, and allocation-count observations.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Each allocation trial uses the same two observations and explicit maximal period.
//
// Complexity:
//   - O(n) per call; no oracle window is ready.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Allocation count is a supplementary check, not a byte-size or runtime-OOM guarantee.
func TestDonchian_ResourceBoundaryMaxInt(t *testing.T) {
	high, low := []float64{2, 4}, []float64{1, 3}
	for _, exclude := range []bool{false, true} {
		opts := donchian.Options{Period: math.MaxInt, ExcludeCurrent: exclude}
		assertResult(t, mustCompute(t, high, low, opts), exactOracle(high, low, opts))
		allocations := testing.AllocsPerRun(20, func() {
			benchmarkResult = mustCompute(t, high, low, opts)
		})
		if allocations > 4 {
			t.Fatalf("masked n=2 path allocated %g objects", allocations)
		}
	}
}

// TestDonchian_WrongBarsErrBar rejects reversed finite ranges even during full warmup.
// A huge requested period does not waive candle-domain admission.
//
// Implementation:
//   - Stage 1: Supply a negative high that is below its negative low.
//   - Stage 2: Exercise a ready period and two wholly masked periods.
//   - Stage 3: Require ErrBar and no published result in every case.
//
// Behavior highlights:
//   - Require ErrBar and no published result in every case.
//
// Inputs:
//   - negative reversed ranges at ordinary and maximal periods.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Period choice must not bypass the same input-geometry failure.
//
// Complexity:
//   - O(1) for the fixed single-bar fixture.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Do not swap high and low or return a masked success on invalid geometry.
func TestDonchian_WrongBarsErrBar(t *testing.T) {
	for _, period := range []int{1, 100, math.MaxInt} {
		result, err := donchian.Compute(context.Background(), []float64{-5}, []float64{-1},
			donchian.Options{Period: period})
		if result != nil || !errors.Is(err, donchian.ErrBar) {
			t.Fatalf("period=%d: result=%v, error=%v", period, result, err)
		}
	}
}

// TestResult_IsNilInterface checks pointer nilability independently of readiness and contents.
// Typed nil is absent; empty, wholly masked, and valid-zero result pointers are present.
//
// Implementation:
//   - Stage 1: Store a typed nil Result pointer behind the structural IsNil interface.
//   - Stage 2: Then store caller-created, empty-success, masked, and measured-zero results.
//   - Stage 3: Require true only for the absent pointer, independently of channel contents.
//
// Behavior highlights:
//   - Require true only for the absent pointer, independently of channel contents.
//
// Inputs:
//   - the structural nilability interface and four non-nil result shapes.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Each interface receiver is fixed before the pointer-presence check.
//
// Complexity:
//   - O(1) fixed-case checks.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Never implement IsNil by inspecting lengths, validity, or numeric zero values.
func TestResult_IsNilInterface(t *testing.T) {
	var absent *donchian.Result
	var value interface{ IsNil() bool } = absent
	if !value.IsNil() {
		t.Fatal("typed nil must report IsNil")
	}
	for _, present := range []*donchian.Result{
		{},
		mustCompute(t, nil, nil, donchian.Options{Period: 1}),
		mustCompute(t, []float64{2}, []float64{1}, donchian.Options{Period: 20}),
		mustCompute(t, []float64{0}, []float64{0}, donchian.Options{Period: 1}),
	} {
		value = present
		if value.IsNil() {
			t.Fatal("non-nil empty result must not report IsNil")
		}
	}
}

// TestDonchian_ValidationPrecedence anchors call metadata and global input-error precedence.
// Classification and input immutability are checked separately from diagnostic text.
//
// Implementation:
//   - Stage 1: Prepare explicit context, option, shape, finite-domain, and geometry conflicts.
//   - Stage 2: Call Compute and retain bitwise snapshots of caller inputs.
//   - Stage 3: Check the winning sentinel, additional nil-input membership, and input immutability.
//
// Behavior highlights:
//   - Check the winning sentinel, additional nil-input membership, and input immutability.
//
// Inputs:
//   - nil, canceled, expired, malformed, non-finite, and reversed inputs.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Table order is fixed; cancellation and expired deadlines are established before calls.
//
// Complexity:
//   - O(n) per case, including input snapshots.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Do not introduce substring matching or let a later geometry error hide a non-finite input.
func TestDonchian_ValidationPrecedence(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	expired, release := context.WithDeadline(context.Background(), time.Unix(1, 0))
	defer release()
	cases := []struct {
		name      string
		ctx       context.Context
		high, low []float64
		period    int
		want      error
		alsoNil   bool
	}{
		{"nil context", nil, nil, []float64{math.NaN()}, 0, donchian.ErrNilContext, false},
		{"canceled", canceled, nil, []float64{math.NaN()}, 0, context.Canceled, false},
		{"deadline", expired, nil, nil, 0, context.DeadlineExceeded, false},
		{"zero period", context.Background(), nil, []float64{1}, 0, donchian.ErrInvalidOption, false},
		{"negative period", context.Background(), nil, nil, math.MinInt, donchian.ErrInvalidOption, false},
		{"shape", context.Background(), []float64{math.NaN()}, []float64{0, 0}, 1, donchian.ErrShape, false},
		{"nil high", context.Background(), nil, []float64{1}, 1, donchian.ErrShape, true},
		{"nil low", context.Background(), []float64{1}, nil, 1, donchian.ErrShape, true},
		{"empty not nil", context.Background(), []float64{}, []float64{1}, 1, donchian.ErrShape, false},
		{"NaN high", context.Background(), []float64{math.NaN()}, []float64{0}, 1, donchian.ErrNonFinite, false},
		{"Inf high", context.Background(), []float64{math.Inf(1)}, []float64{0}, 1, donchian.ErrNonFinite, false},
		{"NaN low", context.Background(), []float64{0}, []float64{math.NaN()}, 1, donchian.ErrNonFinite, false},
		{"negative Inf low", context.Background(), []float64{0}, []float64{math.Inf(-1)}, 1, donchian.ErrNonFinite, false},
		{"later nonfinite", context.Background(), []float64{0, 1}, []float64{1, math.Inf(1)}, 1, donchian.ErrNonFinite, false},
		{"finite geometry", context.Background(), []float64{0, -5}, []float64{1, -1}, 1, donchian.ErrBar, false},
		{"masked invalid", context.Background(), []float64{math.NaN()}, []float64{0}, math.MaxInt, donchian.ErrNonFinite, false},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			highCopy, lowCopy := append([]float64(nil), test.high...), append([]float64(nil), test.low...)
			result, err := donchian.Compute(test.ctx, test.high, test.low, donchian.Options{Period: test.period})
			if result != nil || !errors.Is(err, test.want) {
				t.Fatalf("result=%v error=%v, want %v", result, err, test.want)
			}
			if errors.Is(err, donchian.ErrNilInput) != test.alsoNil {
				t.Fatalf("unexpected nil-input classification: %v", err)
			}
			assertFloats(t, test.high, highCopy)
			assertFloats(t, test.low, lowCopy)
		})
	}
}

// TestDonchian_EmptyAndReadinessBoundaries checks empty ownership and the exact warmup transitions.
// Nil/empty combinations succeed, while include and exclude become ready one row apart.
//
// Implementation:
//   - Stage 1: Enumerate nil/empty channel combinations, then short histories around Period=3.
//   - Stage 2: Compute both window modes with positive periods.
//   - Stage 3: Compare non-nil empty channels, warmup payloads, and first-ready rows with the oracle.
//
// Behavior highlights:
//   - Compare non-nil empty channels, warmup payloads, and first-ready rows with the oracle.
//
// Inputs:
//   - all empty-shape combinations and lengths around Period=3.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Nilness combinations and history lengths are enumerated explicitly.
//
// Complexity:
//   - O(n*P) per bounded direct-window comparison.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - A short positive-period history is not an error or a ready zero measurement.
func TestDonchian_EmptyAndReadinessBoundaries(t *testing.T) {
	for _, high := range [][]float64{nil, {}} {
		for _, low := range [][]float64{nil, {}} {
			for _, exclude := range []bool{false, true} {
				opts := donchian.Options{Period: math.MaxInt, ExcludeCurrent: exclude}
				assertResult(t, mustCompute(t, high, low, opts), exactOracle(high, low, opts))
			}
		}
	}
	for _, n := range []int{1, 2, 3, 4} {
		high, low := syntheticBars(n)
		for _, exclude := range []bool{false, true} {
			opts := donchian.Options{Period: 3, ExcludeCurrent: exclude}
			assertResult(t, mustCompute(t, high, low, opts), exactOracle(high, low, opts))
		}
	}
}

// TestDonchian_Ownership checks detached inputs, independent calls, and append safety.
// Mutating one owning result must not alter input observations or another result.
//
// Implementation:
//   - Stage 1: Compute independent artifacts and preserve caller-input snapshots.
//   - Stage 2: Mutate one artifact, append to an upper channel, and finally change caller input.
//   - Stage 3: Require sibling results, sibling channels, and inputs unaffected by unrelated writes.
//
// Behavior highlights:
//   - Require sibling results, sibling channels, and inputs unaffected by unrelated writes.
//
// Inputs:
//   - two matching results, an input snapshot, and appended output data.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - All mutations happen after successful calls, in a fixed single-goroutine sequence.
//
// Complexity:
//   - O(n*P) oracle work plus O(n) copying.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - A shallow slice assignment is not an owning result copy.
func TestDonchian_Ownership(t *testing.T) {
	high, low := syntheticBars(30)
	highCopy, lowCopy := append([]float64(nil), high...), append([]float64(nil), low...)
	opts := donchian.Options{Period: 3}
	first, second := mustCompute(t, high, low, opts), mustCompute(t, high, low, opts)
	want := exactOracle(high, low, opts)
	assertResult(t, first, second)
	first.Upper[2], first.Middle[2], first.Lower[2], first.Valid[2] = 777, 778, 779, false
	assertResult(t, second, want)
	assertFloats(t, high, highCopy)
	assertFloats(t, low, lowCopy)
	third := mustCompute(t, high, low, opts)
	third.Upper = append(third.Upper, 999)
	assertFloats(t, third.Middle, want.Middle)
	assertFloats(t, third.Lower, want.Lower)
	high[2], low[2] = 1000, -1000
	assertResult(t, second, want)
}

// TestDonchian_CancellationAtEveryPoll checks all observed cancellation boundaries without sleeps.
// Both queue burst directions and the empty path must return nil on interruption.
//
// Implementation:
//   - Stage 1: Build upper-tail and lower-tail removal bursts and measure successful poll counts.
//   - Stage 2: Cancel a real child context at every observed poll for each mode.
//   - Stage 3: Require context.Canceled and no partial result on every interrupted run.
//
// Behavior highlights:
//   - Require context.Canceled and no partial result on every interrupted run.
//
// Inputs:
//   - deterministic context polling and long monotonic tail-removal bursts.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Cancellation is tied to Err observation counts, not scheduler timing.
//
// Complexity:
//   - O(q*n+n*P) for q observed polling positions.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Do not replace logical cancellation with scheduler-sensitive deadlines.
func TestDonchian_CancellationAtEveryPoll(t *testing.T) {
	for _, upperBurst := range []bool{false, true} {
		high, low := burstBars(4100, upperBurst)
		for _, exclude := range []bool{false, true} {
			opts := donchian.Options{Period: 4099, ExcludeCurrent: exclude}
			probe := newPollContext(0)
			result, err := donchian.Compute(probe, high, low, opts)
			if err != nil {
				t.Fatal(err)
			}
			assertResult(t, result, exactOracle(high, low, opts))
			polls := probe.calls.Load()
			probe.cancel()
			for stop := int64(1); stop <= polls; stop++ {
				ctx := newPollContext(stop)
				result, err = donchian.Compute(ctx, high, low, opts)
				ctx.cancel()
				if result != nil || !errors.Is(err, context.Canceled) {
					t.Fatalf("upper=%t exclude=%t poll=%d: result=%v error=%v", upperBurst, exclude, stop, result, err)
				}
			}
		}
	}
	// The empty path must check both post-validation and publication cancellation.
	for stop := int64(1); stop <= 3; stop++ {
		ctx := newPollContext(stop)
		result, err := donchian.Compute(ctx, nil, nil, donchian.Options{Period: 1})
		ctx.cancel()
		if result != nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("empty poll=%d: result=%v error=%v", stop, result, err)
		}
	}
}

// TestDonchian_ConcurrentIndependentCalls checks read-only input sharing between independent calls.
// Parallel subtests own their test goroutines and compare detached outputs to one oracle.
//
// Implementation:
//   - Stage 1: Build one read-only shared history and an independent expected artifact.
//   - Stage 2: Run multiple public computations against the same immutable inputs.
//   - Stage 3: Collect worker outcomes before performing fatal assertions in the owning test.
//
// Behavior highlights:
//   - Collect worker outcomes before performing fatal assertions in the owning test.
//
// Inputs:
//   - one immutable procedural input and eight independent calls.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Worker completion order may vary; every returned artifact has the same fixed expectation.
//
// Complexity:
//   - O(n*P) oracle preparation and O(n) per call.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Never call Fatal from an unmanaged worker goroutine or mutate the shared input.
func TestDonchian_ConcurrentIndependentCalls(t *testing.T) {
	high, low := syntheticBars(513)
	opts := donchian.Options{Period: 64, ExcludeCurrent: true}
	want := exactOracle(high, low, opts)
	for i := 0; i < 8; i++ {
		t.Run(fmt.Sprintf("reader-%d", i), func(t *testing.T) {
			t.Parallel()
			assertResult(t, mustCompute(t, high, low, opts), want)
		})
	}
}
