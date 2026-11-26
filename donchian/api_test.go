// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian_test protects named-facade parity, error identity, cancellation, and ownership.
// This file uses only exported production operations and preserves explicit fixture ownership.
// AI-Hints: A wrapper must not introduce its own polling, defaults, or alternate numerical path.
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

// TestComputeModeWrappers_Parity checks both named facades against canonical calls and the oracle.
// Mode selection must preserve every boundary bit, mask bit, and empty-channel invariant.
//
// Implementation:
//   - Stage 1: Prepare nil, empty, signed, extreme, subnormal, and frozen market inputs.
//   - Stage 2: Run both wrappers and their matching canonical Options for six explicit periods.
//   - Stage 3: Compare every channel against both canonical output and the direct-window oracle.
//
// Behavior highlights:
//   - Compare every channel against both canonical output and the direct-window oracle.
//
// Inputs:
//   - empty, signed, extreme, subnormal, and sourced market histories at six periods.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Wrapper descriptors, input cases, and period values are visited in declared slice order.
//
// Complexity:
//   - O(n*P) per independent oracle comparison.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Wrapper parity alone is insufficient when both paths share one incorrect kernel.
func TestComputeModeWrappers_Parity(t *testing.T) {
	negativeZero := math.Copysign(0, -1)
	marketHigh, marketLow := candleChannels(ibmDailyCandles())
	cases := []struct {
		name      string
		high, low []float64
	}{
		{"nil", nil, nil},
		{"empty", []float64{}, []float64{}},
		{"mixed-empty", nil, []float64{}},
		{"golden", []float64{2, 4, 3}, []float64{1, 2, 0}},
		{"signed-zero", []float64{0, negativeZero, 0}, []float64{negativeZero, 0, negativeZero}},
		{"negative", []float64{-1, -3, -2, -4}, []float64{-5, -6, -4, -7}},
		{"extreme", []float64{math.MaxFloat64, math.MaxFloat64}, []float64{math.MaxFloat64, -math.MaxFloat64}},
		{"subnormal", []float64{math.SmallestNonzeroFloat64, 0}, []float64{0, -math.SmallestNonzeroFloat64}},
		{"vendor-snapshot", marketHigh, marketLow},
	}
	for _, facade := range modeFacades() {
		for _, test := range cases {
			for _, period := range []int{1, 2, 10, 20, 50, math.MaxInt} {
				t.Run(fmt.Sprintf("%s/%s/P=%d", facade.name, test.name, period), func(t *testing.T) {
					opts := donchian.Options{Period: period, ExcludeCurrent: facade.exclude}
					want := mustCompute(t, test.high, test.low, opts)
					got, err := facade.call(context.Background(), test.high, test.low, period)
					if err != nil {
						t.Fatal(err)
					}
					assertResult(t, got, want)
					assertResult(t, got, exactOracle(test.high, test.low, opts))
				})
			}
		}
	}
}

// TestComputeModeWrappers_ErrorPrecedence checks the complete failure-class set of both wrappers.
// Secondary nil-input identity must survive alongside the primary shape category.
//
// Implementation:
//   - Stage 1: Build canceled/deadline contexts and overlapping metadata/domain failures.
//   - Stage 2: Call each wrapper with the independently specified expected category set.
//   - Stage 3: Require nil results, exact errors.Is membership, and unchanged input bits.
//
// Behavior highlights:
//   - Require nil results, exact errors.Is membership, and unchanged input bits.
//
// Inputs:
//   - metadata/domain conflicts and independently declared expected error categories.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Case order is fixed; contexts are canceled or already expired before the tested call.
//
// Complexity:
//   - O(n) per case including input snapshots.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Do not compare joined error object pointers or diagnostic strings.
func TestComputeModeWrappers_ErrorPrecedence(t *testing.T) {
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
	}{
		{"nil-context-first", nil, nil, []float64{math.NaN()}, 0, donchian.ErrNilContext},
		{"cancel-first", canceled, nil, []float64{math.NaN()}, 0, context.Canceled},
		{"deadline-first", expired, nil, nil, 0, context.DeadlineExceeded},
		{"zero-period", context.Background(), nil, []float64{math.NaN()}, 0, donchian.ErrInvalidOption},
		{"min-period", context.Background(), nil, nil, math.MinInt, donchian.ErrInvalidOption},
		{"shape-first", context.Background(), []float64{math.NaN()}, []float64{0, 0}, 1, donchian.ErrShape},
		{"nil-high", context.Background(), nil, []float64{1}, 1, errors.Join(donchian.ErrShape, donchian.ErrNilInput)},
		{"nil-low", context.Background(), []float64{1}, nil, 1, errors.Join(donchian.ErrShape, donchian.ErrNilInput)},
		{"empty-not-nil", context.Background(), []float64{}, []float64{1}, 1, donchian.ErrShape},
		{"global-finite-first", context.Background(), []float64{0, 2}, []float64{1, math.NaN()}, 1, donchian.ErrNonFinite},
		{"geometry", context.Background(), []float64{-5}, []float64{-1}, 1, donchian.ErrBar},
		{"masked-invalid", context.Background(), []float64{math.Inf(-1)}, []float64{0}, math.MaxInt, donchian.ErrNonFinite},
	}
	for _, facade := range modeFacades() {
		for _, test := range cases {
			t.Run(facade.name+"/"+test.name, func(t *testing.T) {
				highCopy, lowCopy := append([]float64(nil), test.high...), append([]float64(nil), test.low...)
				got, err := facade.call(test.ctx, test.high, test.low, test.period)
				if got != nil {
					t.Fatal("failure published a result")
				}
				assertErrorCategories(t, err, test.want)
				assertFloats(t, test.high, highCopy)
				assertFloats(t, test.low, lowCopy)
			})
		}
	}
}

// TestComputeModeWrappers_CancellationAtEveryPoll checks cancellation parity at every reached context observation.
// Wrappers must add neither polling calls nor opportunities to publish partial state.
//
// Implementation:
//   - Stage 1: Measure successful wrapper and canonical poll counts on empty, masked, and burst inputs.
//   - Stage 2: Rerun the wrapper with cancellation at each observed poll count.
//   - Stage 3: Require identical poll counts on success and nil plus context.Canceled at every interruption.
//
// Behavior highlights:
//   - Require identical poll counts on success and nil plus context.Canceled at every interruption.
//
// Inputs:
//   - empty, masked, upper-burst, and lower-burst histories.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - An atomic observation counter, not elapsed time, chooses each cancellation point.
//
// Complexity:
//   - O(q*n+n*P) for q polling positions.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - A wrapper-local context precheck would change observable detection order.
func TestComputeModeWrappers_CancellationAtEveryPoll(t *testing.T) {
	upperHigh, upperLow := burstBars(4100, true)
	lowerHigh, lowerLow := burstBars(4100, false)
	maskedHigh, maskedLow := syntheticBars(3073)
	cases := []struct {
		name      string
		high, low []float64
		period    int
	}{
		{"empty", nil, nil, 1},
		{"masked", maskedHigh, maskedLow, math.MaxInt},
		{"upper-burst", upperHigh, upperLow, 4099},
		{"lower-burst", lowerHigh, lowerLow, 4099},
	}
	for _, facade := range modeFacades() {
		for _, test := range cases {
			t.Run(facade.name+"/"+test.name, func(t *testing.T) {
				probe := newPollContext(0)
				got, err := facade.call(probe, test.high, test.low, test.period)
				polls := probe.calls.Load()
				probe.cancel()
				if err != nil {
					t.Fatal(err)
				}
				opts := donchian.Options{Period: test.period, ExcludeCurrent: facade.exclude}
				assertResult(t, got, exactOracle(test.high, test.low, opts))
				// Extra wrapper polls could change which concurrent error is discovered first.
				canonicalProbe := newPollContext(0)
				_, err = donchian.Compute(canonicalProbe, test.high, test.low, opts)
				canonicalPolls := canonicalProbe.calls.Load()
				canonicalProbe.cancel()
				if err != nil || canonicalPolls != polls {
					t.Fatalf("canonical polls=%d wrapper polls=%d error=%v", canonicalPolls, polls, err)
				}
				for stop := int64(1); stop <= polls; stop++ {
					ctx := newPollContext(stop)
					got, err = facade.call(ctx, test.high, test.low, test.period)
					ctx.cancel()
					if got != nil {
						t.Fatalf("poll %d published a partial result", stop)
					}
					assertErrorCategories(t, err, context.Canceled)
				}
			})
		}
	}
}

// TestComputeModeWrappers_Ownership checks detached ownership through each named mode.
// Neither wrapper may reuse an input buffer or a previously returned artifact.
//
// Implementation:
//   - Stage 1: Compute two independent results from the same caller-owned inputs.
//   - Stage 2: Mutate one result and then mutate the input slices after the calls finish.
//   - Stage 3: Check that the other result and the original input snapshots remain detached as appropriate.
//
// Behavior highlights:
//   - Check that the other result and the original input snapshots remain detached as appropriate.
//
// Inputs:
//   - independent calls and post-call mutations in both directions.
//
// Returns:
//   - No value; a violated assertion fails the owning test goroutine.
//
// Errors:
//   - Unexpected public errors fail the test; expected failures are classified with errors.Is.
//
// Determinism:
//   - Mutations occur sequentially after computation; no concurrent input mutation is assumed.
//
// Complexity:
//   - O(n*P) oracle work plus O(n) copying.
//
// Notes:
//   - The checks use only public results and error identities; no private state is inspected.
//
// AI-Hints:
//   - Keep ownership identical to Compute rather than adding a shortcut that returns aliases.
func TestComputeModeWrappers_Ownership(t *testing.T) {
	for _, facade := range modeFacades() {
		t.Run(facade.name, func(t *testing.T) {
			high, low := syntheticBars(31)
			highCopy, lowCopy := append([]float64(nil), high...), append([]float64(nil), low...)
			const period = 3
			first, err := facade.call(context.Background(), high, low, period)
			if err != nil {
				t.Fatal(err)
			}
			second, err := facade.call(context.Background(), high, low, period)
			if err != nil {
				t.Fatal(err)
			}
			want := exactOracle(high, low, donchian.Options{Period: period, ExcludeCurrent: facade.exclude})
			// Result mutation must not alter input data or a separately computed result.
			first.Upper[period], first.Middle[period], first.Lower[period], first.Valid[period] = 8, 7, 6, false
			assertResult(t, second, want)
			assertFloats(t, high, highCopy)
			assertFloats(t, low, lowCopy)
			// Input mutation after return must not retroactively alter the artifact.
			high[0], low[0] = math.MaxFloat64, -math.MaxFloat64
			assertResult(t, second, want)
		})
	}
}
