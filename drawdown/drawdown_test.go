// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File drawdown_test.go anchors public error, witness, ownership and cancellation laws.
package drawdown_test

import (
	"context"
	"math"
	"slices"
	"testing"
	"time"

	"github.com/lvlath/go/drawdown"
)

// TestDrawdown_TwoDrawdownsGolden anchors the deeper second episode against the original earliest equal peak.
func TestDrawdown_TwoDrawdownsGolden(t *testing.T) {
	mustGolden(t, []float64{100, 120, 90, 120, 80}, []float64{0, 0, .25, 0, 1.0 / 3}, 1, 4, -1)
}

// TestDrawdown_RecoveryGolden anchors equality-based recovery immediately after the trough.
func TestDrawdown_RecoveryGolden(t *testing.T) {
	mustGolden(t, []float64{100, 50, 100}, []float64{0, .5, 0}, 0, 1, 2)
}

// TestDrawdown_EqualPeaks keeps the first occurrence of a running maximum through plateaus.
func TestDrawdown_EqualPeaks(t *testing.T) {
	mustGolden(t, []float64{100, 100, 50, 100}, []float64{0, 0, .5, 0}, 0, 2, 3)
	mustGolden(t, []float64{50, 100, 100, 40, 101}, []float64{0, 0, 0, .6, 0}, 1, 3, 4)
}

// TestDrawdown_EqualMaxDepthsTie protects first-wins ordering for equal exact ratios across peaks.
func TestDrawdown_EqualMaxDepthsTie(t *testing.T) {
	mustGolden(t, []float64{100, 50, 100, 50, 100}, []float64{0, .5, 0, .5, 0}, 0, 1, 2)
	// Different peaks but exactly equal rational depths.
	mustGolden(t, []float64{100, 25, 200, 50, 200}, []float64{0, .75, 0, .75, 0}, 0, 1, 2)
	mustGolden(t, []float64{3, 1, 6, 2, 6}, []float64{0, 2.0 / 3, 0, 2.0 / 3, 0}, 0, 1, 2)
}

// TestDrawdown_NewDeeperTrough requires strict improvement to replace the full witness.
func TestDrawdown_NewDeeperTrough(t *testing.T) {
	mustGolden(t, []float64{100, 75, 100, 25, 100}, []float64{0, .25, 0, .75, 0}, 0, 3, 4)
	mustGolden(t, []float64{100, 75, 200, 50, 201}, []float64{0, .25, 0, .75, 0}, 2, 3, 4)
}

// TestDrawdown_NoDrawdown checks singleton, plateau and rising histories without an episode.
func TestDrawdown_NoDrawdown(t *testing.T) {
	mustGolden(t, []float64{1}, []float64{0}, 0, 0, 0)
	mustGolden(t, []float64{1, 1, 2, 3}, []float64{0, 0, 0, 0}, 0, 0, 0)
}

// TestDrawdown_TotalLoss distinguishes true zero capital from other severe losses and signed zero.
func TestDrawdown_TotalLoss(t *testing.T) {
	mustGolden(t, []float64{100, 0, 0, 100}, []float64{0, 1, 1, 0}, 0, 1, 3)
	mustGolden(t, []float64{100, math.Copysign(0, -1), 120, 0}, []float64{0, 1, 0, 1}, 0, 1, 2)
	mustGolden(t, []float64{100, 0, 120, 1}, []float64{0, 1, 0, 119.0 / 120}, 0, 1, 2)
	mustGolden(t, []float64{100, 1, 200, 0}, []float64{0, .99, 0, 1}, 2, 3, -1)
}

// TestDrawdown_MediumRecoveryScope locates recovery relative to the selected maximum, not a later episode.
func TestDrawdown_MediumRecoveryScope(t *testing.T) {
	// A later smaller, unrecovered episode must not replace the earlier recovery.
	mustGolden(t, []float64{100, 120, 60, 80, 110, 121, 130, 100, 125},
		[]float64{0, 0, .5, 1.0 / 3, 1.0 / 12, 0, 0, 3.0 / 13, 1.0 / 26}, 1, 2, 5)
	mustGolden(t, []float64{100, 90, 95, 80, 90, 99, 80}, []float64{0, .1, .05, .2, .1, .01, .2}, 0, 3, -1)
}

// TestDrawdown_InputValidationErrors checks all public invalid-equity classes, precedence and immutability.
func TestDrawdown_InputValidationErrors(t *testing.T) {
	negativeZero := math.Copysign(0, -1)
	cases := []struct {
		name   string
		equity []float64
		want   error
	}{
		{"nil", nil, drawdown.ErrInsufficient},
		{"empty", []float64{}, drawdown.ErrInsufficient},
		{"zero", []float64{0}, drawdown.ErrEquity},
		{"negative_zero", []float64{negativeZero}, drawdown.ErrEquity},
		{"negative_initial", []float64{-1}, drawdown.ErrEquity},
		{"negative_later", []float64{1, -1}, drawdown.ErrEquity},
		{"nan_initial", []float64{math.NaN()}, drawdown.ErrNonFinite},
		{"nan_later", []float64{1, math.NaN()}, drawdown.ErrNonFinite},
		{"positive_inf_initial", []float64{math.Inf(1)}, drawdown.ErrNonFinite},
		{"negative_inf_initial", []float64{math.Inf(-1)}, drawdown.ErrNonFinite},
		{"positive_inf_later", []float64{1, math.Inf(1)}, drawdown.ErrNonFinite},
		{"negative_inf_later", []float64{1, math.Inf(-1)}, drawdown.ErrNonFinite},
		{"initial_domain_before_later_nan", []float64{0, math.NaN()}, drawdown.ErrEquity},
		{"earliest_negative_before_later_nan", []float64{1, -1, math.NaN()}, drawdown.ErrEquity},
		{"earliest_nan_before_later_negative", []float64{1, math.NaN(), -1}, drawdown.ErrNonFinite},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			before := slices.Clone(tc.equity)
			got, err := drawdown.Compute(context.Background(), tc.equity)
			mustErrorIs(t, err, tc.want)
			mustNilState(t, got, true, "error result")
			mustEqualSlice(t, tc.equity, before)
		})
	}
}

// TestDrawdown_ContextPrecedence establishes nil/canceled/deadline admission before data access.
func TestDrawdown_ContextPrecedence(t *testing.T) {
	var nilContext context.Context
	got, err := drawdown.Compute(nilContext, nil)
	mustErrorIs(t, err, drawdown.ErrNilContext)
	mustNilState(t, got, true, "nil context")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	got, err = drawdown.Compute(ctx, []float64{0, math.NaN()})
	mustErrorIs(t, err, context.Canceled)
	mustNilState(t, got, true, "canceled context")
	ctx, cancel = context.WithDeadline(context.Background(), time.Unix(0, 0))
	defer cancel()
	got, err = drawdown.Compute(ctx, nil)
	mustErrorIs(t, err, context.DeadlineExceeded)
	mustNilState(t, got, true, "deadline context")
}

// TestDrawdown_CancellationBoundaries cancels deterministically at scan and publication checkpoints.
func TestDrawdown_CancellationBoundaries(t *testing.T) {
	// Initial poll, four first-pass stride polls, four recovery polls,
	// then publication. First exact trough is index 1 and never recovers.
	equity := make([]float64, 4097)
	for i := range equity {
		equity[i] = 75
	}
	equity[0], equity[1] = 100, 25
	before := slices.Clone(equity)
	for _, tc := range []struct {
		name     string
		cancelAt int
		input    []float64
	}{
		{"depth_pass", 2, equity},
		{"last_depth_stride", 5, equity},
		{"recovery_start", 6, equity},
		{"recovery_stride", 7, equity},
		{"publication", 10, equity},
		{"publication_without_recovery", 2, []float64{1, 2}},
		{"publication_with_empty_recovery_suffix", 2, []float64{2, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := newPollContext(tc.cancelAt)
			got, err := drawdown.Compute(ctx, tc.input)
			mustErrorIs(t, err, context.Canceled)
			mustNilState(t, got, true, "canceled result")
			mustEqualInt(t, ctx.polls, tc.cancelAt, "polls")
		})
	}
	mustEqualSlice(t, equity, before)
}

// TestDrawdown_Ownership proves returned profiles do not alias input or one another.
func TestDrawdown_Ownership(t *testing.T) {
	equity := []float64{100, 70, 80, 101}
	before := slices.Clone(equity)
	first, second := mustCompute(t, equity), mustCompute(t, equity)
	mustResult(t, first, second)
	first.Depth[1] = 123
	mustEqualBits(t, second.Depth[1], .3, "independent results")
	mustEqualSlice(t, equity, before)
	equity[1] = 1
	mustEqualBits(t, second.Depth[1], .3, "input detached")
}

// TestResult_IsNilInterface checks typed nil, ordinary nil, zero-value and computed results.
func TestResult_IsNilInterface(t *testing.T) {
	var result *drawdown.Result
	var nilable interface{ IsNil() bool } = result
	mustEqualBool(t, nilable.IsNil(), true, "nil receiver")
	mustNilState(t, nil, true, "nil interface")
	mustNilState(t, &drawdown.Result{}, false, "zero result")
	mustNilState(t, mustCompute(t, []float64{1}), false, "computed result")
}

// TestDrawdown_LateInvalidObservation verifies that even a recovered witness
// does not authorize early success: every later observation must be admitted.
func TestDrawdown_LateInvalidObservation(t *testing.T) {
	const history = 4098
	equity := make([]float64, history)
	for i := range equity {
		equity[i] = 100
	}
	equity[1] = 0 // A mathematically maximal, already recovered drawdown.
	for _, value := range []float64{-1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		equity[history-1] = value
		before := slices.Clone(equity)
		got, err := drawdown.Compute(context.Background(), equity)
		mustErrorIs(t, err, oracleError(equity))
		mustNilState(t, got, true, "late invalid result")
		mustEqualSlice(t, equity, before)
	}
}

// TestDrawdown_ConcurrentReadOnlyHistory exercises independent ownership under
// concurrent calls; only the test goroutine performs fatal assertions.
func TestDrawdown_ConcurrentReadOnlyHistory(t *testing.T) {
	const workers = 8
	equity := backtestFixture()
	before := slices.Clone(equity)
	expected := mustCompute(t, equity)
	type outcome struct {
		result *drawdown.Result
		err    error
	}
	outcomes := make(chan outcome, workers)
	for i := 0; i < workers; i++ {
		go func() {
			result, err := drawdown.Compute(context.Background(), equity)
			outcomes <- outcome{result, err}
		}()
	}
	results := make([]*drawdown.Result, 0, workers)
	for i := 0; i < workers; i++ {
		got := <-outcomes
		mustErrorIs(t, got.err, nil)
		mustResult(t, got.result, expected)
		results = append(results, got.result)
	}
	results[0].Depth[expected.TroughIndex] = 0
	for _, result := range results[1:] {
		mustResult(t, result, expected)
	}
	mustEqualSlice(t, equity, before)
}
