// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema_test checks that named facades select only their declared seed policy.
// Every expected measurement comes from the independent rational history oracle.
package ema_test

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/lvlath/go/ema"
)

// TestFacade_NamedPolicyParity checks all named facades against the exact oracle and canonical configuration.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Compare batch payloads/masks and stream state for each seed and period.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Empty, cancellation-prone, and mixed-exponent histories.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Wrappers select only Seed; they must not introduce a default Period.
func TestFacade_NamedPolicyParity(t *testing.T) {
	for _, facade := range namedFacades() {
		t.Run(facade.name, func(t *testing.T) {
			for _, period := range []int{1, 2, 3, 14, 64, math.MaxInt - 1} {
				for _, x := range [][]float64{nil, {}, {1e308, 1, -1e308, 0}, finiteFixture(80)} {
					opts := ema.Options{Period: period, Seed: facade.seed}
					want := oracleHistory(x, opts)
					got, err := facade.compute(context.Background(), x, period)
					if err != nil {
						t.Fatal(err)
					}
					assertResult(t, got, want)
					stream, err := facade.newStream(period)
					if err != nil {
						t.Fatal(err)
					}
					canonical := newStream(t, opts)
					if !sameSnapshot(stream.Snapshot(), canonical.Snapshot()) {
						t.Fatal("constructor wrapper changed the configuration")
					}
					for i, value := range x {
						assertValue(t, pushValue(t, stream, value), want[i])
					}
				}
			}
		})
	}
}

// TestFacade_ValidationPrecedence checks that named wrappers preserve canonical error categories and order.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Exercise invalid periods, nil/canceled/expired contexts, and nonfinite inputs.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Both named policies and deterministic already-expired deadlines.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - A convenience wrapper cannot bypass context or finite-input admission.
func TestFacade_ValidationPrecedence(t *testing.T) {
	active := context.Background()
	canceled, cancel := context.WithCancel(active)
	cancel()
	// A deadline in year one is unconditionally expired; no scheduler race is involved.
	expired, stop := context.WithDeadline(active, time.Time{})
	defer stop()
	for _, facade := range namedFacades() {
		for _, period := range []int{0, -1, math.MinInt, math.MaxInt} {
			stream, err := facade.newStream(period)
			if stream != nil || !errors.Is(err, ema.ErrInvalidOption) {
				t.Fatalf("constructor: %v", err)
			}
			for _, history := range [][]float64{nil, {}, {math.NaN()}} {
				result, callErr := facade.compute(active, history, period)
				if result != nil || !errors.Is(callErr, ema.ErrInvalidOption) {
					t.Fatalf("options: %v", callErr)
				}
			}
		}
		for _, tc := range []struct {
			ctx  context.Context
			want error
		}{
			{nil, ema.ErrNilContext}, {canceled, context.Canceled}, {expired, context.DeadlineExceeded},
		} {
			result, err := facade.compute(tc.ctx, []float64{math.NaN()}, 0)
			if result != nil || !errors.Is(err, tc.want) {
				t.Fatalf("context precedence: %v", err)
			}
		}
		for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			result, err := facade.compute(active, []float64{1, bad}, 14)
			if result != nil || !errors.Is(err, ema.ErrNonFinite) {
				t.Fatalf("input: %v", err)
			}
		}
	}
}

// TestFacade_CancellationAndOwnership checks wrapper publication and ownership behavior beyond numerical parity.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Cancel every checkpoint and mutate independently returned arrays.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Both wrappers on a multi-block scan and a detached short history.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Do not let a wrapper return only Values and silently discard readiness.
func TestFacade_CancellationAndOwnership(t *testing.T) {
	for _, facade := range namedFacades() {
		// Exercise every scheduled checkpoint on both linear batch traversals.
		for check := int32(1); check <= 9; check++ {
			ctx, cancel := cancelAfter(check)
			result, err := facade.compute(ctx, benchmarkHistory(2050), 14)
			cancel()
			if result != nil || !errors.Is(err, context.Canceled) {
				t.Fatalf("check=%d: %v", check, err)
			}
		}
		x := failoverHistory()
		want := oracleHistory(x, ema.Options{Period: 14, Seed: facade.seed})
		first, err := facade.compute(context.Background(), x, 14)
		if err != nil {
			t.Fatal(err)
		}
		second, err := facade.compute(context.Background(), x, 14)
		if err != nil {
			t.Fatal(err)
		}
		// Mutation of x and a separate result must not leak into second.
		x[0] = -99
		first.Values[0], first.Valid[0] = -99, !first.Valid[0]
		assertResult(t, second, want)
		var present nilable = second
		if present.IsNil() {
			t.Fatal("successful wrapper result is nil")
		}
	}
}

// TestFacade_RestoreDoesNotSwitchSeed checks immutable wrapper-selected configuration through restore and reset.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Reject a switched seed, compare unchanged state, then compare Reset with a fresh wrapper.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - SMA and First stream wrappers with identical periods.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Matching period alone does not establish snapshot compatibility.
func TestFacade_RestoreDoesNotSwitchSeed(t *testing.T) {
	for _, facade := range namedFacades() {
		stream, err := facade.newStream(14)
		if err != nil {
			t.Fatal(err)
		}
		pushValue(t, stream, 201.125)
		before := stream.Snapshot()
		changed := before
		if facade.seed == ema.SeedSMA {
			changed.Options.Seed = ema.SeedFirst
		} else {
			changed.Options.Seed = ema.SeedSMA
		}
		if err = stream.Restore(changed); !errors.Is(err, ema.ErrStateVersion) {
			t.Fatalf("seed switch: %v", err)
		}
		if !sameSnapshot(before, stream.Snapshot()) {
			t.Fatal("rejected configuration changed state")
		}
		stream.Reset()
		fresh, err := facade.newStream(14)
		if err != nil {
			t.Fatal(err)
		}
		if !sameSnapshot(stream.Snapshot(), fresh.Snapshot()) {
			t.Fatal("Reset changed wrapper policy")
		}
	}
}
