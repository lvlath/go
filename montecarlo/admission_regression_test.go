// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo_test defends public experiment admission and resource arithmetic.
// These tests distinguish logical path counts from materialized native allocations.
package montecarlo_test

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"testing"

	"github.com/lvlath/go/montecarlo"
	"github.com/lvlath/go/rng"
)

// TestAdmissionValidationPrecedence exercises the missing public validation exits.
// Invalid context and plan must win before sampler presence or dimension access;
// a nonpositive dimension must win before memory sizing.
// Implementation: table-driven public Run calls with a side-effect-counting sampler.
// Determinism: no path is admitted and no timing assumption is required.
// Complexity: O(q) per case; hostile dimensions are never allocated.
// AI-Hints: assert absence and invocation counts in addition to errors.Is.
func TestAdmissionValidationPrecedence(t *testing.T) {
	ctx := context.Background()
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	valid := montecarlo.Plan{Paths: 2}
	for _, tc := range []struct {
		name           string
		ctx            context.Context
		plan           montecarlo.Plan
		dimension      int
		dimensionCalls int64
		want           error
	}{
		{"nil-context-before-invalid-plan", nil, montecarlo.Plan{}, -1, 0, montecarlo.ErrInvalidOption},
		{"canceled-before-invalid-plan", canceled, montecarlo.Plan{}, -1, 0, context.Canceled},
		{"negative-workers", ctx, montecarlo.Plan{Paths: 2, Workers: -1}, 1, 0, montecarlo.ErrInvalidOption},
		{"negative-quantile", ctx, montecarlo.Plan{Paths: 2, Quantiles: []float64{-.1}}, 1, 0, montecarlo.ErrInvalidOption},
		{"infinite-quantile", ctx, montecarlo.Plan{Paths: 2, Quantiles: []float64{math.Inf(1)}}, 1, 0, montecarlo.ErrInvalidOption},
		{"zero-dimension", ctx, valid, 0, 1, montecarlo.ErrInvalidOption},
		{"negative-dimension", ctx, valid, -1, 1, montecarlo.ErrInvalidOption},
		{"dimension-before-budget", ctx, montecarlo.Plan{Paths: math.MaxUint64, RetainSamples: true, MaxSamplesBytes: 1}, -1, 1, montecarlo.ErrInvalidOption},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mustAdmissionFailure(t, tc.ctx, tc.plan, tc.dimension, tc.dimensionCalls, tc.want)
		})
	}
	// A genuine nil interface is distinct from a typed-nil sampler pointer.
	result, err := montecarlo.Run(ctx, valid, nil)
	mustErrorIs(t, err, montecarlo.ErrNilInput)
	mustNilState(t, result, true, "nil sampler result")
	result, err = montecarlo.Run(ctx, montecarlo.Plan{Paths: 1}, nil)
	mustErrorIs(t, err, montecarlo.ErrInvalidOption)
	mustNilState(t, result, true, "invalid plan before nil sampler")
}

// TestAdmissionNativeBackingSize rejects independently impossible native buffers.
// Each case isolates a different arithmetic constraint without attempting a huge make.
// Implementation: derive boundaries from math.MaxInt, not an assumed 64-bit platform.
// Determinism: ErrOverflow precedes the user budget and Sample is never called.
// Complexity: O(1) arithmetic plus the fixed small level list.
// AI-Hints: fitting an element count in int does not prove its byte storage fits.
func TestAdmissionNativeBackingSize(t *testing.T) {
	const float64Bytes = 8
	const unlimited uint64 = math.MaxUint64
	nativeMax := uint64(math.MaxInt)
	for _, tc := range []struct {
		name      string
		plan      montecarlo.Plan
		dimension int
	}{
		{"materialized-row-count", montecarlo.Plan{Paths: nativeMax + 1, RetainSamples: true, MaxSamplesBytes: unlimited}, 1},
		{"raw-element-product", montecarlo.Plan{Paths: nativeMax, RetainSamples: true, MaxSamplesBytes: unlimited}, 3},
		{"raw-byte-product", montecarlo.Plan{Paths: nativeMax/float64Bytes + 1, RetainSamples: true, MaxSamplesBytes: unlimited}, 1},
		{"quantile-only-raw-byte-product", montecarlo.Plan{Paths: nativeMax/float64Bytes + 1, Quantiles: []float64{.5}, MaxSamplesBytes: unlimited}, 1},
		// 2*d*8 fits; 4*d*8 does not. No large level slice is needed.
		{"quantile-output-byte-product", montecarlo.Plan{Paths: 2, Quantiles: []float64{0, .25, .75, 1}, MaxSamplesBytes: unlimited}, math.MaxInt / (3 * float64Bytes)},
		{"streaming-feature-vector", montecarlo.Plan{Paths: 2}, math.MaxInt/float64Bytes + 1},
		{"worker-window", montecarlo.Plan{Paths: math.MaxUint64, Workers: math.MaxInt}, 1},
		// 2*Workers fits int, but the channel payload is not addressable.
		{"worker-outcome-buffer", montecarlo.Plan{Paths: math.MaxUint64, Workers: math.MaxInt / 2}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mustAdmissionFailure(t, context.Background(), tc.plan, tc.dimension, 1, montecarlo.ErrOverflow)
		})
	}
}

// TestAdmissionExactBudgetAcrossModes verifies exact optional-buffer accounting.
// Retention and quantiles must share the raw owner; scratch is one column only.
// Implementation: test one byte below each independent analytic byte total,
// then verify that equality admits a complete experiment with the requested artifacts.
// Complexity: O(Paths*d) plus sorting for the tiny complete fixtures.
// AI-Hints: never infer the budget from the implementation's private helpers.
func TestAdmissionExactBudgetAcrossModes(t *testing.T) {
	const paths = 4
	const dimension = 2
	const float64Bytes = 8
	levels := []float64{0, .5, 1}
	raw := uint64(paths * dimension * float64Bytes)
	quantileCost := uint64((paths + len(levels)*dimension) * float64Bytes)
	for _, tc := range []struct {
		name   string
		retain bool
		levels []float64
		bytes  uint64
	}{
		{"retain-only", true, nil, raw},
		{"quantiles-only", false, levels, raw + quantileCost},
		{"retain-and-quantiles", true, levels, raw + quantileCost},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan := montecarlo.Plan{Paths: paths, RetainSamples: tc.retain, Quantiles: tc.levels, MaxSamplesBytes: tc.bytes - 1}
			mustAdmissionFailure(t, context.Background(), plan, dimension, 1, montecarlo.ErrBudget)
			plan.MaxSamplesBytes = tc.bytes
			result, err := montecarlo.Run(context.Background(), plan, oracleSampler())
			if err != nil || result == nil || !result.Complete || result.Completed != paths {
				t.Fatalf("exact budget must succeed: result=%+v err=%v", result, err)
			}
			mustNilState(t, result.Samples, !tc.retain, "retained samples")
			mustNilState(t, result.Quantiles, len(tc.levels) == 0, "quantile output")
			closeFloat(t, result.Mean[0], 1.5)
		})
	}
	// A one-byte optional-storage ceiling must not reject a streaming-only run.
	result, err := montecarlo.Run(context.Background(), montecarlo.Plan{Paths: paths, MaxSamplesBytes: 1}, oracleSampler())
	if err != nil || result == nil || !result.Complete || result.Samples != nil || result.Quantiles != nil {
		t.Fatalf("streaming mode consumed an optional buffer budget: %+v %v", result, err)
	}
}

// TestAdmissionUnsignedPathsAndWorkerClamp preserves the complete uint64 path domain.
// A streaming-only plan with MaxUint64 requested paths can fail immediately at
// path zero without allocating a path matrix or narrowing the logical counter.
// Determinism: every started path returns the same error; at most 2W may be issued.
// Complexity: O(d+W*d) memory, bounded work by the first observed terminal error.
func TestAdmissionUnsignedPathsAndWorkerClamp(t *testing.T) {
	cause := errors.New("model rejected the first trial")
	var calls atomic.Int64
	sampler := &funcSampler{1, func(_ context.Context, _ uint64, _ rng.RNG) ([]float64, error) {
		calls.Add(1)
		return nil, cause
	}}
	result, err := montecarlo.Run(context.Background(), montecarlo.Plan{Paths: math.MaxUint64, Workers: 1, MaxSamplesBytes: 1}, sampler)
	mustErrorIs(t, err, cause)
	if result == nil || result.Requested != math.MaxUint64 || result.Completed != 0 || result.Complete || result.Mean != nil || result.Samples != nil {
		t.Fatalf("wrong unsigned-path partial result: %+v", result)
	}
	if count := calls.Load(); count < 1 || count > 2 {
		t.Fatalf("unexpected in-flight work: %d", count)
	}
	// The worker clamp must occur before rejecting an oversized dispatch window.
	result, err = montecarlo.Run(context.Background(), montecarlo.Plan{Paths: 2, Workers: math.MaxInt}, oracleSampler())
	if err != nil || result == nil || !result.Complete || result.Completed != 2 {
		t.Fatalf("worker clamp failed: %+v %v", result, err)
	}
}

// TestAdmissionAggregateOverflowAfterNativeBuffers verifies the total-byte
// overflow independently of per-buffer limits. Three rows and three levels
// make both the raw and output buffers individually native-addressable.
// Implementation: use a dimension derived from the target architecture and
// a one-byte ceiling; no storage or sampler call is admitted on either word size.
// Behavior highlights: on 64-bit the aggregate exceeds MaxUint64; on 32-bit
// the aggregate is representable in uint64, so the configured budget rejects it.
// Determinism: exact unsigned arithmetic and fixed error precedence.
// Complexity: O(1) size checks, O(1) fixture storage.
// AI-Hints: separately test native allocation arithmetic and aggregate totals.
func TestAdmissionAggregateOverflowAfterNativeBuffers(t *testing.T) {
	const paths = 3
	const float64Bytes = 8
	dimension := math.MaxInt / (paths * float64Bytes)
	plan := montecarlo.Plan{
		Paths: paths, Quantiles: []float64{0, .5, 1}, MaxSamplesBytes: 1,
	}
	// On 64-bit, MaxInt mod 24 is 7, so raw=output=MaxInt-7 and
	// scratch=24. The total 2*MaxInt+10 exceeds MaxUint64.
	want := montecarlo.ErrBudget
	if uint64(math.MaxInt) == math.MaxUint64/2 {
		want = montecarlo.ErrOverflow
	}
	mustAdmissionFailure(t, context.Background(), plan, dimension, 1, want)
}
