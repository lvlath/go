// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo_test tests adversarial contract boundaries through public APIs.
// These independent oracles defend prefix, error, memory and numerical semantics.
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

// TestSESubnormal_NoPrematureUnderflow anchors the requirement that a tiny but
// representable sample variance can still yield a nonzero standard error.
func TestSESubnormal_NoPrematureUnderflow(t *testing.T) {
	// For [2^-536,0,0,0], sample variance is exactly 2^-1074.
	value := math.Ldexp(1, -536)
	sampler := &funcSampler{1, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if id == 0 {
			return []float64{value}, nil
		}
		return []float64{0}, nil
	}}
	result, err := montecarlo.Run(context.Background(), montecarlo.Plan{Paths: 4}, sampler)
	if err != nil {
		t.Fatal(err)
	}
	if result.SampleVariance[0] != math.SmallestNonzeroFloat64 {
		t.Fatalf("sample variance %.17g, want smallest subnormal", result.SampleVariance[0])
	}
	want := math.Sqrt(math.SmallestNonzeroFloat64) / 2
	if result.MeanStandardError[0] != want || result.MeanStandardError[0] == 0 {
		t.Fatalf("SE underflow: got %.17g want %.17g", result.MeanStandardError[0], want)
	}
}

// TestSummaryAtomicOnLateCoordinateOverflow ensures no partial mean/variance
// vectors leak when a high-index dimension has unrepresentable variance.
func TestSummaryAtomicOnLateCoordinateOverflow(t *testing.T) {
	sampler := &funcSampler{2, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if id == 0 {
			return []float64{1, -math.MaxFloat64}, nil
		}
		return []float64{3, math.MaxFloat64}, nil
	}}
	result, err := montecarlo.Run(context.Background(), montecarlo.Plan{
		Paths: 2, RetainSamples: true, Quantiles: []float64{.5},
	}, sampler)
	mustErrorIs(t, err, montecarlo.ErrOverflow)
	if result == nil || result.Complete || result.Completed != 2 {
		t.Fatalf("failed numeric finalization must preserve the truthful full prefix: %+v", result)
	}
	if result.Mean != nil || result.SampleVariance != nil || result.MeanStandardError != nil {
		t.Fatal("partially initialized moment vectors leaked after summary overflow")
	}
	if result.Quantiles != nil {
		t.Fatal("quantiles must not be published after a failed summary")
	}
	if result.Samples == nil || result.Samples.Rows() != 2 || result.Samples.Cols() != 2 {
		t.Fatal("valid retained rows must remain accessible with the error result")
	}
}

// TestSamplerReusedBufferDetached forces serial reuse of one mutable result
// slice, a common zero-allocation sampler optimization.
func TestSamplerReusedBufferDetached(t *testing.T) {
	reused := make([]float64, 2)
	sampler := &funcSampler{2, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		reused[0], reused[1] = float64(id), -float64(id)
		return reused, nil
	}}
	result, err := montecarlo.RunWithSamples(context.Background(), montecarlo.Plan{Paths: 64}, sampler)
	if err != nil {
		t.Fatal(err)
	}
	for row := 0; row < result.Samples.Rows(); row++ {
		left, _ := result.Samples.At(row, 0)
		right, _ := result.Samples.At(row, 1)
		if left != float64(row) || right != -float64(row) {
			t.Fatalf("sampler buffer was aliased at row %d: %.17g %.17g", row, left, right)
		}
	}
	reused[0] = 1e10
	value, _ := result.Samples.At(0, 0)
	if value != 0 {
		t.Fatal("published samples alias a sampler-owned buffer")
	}
}

// TestSignedQuantileInterpolationFinite probes type-7 quantiles with large
// opposing signs while retaining a mathematically representable variance.
func TestSignedQuantileInterpolationFinite(t *testing.T) {
	sampler := &funcSampler{1, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if id == 0 {
			return []float64{-1e150}, nil
		}
		return []float64{1e150}, nil
	}}
	result, err := montecarlo.RunWithQuantiles(context.Background(), montecarlo.Plan{
		Paths: 2, Workers: 2,
	}, []float64{0, .25, .5, .75, 1}, sampler)
	if err != nil {
		t.Fatal(err)
	}
	for row, expected := range []float64{-1e150, -5e149, 0, 5e149, 1e150} {
		value, readErr := result.Quantiles.At(row, 0)
		if readErr != nil {
			t.Fatal(readErr)
		}
		if math.Abs(value-expected) > 1e-15*math.Max(1, math.Abs(expected)) {
			t.Fatalf("signed quantile row %d: %.17g want %.17g", row, value, expected)
		}
	}
}

// TestOutOfOrderTerminalErrorDoesNotLeakLaterResults exercises a gap at ID1.
// ID0 cannot succeed once ID1 triggers child cancellation, so Completed stays zero.
func TestOutOfOrderTerminalErrorDoesNotLeakLaterResults(t *testing.T) {
	cause := errors.New("instrument failure")
	sampler := &funcSampler{1, func(ctx context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		switch id {
		case 0:
			<-ctx.Done()
			return nil, ctx.Err()
		case 1:
			return nil, cause
		default:
			return []float64{float64(id)}, nil
		}
	}}
	result, err := montecarlo.Run(context.Background(), montecarlo.Plan{
		Paths: 32, Workers: 4, RetainSamples: true, Quantiles: []float64{.5},
	}, sampler)
	mustErrorIs(t, err, cause)
	if result == nil || result.Complete || result.Completed != 0 {
		t.Fatalf("gap must hide all later success rows: %+v", result)
	}
	if result.Samples != nil || result.Quantiles != nil || result.Mean != nil ||
		result.SampleVariance != nil || result.MeanStandardError != nil {
		t.Fatal("zero prefix has non-nil statistical artifacts")
	}
}

// TestDimensionAndBudgetPreflightBeforeSample validates the exact public
// error precedence and prevents any invalid work from starting.
func TestDimensionAndBudgetPreflightBeforeSample(t *testing.T) {
	var dimensionCalls, sampleCalls atomic.Int64
	sampler := &countingSampler{
		dimension: func() int { dimensionCalls.Add(1); return 2 },
		sample: func(_ context.Context, _ uint64, _ rng.RNG) ([]float64, error) {
			sampleCalls.Add(1)
			return []float64{1, 1}, nil
		},
	}
	_, err := montecarlo.Run(context.Background(), montecarlo.Plan{
		Paths: 100, RetainSamples: true, MaxSamplesBytes: 10,
	}, sampler)
	mustErrorIs(t, err, montecarlo.ErrBudget)
	if dimensionCalls.Load() != 1 || sampleCalls.Load() != 0 {
		t.Fatal("budget must be checked after one Dimension and before any Sample")
	}
	_, err = montecarlo.Run(context.Background(), montecarlo.Plan{Paths: 1}, sampler)
	mustErrorIs(t, err, montecarlo.ErrInvalidOption)
	if dimensionCalls.Load() != 1 {
		t.Fatal("invalid plan must not query Dimension")
	}
}

// TestMemorySizeOverflowPrecedesSampling rejects impossible full matrices even
// with a nominally unlimited sample budget.
func TestMemorySizeOverflowPrecedesSampling(t *testing.T) {
	var called atomic.Int64
	sampler := &funcSampler{2, func(_ context.Context, _ uint64, _ rng.RNG) ([]float64, error) {
		called.Add(1)
		return []float64{1, 2}, nil
	}}
	for _, plan := range []montecarlo.Plan{
		{Paths: math.MaxUint64, RetainSamples: true, MaxSamplesBytes: math.MaxUint64},
		{Paths: math.MaxUint64, Quantiles: []float64{.5}, MaxSamplesBytes: math.MaxUint64},
	} {
		result, err := montecarlo.Run(context.Background(), plan, sampler)
		mustErrorIs(t, err, montecarlo.ErrOverflow)
		if result != nil || called.Load() != 0 {
			t.Fatal("integer overflow reached sampler or published a result")
		}
	}
}

// TestQuantileLevelsAndStatisticsDetached checks Plan and result slice ownership.
func TestQuantileLevelsAndStatisticsDetached(t *testing.T) {
	levels := []float64{.25, .5, .75}
	result, err := montecarlo.Run(context.Background(), montecarlo.Plan{
		Paths: 4, Quantiles: levels, RetainSamples: true,
	}, oracleSampler())
	if err != nil {
		t.Fatal(err)
	}
	levels[0] = .1
	if result.QuantileLevels[0] != .25 {
		t.Fatal("QuantileLevels aliases the caller's Plan")
	}
	result.QuantileLevels[0] = .2
	if levels[0] != .1 {
		t.Fatal("Result modifications escaped into caller input")
	}
	result.Mean[0] = 100
	result.SampleVariance[0] = 100
	result.MeanStandardError[0] = 100
	quantile, _ := result.Quantiles.At(0, 0)
	if quantile != .75 {
		t.Fatal("quantiles and moment vectors share writable storage")
	}
}

// TestConvenienceFacadesEquivalent confirms wrappers introduce no hidden
// alternative solver, seed, quantile semantics or result ownership.
func TestConvenienceFacadesEquivalent(t *testing.T) {
	ctx := context.Background()
	plan := montecarlo.Plan{Paths: 32, Seed: 7, Workers: 4}
	compare := func(name string, got, want *montecarlo.Result, gotErr, wantErr error) {
		t.Helper()
		if gotErr != nil || wantErr != nil {
			t.Fatalf("%s error: %v, %v", name, gotErr, wantErr)
		}
		if !got.Complete || !want.Complete || got.Completed != want.Completed || got.RNG != want.RNG {
			t.Fatalf("%s metadata mismatch", name)
		}
		for i := range want.Mean {
			if math.Float64bits(got.Mean[i]) != math.Float64bits(want.Mean[i]) ||
				math.Float64bits(got.SampleVariance[i]) != math.Float64bits(want.SampleVariance[i]) ||
				math.Float64bits(got.MeanStandardError[i]) != math.Float64bits(want.MeanStandardError[i]) {
				t.Fatalf("%s moment mismatch at dimension %d", name, i)
			}
		}
	}
	serial, se := montecarlo.Estimate(ctx, plan.Paths, plan.Seed, oracleSampler())
	serialReference, sr := montecarlo.Run(ctx, montecarlo.Plan{Paths: plan.Paths, Seed: plan.Seed}, oracleSampler())
	compare("Estimate", serial, serialReference, se, sr)

	parallel, pe := montecarlo.EstimateParallel(ctx, plan.Paths, plan.Seed, plan.Workers, oracleSampler())
	parallelReference, pr := montecarlo.Run(ctx, plan, oracleSampler())
	compare("EstimateParallel", parallel, parallelReference, pe, pr)

	withSamples, re := montecarlo.RunWithSamples(ctx, plan, oracleSampler())
	retained := plan
	retained.RetainSamples = true
	retainedReference, rr := montecarlo.Run(ctx, retained, oracleSampler())
	compare("RunWithSamples", withSamples, retainedReference, re, rr)
	if withSamples.Samples == nil || retainedReference.Samples == nil {
		t.Fatal("retention shortcut failed")
	}

	levels := []float64{.05, .5, .95}
	withQuantiles, qe := montecarlo.RunWithQuantiles(ctx, plan, levels, oracleSampler())
	quantilePlan := plan
	quantilePlan.Quantiles = levels
	quantileReference, qr := montecarlo.Run(ctx, quantilePlan, oracleSampler())
	compare("RunWithQuantiles", withQuantiles, quantileReference, qe, qr)
	if withQuantiles.Quantiles == nil || quantileReference.Quantiles == nil {
		t.Fatal("quantile shortcut failed")
	}
	for i := 0; i < len(levels); i++ {
		for j := 0; j < 2; j++ {
			a, _ := withQuantiles.Quantiles.At(i, j)
			b, _ := quantileReference.Quantiles.At(i, j)
			if math.Float64bits(a) != math.Float64bits(b) {
				t.Fatalf("quantile wrapper mismatch at (%d,%d)", i, j)
			}
		}
	}
}

// TestCancellationFirstPathHasNoMoments validates prefix publication at zero.
func TestCancellationFirstPathHasNoMoments(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{}, 1)
	sampler := &funcSampler{1, func(ctx context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if id == 0 {
			started <- struct{}{}
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return []float64{float64(id)}, nil
	}}
	outcome := make(chan struct {
		result *montecarlo.Result
		err    error
	}, 1)
	go func() {
		r, e := montecarlo.Run(ctx, montecarlo.Plan{Paths: 100, Workers: 1, RetainSamples: true}, sampler)
		outcome <- struct {
			result *montecarlo.Result
			err    error
		}{r, e}
	}()
	<-started
	cancel()
	got := <-outcome
	mustErrorIs(t, got.err, context.Canceled)
	if got.result == nil || got.result.Complete || got.result.Completed != 0 ||
		got.result.Samples != nil || got.result.Mean != nil {
		t.Fatalf("invalid zero prefix %+v", got.result)
	}
}

// TestPartialMatrixNilabilityThroughInterface ensures the unexported prefix
// backend obeys the public nilability and shape contracts via matrix.Matrix.
func TestPartialMatrixNilabilityThroughInterface(t *testing.T) {
	cause := errors.New("instrument fail at third trial")
	sampler := &funcSampler{2, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if id == 2 {
			return nil, cause
		}
		return []float64{float64(id), float64(id) + 100}, nil
	}}
	result, err := montecarlo.Run(context.Background(), montecarlo.Plan{
		Paths: 20, Workers: 1, RetainSamples: true,
	}, sampler)
	mustErrorIs(t, err, cause)
	if result.Samples.Rows() != 2 || result.Samples.Cols() != 2 {
		t.Fatal("prefix view reported allocated capacity instead of committed shape")
	}
	presence, ok := result.Samples.(interface{ IsNil() bool })
	if !ok || presence.IsNil() {
		t.Fatal("prefix view lost the optional nilability interface")
	}
	if err := result.Samples.Set(0, 1, 777); err != nil {
		t.Fatal(err)
	}
	value, err := result.Samples.At(0, 1)
	if err != nil || value != 777 {
		t.Fatalf("visible prefix is not mutable: %v %.17g", err, value)
	}
	if err := result.Samples.Set(0, 2, 9); err == nil {
		t.Fatal("prefix accepted a column beyond Dimension")
	}
}

// TestConstantMaximumWithQuantiles defends exact max-float storage, equal
// order-statistic interpolation, and zero-variance sample summaries together.
func TestConstantMaximumWithQuantiles(t *testing.T) {
	sampler := &funcSampler{1, func(_ context.Context, _ uint64, _ rng.RNG) ([]float64, error) {
		return []float64{math.MaxFloat64}, nil
	}}
	result, err := montecarlo.RunWithQuantiles(context.Background(), montecarlo.Plan{
		Paths: 6, Workers: 3,
	}, []float64{0, .25, .5, 1}, sampler)
	if err != nil {
		t.Fatal(err)
	}
	if result.Mean[0] != math.MaxFloat64 || result.SampleVariance[0] != 0 ||
		result.MeanStandardError[0] != 0 {
		t.Fatal("constant extreme value should not overflow mean or sample variance")
	}
	for row := 0; row < result.Quantiles.Rows(); row++ {
		value, err := result.Quantiles.At(row, 0)
		if err != nil || value != math.MaxFloat64 {
			t.Fatalf("constant quantile row %d: %.17g %v", row, value, err)
		}
	}
}

// countingSampler observes the validation boundary independently of Sample work.
type countingSampler struct {
	dimension func() int
	sample    func(context.Context, uint64, rng.RNG) ([]float64, error)
}

func (s *countingSampler) Dimension() int { return s.dimension() }
func (s *countingSampler) Sample(ctx context.Context, id uint64, r rng.RNG) ([]float64, error) {
	return s.sample(ctx, id, r)
}
