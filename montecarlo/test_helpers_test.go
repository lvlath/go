// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo_test provides reflection-free external contract fixtures.
// Helpers compare explicit sentinel, nil-state and numerical invariants.
package montecarlo_test

import (
	"context"
	"errors"
	"math"
	"math/big"
	"sync/atomic"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/montecarlo"
	"github.com/lvlath/go/rng"
)

// funcSampler is a configurable external model fixture, not a driver backend.
// Its callbacks let each test choose a known analytic law or controlled error.
type funcSampler struct {
	dimension int
	sample    func(context.Context, uint64, rng.RNG) ([]float64, error)
}

// Dimension declares the fixture's fixed positive number of output coordinates.
// The driver must read this at most once on each validated Run call.
func (s *funcSampler) Dimension() int { return s.dimension }

// Sample delegates directly to the fixture's deterministic path model.
// Each callback owns its RNG only for that logical invocation.
func (s *funcSampler) Sample(ctx context.Context, id uint64, random rng.RNG) ([]float64, error) {
	return s.sample(ctx, id, random)
}

// IsNil implements the optional nilable convention for typed-nil sampler tests.
// It must not dereference its receiver or modify any model state.
func (s *funcSampler) IsNil() bool { return s == nil }

// oracleSampler returns the analytic vector [id,2*id] for every logical path.
// The known mean/variance and type-7 quantiles are independent of driver code.
func oracleSampler() *funcSampler {
	return &funcSampler{2, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		return []float64{float64(id), 2 * float64(id)}, nil
	}}
}

// mustErrorIs compares sentinel identity rather than diagnostic string content.
// Fatal assertions are invoked only on the test's owning goroutine.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("expected errors.Is(%v,%v)", err, target)
	}
}

// closeFloat checks a scale-relative numerical oracle without string equality.
// The tolerance permits routine final rounding but not order-of-magnitude drift.
// A nonfinite value must never pass because a NaN comparison evaluates false.
func closeFloat(t *testing.T, got, want float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) ||
		math.Abs(got-want) > 1e-12*math.Max(1, math.Abs(want)) {
		t.Fatalf("got %.17g want %.17g", got, want)
	}
}

// isNilLike tests interface absence and the safe core.Nilable-style contract.
// It intentionally never uses reflect to probe arbitrary typed-nil pointers.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if nilable, ok := value.(interface{ IsNil() bool }); ok {
		return nilable.IsNil()
	}
	return false
}

// mustNilState checks absence versus a present zero-valued result.
// Operation labels are diagnostics and are never used as error protocol.
func mustNilState(t *testing.T, value any, wantNil bool, operation string) {
	t.Helper()
	if got := isNilLike(value); got != wantNil {
		t.Fatalf("%s nil state: got %t want %t", operation, got, wantNil)
	}
}

// mustEqualBool checks a public state flag without converting it to prose.
// A diagnostic operation label identifies the invariant that failed.
func mustEqualBool(t *testing.T, got, want bool, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %t want %t", operation, got, want)
	}
}

// mustEqualInt checks an exact dimension or index-domain invariant.
// It is deliberately independent of floating-point tolerances.
func mustEqualInt(t *testing.T, got, want int, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d want %d", operation, got, want)
	}
}

// mustEqualString checks literal metadata rather than matching error messages.
// Errors continue to be classified exclusively through mustErrorIs.
func mustEqualString(t *testing.T, got, want, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %q want %q", operation, got, want)
	}
}

// admissionProbe observes validation stages without allocating a simulated vector.
// An unexpected Sample call returns a conspicuous error instead of allocating
// memory from a deliberately hostile dimension.
// Implementation: atomic counters keep the fixture safe even for an erroneous dispatch.
// Complexity: O(1) per method; no allocation proportional to the declared dimension.
type admissionProbe struct {
	dimension      int
	dimensionCalls atomic.Int64
	sampleCalls    atomic.Int64
}

// Dimension records the one permitted preflight dimension query.
// The declared dimension may intentionally be invalid or unaddressable.
func (s *admissionProbe) Dimension() int {
	s.dimensionCalls.Add(1)
	return s.dimension
}

// Sample records forbidden preflight work and returns a recognizable error.
// No path vector is allocated, even when Dimension is enormous.
func (s *admissionProbe) Sample(_ context.Context, _ uint64, _ rng.RNG) ([]float64, error) {
	s.sampleCalls.Add(1)
	return nil, errors.New("unexpected sample during preflight")
}

// mustAdmissionFailure verifies the entire preflight publication boundary.
// Both sentinel identity and the absence of worker side effects are required.
// Inputs: a plan, declared sampler dimension, expected dimension-query count and error.
// Returns: no value; a failed invariant stops only the calling test goroutine.
// Determinism: no timers or scheduling-dependent assertions.
// Complexity: O(q) level validation; no model-sized allocation on the valid refusal path.
// AI-Hints: a probe returning a huge dimension must never allocate that dimension itself.
// nolint:revive
func mustAdmissionFailure(t *testing.T, ctx context.Context, plan montecarlo.Plan,
	dimension int, dimensionCalls int64, target error,
) {
	t.Helper()
	probe := &admissionProbe{dimension: dimension}
	result, err := montecarlo.Run(ctx, plan, probe)
	mustErrorIs(t, err, target)
	mustNilState(t, result, true, "preflight result")
	if got := probe.dimensionCalls.Load(); got != dimensionCalls {
		t.Fatalf("Dimension calls=%d want %d", got, dimensionCalls)
	}
	if got := probe.sampleCalls.Load(); got != 0 {
		t.Fatalf("rejected plan invoked Sample %d times", got)
	}
}

// mustMatrixBits checks a published matrix cell against exact binary64 bits.
// It distinguishes signed zeros and subnormal rounding, unlike an absolute tolerance.
// Errors from the public matrix reader are never ignored.
func mustMatrixBits(t *testing.T, m matrix.Matrix, row, column int, want float64) {
	t.Helper()
	mustNilState(t, m, false, "matrix")
	got, err := m.At(row, column)
	if err != nil {
		t.Fatalf("At(%d,%d): %v", row, column, err)
	}
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("At(%d,%d)=%g (%016x), want %g (%016x)",
			row, column, got, math.Float64bits(got), want, math.Float64bits(want))
	}
}

// rationalSampleMoments is an independent exact-binary-input statistical oracle.
// It computes the arithmetic mean and centered sum of squares as rational values,
// rather than reproducing the driver's Welford recurrence.
// Implementation:
//   - Stage 1: convert each finite observation to an exact rational and sum it.
//   - Stage 2: center the original values using the exact mean and sum squares.
//   - Stage 3: divide by n-1 and round only the final mean and variance.
//
// Inputs: at least two finite binary64 observations.
// Returns: rounded mean and sample variance, for a representable oracle result.
// Determinism: exact rational arithmetic is independent of grouping and workers.
// Complexity: O(n) rational operations; arbitrary-precision storage is test-only.
// AI-Hints: do not use the production stats.Moments as its own correctness oracle.
func rationalSampleMoments(t *testing.T, values []float64) (float64, float64) {
	t.Helper()
	if len(values) < 2 {
		t.Fatal("oracle requires at least two observations")
	}
	sum := new(big.Rat)
	for _, value := range values {
		exact := new(big.Rat).SetFloat64(value)
		if exact == nil {
			t.Fatal("nonfinite oracle input")
		}
		sum.Add(sum, exact)
	}
	mean := new(big.Rat).Quo(sum, big.NewRat(int64(len(values)), 1))
	m2 := new(big.Rat)
	for _, value := range values {
		delta := new(big.Rat).Sub(new(big.Rat).SetFloat64(value), mean)
		m2.Add(m2, new(big.Rat).Mul(delta, delta))
	}
	variance := new(big.Rat).Quo(m2, big.NewRat(int64(len(values)-1), 1))
	meanFloat, _ := mean.Float64()
	varianceFloat, _ := variance.Float64()
	return meanFloat, varianceFloat
}

// mustRelativeFinite uses a true relative error for nonzero reference values.
// This avoids an absolute floor masking tiny but representable variances or SEs.
// A zero reference requires a genuine zero, and all nonfinite outputs fail.
func mustRelativeFinite(t *testing.T, got, want, tolerance float64, operation string) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) ||
		(want == 0 && got != 0) || (want != 0 && math.Abs((got-want)/want) > tolerance) {
		t.Fatalf("%s: got %.17g want %.17g (relative tolerance %g)", operation, got, want, tolerance)
	}
}

// Compile-time identity check mirrors core.Nilable without implementation access.
var _ interface{ IsNil() bool } = (*montecarlo.Result)(nil)
