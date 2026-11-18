// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo_test provides reflection-free external contract fixtures.
// Helpers compare explicit sentinel, nil-state and numerical invariants.
package montecarlo_test

import (
	"context"
	"errors"
	"math"
	"testing"

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
func closeFloat(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12*math.Max(1, math.Abs(want)) {
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

// Compile-time identity check mirrors core.Nilable without implementation access.
var _ interface{ IsNil() bool } = (*montecarlo.Result)(nil)
