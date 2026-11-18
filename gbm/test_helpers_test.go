// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm_test supplies black-box fake RNGs and semantic assertions.
// These helpers invoke only public contracts and never use reflection.
package gbm_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/gbm"
)

// scripted is an intentionally deterministic one-stream RNG implementation.
// Normal draws are consumed in index order; the optional hook coordinates
// cancellation without scheduling races, clocks, sleeps, or shared state.
type scripted struct {
	draws        []float64
	count        int
	rawCalls     int
	uniformCalls int
	onDraw       func(int)
}

// Uint64 records forbidden raw entropy consumption in deterministic GBM mode.
func (s *scripted) Uint64() uint64 {
	s.rawCalls++
	return 0
}

// UniformOpen records forbidden uniform entropy consumption in deterministic mode.
func (s *scripted) UniformOpen() float64 {
	s.uniformCalls++
	return 0.5
}

// Normal returns the next scripted finite/nonfinite draw, defaulting to zero.
// The hook executes before the sample is returned to support cancellation tests.
func (s *scripted) Normal() float64 {
	s.count++
	if s.onDraw != nil {
		s.onDraw(s.count)
	}
	if s.count <= len(s.draws) {
		return s.draws[s.count-1]
	}
	return 0
}

// IsNil makes typed-nil scripted streams discoverable without reflect.
func (s *scripted) IsNil() bool { return s == nil }

// mustErrorIs verifies sentinel identity through arbitrary wrapping.
func mustErrorIs(t *testing.T, err, expected error) {
	t.Helper()
	if !errors.Is(err, expected) {
		t.Fatalf("got error %v; expected errors.Is(..., %v)", err, expected)
	}
}

// isNilLike checks either nil interfaces or the supported optional Nilable ABI.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if nilable, ok := value.(interface{ IsNil() bool }); ok {
		return nilable.IsNil()
	}
	return false
}

// mustNilState checks the identity of optional result values without reflection.
func mustNilState(t *testing.T, value any, expected bool, operation string) {
	t.Helper()
	mustEqualBool(t, isNilLike(value), expected, operation)
}

// mustEqualBool compares contract flags with a failure message naming the rule.
func mustEqualBool(t *testing.T, got, expected bool, operation string) {
	t.Helper()
	if got != expected {
		t.Fatalf("%s: got %t; expected %t", operation, got, expected)
	}
}

// mustEqualInt compares counts, lengths, and index positions exactly.
func mustEqualInt(t *testing.T, got, expected int, operation string) {
	t.Helper()
	if got != expected {
		t.Fatalf("%s: got %d; expected %d", operation, got, expected)
	}
}

// mustNear verifies both finiteness and a relative-or-absolute error bound.
func mustNear(t *testing.T, got, expected, tolerance float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-expected) >
		tolerance*math.Max(1, math.Abs(expected)) {
		t.Fatalf("got %.17g; expected %.17g with tolerance %.3g", got, expected, tolerance)
	}
}

// mustPrefix asserts atomic publication of a nonempty failed trajectory prefix.
func mustPrefix(t *testing.T, result *gbm.Result, length int) {
	t.Helper()
	if result == nil || result.IsNil() || result.Complete {
		t.Fatalf("expected nonnil incomplete result; got %+v", result)
	}
	mustEqualInt(t, len(result.Times), length, "prefix Times length")
	mustEqualInt(t, len(result.Values), length, "prefix Values length")
	if length > 0 {
		mustNear(t, result.Times[0], 0, 0)
		for index, value := range result.Values {
			if !isPositiveFinite(value) {
				t.Fatalf("prefix Values[%d]=%g must be positive and finite", index, value)
			}
		}
	}
}

// isPositiveFinite separates valid small subnormals from zero/NaN/infinity.
func isPositiveFinite(value float64) bool {
	return value > 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}
