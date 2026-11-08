// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External test helpers compare observable series contracts, not private state.
// Each comparison reports the first failing index and preserves nil/empty laws.
package series_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/series"
)

// mustErrorIs verifies public sentinel identity, including wrapped errors.
// Error strings are deliberately excluded from the comparison protocol.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("error = %v, expected errors.Is(error, %v)", err, target)
	}
}

// assertTransform checks both length-preserving buffers and every paired cell.
// It also rejects nonfinite valid results and any nonzero or negative-zero warmup.
func assertTransform(t *testing.T, got *series.TransformResult, wantValues []float64, wantValid []bool) {
	t.Helper()
	if got == nil {
		t.Fatal("nil TransformResult on successful call")
	}
	if got.Values == nil || got.Valid == nil {
		t.Fatalf("nil result buffer: values=%v valid=%v", got.Values, got.Valid)
	}
	if len(wantValues) != len(wantValid) || len(got.Values) != len(wantValues) || len(got.Valid) != len(wantValid) {
		t.Fatalf("shape mismatch: values=%d valid=%d wantValues=%d wantValid=%d", len(got.Values), len(got.Valid), len(wantValues), len(wantValid))
	}
	for i := range wantValues {
		if got.Valid[i] != wantValid[i] {
			t.Fatalf("Valid[%d] = %v; want %v", i, got.Valid[i], wantValid[i])
		}
		if got.Values[i] != wantValues[i] || math.IsNaN(got.Values[i]) {
			t.Fatalf("Values[%d] = %g; want %g", i, got.Values[i], wantValues[i])
		}
		if !got.Valid[i] && (got.Values[i] != 0 || math.Signbit(got.Values[i])) {
			t.Fatalf("warmup[%d] must be positive zero, got %g", i, got.Values[i])
		}
		if got.Valid[i] && math.IsInf(got.Values[i], 0) {
			t.Fatalf("valid result[%d] must be finite, got %g", i, got.Values[i])
		}
	}
}

// assertGaps compares deterministic gap order, indices, counts, and nil state.
// An empty successful result must retain a non-nil slice.
func assertGaps(t *testing.T, got, want []series.Gap) {
	t.Helper()
	if got == nil || len(got) != len(want) {
		t.Fatalf("gap shape: got nil=%v len=%d, want len=%d", got == nil, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("gap[%d] = %+v; want %+v", i, got[i], want[i])
		}
	}
}

// assertRanges checks exact half-open boundaries and stable emission order.
// Empty successful results must have an allocated, non-nil slice header.
func assertRanges(t *testing.T, got, want []series.Range) {
	t.Helper()
	if got == nil || len(got) != len(want) {
		t.Fatalf("range shape: got nil=%v len=%d, want len=%d", got == nil, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("range[%d] = %+v; want %+v", i, got[i], want[i])
		}
	}
}
