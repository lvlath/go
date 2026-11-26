// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema_test fuzzes public boundaries and exact arithmetic parity.
// All oracle helpers and deterministic input generation live in test_helper_test.go.
package ema_test

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/ema"
)

// FuzzEMA_RationalParity checks the public batch and stream boundaries against exact rational arithmetic.
// All production calls use exported ema methods from the external test package.
//
// Implementation:
//   - Stage 1: decode a bounded binary64 history and explicit seed/period choice.
//   - Stage 2: reject any nonfinite history through the public batch facade.
//   - Stage 3: compare every finite batch and Push result with an independent oracle.
//
// Behavior highlights:
//   - Malformed inputs are expected outcomes, not permission to mutate a live stream.
//
// Inputs:
//   - Byte-encoded observations, a raw period, and a raw seed; history is capped at 64 entries.
//
// Returns:
//   - No library artifact; mismatches fail the current fuzz case.
//
// Errors:
//   - Expected nonfinite/state errors use errors.Is; unexpected outcomes fail immediately.
//
// Determinism:
//   - Each fuzz case has deterministic replay and comparison behavior.
//
// Complexity:
//   - At most 64 observations: O(n*W) production work plus bounded exact-rational oracle arithmetic.
//   - Workspace includes the owning history/results and the independent rational state.
//
// Notes:
//   - Fuzz resource bounds limit oracle cost, not the public production input domain.
//
// AI-Hints:
//   - Do not use batch/stream agreement alone as a mathematical oracle.
func FuzzEMA_RationalParity(f *testing.F) {
	for _, values := range [][]float64{{1, 2, 3, 4}, {1e308, 1, -1e308}, {math.MaxFloat64, -math.MaxFloat64}, {math.SmallestNonzeroFloat64, 0}, {math.NaN()}} {
		data := make([]byte, 8*len(values))
		for i, value := range values {
			binary.LittleEndian.PutUint64(data[i*8:], math.Float64bits(value))
		}
		f.Add(data, uint16(3), uint8(0))
		f.Add(data, uint16(3), uint8(1))
	}
	f.Fuzz(func(t *testing.T, data []byte, rawPeriod uint16, rawSeed uint8) {
		count := len(data) / 8
		if count > 64 {
			count = 64
		}
		x := make([]float64, count)
		nonfinite := false
		for i := range x {
			x[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:]))
			nonfinite = nonfinite || math.IsNaN(x[i]) || math.IsInf(x[i], 0)
		}
		opts := ema.Options{Period: int(rawPeriod%65) + 1, Seed: ema.Seed(rawSeed % 2)}
		got, err := ema.Compute(context.Background(), x, opts)
		if nonfinite {
			if got != nil || !errors.Is(err, ema.ErrNonFinite) {
				t.Fatalf("nonfinite boundary: %v", err)
			}
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		assertResult(t, got, oracleHistory(x, opts))
		assertOracle(t, x, opts)
	})
}

// FuzzSnapshot_RestoreBoundary checks snapshot admission as well as rollback and deterministic continuation.
// All production calls use exported ema methods from the external test package.
//
// Implementation:
//   - Stage 1: edit untrusted count, scalar bits, decimal seed, and schema version.
//   - Stage 2: compare Restore acceptance with an independent bigint phase/grammar model.
//   - Stage 3: verify rejection rollback or accepted-state round-trip and continued equality.
//
// Behavior highlights:
//   - Malformed inputs are expected outcomes, not permission to mutate a live stream.
//
// Inputs:
//   - Arbitrary decimal text, uint64 count, float64 bits, and uint32 schema version.
//
// Returns:
//   - No library artifact; mismatches fail the current fuzz case.
//
// Errors:
//   - Expected nonfinite/state errors use errors.Is; unexpected outcomes fail immediately.
//
// Determinism:
//   - Each fuzz case has deterministic replay and comparison behavior.
//
// Complexity:
//   - O(D*W) parsing/formatting plus bounded bigint checks, D<=660 and W=34.
//   - Oversized text is rejected before any integer parsing by both the library and oracle.
//
// Notes:
//   - The separate oracle bounds text before bigint parsing; oversized strings cannot grow its integer workspace.
//
// AI-Hints:
//   - A canonical round-trip alone cannot prove that an invalid state was correctly rejected.
func FuzzSnapshot_RestoreBoundary(f *testing.F) {
	f.Add("0", uint64(0), uint64(0), uint32(1))
	f.Add("14", uint64(5), uint64(0), uint32(1))
	f.Add("-14", uint64(5), uint64(0), uint32(1))
	f.Add("01", uint64(5), uint64(0), uint32(1))
	f.Add("0", uint64(math.MaxUint64), math.Float64bits(1), uint32(1))
	f.Fuzz(func(t *testing.T, text string, count, raw uint64, version uint32) {
		stream := newStream(t, ema.Options{Period: 14})
		pushValue(t, stream, 42)
		before := stream.Snapshot()
		state := before
		state.SeedSum, state.Count, state.Current, state.Version = text, count, math.Float64frombits(raw), version
		// Compare admission against an independent parser and phase model.
		wantAccepted := snapshotAdmitted(state, before.Options)
		err := stream.Restore(state)
		if (err == nil) != wantAccepted {
			t.Fatalf("admission mismatch: accepted=%t want=%t state=%+v err=%v", err == nil, wantAccepted, state, err)
		}
		if err != nil {
			if !errors.Is(err, ema.ErrStateVersion) || !sameSnapshot(before, stream.Snapshot()) {
				t.Fatalf("bad rejection: %v", err)
			}
			return
		}
		if !sameSnapshot(state, stream.Snapshot()) {
			t.Fatal("accepted state not canonical")
		}
		clone := newStream(t, state.Options)
		if err = clone.Restore(state); err != nil {
			t.Fatal(err)
		}
		a, ea := stream.Push(context.Background(), -1)
		b, eb := clone.Push(context.Background(), -1)
		if !sameValue(a, b) || !errors.Is(ea, eb) || !sameSnapshot(stream.Snapshot(), clone.Snapshot()) {
			t.Fatal("continuation mismatch")
		}
	})
}
