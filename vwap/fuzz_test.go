// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap_test exercises public batch/stream and checkpoint boundaries.
// This file compares valid inputs to exact rational arithmetic; no recovery hides failures.
package vwap_test

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	"github.com/lvlath/go/vwap"
)

// FuzzVWAP_PublicBoundaries fuzzes the public batch, stream, and restart surfaces together.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: decode bounded byte rows into arbitrary float encodings and explicit reset flags.
//   - Stage 3: classify invalid inputs independently or compare valid prefixes to big.Rat and round-trip checkpoints.
//
// Behavior highlights:
//   - This test fuzzes the public batch, stream, and restart surfaces together without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - No recovery should hide a panic; retain failing inputs rather than weakening the oracle.
func FuzzVWAP_PublicBoundaries(f *testing.F) {
	f.Add([]byte{})
	f.Add(make([]byte, 34))
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1})
	f.Fuzz(func(t *testing.T, data []byte) {
		n := min(len(data)/17, 32)
		p, v, r := make([]float64, n), make([]float64, n), make([]bool, n)
		var want error
		for i := 0; i < n; i++ {
			row := data[i*17 : (i+1)*17]
			p[i] = math.Float64frombits(binary.LittleEndian.Uint64(row[:8]))
			v[i] = math.Float64frombits(binary.LittleEndian.Uint64(row[8:16]))
			r[i] = row[16]&1 != 0
			if i == 0 && !r[0] {
				want = vwap.ErrAnchor
			}
			if want == nil {
				if !finite(p[i]) || !finite(v[i]) {
					want = vwap.ErrNonFinite
				} else if v[i] < 0 {
					want = vwap.ErrVolume
				}
			}
		}
		result, err := vwap.Compute(context.Background(), p, v, r, vwap.Options{})
		if want != nil {
			requireError(t, err, want)
			if result != nil {
				t.Fatal("fuzz error returned partial result")
			}
		} else {
			if err != nil {
				t.Fatal(err)
			}
			verifyParity(t, p, v, r)
		}
		stream := newStream(t)
		for i := 0; i < n; i++ {
			before := stream.Snapshot()
			value, pushErr := stream.Push(context.Background(), p[i], v[i], r[i])
			if pushErr != nil {
				requireValue(t, value, vwap.Value{})
				requireUnchanged(t, stream, before)
			} else {
				clone := newStream(t)
				if restoreErr := clone.Restore(stream.Snapshot()); restoreErr != nil {
					t.Fatal(restoreErr)
				}
				if clone.Snapshot() != stream.Snapshot() {
					t.Fatal("fuzz restart changed state")
				}
			}
		}
	})
}

// FuzzSnapshot_PublicBoundary fuzzes schema, ASCII sums, phase, and exact snapshot bounds.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: construct public snapshots from bounded fuzz fields and compare independent structural admission.
//   - Stage 3: require atomic rejection or an unchanged canonical accepted snapshot.
//
// Behavior highlights:
//   - This test fuzzes schema, ASCII sums, phase, and exact snapshot bounds without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Do not reuse the production parser inside the independent acceptance oracle.
func FuzzSnapshot_PublicBoundary(f *testing.F) {
	f.Add("0", "0", uint64(0), false, uint32(1), "vwap/exact-weighted/v1")
	f.Add("1", "-1", uint64(1), true, uint32(1), "vwap/exact-weighted/v1")
	f.Add("-0", "+1", uint64(0), true, uint32(0), "")
	f.Fuzz(func(t *testing.T, volume, weighted string, count uint64, started bool, version uint32, algorithm string) {
		// Oversized strings are legitimate rejection probes, but cap the fuzz worker's own work.
		if len(volume) > 2000 || len(weighted) > 2000 || len(algorithm) > 2000 {
			return
		}
		stream := newStream(t)
		mustPush(t, stream, -7, 3, true)
		before := stream.Snapshot()
		state := vwap.Snapshot{Version: version, Algorithm: algorithm, Count: count, Started: started, VolumeSum: volume, WeightedSum: weighted}
		err := stream.Restore(state)
		expectedValid := validSnapshotOracle(state)
		if (err == nil) != expectedValid {
			t.Fatalf("Restore classification differs from independent state oracle: error=%v valid=%t", err, expectedValid)
		}
		if err != nil {
			requireError(t, err, vwap.ErrStateVersion)
			requireUnchanged(t, stream, before)
		} else if stream.Snapshot() != state {
			t.Fatal("accepted checkpoint was not canonical")
		}
	})
}
