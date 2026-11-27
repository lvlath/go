// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap_test verifies binary64 decomposition and the full-width rounding boundary.
// This file exercises private arithmetic only through public Push, Compute, and Restore.
// AI-Hints: expected values come from identities or math/big, never production decomposition.
package vwap_test

import (
	"context"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/vwap"
)

// TestVWAP_Binary64ExponentDecompositionRegression covers every finite exponent field.
// A one-row positive-weight mean is exactly its price, even if its exact product over/underflows binary64.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: cover every finite exponent, three significands, and both signs with positive weights.
//   - Stage 3: require the one-row identity and independently checked exact sums at exponent boundaries.
//
// Behavior highlights:
//   - This test protects normal and subnormal decoding without narrowing exponent conversions without altering production visibility.
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
//   - Normal Frexp decomposition must not add an implicit bit to a subnormal.
func TestVWAP_Binary64ExponentDecompositionRegression(t *testing.T) {
	stream := newStream(t)
	// Three significands per exponent cover the lower endpoint, low retained bit,
	// and upper endpoint; both signs exercise negative normal/subnormal values.
	for exponent := uint64(0); exponent < 0x7ff; exponent++ {
		for _, fraction := range []uint64{0, 1, 1<<52 - 1} {
			positive := math.Float64frombits(exponent<<52 | fraction)
			volume := positive
			if volume == 0 {
				volume = math.SmallestNonzeroFloat64
			}
			for _, sign := range []float64{1, -1} {
				price := math.Copysign(positive, sign)
				expected := price
				if expected == 0 {
					expected = 0 // Exact zero, unlike negative nonzero underflow, is canonical +0.
				}
				requireValue(t, mustPush(t, stream, price, volume, true), vwap.Value{Value: expected, Valid: true})
				// Check the units independently at regular exponent boundaries as well.
				if exponent%127 == 0 || exponent == 2046 {
					wantVolume, wantWeighted := oracleSums([]float64{price}, []float64{volume})
					state := stream.Snapshot()
					if state.VolumeSum != wantVolume || state.WeightedSum != wantWeighted {
						t.Fatalf("wrong exact sums at exponent=%d fraction=%x sign=%g", exponent, fraction, sign)
					}
				}
			}
		}
	}
}

// TestSnapshot_DivisorNormalizationAndTieNeighborhood checks all divisor normalization distances.
// Structural snapshots supply exact rationals around half-way cases without exposing a test-only API.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: Restore independently constructed rational states across leading bits, scales, signs, and offsets.
//   - Stage 3: require exact rounded bits and unchanged sums after a zero-weight continuation.
//
// Behavior highlights:
//   - This test protects every divisor normalization shift and nearest-half neighborhood without altering production visibility.
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
//   - Do not expose a test-only private arithmetic facade to reach rounding boundaries.
func TestSnapshot_DivisorNormalizationAndTieNeighborhood(t *testing.T) {
	stream := newStream(t)
	for leading := uint(0); leading < 64; leading++ {
		for _, scale := range []uint{0, 1024, 2044} {
			for _, quotient := range []int64{0, 1, 1 << 52, 1<<53 - 1} {
				for _, offset := range []int64{-1, 0, 1} {
					for _, negative := range []bool{false, true} {
						state, expected := quotientStateFixture(leading, scale, quotient, offset, negative)
						if !validSnapshotOracle(state) {
							t.Fatal("structural rounding fixture violates the independent magnitude bounds")
						}
						if err := stream.Restore(state); err != nil {
							t.Fatal(err)
						}
						// A zero-volume read accepts a new row but must not change the cached mean.
						requireValue(t, mustPush(t, stream, 0, 0, false), vwap.Value{Value: expected, Valid: true})
						state.Count++
						if stream.Snapshot() != state {
							t.Fatal("zero-volume continuation altered exact sums or sign")
						}
					}
				}
			}
		}
	}
}

// TestStream_LastCountHeadroomKeepsExactWideSums exercises the final legal accumulation.
// The next ordinary Push must fail atomically, while an explicit reset still starts a fresh segment.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: Restore wide exact sums at MaxUint64-1 and add one opposite-sign extreme row.
//   - Stage 3: require exact final sums, atomic subsequent overflow, and successful explicit reset.
//
// Behavior highlights:
//   - This test protects the final legal carry-bearing accumulation before count exhaustion without altering production visibility.
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
//   - Count headroom and accumulator bit capacity are distinct proof obligations.
func TestStream_LastCountHeadroomKeepsExactWideSums(t *testing.T) {
	maximumUnits := units(math.MaxFloat64)
	volume := new(big.Int).Mul(new(big.Int).SetUint64(^uint64(0)-1), maximumUnits)
	weighted := new(big.Int).Mul(new(big.Int).Set(volume), maximumUnits)
	for _, negative := range []bool{false, true} {
		stream := newStream(t)
		numerator := new(big.Int).Set(weighted)
		price := -math.MaxFloat64
		if negative {
			numerator.Neg(numerator)
			price = math.MaxFloat64
		}
		state := snapshotFromInts(^uint64(0)-1, volume, numerator)
		if err := stream.Restore(state); err != nil {
			t.Fatal(err)
		}
		wantVolume := new(big.Int).Add(volume, maximumUnits)
		wantWeighted := new(big.Int).Add(numerator, new(big.Int).Mul(units(price), maximumUnits))
		denominator := new(big.Int).Lsh(new(big.Int).Set(wantVolume), 1074)
		expected, _ := new(big.Rat).SetFrac(wantWeighted, denominator).Float64()
		requireValue(t, mustPush(t, stream, price, math.MaxFloat64, false), vwap.Value{Value: expected, Valid: true})
		before := snapshotFromInts(^uint64(0), wantVolume, wantWeighted)
		if stream.Snapshot() != before {
			t.Fatal("last legal accumulation lost an integer carry")
		}
		value, err := stream.Push(context.Background(), 1, 0, false)
		requireError(t, err, vwap.ErrOverflow)
		requireValue(t, value, vwap.Value{})
		requireUnchanged(t, stream, before)
		requireValue(t, mustPush(t, stream, 1, 1, true), vwap.Value{Value: 1, Valid: true})
	}
}

// TestStream_SignedZeroCacheSurvivesRestart checks output bits, not float equality.
// A negative nonzero mean rounded to -0 remains valid and unchanged by zero-volume rows.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: form a tiny negative mean, Restore it, and submit both zero-volume signs.
//   - Stage 3: require unchanged -0 output bits until an explicit masked reset.
//
// Behavior highlights:
//   - This test protects valid negative underflow zero through restart and zero weights without altering production visibility.
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
//   - Float equality cannot distinguish the -0 cache from a forbidden sign erasure.
func TestStream_SignedZeroCacheSurvivesRestart(t *testing.T) {
	stream := newStream(t)
	mustPush(t, stream, -math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64, true)
	want := vwap.Value{Value: math.Copysign(0, -1), Valid: true}
	requireValue(t, mustPush(t, stream, 0, math.MaxFloat64, false), want)
	state := stream.Snapshot()
	restored := newStream(t)
	if err := restored.Restore(state); err != nil {
		t.Fatal(err)
	}
	for _, zero := range []float64{0, math.Copysign(0, -1)} {
		requireValue(t, mustPush(t, stream, 17, zero, false), want)
		requireValue(t, mustPush(t, restored, -17, zero, false), want)
		if stream.Snapshot() != restored.Snapshot() {
			t.Fatal("zero-volume prices changed exact state")
		}
	}
	requireValue(t, mustPush(t, restored, 17, 0, true), vwap.Value{})
}
