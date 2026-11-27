// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap_test audits the exact checkpoint grammar and continuation bounds.
// This file generates inputs procedurally and verifies tamper rejection through the public API.
package vwap_test

import (
	"encoding/json"
	"math"
	"math/big"
	"strings"
	"testing"

	"github.com/lvlath/go/vwap"
)

// TestSnapshot_TamperAndRestore protects exact continuation at every split of a multi-phase history.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: JSON-round-trip each prefix snapshot and replace an existing destination state.
//   - Stage 3: require exact snapshots and oracle outputs after every resumed row.
//
// Behavior highlights:
//   - This test protects exact continuation at every split of a multi-phase history without altering production visibility.
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
//   - Persist exact sums and typed uint64 counts, not the rounded current mean alone.
func TestSnapshot_TamperAndRestore(t *testing.T) {
	fixture := restartFixture()
	p, v, r := fixture.price, fixture.volume, fixture.reset
	want := rationalOracle(p, v, r)
	for split := 0; split <= len(p); split++ {
		original := newStream(t)
		for i := 0; i < split; i++ {
			requireValue(t, mustPush(t, original, p[i], v[i], r[i]), want[i])
		}
		before := original.Snapshot()
		encoded, err := json.Marshal(before)
		if err != nil {
			t.Fatal(err)
		}
		var decoded vwap.Snapshot
		if err := json.Unmarshal(encoded, &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded != before {
			t.Fatal("JSON round trip lost checkpoint data")
		}
		restored := newStream(t)
		mustPush(t, restored, 999, 3, true)
		if err := restored.Restore(decoded); err != nil {
			t.Fatal(err)
		}
		if restored.Snapshot() != before {
			t.Fatal("snapshot did not round trip")
		}
		for i := split; i < len(p); i++ {
			requireValue(t, mustPush(t, restored, p[i], v[i], r[i]), want[i])
			requireValue(t, mustPush(t, original, p[i], v[i], r[i]), want[i])
			if restored.Snapshot() != original.Snapshot() {
				t.Fatal("restart exact state diverged")
			}
		}
	}
}

// TestSnapshot_ExactSumsAndNegativeCancellation checks serialized sums independently of the rounded output.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: construct extreme, canceling, and zero-weight segments.
//   - Stage 3: compare both decimal strings with unbounded integer oracle sums and Restore them.
//
// Behavior highlights:
//   - This test checks serialized sums independently of the rounded output without altering production visibility.
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
//   - Matching mean bits alone can conceal incorrect checkpoint numerators.
func TestSnapshot_ExactSumsAndNegativeCancellation(t *testing.T) {
	histories := []struct{ p, v []float64 }{
		{[]float64{1e308, 1, -1e308}, []float64{1, 1, 1}},
		{[]float64{-math.MaxFloat64, math.SmallestNonzeroFloat64}, []float64{math.MaxFloat64, math.SmallestNonzeroFloat64}},
		{[]float64{1, -1, 0}, []float64{1, 1, 1}},
		{[]float64{0, 0, 0}, []float64{0, 0, 0}},
	}
	for _, history := range histories {
		stream := newStream(t)
		for i := range history.p {
			mustPush(t, stream, history.p[i], history.v[i], i == 0)
		}
		state := stream.Snapshot()
		volume, weighted := oracleSums(history.p, history.v)
		if state.VolumeSum != volume || state.WeightedSum != weighted {
			t.Fatal("checkpoint sums differ from independent exact integer oracle")
		}
		restored := newStream(t)
		if err := restored.Restore(state); err != nil {
			t.Fatal(err)
		}
		if restored.Snapshot() != state {
			t.Fatal("exact sums changed after restoration")
		}
	}
}

// TestSnapshot_AllFieldsAndGrammar protects all mutable schema/phase fields and canonical ASCII grammar.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: tamper individual fields and submit malformed signed integer spellings.
//   - Stage 3: require ErrStateVersion and a completely unchanged destination.
//
// Behavior highlights:
//   - This test protects all mutable schema/phase fields and canonical ASCII grammar without altering production visibility.
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
//   - Options has one value, so do not invent a configuration mutation that cannot exist.
func TestSnapshot_AllFieldsAndGrammar(t *testing.T) {
	stream := newStream(t)
	mustPush(t, stream, -9, 3, true)
	before := stream.Snapshot()
	mutations := []struct {
		name   string
		mutate func(*vwap.Snapshot)
	}{
		{"version-zero", func(s *vwap.Snapshot) { s.Version = 0 }},
		{"version-two", func(s *vwap.Snapshot) { s.Version = 2 }},
		{"algorithm", func(s *vwap.Snapshot) { s.Algorithm = "vwap/exact-weighted/v2" }},
		{"algorithm-empty", func(s *vwap.Snapshot) { s.Algorithm = "" }},
		{"count-phase", func(s *vwap.Snapshot) { s.Count = 0 }},
		{"started-phase", func(s *vwap.Snapshot) { s.Started = false }},
		{"empty-phase-nonzero", func(s *vwap.Snapshot) { s.Count = 0; s.Started = false }},
		{"zero-volume-with-weight", func(s *vwap.Snapshot) { s.VolumeSum = "0" }},
	}
	for _, tc := range mutations {
		t.Run(tc.name, func(t *testing.T) {
			bad := before
			tc.mutate(&bad)
			requireError(t, stream.Restore(bad), vwap.ErrStateVersion)
			requireUnchanged(t, stream, before)
		})
	}
	malformed := []string{"", "+0", "+1", "-0", "00", "01", "-01", "-", "--1", " 1", "1 ", "1\n", "1.0", "1e3", "0x1", "1_000", "1\x00", "\xef\xbc\x91", "\xd9\xa1", strings.Repeat("9", 1301), strings.Repeat("9", 1300)}
	for _, text := range malformed {
		for field := 0; field < 2; field++ {
			bad := before
			if field == 0 {
				bad.VolumeSum = text
			} else {
				bad.WeightedSum = text
			}
			requireError(t, stream.Restore(bad), vwap.ErrStateVersion)
			requireUnchanged(t, stream, before)
		}
	}
	bad := before
	bad.VolumeSum = "-1"
	requireError(t, stream.Restore(bad), vwap.ErrStateVersion)
	requireUnchanged(t, stream, before)
	// Options has no fields: there is no distinct value available for a tamper test.
}

// TestSnapshot_BitLimitsAndMagnitudeInequalities protects exact snapshot capacity and cross-field bounds.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: test maximum admissible sums and violations one integer unit beyond each bound.
//   - Stage 3: accept legal extremes and atomically reject oversized or contradictory states.
//
// Behavior highlights:
//   - This test protects exact snapshot capacity and cross-field bounds without altering production visibility.
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
//   - String length admission alone is not enough to prove bounded magnitude.
func TestSnapshot_BitLimitsAndMagnitudeInequalities(t *testing.T) {
	stream := newStream(t)
	mustPush(t, stream, 17, 3, true)
	before := stream.Snapshot()
	unitMax := units(math.MaxFloat64)
	one := big.NewInt(1)
	maxCount := new(big.Int).SetUint64(^uint64(0))
	maxVolume := new(big.Int).Mul(unitMax, maxCount)
	maxWeighted := new(big.Int).Mul(maxVolume, unitMax)
	valid := snapshotFromInts(^uint64(0), maxVolume, maxWeighted)
	if err := stream.Restore(valid); err != nil {
		t.Fatal(err)
	}
	if stream.Snapshot() != valid {
		t.Fatal("maximum exact checkpoint did not round trip")
	}
	// Explicit reset at the maximum count remains legal.
	requireValue(t, mustPush(t, stream, -math.MaxFloat64, math.MaxFloat64, true), vwap.Value{Value: -math.MaxFloat64, Valid: true})
	if err := stream.Restore(before); err != nil {
		t.Fatal(err)
	}
	invalid := []vwap.Snapshot{
		snapshotFromInts(1, new(big.Int).Add(unitMax, one), big.NewInt(0)),
		snapshotFromInts(1, unitMax, new(big.Int).Add(new(big.Int).Mul(unitMax, unitMax), one)),
		snapshotFromInts(1, unitMax, new(big.Int).Neg(new(big.Int).Add(new(big.Int).Mul(unitMax, unitMax), one))),
		snapshotFromInts(^uint64(0), new(big.Int).Lsh(one, 2162), big.NewInt(0)),
		snapshotFromInts(^uint64(0), maxVolume, new(big.Int).Lsh(one, 4260)),
		snapshotFromInts(^uint64(0), maxVolume, new(big.Int).Add(maxWeighted, one)),
		snapshotFromInts(0, big.NewInt(0), one),
		snapshotFromInts(1, big.NewInt(0), one),
	}
	for _, bad := range invalid {
		requireError(t, stream.Restore(bad), vwap.ErrStateVersion)
		requireUnchanged(t, stream, before)
	}
	zero := snapshotFromInts(^uint64(0), big.NewInt(0), big.NewInt(0))
	if err := stream.Restore(zero); err != nil {
		t.Fatal(err)
	}
	if stream.Snapshot() != zero {
		t.Fatal("maximum-count masked state rejected")
	}
	requireValue(t, mustPush(t, stream, 0, 0, true), vwap.Value{})
}

// TestSnapshot_RoundingArbitraryStructuralStates protects rounding of admitted rational states beyond sampled trade histories.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: construct divisors and signed numerators at explicit integer boundaries.
//   - Stage 3: compare Restore-derived output with independent big.Rat rounding.
//
// Behavior highlights:
//   - This test protects rounding of admitted rational states beyond sampled trade histories without altering production visibility.
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
//   - Structural acceptance is deliberately not a proof of historical realizability.
func TestSnapshot_RoundingArbitraryStructuralStates(t *testing.T) {
	// Structural validity does not imply historical reachability. These valid
	// artificial states independently exercise quotient ties and limb boundaries.
	denominators := []*big.Int{
		big.NewInt(1), big.NewInt(2), big.NewInt(3),
		new(big.Int).Lsh(big.NewInt(1), 63),
		new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 64), big.NewInt(1)),
		new(big.Int).Add(new(big.Int).Lsh(big.NewInt(1), 130), big.NewInt(17)),
		new(big.Int).Add(new(big.Int).Lsh(big.NewInt(1), 2100), big.NewInt(47)),
	}
	scale := new(big.Int).Lsh(big.NewInt(1), 1074)
	for _, denominator := range denominators {
		for _, exponent := range []uint{0, 1, 51, 52, 53, 64, 1024, 2097} {
			numerator := new(big.Int).Lsh(big.NewInt(1), exponent)
			numerator.Mul(numerator, denominator)
			numerator.Add(numerator, new(big.Int).Rsh(new(big.Int).Set(denominator), 1))
			for _, sign := range []int64{-1, 1} {
				signed := new(big.Int).Mul(numerator, big.NewInt(sign))
				state := snapshotFromInts(^uint64(0)-1, denominator, signed)
				stream := newStream(t)
				if err := stream.Restore(state); err != nil {
					t.Fatal(err)
				}
				ratio := new(big.Rat).SetFrac(signed, new(big.Int).Mul(denominator, scale))
				expected, _ := ratio.Float64()
				got := mustPush(t, stream, 0, 0, false)
				requireValue(t, got, vwap.Value{Value: expected, Valid: true})
			}
		}
	}
}

// TestSnapshot_NoInternalHeapAllocationsOnRestore protects bounded zero-allocation Restore parsing and rejection.
// Assertions exercise only exported VWAP operations and independently formed expectations.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: measure a long valid snapshot and an oversized rejected string after setup.
//   - Stage 3: require the correct classification and zero measured allocations.
//
// Behavior highlights:
//   - This test protects bounded zero-allocation Restore parsing and rejection without altering production visibility.
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
//   - Do not include JSON parsing or Snapshot formatting in a Restore-only measurement.
func TestSnapshot_NoInternalHeapAllocationsOnRestore(t *testing.T) {
	stream := newStream(t)
	mustPush(t, stream, -1e308, 1e308, true)
	state := stream.Snapshot()
	bad := state
	bad.WeightedSum = strings.Repeat("9", 1301)
	for _, checkpoint := range []vwap.Snapshot{state, bad} {
		var got error
		allocations := testing.AllocsPerRun(100, func() { got = stream.Restore(checkpoint) })
		if checkpoint == bad {
			requireError(t, got, vwap.ErrStateVersion)
		} else if got != nil {
			t.Fatal(got)
		}
		if allocations != 0 {
			t.Fatalf("Restore allocated %g times", allocations)
		}
	}
}
