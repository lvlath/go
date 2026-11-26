// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema_test verifies bounded snapshot parsing, schema checks, and continuation.
// Independent big integers supply decimal fixtures; production internals are inaccessible.
package ema_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"math/big"
	"strings"
	"testing"

	"github.com/lvlath/go/ema"
)

// TestSnapshot_TamperAndPhaseParity anchors exact continuation at every possible history split.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Serialize typed snapshots, restore, and compare every resumed value and final state.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Both seeds, P1/P3/P14, before/during/after warmup.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Comparing only the last output can hide intermediate readiness errors.
func TestSnapshot_TamperAndPhaseParity(t *testing.T) {
	x := []float64{1e308, 1, -1e308, 62000, 62001.5, 61998, 62003, 62006, 62001, 62004, 62003, 62008, 62007, 62009, 62011, 62010}
	for _, period := range []int{1, 3, 14} {
		for _, seed := range []ema.Seed{ema.SeedSMA, ema.SeedFirst} {
			opts := ema.Options{Period: period, Seed: seed}
			want := oracleHistory(x, opts)
			for split := 0; split <= len(x); split++ {
				source := newStream(t, opts)
				for _, value := range x[:split] {
					pushValue(t, source, value)
				}
				state := source.Snapshot()
				expectedSum := "0"
				if seed == ema.SeedSMA && split < period {
					expectedSum = exactSeedText(x[:split])
				}
				if state.SeedSum != expectedSum {
					t.Fatalf("split=%d sum mismatch", split)
				}
				encoded, err := json.Marshal(state)
				if err != nil {
					t.Fatal(err)
				}
				var decoded ema.Snapshot
				if err = json.Unmarshal(encoded, &decoded); err != nil {
					t.Fatal(err)
				}
				if !sameSnapshot(state, decoded) {
					t.Fatal("typed JSON changed bits")
				}
				restored := newStream(t, opts)
				if err = restored.Restore(decoded); err != nil {
					t.Fatal(err)
				}
				for i := split; i < len(x); i++ {
					assertValue(t, pushValue(t, restored, x[i]), want[i])
					assertValue(t, pushValue(t, source, x[i]), want[i])
				}
				if !sameSnapshot(source.Snapshot(), restored.Snapshot()) {
					t.Fatal("continuation state differs")
				}
			}
		}
	}
}

// TestSnapshot_RejectEveryMalformedField rejects individual schema, scalar, phase, and decimal corruption.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Change one field at a time and require ErrStateVersion with an unchanged snapshot.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Every external field plus adversarial canonical-decimal failures.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Do not normalize a malformed checkpoint into a different accepted state.
func TestSnapshot_RejectEveryMalformedField(t *testing.T) {
	stream := newStream(t, ema.Options{Period: 3})
	pushValue(t, stream, 100)
	before := stream.Snapshot()
	cases := []struct {
		name  string
		alter func(*ema.Snapshot)
	}{
		{"version", func(s *ema.Snapshot) { s.Version++ }},
		{"algorithm", func(s *ema.Snapshot) { s.Algorithm = "ema/affine-rn/v2" }},
		{"period", func(s *ema.Snapshot) { s.Options.Period++ }},
		{"seed-mode", func(s *ema.Snapshot) { s.Options.Seed = ema.SeedFirst }},
		{"unknown-seed", func(s *ema.Snapshot) { s.Options.Seed = 255 }},
		{"count-zero", func(s *ema.Snapshot) { s.Count = 0 }},
		{"phase", func(s *ema.Snapshot) { s.Count = 3 }},
		{"warmup-current", func(s *ema.Snapshot) { s.Current = 1 }},
		{"nan", func(s *ema.Snapshot) { s.Current = math.NaN() }},
		{"inf", func(s *ema.Snapshot) { s.Current = math.Inf(1) }},
		{"negative-zero", func(s *ema.Snapshot) { s.Current = math.Copysign(0, -1) }},
	}
	for _, fixture := range cases {
		t.Run(fixture.name, func(t *testing.T) {
			invalid := before
			fixture.alter(&invalid)
			if err := stream.Restore(invalid); !errors.Is(err, ema.ErrStateVersion) {
				t.Fatalf("Restore: %v", err)
			}
			if !sameSnapshot(before, stream.Snapshot()) {
				t.Fatal("partial mutation")
			}
		})
	}
	invalidTexts := []string{"", "-", "+1", "00", "01", "-0", "-01", " 1", "1 ", "1\n", "1e3", "1.0", "--1", "1+1", "\x001", "\xd9\xa1", strings.Repeat("9", 661), strings.Repeat("9", 660), new(big.Int).Lsh(big.NewInt(1), 2176).String()}
	for _, text := range invalidTexts {
		invalid := before
		invalid.SeedSum = text
		if err := stream.Restore(invalid); !errors.Is(err, ema.ErrStateVersion) {
			t.Fatalf("invalid text len=%d: %v", len(text), err)
		}
		if !sameSnapshot(before, stream.Snapshot()) {
			t.Fatal("text error mutated state")
		}
	}
}

// TestSnapshot_ExactMagnitudeBound checks the inclusive seed bound and its immediate neighbors.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Restore both signs of Count*MaxUnits, reject bound+1, then finish maximal warmup.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Count zero, small counts, and a maximum accepted period.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Use exact integers for the physical bound, not a floating-point approximation.
func TestSnapshot_ExactMagnitudeBound(t *testing.T) {
	period := math.MaxInt - 1
	stream := newStream(t, ema.Options{Period: period})
	state := stream.Snapshot()
	for _, count := range []uint64{0, 1, 2, uint64(period - 1)} {
		state.Count = count
		bound := new(big.Int).Mul(maxUnits(), new(big.Int).SetUint64(count))
		for _, negative := range []bool{false, true} {
			legal := new(big.Int).Set(bound)
			if negative {
				legal.Neg(legal)
			}
			state.SeedSum = legal.String()
			if err := stream.Restore(state); err != nil {
				t.Fatalf("bound count=%d: %v", count, err)
			}
			before := stream.Snapshot()
			if before.SeedSum != state.SeedSum {
				t.Fatal("decimal round trip")
			}
			bad := new(big.Int).Add(bound, big.NewInt(1))
			if negative {
				bad.Neg(bad)
			}
			invalid := state
			invalid.SeedSum = bad.String()
			if err := stream.Restore(invalid); !errors.Is(err, ema.ErrStateVersion) {
				t.Fatalf("bound+1 accepted: %v", err)
			}
			if !sameSnapshot(before, stream.Snapshot()) {
				t.Fatal("bound rejection changed state")
			}
		}
	}
	// Complete the longest legal SMA warmup without storing its history.
	state.Count = uint64(period - 1)
	state.SeedSum = new(big.Int).Mul(maxUnits(), new(big.Int).SetUint64(state.Count)).String()
	if err := stream.Restore(state); err != nil {
		t.Fatal(err)
	}
	assertValue(t, pushValue(t, stream, math.MaxFloat64), ema.Value{Value: math.MaxFloat64, Valid: true})
	if stream.Snapshot().SeedSum != "0" {
		t.Fatal("seed not cleared at maturity")
	}
}

// TestSnapshot_FirstAndMatureInvariants checks empty First and mature seed-clearing constraints.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Reject inconsistent fields and preserve MaxUint64 through typed JSON.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - P1, First, and mature SMA states.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - A saturated count is restorable but cannot accept another observation.
func TestSnapshot_FirstAndMatureInvariants(t *testing.T) {
	for _, opts := range []ema.Options{{Period: 1}, {Period: 3, Seed: ema.SeedFirst}} {
		stream := newStream(t, opts)
		state := stream.Snapshot()
		if err := stream.Restore(state); err != nil {
			t.Fatal(err)
		}
		state.Current = 1
		if err := stream.Restore(state); !errors.Is(err, ema.ErrStateVersion) {
			t.Fatal("unobserved current accepted")
		}
		state.Current = 0
		state.SeedSum = "1"
		if err := stream.Restore(state); !errors.Is(err, ema.ErrStateVersion) {
			t.Fatal("first seed sum accepted")
		}
		state.SeedSum = "0"
		state.Count = math.MaxUint64
		state.Current = -3.5
		if err := stream.Restore(state); err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(stream.Snapshot())
		if err != nil {
			t.Fatal(err)
		}
		var decoded ema.Snapshot
		if err = json.Unmarshal(encoded, &decoded); err != nil {
			t.Fatal(err)
		}
		if decoded.Count != math.MaxUint64 {
			t.Fatal("Count precision lost")
		}
	}
	stream := newStream(t, ema.Options{Period: 3})
	for _, x := range []float64{1, 2, 3} {
		pushValue(t, stream, x)
	}
	before := stream.Snapshot()
	invalid := before
	invalid.SeedSum = "1"
	if err := stream.Restore(invalid); !errors.Is(err, ema.ErrStateVersion) {
		t.Fatal("mature sum accepted")
	}
	invalid = before
	invalid.Count = 1
	if err := stream.Restore(invalid); !errors.Is(err, ema.ErrStateVersion) {
		t.Fatal("count/current phase mismatch accepted")
	}
	if !sameSnapshot(before, stream.Snapshot()) {
		t.Fatal("mature rollback")
	}
}

// TestSnapshot_NoAllocationOnRestore checks allocation-free snapshot admission and rejection.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Measure valid, oversized, arithmetic-overflow, and physical-bound failures.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - A negative extreme seed and a one-megabyte hostile string.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Build hostile input before timing; Restore must not copy or parse its full oversized content.
func TestSnapshot_NoAllocationOnRestore(t *testing.T) {
	stream := newStream(t, ema.Options{Period: 14})
	pushValue(t, stream, -math.MaxFloat64)
	before := stream.Snapshot()
	states := []ema.Snapshot{before, before, before, before}
	states[1].SeedSum = strings.Repeat("9", 1<<20)
	states[2].SeedSum = strings.Repeat("9", 660)
	states[3].SeedSum = new(big.Int).Add(maxUnits(), big.NewInt(1)).String()
	for i, state := range states {
		var err error
		allocations := testing.AllocsPerRun(1000, func() { err = stream.Restore(state) })
		if allocations != 0 {
			t.Fatalf("case=%d allocs=%g", i, allocations)
		}
		if (i == 0 && err != nil) || (i > 0 && !errors.Is(err, ema.ErrStateVersion)) {
			t.Fatalf("case=%d err=%v", i, err)
		}
		if !sameSnapshot(before, stream.Snapshot()) {
			t.Fatal("Restore changed canonical state")
		}
	}
}

// TestSnapshot_DetachedState checks snapshot independence and structural rather than historical authentication.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Change copied snapshot fields, then restore a coherent edited state.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Partial SMA seed and detached decimal text.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - A valid format cannot prove that the record originated from a genuine history.
func TestSnapshot_DetachedState(t *testing.T) {
	stream := newStream(t, ema.Options{Period: 14})
	pushValue(t, stream, -123)
	first := stream.Snapshot()
	originalText := first.SeedSum
	pushValue(t, stream, 124)
	if first.SeedSum != originalText || originalText != exactSeedText([]float64{-123}) {
		t.Fatal("snapshot aliases live accumulator")
	}
	first.Options.Period = 1
	first.Count = 0
	first.SeedSum = "0"
	if stream.Snapshot().Options.Period != 14 || stream.Snapshot().Count != 2 {
		t.Fatal("external snapshot mutation leaked")
	}
	// A structurally coherent edited state is allowed; this ABI is not authentication.
	coherent := stream.Snapshot()
	coherent.SeedSum = "14"
	if err := stream.Restore(coherent); err != nil {
		t.Fatal(err)
	}
	_, err := stream.Push(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}
}

// TestSnapshot_RationalSeedCompletion compares arbitrary admitted exact seed states with an independent rational mean.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Generate bounded signed bigint seeds, finish warmup, and check the next recurrence.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Four periods and deterministic structurally valid wide integers.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Reachability from a specific observation list is not required by structural Restore.
func TestSnapshot_RationalSeedCompletion(t *testing.T) {
	rng := uint64(0x31a01)
	samples := finiteFixture(64)
	units := new(big.Int).Lsh(big.NewInt(1), 1074)
	for _, period := range []int{2, 3, 14, math.MaxInt - 1} {
		stream := newStream(t, ema.Options{Period: period})
		state := stream.Snapshot()
		state.Count = uint64(period - 1)
		bound := new(big.Int).Mul(maxUnits(), new(big.Int).SetUint64(state.Count))
		modulus := new(big.Int).Add(bound, big.NewInt(1))
		for _, sample := range samples {
			sum := new(big.Int)
			for word := 0; word < 34; word++ {
				sum.Lsh(sum, 64)
				sum.Add(sum, new(big.Int).SetUint64(fixtureBits(&rng)))
			}
			sum.Mod(sum, modulus)
			if fixtureBits(&rng)&1 != 0 {
				sum.Neg(sum)
			}
			state.SeedSum = sum.String()
			if err := stream.Restore(state); err != nil {
				t.Fatal(err)
			}
			rational := new(big.Rat).SetFrac(sum, units)
			rational.Add(rational, new(big.Rat).SetFloat64(sample))
			rational.Quo(rational, new(big.Rat).SetInt64(int64(period)))
			expected := ratRound(rational)
			assertValue(t, pushValue(t, stream, sample), ema.Value{Value: expected, Valid: true})
			if stream.Snapshot().SeedSum != "0" {
				t.Fatal("seed was not cleared")
			}
			next := oracleHistory([]float64{expected, -sample}, ema.Options{Period: period, Seed: ema.SeedFirst})
			assertValue(t, pushValue(t, stream, -sample), next[1])
		}
	}
}

// TestSnapshot_ReplacementAcrossAllPhases checks replacement of already-populated streams without residue.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Restore each source phase over every target phase and compare the continuation.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Empty, partial, completion, mature, identity, and First configurations.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Clearing old seed state matters when replacing warmup by a mature snapshot.
func TestSnapshot_ReplacementAcrossAllPhases(t *testing.T) {
	inputs := failoverHistory()
	for _, opts := range []ema.Options{{Period: 1}, {Period: 14}, {Period: 14, Seed: ema.SeedFirst}} {
		want := oracleHistory(inputs, opts)
		for _, sourceCount := range []int{0, 1, 5, 13, 14, 20} {
			for _, targetCount := range []int{0, 1, 5, 13, 14, 20} {
				source, target := newStream(t, opts), newStream(t, opts)
				for _, x := range inputs[:sourceCount] {
					pushValue(t, source, x)
				}
				for _, x := range inputs[:targetCount] {
					pushValue(t, target, -x)
				}
				state := source.Snapshot()
				if err := target.Restore(state); err != nil {
					t.Fatal(err)
				}
				if !sameSnapshot(state, target.Snapshot()) {
					t.Fatal("restore retained target residue")
				}
				for i := sourceCount; i < len(inputs); i++ {
					assertValue(t, pushValue(t, target, inputs[i]), want[i])
				}
			}
		}
	}
}

// TestSnapshot_CarryBorrowAtEverySeedLimb checks exact seed arithmetic across every occupied word boundary.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Round-trip decimal units and force carry or borrow before the final seed division.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Both signs at every reachable 64-bit boundary through bit 2112, including the highest limb.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Exercise through Restore and Push rather than exporting private arithmetic for tests.
func TestSnapshot_CarryBorrowAtEverySeedLimb(t *testing.T) {
	// The last storage boundary is 2112; 2176 lies outside the admitted count bound.
	const highestSeedBoundary = 2112
	const highBoundaryPeriod = 1 << 16
	for bit := uint(64); bit <= highestSeedBoundary; bit += 64 {
		period := 3
		if bit == highestSeedBoundary {
			// Count=65535 admits these units below Count*MaxBinary64Units on both int sizes.
			period = highBoundaryPeriod
		}
		stream := newStream(t, ema.Options{Period: period})
		for _, subtract := range []bool{false, true} {
			sum := new(big.Int).Lsh(big.NewInt(1), bit)
			sample := -math.SmallestNonzeroFloat64
			if !subtract {
				sum.Sub(sum, big.NewInt(1))
				sample = math.SmallestNonzeroFloat64
			}
			for _, negative := range []bool{false, true} {
				units := new(big.Int).Set(sum)
				x := sample
				if negative {
					units.Neg(units)
					x = -x
				}
				state := stream.Snapshot()
				state.Count, state.Current, state.SeedSum = uint64(period-1), 0, units.String()
				if err := stream.Restore(state); err != nil {
					t.Fatal(err)
				}
				if stream.Snapshot().SeedSum != state.SeedSum {
					t.Fatal("canonical decimal round-trip failed")
				}
				assertValue(t, pushValue(t, stream, x), completedSeedValue(units, x, period))
			}
		}
	}
}

// TestSnapshot_TypedJSONPreservesFiniteBits checks finite scalar round-trip and continuation through typed JSON.
// The public EMA boundary is exercised without access to private implementation state.
//
// Implementation:
//   - Stage 1: construct explicit deterministic fixtures and expected outcomes.
//   - Stage 2: Marshal, decode, compare every field, restore, and verify the next step.
//   - Stage 3: fail on any mismatched value, mask, error class, or state invariant.
//
// Behavior highlights:
//   - Mixed finite exponents, signed subnormals, MaxFloat64, and zero.
//
// Inputs:
//   - t: subtest-owning testing context; helper assertions report through this owner.
//
// Returns:
//   - No library result; success means every enumerated assertion held.
//
// Errors:
//   - Unexpected errors and contract violations fail the test; expected errors use errors.Is.
//
// Determinism:
//   - Fixture order and bit comparisons are fixed; no timing-dependent tolerance is used.
//
// Complexity:
//   - Finite fixture workload; production complexity is documented on the exercised API.
//
// Notes:
//   - The test checks its stated property, not all possible input histories.
//
// AI-Hints:
//   - Decoding Count into a float64 map would destroy large integer precision.
func TestSnapshot_TypedJSONPreservesFiniteBits(t *testing.T) {
	values := append(finiteFixture(128), math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64, math.MaxFloat64, 0)
	stream := newStream(t, ema.Options{Period: 14, Seed: ema.SeedFirst})
	for _, x := range values {
		stream.Reset()
		pushValue(t, stream, x)
		state := stream.Snapshot()
		wire, err := json.Marshal(state)
		if err != nil {
			t.Fatal(err)
		}
		var restored ema.Snapshot
		if err = json.Unmarshal(wire, &restored); err != nil {
			t.Fatal(err)
		}
		if !sameSnapshot(restored, state) {
			t.Fatal("typed JSON changed finite bits")
		}
		if err = stream.Restore(restored); err != nil {
			t.Fatal(err)
		}
		want := oracleHistory([]float64{x, -x}, state.Options)
		assertValue(t, pushValue(t, stream, -x), want[1])
	}
}
