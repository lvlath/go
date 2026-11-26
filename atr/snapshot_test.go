// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr_test protects bounded checkpoint parsing and exact restart parity.
// AI-Hints: reject malformed state, but do not mistake validation for authentication.
package atr_test

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"strings"
	"testing"

	"github.com/lvlath/go/atr"
)

// TestSnapshot_TamperAndPhaseParity compares typed serialization and restored continuation at every phase.
// empty, warming, first-ready, and mature checkpoints.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - empty, warming, first-ready, and mature checkpoints.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - A saved rounded seed mean cannot replace exact SeedSum.
func TestSnapshot_TamperAndPhaseParity(t *testing.T) {
	h, l, c := marketBars(24)
	for _, period := range []int{1, 2, 14, 64} {
		uninterrupted := newStream(t, period)
		for split := 0; split <= len(h); split++ {
			state := uninterrupted.Snapshot()
			encoded, err := json.Marshal(state)
			if err != nil {
				t.Fatal(err)
			}
			var decoded atr.Snapshot
			if err = json.Unmarshal(encoded, &decoded); err != nil {
				t.Fatal(err)
			}
			mustSnapshot(t, decoded, state)
			a, b := newStream(t, period), newStream(t, period)
			if err = a.Restore(state); err != nil {
				t.Fatal(err)
			}
			if err = b.Restore(decoded); err != nil {
				t.Fatal(err)
			}
			for i := split; i < len(h); i++ {
				mustValue(t, push(t, a, candle{h[i], l[i], c[i]}), push(t, b, candle{h[i], l[i], c[i]}))
			}
			// Both restorations must also match a full-prefix batch oracle.
			want := compute(t, h, l, c, period)
			last := a.Snapshot()
			mustFloatBits(t, last.Current, want.ATR[len(h)-1])
			if split < len(h) {
				push(t, uninterrupted, candle{h[split], l[split], c[split]})
			}
		}
	}
}

// TestSnapshot_FieldTamperingIsTransactional rejects invalid schema, configuration, scalars, and phase combinations.
// each public snapshot field varied against a retained original state.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - each public snapshot field varied against a retained original state.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - A failed Restore must not partially publish any field.
func TestSnapshot_FieldTamperingIsTransactional(t *testing.T) {
	mutations := []struct {
		name   string
		change func(*atr.Snapshot)
	}{
		{"version", func(s *atr.Snapshot) { s.Version++ }},
		{"algorithm", func(s *atr.Snapshot) { s.Algorithm = "atr/ema/v1" }},
		{"options", func(s *atr.Snapshot) { s.Options.Period++ }},
		{"invalid-options", func(s *atr.Snapshot) { s.Options.Period = 0 }},
		{"previous-nan", func(s *atr.Snapshot) { s.PreviousClose = math.NaN() }},
		{"previous-plus-inf", func(s *atr.Snapshot) { s.PreviousClose = math.Inf(1) }},
		{"previous-minus-inf", func(s *atr.Snapshot) { s.PreviousClose = math.Inf(-1) }},
		{"current-nan", func(s *atr.Snapshot) { s.Current = math.NaN() }},
		{"current-inf", func(s *atr.Snapshot) { s.Current = math.Inf(1) }},
		{"current-negative", func(s *atr.Snapshot) { s.Current = -1 }},
		{"empty-seed", func(s *atr.Snapshot) { s.SeedSum = "" }},
		{"negative-seed", func(s *atr.Snapshot) { s.SeedSum = "-1" }},
		{"plus-seed", func(s *atr.Snapshot) { s.SeedSum = "+1" }},
		{"negative-zero-seed", func(s *atr.Snapshot) { s.SeedSum = "-0" }},
		{"leading-zero", func(s *atr.Snapshot) { s.SeedSum = "01" }},
		{"double-zero", func(s *atr.Snapshot) { s.SeedSum = "00" }},
		{"space-seed", func(s *atr.Snapshot) { s.SeedSum = "1 " }},
		{"nondigit", func(s *atr.Snapshot) { s.SeedSum = "1x" }},
		{"decimal-fraction", func(s *atr.Snapshot) { s.SeedSum = "1.0" }},
		{"exponent", func(s *atr.Snapshot) { s.SeedSum = "1e2" }},
		{"too-long", func(s *atr.Snapshot) { s.SeedSum = strings.Repeat("9", 661) }},
		{"fixed-capacity-overflow", func(s *atr.Snapshot) { s.SeedSum = strings.Repeat("9", 660) }},
		{"sum-bound", func(s *atr.Snapshot) { s.SeedSum = new(big.Int).Lsh(big.NewInt(1), 2163).String() }},
	}
	for _, count := range []int{0, 1, 2, 3, 4} {
		for _, mutation := range mutations {
			t.Run(fmt.Sprintf("phase%d/%s", count, mutation.name), func(t *testing.T) {
				s := newStream(t, 3)
				for i := 0; i < count; i++ {
					push(t, s, candle{3, 1, 2})
				}
				before := s.Snapshot()
				changed := before
				mutation.change(&changed)
				mustErrorIs(t, s.Restore(changed), atr.ErrStateVersion)
				mustSnapshot(t, s.Snapshot(), before)
			})
		}
	}
	empty := newStream(t, 14)
	for _, previous := range []float64{1, -1} {
		state := empty.Snapshot()
		state.PreviousClose = previous
		mustErrorIs(t, empty.Restore(state), atr.ErrStateVersion)
	}
	for _, current := range []float64{1, math.MaxFloat64} {
		state := empty.Snapshot()
		state.Current = current
		mustErrorIs(t, empty.Restore(state), atr.ErrStateVersion)
	}
	state := empty.Snapshot()
	state.SeedSum = "1"
	mustErrorIs(t, empty.Restore(state), atr.ErrStateVersion)
	warming := newStream(t, 3)
	push(t, warming, candle{3, 1, 2})
	state = warming.Snapshot()
	state.Count = 3
	mustErrorIs(t, warming.Restore(state), atr.ErrStateVersion)
	state = warming.Snapshot()
	state.Count = 0
	mustErrorIs(t, warming.Restore(state), atr.ErrStateVersion)
	ready := newStream(t, 1)
	push(t, ready, candle{3, 1, 2})
	state = ready.Snapshot()
	state.Count = 0
	mustErrorIs(t, ready.Restore(state), atr.ErrStateVersion)
}

// TestSnapshot_ExactSeedUnitsAndBoundaries checks exact decimal sums and their maximum phase-dependent bounds.
// subnormal units, MaxFloat64 multiples, equality and one-unit excess.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - subnormal units, MaxFloat64 multiples, equality and one-unit excess.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Use an independent big.Int bound rather than production encoding.
func TestSnapshot_ExactSeedUnitsAndBoundaries(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	s := newStream(t, maxInt)
	sum := new(big.Int)
	values := []float64{0, math.SmallestNonzeroFloat64, math.MaxFloat64, math.MaxFloat64, math.Nextafter(2, 0)}
	for _, tr := range values {
		push(t, s, candle{tr, 0, 0})
		sum.Add(sum, binary64Units(tr))
		if s.Snapshot().SeedSum != sum.String() {
			t.Fatal("exact units changed")
		}
	}
	state := s.Snapshot()
	state.Count = uint64(maxInt - 1)
	maxUnits := binary64Units(math.MaxFloat64)
	bound := new(big.Int).Mul(maxUnits, new(big.Int).SetUint64(state.Count))
	state.SeedSum = bound.String()
	if err := s.Restore(state); err != nil {
		t.Fatal(err)
	}
	got := push(t, s, candle{math.MaxFloat64, 0, 0})
	mustValue(t, got, atr.Value{TR: math.MaxFloat64, ATR: math.MaxFloat64, Valid: true})
	mustSnapshot(t, s.Snapshot(), atr.Snapshot{Version: 1, Algorithm: "atr/wilder-rn/v1", Options: atr.Options{Period: maxInt}, Count: uint64(maxInt), Current: math.MaxFloat64, SeedSum: "0"})
	// Exact upper bound is admissible, one unit above it is not.
	state.SeedSum = new(big.Int).Add(bound, big.NewInt(1)).String()
	before := s.Snapshot()
	mustErrorIs(t, s.Restore(state), atr.ErrStateVersion)
	mustSnapshot(t, s.Snapshot(), before)
	// Zero seed is admissible with a nonzero warmup count (flat candles).
	state.SeedSum = "0"
	if err := s.Restore(state); err != nil {
		t.Fatal(err)
	}
	got = push(t, s, candle{1, 0, 0})
	exact := new(big.Rat).SetFrac(big.NewInt(1), new(big.Int).SetInt64(int64(maxInt)))
	want, _ := exact.Float64()
	mustValue(t, got, atr.Value{TR: 1, ATR: want, Valid: true})
}

// TestSnapshot_SignedZeroAndNoAuthenticationClaim separates exact scalar preservation from historical authentication.
// signed previous-close zero and a structurally plausible altered close.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - signed previous-close zero and a structurally plausible altered close.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - A valid snapshot is not evidence that a particular feed produced it.
func TestSnapshot_SignedZeroAndNoAuthenticationClaim(t *testing.T) {
	s := newStream(t, 2)
	minusZero := math.Copysign(0, -1)
	push(t, s, candle{minusZero, 0, minusZero})
	state := s.Snapshot()
	mustFloatBits(t, state.PreviousClose, minusZero)
	target := newStream(t, 2)
	if err := target.Restore(state); err != nil {
		t.Fatal(err)
	}
	mustSnapshot(t, target.Snapshot(), state)
	// A different finite previous close is structurally plausible. The ABI has
	// no bar history or MAC with which to authenticate its provenance.
	state.PreviousClose = 123
	if err := target.Restore(state); err != nil {
		t.Fatal(err)
	}
	got := push(t, target, candle{3, 1, 2})
	mustFloatBits(t, got.TR, 122)
}

// TestSnapshot_OversizedRejectAndRestoreAllocateNothing measures bounded rejection and accepted Restore allocations.
// oversized decimal text and valid canonical snapshots.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - oversized decimal text and valid canonical snapshots.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Input size must be checked before arbitrary-precision parsing.
func TestSnapshot_OversizedRejectAndRestoreAllocateNothing(t *testing.T) {
	s := newStream(t, 14)
	push(t, s, candle{math.MaxFloat64, 0, 0})
	good := s.Snapshot()
	bad := good
	bad.SeedSum = strings.Repeat("1", 100000)
	var failure error
	n := testing.AllocsPerRun(100, func() { failure = s.Restore(good) })
	if failure != nil {
		t.Fatal(failure)
	}
	if n != 0 {
		t.Fatalf("Restore allocated %g", n)
	}
	n = testing.AllocsPerRun(100, func() { failure = s.Restore(bad) })
	mustErrorIs(t, failure, atr.ErrStateVersion)
	if n != 0 {
		t.Fatalf("oversized rejection allocated %g", n)
	}
	mustSnapshot(t, s.Snapshot(), good)
}

// TestSnapshot_ZeroSignsFollowTheDeclaredPhaseDomain anchors distinct warmup and mature zero-sign admission.
// positive-zero warmup payload and nonnegative mature scalar domain.
//
// Implementation:
//   - Stage 1: Prepare the fixed input regime described above.
//   - Stage 2: Exercise only the exported atr facade and compare the stated witness.
//   - Stage 3: Report any violated invariant through the owning test or benchmark.
//
// Behavior highlights:
//   - positive-zero warmup payload and nonnegative mature scalar domain.
//
// Inputs:
//   - Prepare the fixed input regime described above.
//
// Returns:
//   - Report any violated invariant through the owning test or benchmark.
//
// Errors:
//   - Unexpected errors fail; expected errors use errors.Is.
//
// Determinism:
//   - Fixed fixture and traversal order; no implicit random seed or timing threshold.
//
// Complexity:
//   - Bounded fixture workload; measured algorithm costs are documented in atr/doc.go.
//
// Notes:
//   - Test assertions run only in the owning test goroutine.
//
// AI-Hints:
//   - Do not silently strengthen a serialized mature-zero contract.
func TestSnapshot_ZeroSignsFollowTheDeclaredPhaseDomain(t *testing.T) {
	negativeZero := math.Copysign(0, -1)
	// An empty checkpoint permits either sign of zero PreviousClose.
	// Count=0 still prevents this field from participating in the first TR.
	fresh := newStream(t, 3)
	empty := fresh.Snapshot()
	empty.PreviousClose = negativeZero
	if err := fresh.Restore(empty); err != nil {
		t.Fatal(err)
	}
	mustSnapshot(t, fresh.Snapshot(), empty)
	mustValue(t, push(t, fresh, candle{-1, -3, -2}), atr.Value{TR: 2})
	// Warmup payload is explicitly +0, so a -0 Current cannot enter that phase.
	for _, count := range []uint64{0, 1, 2} {
		s := newStream(t, 3)
		state := s.Snapshot()
		state.Count = count
		state.Current = negativeZero
		mustErrorIs(t, s.Restore(state), atr.ErrStateVersion)
	}
	// In the mature phase the declared domain is finite Current>=0. Negative
	// zero satisfies it and is preserved on restore, not silently normalized.
	s := newStream(t, 3)
	state := s.Snapshot()
	state.Count = 3
	state.Current = negativeZero
	if err := s.Restore(state); err != nil {
		t.Fatal(err)
	}
	mustSnapshot(t, s.Snapshot(), state)
	wire, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	var decoded atr.Snapshot
	if err = json.Unmarshal(wire, &decoded); err != nil {
		t.Fatal(err)
	}
	mustSnapshot(t, decoded, state)
	mustValue(t, push(t, s, candle{0, 0, 0}), atr.Value{Valid: true})
}
