// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Independent exact-price oracle and explicit public-state fixtures.
package mfi_test

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/mfi"
)

const (
	mfiSnapshotV1         = "mfi/rolling-neumaier/v1"
	mfiSnapshotV2         = "mfi/rolling-neumaier/v2"
	mfiHeader             = 5
	mfiPrevious           = 0
	mfiPositiveSum        = 1
	mfiNegativeSum        = 2
	mfiPositiveCorrection = 3
	mfiNegativeCorrection = 4
	mfiUsed               = 0
	mfiHead               = 1
	mfiExtra              = 2
)

// exactTypicalOracle sums exact rational representations of the original binary64
// inputs and rounds exactly once via Rat.Float64. It shares no production arithmetic.
func exactTypicalOracle(bar mfi.Bar) float64 {
	var sum big.Rat
	for _, price := range [...]float64{bar.High, bar.Low, bar.Close} {
		sum.Add(&sum, new(big.Rat).SetFloat64(price))
	}
	sum.Quo(&sum, big.NewRat(3, 1))
	value, _ := sum.Float64()
	return value
}

// sortedPriceBar places the maximum and minimum in H/L and uses the remaining
// value for Open/Close, preserving all three mean operands and valid geometry.
func sortedPriceBar(a, b, c float64) mfi.Bar {
	if a > b {
		a, b = b, a
	}
	if b > c {
		b, c = c, b
	}
	if a > b {
		a, b = b, a
	}
	return mfi.Bar{Open: b, High: c, Low: a, Close: b, Volume: 0}
}

// assertTypicalBits observes the canonical price through the detached Snapshot
// ABI after the first accepted bar, which cannot contain a directional flow.
func assertTypicalBits(t *testing.T, s *mfi.Stream, bar mfi.Bar) {
	t.Helper()
	s.Reset()
	value, ready, err := s.Push(bar)
	if err != nil || ready || value != 0 {
		t.Fatalf("seed: (%g,%t,%v) for %+v", value, ready, err, bar)
	}
	got := s.Snapshot().FloatState[mfiPrevious]
	want := exactTypicalOracle(bar)
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("H=%016x L=%016x C=%016x: got=%016x want=%016x", math.Float64bits(bar.High), math.Float64bits(bar.Low), math.Float64bits(bar.Close), math.Float64bits(got), math.Float64bits(want))
	}
}

// snapshotDigest independently encodes the public wire identity; no package
// helper, recurrence or private state is used.
func snapshotDigest(algorithm string, period int) string {
	var encoded [8]byte
	binary.LittleEndian.PutUint64(encoded[:], uint64(period))
	data := append([]byte(algorithm), encoded[:]...)
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

// restoredRingFixture builds a documented wire state with a full ring and a
// specific physical cursor. This is a synthetic but structurally admissible
// Restore input, not a claim that its compensators arise in normal trading.
// All ring contributions and effective sums are checked by the real Restore.
func restoredRingFixture(t *testing.T, period int, count uint64, previous float64, pos, neg []float64, posSum, negSum, posCorr, negCorr float64, extra uint64) *mfi.Stream {
	t.Helper()
	s, err := mfi.NewPeriod(period)
	if err != nil {
		t.Fatal(err)
	}
	state := s.Snapshot()
	state.Count = count
	state.FloatState[mfiPrevious] = previous
	state.FloatState[mfiPositiveSum] = posSum
	state.FloatState[mfiNegativeSum] = negSum
	state.FloatState[mfiPositiveCorrection] = posCorr
	state.FloatState[mfiNegativeCorrection] = negCorr
	copy(state.FloatState[mfiHeader:mfiHeader+period], pos)
	copy(state.FloatState[mfiHeader+period:], neg)
	state.IntState[mfiUsed] = uint64(period)
	state.IntState[mfiHead] = (count - 1) % uint64(period)
	state.IntState[mfiExtra] = extra
	if err = s.Restore(state); err != nil {
		t.Fatalf("fixture not admitted by Restore: %v (state=%+v)", err, state)
	}
	mustSnapshotBits(t, s.Snapshot(), state)
	return s
}

// assertOverflowAtomic checks the only error result permitted for arithmetic
// failure and proves that all committed wire fields are unchanged.
func assertOverflowAtomic(t *testing.T, s *mfi.Stream, bar mfi.Bar) {
	t.Helper()
	before := s.Snapshot()
	value, ready, err := s.Push(bar)
	mustErrorIs(t, err, mfi.ErrOverflow)
	if value != 0 || ready {
		t.Fatalf("partial publication: (%g,%t)", value, ready)
	}
	mustSnapshotBits(t, s.Snapshot(), before)
}
