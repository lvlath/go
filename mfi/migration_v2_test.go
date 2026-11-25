// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Snapshot migration and source replay are tested through public wire data.
package mfi_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/mfi"
)

// TestSnapshot_V1_Rejected uses a known v1 directional artifact, not just a
// changed header on v2 data. Replaying OHLCV reconstructs its correct flat slot.
func TestSnapshot_V1_Rejected(t *testing.T) {
	const p = 1
	s, err := mfi.NewPeriod(p)
	if err != nil {
		t.Fatal(err)
	}
	a := mfi.Bar{Open: 2, High: 4, Low: 1, Close: 2, Volume: 1}
	b := mfi.Bar{Open: 2, High: 3, Low: 2, Close: 2, Volume: 1}
	for _, bar := range []mfi.Bar{a, b} {
		if _, _, err = s.Push(bar); err != nil {
			t.Fatal(err)
		}
	}
	before := s.Snapshot()
	if before.Algorithm != mfiSnapshotV2 || before.Config != snapshotDigest(mfiSnapshotV2, p) {
		t.Fatalf("wrong v2 identity: %+v", before)
	}
	oldTP := math.Float64frombits(0x4002aaaaaaaaaaaa) // v1 rounded B downward.
	old := mfi.Snapshot{Algorithm: mfiSnapshotV1, Config: snapshotDigest(mfiSnapshotV1, p), Count: 2, FloatState: []float64{oldTP, 0, oldTP, 0, 0, 0, oldTP}, IntState: []uint64{1, 0, 0}}
	mustErrorIs(t, s.Restore(old), mfi.ErrStateVersion)
	mustSnapshotBits(t, s.Snapshot(), before)
	// A v2 identifier alone cannot authorize the legacy configuration digest.
	old.Algorithm = mfiSnapshotV2
	mustErrorIs(t, s.Restore(old), mfi.ErrStateVersion)
	mustSnapshotBits(t, s.Snapshot(), before)
	rebuilt, err := mfi.NewPeriod(p)
	if err != nil {
		t.Fatal(err)
	}
	for i, bar := range []mfi.Bar{a, b} {
		value, ready, err := rebuilt.Push(bar)
		if err != nil || ready != (i == 1) || (ready && value != 50) {
			t.Fatalf("rebuild %d: (%g,%t,%v)", i, value, ready, err)
		}
	}
	mustSnapshotBits(t, rebuilt.Snapshot(), before)
}

// TestSnapshot_RepresentableNearMaxWindowRoundTrip preserves a successful
// compensated window whose exact sum is just beyond MaxFloat64 but whose
// nearest-even effective sum is finite. Restore must agree with Push.
func TestSnapshot_RepresentableNearMaxWindowRoundTrip(t *testing.T) {
	s, err := mfi.NewPeriod(2)
	if err != nil {
		t.Fatal(err)
	}
	for _, bar := range []mfi.Bar{sampleBar(1, 0), sampleBar(2, math.Ldexp(1, 967)), sampleBar(4, math.MaxFloat64/4)} {
		if _, _, err = s.Push(bar); err != nil {
			t.Fatal(err)
		}
	}
	saved := s.Snapshot()
	replica, err := mfi.NewPeriod(2)
	if err != nil {
		t.Fatal(err)
	}
	if err = replica.Restore(saved); err != nil {
		t.Fatalf("successful near-max checkpoint rejected: %v", err)
	}
	mustSnapshotBits(t, replica.Snapshot(), saved)
}

// TestSnapshot_RejectsUnrepresentableRingTotals covers the two independent
// reference sums and the two effective-sum overflow checks during Restore.
// Corrupt wire artifacts cannot bypass the replay even with finite saved fields.
func TestSnapshot_RejectsUnrepresentableRingTotals(t *testing.T) {
	const p = 3
	maxValue := math.MaxFloat64
	small := math.Ldexp(1, 969)
	for _, positive := range []bool{true, false} {
		for _, flows := range [][p]float64{{maxValue, maxValue, 0}, {maxValue, small, small}} {
			s, err := mfi.NewPeriod(p)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err = s.Push(sampleBar(10, 0)); err != nil {
				t.Fatal(err)
			}
			original := s.Snapshot()
			bad := s.Snapshot()
			bad.Count = p + 1
			bad.IntState[mfiUsed] = p
			bad.IntState[mfiHead] = 0
			start := mfiHeader + p
			if positive {
				start = mfiHeader
			}
			copy(bad.FloatState[start:start+p], flows[:])
			mustErrorIs(t, s.Restore(bad), mfi.ErrStateVersion)
			mustSnapshotBits(t, s.Snapshot(), original)
		}
	}
}
