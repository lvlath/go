// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad_test validates portable checkpoint identity, corruption rejection and ownership.
// Snapshot tests use exported values only and preserve original binary64 coordinates.
package ad_test

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"testing"

	"github.com/lvlath/go/ad"
)

// TestAD10_SnapshotEveryPrefix checks all split positions against a full uninterrupted run.
func TestAD10_SnapshotEveryPrefix(t *testing.T) {
	candles := append(standardBars(), variedBars(128)...)
	baseline, err := ad.ComputeCandles(candles)
	if err != nil {
		t.Fatal(err)
	}

	for split := 0; split <= len(candles); split++ {
		var primary ad.Stream
		for _, bar := range candles[:split] {
			if _, _, pushErr := primary.Push(bar); pushErr != nil {
				t.Fatalf("split=%d primary: %v", split, pushErr)
			}
		}

		checkpoint := primary.Snapshot()
		if checkpoint.Count != uint64(split) {
			t.Fatalf("split=%d Count=%d", split, checkpoint.Count)
		}
		mustEqualInt(t, len(checkpoint.FloatState), 2, "FloatState ABI")
		mustEqualInt(t, len(checkpoint.IntState), 0, "IntState ABI")

		replica := ad.New()
		if restoreErr := replica.Restore(checkpoint); restoreErr != nil {
			t.Fatalf("split=%d Restore: %v", split, restoreErr)
		}
		mustSnapshotEqual(t, replica.Snapshot(), checkpoint)

		for index := split; index < len(candles); index++ {
			value, valid, pushErr := replica.Push(candles[index])
			if pushErr != nil {
				t.Fatalf("split=%d index=%d Push: %v", split, index, pushErr)
			}
			mustEqualBool(t, valid, true, "replica validity")
			mustBits(t, value, baseline.Values[index])
		}
	}
}

// TestSnapshot_SchemaDigestMatchesIndependentSHA256 verifies the fixed Config literal.
func TestSnapshot_SchemaDigestMatchesIndependentSHA256(t *testing.T) {
	checkpoint := ad.New().Snapshot()
	mustEqualString(t, checkpoint.Algorithm, "ad/chaikin-neumaier/v1", "Algorithm")
	sha := sha256.Sum256([]byte(checkpoint.Algorithm))
	wantConfig := hex.EncodeToString(sha[:])
	mustEqualString(t, checkpoint.Config, wantConfig, "SHA256 Config")
	mustEqualInt(t, len(checkpoint.Config), 64, "lowercase SHA256 length")
	if checkpoint.FloatState == nil || checkpoint.IntState == nil {
		t.Fatal("non-nil Snapshot must publish FloatState and empty IntState")
	}
	mustEqualInt(t, len(checkpoint.FloatState), 2, "FloatState length")
	mustEqualInt(t, len(checkpoint.IntState), 0, "IntState length")
	mustBits(t, checkpoint.FloatState[0], 0)
	mustBits(t, checkpoint.FloatState[1], 0)
}

// TestSnapshot_RestoreCorruptionTransactional rejects each incompatible public state field.
func TestSnapshot_RestoreCorruptionTransactional(t *testing.T) {
	stream := ad.New()
	for _, bar := range standardBars()[:2] {
		if _, _, err := stream.Push(bar); err != nil {
			t.Fatal(err)
		}
	}
	valid := stream.Snapshot()

	cases := []struct {
		name   string
		mutate func(*ad.Snapshot)
	}{
		{"wrongAlgorithm", func(s *ad.Snapshot) { s.Algorithm = "obv/neumaier/v1" }},
		{"missingAlgorithm", func(s *ad.Snapshot) { s.Algorithm = "" }},
		{"wrongConfig", func(s *ad.Snapshot) { s.Config = "ffff" }},
		{"uppercaseConfig", func(s *ad.Snapshot) { s.Config = "DF3132E9B6A35BDBC7D3567D9F8E3C4675DC6736013E4A408C6E150B5441A2DF" }},
		{"noFloats", func(s *ad.Snapshot) { s.FloatState = nil }},
		{"shortFloats", func(s *ad.Snapshot) { s.FloatState = s.FloatState[:1] }},
		{"extraFloats", func(s *ad.Snapshot) { s.FloatState = append(s.FloatState, 0) }},
		{"intCursor", func(s *ad.Snapshot) { s.IntState = []uint64{1} }},
		{"nonfiniteSumNaN", func(s *ad.Snapshot) { s.FloatState[0] = math.NaN() }},
		{"nonfiniteCorrectionNaN", func(s *ad.Snapshot) { s.FloatState[1] = math.NaN() }},
		{"nonfiniteSumInf", func(s *ad.Snapshot) { s.FloatState[0] = math.Inf(1) }},
		{"nonfiniteCorrectionNegInf", func(s *ad.Snapshot) { s.FloatState[1] = math.Inf(-1) }},
		{"overflowedEffectiveTotal", func(s *ad.Snapshot) {
			s.FloatState[0], s.FloatState[1] = math.MaxFloat64, math.MaxFloat64
		}},
		{"zeroCountWithPositiveSum", func(s *ad.Snapshot) { s.Count, s.FloatState[0] = 0, 1 }},
		{"zeroCountWithCorrection", func(s *ad.Snapshot) {
			s.Count, s.FloatState[0], s.FloatState[1] = 0, 0, 0.25
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			candidate := valid
			candidate.FloatState = append([]float64(nil), valid.FloatState...)
			candidate.IntState = append([]uint64(nil), valid.IntState...)
			tc.mutate(&candidate)
			mustErrorIs(t, stream.Restore(candidate), ad.ErrStateVersion)
			mustSnapshotEqual(t, stream.Snapshot(), valid)
		})
	}
}

// TestSnapshot_NilReceiverAndDetachedRestore defends ownership at each transfer boundary.
func TestSnapshot_NilReceiverAndDetachedRestore(t *testing.T) {
	var missing *ad.Stream
	mustSnapshotEqual(t, missing.Snapshot(), ad.Snapshot{})
	mustErrorIs(t, missing.Restore(ad.New().Snapshot()), ad.ErrNilInput)

	primary := ad.New()
	if _, _, err := primary.Push(standardBars()[0]); err != nil {
		t.Fatal(err)
	}
	first := primary.Snapshot()
	second := primary.Snapshot()
	mustSnapshotEqual(t, first, second)

	// Each export is detached from both the stream and every other export.
	first.FloatState[0] = 10_000
	mustBits(t, primary.Snapshot().FloatState[0], 3)
	mustBits(t, second.FloatState[0], 3)

	replica := ad.New()
	if err := replica.Restore(second); err != nil {
		t.Fatal(err)
	}
	second.FloatState[0] = -999
	mustBits(t, replica.Snapshot().FloatState[0], 3)
	if _, _, err := replica.Push(standardBars()[1]); err != nil {
		t.Fatal(err)
	}
	mustBits(t, primary.Snapshot().FloatState[0], 3)
}

// TestSnapshot_ZeroCountCanonicalization anchors zero-value Stream equivalence.
func TestSnapshot_ZeroCountCanonicalization(t *testing.T) {
	stream := ad.New()
	if _, _, err := stream.Push(standardBars()[0]); err != nil {
		t.Fatal(err)
	}
	initial := ad.New().Snapshot()
	if err := stream.Restore(initial); err != nil {
		t.Fatal(err)
	}
	mustSnapshotEqual(t, stream.Snapshot(), initial)

	// IEEE -0 compares equal to +0 for the documented zero-count invariant.
	// It is canonicalized to +0 so restored unseeded streams equal New/Stream{}.
	initial.FloatState[0] = math.Copysign(0, -1)
	initial.FloatState[1] = math.Copysign(0, -1)
	if err := stream.Restore(initial); err != nil {
		t.Fatal(err)
	}
	mustSnapshotEqual(t, stream.Snapshot(), ad.New().Snapshot())
	value, valid, err := stream.Push(standardBars()[0])
	if err != nil {
		t.Fatal(err)
	}
	mustEqualBool(t, valid, true, "reset first valid")
	mustBits(t, value, 3)
}

// TestSnapshot_RestoredCorrectionAffectsNextOutput proves sum alone is insufficient.
func TestSnapshot_RestoredCorrectionAffectsNextOutput(t *testing.T) {
	stream := ad.New()
	for _, bar := range []ad.Bar{
		{Open: 1.5, Low: 1, High: 2, Close: 2, Volume: 1e16},
		{Open: 1.5, Low: 1, High: 2, Close: 2, Volume: 1},
	} {
		if _, _, err := stream.Push(bar); err != nil {
			t.Fatal(err)
		}
	}
	checkpoint := stream.Snapshot()
	mustBits(t, checkpoint.FloatState[0], 1e16)
	mustBits(t, checkpoint.FloatState[1], 1)
	replica := ad.New()
	if err := replica.Restore(checkpoint); err != nil {
		t.Fatal(err)
	}
	terminal := ad.Bar{Open: 1.5, Low: 1, High: 2, Close: 1, Volume: 1e16}
	originalValue, _, err := stream.Push(terminal)
	if err != nil {
		t.Fatal(err)
	}
	replicatedValue, _, err := replica.Push(terminal)
	if err != nil {
		t.Fatal(err)
	}
	mustBits(t, originalValue, 1)
	mustBits(t, replicatedValue, 1)
}
