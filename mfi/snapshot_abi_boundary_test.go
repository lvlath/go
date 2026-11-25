// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Snapshot ABI boundary checks use only mfi's exported constructor and methods.
// The digest oracle deliberately serializes the period independently of the package.
package mfi_test

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"testing"

	"github.com/lvlath/go/mfi"
)

// TestSnapshotABI_CursorEpochAndCheckpoint verifies the relationship between
// Count, transition count and physical ring head at every meaningful boundary.
// It also verifies the stable LE64 digest and exact continuation after Restore.
func TestSnapshotABI_CursorEpochAndCheckpoint(t *testing.T) {
	for _, period := range []int{1, 2, 3, 14, 257, 1000} {
		t.Run(fmt.Sprintf("period=%d", period), func(t *testing.T) {
			stream, err := mfi.NewPeriod(period)
			if err != nil {
				t.Fatal(err)
			}
			var bytes [8]byte
			binary.LittleEndian.PutUint64(bytes[:], uint64(period))
			material := append([]byte("mfi/rolling-neumaier/v2"), bytes[:]...)
			digest := sha256.Sum256(material)
			wantConfig := hex.EncodeToString(digest[:])

			type checkpoint struct {
				at    int
				state mfi.Snapshot
			}
			checkpoints := make([]checkpoint, 0, 12)
			bars := make([]mfi.Bar, 2*period+5)
			values := make([]float64, len(bars))
			valid := make([]bool, len(bars))
			for i := range bars {
				price := 100 + float64((i*7)%13)
				volume := 5 + float64(i%7)
				if i%17 == 0 {
					volume = 0 // This still consumes a physical transition slot.
				}
				bars[i] = mfi.Bar{Open: price, High: price, Low: price, Close: price, Volume: volume}
				values[i], valid[i], err = stream.Push(bars[i])
				if err != nil {
					t.Fatalf("push[%d]: %v", i, err)
				}
				snapshot := stream.Snapshot()
				if snapshot.Count != uint64(i+1) || snapshot.Config != wantConfig {
					t.Fatalf("checkpoint[%d]: count=%d config=%q", i, snapshot.Count, snapshot.Config)
				}
				expectedUsed := i
				if expectedUsed > period {
					expectedUsed = period
				}
				if snapshot.IntState[0] != uint64(expectedUsed) || snapshot.IntState[1] != uint64(i%period) {
					t.Fatalf("cursor[%d]: got=%v, want used=%d head=%d", i, snapshot.IntState, expectedUsed, i%period)
				}
				if i > 0 && i%period == 0 && snapshot.IntState[2] != 0 {
					t.Fatalf("scheduled rebase retained unscheduled flag at transition %d", i)
				}
				if i == 0 || i == 1 || i == period-1 || i == period || i == period+1 ||
					i == 2*period-1 || i == 2*period || i == 2*period+1 {
					checkpoints = append(checkpoints, checkpoint{at: i, state: snapshot})
				}
			}

			for _, item := range checkpoints {
				standby, newErr := mfi.NewPeriod(period)
				if newErr != nil {
					t.Fatal(newErr)
				}
				if err = standby.Restore(item.state); err != nil {
					t.Fatalf("restore[%d]: %v", item.at, err)
				}
				for j := item.at + 1; j < len(bars); j++ {
					value, ready, pushErr := standby.Push(bars[j])
					if pushErr != nil || ready != valid[j] || math.Float64bits(value) != math.Float64bits(values[j]) {
						t.Fatalf("replay checkpoint=%d bar=%d got=(%v,%t,%v) want=(%v,%t)",
							item.at, j, value, ready, pushErr, values[j], valid[j])
					}
				}
			}
		})
	}
}

// TestSnapshotABI_MaxPeriod checks the resource ceiling's LE64 fingerprint
// without generating millions of bar updates or changing the accepted limit.
func TestSnapshotABI_MaxPeriod(t *testing.T) {
	stream, err := mfi.NewPeriod(mfi.MaxPeriod)
	if err != nil {
		t.Fatal(err)
	}
	state := stream.Snapshot()
	var encoded [8]byte
	binary.LittleEndian.PutUint64(encoded[:], uint64(mfi.MaxPeriod))
	input := append([]byte("mfi/rolling-neumaier/v2"), encoded[:]...)
	sum := sha256.Sum256(input)
	if state.Config != hex.EncodeToString(sum[:]) {
		t.Fatalf("maximum-period digest changed: %s", state.Config)
	}
	if state.Count != 0 || len(state.FloatState) != 5+2*mfi.MaxPeriod ||
		state.IntState[0] != 0 || state.IntState[1] != 0 || state.IntState[2] != 0 {
		t.Fatal("maximum-period snapshot shape/cursor changed")
	}

	// The full resource ceiling is an ABI edge case, not merely an allocation
	// test: the physical head must preserve bit 19 immediately before wrapping.
	// The zero-volume input is valid, occupies a slot, and keeps sums exactly zero.
	zeroFlow := mfi.Bar{Open: 100, High: 100, Low: 100, Close: 100, Volume: 0}
	for i := 0; i < mfi.MaxPeriod; i++ {
		if _, _, err = stream.Push(zeroFlow); err != nil {
			t.Fatalf("maximum-period push[%d]: %v", i, err)
		}
	}
	beforeWrap := stream.Snapshot()
	if beforeWrap.IntState[0] != uint64(mfi.MaxPeriod-1) ||
		beforeWrap.IntState[1] != uint64(mfi.MaxPeriod-1) {
		t.Fatalf("maximum-period cursor before wrap: %v", beforeWrap.IntState)
	}
	value, ready, err := stream.Push(zeroFlow)
	if err != nil || !ready || value != 50 {
		t.Fatalf("scheduled maximum-period rebase: value=%v valid=%t err=%v", value, ready, err)
	}
	afterWrap := stream.Snapshot()
	if afterWrap.IntState[0] != uint64(mfi.MaxPeriod) || afterWrap.IntState[1] != 0 ||
		afterWrap.IntState[2] != 0 || afterWrap.Config != state.Config {
		t.Fatalf("maximum-period cursor after wrap: %v", afterWrap.IntState)
	}
}
