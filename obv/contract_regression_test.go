// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package obv_test checks cross-boundary invariants with external-only probes.
// These tests target public ABI identity, error atomicity and accepted-bar counts.
package obv_test

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"testing"

	"github.com/lvlath/go/obv"
)

// TestSnapshotIdentityAndInitialStates computes the configuration fingerprint
// independently and proves that Count counts accepted candles, not transitions.
func TestSnapshotIdentityAndInitialStates(t *testing.T) {
	const algorithm = "obv/neumaier/v1"
	independentHash := sha256.Sum256([]byte(algorithm))

	stream := obv.New()
	empty := stream.Snapshot()
	mustEqualString(t, empty.Algorithm, algorithm, "algorithm identity")
	mustEqualString(t, empty.Config, hex.EncodeToString(independentHash[:]), "configuration digest")
	mustEqualInt(t, len(empty.FloatState), 3, "float ABI length")
	mustEqualInt(t, len(empty.IntState), 0, "integer ABI length")
	if empty.Count != 0 || math.Float64bits(empty.FloatState[0]) != 0 ||
		math.Float64bits(empty.FloatState[1]) != 0 ||
		math.Float64bits(empty.FloatState[2]) != 0 {
		t.Fatal("new stream did not publish canonical zero state")
	}

	// Even the largest finite first-bar volume contributes nothing.
	value, valid, err := stream.Push(obv.Bar{Open: 105, High: 107, Low: 104, Close: 106, Volume: math.MaxFloat64})
	if err != nil {
		t.Fatal(err)
	}
	mustEqualBool(t, valid, true, "first observation Valid")
	if math.Float64bits(value) != 0 {
		t.Fatalf("first-bar volume polluted OBV: %v", value)
	}
	seed := stream.Snapshot()
	if seed.Count != 1 || seed.FloatState[0] != 106 ||
		math.Float64bits(seed.FloatState[1]) != 0 || math.Float64bits(seed.FloatState[2]) != 0 {
		t.Fatalf("seed-only state is not canonical: %+v", seed)
	}

	// A flat close and a zero-volume change must both advance Count.
	if _, valid, err = stream.Push(obv.Bar{Open: 106, High: 108, Low: 105, Close: 106, Volume: math.MaxFloat64}); err != nil || !valid {
		t.Fatalf("flat close failed: valid=%t error=%v", valid, err)
	}
	if _, valid, err = stream.Push(obv.Bar{Open: 106, High: 108, Low: 105, Close: 108, Volume: 0}); err != nil || !valid {
		t.Fatalf("zero-volume change failed: valid=%t error=%v", valid, err)
	}
	final := stream.Snapshot()
	if final.Count != 3 || final.FloatState[0] != 108 ||
		math.Float64bits(final.FloatState[1]+final.FloatState[2]) != 0 {
		t.Fatalf("accepted-bar count drifted: %+v", final)
	}
}

// TestBatchReturnsNoPartialOnLateOverflow prevents accidental publication of
// otherwise successful prefix values when the last signed addition overflows.
func TestBatchReturnsNoPartialOnLateOverflow(t *testing.T) {
	columns := bars(
		[]float64{100, 101, 102},
		[]float64{math.MaxFloat64, math.MaxFloat64, math.MaxFloat64},
	)
	result, err := obv.Compute(columns)
	mustErrorIs(t, err, obv.ErrOverflow)
	mustNilState(t, result, true, "late-overflow column Result")

	candles := []obv.Bar{
		{Open: 100, High: 100, Low: 100, Close: 100, Volume: math.MaxFloat64},
		{Open: 101, High: 101, Low: 101, Close: 101, Volume: math.MaxFloat64},
		{Open: 102, High: 102, Low: 102, Close: 102, Volume: math.MaxFloat64},
	}
	rowResult, rowErr := obv.ComputeCandles(candles)
	mustErrorIs(t, rowErr, obv.ErrOverflow)
	mustNilState(t, rowResult, true, "late-overflow candle Result")
}

// TestNilReceiverSnapshotIsNotVersioned distinguishes the zero Snapshot{}
// returned by a nil pointer from the valid versioned Count=0 Stream{} state.
func TestNilReceiverSnapshotIsNotVersioned(t *testing.T) {
	var absent *obv.Stream
	nilSnapshot := absent.Snapshot()
	if nilSnapshot.Algorithm != "" || nilSnapshot.Config != "" ||
		nilSnapshot.Count != 0 || nilSnapshot.FloatState != nil || nilSnapshot.IntState != nil {
		t.Fatalf("nil receiver returned non-zero checkpoint: %+v", nilSnapshot)
	}
	var destination obv.Stream
	mustErrorIs(t, destination.Restore(nilSnapshot), obv.ErrStateVersion)
	if destination.Snapshot().Count != 0 {
		t.Fatal("invalid nil checkpoint mutated destination")
	}
}
