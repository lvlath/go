// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng_test anchors literal seed and PCG golden reference vectors.
// This file accesses the implementation only through the exported RNG facade.
package rng_test

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"testing"

	"github.com/lvlath/go/rng"
)

// Independently pinned fixtures from lvlath-spec-20-v1, not from package helpers.
// TestSeedVectors_GoldenReference verifies published digest hi/lo fixtures
// and first-two-word golden vectors without importing implementation internals.
func TestSeedVectors_GoldenReference(t *testing.T) {
	tests := []struct{ seed, path, hi, lo, a, b uint64 }{
		{0, 0, 0x959f7cde48806db1, 0xf075daaa9e93c7fe, 0xde616b2b746d339a, 0x8351bd653a53bc01},
		{0, 1, 0x13acb9e304cdcdb7, 0x4212ef527710d91e, 0x6bd7526a9750e27d, 0xbd45572342df7046},
		{1, 0, 0xb4710acc1507334a, 0x3dbed5e0dcdf52df, 0x22fabff5ca8f6db0, 0x2e9409e6e6c2bca2},
		{^uint64(0), ^uint64(0), 0x9040449b8608bb02, 0x259a7f6793588e02, 0xdf0d2e5fdaa45333, 0x24cf2f9c39b42ed0},
	}
	f := rng.NewFactory()
	for _, tc := range tests {
		input := make([]byte, 0, 38)
		input = append(input, []byte("lvlath/rng/pcg-dxsm-v1")...)
		var tmp [8]byte
		binary.LittleEndian.PutUint64(tmp[:], tc.seed)
		input = append(input, tmp[:]...)
		binary.LittleEndian.PutUint64(tmp[:], tc.path)
		input = append(input, tmp[:]...)
		if len(input) != 38 {
			t.Fatalf("unexpected length %d", len(input))
		}
		sum := sha256.Sum256(input)
		gotHi, gotLo := binary.LittleEndian.Uint64(sum[:8]), binary.LittleEndian.Uint64(sum[8:16])
		if gotHi != tc.hi || gotLo != tc.lo {
			t.Errorf("SHA fixture %x/%x: got %016x %016x digest=%s", tc.seed, tc.path, gotHi, gotLo, hex.EncodeToString(sum[:]))
		}
		stream := f.Stream(tc.seed, tc.path)
		if a, b := stream.Uint64(), stream.Uint64(); a != tc.a || b != tc.b {
			t.Errorf("PCG fixture %x/%x got %016x %016x want %016x %016x", tc.seed, tc.path, a, b, tc.a, tc.b)
		}
	}
}
