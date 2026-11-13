// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// PCG recurrence and DXSM output adapted from Go 1.23.4 math/rand/v2/pcg.go:
// Copyright 2023 The Go Authors. All rights reserved.
// Use of the upstream source is governed by a BSD-style license.
// Upstream: https://github.com/golang/go/blob/go1.23.4/src/math/rand/v2/pcg.go
// See rng/UPSTREAM_LICENSE.txt for the BSD license notice.

// Package rng pins the 128-bit PCG recurrence and little-endian seed derivation.
// This file implements unsigned wraparound without a dependency on a future stdlib PRNG.
package rng

import (
	"crypto/sha256"
	"encoding/binary"
	"math/bits"
)

const (
	// seedDomain identifies RNG initial-state derivation and excludes unrelated SHA contexts.
	seedDomain = "lvlath/rng/pcg-dxsm-v1"

	// seedWordBytes fixes each unsigned seed/path field to a little-endian 64-bit word.
	seedWordBytes = 8

	// seedPreimageBytes fixes the exact SHA-256 input length (22 + 8 + 8 bytes).
	seedPreimageBytes = len(seedDomain) + 2*seedWordBytes
)

const (
	// pcgMulHi/Lo form the upper/lower words of the fixed 128-bit PCG multiplier.
	pcgMulHi uint64 = 0x2360ed051fc65da4
	pcgMulLo uint64 = 0x4385df649fccf645

	// pcgIncHi/Lo form the upper/lower words of the fixed odd 128-bit PCG increment.
	pcgIncHi uint64 = 0x5851f42d4c957f2d
	pcgIncLo uint64 = 0x14057b7ef767814f

	// pcgDXSMMul is the fixed 64-bit DXSM high-word avalanche multiplier.
	pcgDXSMMul uint64 = 0xda942042e4dd58b5
)

// deriveState hashes the fixed domain, seed, and path into an initial 128-bit state.
// It never treats seed zero specially or accesses a process-global entropy source.
//
// Implementation:
//   - Stage 1: create a fixed 38-byte, stack-addressable input buffer.
//   - Stage 2: encode seed and path ID as separate little-endian uint64 words.
//   - Stage 3: interpret digest bytes 0:8 as hi and 8:16 as lo, both little-endian.
//
// Determinism:
//   - Same 38 input bytes always produce the same two initial state words.
//
// Complexity:
//   - Time O(1) for a fixed SHA-256 block, Space O(1).
//
// AI-Hints:
//   - Swapping hi/lo or changing the domain/NUL policy changes every golden vector.
func deriveState(seed, pathID uint64) (uint64, uint64) {
	var input [seedPreimageBytes]byte
	copy(input[:len(seedDomain)], seedDomain)
	binary.LittleEndian.PutUint64(input[len(seedDomain):len(seedDomain)+seedWordBytes], seed)
	binary.LittleEndian.PutUint64(input[len(seedDomain)+seedWordBytes:], pathID)
	digest := sha256.Sum256(input[:])
	return binary.LittleEndian.Uint64(digest[:seedWordBytes]),
		binary.LittleEndian.Uint64(digest[seedWordBytes : 2*seedWordBytes])
}

// Uint64 advances the 128-bit PCG state and mixes that NEW state with DXSM.
// It is the canonical raw-word kernel used by every other draw method.
//
// Implementation:
//   - Stage 1: multiply the prior low word and collect the carry into the high word.
//   - Stage 2: add the fixed 128-bit odd increment and publish the updated state.
//   - Stage 3: fold the updated high word into the updated odd low word via DXSM.
//
// Behavior highlights:
//   - Cross terms and the low-word addition carry are mandatory for modulo-2^128 math.
//   - Gaussian spare fields are not examined or changed.
//
// Inputs:
//   - None: the current hi/lo words supply the mutable recurrence state.
//
// Returns:
//   - uint64: one deterministic output word from the updated 128-bit state.
//
// Errors:
//   - None: all operations use unsigned modular arithmetic.
//
// Determinism:
//   - State update occurs before output; overflow wraps by Go's uint64 rules.
//
// Complexity:
//   - Time O(1), Space O(1), no explicit allocation, lock or system call.
//
// Notes:
//   - The method is not concurrency-safe on a shared pcgStream.
//
// AI-Hints:
//   - Using the old state or missing the carry breaks cross-version replay.
//   - The 128-bit multiplier high word participates through cross products only.
func (p *pcgStream) Uint64() uint64 {
	// The 64x64 low product contributes one full low word and one high carry.
	hi, lo := bits.Mul64(p.lo, pcgMulLo)

	// Cross products only affect the upper half modulo 2^128; high*high vanishes.
	hi += p.hi*pcgMulLo + p.lo*pcgMulHi

	// Low-word increment may carry into the high word; preserve that exact carry.
	lo, carry := bits.Add64(lo, pcgIncLo, 0)
	hi, _ = bits.Add64(hi, pcgIncHi, carry)

	// Commit the new state before output mixing, matching the pinned recurrence.
	p.hi, p.lo = hi, lo

	// DXSM is a 64-bit transform: every multiplication wraps modulo 2^64.
	hi ^= hi >> 32
	hi *= pcgDXSMMul
	hi ^= hi >> 48
	hi *= lo | 1
	return hi
}
