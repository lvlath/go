// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm constructs canonical versioned identities for immutable models.
// This boundary hashes normalized IEEE-754 parameter bits in deterministic row-major order.
package hmm

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"hash"
	"math"
)

// modelIdentityDomain separates SHA-256 input bytes for this HMM identity revision.
// Any modification of the canonical byte grammar must also change this domain
// to prevent a stored FilterState from being accepted by a changed model.
const modelIdentityDomain = "lvlath/hmm/v1/"

// hashParameters derives the canonical SHA-256 identity of immutable model parameters.
// It ties state compatibility to algorithm revision, emission kind, dimensions, and exact copied bits.
//
// Implementation:
//   - Stage 1: Initialize SHA-256 with the versioned domain and emission-family tag.
//   - Stage 2: Write K, emission dimension and length-prefixed parameter arrays.
//   - Stage 3: Canonicalize signed zero and encode the digest as lowercase hexadecimal.
//
// Behavior highlights:
//   - The identifier detects accidental state/model mixing; it is not a signature or authorization token.
//
// Inputs:
//   - kind: family tag; k, dim: positive dimensions; parts: normalized immutable row-major parameter slices.
//
// Returns:
//   - string: 64 hexadecimal SHA-256 characters determined by supplied canonical values.
//
// Errors:
//   - None; only validated finite model values are passed to this internal helper.
//
// Determinism:
//   - Fixed domain, part order and big-endian IEEE-754 bit layout.
//
// Complexity:
//   - Time O(total parameter cells), Space O(1) aside from hash state and result.
//
// Notes:
//   - Hashing inputs does not freeze caller memory; constructors first detach every parameter.
//
// AI-Hints:
//   - Any change to the canonical byte grammar requires a new revision tag and state incompatibility.
func hashParameters(kind string, k, dim int, parts ...[]float64) string {
	h := sha256.New()
	_, _ = h.Write([]byte(modelIdentityDomain + kind))
	writeNumber(h, uint64(k))
	writeNumber(h, uint64(dim))
	for _, p := range parts {
		writeNumber(h, uint64(len(p)))
		for _, v := range p {
			if v == 0 {
				v = 0
			}
			writeNumber(h, math.Float64bits(v))
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// writeNumber appends a single unsigned 64-bit value to a canonical model digest.
// Its fixed big-endian representation avoids architecture-dependent serialization.
//
// Implementation:
//   - Stage 1: Encode the integer into a local eight-byte array.
//   - Stage 2: Write exactly eight bytes into the running hash.
//   - Stage 3: Leave the caller's numeric value and previous hash bytes untouched.
//
// Behavior highlights:
//   - The output is a hash input segment, not a complete external serialization format.
//
// Inputs:
//   - h: private hash.Hash accumulator; v: integer or prevalidated float64 bit pattern.
//
// Returns:
//   - No return value; the accumulator receives an ordered eight-byte segment.
//
// Errors:
//   - None under the fixed in-memory crypto/sha256 writer contract.
//
// Determinism:
//   - Big-endian octet order is stable across CPU architectures.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Only hashParameters calls this with a valid writable digest.
//
// AI-Hints:
//   - Do not encode native int size or byte order into ModelID; that breaks restore validation.
func writeNumber(h hash.Hash, v uint64) {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], v)
	_, _ = h.Write(b[:])
}
