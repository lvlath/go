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
	// Positive Go int dimensions can be widened to int64 without any loss.
	// Their eight-byte big-endian encoding is identical to the previous uint64 layout.
	writeDimension(h, k)
	writeDimension(h, dim)
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

// writeDimension appends a Go int dimension in a platform-independent eight-byte form.
// It preserves the canonical byte encoding of integer dimensions used by ModelID.
//
// Implementation:
//   - Stage 1: Widen dimension from int to int64 without losing any representable int value.
//   - Stage 2: Encode exactly eight big-endian bytes into a fixed-size local buffer.
//   - Stage 3: Feed those bytes into the same ordered SHA-256 accumulator.
//
// Behavior highlights:
//   - Unlike conversion to uint64, conversion to int64 cannot overflow on Go's 32- or 64-bit targets.
//   - Positive dimensions have exactly the same bytes as the former unsigned encoding.
//
// Inputs:
//   - h: non-nil private SHA-256 digest; dimension: validated positive state or emission dimension.
//
// Returns:
//   - No result; exactly eight canonical bytes are appended to h.
//
// Errors:
//   - None: binary.Encode accepts int64 in an eight-byte buffer; SHA-256 writes cannot fail.
//
// Determinism:
//   - Exactly eight big-endian bytes, independent of native int size and CPU byte order.
//
// Complexity:
//   - Time O(1), Space O(1), with a fixed eight-byte staging buffer.
//
// Notes:
//   - Existing FilterState values retain their ModelID; hmm provides no binary state serialization API.
//   - This helper is not a general-purpose encoder for externally supplied unvalidated data.
//
// AI-Hints:
//   - Do not hash native-width int bytes: a 32-bit process must agree with a 64-bit process.
//   - Do not change this representation without advancing modelIdentityDomain and state compatibility policy.
func writeDimension(h hash.Hash, dimension int) {
	var encoded [8]byte
	// The primitive int64 follows the non-reflective, fixed-width encoding path.
	_, _ = binary.Encode(encoded[:], binary.BigEndian, int64(dimension))
	_, _ = h.Write(encoded[:])
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
