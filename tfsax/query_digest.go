// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax derives a stable query-set identity independent of input query ordering.
// This file protects frozen benchmark queries from accidental or post-hoc replacement.
package tfsax

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"math"
	"sort"
)

const queryHashDomain = "lvlath-tfsax-query-set-v1\x00" // Domain separation from raw corpus hashes.

// QueriesHash commits to stable query IDs, family labels, TopK and raw float64 samples.
// Input query order does not affect identity because query IDs determine execution order.
//
// Implementation:
//   - Stage 1: copy and sort query records by unsigned query ID.
//   - Stage 2: reject duplicates, unknown families, missing samples and non-finite values.
//   - Stage 3: SHA-256 encode a domain tag, each ID, family, TopK, length and sample bits.
//
// Behavior highlights:
//   - The hash distinguishes +0 from -0 and changes in TopK or declared family.
//   - Reordering entries with the same IDs and values does not change this digest.
//
// Inputs:
//   - queries: nonempty read-only named queries; each has finite nonempty Values.
//
// Returns:
//   - Lowercase 64-character SHA-256 hex identity, or empty string with error.
//
// Errors:
//   - ErrInvalidInput for empty set, unknown family or invalid TopK/window length.
//   - ErrDuplicateID for a repeated query ID; ErrNonFinite for NaN/Inf samples.
//
// Determinism:
//   - Ascending numeric query IDs and ascending sample indices; all integer words are LE64.
//
// Complexity:
//   - Time O(Q log Q + sum(query lengths)), additional memory O(Q) for sorted records.
//
// Notes:
//   - This hash does not prove truthful raw-time provenance or authenticate an untrusted source.
//
// AI-Hints:
//   - Commit to query identity before tuning and never include timing in this data digest.
func QueriesHash(queries []NamedQuery) (string, error) {
	if len(queries) == 0 {
		return "", ErrInvalidInput
	}
	ordered := append([]NamedQuery(nil), queries...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].ID < ordered[j].ID })
	hasher := sha256.New()
	_, _ = hasher.Write([]byte(queryHashDomain))
	for index, named := range ordered {
		if index != 0 && named.ID == ordered[index-1].ID {
			return "", ErrDuplicateID
		}
		if !knownFamily(named.Family) {
			return "", ErrInvalidInput
		}
		if err := ValidateQuery(named.Query, len(named.Query.Values)); err != nil {
			return "", err
		}
		writeHashWord(hasher, named.ID)
		writeHashWord(hasher, uint64(len(named.Family)))
		_, _ = hasher.Write([]byte(named.Family))
		// ValidateQuery rejects nonpositive TopK. Preserve that local admission
		// boundary before writing a fixed-width query identity.
		if named.Query.TopK < 1 {
			return "", ErrInvalidInput
		}

		// Every Go int fits in int64, including on 32-bit targets. For accepted
		// positive TopK values, little-endian int64 and uint64 have identical
		// eight-byte encodings. This preserves the established hash format
		// without introducing a signed-to-unsigned integer conversion.
		if err := binary.Write(hasher, binary.LittleEndian, int64(named.Query.TopK)); err != nil {
			return "", errors.Join(ErrInvalidInput, err)
		}
		writeHashWord(hasher, uint64(len(named.Query.Values)))
		for _, sample := range named.Query.Values {
			writeHashWord(hasher, math.Float64bits(sample))
		}
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}
