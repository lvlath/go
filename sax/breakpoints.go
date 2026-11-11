// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File: breakpoints.go
// Role: Verify the pinned Gaussian binary64 asset and return detached thresholds.
// AI-Hint: Recomputing quantiles at runtime can invalidate persisted EncodingID values.

package sax

import (
	"compress/zlib"
	"crypto/sha256"
	_ "embed"
	"encoding/binary"
	"encoding/hex"
	"io"
	"math"
	"strings"
	"sync"
)

const (
	// quantileTableVersion identifies the offline standard-normal quantile data.
	// It participates in EncodingID; changes to threshold bits require a new version.
	quantileTableVersion = "gauss-ndtri-f64-v1"

	// quantileTableSHA256 is the SHA-256 of all 32,640 uncompressed little-endian
	// binary64 thresholds. It protects the shipped asset against silent changes.
	quantileTableSHA256 = "3042af4cd15a0f58d3d9fade0cd9c60a3a1dac1b47533a2996f4b4099a9c4f94"

	// quantileAssetSHA256 pins the compressed bytes as a reproducible release
	// asset, independently of the mathematically relevant uncompressed hash.
	quantileAssetSHA256 = "e830f112f33562c8b97db392b2274a131074f7592782268f9a4edb6fc3ef7b34"

	// quantileTableBytes is the exact decoded size: 8 * sum(a-1), a = 2..256.
	quantileTableBytes = 261120
	// quantileCount is the number of binary64 thresholds across all alphabets.
	quantileCount = quantileTableBytes / 8

	// maxAlphabet is the largest supported bin count and constrains all
	// uint16 code values to at most 255, without narrowing an unbounded int.
	maxAlphabet = 256

	// minAlphabet is the smallest useful discretization: two bins, one boundary.
	minAlphabet = 2
)

// compressedQuantiles holds the immutable zlib stream as an embedded string;
// a mutable byte slice must not be exposed to constructors or inference calls.
//
//go:embed quantiles.bin
var compressedQuantiles string

var (
	// quantileOnce serializes the single integrity-checked cache initialization.
	quantileOnce sync.Once

	// quantileData is published only after every checksum and size gate passes;
	// each Encoder receives a detached slice, never this shared backing storage.
	quantileData []float64

	// quantileError permanently classifies a failed embedded-resource admission.
	// It is read only after quantileOnce.Do provides synchronization.
	quantileError error
)

// pinnedBreakpoints returns a detached ascending breakpoint vector for alphabet.
// Every threshold is loaded from the original SHA-256-pinned binary64 table,
// rather than reevaluating a potentially platform-dependent inverse Gaussian CDF.
//
// Implementation:
//   - Stage 1: reject an invalid alphabet before indexing the table.
//   - Stage 2: initialize and verify the complete shared table exactly once.
//   - Stage 3: copy the contiguous (alphabet-1)-value region into private storage.
//
// Behavior highlights:
//   - The embedded compressed asset is immutable and decoded only once.
//   - The output never aliases the shared cached table.
//
// Inputs:
//   - alphabet: number of SAX bins in [2,256].
//
// Returns:
//   - []float64: a detached, strictly ascending threshold array of size a-1.
//
// Errors:
//   - ErrInvalidOption: invalid alphabet or damaged/corrupt embedded table.
//
// Determinism:
//   - Alphabet rows are stored in increasing alphabet order; each row is in
//     increasing quantile order, with unchanged IEEE-754 binary64 bits.
//
// Complexity:
//   - Time O(a), space O(a) per call after the first O(32640) decoding pass;
//     the full shared cache occupies 32640 float64 values.
//
// Notes:
//   - This private function does not expose a mutable view of shared data.
//
// AI-Hints:
//   - Do not replace the pinned table with math.Erfcinv: even a 1-ULP change
//     alters bin boundaries and therefore symbol words and EncodingID.
func pinnedBreakpoints(alphabet int) ([]float64, error) {
	if alphabet < minAlphabet || alphabet > maxAlphabet {
		return nil, ErrInvalidOption
	}

	// A sync.Once gives immutable encoder inference a single read-only data cache.
	// Any integrity failure remains stable for every subsequent constructor.
	quantileOnce.Do(loadQuantileTable)
	if quantileError != nil {
		return nil, quantileError
	}

	// The first 2..(a-1) alphabets occupy the triangular number of values.
	// The interval arithmetic uses only small validated ints (a <= 256).
	offset := (alphabet - minAlphabet) * (alphabet - 1) / 2
	out := make([]float64, alphabet-1)
	copy(out, quantileData[offset:offset+len(out)])
	return out, nil
}

// loadQuantileTable verifies the entire embedded numerical resource before
// publishing it. It performs one bounded decompression and two SHA-256 checks;
// a malformed asset is never allowed to leak a partially initialized cache.
// Called only through sync.Once, so publication does not require another lock.
func loadQuantileTable() {
	// The compressed payload itself is pinned to prevent silently accepting
	// extra bytes after a valid zlib stream (which zlib readers may ignore).
	assetSum := sha256.Sum256([]byte(compressedQuantiles))
	if hex.EncodeToString(assetSum[:]) != quantileAssetSHA256 {
		quantileError = ErrInvalidOption
		return
	}

	reader, err := zlib.NewReader(strings.NewReader(compressedQuantiles))
	if err != nil {
		quantileError = ErrInvalidOption
		return
	}

	// LimitReader caps decoded data at expected+1 bytes, so even an embedded
	// corrupt stream cannot force a giant allocation during validation. The
	// extra byte is a strict trailing-output detector, not part of the table.
	raw, readErr := io.ReadAll(io.LimitReader(reader, quantileTableBytes+1))
	closeErr := reader.Close()
	checksum := sha256.Sum256(raw)
	if readErr != nil || closeErr != nil || len(raw) != quantileTableBytes ||
		hex.EncodeToString(checksum[:]) != quantileTableSHA256 {
		quantileError = ErrInvalidOption
		return
	}

	// Only fully validated binary64 values can be published. A fixed-width
	// little-endian decode has no floating conversion or quantile approximation.
	decoded := make([]float64, quantileCount)
	for i := range decoded {
		decoded[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
	}
	quantileData = decoded
}
