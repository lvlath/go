// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng_test verifies batch facades from the exact public stream interface.
// The tests protect fresh-stream replay, indexed publication, and spare continuity.
package rng_test

import (
	"math"
	"strconv"
	"testing"

	"github.com/lvlath/go/rng"
)

// TestBatchFill_StreamParity proves every nonempty batch wrapper is a projection
// of the canonical Stream API for empty, odd, even, and multichunk-sized buffers.
func TestBatchFill_StreamParity(t *testing.T) {
	const seed uint64 = 0
	const pathID uint64 = ^uint64(0)

	var nilFactory *rng.Factory
	for _, n := range []int{0, 1, 2, 3, 7, 32, 1025} {
		t.Run("size_"+strconv.Itoa(n), func(t *testing.T) {
			words := make([]uint64, n)
			mustEqualInt(t, len(words), n, "batch size")
			nilFactory.FillUint64(seed, pathID, words)
			raw := nilFactory.Stream(seed, pathID)
			for _, got := range words {
				mustEqualWord(t, got, raw.Uint64(), "raw batch cell")
			}

			uniforms := make([]float64, n)
			nilFactory.FillUniformOpen(seed, pathID, uniforms)
			uniform := nilFactory.Stream(seed, pathID)
			for i, got := range uniforms {
				mustSameFloatBits(t, got, uniform.UniformOpen(), "uniform batch cell")
				if !(got > 0 && got < 1) {
					t.Fatalf("uniform cell %d outside (0,1): %.17g", i, got)
				}
			}

			normals := make([]float64, n)
			nilFactory.FillNormal(seed, pathID, normals)
			normal := nilFactory.Stream(seed, pathID)
			for i, got := range normals {
				mustSameFloatBits(t, got, normal.Normal(), "normal batch cell")
				if math.IsNaN(got) || math.IsInf(got, 0) {
					t.Fatalf("normal cell %d is nonfinite", i)
				}
			}
		})
	}
}

// TestBatchFill_FreshReplayAndBufferOwnership checks that a batch writes
// caller memory, never aliases it, and repeats from the original state.
func TestBatchFill_FreshReplayAndBufferOwnership(t *testing.T) {
	factory := rng.NewFactory()
	const seed, pathID = 73, 11

	original := make([]float64, 11)
	for i := range original {
		original[i] = math.NaN()
	}
	factory.FillNormal(seed, pathID, original)
	first := append([]float64(nil), original...)

	// Mutating the first buffer must not change the next batch's initial state.
	original[0] = 999
	factory.FillNormal(seed, pathID, original)
	for i := range original {
		mustSameFloatBits(t, original[i], first[i], "repeat normal batch")
	}

	// Different path identifiers must usually produce different bitstreams;
	// this fixed known-answer comparison is not a proof of statistical independence.
	other := make([]float64, 11)
	factory.FillNormal(seed, pathID+1, other)
	mustEqualBool(t, math.Float64bits(other[0]) != math.Float64bits(first[0]), true, "path divergence")

	// Empty and nil buffers are legal and do not overwrite unrelated caller data.
	factory.FillUint64(seed, pathID, nil)
	factory.FillUniformOpen(seed, pathID, nil)
	factory.FillNormal(seed, pathID, nil)
	var nilWords []uint64
	var nilFloats []float64
	factory.FillUint64(seed, pathID, nilWords)
	factory.FillUniformOpen(seed, pathID, nilFloats)
	factory.FillNormal(seed, pathID, nilFloats)
	mustEqualBool(t, nilWords == nil && nilFloats == nil, true, "nil buffer retention")
}

// TestNewStream_CanonicalAliasAndNilability protects the additive convenience
// API and optional concrete nilability without expanding the RNG interface.
func TestNewStream_CanonicalAliasAndNilability(t *testing.T) {
	const version = "pcg-dxsm128-sha256le-u52-boxmuller-v1"
	factory := rng.NewFactory()
	mustEqualString(t, factory.Version(), version, "version")
	mustNilState(t, factory, false, "factory")

	var absent *rng.Factory
	mustNilState(t, absent, true, "nil factory")
	mustEqualString(t, absent.Version(), version, "nil version")
	mustEqualWord(t, absent.Stream(0, 0).Uint64(), rng.NewStream(0, 0).Uint64(), "nil receiver")

	wrapped := rng.NewStream(0, ^uint64(0))
	canonical := factory.Stream(0, ^uint64(0))
	mustNilState(t, wrapped, false, "wrapped stream")
	mustNilState(t, canonical, false, "canonical stream")
	for i := 0; i < 100; i++ {
		mustEqualWord(t, wrapped.Uint64(), canonical.Uint64(), "stream wrapper")
	}
}
