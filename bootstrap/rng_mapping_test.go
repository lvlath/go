// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap_test compares public index output against independently
// decoded RNG words without access to any private implementation helper.
package bootstrap_test

import (
	"context"
	"strconv"
	"testing"

	"github.com/lvlath/go/bootstrap"
	"github.com/lvlath/go/rng"
)

// referenceWordSource describes the only observable RNG operation required by
// the independent oracle. It does not depend on the sampler's private adapter.
type referenceWordSource interface {
	Uint64() uint64
}

// referenceBoundedInteger implements the mathematical acceptance interval using
// a separate threshold identity and a standard-library integer parser.
//
// Implementation:
//   - Stage 1: Reject no words for the one-element domain.
//   - Stage 2: Draw until the word lies in a multiple-of-bound acceptance interval.
//   - Stage 3: Convert the accepted residue through decimal formatting/parsing.
//
// Behavior highlights:
//   - Decimal decoding is intentionally independent of the production nibble
//     reconstruction and lives only in test code, not the hot sampling path.
//
// Inputs:
//   - stream: private single-replicate stream supplied by the public RNG factory.
//   - bound: strictly positive native-int range limit, guaranteed by each fixture.
//
// Returns:
//   - int: the exact bounded sample; a test failure is raised on impossible parse.
//
// Determinism:
//   - Consumes the same accepted/rejected word sequence from the pinned stream.
//
// Complexity:
//   - Expected O(1) word draws and O(log(bound)) decimal decoding; test-only.
//
// AI-Hints:
//   - Do not call bootstrap's internal converter from this oracle.
//   - This establishes exact mapping/stream consumption, not PRNG independence.
func referenceBoundedInteger(t *testing.T, stream referenceWordSource, bound int) int {
	t.Helper()
	if bound == 1 {
		return 0
	}

	// Since MaxUint64=2^64-1, this alternate identity evaluates 2^64 mod b.
	b := uint64(bound)
	threshold := ((^uint64(0) % b) + 1) % b
	for {
		word := stream.Uint64()
		if word < threshold {
			continue
		}
		value, err := strconv.Atoi(strconv.FormatUint(word%b, 10))
		if err != nil {
			t.Fatalf("accepted residue is not representable in int: %v", err)
		}
		return value
	}
}

// TestSampleIndices_NativeRemainderExactOracle compares every output against an
// independent integer oracle for bounds across both 32-bit and 64-bit ranges.
// It covers nibble boundaries that a wrongly shifted decoder would corrupt.
//
// Implementation:
//   - Stage 1: Choose several crossing-boundary sizes up to native MaxInt.
//   - Stage 2: Sample the public B=1 MovingBlocks facade with fixed seeds.
//   - Stage 3: Replay the factory stream using an independent decimal oracle.
//
// Behavior highlights:
//   - A wrong nibble mask/shift is caught by exact per-index comparisons.
//   - No statistical tolerance or randomness-dependent threshold is used.
//
// Determinism:
//   - The expected row uses precisely Seed and pathID=replicate.
//
// Complexity:
//   - O(seeds*bounds*R*L) expected RNG words and fixed-size test memory.
//
// AI-Hints:
//   - Source-domain size is not output allocation size; MaxInt source rows
//     are legal for a very short result, and must not trigger ErrBudget.
func TestSampleIndices_NativeRemainderExactOracle(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	bounds := []int{
		1, 2, 3, 7, 15, 16, 17, 31, 32, 33,
		255, 256, 257, 4095, 4096, 4097, 10000,
		maxInt / 1024, maxInt / 16, maxInt/2 - 17, maxInt,
	}
	const rowLength = 41
	const replicates = 3
	for _, rows := range bounds {
		for _, seed := range []uint64{0, 1, 2026, ^uint64(0)} {
			plan := bootstrap.Plan{
				Kind: bootstrap.MovingBlocks, Length: rowLength,
				BlockLength: 1, Replicates: replicates, Seed: seed,
			}
			actual, err := bootstrap.SampleIndices(context.Background(), rows, plan)
			if err != nil {
				t.Fatalf("rows=%d seed=%d: %v", rows, seed, err)
			}
			mustResultShape(t, actual, rows, rowLength, replicates)
			factory := rng.NewFactory()
			pathID := uint64(0)
			for replicate, row := range actual.Indices {
				stream := factory.Stream(seed, pathID)
				pathID++
				for position, got := range row {
					want := referenceBoundedInteger(t, stream, rows)
					if got != want {
						t.Fatalf("rows=%d seed=%d replicate=%d position=%d: got=%d want=%d",
							rows, seed, replicate, position, got, want)
					}
				}
			}
		}
	}
}

// TestSampleIndices_BlockKernelWordOracle checks exact logical stream consumption
// and block transitions with an independent implementation of the three laws.
//
// Implementation:
//   - Stage 1: Sample one nontrivial non-IID block length for each Kind.
//   - Stage 2: Independently decode block starts and stationary restart trials.
//   - Stage 3: Compare every published index without deducing random boundaries.
//
// Determinism:
//   - The oracle uses the same public RNG Stream(seed, pathID), but no bootstrap
//     private helpers or test-only public exports.
//
// Complexity:
//   - Expected O(R*L) sampled words and comparisons, O(1) oracle workspace.
//
// AI-Hints:
//   - Do not infer stationary restart decisions from adjacent observed values;
//     a restart can legitimately choose the same successor as continuation.
func TestSampleIndices_BlockKernelWordOracle(t *testing.T) {
	const (
		rows       = 29
		blockSize  = 7
		rowLength  = 113
		replicates = 3
		seed       = uint64(0)
	)
	for _, kind := range []bootstrap.Kind{
		bootstrap.MovingBlocks, bootstrap.CircularBlocks, bootstrap.StationaryBlocks,
	} {
		plan := bootstrap.Plan{
			Kind: kind, Length: rowLength, BlockLength: blockSize,
			Replicates: replicates, Seed: seed,
		}
		actual, err := bootstrap.SampleIndices(context.Background(), rows, plan)
		if err != nil {
			t.Fatal(err)
		}
		pathID := uint64(0)
		for _, row := range actual.Indices {
			stream := rng.NewFactory().Stream(seed, pathID)
			pathID++
			var want []int
			if kind == bootstrap.StationaryBlocks {
				want = referenceStationaryPath(t, stream, rows, blockSize, rowLength)
			} else {
				want = referenceFixedBlockPath(t, stream, rows, blockSize, rowLength, kind)
			}
			mustSameRows(t, [][]int{row}, [][]int{want}, kindName(kind))
		}
	}
}

// referenceFixedBlockPath creates an independent fixed-block oracle whose block
// boundaries are explicit rather than inferred from repeated adjacent values.
// It preserves the final truncated block without extra uniform draws.
func referenceFixedBlockPath(
	t *testing.T,
	stream referenceWordSource,
	rows, blockLength, length int,
	kind bootstrap.Kind,
) []int {
	t.Helper()
	out := make([]int, 0, length)
	for len(out) < length {
		startBound := rows
		if kind == bootstrap.MovingBlocks {
			startBound = rows - blockLength + 1
		}
		index := referenceBoundedInteger(t, stream, startBound)
		for count := 0; count < blockLength && len(out) < length; count++ {
			out = append(out, index)
			index++
			if kind == bootstrap.CircularBlocks && index == rows {
				index = 0
			}
		}
	}
	return out
}

// referenceStationaryPath builds a stationary Markov-chain index sequence from
// independent rational restart trials. It uses no float64 probability threshold.
func referenceStationaryPath(t *testing.T, stream referenceWordSource, rows, blockLength, length int) []int {
	t.Helper()
	out := make([]int, length)
	previous := referenceBoundedInteger(t, stream, rows)
	out[0] = previous
	for position := 1; position < length; position++ {
		if referenceBoundedInteger(t, stream, blockLength) == 0 {
			previous = referenceBoundedInteger(t, stream, rows)
		} else {
			previous++
			if previous == rows {
				previous = 0
			}
		}
		out[position] = previous
	}
	return out
}
