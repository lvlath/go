// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap_test exercises the public facade with structural index oracles.
// No test accesses private kernels or derives expected blocks from random chance.
package bootstrap_test

import (
	"context"
	"errors"
	"testing"

	"github.com/lvlath/go/bootstrap"
	"github.com/lvlath/go/core"
	"github.com/lvlath/go/rng"
)

// core.Nilable compatibility is asserted through the public type, not reflection.
var _ core.Nilable = (*bootstrap.Result)(nil)

// TestSampleIndices_ExactFixtures checks deterministic degenerate cases and
// parses every fixed block independently of its random starting value.
func TestSampleIndices_ExactFixtures(t *testing.T) {
	for _, kind := range []bootstrap.Kind{
		bootstrap.MovingBlocks, bootstrap.CircularBlocks, bootstrap.StationaryBlocks,
	} {
		t.Run(kindName(kind), func(t *testing.T) {
			result, err := bootstrap.SampleIndices(context.Background(), 1, bootstrap.Plan{
				Kind: kind, Length: 5, BlockLength: 1, Replicates: 3, Seed: 0,
			})
			if err != nil {
				t.Fatal(err)
			}
			mustResultShape(t, result, 1, 5, 3)
			for replicate, indices := range result.Indices {
				for col, index := range indices {
					if index != 0 {
						t.Fatalf("replicate=%d position=%d: want zero, got %d", replicate, col, index)
					}
				}
			}
		})
	}

	moving, err := bootstrap.SampleIndices(context.Background(), 5, bootstrap.Plan{
		Kind: bootstrap.MovingBlocks, Length: 5, BlockLength: 3,
		Replicates: 32, Seed: 7,
	})
	if err != nil {
		t.Fatal(err)
	}
	mustResultShape(t, moving, 5, 5, 32)
	for _, indices := range moving.Indices {
		// A length-five row is exactly [full block 3, truncated block 2].
		checkMovingBlock(t, indices[:3], 5, 3)
		checkMovingBlock(t, indices[3:], 5, 3)
	}

	circular, err := bootstrap.SampleIndices(context.Background(), 5, bootstrap.Plan{
		Kind: bootstrap.CircularBlocks, Length: 5, BlockLength: 3,
		Replicates: 32, Seed: 7,
	})
	if err != nil {
		t.Fatal(err)
	}
	mustResultShape(t, circular, 5, 5, 32)
	for _, indices := range circular.Indices {
		checkCircularBlock(t, indices[:3], 5)
		checkCircularBlock(t, indices[3:], 5)
	}
}

// checkMovingBlock validates a known block boundary without trying to infer
// boundaries from adjacent values, which can coincide across different blocks.
func checkMovingBlock(t *testing.T, block []int, rows, fullBlockLength int) {
	t.Helper()
	if len(block) == 0 || block[0] < 0 || block[0] > rows-fullBlockLength {
		t.Fatalf("invalid moving-block start: %v; rows=%d fullBlockLength=%d", block, rows, fullBlockLength)
	}
	for offset, got := range block {
		if got != block[0]+offset {
			t.Fatalf("broken moving block at %d: %v", offset, block)
		}
	}
}

// checkCircularBlock validates cyclic continuation at a known block boundary.
func checkCircularBlock(t *testing.T, block []int, rows int) {
	t.Helper()
	if len(block) == 0 || block[0] < 0 || block[0] >= rows {
		t.Fatalf("invalid circular-block start: %v", block)
	}
	expected := block[0]
	for offset, got := range block {
		if got != expected {
			t.Fatalf("broken circular block at %d: %v", offset, block)
		}
		if expected == rows-1 {
			expected = 0
		} else {
			expected++
		}
	}
}

// kindName gives stable case names without requiring a public Kind.String API.
func kindName(kind bootstrap.Kind) string {
	switch kind {
	case bootstrap.MovingBlocks:
		return "moving"
	case bootstrap.CircularBlocks:
		return "circular"
	case bootstrap.StationaryBlocks:
		return "stationary"
	default:
		return "invalid"
	}
}

// TestSampleIndices_IndexBounds samples diverse legal dimensions, including
// smaller output than B and rows=1, to protect bounds and row-length laws.
func TestSampleIndices_IndexBounds(t *testing.T) {
	shapes := []struct {
		rows, length, blockLength, replicates int
	}{
		{1, 1, 1, 1}, {2, 17, 2, 9}, {5, 1, 5, 7},
		{7, 43, 3, 12}, {100, 1025, 17, 5}, {1024, 64, 64, 3},
	}
	for _, kind := range []bootstrap.Kind{
		bootstrap.MovingBlocks, bootstrap.CircularBlocks, bootstrap.StationaryBlocks,
	} {
		for _, shape := range shapes {
			result, err := bootstrap.SampleIndices(context.Background(), shape.rows, bootstrap.Plan{
				Kind: kind, Length: shape.length, BlockLength: shape.blockLength,
				Replicates: shape.replicates, Seed: ^uint64(0),
			})
			if err != nil {
				t.Fatalf("kind=%d shape=%+v: %v", kind, shape, err)
			}
			mustResultShape(t, result, shape.rows, shape.length, shape.replicates)
		}
	}
}

// TestSampleIndices_ValidationPrecedence anchors the canonical context->options
// ->shape->overflow->budget order without classifying error strings.
func TestSampleIndices_ValidationPrecedence(t *testing.T) {
	valid := bootstrap.Plan{Kind: bootstrap.MovingBlocks, Length: 16, BlockLength: 4, Replicates: 2}
	bad := []struct {
		name string
		rows int
		plan bootstrap.Plan
	}{
		{"unknown kind", 5, bootstrap.Plan{Kind: 255, Length: 16, BlockLength: 4, Replicates: 2}},
		{"zero rows", 0, valid},
		{"negative rows", -1, valid},
		{"zero length", 5, bootstrap.Plan{Kind: bootstrap.MovingBlocks, BlockLength: 4, Replicates: 2}},
		{"zero replicates", 5, bootstrap.Plan{Kind: bootstrap.MovingBlocks, Length: 16, BlockLength: 4}},
		{"zero block", 5, bootstrap.Plan{Kind: bootstrap.MovingBlocks, Length: 16, Replicates: 2}},
		{"block exceeds rows", 3, valid},
		{"stationary block exceeds rows", 3, bootstrap.Plan{Kind: bootstrap.StationaryBlocks, Length: 16, BlockLength: 4, Replicates: 2}},
		{"circular block exceeds rows", 3, bootstrap.Plan{Kind: bootstrap.CircularBlocks, Length: 16, BlockLength: 4, Replicates: 2}},
	}
	for _, tc := range bad {
		t.Run(tc.name, func(t *testing.T) {
			result, err := bootstrap.SampleIndices(context.Background(), tc.rows, tc.plan)
			mustErrorIs(t, err, bootstrap.ErrInvalidOption)
			mustNilState(t, result, true, "invalid plan result")
		})
	}

	result, err := bootstrap.SampleIndices(nil, 5, valid)
	mustErrorIs(t, err, bootstrap.ErrInvalidOption)
	mustNilState(t, result, true, "nil context result")

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	result, err = bootstrap.SampleIndices(canceled, 0, bootstrap.Plan{Kind: 255})
	mustErrorIs(t, err, context.Canceled)
	mustNilState(t, result, true, "pre-canceled result")
	if errors.Is(err, bootstrap.ErrInvalidOption) {
		t.Fatal("context cancellation must take precedence over invalid options")
	}
}

// TestSampleIndices_BudgetAndOverflowProtection ensures huge plans fail without
// attempting result allocation or consuming a random stream.
func TestSampleIndices_BudgetAndOverflowProtection(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	cases := []struct {
		name string
		plan bootstrap.Plan
		want error
	}{
		{"row byte overflow", bootstrap.Plan{Length: maxInt, BlockLength: 1, Replicates: 1}, bootstrap.ErrOverflow},
		{"replicate byte overflow", bootstrap.Plan{Length: maxInt / 64, BlockLength: 1, Replicates: 512}, bootstrap.ErrOverflow},
		{"large valid arithmetic", bootstrap.Plan{Length: 1 << 25, BlockLength: 1, Replicates: 3}, bootstrap.ErrBudget},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := bootstrap.SampleIndices(context.Background(), 8, tc.plan)
			mustErrorIs(t, err, tc.want)
			mustNilState(t, result, true, "allocation rejection")
		})
	}
}

// TestSampleIndices_WrappersParity confirms that the three named conveniences
// are exact projections of the canonical Plan, including their metadata.
func TestSampleIndices_WrappersParity(t *testing.T) {
	const rows, length, blockLength, replicates = 17, 39, 4, 5
	const seed uint64 = 14
	cases := []struct {
		kind bootstrap.Kind
		call func(context.Context, int, int, int, int, uint64) (*bootstrap.Result, error)
	}{
		{bootstrap.MovingBlocks, bootstrap.SampleMovingIndices},
		{bootstrap.CircularBlocks, bootstrap.SampleCircularIndices},
		{bootstrap.StationaryBlocks, bootstrap.SampleStationaryIndices},
	}
	for _, tc := range cases {
		expected, err := bootstrap.SampleIndices(context.Background(), rows, bootstrap.Plan{
			Kind: tc.kind, Length: length, BlockLength: blockLength,
			Replicates: replicates, Seed: seed,
		})
		if err != nil {
			t.Fatal(err)
		}
		actual, err := tc.call(context.Background(), rows, length, blockLength, replicates, seed)
		if err != nil {
			t.Fatal(err)
		}
		mustSameRows(t, actual.Indices, expected.Indices, kindName(tc.kind))
		mustEqualString(t, string(actual.RNG), string(expected.RNG), "wrapper RNG identity")
	}
}

// TestSampleIndices_BlockOneMethodEquivalence anchors the strongest available
// version-independent IID oracle: with B=1 all three methods must consume
// the same one uniform source draw for each destination index, with identical
// replay streams and without additional Bernoulli words.
func TestSampleIndices_BlockOneMethodEquivalence(t *testing.T) {
	for _, seed := range []uint64{0, 1, 2026, ^uint64(0)} {
		plan := bootstrap.Plan{
			Kind: bootstrap.MovingBlocks, Length: 257,
			BlockLength: 1, Replicates: 5, Seed: seed,
		}
		moving, err := bootstrap.SampleIndices(context.Background(), 11, plan)
		if err != nil {
			t.Fatal(err)
		}
		plan.Kind = bootstrap.CircularBlocks
		circular, err := bootstrap.SampleIndices(context.Background(), 11, plan)
		if err != nil {
			t.Fatal(err)
		}
		plan.Kind = bootstrap.StationaryBlocks
		stationary, err := bootstrap.SampleIndices(context.Background(), 11, plan)
		if err != nil {
			t.Fatal(err)
		}
		mustSameRows(t, moving.Indices, circular.Indices, "moving vs circular B=1")
		mustSameRows(t, moving.Indices, stationary.Indices, "moving vs stationary B=1")
	}
}

// TestSampleIndices_CircularWholeRowPermutation proves that a circular block
// of length rows visits every source position once before a new block starts.
func TestSampleIndices_CircularWholeRowPermutation(t *testing.T) {
	const rows = 17
	result, err := bootstrap.SampleIndices(context.Background(), rows, bootstrap.Plan{
		Kind: bootstrap.CircularBlocks, Length: 2*rows + 5,
		BlockLength: rows, Replicates: 40, Seed: 314,
	})
	if err != nil {
		t.Fatal(err)
	}
	mustResultShape(t, result, rows, 2*rows+5, 40)
	for _, indices := range result.Indices {
		for start := 0; start < len(indices); start += rows {
			stop := start + rows
			if stop > len(indices) {
				stop = len(indices)
			}
			checkCircularBlock(t, indices[start:stop], rows)
			if stop-start == rows {
				seen := make([]bool, rows)
				for _, index := range indices[start:stop] {
					if seen[index] {
						t.Fatalf("repeated observation in one circular full-row block: %v", indices[start:stop])
					}
					seen[index] = true
				}
			}
		}
	}
}

// TestSampleIndices_AlgorithmIDRevision verifies that the public result names
// the installed RNG factory revision and the bootstrap mapping version together.
// It does not falsely pin the identity of a temporary test-double generator.
func TestSampleIndices_AlgorithmIDRevision(t *testing.T) {
	result, err := bootstrap.SampleIndices(context.Background(), 9, bootstrap.Plan{
		Kind: bootstrap.StationaryBlocks, Length: 14,
		BlockLength: 3, Replicates: 2, Seed: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Validate the contract against the installed factory rather than a version
	// string from a fixture generator. A factory upgrade requires a new replay ID.
	const indexMappingRevision = "/bootstrap-index-v1"
	expected := string(rng.NewFactory().Version()) + indexMappingRevision
	mustEqualString(t, string(result.RNG), expected, "accepted RNG revision")
}

// TestSampleIndices_MaxSourceRows proves that large source-index domains do not
// force a correspondingly large allocation or overflow a circular increment.
func TestSampleIndices_MaxSourceRows(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, kind := range []bootstrap.Kind{
		bootstrap.MovingBlocks, bootstrap.CircularBlocks, bootstrap.StationaryBlocks,
	} {
		result, err := bootstrap.SampleIndices(context.Background(), maxInt, bootstrap.Plan{
			Kind: kind, Length: 9, BlockLength: maxInt,
			Replicates: 2, Seed: 0,
		})
		if err != nil {
			t.Fatalf("kind=%d MaxInt rows: %v", kind, err)
		}
		mustResultShape(t, result, maxInt, 9, 2)
	}
}

// TestSampleIndices_NativeIntegerDomain exercises the bounded-word narrowing
// path at the largest legal native-int source count. The result allocation
// remains tiny; the source-domain size must not be mistaken for output size.
func TestSampleIndices_NativeIntegerDomain(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	for _, kind := range []bootstrap.Kind{
		bootstrap.MovingBlocks, bootstrap.CircularBlocks, bootstrap.StationaryBlocks,
	} {
		plan := bootstrap.Plan{
			Kind: kind, Length: 2048, BlockLength: 1, Replicates: 2, Seed: ^uint64(0),
		}
		res, err := bootstrap.SampleIndices(context.Background(), maxInt, plan)
		if err != nil {
			t.Fatalf("kind=%d maximum native-int domain: %v", kind, err)
		}
		mustResultShape(t, res, maxInt, plan.Length, plan.Replicates)
	}
}

// TestSampleIndices_UniformSingleSourcePosition proves all three algorithms
// produce only zero when there is one available observation, including long
// block paths and multiple logical replicates.
func TestSampleIndices_UniformSingleSourcePosition(t *testing.T) {
	for _, kind := range []bootstrap.Kind{
		bootstrap.MovingBlocks, bootstrap.CircularBlocks, bootstrap.StationaryBlocks,
	} {
		res, err := bootstrap.SampleIndices(context.Background(), 1, bootstrap.Plan{
			Kind: kind, Length: 1129, BlockLength: 1, Replicates: 3, Seed: 180,
		})
		if err != nil {
			t.Fatal(err)
		}
		mustResultShape(t, res, 1, 1129, 3)
		for r, row := range res.Indices {
			for pos, value := range row {
				if value != 0 {
					t.Fatalf("singleton domain: replicate=%d position=%d index=%d", r, pos, value)
				}
			}
		}
	}
}
