// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap_test anchors per-replicate replay and detached index ownership.
// The output equality oracle is exact: integer samples have no numeric tolerance.
package bootstrap_test

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/lvlath/go/bootstrap"
)

// TestSampleIndices_DeterministicPrefixPreservation rejects any RNG stream
// coupling to total replicate count for each of the three supported methods.
func TestSampleIndices_DeterministicPrefixPreservation(t *testing.T) {
	for _, kind := range []bootstrap.Kind{
		bootstrap.MovingBlocks, bootstrap.CircularBlocks, bootstrap.StationaryBlocks,
	} {
		short := bootstrap.Plan{Kind: kind, Length: 97, BlockLength: 8, Replicates: 2, Seed: 42}
		first, err := bootstrap.SampleIndices(context.Background(), 29, short)
		if err != nil {
			t.Fatal(err)
		}
		long := short
		long.Replicates = 5
		second, err := bootstrap.SampleIndices(context.Background(), 29, long)
		if err != nil {
			t.Fatal(err)
		}
		mustSameRows(t, second.Indices[:2], first.Indices, kindName(kind)+" prefix")
		mustEqualString(t, string(first.RNG), string(second.RNG), "algorithm identity under R extension")
	}
}

// TestSampleIndices_SeedZeroReproducible makes Seed=0 a stable explicit input,
// not a trigger for entropy derived from the system clock or process state.
func TestSampleIndices_SeedZeroReproducible(t *testing.T) {
	for _, kind := range []bootstrap.Kind{
		bootstrap.MovingBlocks, bootstrap.CircularBlocks, bootstrap.StationaryBlocks,
	} {
		plan := bootstrap.Plan{Kind: kind, Length: 96, BlockLength: 11, Replicates: 7, Seed: 0}
		first, err := bootstrap.SampleIndices(context.Background(), 83, plan)
		if err != nil {
			t.Fatal(err)
		}
		second, err := bootstrap.SampleIndices(context.Background(), 83, plan)
		if err != nil {
			t.Fatal(err)
		}
		mustSameRows(t, first.Indices, second.Indices, kindName(kind)+" seed zero")
	}
}

// TestSampleIndices_DetachedRows verifies that caller mutation does not affect
// subsequent rows or later calls using the same deterministic seed and plan.
func TestSampleIndices_DetachedRows(t *testing.T) {
	plan := bootstrap.Plan{Kind: bootstrap.StationaryBlocks, Length: 128, BlockLength: 9, Replicates: 4, Seed: 7}
	result, err := bootstrap.SampleIndices(context.Background(), 53, plan)
	if err != nil {
		t.Fatal(err)
	}
	backup, err := bootstrap.SampleIndices(context.Background(), 53, plan)
	if err != nil {
		t.Fatal(err)
	}
	result.Indices[0][0] = -1
	result.Indices[0][1] = -2
	mustResultShape(t, backup, 53, 128, 4)
	for row := 1; row < result.Completed; row++ {
		for col, got := range result.Indices[row] {
			if expected := backup.Indices[row][col]; got != expected {
				t.Fatalf("row alias at (%d,%d): got=%d want=%d", row, col, got, expected)
			}
		}
	}
}

// TestSampleIndices_ConcurrentReplay ensures each invocation owns its own
// streams and result buffers and does not depend on goroutine scheduling.
func TestSampleIndices_ConcurrentReplay(t *testing.T) {
	const workers = 12
	plan := bootstrap.Plan{Kind: bootstrap.CircularBlocks, Length: 307, BlockLength: 17, Replicates: 9, Seed: 0}
	anchor, err := bootstrap.SampleIndices(context.Background(), 97, plan)
	if err != nil {
		t.Fatal(err)
	}

	failure := make(chan error, workers)
	var wg sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			result, callErr := bootstrap.SampleIndices(context.Background(), 97, plan)
			if callErr != nil {
				failure <- callErr
				return
			}
			for row, indices := range result.Indices {
				for col, index := range indices {
					if index != anchor.Indices[row][col] {
						failure <- fmt.Errorf("concurrent replay mismatch at (%d,%d)", row, col)
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	close(failure)
	for err := range failure {
		t.Error(err)
	}
}

// TestSampleIndices_IndependentReplicatePaths confirms that each output row
// is reproducible across complete runs and unaffected by another request's
// subsequent mutation or a different number of requested replicate rows.
//
// This is an exact replay oracle, not a claim that a specific PRNG is
// cryptographically independent across path identifiers.
func TestSampleIndices_IndependentReplicatePaths(t *testing.T) {
	for _, kind := range []bootstrap.Kind{
		bootstrap.MovingBlocks, bootstrap.CircularBlocks, bootstrap.StationaryBlocks,
	} {
		plan := bootstrap.Plan{
			Kind: kind, Length: 1536, BlockLength: 19, Replicates: 8, Seed: 0,
		}
		baseline, err := bootstrap.SampleIndices(context.Background(), 67, plan)
		if err != nil {
			t.Fatal(err)
		}
		baseline.Indices[0][0] = -1
		plan.Replicates = 12
		extended, err := bootstrap.SampleIndices(context.Background(), 67, plan)
		if err != nil {
			t.Fatal(err)
		}
		for r := 1; r < 8; r++ {
			mustSameRows(t, [][]int{extended.Indices[r]}, [][]int{baseline.Indices[r]}, kindName(kind))
		}
	}
}
