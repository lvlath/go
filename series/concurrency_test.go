// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External concurrency tests exercise independent calls with shared immutable input.
// They do not claim concurrent caller mutation is supported.

package series_test

import (
	"sync"
	"testing"

	"github.com/lvlath/go/series"
)

// TestSeries_ConcurrentReadOnlyCalls checks independently allocated results
// when callers share an immutable source array. No input writes are performed.
func TestSeries_ConcurrentReadOnlyCalls(t *testing.T) {
	// The source is shared read-only across all workers; returned storage is not.
	values := []float64{1, 2, 4, 8, 16, 32}
	times := []int64{0, 10, 20, 30, 40, 50}
	// Wait for every invocation before completing this race-test boundary.
	var wg sync.WaitGroup
	for worker := 0; worker < 12; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for iteration := 0; iteration < 20; iteration++ {
				grid, err := series.Validate(times, values, series.GridOptions{CadenceNS: 10})
				if err != nil || grid == nil || grid.Samples != len(values) {
					t.Errorf("Validate: result=%v err=%v", grid, err)
					return
				}
				result, err := series.LogReturns(values)
				if err != nil || result == nil || !result.Valid[5] {
					t.Errorf("LogReturns: result=%v err=%v", result, err)
					return
				}
				result.Values[5] = 100 // only the caller's output is mutated.
			}
		}()
	}
	wg.Wait()
}
