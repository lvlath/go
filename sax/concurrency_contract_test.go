// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Concurrency tests use externally constructed immutable inputs and never
// invoke testing.Fatal from a worker goroutine. race must be run in CI.
package sax_test

import (
	"fmt"
	"math"
	"slices"
	"sync"
	"testing"

	"github.com/lvlath/go/sax"
)

func TestEncoder_ConcurrentInferenceAndDetachedMutation(t *testing.T) {
	opts := sax.DefaultOptions(12)
	opts.ReturnPAA = true
	enc, err := sax.NewEncoder(opts)
	if err != nil {
		t.Fatal(err)
	}
	values := make([]float64, 127)
	for i := range values {
		values[i] = math.Sin(float64(i)*0.15) + math.Cos(float64(i)*0.07)
	}
	reference, err := enc.Encode(values)
	if err != nil {
		t.Fatal(err)
	}
	const workers, iterations = 12, 20
	failures := make(chan error, workers)
	var group sync.WaitGroup
	for worker := 0; worker < workers; worker++ {
		group.Add(1)
		go func(worker int) {
			defer group.Done()
			for iteration := 0; iteration < iterations; iteration++ {
				result, encodeErr := enc.Encode(values)
				if encodeErr != nil {
					failures <- fmt.Errorf("worker %d: Encode: %w", worker, encodeErr)
					return
				}
				if !slices.Equal(result.Symbols, reference.Symbols) || result.EncodingID != reference.EncodingID {
					failures <- fmt.Errorf("worker %d: non-deterministic encoding", worker)
					return
				}
				bound, compareErr := enc.Mindist(reference, result)
				if compareErr != nil || bound != 0 {
					failures <- fmt.Errorf("worker %d: bound=%g compareErr=%w", worker, bound, compareErr)
					return
				}
				// Mutate only the caller-owned result after comparison; other workers
				// and the immutable model must remain unaffected by this mutation.
				result.Symbols[0] = 65535
				result.PAA[0] = math.NaN()
			}
		}(worker)
	}
	group.Wait()
	close(failures)
	for failure := range failures {
		t.Error(failure)
	}
	fresh, err := enc.Encode(values)
	if err != nil || !slices.Equal(fresh.Symbols, reference.Symbols) {
		t.Fatalf("model corrupted by caller-owned Result mutation: %v", err)
	}
}
