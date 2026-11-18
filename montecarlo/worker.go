// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo isolates stochastic path generation from ordered reduction.
// This file owns per-path RNG streams, output validation and sample copying.
package montecarlo

import (
	"context"

	"github.com/lvlath/go/rng"
)

// sampleOutcome is a detached message for one accepted logical path.
// Only the coordinator owns reductions; workers publish a valid vector or error.
type sampleOutcome struct {
	id     uint64
	values []float64
	err    error
}

// worker consumes assigned IDs and emits one independently owned outcome each.
// It derives RNG state from Seed and pathID, never from the worker's identity
// or completion order, and forwards sampler errors without replacing the cause.
//
// Implementation:
//   - Stage 1: skip sampling if child context was already cancelled.
//   - Stage 2: invoke Sample with a newly created path stream and copy its vector.
//   - Stage 3: validate shape/finiteness of detached data and send exactly one outcome.
//
// Inputs:
//   - ctx: child cancellation context shared only for interruption signaling.
//   - sampler: safe for concurrent Sample calls when more than one worker runs.
//   - factory: immutable stream factory shared safely across goroutines.
//   - seed, dimension: immutable experiment metadata.
//   - jobs: unique IDs assigned by the coordinator; outcomes: result channel.
//
// Returns: no value; each accepted job produces exactly one channel result.
// Errors: ErrShape/ErrNonFinite classify invalid sampler output; sampler errors
//
//	and context cancellation are forwarded unmodified for errors.Is.
//
// Determinism: stream identity is Stream(seed,pathID); no floating reduction here.
// Complexity: Time O(d) and Space O(d) per successful path, excluding Sample.
// Notes: a sampler must not mutate a returned slice after returning from Sample.
// AI-Hints: never recover a sampler panic and substitute an all-zero vector.
func worker(ctx context.Context, sampler Sampler, factory *rng.Factory, seed uint64,
	dimension int, jobs <-chan uint64, outcomes chan<- sampleOutcome,
) {
	for pathID := range jobs {
		outcome := sampleOutcome{id: pathID}
		if err := ctx.Err(); err != nil {
			outcome.err = err
		} else {
			// Each call owns exactly one mutable RNG instance until Sample returns.
			values, err := sampler.Sample(ctx, pathID, factory.Stream(seed, pathID))
			switch {
			case err != nil:
				outcome.err = err
			case len(values) != dimension:
				outcome.err = ErrShape
			default:
				// Copy first so subsequent verification and publishing never read an
				// external slice after ownership has been transferred back to the sampler.
				outcome.values = append([]float64(nil), values...)
				for _, value := range outcome.values {
					if !finiteScalar(value) {
						outcome.values = nil
						outcome.err = ErrNonFinite
						break
					}
				}
			}
		}
		// The coordinator always drains accepted outcomes, including after cancel.
		outcomes <- outcome
	}
}
