// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo validates experiment configuration before dispatch.
// This file freezes dimensions, worker capacity, levels and safe index bounds.
package montecarlo

import (
	"context"
	"math"
)

// outcomeSlotUpperWords bounds the payload of one queued sampleOutcome on
// supported 32-/64-bit Go implementations: one uint64 ID, three native words
// for the slice header, and two native words for the error interface. Each
// native word is at most eight bytes; the bound excludes allocator overhead.
// This is an addressability guard, not a charge to MaxSamplesBytes.
const outcomeSlotUpperWords uint64 = 1 + 3 + 2

// configuration is a private immutable copy of the validated execution policy.
// pathRows is used only when an actual Paths-by-d buffer is allocated.
// An unrestricted mean-only run retains uint64 path counters throughout.
type configuration struct {
	plan      Plan      // Value-copy of the requested experiment metadata.
	dimension int       // Fixed positive feature dimension of every path.
	workers   int       // Resolved worker count, capped at Paths.
	window    uint64    // Maximum issued-but-not-reduced paths.
	pathRows  int       // Checked native row count; zero in streaming-only mode.
	levels    []float64 // Validated levels; detached before returning configuration.
	keepRaw   bool      // Whether retention or exact quantiles need raw samples.
}

// validate resolves a Plan without executing Sample or allocating full path storage.
// Its stages preserve the public precedence: context, plan, sampler, dimension, budget.
// The only sampler method it calls is Dimension, exactly once on valid presence.
// Implementation:
//   - Stage 1: validate context, scalar options and quantile order before touching the sampler.
//   - Stage 2: resolve positive dimension, worker count and checked dense row indices.
//   - Stage 3: admit all required backing sizes and detach quantile levels.
//
// Behavior highlights:
//   - A large logical path count is valid in streaming-only mode.
//   - Native row conversion is performed only when raw storage is requested.
//
// Inputs:
//   - ctx: non-nil execution context; an already canceled context takes precedence.
//   - plan: read-only execution parameters, including caller-owned quantile levels.
//   - sampler: a non-nil Sampler whose Dimension is fixed during the experiment.
//
// Returns:
//   - configuration: detached levels and checked integer indices for any dense storage.
//   - error: classification of the first failing validation stage.
//
// Errors:
//   - ErrInvalidOption: nil context, invalid path/worker count, levels or dimension.
//   - ErrNilInput: absent sampler or recognized typed-nil sampler.
//   - ErrOverflow/ErrBudget: unaddressable sizes or insufficient optional buffer budget.
//   - Context cancellation: original context error before any input access.
//
// Determinism:
//   - Quantile levels are inspected in their original order; no sorting occurs.
//
// Complexity:
//   - Time O(q), Space O(q), q=len(plan.Quantiles); no path vectors are generated.
//
// Notes: the caller must keep the Plan slice stable during Run.
// AI-Hints:
//   - Do not convert Paths to int for mean-only experiments; uint64 is the public domain.
//   - Without reflect, a typed-nil custom Sampler must implement IsNil for detection.
func validate(ctx context.Context, plan Plan, sampler Sampler) (configuration, error) {
	// Context validation precedes all reads of a possibly invalid sampler.
	if ctx == nil {
		return configuration{}, ErrInvalidOption
	}
	if err := ctx.Err(); err != nil {
		return configuration{}, err
	}

	// Validate scalar options before the potentially expensive sampler.Dimension call.
	if plan.Paths < minimumPaths {
		return configuration{}, ErrInvalidOption
	}
	// Resolve signed worker input once; a negative value is an invalid option.
	workerCount, err := nonnegativeUint64(plan.Workers)
	if err != nil {
		return configuration{}, ErrInvalidOption
	}
	if err := validateQuantileLevels(plan.Quantiles); err != nil {
		return configuration{}, err
	}
	if sampler == nil {
		return configuration{}, ErrNilInput
	}
	// core.Nilable-like detection is optional for third-party sampler types.
	if nilable, ok := sampler.(interface{ IsNil() bool }); ok && nilable.IsNil() {
		return configuration{}, ErrNilInput
	}
	constDimension := sampler.Dimension()
	// Convert the dimension at its semantic validation boundary, not in each
	// buffer calculation. Zero and negative dimensions share one public error.
	dimensionCount, err := nonnegativeUint64(constDimension)
	if err != nil || dimensionCount == 0 {
		return configuration{}, ErrInvalidOption
	}

	// Normalize zero workers to one; positive values are never silently clamped
	// except to the maximum number of paths that can execute concurrently.
	workers := plan.Workers
	if workers == 0 {
		workers = defaultWorkers
		workerCount = defaultWorkers
	}
	if workerCount > plan.Paths {
		// plan.Paths < workerCount <= MaxInt, so its conversion is checked here.
		workers, err = checkedInt(plan.Paths)
		if err != nil {
			return configuration{}, err
		}
		workerCount = plan.Paths
	}
	if workers > math.MaxInt/schedulerWindowFactor {
		return configuration{}, ErrOverflow
	}
	// The buffered outcome channel has one slot per worker. Its metadata is
	// not covered by the optional sample budget, but its backing byte product
	// must still fit the native address domain before starting any goroutine.
	if _, err := checkedBytes(workerCount, outcomeSlotUpperWords); err != nil {
		return configuration{}, err
	}

	// Whether samples are kept publicly or only for sorting, preflight proves
	// all backing sizes representable before any int-based row indexing.
	keepRaw := plan.RetainSamples || len(plan.Quantiles) != 0
	c := configuration{
		plan:      plan,
		dimension: constDimension,
		workers:   workers,
		window:    workerCount * schedulerWindowFactor,
		levels:    plan.Quantiles, // Read-only during preflight; copied after admission.
		keepRaw:   keepRaw,
	}
	if keepRaw {
		// A materialized row count is a native int, unlike logical path IDs.
		// Reject impossible rows before computing dependent backing products.
		c.pathRows, err = checkedInt(plan.Paths)
		if err != nil {
			return configuration{}, err
		}
	}
	if err := checkBudget(c, dimensionCount); err != nil {
		return configuration{}, err
	}
	// No detached level allocation is needed for a rejected experiment.
	c.levels = append([]float64(nil), plan.Quantiles...)
	return c, nil
}
