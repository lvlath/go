// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo validates experiment configuration before dispatch.
// This file freezes dimensions, worker capacity, levels and safe index bounds.
package montecarlo

import (
	"context"
	"math"
)

// configuration is a private immutable copy of the validated execution policy.
// pathRows is used only when an actual Paths-by-d buffer is allocated.
// An unrestricted mean-only run retains uint64 path counters throughout.
type configuration struct {
	plan      Plan
	dimension int
	workers   int
	window    uint64
	pathRows  int
	levels    []float64
	keepRaw   bool
}

// validate resolves a Plan without executing Sample or allocating full path storage.
// Its stages preserve the public precedence: context, plan, sampler, dimension, budget.
// The only sampler method it calls is Dimension, exactly once on valid presence.
//
// Inputs:
//   - ctx: non-nil execution context; an already cancelled context takes precedence.
//   - plan: read-only execution parameters, including caller-owned quantile levels.
//   - sampler: a non-nil Sampler whose Dimension is fixed during the experiment.
//
// Returns:
//   - configuration: detached levels and checked integer indices for any dense storage.
//   - error: classification of the first failing validation stage.
//
// Determinism:
//   - Quantile levels are inspected in their original order; no sorting occurs.
//
// Complexity:
//   - Time O(q), Space O(q), q=len(plan.Quantiles); no path vectors are generated.
//
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
	if plan.Paths < minimumPaths || plan.Workers < 0 {
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
	if constDimension <= 0 {
		return configuration{}, ErrInvalidOption
	}

	// Normalize zero workers to one; positive values are never silently clamped
	// except to the maximum number of paths that can execute concurrently.
	workers := plan.Workers
	if workers == 0 {
		workers = defaultWorkers
	}
	workerCount, err := nonnegativeUint64(workers)
	if err != nil {
		return configuration{}, err
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

	// Whether samples are kept publicly or only for sorting, preflight proves
	// all backing sizes representable before any int-based row indexing.
	keepRaw := plan.RetainSamples || len(plan.Quantiles) != 0

	// Freeze the caller-owned levels before any worker can start. Plan remains a
	// value-copy metadata record; only the private c.levels slice is consulted.
	c := configuration{
		plan:      plan,
		dimension: constDimension,
		workers:   workers,
		window:    workerCount * schedulerWindowFactor,
		pathRows:  0,
		levels:    append([]float64(nil), plan.Quantiles...),
		keepRaw:   keepRaw,
	}
	if err := checkBudget(c); err != nil {
		return configuration{}, err
	}
	if keepRaw {
		// Since dimension is positive, checkedInt(Paths*d) in checkBudget
		// already guarantees Paths is in the native int range.
		c.pathRows, err = checkedInt(plan.Paths)
		if err != nil {
			return configuration{}, err
		}
	}
	return c, nil
}
