// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw provides inclusive-limit alignment over the existing canonical DTW recurrence.
// This file classifies completed results without introducing an alternate DTW engine.
package dtw

import (
	"context"
	"errors"
	"math"
	"math/bits"
)

// AlignBounded aligns two scalar sequences against an inclusive accumulated-cost limit.
// It preserves Align's local-cost metric, window, penalty, and strict path witness;
// the initial exact-first implementation computes the complete alignment before classification.
//
// Implementation:
//   - Stage 1: check context, assemble options exactly once, and validate input/domain order.
//   - Stage 2: invoke the existing classified scalar recurrence and preserve numeric errors.
//   - Stage 3: distinguish structural no-path, exact score within the limit, and above-limit.
//
// Behavior highlights:
//   - A score equal to limit yields BoundExact; AboveLimit never exposes a finite approximate score.
//   - No premature abandonment occurs; this preserves overflow classification and full accuracy.
//   - ReturnLocalCost/ReturnAccumulated cannot be published in BoundedResult and are rejected.
//
// Inputs:
//   - ctx: non-nil context; cancellation is checked before and after the full kernel.
//   - a, b: caller-owned input sequences; concurrent mutation during the call is unsupported.
//   - limit: non-negative finite float64 or +Inf, in the configured accumulated cost units.
//   - opts: ordinary DTW options, applied in the original order without replaying user callbacks.
//
// Returns:
//   - *BoundedResult: detached outcome with optional Path and total admitted CellsVisited.
//   - BoundUnreachable uses two +Inf fields; AboveLimit uses NaN Distance and an exact lower bound.
//
// Errors:
//   - ErrInvalidOption: nil context, invalid limit, or unsupported artifact-output request.
//   - ErrNilOption and existing DTW option/input errors retain their published sentinel identities.
//   - ErrOverflow joined with ErrNaNInf: unrepresentable terminal optimum; no result.
//   - context.Canceled/DeadlineExceeded: context stopped before or during the invocation.
//
// Determinism:
//   - Uses Align's row-major recurrence and strict diagonal/up/left tie order.
//
// Complexity:
//   - Time O(n*m), Space O(m) for distance-only, O(n*m) for path/full-matrix modes.
//   - Counting visited band cells adds O(n) time and O(1) storage after computation.
//
// Notes:
//   - This is an exact-first implementation, not an optimized early-abandonment guarantee.
//   - It does not add fields to the existing exported Result or Options structs.
//
// AI-Hints:
//   - Do not turn a completed overflow into BoundAboveLimit merely because the limit is smaller.
//   - Never call Align again with the same Option values: custom options may have stateful assembly.
func AlignBounded(ctx context.Context, a, b []float64, limit float64, opts ...Option) (*BoundedResult, error) {
	if ctx == nil {
		return nil, ErrInvalidOption
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	// Preserve canonical option validation priority over malformed sequence inputs.
	cfg, err := applyOptions(opts...)
	if err != nil {
		return nil, err
	}
	if cfg.returnAccumulated || cfg.returnLocalCost {
		return nil, ErrInvalidOption
	}

	// Validate the sequence before the independently invalid limit, as specified
	// for new context-bearing entry points. Validation is intentionally repeated
	// by compute to preserve its existing internal contract and error identities.
	if err = validateSequences(a, b, cfg.validateFinite); err != nil {
		return nil, err
	}
	if math.IsNaN(limit) || math.IsInf(limit, -1) || limit < 0 {
		return nil, ErrInvalidOption
	}

	// The existing classified recurrence detects terminal overflow and preserves
	// a predecessor witness. In particular, a discarded overflow is not fatal.
	result, err := compute(a, b, cfg)
	if stopped := ctx.Err(); stopped != nil {
		return nil, stopped
	}
	if result == nil {
		return nil, err
	}
	if err != nil && !(errors.Is(err, ErrNoPath) && !result.Reachable) {
		return nil, err
	}

	// The existing kernel evaluates local cost on every admitted cell, including
	// structurally unreachable interior cells, but never on out-of-band cells.
	visited, countErr := countAdmittedCells(len(a), len(b), cfg.window)
	if countErr != nil {
		return nil, countErr
	}

	if !result.Reachable {
		return &BoundedResult{
			Status: BoundUnreachable, Distance: math.Inf(1),
			LowerBound: math.Inf(1), CellsVisited: visited,
		}, nil
	}
	if result.Distance > limit {
		return &BoundedResult{
			Status: BoundAboveLimit, Distance: math.NaN(),
			LowerBound: result.Distance, CellsVisited: visited,
		}, nil
	}
	return &BoundedResult{
		Status: BoundExact, Distance: result.Distance,
		LowerBound: result.Distance, Path: result.Path, CellsVisited: visited,
	}, nil
}

// countAdmittedCells counts the cost evaluations performed by the current full-grid kernel.
// It uses inclusive row band endpoints and checked uint64 addition rather than n*m overflow.
//
// Implementation:
//   - Stage 1: compute each row's band intersection with the column domain.
//   - Stage 2: add its non-negative width with carry detection.
//   - Stage 3: return the exact count or ErrOverflow if it exceeds uint64.
//
// Behavior highlights:
//   - Window=-1 admits the full rectangle; Window=0 admits only matching indices.
//
// Inputs:
//   - rows, cols: finite nonnegative lengths; radius: -1 or nonnegative.
//
// Returns:
//   - uint64: exact number of admitted DP cells.
//
// Errors:
//   - ErrOverflow: uint64 count cannot be represented.
//
// Determinism:
//   - Scans rows increasing, with no floating-point operations.
//
// Complexity:
//   - Time O(rows), Space O(1).
//
// Notes:
//   - This helper describes current computation, not a promise of early-abandonment.
//
// AI-Hints:
//   - Do not estimate window complexity as n*(2r+1) without clipping boundary rows.
//   - Avoid i+radius integer overflow for very large index domains.
func countAdmittedCells(rows, cols, radius int) (uint64, error) {
	var count uint64
	for i := 0; i < rows; i++ {
		start, end := 0, cols
		if radius >= 0 {
			if i > radius {
				start = i - radius
			}
			// end is exclusive; this subtraction form cannot overflow int.
			if i < cols && radius < cols-1-i {
				end = i + radius + 1
			}
		}
		if start >= end {
			continue
		}
		var carry uint64
		count, carry = bits.Add64(count, uint64(end-start), 0)
		if carry != 0 {
			return 0, errors.Join(ErrOverflow, ErrNaNInf)
		}
	}
	return count, nil
}
