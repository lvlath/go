// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap implements stationary block resampling by random restarts.
// The transition kernel has an exactly uniform stationary index distribution.
package bootstrap

import "context"

// fillStationary populates one unpublished row with a stationary block process.
// Each restart is Bernoulli with the rational probability 1/BlockLength.
//
// Implementation:
//   - Stage 1: Select a uniformly distributed first source index.
//   - Stage 2: Before each later output, draw a restart indicator; restart with
//     a fresh uniform index, or advance the previous index cyclically.
//   - Stage 3: Verify context and let the facade publish only a complete row.
//
// Behavior highlights:
//   - B=1 is exactly IID sampling; no meaningless Bernoulli draws are consumed.
//   - For B>1, UniformInteger(B)==0 is a rational, floating-point-free restart.
//   - Restart may randomly draw the same index as a cyclic continuation would.
//
// Inputs:
//   - ctx: non-nil context, checked for every random word and per 1024 indices.
//   - stream: single-owner word stream derived from the replicate's logical ID.
//   - dst: unpublished destination row with length at least one.
//   - rows: number of source observations >=1; blockLength: B in [1, rows].
//
// Returns:
//   - dst filled on nil error; intermediate dst must be discarded on error.
//
// Errors:
//   - ctx.Err(): cancellation while drawing or writing a row.
//
// Determinism:
//   - Initial uniform draw; then, per position, Bernoulli before optional
//     uniform restart; all decisions use the same replicate-local stream.
//
// Complexity:
//   - Expected Time O(len(dst)), Space O(1) beyond dst.
//
// Notes:
//   - Geometric block lengths are unbounded; BlockLength is their expectation.
//   - Stationary output indices do not prove validity of arbitrary statistical CIs.
//
// AI-Hints:
//   - Never substitute float64(p) comparisons for the exact rational trial.
//   - B=1 changes word consumption deliberately and is a versioned fast path.
func fillStationary(ctx context.Context, stream wordStream, dst []int, rows, blockLength int) error {
	// Mean block size one means each observation is an independent uniform index.
	if blockLength == 1 {
		for i := range dst {
			if err := ctx.Err(); err != nil {
				return err
			}
			index, err := uniformInteger(ctx, stream, rows)
			if err != nil {
				return err
			}
			dst[i] = index
		}
		return ctx.Err()
	}

	// The first position is uniform, which initializes the stationary chain.
	index, err := uniformInteger(ctx, stream, rows)
	if err != nil {
		return err
	}
	for i := range dst {
		if i%cancellationStride == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if i != 0 {
			// Bernoulli is evaluated BEFORE the candidate index for this position.
			restart, restartErr := uniformInteger(ctx, stream, blockLength)
			if restartErr != nil {
				return restartErr
			}
			if restart == 0 {
				// A restart is independent of the preceding index, including at wrap.
				index, err = uniformInteger(ctx, stream, rows)
				if err != nil {
					return err
				}
			} else if index == rows-1 {
				index = 0
			} else {
				index++
			}
		}
		dst[i] = index
	}
	return ctx.Err()
}
