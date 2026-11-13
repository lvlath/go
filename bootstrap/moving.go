// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap implements fixed-length moving-block index resampling.
// Every block is contiguous in the source, without circular continuation.
package bootstrap

import "context"

// cancellationStride limits the length of a block segment between context checks.
// A context check also occurs at each block start and inside every random draw.
const cancellationStride = 1024

// fillMoving populates one unpublished row with uniformly selected moving blocks.
// Its final block is truncated to the remaining requested output length.
//
// Implementation:
//   - Stage 1: Check context; select a uniform start from [0, rows-B].
//   - Stage 2: Emit min(B, remaining) consecutive source positions.
//   - Stage 3: Repeat until full and verify context before returning.
//
// Behavior highlights:
//   - No block may cross rows-1 or wrap to zero.
//   - A truncated final block still consumes exactly one block-start draw.
//   - B=rows has only one start and consumes no RNG words for the start.
//
// Inputs:
//   - ctx: active context; inspected per block and per cancellationStride indices.
//   - stream: exclusive source of 64-bit random words for this replicate.
//   - dst: private, preallocated output buffer of the requested row length.
//   - rows: positive source observation count; B is in [1, rows].
//   - blockLength: fixed block size B, independent of destination length.
//
// Returns:
//   - The filled dst on nil error; dst content is invalid on error.
//
// Errors:
//   - ctx.Err() on interruption, including a failed uniform draw.
//
// Determinism:
//   - Blocks and destinations are visited strictly left-to-right.
//
// Complexity:
//   - Expected Time O(len(dst)), Space O(1) beyond dst.
//
// Notes:
//   - The caller must validate dimensions; dst is never published by this kernel.
//
// AI-Hints:
//   - Do not substitute wrap-around for short blocks at the source boundary.
//   - Check cancellation before publication, even if the final write completed.
func fillMoving(ctx context.Context, stream wordStream, dst []int, rows, blockLength int) error {
	for offset := 0; offset < len(dst); {
		// One fixed block begins only after cancellation is ruled out.
		if err := ctx.Err(); err != nil {
			return err
		}

		// rows-B+1 is safe: 1<=B<=rows<=MaxInt by the validated plan.
		start, err := uniformInteger(ctx, stream, rows-blockLength+1)
		if err != nil {
			return err
		}

		// Truncate only the last block; no implicit reselection is performed.
		count := blockLength
		if remaining := len(dst) - offset; remaining < count {
			count = remaining
		}
		for j := 0; j < count; j++ {
			// Long blocks remain cancellable even when the start bound is one.
			if j%cancellationStride == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			dst[offset+j] = start + j
		}
		offset += count
	}
	return ctx.Err()
}
