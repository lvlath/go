// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap implements fixed-length circular block resampling.
// Source positions wrap to the first row only within the selected block.
package bootstrap

import "context"

// fillCircular populates one unpublished row with cyclic fixed-length blocks.
// It uses checked per-position wrap logic rather than overflow-prone start+j.
//
// Implementation:
//   - Stage 1: Check context and choose a uniform starting index in [0, rows).
//   - Stage 2: Emit min(B, remaining) sequential positions with wrap at rows-1.
//   - Stage 3: Repeat at a new random start and verify context at completion.
//
// Behavior highlights:
//   - A block beginning at rows-1 continues at index 0, then 1.
//   - The final block is truncated exactly at the requested row length.
//   - The wrap check prevents addition overflow for input rows close to MaxInt.
//
// Inputs:
//   - ctx: non-nil context; checked at least once per block and every 1024 indices.
//   - stream: exclusive deterministic 64-bit source for this replicate.
//   - dst: caller-allocated unpublished index row.
//   - rows: source observation count >=1; blockLength: B in [1, rows].
//
// Returns:
//   - Fully populated dst with nil error, or invalid local dst and a context error.
//
// Errors:
//   - ctx.Err() if computation is interrupted.
//
// Determinism:
//   - Blocks are visited in destination order, starts drawn before writes.
//
// Complexity:
//   - Expected Time O(len(dst)), Space O(1) beyond dst.
//
// Notes:
//   - No cross-replicate RNG state is shared by this kernel.
//
// AI-Hints:
//   - Do not calculate (start+j)%rows in int: start+j can overflow.
//   - Circular continuation is not an independent draw at each position.
func fillCircular(ctx context.Context, stream wordStream, dst []int, rows, blockLength int) error {
	for offset := 0; offset < len(dst); {
		if err := ctx.Err(); err != nil {
			return err
		}

		// A circular block has rows equally likely starts, unlike MovingBlocks.
		index, err := uniformInteger(ctx, stream, rows)
		if err != nil {
			return err
		}

		count := blockLength
		if remaining := len(dst) - offset; remaining < count {
			count = remaining
		}
		for j := 0; j < count; j++ {
			if j%cancellationStride == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			dst[offset+j] = index

			// No addition is needed at rows-1; this is safe on 32- and 64-bit int.
			if index == rows-1 {
				index = 0
			} else {
				index++
			}
		}
		offset += count
	}
	return ctx.Err()
}
