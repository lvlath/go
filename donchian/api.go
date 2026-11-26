// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian exposes one canonical batch facade and explicit window-mode wrappers.
// This file owns validation sequencing, checked allocation, and all-or-nothing publication.
package donchian

import (
	"context"
	"math"
)

// Compute constructs aligned Donchian boundaries from ordered high and low prices.
// It computes exact observed extrema and a nearest-even binary64 boundary mean.
//
// Implementation:
//   - Stage 1: validate context, options, shape, all finite values, then geometry.
//   - Stage 2: allocate owning channels and run one monotonic-deque kernel.
//   - Stage 3: check cancellation again and publish the complete result atomically.
//
// Behavior highlights:
//   - Include-current uses [i-P+1,i], first ready at P-1.
//   - Exclude-current uses [i-P,i-1], first ready at P.
//   - All-empty inputs succeed; a short history succeeds with a false mask.
//
// Inputs:
//   - ctx: required context; cancellation is polled at least every 1024 bars,
//     every 1024 removals in a long deque burst, and before publication.
//   - high, low: equal-length, read-only finite prices with high[i] >= low[i].
//   - opts: explicit positive period and include/exclude-current policy.
//
// Returns:
//   - *Result: non-nil owning channels on success; nil on every error.
//   - error: stable errors.Is category or the original context error.
//
// Errors:
//   - ErrNilContext or ctx.Err() precedes all input validation.
//   - ErrInvalidOption precedes ErrShape; nil/nonempty also matches ErrNilInput.
//   - ErrNonFinite precedes ErrBar globally, even in entirely masked histories.
//   - ErrOverflow reports an unrepresentable packed float-output byte size.
//
// Determinism:
//   - Increasing index traversal; latest equal extrema; nearest-even midpoint.
//   - Concurrent cancellation and an input failure have no prescribed race order.
//
// Complexity:
//   - Time O(n), including validation and output initialization.
//   - Space O(n) outputs plus O(min(P,n)) indices; wholly masked inputs use no rings.
//
// Notes:
//   - No input mutation or input/output aliasing. Inputs must remain unchanged
//     throughout a call. Independent calls may share read-only inputs.
//   - Byte-count checks do not promise available RAM or recover runtime OOM.
//   - The final successful context check is the publication boundary; cancellation
//     immediately afterward cannot retroactively withdraw a returned result.
//   - Typed-nil or misbehaving implementations of context.Context are unsupported;
//     nil-interface detection uses no reflection and calls no recovery machinery.
//
// AI-Hints:
//   - Validate bar finality, time ordering, missing observations and execution
//     assumptions outside this package; ExcludeCurrent does not provide a fill.
//   - A finite midpoint cannot overflow; do not add false numeric failure paths.
func Compute(ctx context.Context, high, low []float64, opts Options) (*Result, error) {
	// Validate cheap call-level requirements before reading any input price.
	if err := validateHeader(ctx, high, low, opts); err != nil {
		// Preserve the context/options/shape error and publish no artifact.
		return nil, err
	}
	// Validate every supplied row, even if the requested window never becomes ready.
	if err := validateHistory(ctx, high, low); err != nil {
		// A masked or last excluded row is still part of the admitted input domain.
		return nil, err
	}
	// Avoid allocating outputs after cancellation observed during validation.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Check the mandatory packed float buffer before multiplying its size.
	// This is allocation arithmetic, not a restriction on the window period.
	n := len(high)
	const (
		// boundaryChannels counts Upper, Middle, and Lower; Valid is separate.
		boundaryChannels = 3
		// binary64Bytes is the storage width of a Go float64 on every target.
		binary64Bytes = 8
		// floatBytesPerRow bounds both element-count and byte-count multiplication.
		floatBytesPerRow = boundaryChannels * binary64Bytes
	)
	if n > math.MaxInt/floatBytesPerRow {
		// This is an allocation-size failure, not a midpoint or price-domain failure.
		return nil, ErrOverflow
	}
	// One zeroed allocation supplies three disjoint owning boundary channels.
	values := make([]float64, boundaryChannels*n)
	// Full-slice expressions cap each channel so append cannot overwrite its neighbor.
	// Keeping one channel alive retains this entire backing array; see Result.
	result := &Result{
		Upper:  values[:n:n],
		Middle: values[n : 2*n : 2*n],
		Lower:  values[2*n : 3*n : 3*n],
		// The separate zeroed mask distinguishes warmup +0 from a measured zero.
		Valid: make([]bool, n),
	}
	// Only this private kernel writes ready rows; unready rows keep allocation zeros.
	if err := compute(ctx, high, low, opts, result); err != nil {
		// Discard all private rows and deque state after an interrupted computation.
		return nil, err
	}
	// Cancellation observed at publication also discards an otherwise complete result.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Transfer ownership only after every fallible stage has succeeded.
	return result, nil
}

// ComputeIncludeCurrent constructs a channel containing each row's current bar.
// Use it for a finalized-bar range overlay, not as a previous-bar breakout threshold.
//
// Implementation:
//   - Stage 1: assemble Options with the supplied period and ExcludeCurrent=false.
//   - Stage 2: delegate validation, allocation, and the sole numerical kernel to Compute.
//   - Stage 3: return its complete result or unchanged error; discard no result channels.
//
// Behavior highlights:
//   - The window is [i-period+1,i]; the first ready row is period-1.
//   - Period has no default; zero and negative values remain invalid.
//
// Inputs:
//   - ctx: required non-nil context; cancellation follows Compute exactly.
//   - high, low: equal-length read-only finite prices, with high[i] >= low[i].
//   - period: positive number of observations, not a number of calendar days.
//
// Returns:
//   - *Result: all four detached channels of input length, or nil on failure.
//   - error: nil on success; otherwise the unchanged error produced by Compute.
//
// Errors:
//   - ErrNilContext or ctx.Err(): before option or input validation.
//   - ErrInvalidOption: period < 1; ErrShape: unequal lengths.
//   - ErrNilInput: additionally matched for a nil/nonempty shape mismatch.
//   - ErrNonFinite precedes ErrBar globally; ErrOverflow guards output allocation size.
//
// Determinism:
//   - Exactly the same bits, mask, tie policy, and error precedence as matching Compute.
//   - The wrapper performs no extra context polls or input traversal.
//
// Complexity:
//   - Wrapper overhead is O(1) time/space, without an additional heap buffer.
//   - Including Compute: O(n) time, O(n) output space and O(min(period,n)) workspace.
//
// Notes:
//   - Empty input succeeds with non-nil empty channels; no partial results are returned.
//   - n < period is entirely masked and allocates no deque rings.
//   - Inputs must remain unchanged during the call; output ownership matches Result.
//
// AI-Hints:
//   - For prior-range comparisons, call ComputeExcludeCurrent instead.
//   - Use Compute when a mode is already stored in an Options value; do not reimplement it.
func ComputeIncludeCurrent(ctx context.Context, high, low []float64, period int) (*Result, error) {
	// Freeze the named window policy without changing canonical validation or arithmetic.
	return Compute(ctx, high, low, Options{Period: period, ExcludeCurrent: false})
}

// ComputeExcludeCurrent constructs a previous-bar reference channel without the current bar.
// Use it to compare a finalized observation against an already established range.
//
// Implementation:
//   - Stage 1: assemble Options with the supplied period and ExcludeCurrent=true.
//   - Stage 2: delegate validation, allocation, and the sole numerical kernel to Compute.
//   - Stage 3: return its complete result or unchanged error; discard no result channels.
//
// Behavior highlights:
//   - The window is [i-period,i-1]; the first ready row is period.
//   - Period has no default; zero and negative values remain invalid.
//
// Inputs:
//   - ctx: required non-nil context; cancellation follows Compute exactly.
//   - high, low: equal-length read-only finite prices, with high[i] >= low[i].
//   - period: positive number of observations, not a number of calendar days.
//
// Returns:
//   - *Result: all four detached channels of input length, or nil on failure.
//   - error: nil on success; otherwise the unchanged error produced by Compute.
//
// Errors:
//   - ErrNilContext or ctx.Err(): before option or input validation.
//   - ErrInvalidOption: period < 1; ErrShape: unequal lengths.
//   - ErrNilInput: additionally matched for a nil/nonempty shape mismatch.
//   - ErrNonFinite precedes ErrBar globally; ErrOverflow guards output allocation size.
//
// Determinism:
//   - Exactly the same bits, mask, tie policy, and error precedence as matching Compute.
//   - The wrapper performs no extra context polls or input traversal.
//
// Complexity:
//   - Wrapper overhead is O(1) time/space, without an additional heap buffer.
//   - Including Compute: O(n) time, O(n) output space and O(min(period,n)) workspace.
//
// Notes:
//   - Empty input succeeds with non-nil empty channels; no partial results are returned.
//   - n <= period is entirely masked and allocates no deque rings.
//   - Inputs must remain unchanged during the call; output ownership matches Result.
//
// AI-Hints:
//   - A signal at close does not imply an executable fill at that same close.
//   - Use Compute when a mode is already stored in an Options value; do not reimplement it.
func ComputeExcludeCurrent(ctx context.Context, high, low []float64, period int) (*Result, error) {
	// Freeze the named window policy without changing canonical validation or arithmetic.
	return Compute(ctx, high, low, Options{Period: period, ExcludeCurrent: true})
}
