// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema provides exact affine exponential moving average smoothing.
// This file exposes the validated batch and stream construction facades.
package ema

import (
	"context"
	"math"
)

// New constructs an initialized empty Stream with explicit period and seed policy.
// It owns fixed-size state and retains no references to caller observations.
//
// Implementation:
//   - Stage 1: validate Period and Seed without allocation.
//   - Stage 2: store the exact unsigned period and copied options.
//   - Stage 3: return the initialized single-owner stream.
//
// Behavior highlights:
//   - New never accepts Options{} as an implicit default configuration.
//
// Inputs:
//   - opts: Period in [1, MaxInt-1], SeedSMA or SeedFirst.
//
// Returns:
//   - *Stream: ready for Push or Restore, with Count zero.
//
// Errors:
//   - ErrInvalidOption: invalid period or unknown initialization policy.
//
// Determinism:
//   - The same options create identical empty snapshots.
//
// Complexity:
//   - Time O(W), Space O(W), W=34 fixed words; one owning stream allocation.
//
// Notes:
//   - Independent instances can be used concurrently; one instance cannot.
//
// AI-Hints:
//   - Construct outside the hot loop and reuse the stream through Push.
//   - Restore requires a New-created receiver with the exact same options.
func New(opts Options) (*Stream, error) {
	// Freeze validated integer weights before constructing any mutable state.
	period, err := validateOptions(opts)
	if err != nil {
		// No stream or result is exposed for an invalid configuration.
		return nil, err
	}
	// Copy options by value; the accumulator starts empty and detached.
	return &Stream{options: opts, period: period, initialized: true}, nil
}

// Compute returns an owning, input-aligned history of exact affine EMA measurements.
// Its numerical transition is the same transition used by successive Stream.Push calls.
//
// Implementation:
//   - Stage 1: validate context, options, shape capacity, then all finite inputs.
//   - Stage 2: allocate the two result channels and advance one local Stream.
//   - Stage 3: check context immediately before publishing the complete Result.
//
// Behavior highlights:
//   - Nil and empty histories succeed with non-nil, empty output slices.
//   - Short SMA histories succeed with +0 values and false readiness masks.
//
// Inputs:
//   - ctx: non-nil context, checked during validation and calculation.
//   - x: finite ordered observations; read only and never retained.
//   - opts: explicit period and initialization policy.
//
// Returns:
//   - *Result: independent Values and Valid channels of length len(x).
//
// Errors:
//   - ErrNilContext or ctx.Err(): context validation or cancellation checks.
//   - ErrInvalidOption: invalid Period or Seed, before inspecting input values.
//   - ErrOverflow: combined output byte count exceeds the platform int range.
//   - ErrNonFinite: first nonfinite observation in ascending input order.
//   - Every error returns nil, never an externally visible partial result.
//
// Determinism:
//   - Uses ascending observation order and one nearest-even rounding per transition.
//
// Complexity:
//   - Time O(n*W), Space O(n+W), W=34 fixed limbs; two linear traversals.
//
// Notes:
//   - Caller must not mutate x during the call; returned channels are caller-owned.
//   - Integer-size checks do not promise recovery from runtime memory exhaustion.
//
// AI-Hints:
//   - Never prefill an SMA seed from x[0] or read future values to find an anchor.
//   - Use Stream to avoid allocating a complete output history.
func Compute(ctx context.Context, x []float64, opts Options) (*Result, error) {
	// Stage 1: context has precedence even for an empty or malformed input.
	if err := checkContext(ctx); err != nil {
		return nil, err
	}
	// Freeze validated integer weights before constructing any mutable state.
	period, err := validateOptions(opts)
	if err != nil {
		// No stream or result is exposed for an invalid configuration.
		return nil, err
	}
	// Bound the combined payload before make; this does not reserve physical RAM.
	if len(x) > math.MaxInt/resultBytesPerObservation {
		return nil, ErrOverflow
	}
	// Complete finite-input admission in original index order before arithmetic.
	if err := validateHistory(ctx, x); err != nil {
		return nil, err
	}
	// Stage 2: allocate owning channels; make preserves non-nil empty slices.
	result := &Result{Values: make([]float64, len(x)), Valid: make([]bool, len(x))}
	// The local stream uses exactly the same transition as Push, without copies.
	state := Stream{options: opts, period: period, initialized: true}
	// Preserve index alignment and check cancellation in each bounded block.
	for i, value := range x {
		if i%contextCheckInterval == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		// advance publishes +0/false until the exact SMA seed becomes ready.
		next := state.advance(value)
		// Store payload and mask together; a valid numerical zero stays valid.
		result.Values[i], result.Valid[i] = next.Value, next.Valid
	}
	// Stage 3: discard the entire private result if cancellation is observed.
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Publication transfers both output slices to the caller.
	return result, nil
}

// ComputeSMA selects an exact P-observation arithmetic-mean seed.
// It is a policy-only shortcut to Compute, not a second smoothing implementation.
//
// Implementation:
//   - Stage 1: form Options{Period: period, Seed: SeedSMA} by value.
//   - Stage 2: delegate all admission and state construction to Compute.
//   - Stage 3: return its result and error unchanged.
//
// Behavior highlights:
//   - Index P-1 is the first valid output; shorter histories remain +0/false.
//   - Period one selects the input directly, canonicalizing any zero to +0.
//
// Inputs:
//   - ctx: non-nil context; cancellation and precedence are handled by Compute.
//   - x: finite, ordered, read-only observations; nil and empty are accepted.
//   - period: explicit observation count in [1, MaxInt-1], never a calendar duration.
//
// Returns:
//   - *Result: detached input-length Values and Valid; non-nil empty channels on success.
//   - error: nil on success; every failure returns a nil result.
//
// Errors:
//   - ErrNilContext or ctx.Err(): same context checkpoints as Compute.
//   - ErrInvalidOption: period outside [1, MaxInt-1], before input validation.
//   - ErrNonFinite: a nonfinite observation, in ascending index order.
//   - ErrOverflow: combined output payload exceeds the addressable int range.
//
// Determinism:
//   - Identical bits, readiness, and sentinel precedence to Compute with SeedSMA.
//
// Complexity:
//   - Time O(n*W), Space O(n+W), W=34 fixed limbs; wrapper overhead O(1).
//
// Notes:
//   - No period default, fallback, or output projection is introduced; use Compute for explicit Options.
//   - The seed mode remains part of the snapshot configuration and cannot be changed by Restore.
//
// AI-Hints:
//   - Use ComputeFirst or NewFirst only when immediate first-observation seeding is intended.
//   - Keep validity masks when joining this series with other measurements.
func ComputeSMA(ctx context.Context, x []float64, period int) (*Result, error) {
	// Select only the named seed policy; preserve the complete canonical result.
	return Compute(ctx, x, Options{Period: period, Seed: SeedSMA})
}

// ComputeFirst selects the first observation as its seed.
// It is a policy-only shortcut to Compute, not a second smoothing implementation.
//
// Implementation:
//   - Stage 1: form Options{Period: period, Seed: SeedFirst} by value.
//   - Stage 2: delegate all admission and state construction to Compute.
//   - Stage 3: return its result and error unchanged.
//
// Behavior highlights:
//   - Index zero is already valid; no artificial P-observation delay is introduced.
//   - Period one selects the input directly, canonicalizing any zero to +0.
//
// Inputs:
//   - ctx: non-nil context; cancellation and precedence are handled by Compute.
//   - x: finite, ordered, read-only observations; nil and empty are accepted.
//   - period: explicit observation count in [1, MaxInt-1], never a calendar duration.
//
// Returns:
//   - *Result: detached input-length Values and Valid; non-nil empty channels on success.
//   - error: nil on success; every failure returns a nil result.
//
// Errors:
//   - ErrNilContext or ctx.Err(): same context checkpoints as Compute.
//   - ErrInvalidOption: period outside [1, MaxInt-1], before input validation.
//   - ErrNonFinite: a nonfinite observation, in ascending index order.
//   - ErrOverflow: combined output payload exceeds the addressable int range.
//
// Determinism:
//   - Identical bits, readiness, and sentinel precedence to Compute with SeedFirst.
//
// Complexity:
//   - Time O(n*W), Space O(n+W), W=34 fixed limbs; wrapper overhead O(1).
//
// Notes:
//   - No period default, fallback, or output projection is introduced; use Compute for explicit Options.
//   - The seed mode remains part of the snapshot configuration and cannot be changed by Restore.
//
// AI-Hints:
//   - Do not use first seeding merely to conceal an unfinished SMA warmup.
//   - Keep validity masks when joining this series with other measurements.
func ComputeFirst(ctx context.Context, x []float64, period int) (*Result, error) {
	// Select only the named seed policy; preserve the complete canonical result.
	return Compute(ctx, x, Options{Period: period, Seed: SeedFirst})
}

// NewSMA selects an exact P-observation arithmetic-mean seed.
// It is a policy-only shortcut to New, not a second smoothing implementation.
//
// Implementation:
//   - Stage 1: form Options{Period: period, Seed: SeedSMA} by value.
//   - Stage 2: delegate all admission and state construction to New.
//   - Stage 3: return its result and error unchanged.
//
// Behavior highlights:
//   - Index P-1 is the first valid output; shorter histories remain +0/false.
//   - Period one selects the input directly, canonicalizing any zero to +0.
//
// Inputs:
//   - period: explicit observation count in [1, MaxInt-1], never a calendar duration.
//
// Returns:
//   - *Stream: initialized, empty, single-owner stream; nil on failure.
//
// Errors:
//   - ErrInvalidOption: period outside [1, MaxInt-1]; no stream is returned.
//
// Determinism:
//   - Identical bits, readiness, and sentinel precedence to New with SeedSMA.
//
// Complexity:
//   - Time O(W), Space O(W), W=34; wrapper overhead O(1), no additional workspace.
//
// Notes:
//   - No period default, fallback, or output projection is introduced; use New for explicit Options.
//   - The seed mode remains part of the snapshot configuration and cannot be changed by Restore.
//
// AI-Hints:
//   - Use ComputeFirst or NewFirst only when immediate first-observation seeding is intended.
//   - Keep validity masks when joining this series with other measurements.
func NewSMA(period int) (*Stream, error) {
	// Delegate validation before allocating or publishing stream state.
	return New(Options{Period: period, Seed: SeedSMA})
}

// NewFirst selects the first observation as its seed.
// It is a policy-only shortcut to New, not a second smoothing implementation.
//
// Implementation:
//   - Stage 1: form Options{Period: period, Seed: SeedFirst} by value.
//   - Stage 2: delegate all admission and state construction to New.
//   - Stage 3: return its result and error unchanged.
//
// Behavior highlights:
//   - Index zero is already valid; no artificial P-observation delay is introduced.
//   - Period one selects the input directly, canonicalizing any zero to +0.
//
// Inputs:
//   - period: explicit observation count in [1, MaxInt-1], never a calendar duration.
//
// Returns:
//   - *Stream: initialized, empty, single-owner stream; nil on failure.
//
// Errors:
//   - ErrInvalidOption: period outside [1, MaxInt-1]; no stream is returned.
//
// Determinism:
//   - Identical bits, readiness, and sentinel precedence to New with SeedFirst.
//
// Complexity:
//   - Time O(W), Space O(W), W=34; wrapper overhead O(1), no additional workspace.
//
// Notes:
//   - No period default, fallback, or output projection is introduced; use New for explicit Options.
//   - The seed mode remains part of the snapshot configuration and cannot be changed by Restore.
//
// AI-Hints:
//   - Do not use first seeding merely to conceal an unfinished SMA warmup.
//   - Keep validity masks when joining this series with other measurements.
func NewFirst(period int) (*Stream, error) {
	// Delegate validation before allocating or publishing stream state.
	return New(Options{Period: period, Seed: SeedFirst})
}
