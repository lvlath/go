// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package drawdown defines the validated public computation and scalar projection facades.
// This file delegates every calculation to one canonical kernel.
package drawdown

import "context"

// Compute measures the sampled drawdown of chronological equity observations.
// It returns a correctly rounded depth profile and an exactly selected maximum episode.
//
// Implementation:
//   - Stage 1: validate context, nonempty shape and initial equity.
//   - Stage 2: allocate detached output, then validate and scan each later sample
//     in order, tracking the earliest running peak and exact maximum witness.
//   - Stage 3: find the first recovery after the selected trough and check
//     context immediately before publishing the complete result.
//
// Behavior highlights:
//   - Depth is RN64((runningPeak-equity)/runningPeak), nearest-even, in [0,1].
//   - The exact maximum ratio wins even when rounded depths collide.
//   - Later equity zero is valid total loss; positive equity may round to depth 1.
//   - No drawdown gives all witness indices zero. No recovery gives -1.
//
// Inputs:
//   - ctx: non-nil context; cancellation checked at least every 1024 observations.
//   - equity: nonempty chronological finite values in one consistent unit;
//     first value >0, later values >=0. Read-only for the duration of the call.
//
// Returns:
//   - *Result: complete, detached, caller-owned profile and witness on success.
//   - On any error, nil Result; inputs unchanged, no partial publication.
//
// Errors:
//   - ErrNilContext: nil context, before input access.
//   - ctx.Err(): cancellation or deadline at a poll or publication boundary.
//   - ErrInsufficient: nil or empty equity, after context validation.
//   - ErrNonFinite: NaN/Inf at the first invalid observation.
//   - ErrEquity: finite nonpositive first or negative later observation.
//     Values are checked by index, with finiteness before sign at each index.
//     Concurrent cancellation and input errors follow first detection.
//
// Determinism:
//   - Ascending index order; earliest equal running peak and exact equal trough.
//   - Maximum is bit-identical to Depth[TroughIndex].
//
// Complexity:
//   - Time O(n), one profile pass and at most one recovery suffix pass.
//   - Space O(n) output, O(1) auxiliary; fixed-width arithmetic, no per-sample heap allocation on success.
//
// Notes:
//   - Independent calls may run concurrently on read-only inputs. No shared state.
//   - No cashflow adjustment, intrabar estimation, sorting, interpolation or calendar.
//   - A successful final context check is the publication boundary.
//
// AI-Hints:
//   - Do not use 1-equity/peak or compare rounded depths for witness selection.
//   - Use ComputeMaximum or ComputeCurrentDepth only when the discarded
//     evidence is unnecessary; neither wrapper reduces the scan or storage cost.
//   - Do not use the retrospective maximum as a feature at an earlier timestamp.
func Compute(ctx context.Context, equity []float64) (*Result, error) {
	// Stage 1: reject invalid context, shape, or initial equity before allocation.
	if err := validateInput(ctx, equity); err != nil {
		return nil, err
	}
	// Stages 2-3: keep the complete profile private until both scans succeed.
	return compute(ctx, equity)
}

// ComputeMaximum returns only the maximum sampled drawdown of an equity history.
// Use it for report columns or path-level risk summaries that do not need the
// profile or witness; it projects Maximum from Compute without changing arithmetic.
//
// Implementation:
//   - Stage 1: delegate context and equity validation to Compute.
//   - Stage 2: let Compute calculate the exact witness and full depth profile.
//   - Stage 3: propagate any error unchanged, otherwise return Maximum.
//
// Behavior highlights:
//   - The returned scalar is in [0,1], rounded once to nearest-even binary64.
//   - A valid zero denotes no drawdown; a rounded one need not imply zero equity.
//
// Inputs:
//   - ctx: non-nil active context, with the cancellation semantics of Compute.
//   - equity: chronological finite history, first >0, later >=0, read-only.
//
// Returns:
//   - float64: exact-witness rounded maximum, or +0 on error (not a measurement).
//   - error: unchanged error from Compute; nil on success.
//
// Errors:
//   - Same sentinels, ordering and context errors as Compute.
//
// Determinism:
//   - Bit-identical to Compute(ctx, equity).Maximum on successful calls.
//
// Complexity:
//   - Time O(n), space O(n) transient output and O(1) auxiliary workspace;
//     this wrapper still allocates the entire canonical Result and discards it.
//
// Notes:
//   - No retained input or output; the scalar is retrospective over the full history.
//   - Concurrent input mutation is unsupported, just as for Compute.
//
// AI-Hints:
//   - Do not treat a returned zero as risk-free when err != nil.
//   - Use Compute when exact peak/trough/recovery evidence is needed. Do not
//     substitute a rounded argmax or advertise a constant-memory fast path.
func ComputeMaximum(ctx context.Context, equity []float64) (float64, error) {
	// Delegate the entire mathematical contract and cancellation boundary.
	result, err := Compute(ctx, equity)
	if err != nil {
		return 0, err // Never expose a scalar from a failed calculation.
	}
	// Project only the historical summary, not the last observation's depth.
	return result.Maximum, nil
}

// ComputeCurrentDepth returns the sampled drawdown at the final supplied observation.
// It is useful for an as-of risk dashboard consuming an admitted history prefix;
// the reference is the running peak of that prefix, not the initial balance.
//
// Implementation:
//   - Stage 1: delegate validation and full batch computation to Compute.
//   - Stage 2: propagate failure without indexing an absent result.
//   - Stage 3: read the final element of the successful nonempty Depth profile.
//
// Behavior highlights:
//   - Returns a current sampled depth, not the historical maximum.
//   - A recovered account may have current depth +0 and a positive historical MDD.
//
// Inputs:
//   - ctx: non-nil active context, with the cancellation semantics of Compute.
//   - equity: chronological admitted prefix, first >0, later >=0, all finite.
//
// Returns:
//   - float64: last rounded depth in [0,1], or +0 on error (not a measurement).
//   - error: unchanged error from Compute; nil on success.
//
// Errors:
//   - Same sentinels, ordering and context errors as Compute.
//
// Determinism:
//   - Bit-identical to the last Depth element from Compute for the same prefix.
//
// Complexity:
//   - Time O(n), space O(n) transient output and O(1) auxiliary workspace.
//     Recomputing every growing prefix is quadratic in total history length.
//
// Notes:
//   - No input retention, incremental state, threshold policy or account resets.
//   - The caller owns source availability, cashflow adjustment and risk limits.
//
// AI-Hints:
//   - Check err before threshold comparisons; rounded depths are not exact
//     rational threshold certificates. This is not a streaming update API.
func ComputeCurrentDepth(ctx context.Context, equity []float64) (float64, error) {
	// Compute rejects nil/empty histories and preserves validation precedence.
	result, err := Compute(ctx, equity)
	if err != nil {
		return 0, err // A failure has no valid current depth.
	}
	// Successful profiles are nonempty, so the final index is safe.
	return result.Depth[len(result.Depth)-1], nil
}
