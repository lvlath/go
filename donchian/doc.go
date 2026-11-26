// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian computes causal rolling high/low ranges with an explicit
// current-bar policy and a correctly rounded binary64 midpoint.
// This file defines the batch contract, ownership boundaries, and operational limits.
//
// # What and why
//
// Compute returns aligned Upper, Middle, Lower, and Valid channels. It supplies
// observed range boundaries for overlays and caller-defined comparisons, not a
// trading system, prediction interval, or profitability claim. ComputeIncludeCurrent
// and ComputeExcludeCurrent are named policy wrappers around Compute; neither
// changes validation, numerical evaluation, context polling, or returned channels.
// No other lvlath package is imported. Time-grid preparation belongs to the caller,
// optionally using series; ATR-based normalization is a separate application step.
//
// # Implementation
//
//   - Stage 1: validate context, period, shape, all finite values, then candle geometry.
//   - Stage 2: check packed-output size, allocate zeroed owning channels, and scan
//     with two fixed-capacity monotonic index deques when any row can become ready.
//   - Stage 3: check cancellation before returning the complete result; every error
//     discards private work and returns a nil *Result.
//
// # Behavior highlights
//
// For P=Options.Period, include-current uses [i-P+1,i], first ready at P-1;
// exclude-current uses [i-P,i-1], first ready at P. Upper selects the maximum high
// and Lower selects the minimum low. New equal values replace older candidates,
// so the latest index wins, including ties between positive and negative zero.
// At P=1, include selects current H/L and exclude selects previous H/L.
//
// Middle is RN64((Upper+Lower)/2): nearest-even rounding of the exact mean of the
// selected binary64 boundaries. Magnitude-dependent evaluation prevents false
// sum/difference overflow and destructive intermediate subnormal underflow.
// Exact-zero means are +0; a negative nonzero mean may round to -0. Selected H/L
// retain their original bits. All three unready payloads are strictly +0.
//
// Include-current is causal after the current bar is available, but is not a
// previous-bar threshold: a consistent close cannot exceed its own inclusive
// upper boundary. Exclude-current reads the prior window before expiring its
// oldest bar or inserting the current bar. Neither policy authenticates historical
// availability or permits an assumed fill at the signal close.
//
// # Inputs and options
//
// ctx must be a non-nil usable context.Context. Period is an observation count in
// [1, MaxInt], with no default; Options{} is invalid. high and low are equal-length,
// read-only, already ordered slices in consistent units, with high[i] >= low[i].
// Finite negative values, both zero signs, and subnormals are admitted. NaN and
// either infinity are rejected. The package does not accept or validate closes,
// timestamps, corporate-action adjustments, or missing-bar metadata.
//
// # Returns and nilability
//
// Success returns a non-nil *Result with four input-length channels. All-empty
// inputs, including nil/empty combinations, yield four non-nil empty slices.
// n<P in include mode, or n<=P in exclude mode, yields an entirely false mask
// without deque allocation. Ready zero-valued rows remain valid.
// Result.IsNil is exactly a pointer-presence test, including through interfaces;
// it does not inspect readiness, shape, or numeric values.
//
// # Errors
//
// ErrNilContext or an already observed ctx.Err() precedes ErrInvalidOption, then
// ErrShape. A nil slice paired with a nonempty slice matches both ErrShape and
// ErrNilInput via errors.Is. After shape admission, high is checked before low
// at each increasing index. ErrNonFinite has global priority over a remembered
// finite ErrBar; only after the entire finite scan is the first bad bar reported.
// Every supplied row is checked, including warmup rows and the last excluded bar.
// An invalid later row rejects the whole batch, not merely that row or a suffix.
//
// ErrOverflow guards an unrepresentable packed float-output byte count; the mean
// of two finite bounds cannot itself overflow. No partial result accompanies any
// error. Error text is diagnostic; classify categories with errors.Is. Custom
// typed-nil or misbehaving context implementations are unsupported and not recovered.
//
// # Determinism
//
// The kernel visits indices ascending, updates the upper deque before the lower
// deque, and retains latest equal extrema. Admissible prefixes are numerically
// invariant under appending admissible future bars. On the same accepted history,
// exclude row i equals include row i-1 bit for bit when those rows are ready.
// Concurrent cancellation and an input error after entry have no prescribed winner.
//
// # Complexity and resources
//
// Validation, initialization, and the kernel take O(n) total time. Each index
// enters each deque once and leaves it at most once. A single insertion can remove
// O(P) candidates; O(1) per bar is amortized, not a worst-case latency bound.
// Outputs use O(n) storage; deque workspace is O(min(P,n)), with exactly P integer
// slots per deque only when a ready row is possible. No P+1 allocation is needed.
//
// Three float64 channels share a packed 3*n array. Its 24*n byte arithmetic is
// checked before multiplication; Valid has a separate n-element bool allocation.
// Allocation checks do not reserve RAM, interrupt runtime allocation, or make
// runtime out-of-memory termination recoverable. Huge P with small n never causes
// a period-sized allocation. This owning batch API is not allocation-free.
//
// # Ownership, cancellation, and concurrency
//
// Inputs are never mutated or retained, and outputs do not alias them. The three
// numerical channels are non-overlapping views of one owning backing array, each
// capped at len==n. Append cannot overwrite a neighboring channel, but keeping
// one numerical channel alive retains the shared allocation. Copy it when an
// independent lifetime matters. Copying Result by value does not deep-copy slices.
//
// Independent calls may share stable read-only input, including overlapping input
// slices. Concurrent input mutation or writes to a shared result require caller
// synchronization. There is no persistent processing state or internal lock.
// Context is polled every 1024 input indices in each scan, additionally every
// 1024 removals within a long single-deque insertion, after validation, and before
// publication. These are phase-local iteration bounds, not a universal bound of
// 1024 operations or a wall-clock guarantee. The final successful check cannot
// retroactively withdraw a result if cancellation occurs immediately afterward.
//
// # Notes and AI-Hints
//
// There is no Stream, Snapshot, default period, width output, timestamp handling,
// order execution, or private risk dependency. Source identity, revision, units,
// cadence, and Options must accompany saved artifacts outside Result; the result
// does not carry policy metadata or prove provenance.
// Never interpret a masked zero as a measurement or use IsNil as a readiness test.
// Validate a close separately before comparing it with a prior boundary. Upper-Lower
// can overflow despite a finite Middle; widths, ratios, and ATR denominators need
// explicit consumer policy. For a pre-event scale, align a ready ATR[i-1], not an
// unexamined ATR[i]. Recompute affected windows after historical corrections.
// Production uses no reflection, unsafe operations, arbitrary-precision arithmetic,
// embedded fixtures, panic/recovery calls, or test-only public exports.
package donchian
