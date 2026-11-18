// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ad computes the Chaikin Accumulation/Distribution (A/D) line from
// validated OHLCV candles. It accumulates each candle's close-location-weighted
// base volume, not the Advance/Decline breadth index or automatic differentiation.
//
// # Domain, shape, and origin
//
// A Bar holds one Open, High, Low, Close, Volume observation. All prices must
// be finite and strictly positive; Volume must be finite and nonnegative.
// Low <= Open,Close <= High is mandatory, including for Open, which does not
// enter the multiplier. A flat bar (High == Low) must therefore have all four
// prices equal. Bars supplies five non-nil, nonempty, equally long columns;
// ComputeCandles accepts a non-nil, nonempty []Bar in its existing order.
//
// The A/D origin is zero BEFORE the first observation. For a nonflat candle,
// define left = Close-Low, right = High-Close, and spread = High-Low. Its
// multiplier is (left-right)/spread and its contribution is multiplier*Volume.
// A flat bar contributes exactly zero, even if Volume is nonzero. Every
// accepted bar, INCLUDING the first, updates the accumulator and has Valid=true.
// This differs from obv, whose first candle only establishes a previous close;
// ad has no previous-close state and no warmup window.
//
// # Numerical contract
//
// The distance-based formula never evaluates 2*Close. Given positive finite
// validated OHLC values, left, right, and spread are finite, nonnegative, and
// left,right <= spread; the computed multiplier stays within [-1,1] without
// clipping or an arbitrary epsilon. A candle's contribution is bounded in
// magnitude by its Volume, but its cumulative A/D balance is not so bounded.
// No output is NaN or infinite on success.
//
// The recurrence stores a Neumaier high component (sum) and low-order
// correction (compensation); the displayed value is sum+compensation. On an
// overflowing direct intermediate operation, a deterministic cancellation-
// first rebase attempts to form a finite two-component candidate. If neither
// documented route is representable, the operation returns ErrOverflow. This
// improves cancellation such as +1e16,+1,-1e16, but does not promise exact
// rational accumulation, a universal error bound, or unlimited numeric range.
//
// # API and publication
//
// Compute(Bars) is the canonical column-oriented batch facade. ComputeCandles
// processes chronological []Bar rows without allocating five transpose arrays.
// Both use the same private nextAD transition as Stream.Push. Latest and
// LatestCandles are projections of those COMPLETE batch results; they still
// require O(n) time and result allocation, rather than a last-window shortcut.
// For ongoing O(1) work, New() and a zero-valued Stream{} are equivalent.
// Stream.Push validates one bar, builds a complete local candidate, and commits
// it only after all checks pass. A successful Push returns (value,true,nil);
// a failure returns (0,false,error) without modifying the receiver.
//
// A successful *Result owns detached Values and Valid slices, each of length
// n>0, with all Valid entries true. Result.IsNil implements core.Nilable by
// checking the receiver pointer, not its fields or contents. On ANY batch
// error, the result pointer is nil: no usable prefix is published. Returned
// slices may be changed by their caller without affecting package state.
//
// # Error law and precedence
//
// Sentinels support errors.Is and never include caller-supplied values:
//   - ErrNilInput: a nil batch column, nil candle slice, or nil Push/Restore receiver.
//   - ErrEmptyInput: a present, non-nil empty observation collection.
//   - ErrShape: lengths of the five non-nil Bars columns disagree.
//   - ErrNonFinite: input OHLCV contains NaN or either infinity.
//   - ErrDomain: nonpositive price, negative volume, or invalid candle geometry.
//   - ErrOverflow: result allocation byte arithmetic, Count, or accumulator fails.
//   - ErrStateVersion: incompatible or invalid Snapshot metadata or numeric state.
//
// The COMPLETE batch preflight follows Nil -> Empty -> Shape (for Bars), then
// scans all rows for NonFinite, then scans all rows for Domain, and only then
// computes. Thus a late NaN outranks an earlier geometric error in a batch.
// Stream.Push checks nil receiver, then the current bar's NonFinite and Domain,
// then arithmetic. Restore checks nil receiver before snapshot validity.
// There are no Options, context values, periods, hidden fallbacks, or separate
// ErrInvalidOption policy in this package.
//
// # Determinism and state transfer
//
// Accepted observations are processed strictly in caller-supplied index order.
// Identical input bits, package version, and floating-point execution
// conditions produce bit-identical results from Compute, ComputeCandles and
// consecutive Push calls. No sorting, price epsilon, clipping, or implicit
// timestamp handling can change that order.
//
// Stream.Snapshot returns a detached v1 checkpoint: Algorithm is
// "ad/chaikin-neumaier/v1", Config is lowercase SHA-256 of those exact ASCII
// bytes, Count is the accepted-bar count, FloatState is [sum,compensation] of
// length 2, and IntState has length 0. No previous Close is stored. Restore
// checks identity, lengths, finite coordinates, finite effective total, and
// Count=0 zero-state invariants before atomically replacing the receiver.
// Count=0 signed zeros are canonicalized to Stream{}; Count=MaxUint64 is
// restorable but the next Push returns ErrOverflow. Snapshot(nil) returns an
// empty Snapshot{}, Reset(nil) is a no-op, and Push/Restore on nil return
// ErrNilInput. Reset establishes a fresh zero origin, NOT continuity with the
// prior cumulative window.
//
// Config identifies schema compatibility ONLY. It does not authenticate the
// snapshot payload, source feed, source offset, instrument, or volume units.
// Services must authenticate the entire checkpoint and enforce exactly-once
// replay/offset coordination outside this package.
//
// # Ownership, concurrency, and resource cost
//
// Input slices are borrowed read-only and never retained; concurrent caller
// mutation during a batch call is unsupported. A Stream has no locks and must
// be owned by one goroutine or externally synchronized for all Push, Reset,
// Snapshot, and Restore operations. Independent streams and immutable inputs
// can be processed in parallel. No wall clock, network, RNG, or global mutable
// state participates in the recurrence.
//
// Compute and ComputeCandles require O(n) time and O(n) detached output space;
// batch preflight itself uses O(1) extra space. Latest wrappers inherit those
// same costs. Push takes O(1) time, O(1) state and zero steady-state heap
// allocations. Reset, Snapshot, and Restore are O(1); Snapshot allocates a
// fixed-size detached FloatState. Allocation byte overflow is checked before
// output allocation, but runtime memory exhaustion is not converted to an error.
//
// # Explicit non-goals and integration
//
// ad does not validate timestamps, align calendars, infer a grid, fill missing
// bars, normalize volume currencies, issue trading decisions, detect price/A/D
// divergence, attribute buyer identity, or calculate an MFI-style oscillator.
// A caller may use series to validate timestamps and must Reset explicitly at
// a known discontinuity, retaining its external epoch and feed-offset provenance.
// Mixing A/D levels across different origins or volume units is not meaningful
// without a separate application-level comparison rule.
//
// AI-Hints:
//   - Never initialize the first output to OBV's seed zero: ad includes bar zero.
//   - Never replace the two bounded distances with 2*Close-High-Low.
//   - Never serialize only the displayed total: future cancellation needs both coordinates.
//   - Do not interpret Valid=true as evidence of a trading signal or market quality.
//   - Do not use Config as a MAC, and do not resume a checkpoint at a replayed source offset.
package ad
