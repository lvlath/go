// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package donchian provides causal Donchian Channel computations over finite,
// ordered high/low observations using two bounded monotonic index deques.
//
// # What and why
//
// Compute returns aligned Upper, Middle, Lower and Valid channels. It provides
// a range baseline for caller-defined features and historical comparisons,
// not trading decisions or a profitability claim. No other lvlath package is
// imported. Time-grid preparation belongs to the caller, optionally using series.
//
// # Windows and mathematical laws
//
// With P=Options.Period, include-current uses [i-P+1,i], first valid at P-1.
// ExcludeCurrent uses [i-P,i-1], first valid at P. Upper is the maximum high;
// Lower is the minimum low. Equal extrema retain their latest observation.
// This also defines deterministic signed-zero selection at the boundaries.
// Middle is RN64((Upper+Lower)/2), with no overflow or intermediate underflow
// loss. Exact-zero means are +0; negative nonzero underflow may round to -0.
//
// Include-current is causal after bar finality but includes the bar being
// compared. Exclude-current removes that same-bar threshold contamination.
// Neither mode certifies historical availability or same-close execution.
//
// # Options, domain, errors and nilability
//
// Period must be positive; there is no implicit default. Huge positive periods
// are valid. Negative prices and signed zeros are valid; NaN/Inf are not.
// Every high must be at least its corresponding low. Validation precedence is
// context, options, shape, non-finiteness over the complete input, then geometry.
// Nil/nonempty mismatches match both ErrShape and ErrNilInput via errors.Is.
// All-empty inputs, including nil slices, yield a non-nil result with non-nil
// empty channels. A short history yields positive-zero payloads and false masks.
// A ready zero stays valid. Result.IsNil checks only pointer presence.
//
// ErrNilContext, ErrNilInput, ErrInvalidOption, ErrShape, ErrNonFinite, ErrBar
// and ErrOverflow are stable categories. Cancellation returns ctx.Err(). Every
// batch error returns nil; no partially populated result is ever published.
// ErrOverflow protects packed output size, not the mathematically finite mean.
//
// # Ownership, cancellation and concurrency
//
// Inputs are read-only for the duration of a call. Outputs do not alias inputs;
// output channels are disjoint, capped and caller-owned. Independent calls on
// unchanged inputs are safe. Concurrent mutation of inputs or of the same result
// requires caller synchronization. The package has no mutable processing state.
// Context is polled at least every 1024 bars, every 1024 removals in long deque
// bursts, after validation and immediately before publication. Concurrent input
// errors and cancellation are reported in whichever order they are observed.
// A final context check is not a scheduling barrier against later cancellation.
//
// # Complexity and resource bounds
//
// Validation, kernel and output initialization take O(n) time. Every index is
// inserted once into each deque and removed at most once. Rings use O(min(P,n))
// indices, with no sentinel slot; wholly masked histories allocate no rings.
// Output payload is 25*n bytes, excluding headers and allocator rounding.
// Checked packed-array byte arithmetic prevents integer overflow, but cannot
// guarantee available RAM or make runtime out-of-memory termination recoverable.
//
// # Non-goals and AI-Hints
//
// No Stream, Snapshot, calendar, interpolation, sorting, revision handling,
// ATR normalization, width output or private execution/risk import is provided.
// General Upper-Lower can overflow even though Middle is finite; callers must
// separately govern derived width or ratio arithmetic. Never trade on warmup
// zeros or treat include-current boundaries as prior-bar breakout thresholds.
// Production uses no reflection, unsafe operations, arbitrary-precision arithmetic,
// embedded fixtures, panic calls, recovery calls, or test-only public exports.
package donchian
