// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sessionlevels provides previous completed session High/Low/Close levels without look-ahead bias.
// This file defines the package charter, numerical laws, and operational boundaries.
//
// # What and why
//
// Compute selects the maximum high, minimum low, and final close of the strictly
// preceding completed session. Every bar in one session receives the same levels.
// The first session has no predecessor: Valid is false and all payloads are +0.
// These are observed reference levels, not predictions, orders, fills, or profit claims.
//
// # Public surface and explicit policy
//
// Session carries the caller's half-open observed-index interval and Complete
// assertion. Options{} is the only configuration. Compute is canonical;
// ComputeDefault delegates with Options{}, and ComputeDaily and ComputeWeekly
// are names for the same delegation with caller-prepared day/week partitions.
// None derives, validates, or changes a calendar frequency. The same manifest
// produces the same bits and error classification through all four facades.
// There is no period, current-session overlay, or implicit previous-day policy.
//
// Result owns PreviousHigh, PreviousLow, PreviousClose and one shared Valid
// slice. Each channel has length n, including non-nil zero-length channels on
// successful empty input. The zero-value Result is not an allocated empty result.
//
// # Implementation and causality
//
// Validate the entire batch, then scan sessions and bars in ascending index order.
// For each bar, copy only a saved previous summary. Aggregate current prices in
// separate local scalars. Commit those scalars only after the session ends and
// only if Complete is true. Therefore a change confined to session j cannot change
// any successful output in sessions 0 through j. Its summary can affect session
// j+1 only, assuming unchanged segmentation. Extending an admitted prefix with
// consistent intervals preserves its output bits; a cut inside a session clips
// End and sets Complete=false. An invalid later price rejects the whole batch.
// This is numerical causality on admitted histories, not proof of when a source
// released or revised a price. The caller must establish that availability.
//
// # Inputs and options
//
// Session ranges are nonempty, half-open, ordered, adjacent, and cover [0,n).
// Only the final session may have Complete=false. For empty prices, sessions must
// also be empty. Complete is a caller assertion, not proof of exchange coverage.
// All three price slices must be non-nil, equally sized, finite, and satisfy
// low <= close <= high. Negative prices, signed zeros, and subnormals are valid.
// Options{} is valid and introduces no defaults or configuration errors.
// A nil H/L/C slice fails even when all three are nil. Three non-nil empty
// slices with no sessions succeed. Both nil and empty session lists are admitted
// only for empty prices. Every nonfinal session must be complete.
//
// # Numeric policy and determinism
//
// No prices are added, subtracted, averaged, rounded, or normalized. Extrema use
// strict comparisons starting from the first bar: earliest equal values win,
// including ties between +0 and -0. The final close is copied bit for bit. A ready
// zero remains valid; an unready payload is always +0. Finite extrema cannot create
// arithmetic overflow. There is no epsilon or clamp.
//
// # Errors and nilability
//
// A nil context returns ErrNilContext; an already canceled or expired context
// returns ctx.Err(). Next come ErrNilInput for any nil H/L/C slice, ErrShape
// for unequal lengths, and ErrSession for invalid coverage or completion.
// Only after structural admission does price validation scan increasing bar
// indices, checking high, low, close for NaN/Inf. Any ErrNonFinite takes
// precedence over the first remembered finite ErrBar (low <= close <= high).
// Masked and final incomplete bars are not exempt from validation.
// There is no invalid-option or arithmetic-overflow category for this selection.
//
// Every failure returns nil, error. Classify with errors.Is, not error text.
// Cancellation observed during work returns ctx.Err(); a racing input error
// after entry has no prescribed winner. Successful empty or masked results are
// not errors. IsNil is exactly a receiver-pointer comparison; it does not
// validate channel shape, determine readiness, or certify source history.
//
// # Ownership and concurrency
//
// Successful results have four non-nil input-length slices, allocated independently.
// Each has capacity n. No output aliases an input, another channel, or another
// call. Inputs are neither modified nor retained. Copying Result by value is
// shallow. Independent calls may
// share read-only input; input mutation during a call is unsupported. Concurrent
// writes to a returned result require caller synchronization. No locks or shared
// algorithm state are needed.
//
// # Complexity, allocation, and cancellation
//
// With n bars and s sessions, total work including validation and output zeroing
// is O(n+s), output space O(n), and auxiliary space O(1). The sequential kernel is
// one pass; the whole API also performs validation scans. Four output buffers and
// a result object form the owning allocation layout for a nonempty call. The
// facade wrappers introduce no additional buffers or passes.
// There are no per-session arrays, matrices, or heap allocations inside hot loops.
//
// Each float buffer is allocated with an already existing float-slice length n;
// no multiplied or summed allocation length is constructed. A valid Go input
// slice already witnesses representability of that individual float buffer size.
// The bool buffer is smaller. This does not reserve total RAM or recover runtime
// out-of-memory failure. There is no arbitrary user-visible memory limit.
//
// Context is checked at entry, every 1024 descriptors in session validation,
// every 1024 bars in each price/kernel scan, before allocation, and immediately
// before publication. These are phase-local iteration bounds, not time bounds.
// Runtime allocation is not interruptible. Cancellation following the final check
// cannot retroactively retract a result. Typed-nil or misbehaving custom contexts
// are outside the contract and are not detected through reflection or recovery.
//
// # Composition and operational boundaries
//
// The caller may supply ordered grid-checked observations from series, but
// sessionlevels imports no other lvlath package. Donchian uses a rolling bar
// count with explicit current-bar policy; VWAP uses a weighted prefix with
// explicit reset flags; ATR measures and smooths ranges including price gaps.
// None of those values substitutes for the previous completed session triple.
// When combining features, align original row identity, units, readiness and
// availability time. A previous-session threshold can be known before the
// current bar only if its predecessor was actually finalized and available then.
// A finalized current close comparison becomes observable at that close, not
// before it. Consumer differences, ratios and execution rules are external.
//
// # Non-goals and AI-Hints
//
// No calendar, timezone, network, timestamp, series, or other lvlath package is
// imported. A caller such as katalvlaran supplies exchange/session policy, DST,
// holidays, shortened sessions, data revisions, missing-bar evidence, and finality.
// Daily and weekly levels require separate partitions and separate calls.
// A replay cut clips its final End to n and marks a truncated interval incomplete.
// Zero-trade days are not represented by fabricated zero candles or empty sessions.
// Future-known schedule boundaries are not permission to consume future price totals.
// The package has no Stream, Snapshot, execution model, or mutable model state.
// An isolated new-session batch loses its predecessor. Replay with an admitted
// complete preceding interval and rebase indices; never mark a truncated left
// fragment complete to manufacture readiness. With fixed boundaries a price
// correction changes at most the following session's levels; changed boundaries,
// completion or source admission require explicit revalidation and recomputation.
// Check err before Valid, never treat a warmup zero as a level or compare across
// different adjustment bases. Do not normalize or subtract extreme finite
// levels without independently checking consumer arithmetic.
// No reflection, unsafe operations, embedded fixtures, panic, or recovery calls are used.
package sessionlevels
