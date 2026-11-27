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
// # Implementation and causality
//
// Validate the entire batch, then scan sessions and bars in ascending index order.
// For each bar, copy only a saved previous summary. Aggregate current prices in
// separate local scalars. Commit those scalars only after the session ends and
// only if Complete is true. Therefore a change confined to session j cannot change
// any successful output in sessions 0 through j. Its summary can affect session
// j+1 only. An invalid later price still rejects the whole batch; success/error
// prefix invariance is not promised for malformed extensions.
//
// # Inputs and options
//
// Session ranges are nonempty, half-open, ordered, adjacent, and cover [0,n).
// Only the final session may have Complete=false. For empty prices, sessions must
// also be empty. Complete is a caller assertion, not proof of exchange coverage.
// All three price slices must be non-nil, equally sized, finite, and satisfy
// low <= close <= high. Negative prices, signed zeros, and subnormals are valid.
// Options{} is valid and introduces no defaults or configuration errors.
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
// Precedence is nil/canceled context, nil prices, shape, empty/session structure,
// global non-finiteness, finite candle geometry, computation. Within the finite
// scan, visit increasing indices and high, low, close at each index. Remember a
// bad finite candle until all prices are known finite. Classify through errors.Is.
// Every error returns nil, never a partially useful Result. Concurrent cancellation
// and an input error after entry have no prescribed winner. ErrEmptyInput is a
// reserved, deprecated identity; valid empty input succeeds. IsNil is exactly a
// receiver-pointer comparison, not readiness or channel validation.
//
// # Ownership and concurrency
//
// Successful results have four non-nil input-length slices, allocated independently.
// No output aliases an input, another channel, or another call. Inputs are neither
// modified nor retained. Copying Result by value is shallow. Independent calls may
// share read-only input; input mutation during a call is unsupported. Concurrent
// writes to a returned result require caller synchronization. No locks or shared
// algorithm state are needed.
//
// # Complexity, allocation, and cancellation
//
// With n bars and s sessions, total work including validation and output zeroing
// is O(n+s), output space O(n), and auxiliary space O(1). The sequential kernel is
// one pass; the whole API also performs validation scans. Four output buffers and
// a result object are required for a nonempty call; this owning API is not allocation-free.
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
// No reflection, unsafe operations, embedded fixtures, panic, or recovery calls are used.
package sessionlevels
