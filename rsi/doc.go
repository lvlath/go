// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rsi implements J. Welles Wilder's Relative Strength Index as a
// deterministic, allocation-free-per-tick numerical primitive for ordered prices.
// It supports batch histories and online updates through one shared transition
// kernel, with versioned scalar checkpoints for continuation after failover.
//
// Domain and scope:
//   - Inputs are closing prices, or other explicitly chosen positive price marks,
//     represented by finite float64 values in one consistent price unit.
//   - Compute is the canonical batch facade and returns one detached Result with
//     index-aligned Values and Valid slices of the same length as its input.
//   - New constructs one independently owned Stream; each successful Push consumes
//     exactly one price and updates one scalar recurrence state.
//   - ComputeDefault, ComputePeriod, NewDefault, and NewPeriod are thin facade
//     wrappers; Latest projects the last observation of a complete Compute Result.
//   - Snapshot and Restore transport the exact state of a configured Wilder stream.
//
// Shape and phase laws:
//   - Options.Period=0 selects 14; a negative Period or overflowing Period+1 is invalid.
//   - The first accepted price is the baseline only and cannot produce RSI.
//   - The first P price differences seed arithmetic averages of gains and losses.
//   - The first valid RSI is returned after P+1 accepted prices, at batch index P.
//   - Every preceding Values entry is +0 with Valid=false, including short histories.
//   - Count includes all successful prices, not the number of published RSI values.
//   - A short but nonempty valid batch history returns successfully with invalid masks.
//   - Latest projects that final mask: before warmup it returns (0,false,nil).
//
// Numerical laws:
//   - Non-finite prices are rejected with ErrNonFinite; finite prices <=0 with ErrDomain.
//   - Delta is the new close minus the previously accepted close. Gain is the
//     positive delta, or zero; loss is the magnitude of a negative delta, or zero.
//   - Seed means are updated incrementally without forming potentially overflowing
//     sums. After warmup, both means follow Wilder's update
//     mean = mean + (observation-mean)/P, using one shared kernel for all entry points.
//   - When both means are zero RSI=50; when only loss is zero RSI=100; when only
//     gain is zero RSI=0. Otherwise the smaller/larger ratio is used to compute
//     RSI without forming a potentially overflowing gain+loss or gain/loss.
//   - Any nonfinite or out-of-range computed result returns ErrOverflow. The
//     kernel never clamps a failed calculation into the oscillator domain.
//   - Binary64 arithmetic is deterministic under one floating-point environment;
//     no universal exact-real-arithmetic or cross-architecture bitwise guarantee
//     is claimed. Very small contributions can disappear through rounding.
//
// Error precedence and atomicity:
//   - Compute checks options, nil input, empty input, then all prices in ascending
//     index order before allocating output or performing recurrence steps.
//   - Compute returns nil Result on any error; no partial result is published.
//   - Push and Restore reject nil receivers with ErrNilInput and zero-value
//     Stream receivers with ErrStateVersion before reading their input state.
//   - Push validates the price before checking Count exhaustion in the kernel;
//     an invalid price therefore takes precedence over Count overflow.
//   - Push computes on a copied candidate and commits once only on success;
//     failed steps preserve Count, previous, and both averages.
//   - Count overflow returns ErrOverflow before any arithmetic or state mutation.
//   - Restore rejects foreign algorithm/configuration IDs, wrong slice lengths,
//     nonfinite/negative means, invalid previous, and impossible phase states
//     with ErrStateVersion, leaving the destination unchanged.
//
// Snapshot ABI:
//   - Algorithm is exactly "rsi/wilder/v1".
//   - Config is lowercase SHA-256 hex over the Algorithm UTF-8 bytes followed by
//     the normalized period encoded as one uint64 in little-endian order.
//   - FloatState=[previous, avgGain, avgLoss] and IntState has length zero;
//     Restore accepts either nil or allocated-empty IntState with zero length.
//   - Count=0 requires three numeric zeros; Count=1 requires a positive previous
//     and zero means. Count=2 has one signed change and cannot have two positive means.
//   - Config identifies period/version and does not authenticate Count or averages.
//     External checkpoint integrity requires a MAC or signature over encoded bytes.
//   - Transport encoding must preserve floating-point bits; the ABI does not prove
//     that a structurally valid checkpoint originates from a genuine price history.
//
// Ownership, concurrency, and cancellation:
//   - Compute never mutates or retains its input; returned Result slices are detached
//     and owned by the caller, even after multiple invocations on the same input.
//   - Snapshot allocates detached state slices; Restore never retains their backing
//     arrays, so subsequent caller mutation cannot affect the restored stream.
//   - Streams are single-owner mutable objects: concurrent Push, Reset, Snapshot,
//     and Restore on one receiver are unsupported without external synchronization.
//     No locks are owned by this package; there is no internal lock ordering.
//   - Reset(nil) is a safe no-op; Snapshot(nil) and Snapshot(&Stream{}) return
//     Snapshot{}, which is not a restorable initialized checkpoint.
//   - An uninitialized Stream{} is not interchangeable with a stream
//     initialized by New.
//   - The package has no context-bearing entry point and no cancellation contract;
//     consumers manage their own job or feed lifetime outside the numeric kernel.
//
// Determinism and complexity:
//   - Each history is processed strictly in ascending input index/Push call order.
//     No sorting, global random state, wall clock, internal concurrency, or gap
//     heuristics alter that order. Batch and streaming invoke exactly stepWilder.
//   - Compute and batch convenience wrappers take O(n) time and O(n) detached
//     output memory, with O(1) temporary recurrence state.
//   - Latest has O(n) time and O(n) transient output memory by delegation to Compute.
//   - New/Reset/Push/Snapshot/Restore take O(1) time and state memory; Push does
//     not allocate after initialization, including seed, valid, and rejected steps.
//
// Non-goals and operational boundaries:
//   - No 30/70 thresholds, trade signals, position sizing, parameter optimization,
//     forecasting, exchange API, persistence, time series schema, or gap interpolation.
//   - Caller code validates timestamp order and sequence continuity, computes
//     correct bar/quote prices, and explicitly calls Reset on true discontinuities.
//   - Persist feed offsets separately from Snapshot and do not replay the last
//     checkpointed price: Count already includes every accepted close.
//   - Market data sources may encode missing prices as zero; that convention is
//     not accepted here because a missing bar is not a mathematical zero price.
//   - The package depends only on the Go standard library and does not import
//     core, matrix, series, stats, or any trading-domain functionality.
//
// AI-Hints:
//   - Never use Values[i] without Valid[i]; a warmup zero is not evidence of RSI=0.
//   - Never substitute the classic unbounded gain/loss quotient for the stable branch.
//   - Keep all batch and streaming arithmetic inside the same stepWilder transition.
//   - Restore only after external integrity verification, and never reinterpret
//     a checkpoint created for a different normalized period.
//   - Do not reuse a stream across instruments or process concurrent feed mutations
//     without explicit serialization of access.
package rsi
