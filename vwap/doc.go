// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap computes exact Volume-Weighted Average Price over explicitly
// anchored segments of finite price and nonnegative volume observations.
// This file defines the package boundary, continuation law, and caller responsibilities.
//
// # What and why
//
// [Compute] produces owning aligned Values and Valid channels. [New] constructs a
// single-owner Stream whose Push method uses exactly the same numerical transition.
// ComputeDefault and NewDefault delegate with Options{}; they never create an anchor,
// infer session times, allocate a reset channel, or change error precedence.
// Snapshot/Restore serializes exact accumulation for reproducible continuation.
// Session VWAP and anchored VWAP are the same arithmetic with different externally
// supplied reset schedules. No other lvlath package is imported.
//
// # Implementation
//
// Stage 1: validate context, shape, the first anchor, and rows in increasing index
// order. Options has no fields; there is no rejecting option branch. Compute
// completes a validation pass before allocating output or performing arithmetic.
// Stage 2: use one accepted-row transition for Compute and Push, resetting before
// the accepted row and accumulating exact signed products and nonnegative weights.
// Stage 3: round each ready prefix once; publish only a complete Result or a
// fully validated stream candidate after its final context check.
//
// # Inputs and results
//
// Price, volume, and reset channels have equal lengths and retain caller order.
// Prices have any finite sign; volumes are finite and nonnegative. Units must
// be compatible within a segment. Result owns Values and Valid of input length;
// masked entries are +0/false, not measurements. A ready zero remains valid.
// Copying Result is shallow because its slices share their existing backing
// arrays. Stream retains only fixed value state, not input arrays or trade history.
//
// # Mathematical and numerical laws
//
// A finite binary64 x is an integer multiple of 2^-1074. The implementation keeps
// VolumeSum=sum(V_i) in those units and WeightedSum=sum(P_i*V_i) in 2^-2148 units,
// with exact integer products. Published VWAP is RN64(WeightedSum/VolumeSum*2^-1074).
// Rounding is once, to nearest with ties to even; exact zero becomes +0, while a
// negative nonzero mean whose magnitude is at most half the least subnormal
// rounds to -0.
// No floating-point price*volume product is formed and no small term is dropped.
// A positive-weighted average lies between the finite prices carrying positive
// weight, so a valid average cannot overflow merely because a product or sum would.
//
// Umax=(2^53-1)<<2045 has 2098 bits. A uint64 count bounds the volume integer
// to 2162 bits and the weighted magnitude to 4260 bits. Fixed 67-word magnitudes
// provide 4288 bits, including bounded workspace for exact quotient rounding.
// A normalized two-limb quotient estimate, exact correction, and remainder test
// replace a floating estimate. Production code needs no arbitrary-precision heap.
//
// # Anchors, zero volume, and causality
//
// Every nonempty batch requires reset[0]=true. A fresh or Reset stream requires
// reset=true on its first accepted observation. A reset occurs before that row,
// and failed input or cancellation commits neither the reset nor the row.
// Before the first positive segment volume, output is Value{0,false}; afterward,
// a zero-volume row retains the previous valid average and increments Count.
// Signed finite prices and either zero sign are admitted. Invalid finite-domain
// input is still rejected on masked or zero-volume rows.
//
// An explicit reset is necessary but cannot prove causal availability: this API
// has no timestamp, AvailableAt, source identity, or event revision. The caller
// owns session calendars, bar finality, pivot confirmation, revisions, and replay.
// An HLC3 proxy supplied by the caller is not the exact mean of all underlying ticks.
//
// # Validation, errors, and ownership
//
// Context precedes options (vacuous in v1), shape, first anchor, then row validation
// in increasing index order. Within each row, nonfinite input precedes negative
// volume. Thus an earlier finite negative volume precedes a later NaN. Empty input
// succeeds with a non-nil Result and non-nil empty slices. A mixed nil/nonempty
// shape matches both ErrShape and ErrNilInput; ordinary length mismatch is ErrShape.
// Stable categories are ErrNilContext, ErrNilInput, ErrShape, ErrAnchor,
// ErrNonFinite, ErrVolume, ErrOverflow, and ErrStateVersion. There is no option
// error sentinel: every value of the empty Options domain is valid.
// Errors are classified with errors.Is, never by error text.
//
// Every batch error returns nil; every Push error returns Value{} and preserves
// all prior state. [Result.IsNil] is exactly a pointer-presence check. Inputs are
// read-only for each call; output channels are detached and caller-owned.
//
// Push validates context, receiver construction, first anchor, current-row
// finiteness, volume domain, and non-reset count exhaustion, in that order.
// Count includes every accepted zero-volume observation. At MaxUint64, another
// non-reset Push returns ErrOverflow, while a valid reset starts Count=1.
// Compute's ErrOverflow protects output-size arithmetic, not finite exact means.
//
// # Cancellation and concurrency
//
// Both linear batch passes poll at least every 1024 rows, and before publication.
// Push polls before work and just before commit. Concurrent cancellation and input
// errors have no promised interleaving order. Cancellation after the final poll
// does not retroactively revoke publication. Allocation itself is not cancellable.
// One Stream has one owner; concurrent same-stream operations are unsupported.
// Independent streams and batch calls share no mutable state. No internal locks
// or process-wide numerical configuration are used.
//
// # Snapshot ABI and bounds
//
// Version=1, Algorithm="vwap/exact-weighted/v1", and Options must match New.
// Started equals Count!=0; zero count requires zero sums. Each decimal sum is at
// most 1300 bytes and uses ASCII grammar 0|-?[1-9][0-9]*. VolumeSum is nonnegative.
// Bit bounds and exact inequalities V<=Count*Umax and |W|<=V*Umax are checked in
// fixed buffers before state assignment, without input-dependent heap allocation.
// These conditions imply W=0 when V=0, but do not prove trade-history provenance.
// Checkpoints require external authenticity and lossless uint64 serialization.
// A constructed empty snapshot is versioned and has both sums equal to "0";
// it is not Snapshot{}. No rounded mean is serialized: Restore reconstructs it.
// Restore has no context parameter and validates before one final assignment.
// Count is a segment observation count, never a source cursor or deduplication ID.
// Persist source identity/revision, cursor, and anchor policy atomically with state.
// Bound untrusted serialized input before decoding; Restore does not bound the
// allocations of an application JSON decoder or storage client.
// Nil/zero Snapshot returns Snapshot{}; nil/zero Reset is a no-op. Push/Restore
// reject nil with ErrNilInput and unconstructed Stream{} with ErrStateVersion.
//
// # Replay and composition
//
// [Stream.Reset] discards history immediately; it is not transactional with a
// separate later Push. Use Push with reset=true for reset-plus-observation rollback.
// Independent Compute calls cannot continue a prior segment. Retain a Stream or
// Restore into New/NewDefault and process only the unread suffix. Overlap without
// source admission control double-counts observations; no public merge/remove API
// exists. Corrected prices or weights require replay from before the correction.
// An unchanged next reset isolates later values; changed segmentation or source
// availability requires rebuilding the affected admission/reset schedule.
//
// VWAP supplies a weighted level, not ATR's range or Donchian's extrema. Caller
// composition must align rows, source revisions, masks, units, and information sets.
// A pre-observation benchmark comes from an already available prefix, not the
// current-inclusive VWAP or a prior session silently carried across a new anchor.
// Finite means do not make a subsequent difference or ratio finite or meaningful.
//
// # Complexity and non-goals
//
// L=67 words and D<=1300 decimal bytes are fixed protocol bounds. Push takes
// O(L) time/space, hence strict O(1) in accepted history length. Compute uses
// O(n*L) time across two linear scans, O(n) output, and O(L) workspace.
// Reset uses O(L) work and preserves construction/options. Snapshot/Restore take O(D*L)
// bounded time; Snapshot allocates returned nonzero strings, while Push and
// Restore need no internal heap allocations. JSON, logging, and allocating custom
// contexts are separate costs. Byte overflow checks do not guarantee
// available RAM or promise recovery from runtime out-of-memory termination.
//
// No moving window, trading strategy, profit guarantee, execution model, calendar,
// order book, interpolation, sorting, or retrospective anchor search is implemented.
// There are no production imports of reflect, unsafe, or math/big; no calls to
// panic or recover; and no embedded data. Exact rational arithmetic in tests is
// independent of the production integer representation and quotient algorithm.
//
// # AI-Hints
//
// Keep default wrappers as direct delegation; they must not fabricate an anchor.
// Preserve exact integer checkpoints rather than rounded means, and keep source
// cursor/session identity in an authenticated application envelope. Do not use
// Count as that cursor or mark a pending reset consumed before successful Push.
// A positive volume mask is not a profitability signal or evidence of contemporaneous data.
// Keep frozen vendor-demo bars separate from engineered tapes; a close-volume
// benchmark is not underlying-tick VWAP. Execute the five package scenarios in
// example_test.go and use docs/VWAP.md for proofs, lifecycle tables, and Go-only
// verification. All tests use package vwap_test with independent rational oracles.
package vwap
