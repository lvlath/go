// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap computes exact Volume-Weighted Average Price over explicitly
// anchored segments of finite price and nonnegative volume observations.
//
// # What and why
//
// Compute produces owning aligned Values and Valid channels. New constructs a
// single-owner Stream whose Push method uses exactly the same numerical transition.
// ComputeDefault and NewDefault delegate with Options{}; they never create an anchor,
// infer session times, allocate a reset channel, or change error precedence.
// Snapshot/Restore serializes exact accumulation for reproducible continuation.
// Session VWAP and anchored VWAP are the same arithmetic with different externally
// supplied reset schedules. No other lvlath package is imported.
//
// # Mathematical and numerical laws
//
// A finite binary64 x is an integer multiple of 2^-1074. The implementation keeps
// VolumeSum=sum(V_i) in those units and WeightedSum=sum(P_i*V_i) in 2^-2148 units,
// with exact integer products. Published VWAP is RN64(WeightedSum/VolumeSum*2^-1074).
// Rounding is once, to nearest with ties to even; exact zero becomes +0, while a
// negative nonzero value below half the least subnormal may round to -0.
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
// ErrNonFinite, ErrVolume, ErrOverflow, and ErrStateVersion. ErrInvalidOption is
// retained for compatibility; it is unreachable for the empty Options domain.
// Errors are classified with errors.Is, never by error text.
//
// Every batch error returns nil; every Push error returns Value{} and preserves
// all prior state. Result.IsNil is exactly a pointer-presence check. Inputs are
// read-only for each call; output channels are detached and caller-owned.
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
// Nil/zero Snapshot returns Snapshot{}; nil/zero Reset is a no-op. Push/Restore
// reject nil with ErrNilInput and unconstructed Stream{} with ErrStateVersion.
//
// # Complexity and non-goals
//
// L=67 words and D<=1300 decimal bytes are fixed protocol bounds. Push takes
// O(L) time/space, hence strict O(1) in accepted history length. Compute uses
// O(n*L) time, O(n) output, and O(L) workspace. Snapshot/Restore take O(D*L)
// bounded time; Snapshot allocates returned nonzero strings, while Push and
// Restore need no internal heap allocations. Byte overflow checks do not guarantee
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
// cursor/session identity in an authenticated application envelope. A positive
// volume mask is not a profitability signal or evidence of contemporaneous data.
package vwap
