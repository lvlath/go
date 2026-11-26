// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package atr provides exact True Range and Wilder Average True Range for
// ordered finite candles. It measures range in price units, not trading decisions.
//
// # What and why
//
// Compute returns aligned TR, ATR, and an ATR-readiness mask. New constructs a
// constant-state Stream. ComputePeriod/NewPeriod select an explicit period;
// ComputeDefault/NewDefault explicitly select DefaultPeriod=14. Canonical
// Options{} remains invalid; these wrappers add no new numerical modes.
// Snapshot/Restore preserves exact warmup and subsequent
// continuation. ATR can supply an independent width to a caller's volatility
// channel. No other lvlath package is imported and no EMA kernel is reused.
//
// # Mathematical laws
//
// The first TR is RN64(high-low). Later TR is the nearest-even rounding of the
// exact maximum of high-low, |high-previousClose|, and |low-previousClose|.
// This maximum equals max(high,previousClose)-min(low,previousClose), so the
// kernel selects endpoints without rounding and performs one subtraction.
// No previous close exists before the first accepted bar. Only successful bars
// advance PreviousClose; Reset removes it together with the entire history.
//
// The first valid ATR is RN64(sum(first P rounded TR)/P) at index P-1. Later
// ATR is RN64(((P-1)*previousATR+TR)/P) with exact integer weights and exactly
// one rounding. This is a sequentially rounded recurrence, not an unrounded
// whole-history formula. P=1 assigns ATR=TR bit for bit. All zero range/ATR
// outputs are canonical +0. TR is always present; Valid describes ATR only.
//
// # Arithmetic bounds
//
// A fixed unsigned 34-word accumulator represents units of 2^-1074.
// MaxFloat64Units=(2^53-1)<<2045 needs 2098 bits; even a uint64 multiple needs
// at most 2162 bits, within 2176. Seed and affine numerator are exact integers.
// Quotient, remainder, guard, and sticky bits determine nearest-even rounding,
// including subnormal underflow to +0. No floating numerator or rounded alpha
// is formed. Convexity bounds a valid ATR between finite nonnegative operands.
// Required TR overflow remains an error even when averaging could hide it.
//
// # Domains, shape, and errors
//
// Period is in [1, MaxInt]; there is no implicit default. Prices may be negative or
// signed zero but must be finite and satisfy low<=close<=high. There is no
// positive-price policy. Context precedes options, then shape, then per-index
// finite-domain and geometry validation, then numerical computation. A nil
// channel mixed with any nonempty channel gives ErrNilInput; other length
// mismatches give ErrShape. All-empty inputs, including nil slices, succeed
// with a non-nil Result and non-nil empty channels. A short history is warmup,
// not failure. Unavailable ATR is +0; a ready zero remains valid.
//
// ErrNilContext, ErrNilInput, ErrInvalidOption, ErrShape, ErrNonFinite, ErrBar,
// ErrOverflow, and ErrStateVersion are stable errors.Is categories. Context
// cancellation returns ctx.Err(). All batch errors return nil; all Push errors
// return Value{} without changing state. Result.IsNil tests pointer presence.
//
// # Cancellation, ownership, and concurrency
//
// Both linear batch passes poll context at least every 1024 bars and Compute
// checks again before publication. Push checks before work and before commit.
// Concurrent cancellation/input errors have no prescribed race ordering.
// A successful final context check is the publication boundary, not a scheduler
// barrier against cancellation immediately afterward. Caller inputs are read-only
// for the call; output slices are detached and caller-owned.
//
// One Stream has one owner and no internal locks. Concurrent calls on the same
// stream are unsupported; independent streams and batch calls are independent.
// Push is transactional with respect to validation/numerical/cancellation errors,
// not a concurrent-memory transaction. Nil/zero Snapshot returns Snapshot{};
// nil/zero Reset is a no-op. Push/Restore reject nil with ErrNilInput and zero
// values with ErrStateVersion. New is the canonical constructor; NewPeriod
// and NewDefault delegate to it without bypassing validation.
//
// # Snapshot ABI
//
// Version=1; Algorithm="atr/wilder-rn/v1"; complete Options must match New.
// Count zero requires a zero PreviousClose and SeedSum="0". During warmup,
// Current=+0 and SeedSum is the exact nonnegative seed integer. After readiness,
// SeedSum="0" and Current is finite and nonnegative (either zero sign is allowed).
// PreviousClose is finite and retains either zero sign; warmup Current must be +0.
// SeedSum grammar is 0|[1-9][0-9]*, at most 660 bytes; its magnitude cannot
// exceed Count*MaxFloat64Units. Restore validates before assignment and without
// heap allocation. Count=MaxUint64 may be restored, but another Push overflows.
// A valid snapshot is continuation data, not proof of authentic history;
// externally authenticate storage and preserve scalar bits/counts losslessly.
//
// # Complexity and operational limits
//
// W=34 limbs and D<=660 bytes are fixed by the numeric domain, not Period.
// Push takes O(W) time/space and zero internal heap allocations, including seed
// and first-ready steps. Compute takes O(n*W) time, O(n+W) space, with one
// domain-validation pass and one numerical pass. Snapshot/Restore take bounded
// O(D*W) time; Snapshot may allocate its returned decimal string. A custom
// context can itself allocate. Arithmetic byte-size checks do not guarantee
// available RAM; runtime out-of-memory termination is not recoverable here.
//
// # Non-goals and AI-Hints
//
// No position sizing, stop prices, NATR, division by ATR, profit guarantees,
// calendars, sorting, interpolation, or data delivery are implemented.
// Callers govern bar finality, missing observations, revisions, and event time.
// Historical corrections require explicit recomputation, not silent mutation.
// Do not replace Wilder with EMA, mask TR during warmup, fabricate a first close,
// or approximate SeedSum. There are no calls to panic/recover, no reflection,
// no unsafe code, and no arbitrary-precision allocation in the production kernel.
package atr
