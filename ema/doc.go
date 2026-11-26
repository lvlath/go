// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema provides exact affine exponential moving average smoothing for batch
// and streaming pipelines. It measures a finite numeric history, not trading decisions.
//
// # What and why
//
// Compute provides aligned values and readiness masks. New creates a constant-state
// Stream. ComputeSMA/ComputeFirst and NewSMA/NewFirst select a named seed policy
// while keeping the period explicit and delegating to the canonical facades.
// Snapshot/Restore preserves exact warmup state for deterministic failover.
// Typical consumers are noise filters and trend baselines such as a Keltner centerline;
// this package does not implement channel widths, ATR, execution, or market calendars.
//
// # Mathematical contract
//
// SeedSMA first publishes RN64(sum(x[0:P])/P) at index P-1, using an exact sum.
// SeedFirst publishes x[0] at index zero. Subsequent outputs are
// RN64(((P-1)*previous+2*x)/(P+1)), with exact represented binary64 operands and
// exact integer coefficients. RN64 means round-to-nearest, ties-to-even.
// This is a sequentially rounded recurrence, not an unrounded entire-history formula.
// Period one directly returns the input. Every zero output is canonical positive zero,
// so the period-one bit identity excludes the sign of an input negative zero.
//
// # Numeric domain and bounded arithmetic
//
// All finite binary64 observations are admitted, including negative values and
// subnormals. A 34-word sign-magnitude accumulator measures units of 2^-1074.
// MaxFloat64 needs 2098 bits in those units; multiplying by any uint64 count needs
// at most 2162 bits, below the allocated 2176. Affine weights are nonnegative and
// sum to their denominator, so a valid result lies within its finite operands.
// No floating-point difference, epsilon clamp, arbitrary-precision allocation,
// reflection, panic, or recovery is used by the package.
//
// # Options, shape, and errors
//
// Period must satisfy 1 <= Period < MaxInt; no defaults are applied. Unknown Seed
// values are errors. Compute checks context, options, output capacity, then input
// values in ascending order before calculation. Empty input returns non-nil empty
// channels. Short SMA histories return only +0/false entries, not an error.
// ErrNilContext, ErrInvalidOption, ErrNonFinite, and ErrOverflow classify batch
// failures. Push additionally reports ErrNilInput and ErrStateVersion for invalid
// receivers. Restore uses ErrNilInput or ErrStateVersion. Use errors.Is.
// ErrShape is deliberately absent: this API has no pair of input lengths to compare.
//
// # Cancellation, ownership, and determinism
//
// Context is checked before input access, at least every 1024 observations in each
// batch pass, and immediately before publication. Push has an initial and a final
// context check. All errors publish no batch result and change no stream field.
// Concurrent cancellation and a detected input error have no fixed relative order.
// A successful final check is a publication boundary, not a scheduler barrier.
// Results are caller-owned and never alias inputs. One Stream has one owner and
// requires external synchronization for concurrent calls. Independent streams and
// independent batch computations may run concurrently.
//
// # Snapshot ABI
//
// Version=1, Algorithm="ema/affine-rn/v1". Options must exactly match New.
// SMA warmup has Current=+0 and exact SeedSum; outside warmup SeedSum="0".
// Count zero requires an empty mathematical state. SeedSum is canonical signed
// decimal ASCII, at most 660 bytes, with magnitude <= Count*MaxBinary64Units.
// Restore performs no heap allocation, including rejection of oversized numbers.
// Current is finite and zero-canonical. Count=MaxUint64 is restorable, but another
// Push fails with ErrOverflow. Persist all scalar bits/counts losslessly and
// authenticate state externally; validation is not a proof of history provenance.
// Nil/zero Stream Snapshot returns Snapshot{}; Reset is a no-op on either.
//
// # Complexity and limitations
//
// Let W=34 limbs and D<=660 decimal bytes. Push takes O(W) time and fixed O(W)
// candidate storage, with no internal heap allocation in warmup or maturity.
// Compute has two linear scans, O(n*W) time and O(n+W) storage. Snapshot and Restore
// take O(D*W) bounded time; Snapshot may allocate its returned decimal string.
// An allocating custom context can introduce allocations outside the kernel.
// Runtime out-of-memory termination is not converted into a recoverable error.
// Inputs are already ordered observations: timestamps, gaps, revisions, source
// checkpoints, and event availability are the caller's responsibilities.
//
// # AI-Hints
//
// Do not reuse the MACD private kernel: its published rounding/seed contract differs.
// Do not replace exact SMA seeding with a running float mean or first-price shortcut.
// Do not assume translation commutes bitwise with binary64 rounding; use a rational
// oracle for the transformed represented inputs. Keep snapshot I/O out of Push loops.
package ema
