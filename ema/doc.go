// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ema provides exact affine exponential moving average smoothing for batch
// and streaming pipelines over finite binary64 observations of either sign.
// This file defines the numerical, readiness, ownership, and continuation contracts.
//
// # What and why
//
// [Compute] returns index-aligned values and readiness masks; [New] creates a
// single-owner stream without retaining input history. [ComputeSMA], [ComputeFirst],
// [NewSMA], and [NewFirst] select a named seed policy while keeping the period
// explicit. They delegate to the canonical facades without changing arithmetic,
// validation precedence, ownership, or result shape. Options{} is invalid.
//
// Applications include signed sensor baselines, quote-noise filtering, and an EMA
// centerline for a separately implemented channel. The package does not compute
// ATR, channel widths, trading signals, statistical confidence, or safety limits.
// Timestamps, gaps, revisions, instrument identity, and event availability belong
// to the caller. The mathematical relationship to MACD is not a shared dependency
// or a promise of compatibility with its private seed and rounding implementation.
//
// # Implementation
//
// Compute validates context, options, output capacity, and every input value before
// allocating its result. A second ascending-index pass uses the same state
// transition as [Stream.Push]. Push validates first, advances a fixed-size private
// copy, rechecks context, and commits only on success. [Stream.Restore] validates
// schema, configuration, numeric fields, and phase before replacing any field.
//
// # Mathematical contract
//
// For period P, SeedSMA first publishes RN64(sum(x[0:P])/P) at index P-1. The sum
// of the represented binary64 inputs is exact; only the division result is rounded.
// SeedFirst publishes x[0] at index zero, after canonicalizing zero. Later steps are
//
//	RN64(((P-1)*previous + 2*x)/(P+1)).
//
// RN64 is round-to-nearest, ties-to-even. Integer weights and the numerator are
// exact; previous is the already rounded output of the preceding accepted step.
// This is a sequentially rounded recurrence, not an unrounded entire-history
// expression and not a rolling window that discards observations after P samples.
// Period one returns the current input directly. Every zero output is +0, so its
// bit identity excludes the sign of an input -0. Rounded underflow to +0 succeeds.
//
// # Numeric domain and bounded arithmetic
//
// Negative observations, both input zero signs, and finite subnormals are valid.
// NaN and either infinity are rejected, including during warmup and at period one.
// A 34-word sign-magnitude accumulator stores integer multiples of 2^-1074.
// MaxFloat64 needs 2098 bits in these units; multiplication by a uint64 count needs
// at most 2162 bits, within the 2176-bit array. Nonnegative affine weights sum to
// their denominator, so an admitted result remains within its finite operands.
// Neither x-previous nor unbounded floating-point numerator products are formed.
// No epsilon clamp, reflection, panic/recover, or dynamic big-number storage is used.
//
// # Options, shape, and result interpretation
//
// Options require 1 <= Period < MaxInt and SeedSMA or SeedFirst. Period counts
// accepted observations, not elapsed time. No default period or hidden seed change
// is applied. On success, Result.Values and Result.Valid both have len(x)
// entries. A false mask always accompanies +0; a true mask may also accompany +0.
// Empty and nil input slices produce a non-nil Result with non-nil empty channels.
// A nonempty SMA history shorter than P succeeds with only +0/false entries.
// [Result.IsNil] tests pointer presence only; it does not test readiness or shape.
// [Value] is a detached scalar output, not a reference to the live stream.
//
// # Errors and validation precedence
//
// Compute checks context, options, output capacity, then finite values in ascending
// index order, then computes. Its errors are ErrNilContext, ctx.Err(),
// ErrInvalidOption, ErrNonFinite, and ErrOverflow for output-size admission.
// Every batch failure returns a nil Result; no partial prefix is published.
// ErrShape and ErrInsufficient are absent: there is one input channel and short
// histories are successful. Classify sentinel errors with errors.Is.
//
// Push checks context, nil/initialized receiver, finite input, and Count capacity,
// then computes and checks context again. ErrNilInput identifies a nil receiver;
// ErrStateVersion identifies Stream{}; ErrOverflow rejects Count==MaxUint64.
// Any failed Push returns Value{} and leaves every state field unchanged. A
// successful warmup Push also returns Value{}, but with a nil error and an
// incremented Count. Check error before interpreting Valid.
//
// Restore returns ErrNilInput for a nil receiver and ErrStateVersion for an
// uninitialized receiver or malformed/incompatible snapshot. A mismatch in saved
// Options is a state error, not ErrInvalidOption. Reset on nil or Stream{} does
// nothing; Snapshot on either returns Snapshot{}. Reset on an initialized stream
// keeps its configuration and clears all history, including an unfinished seed.
//
// # Cancellation and publication
//
// Context is checked before input access, at least every 1024 observations in each
// batch pass, and immediately before result publication. Push checks before work
// and immediately before commit. Concurrent cancellation and an input error after
// entry have no fixed relative order: the first detected failure is returned.
// Cancellation after the last check cannot revoke an already completed commit.
// The caller must supply a valid Context implementation; its own Err method is
// outside this package's allocation and execution guarantees.
//
// # Ownership, concurrency, and determinism
//
// Compute reads without retaining input and returns separate caller-owned arrays.
// Copying Result by value copies slice headers, not the backing arrays; synchronize
// mutations of shared results. One Stream has one owner. Concurrent mutation or
// Snapshot during mutation requires external synchronization; there are no locks.
// Independent streams and batch calls are safe with stable caller-owned inputs.
//
// Inputs are consumed in ascending index/acceptance order. For the same represented
// values, explicit options, and arithmetic version, batch, successive successful
// Push calls, and a valid Snapshot/Restore continuation have identical output bits
// and masks. This does not promise equivalence to another EMA with a different
// seed or rounded alpha, nor bitwise commutation with arbitrary input rescaling.
//
// # Snapshot ABI and trust boundary
//
// Version is 1 and Algorithm is "ema/affine-rn/v1". Restore requires a target
// constructed with exactly the saved Options. Count is the number of accepted
// observations, not a source sequence number. Current must be finite; zero is +0.
// SeedSum is an exact integer in 2^-1074 units, encoded as canonical signed decimal
// ASCII matching 0|-?[1-9][0-9]* with no more than 660 bytes, including the sign.
//
// SeedSMA with Count<P requires Current=+0 and the bounded exact SeedSum. Its
// magnitude must not exceed Count*MaxBinary64Units. Count zero therefore requires
// SeedSum="0". Mature SMA states and every SeedFirst state require SeedSum="0";
// an unobserved SeedFirst state also requires Current=+0. Count==MaxUint64 is
// restorable in the mature phase, but the next Push fails with ErrOverflow.
// Reset explicitly starts a new history; Restore may select an older saved state.
//
// Parsing and rejection use fixed-size local storage, including oversized input
// rejection. Snapshot returns detached scalar/string state; a nonzero seed may
// allocate its decimal string. Preserve integer counts and binary64 values exactly
// in external codecs. Authenticate snapshots and atomically pair them with caller
// feed identity and offsets; structural validity proves neither authenticity nor
// historical reachability. Restoring an older accepted state is allowed.
//
// # Complexity and resource limits
//
// Let W=34 limbs, D<=660 decimal bytes, and n=len(x). New and Reset take O(W)
// initialization/copy work. Push takes O(W) time and fixed O(W) candidate space,
// with no internal heap allocation in warmup, seed completion, or maturity.
// For fixed binary64 and machine-word limits, these bounds are O(1) in n and P,
// not a claim of one machine instruction or constant timing across branches.
//
// Compute has two linear scans, O(n*W) time and O(n+W) total storage, including
// owning outputs. Snapshot takes O(D*W) bounded work and O(D+W) storage in the
// nonzero-seed case; Restore takes O(D*W) time and O(W) auxiliary space. Output
// byte-count checks are not a memory budget or an out-of-memory recovery mechanism.
// Benchmark constructor, Push, and checkpoint I/O separately on the target toolchain.
//
// # AI-Hints
//
// Do not substitute a running float mean or first-price shortcut for exact SMA.
// Do not rewrite the kernel as previous+alpha*(x-previous), including through FMA.
// Do not drop Valid, treat failed Push as accepted warmup, or change the source
// offset on error. Do not reuse MACD internals or change continuation arithmetic
// under the same Algorithm. Keep JSON, logging, and Snapshot outside the Push
// measurement loop. A previous-state anomaly baseline must be read before Push;
// the returned value already incorporates the current sample. A corrected history
// requires replay from a checkpoint preceding the first corrected observation.
//
// See docs/EMA.md for proofs, phase tables, complete programs, and operational
// recipes. The six functions in example_test.go are executable scenario references.
package ema
