// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd implements the SMA-seeded Moving Average Convergence/Divergence indicator.
// It combines two independently seeded price EMAs, a signed MACD-line signal EMA,
// and a signed histogram in one deterministic batch/streaming recurrence.
//
// Domain and purpose:
//   - Consume one caller-supplied, strictly positive finite closing-price sequence.
//   - Publish MACD Line (fast EMA minus slow EMA), Signal (EMA of valid Lines),
//     and Histogram (Line minus Signal) with independent warmup readiness masks.
//   - Provide a constant-state Stream with deterministic Snapshot/Restore failover.
//   - Leave timestamp cadence, missing-market-data policy, order execution,
//     risk controls and investment decisions to upstream/downstream applications.
//
// Mathematical laws:
//   - The SMA seed of any EMA period P uses its first P accepted observations.
//   - Each subsequent EMA step uses alpha=2/(P+1); seed is never close[0] alone
//     unless the explicitly selected period P equals one.
//   - Fast and Slow both start at close index zero, even though their seeds
//     finish at different indices. Signal starts only at the first valid MACD Line.
//   - MACD uses the Appel-style EMA factor 2/(P+1), not the Wilder 1/P
//     smoothing used by RSI. Matching period numbers do not imply matching kernels.
//   - For zero-based indices, the first valid Line is Slow-1. The first valid
//     Signal and Histogram are Slow+Signal-2, including Signal=1.
//   - With Signal=1, Signal matches Line bitwise and Histogram is exactly +0.
//   - Before a value becomes ready, its published value is +0, but Valid is false.
//     Histogram.Valid equals Signal.Valid at every position.
//   - After C accepted closes, Fast/Slow seed counters are min(C,Fast/Slow).
//     Signal's seed counter is min(max(C-Slow+1,0),Signal), computed without
//     unsigned underflow. Counters saturate, while EMA values keep evolving.
//
// Public API and option policy:
//   - Compute is the canonical batch facade; New constructs the Stream facade.
//   - ComputeDefault/ComputeWithPeriods and NewDefault/NewWithPeriods forward to
//     those canonical functions without introducing alternate arithmetic.
//   - Options{} independently selects Fast=12, Slow=26, Signal=9. After defaults,
//     require 1<=Fast<Slow and Signal>=1; checked Slow+Signal-2 must fit an int.
//   - Compute accepts non-nil, nonempty []float64. Histories shorter than warmup
//     are successful and retain input-length arrays with readiness=false.
//   - Line requires Slow accepted prices; Signal/Histogram require
//     Slow+Signal-1 accepted prices. The separate masks preserve original indexes.
//   - Result.IsNil reports pointer presence only; it does not mean that any
//     output position is ready or that an allocated Result was produced by Compute.
//
// Numeric law:
//   - Input close is finite and strictly positive. Negative or zero prices are
//     rejected, as are NaN and both infinities.
//   - Line, Signal and Histogram are SIGNED and must not reuse price validation.
//   - SMA seeds use bounded online means rather than raw potentially overflowing
//     sums. Opposite-sign signal updates use weighted terms rather than x-old.
//   - Every required intermediate and final float result must be representable;
//     failed computation returns ErrOverflow, never silently published Inf/NaN.
//   - Online SMA is a fixed-order finite-precision running mean, not a guarantee
//     of correctly rounded mathematical sums for every binary64 input.
//   - A valid Line/Signal/Histogram may equal zero or have either sign. Validity
//     is a phase property, never a comparison against a numeric zero threshold.
//   - Finite IEEE-754 arithmetic is deterministic for the same execution target,
//     not an arbitrary-precision or cross-toolchain bit-identity guarantee.
//
// Error law and precedence:
//   - Compute checks options, nil, empty, all nonfinite closes, then all
//     nonpositive closes, in that precedence order; both scans use original
//     ascending indices. It returns (nil,error) on every failure.
//   - ErrInvalidOption: malformed or overflowing periods; ErrNilInput: nil input
//     slice/stream receiver; ErrEmptyInput: allocated empty batch.
//   - ErrNonFinite: nonfinite close; ErrDomain: finite nonpositive close;
//     ErrOverflow: unrepresentable count/EMA/line/histogram; ErrStateVersion:
//     invalid zero Stream or incompatible/malformed Snapshot.
//   - Use errors.Is, never error-message parsing, for program control.
//   - Push errors return Point{} and DO NOT COMMIT ANY recurrence field.
//   - Restore errors leave every Stream field untouched.
//   - Push checks nil/uninitialized receivers, finite/positive close, then
//     arithmetic and the checked total counter. Failed Push returns Point{}.
//   - Reset on nil or zero Stream is a no-op; Snapshot on either returns the
//     empty Snapshot{}. The zero-value Stream cannot be used with Push/Restore.
//
// Ownership, concurrency and state:
//   - Compute reads without retaining input closes; its six output slices are
//     independent caller-owned memory and can be read concurrently if not mutated.
//   - Stream is a single-goroutine mutable object; no synchronization lock exists.
//     Concurrent mutation, or Snapshot during mutation, is unsupported.
//   - Snapshot exports detached arrays of [fast,slow,signal] means and saturated
//     counters, plus Count, Algorithm="macd/sma-ema/v1" and a SHA-256 config identity.
//   - Config = lowercase hex SHA-256(ASCII Algorithm || LE64(Fast) ||
//     LE64(Slow) || LE64(Signal)). It is NOT an authentication tag for the state.
//   - Restore checks the target's normalized periods, slice lengths, saturated
//     counter identities, finite means, positive observed price means, and
//     canonical +0 for unobserved means before changing any field.
//   - A snapshot does not include the last source sequence number, wall-clock
//     timestamp, data provenance or an authenticated checksum of the state.
//   - Nil Push/Restore return ErrNilInput; nil Reset does nothing; nil Snapshot
//     returns Snapshot{}. An uninitialized Stream{} is not usable and reports
//     ErrStateVersion from Push/Restore.
//   - No public context API is provided: cancellation is application-controlled
//     between Push calls rather than a hidden cancellation behavior.
//
// Complexity and resource profile:
//   - Compute: O(n) time and O(n) space for three independently masked series.
//   - New, Push, Reset and Restore: O(1) time and auxiliary memory per call.
//   - Push is allocation-free after New; Snapshot allocates two fixed-size slices.
//   - No price history, global RNG, reflect, panic/recover, network, or clock is used.
//
// Non-goals and operational limitations:
//   - No exchange gateway, gap interpolation, probability/confidence estimate,
//     portfolio allocation, trading recommendation, or cross-sectional ranking API.
//   - Snapshot version/config/phase checks are structural, not proof that the means
//     came from a genuine price history. Authenticate external persistence separately.
//   - Different charting libraries may seed Fast from later windows; this version
//     intentionally starts both Fast and Slow at the same first close index.
//   - A stream is not a clock or sequence validator: a missing bar, duplicate
//     event or changed instrument must be handled explicitly by the caller.
//     A new independent history requires Reset and another complete warmup.
//
// AI-Hints:
//   - Never replace SMA seeding with the first quote or truncate warmup zeros
//     without applying the same index alignment to timestamps and other features.
//   - Never treat a zero-valued Histogram as invalid: read Signal.Valid or SignalValid.
//   - Never introduce a second MACD kernel in Compute or update Fast/Slow in place
//     before validating the signed signal and final histogram.
//   - Use one Stream per instrument/cadence, and guard all derived business
//     decisions with readiness masks and upstream feed-quality checks.
//   - Keep complete checkpoint ownership outside this package: authenticate
//     persisted state, retain exact uint64 and float64 representations, and resume
//     from the next unprocessed source event rather than replaying a committed one.
package macd
