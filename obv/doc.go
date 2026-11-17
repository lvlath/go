// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package obv computes deterministic On-Balance Volume (OBV) from validated
// chronological OHLCV observations. OBV is a cumulative signed-volume feature,
// not a bounded oscillator, buyer-identity classifier, or trading recommendation.
//
// Domain and mathematical law:
//   - The first accepted candle establishes previousClose and produces exactly
//     positive zero, with Valid=true; its Volume is never accumulated.
//   - Every subsequent candle contributes its own Volume when Close increases,
//     minus its Volume when Close decreases, and zero when Close is unchanged.
//   - Comparisons use the exact float64 ordering of successive closes. An equal
//     close or zero volume still advances the accepted-observation Count.
//   - The accumulator retains a Neumaier sum and its separate compensation;
//     the published value is their finite, representable effective total.
//   - A representable cancellation may be rebalanced without discarding the
//     correction. An unrepresentable accepted total returns ErrOverflow.
//
// Input and validation law:
//   - Compute requires five non-nil, equally sized, nonempty OHLCV columns;
//     ComputeCandles accepts a non-nil, nonempty chronological []Bar instead.
//   - Open, High, Low and Close must be finite and strictly positive. Volume
//     must be finite and nonnegative. Candle geometry requires
//     Low <= Open,Close <= High, and High >= Low.
//   - Structural error precedence is ErrNilInput, ErrEmptyInput, ErrShape.
//     Complete batch shape is checked before any element access, after which
//     the earliest bad candle wins; within one candle, ErrNonFinite precedes
//     ErrDomain. The entire batch is validated before allocating output.
//   - Arithmetic or Count overflow is distinct from invalid input and yields
//     ErrOverflow, never an artificial NaN, infinity, or partial Result.
//
// Processing, ownership and determinism:
//   - Compute and ComputeCandles traverse observations in input-index order and
//     use the same nextOBV transition as Stream.Push. Their successful values
//     are bit-identical to successive Push outputs in the same Go environment.
//   - Result.Values and Result.Valid are detached, caller-owned slices; every
//     entry is valid on success, including index zero. Neither facade retains
//     inputs, and callers must not mutate inputs concurrently with a batch call.
//   - New and Stream{} initialize the same state. Push validates, calculates a
//     candidate on a value copy, and commits only after every numeric check;
//     a failed Push leaves previous close, Count and both sums unchanged.
//   - Stream is single-owner mutable state without locks: simultaneous Push,
//     Snapshot, Restore or Reset operations require external synchronization.
//   - Reset begins a new cumulative epoch. Neither gaps nor session boundaries
//     are detected from market timestamps; the caller selects them explicitly.
//
// Versioned checkpoint and trust law:
//   - Snapshot.Algorithm is "obv/neumaier/v1"; Config is lowercase SHA-256 of
//     those exact ASCII bytes. FloatState is [previousClose,sum,compensation],
//     IntState has length zero, and Count includes the first seed-only bar.
//   - Restore validates metadata, lengths, finite numeric coordinates and
//     count-dependent seed invariants before publishing any state. Snapshot
//     and Restore never alias a caller-controlled slice.
//   - Config is a compatibility fingerprint, NOT an integrity check, MAC,
//     signature or proof of source-offset provenance. Authenticate the entire
//     serialized checkpoint and bind its feed cursor in the application.
//
// Error, nil and resource law:
//   - Exported sentinels are ErrNilInput, ErrEmptyInput, ErrShape,
//     ErrNonFinite, ErrDomain, ErrOverflow and ErrStateVersion; use errors.Is.
//   - Push and Restore on a nil *Stream return ErrNilInput; Reset(nil) is a
//     no-op and Snapshot(nil) returns the zero Snapshot{}.
//   - Snapshot on a real, zero-count Stream{} is different: it is a versioned
//     Count=0 snapshot with three zero coordinates, suitable for Restore.
//     A snapshot whose Count is MaxUint64 can be restored but cannot be advanced.
//   - (*Result).IsNil reports pointer nilness, not missing output data.
//   - Compute and ComputeCandles cost O(n) time and O(n) detached output space.
//     Stream.Push, Reset, Snapshot and Restore are O(1); Push allocates no heap
//     memory in its normal or error paths. Runtime OOM is not converted to an
//     error, and a caller-owned input must remain stable during computation.
//
// Explicit non-goals:
//   - No Options, context API, timestamps, hidden gap interpolation, option
//     defaults, implicit quote/base-volume conversion, trading threshold,
//     market participant attribution, or shared mutable indicator engine.
//   - Neumaier compensation reduces cancellation loss but does not guarantee
//     exact integer accumulation above 2^53, arbitrary precision, or universally
//     correctly rounded sums for every possible float64 input history.
//   - Comparing OBV magnitudes between instruments with different volume units
//     is not meaningful without an explicit caller-defined normalization.
//
// AI-Hints:
//   - Do not add the first candle's volume, filter flat/zero-volume candles,
//     replace compensation with plain sum+=delta, or silently repair OHLC.
//   - Do not mistake a rising OBV for proof of institutional accumulation;
//     it measures signed volume conditioned on the closing-price direction.
//   - Do not restore only the displayed value or trust Config as a security MAC;
//     retain both compensated components and authenticate replay metadata.
//   - See docs/OBV.md for independent mathematical oracles, operational
//     examples, error precedence, checkpoint boundaries and verification.
package obv
