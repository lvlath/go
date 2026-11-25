// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mfi computes a compensated rolling Money Flow Index on aligned OHLCV bars.
// It provides a canonical batch result, a zero-allocation streaming transition,
// and a detached, versioned checkpoint of the complete numerical recurrence.
//
// Domain and purpose:
//   - Each Bar records finite positive Open, High, Low, Close and finite Volume>=0.
//   - Low<=Open,Close<=High; Open validates candle geometry but does not enter
//     the typical price TP=RN64((High+Low+Close)/3).
//   - Volume uses one consistent base unit within an instrument's series.
//   - The first bar establishes a comparison baseline without a money-flow term.
//   - Later bars contribute current TP*Volume to the positive sign when TP rises,
//     to the negative sign when TP falls, or zero to both when prices are equal.
//   - Zero-volume and unchanged-price transitions still occupy ring slots.
//
// Mathematical and window laws:
//   - Period counts consecutive transitions, not positive or negative flows.
//   - The first valid value occurs at index Period after Period+1 accepted bars.
//   - Before warmup, Values[i]=0 and Valid[i]=false; afterward Valid is true.
//   - MFI is 100*positive/(positive+negative), evaluated with the smaller sum
//     divided by the larger to avoid overflowing the sum or a money-flow ratio.
//   - Both sums zero produce exactly 50; positive-only produces 100 and
//     negative-only produces 0. The range is [0,100], with no arbitrary clipping.
//
// Numerical law:
//   - Typical price is the nearest-even binary64 rounding of the exact mean of
//     the original binary64 High, Low and Close. Equal exact sums give identical
//     bits; distinct exact means rounding to the same price are flat transitions.
//   - The integer kernel uses one word for narrow exponent spans and a bounded
//     33-word stack accumulator otherwise, with exact quotient/remainder rounding.
//   - Raw flows and both rolling sums must be individually representable float64.
//   - Two Neumaier accumulators compensate the eviction-before-insertion recurrence.
//   - Every Period transitions, both candidate sums are recomputed deterministically
//     from ring elements oldest to newest. One extra rebase per epoch is permitted
//     on negative drift; a repeated failure returns ErrOverflow, never a clamp.
//   - Beyond the canonical price, raw products and rolling totals retain their
//     binary64 arithmetic; no exact-real or decimal guarantee is made. An unsuccessful calculation never commits state.
//
// Shape, options and errors:
//   - Options.Period=0 selects DefaultPeriod; valid explicit values are
//     between 1 and MaxPeriod inclusive. Invalid values return ErrInvalidOption
//     before allocating ring storage or reading batch inputs.
//   - Compute requires five non-nil, nonempty, equal-length OHLCV columns.
//     Shape/presence checks precede all element reads. After that, Compute
//     processes bars in ascending index order; inside each bar, nonfinite
//     fields precede domain checks. The earliest failing bar wins, so an
//     earlier ErrDomain can precede a later bar's ErrNonFinite.
//     ComputeCandles checks period, nil and empty row input before transposition.
//   - ErrNilInput, ErrEmptyInput, ErrShape, ErrInvalidOption, ErrNonFinite,
//     ErrDomain, ErrOverflow and ErrStateVersion are tested with errors.Is.
//   - Compute returns a nil *Result on every error. Push returns value=0,
//     valid=false, and preserves all existing state on every failure.
//   - A nil Stream Push/Restore returns ErrNilInput; uninitialized Stream{}
//     Push/Restore returns ErrStateVersion. Reset(nil) is a safe no-op;
//     Snapshot(nil) is the zero Snapshot.
//
// Public facades:
//   - Compute and New are canonical; their Default/Period wrappers only
//     select a policy and delegate without introducing alternate math.
//   - ComputeCandles adapts row-oriented []Bar into the canonical five columns;
//     Latest projects the last Compute value and its essential validity flag.
//   - The Result implements IsNil() bool for typed-nil interface compatibility.
//   - An allocated Result with an all-false Valid mask is a successful warmup,
//     never an invalid/nil result or evidence of zero directional pressure.
//
// Snapshot ABI:
//   - Algorithm is "mfi/rolling-neumaier/v2". Config is SHA-256 of that ASCII
//     identifier followed by a little-endian uint64 effective period, in hex.
//   - Restore rejects v1 checkpoints with ErrStateVersion. Replay original OHLCV
//     from the intended reset boundary; do not relabel old directional flows.
//   - FloatState=[previousTP,sumPositive,sumNegative,corrPositive,corrNegative,
//     positiveRing[Period],negativeRing[Period]].
//   - IntState=[used,head,extraRebaseUsed], Count counts accepted bars.
//   - Restore checks lengths, count/cursor consistency, ring sign domain and
//     recomputed effective sums within a documented binary64 roundoff envelope.
//     It preserves saved corrections rather than silently resetting the recurrence.
//   - Config is a configuration fingerprint, NOT a MAC over Snapshot bytes.
//     Persisted untrusted snapshots require external authentication of all fields.
//
// Ownership, ordering and concurrency:
//   - Batch inputs are read-only and never retained. Returned Result slices and
//     Snapshot arrays are detached, caller-owned copies. Restore copies back.
//   - Stream is mutable, lock-free, and belongs to one goroutine. Concurrent
//     Push/Reset/Restore/Snapshot on the same engine is unsupported.
//   - Consume bars in supplied order. Timestamp synchronization, missing candles,
//     exchange metadata, aggregation, unit conversion and Reset on gaps belong
//     to the caller. No clocks, network, random source or trading policy is used.
//
// Complexity:
//   - New: O(Period) time and space, with two float64 rings (16 MiB at MaxPeriod).
//   - Push: amortized O(1) time, O(Period) worst-case scheduled rebase,
//     O(1) extra memory, no successful-path allocation.
//   - Compute: amortized O(n+Period) time and O(n+Period) memory for n bars.
//   - ComputeCandles: amortized O(n+Period) time and additional five-column
//     O(n) transposition storage; Latest has the same cost as Compute.
//   - Snapshot/Restore/Reset: O(Period) time; Snapshot allocates O(Period),
//     while Restore/Reset reuse the engine's ring allocations. Restore may
//     allocate a short decimal encoding for checked native-int coordinates;
//     Push remains free of this conversion and its allocation cost.
//
// Non-goals:
//   - Identifying institutions, making trade recommendations, detecting events,
//     imputing missing bars, resampling timeframes, currency/base-unit conversion,
//     arbitrary-precision math, authenticated storage or synchronized multi-owner
//     streaming are outside this package's contract.
//
// AI-Hints:
//   - Never skip a flat or zero-volume bar when updating the ring: it changes
//     which old contribution expires and silently produces false MFI signals.
//   - Never implement a second batch recurrence or normalize flows by arbitrary
//     epsilon; negative drift must take the documented rebase/error path.
//   - Never discard the Valid mask or interpret a warmup zero as a bearish MFI.
//   - Never treat Config as authentication: MAC/sign the complete checkpoint
//     before passing untrusted persisted bytes to Restore.
package mfi
