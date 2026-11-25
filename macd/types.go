// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package macd declares the public, versioned data carriers of the MACD engine.
// This file fixes ownership, warmup masks, result nilability, and stream state.
package macd

// Options selects three distinct smoothing horizons for the MACD recurrence.
// A zero field chooses its conventional default without affecting other fields.
//
// Behavior highlights:
//   - Fast and Slow are SMA-seeded EMA periods over the same close history.
//   - Signal is a separate SMA-seeded EMA over valid, possibly negative MACD lines.
//
// Inputs:
//   - Fast: price observations in the fast seed, default 12; at least 1.
//   - Slow: price observations in the slow seed, default 26; strictly greater than Fast.
//   - Signal: valid MACD lines in the signal seed, default 9; at least 1.
//
// Determinism:
//   - The three normalized periods determine warmup positions and the snapshot config digest.
//
// Complexity:
//   - Time O(1), Space O(1) for option normalization.
//
// Notes:
//   - This value is copied; the package never retains a mutable reference to it.
//   - Invalid periods and Slow+Signal-2 integer overflow yield ErrInvalidOption.
//
// AI-Hints:
//   - Setting just Signal=1 retains the default Fast=12 and Slow=26 periods.
//   - A period of 1 is allowed for Fast and Signal; Fast must still be less than Slow.
//   - These are observation counts, not durations; establish a consistent bar cadence upstream.
type Options struct {
	Fast   int // Fast price EMA period; 0 selects 12.
	Slow   int // Slow price EMA period; 0 selects 26.
	Signal int // Signal-line EMA period; 0 selects 9.
}

// Series carries one length-preserving indicator sequence and its readiness mask.
// Values[i] is meaningful only when Valid[i] is true; invalid positions contain +0.
//
// Behavior highlights:
//   - Each slice has exactly the input history length, with equal index alignment.
//
// Returns:
//   - Values: caller-owned float64 samples, including placeholder zeros during warmup.
//   - Valid: caller-owned boolean readiness flags, independent of numeric zeros.
//
// Determinism:
//   - Observations are stored in the original close order without shifting or truncation.
//
// Complexity:
//   - Storage O(n) per Series; O(1) indexing for either slice.
//
// Notes:
//   - Mutating a returned Series is permitted after Compute returns, but not concurrently
//     with readers unless the caller synchronizes it. The stream does not retain the series.
//
// AI-Hints:
//   - Histogram zero can be an entirely valid result; never infer validity from Values[i].
//   - Align downstream labels/features using Valid, not by dropping a leading prefix blindly.
type Series struct {
	Values []float64 // Indicator values; +0 before the corresponding warmup completes.
	Valid  []bool    // Readiness per original close index.
}

// Result exposes the aligned MACD line, signal EMA, and signed histogram.
// It is the single canonical batch result: a successful Compute owns six detached slices.
//
// Behavior highlights:
//   - Line becomes ready at index Slow-1.
//   - Signal and Histogram become ready at index Slow+Signal-2.
//   - Histogram.Valid is identical to Signal.Valid at every index.
//
// Returns:
//   - Line: fast price EMA minus slow price EMA, possibly negative.
//   - Signal: EMA of valid MACD line values, possibly negative.
//   - Histogram: Line minus Signal, possibly negative or exactly zero.
//
// Determinism:
//   - All three fields use the same index order and the same recurrence as Stream.Push.
//
// Complexity:
//   - Storage O(n) total; Compute is O(n) time and O(n) additional space.
//
// Notes:
//   - Compute returns nil, not a partially populated Result, on any error.
//   - Result implements the reflect-free core.Nilable convention through IsNil.
//
// AI-Hints:
//   - A non-nil &Result{} is not a computed Result; IsNil means pointer presence only.
//   - An invalid signal cannot be used as a numeric zero in model features.
type Result struct {
	Line      Series // Fast minus slow EMA.
	Signal    Series // EMA of the valid MACD line.
	Histogram Series // Line minus Signal; ready iff Signal is ready.
}

// IsNil reports whether the Result pointer itself is nil.
// It implements lvlath's nilable convention without inspecting result fields.
//
// Implementation:
//   - Compare the receiver directly against nil; no reflection or dereference occurs.
//
// Returns:
//   - bool: true for a typed-nil *Result and false for any allocated *Result.
//
// Determinism:
//   - Depends solely on pointer presence, not slice contents or warmup masks.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Safe for a nil receiver; this is not a data-validity validator.
//
// AI-Hints:
//   - Never replace this with a check of Line.Values: an empty struct is still non-nil.
func (r *Result) IsNil() bool { return r == nil }

// Point is one fully indexed streaming MACD observation.
// Its booleans distinguish a completed recurrence phase from a placeholder zero.
//
// Behavior highlights:
//   - LineValid becomes true once the slow EMA has received Slow prices.
//   - SignalValid becomes true after Signal valid MACD line observations.
//   - Histogram is valid exactly when SignalValid is true; no third flag is necessary.
//
// Returns:
//   - Line, Signal, Histogram: finite float64 values on success, +0 when invalid.
//   - LineValid, SignalValid: monotonically ready after each successful warmup.
//
// Determinism:
//   - A Point produced by Push matches the corresponding Compute index bit-for-bit.
//
// Complexity:
//   - O(1) state and size per observation.
//
// Notes:
//   - A failed Push returns an empty Point{} and never advances the Stream.
//
// AI-Hints:
//   - Check SignalValid before using Histogram in scoring or crossover detection.
type Point struct {
	Line      float64 // Difference of the price EMAs after slow warmup.
	Signal    float64 // Smoothed MACD line after signal warmup.
	Histogram float64 // Signed difference Line-Signal after signal warmup.

	LineValid   bool // Slow price EMA has reached its SMA seed.
	SignalValid bool // Signal EMA has reached its SMA seed; histogram is also valid.
}

// Snapshot is a complete detached representation of a Stream's recurrence state.
// It supports deterministic continuity across process restarts with the same algorithm/version.
//
// Behavior highlights:
//   - Algorithm is the literal "macd/sma-ema/v2" for every initialized Stream.
//   - v1 checkpoints are not restorable; replay their original accepted price prefix.
//   - Config is lowercase SHA-256 hex over Algorithm and three little-endian uint64 periods.
//   - Count counts accepted closes, not attempted Push calls.
//   - FloatState has three means in [fast, slow, signal] order.
//   - IntState has three saturated seed counters in the same order.
//
// Returns:
//   - Both slices are detached on Snapshot and their contents are copied on Restore.
//
// Errors:
//   - Invalid version, configuration, lengths, phase counters or means yield ErrStateVersion
//     when the snapshot is passed to Restore.
//
// Determinism:
//   - State fields follow a fixed serialization order, independent of map iteration.
//
// Complexity:
//   - O(1) time and space to export, validate, or import; two three-element slices.
//
// Notes:
//   - Config is NOT a checksum or authentication tag for Count/FloatState/IntState.
//   - Persist snapshots through an authenticated channel when inputs may be hostile.
//
// AI-Hints:
//   - Persist all fields and the stream's declared periods; a digest alone cannot
//     reconstruct Options or establish the provenance of the underlying market data.
type Snapshot struct {
	Algorithm  string    // Algorithm/version identifier.
	Config     string    // Lowercase SHA-256 digest of version and periods.
	Count      uint64    // Number of successfully committed prices.
	FloatState []float64 // Fast, slow, signal mean/EMA, in fixed order.
	IntState   []uint64  // Saturated fast, slow, signal seed counts.
}

// Stream maintains independent fast/slow price EMA and signed signal EMA phases.
// It stores no close history and never publishes a partially committed next state.
//
// Behavior highlights:
//   - opts is a normalized configuration fixed by New until the stream is discarded.
//   - count is the total number of accepted closes; the EMA seed counters saturate.
//   - initialized rejects accidentally constructed zero-value Stream{} receivers.
//
// Determinism:
//   - Input acceptance and recurrence happen strictly in Push call order.
//
// Complexity:
//   - O(1) persistent state, O(1) time and 0 heap allocations per successful Push.
//
// Notes:
//   - Stream is owned by one goroutine; no concurrent Push, Reset, Restore or Snapshot
//     with mutation. Use one stream per independent instrument/timeframe.
//   - Snapshot produces a detached export suitable for externally synchronized replication.
//
// AI-Hints:
//   - Do not copy a live Stream as a synchronization mechanism; use Snapshot/Restore.
//   - Never introduce a full-history buffer into the per-price recurrence.
type Stream struct {
	opts        Options  // Canonical periods, never mutated by Push.
	initialized bool     // New was used; zero Stream is not a valid receiver.
	count       uint64   // Successfully committed closes, not attempted pushes.
	fast        emaState // Price EMA seeded from closes [0:Fast].
	slow        emaState // Price EMA seeded from closes [0:Slow].
	signal      emaState // Signed EMA seeded from the first Signal valid MACD lines.
}
