// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 katalvlaran

// Package series validates integer-nanosecond time grids and provides causal,
// length-preserving transformations of finite float64 observations.
//
// A series is an ordered array, not a calendar or an interpolated signal.
// Validate checks strictly increasing timestamps; a positive cadence also
// requires exact integer multiples of that cadence. Missing grid slots are
// rejected or reported, never filled. Lag, Difference, SimpleReturns, and
// LogReturns consume only observations at or before each output position.
// WindowRanges enumerates complete half-open index intervals.
//
// Input slices must not be mutated concurrently with a call. Every returned
// slice is detached and caller-owned. Calls share no mutable package state.
// Public functions return a nil result on error; no partial results are
// published. Finite inputs are mandatory, and unrepresentable floating-point
// results are errors rather than infinity sentinels.
//
// Options are explicit: a zero GridOptions cadence checks order only, without
// cadence inference. Nil and non-nil empty inputs are distinct. Validation
// checks options before inputs, then shape, finiteness, domain, and arithmetic.
//
// All algorithms run in linear time in the input or output length. Grid gap
// storage is O(number of gaps); transformations allocate O(n) output memory;
// window enumeration allocates O(number of complete windows).
//
// Non-goals include resampling, interpolation, trading calendars, rolling
// statistics, statistical models, graph integration, and streaming state.
package series
