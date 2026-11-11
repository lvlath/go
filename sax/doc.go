// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sax converts finite, uniformly sampled scalar windows into fixed-
// length Symbolic Aggregate approXimation (SAX) words and computes their
// compatible Euclidean MINDIST lower-bound dissimilarity. This package makes
// fractional aggregation, normalization, quantization, comparison identity,
// and returned memory ownership explicit rather than implicit search policy.
//
// Domain and mathematical contract:
//   - A source window contains n nonempty, finite float64 observations at
//     equally spaced sample positions. No timestamps, missing-value masks,
//     interpolation, or irregular sampling are inferred or reconstructed.
//   - The Encoder partitions the whole source step function into w equal-
//     duration intervals [j*n/w,(j+1)*n/w). It delegates their fractional,
//     overlap-weighted averages to paa.Transform. No remainder is discarded.
//   - Each average is mapped to one of a bins (2 <= a <= 256). Bin zero is
//     (-Inf,b[0]); intermediate bins are [b[i-1],b[i]); the final bin is
//     [b[a-2],+Inf). A value equal to a breakpoint enters the RIGHT bin.
//   - A Symbol is an index in [0,a), stored in uint16. Every successful word
//     contains exactly w symbols in ascending segment order.
//   - The distance of equal or neighboring bins is zero. Separated bins use
//     the gap between their nearest endpoints. For compatible words, Mindist
//     returns sqrt(n/w * sum(gap(symbolLeft[j],symbolRight[j])^2)).
//   - In exact real arithmetic this is bounded above by the Euclidean
//     distance between the represented PAA step functions, which is itself
//     bounded above by the Euclidean distance between the corresponding
//     TRANSFORMED source windows. It is not a DTW lower bound.
//
// Normalization and degenerate windows:
//   - ZPopulation fits population mean and standard deviation to EACH supplied
//     window independently, using divisor n. A constant window produces
//     ErrZeroVariance under RejectConstant (the default). ZeroWord instead
//     publishes zero normalized coordinates, original Mean, StdDev=0, and
//     Degenerate=true; its symbols belong to the bin containing numeric zero,
//     which is not necessarily symbol zero.
//   - Raw preserves input coordinates and reports Mean=0, StdDev=1, and
//     Degenerate=false, including for constant input. Raw custom breakpoints
//     should be expressed in the same physical units as the observations.
//   - Nonconstant data whose fitted numeric statistics cannot be represented
//     are rejected with ErrOverflow; they are never relabeled as constants.
//   - With ZPopulation, MINDIST bounds Euclidean distance between separately
//     window-normalized sequences, NOT necessarily original-amplitude distance.
//
// Public API and configuration:
//   - DefaultOptions(w) chooses alphabet eight, ZPopulation, RejectConstant,
//     the pinned Gaussian quantiles, and ReturnPAA=false. A zero Options value
//     is invalid: dimensions are explicit and do not receive silent defaults.
//   - NewEncoder freezes the caller-supplied configuration, including a
//     detached copy of any custom breakpoints. Encode(values,opts) is the
//     canonical one-shot facade; Encoder.Encode reuses the immutable model.
//   - EncodeGaussian and EncodeRaw select the common Gaussian modes.
//     EncodeRawWithBreakpoints uses user-calibrated thresholds and a required
//     audit label. MindistValues is a one-shot pairwise convenience wrapper.
//     All wrappers delegate to the same Encoder kernels without changing math.
//   - Gaussian thresholds for every alphabet in [2,256] are pinned as an
//     integrity-checked embedded binary64 table, not regenerated at runtime.
//
// Compatibility, identity and mutable results:
//   - EncodingID is a lowercase hex SHA-256 digest of a versioned, canonical
//     binary configuration: w and a as fixed-width little-endian integers,
//     normalization and constant-policy enum bytes, normalization/table
//     versions as applicable, and all breakpoint binary64 bits (signed zero
//     canonicalized to positive zero). BreakpointID is audit-only; ReturnPAA
//     controls diagnostics only. Neither field changes EncodingID.
//   - OriginalLength is NOT part of EncodingID, but Mindist separately requires
//     matching source lengths, w, a and EncodingID against both results and
//     the receiver Encoder. It also validates symbol bounds and all exposed
//     shape/fit metadata; optional PAA values are checked when present.
//   - Result is caller-owned and publicly mutable. EncodingID proves matching
//     configuration claims, NOT authenticity or provenance of a manually
//     constructed or subsequently modified Result. IsNil checks pointer
//     presence only, satisfying core.Nilable without reflection.
//
// Error, ownership and concurrency laws:
//   - Errors are classified using errors.Is: ErrInvalidOption, ErrNilInput,
//     ErrEmptyInput, ErrShape, ErrNonFinite, ErrZeroVariance, ErrIncompatible,
//     and ErrOverflow. Options/receiver validation precedes source nil,
//     empty/shape, nonfinite, domain/variance and arithmetic checks as defined
//     by the individual facades. No partial Result is returned on failure;
//     Mindist returns zero alongside its error.
//   - Inputs are read-only during each call and never retained. Symbols and
//     optional transformed PAA slices are detached from the input, the model,
//     and other results. The immutable Encoder supports concurrent Encode and
//     Mindist calls when callers do not mutate source slices or Result fields
//     concurrently with readers. There is no automatic snapshot isolation.
//   - Per-window execution scans observations in ascending index order,
//     computes segment means in increasing segment index, and uses binary
//     search for the first breakpoint greater than an average. Identical
//     inputs, explicit policies, implementation versions, and toolchain
//     numeric behavior produce deterministic published words and error classes.
//   - Encode executes in O(n+w*log(a)) time, with O(n+w) transient space in
//     ZPopulation and O(w) workspace/output in Raw (plus the model O(a)).
//     NewEncoder costs O(a) after one process-wide Gaussian table decode.
//     Mindist is O(w) if PAA is absent and O(w*log(a)) if diagnostic PAA must
//     be revalidated; its additional workspace is O(1).
//
// Numerical proof boundary:
//   - The lower-bound inequalities follow from interval separation and
//     Jensen's inequality in exact arithmetic. Floating-point rounding in
//     normalization, paa.Transform and Mindist is NOT a directed-rounding
//     certificate; callers building a mathematically certified Euclidean
//     pruning engine need a justified numerical error bound as well.
//   - MINDIST is not generally a metric on SAX words: distinct adjacent bins
//     can have zero distance, and triangle inequality can fail. A zero bound
//     does not establish identical original windows.
//
// Non-goals:
//   - No search index, sliding-window scheduler, learned quantile registry,
//     bag-of-patterns model, DTW path, exact nearest-neighbor retrieval, or
//     inference on unknown timestamp quality is provided by this package.
//     series can supply validated windows; stats owns other fitted-statistics
//     models; matrix/core own matrix and graph artifacts; dtw owns warping.
//   - Euclidean MINDIST MUST NOT be used to reject arbitrary DTW candidates.
//     For q=[0,0,1], c=[0,1,1], w=3 and custom [0.25,0.75], MINDIST=0.5
//     while unrestricted DTW can align matching samples with zero cost.
//     Using SAX to shortlist DTW candidates is approximate global retrieval;
//     evaluate Recall@K against an exhaustive DTW baseline for each workload.
//
// AI-Hints:
//   - Never replace fractional PAA with floor(n/w), confuse BreakpointID with
//     EncodingID, drop upper-tail bins, or interpret ZeroWord as symbol index 0.
//   - Never transfer a bound proven for Euclidean distance to DTW merely
//     because both compare time-series windows. A positive SAX lower bound
//     cannot certify DTW rejection; a small SAX score does not prove a match.
//   - Reuse one Encoder for many same-policy windows to avoid repeated hashing
//     and threshold copies. For practical threshold-based monitoring prefer
//     Raw with calibrated physical-unit breakpoints, and preserve the source
//     window length, cadence provenance and immutable result snapshot upstream.
package sax
