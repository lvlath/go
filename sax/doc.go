// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package sax implements Symbolic Aggregate approXimation with fixed-width
// words and a Euclidean MINDIST lower bound for equal-length, equally
// segmented, compatibly transformed observation windows.
//
// Facades: Encode is the canonical one-shot constructor+transform. EncodeGaussian
// and EncodeRaw make common modes explicit; EncodeRawWithBreakpoints accepts
// calibrated physical-unit bins. MindistValues is a one-shot pairwise Euclidean
// convenience projection. Repeated calls should reuse an immutable Encoder.
//
// Domain: Raw or population Z-score window normalization, exact fractional
// PAA through github.com/lvlath/go/paa, right-bin Gaussian/custom threshold
// quantization, and immutable SHA-256 mathematical configuration identifiers.
//
// Laws: w>=1, 2<=a<=256, n>=w; finite observations and custom boundaries;
// each boundary belongs to the bin to its right; upper tail is retained.
// Zero Z variance is rejected by default or converted to a flagged zero-vector.
// Input is read-only and not retained. Encoder configuration is copied, and
// all published Result slices are caller-owned detached values. Independent
// encoder calls are concurrent-safe with immutable input and results.
// Failed Encode returns nil Result and a package-owned errors.Is sentinel;
// no partial results are published. Fixed scan order and pinned quantile bytes
// make encoding deterministic for one specified Go/PAA numeric kernel version.
// The Gaussian table is an embedded versioned binary resource, verified by
// two SHA-256 checks before its immutable cache is published.
//
// Non-goals: SAX does not prove a lower bound for arbitrary DTW, implement a
// search index, or certify recall of an approximate candidate shortlist.
// Mindist is a lower bound only on the declared transformed Euclidean metric.
//
// AI-Hints: Never use positive Euclidean MINDIST to discard a DTW candidate;
// q=[0,0,1] and c=[0,1,1] have MINDIST=0.5 with custom [0.25,0.75]
// while unconstrained DTW=0. Never trust the user-visible BreakpointID as the
// compatibility key, replace fractional PAA by integer blocks, or mutate
// private model thresholds after construction. No reflect, panic, or recover.
package sax
