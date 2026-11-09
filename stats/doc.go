// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats provides compensated, scaled Welford/Chan moments and immutable
// fitted standardization over the public matrix.Matrix abstraction.
//
// Counts and observed extrema remain exact. A bounded fixed-point binary sum
// preserves every binary contribution for the final mean, while compensated Welford/Chan
// M2 uses an extended exponent domain and rounds during its arithmetic. Means
// are rounded once when published; variance is an approximate central moment
// with final float64 materialization. Update/Merge use candidate-state commits
// and never publish a partial change after an error.
//
// Standardizer columns are fitted only from their training matrix; immutable
// model parameters support concurrent read-only inference with detached output.
// Error categories are stable package-owned sentinels classified by errors.Is.
// Summary, ScaleFitResult, and Standardizer implement core.Nilable; their
// IsNil methods report pointer absence rather than fitted or sample validity.
//
// Non-goals: rolling removal, quantiles, implicit missing-value imputation,
// online inference refitting, and snapshot isolation for concurrently edited
// input matrices. Accumulators themselves are mutable and not concurrent-safe.
//
// AI-Hints:
//   - Do not replace central moments with E[x*x]-E[x]*E[x].
//   - Do not detect constants by comparing rounded variance to an epsilon.
//   - Reuse a single fitted model on heldout data to prevent data leakage.
package stats
