// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap classifies invalid requests and bounded-resource failures.
// These package-owned sentinels are stable identities for errors.Is inspection.
package bootstrap

import "errors"

// ErrInvalidOption classifies invalid bootstrap policy or context admission.
// It prevents unsupported block modes and illegal dimensions from reaching an
// allocator or random stream, without changing the caller's plan.
//
// Implementation:
//   - Stage 1: Check nil context before inspecting the Plan.
//   - Stage 2: Check Kind, strictly positive dimensions and BlockLength<=rows.
//   - Stage 3: Return this package-owned sentinel before resource admission.
//
// Behavior highlights:
//   - An already-canceled non-nil context takes precedence over plan errors.
//
// Errors:
//   - errors.Is(err, ErrInvalidOption) for nil context or an invalid Plan.
//
// Determinism:
//   - Validation order is fixed; no RNG state is touched on failure.
//
// Complexity:
//   - Constant-time sentinel identity; validation needs O(1) time and space.
//
// Notes:
//   - Error text is a diagnostic, never the caller's classification protocol.
//
// AI-Hints:
//   - Handle errors.Is, not string matching; do not silently default dimensions.
var ErrInvalidOption = errors.New("bootstrap: invalid option")

// ErrOverflow classifies a native-integer representation failure in sampling.
// A requested matrix whose logical byte size cannot be expressed as a Go int
// must be rejected before allocation, regardless of the configured budget.
//
// Implementation:
//   - Stage 1: Check Length*sizeof(int) and header additions for overflow.
//   - Stage 2: Check Replicates*bytesPerReplicate without multiplying first.
//   - Stage 3: Reject unrepresentable sizes before comparing with MaxResultBytes.
//
// Behavior highlights:
//   - Source index bounds already fit native int by construction; overflow
//     classification concerns the checked output allocation dimensions.
//
// Errors:
//   - errors.Is(err, ErrOverflow) identifies an unrepresentable integer result.
//
// Determinism:
//   - Native architecture fixes int size; overflow precedes budget refusal.
//
// Complexity:
//   - Constant-time sentinel identity; checked size arithmetic is O(1).
//
// AI-Hints:
//   - Never multiply untrusted dimensions before checking their native limits.
var ErrOverflow = errors.New("bootstrap: result size overflow")

// ErrBudget classifies a representable result larger than the index budget.
// The 256-MiB ceiling includes logical index words and outer slice headers,
// but is not an upper bound on total process resident memory.
//
// Implementation:
//   - Stage 1: Validate all dimensions and the native byte arithmetic.
//   - Stage 2: Compare the representable logical byte total with MaxResultBytes.
//   - Stage 3: Reject before allocating the matrix or constructing any RNG stream.
//
// Errors:
//   - errors.Is(err, ErrBudget) when requested logical output exceeds the cap.
//
// Determinism:
//   - Fixed 256-MiB admission limit, independent of seed and random draws.
//
// Complexity:
//   - Constant-time sentinel identity; budget admission is O(1).
//
// AI-Hints:
//   - Reduce Length or Replicates explicitly. Repeating the same Seed in
//     separate calls restarts path IDs at zero; there is no chunk-offset API.
var ErrBudget = errors.New("bootstrap: result exceeds memory budget")
