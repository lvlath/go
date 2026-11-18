// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion aggregates the conditionally Gaussian log-jump sum.
// This file owns the one-Normal conditional sampling rule and scratch ownership.
package jumpdiffusion

import (
	"math/big"

	"github.com/lvlath/go/rng"
)

// aggregateJump generates the total log jump conditional on a sampled Poisson count.
// The sum of count independent Normal(m,s^2) jumps is Normal(count*m,count*s^2).
//
// Implementation:
//   - Stage 1: clear destination and return without draws for a zero count.
//   - Stage 2: accumulate count*m at wide precision; skip Normal when std=0.
//   - Stage 3: draw one standard Normal and add sqrt(count)*std*Z when std>0.
//
// Behavior highlights:
//   - Exactly one jump Normal is used for any positive count with positive std.
//   - Even at count=2^53-1, count is converted exactly to a float64 integer.
//
// Inputs:
//   - count: already range-checked integer in [0,2^53-1].
//   - random: caller-owned RNG; scratch values have distinct owned big.Float storage.
//   - dst/countWide/scale/z/mean/std: mutable workspace valid for this step only.
//
// Returns: nil on success; dst contains the aggregated log jump.
// Errors: ErrNonFinite when the external RNG returns a nonfinite Normal.
// Determinism: zero or one jump Normal, always after Poisson count generation.
// Complexity: Time O(1), scratch O(1) at fixed working precision.
// Notes: this aggregates observations, not individual event times or jump marks.
// AI-Hints: do not draw count independent Normals or round the sum through float64.
func aggregateJump(count uint64, random rng.RNG,
	dst, countWide, scale, z, mean, std *big.Float,
) error {
	// Clearing the scratch buffer also guarantees that N=0 contributes a
	// true zero increment without touching either Normal stream or old data.
	dst.SetFloat64(0)
	if count == 0 {
		return nil
	}

	// Count was admitted only below 2^53, so float64(count) is exact.
	// Multiplying in the wide domain avoids a finite-input overflow here.
	countWide.SetFloat64(float64(count))
	dst.Mul(countWide, mean)
	// Deterministic jump marks require no Gaussian entropy, regardless
	// of how many individual Poisson events occurred inside this step.
	if std.Sign() == 0 {
		return nil
	}

	sample := random.Normal()
	if !finite(sample) {
		return ErrNonFinite
	}
	// Conditional Gaussian closure replaces N independent samples by one
	// Normal with standard deviation sqrt(N)*s, at O(1) draw cost.
	z.SetFloat64(sample)
	scale.Sqrt(countWide)
	scale.Mul(scale, std)
	scale.Mul(scale, z)
	dst.Add(dst, scale)
	return nil
}
