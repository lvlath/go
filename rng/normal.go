// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng maps open uniforms to normal variates using a cached Box-Muller pair.
// This file pins both cosine-first return order and the persistent sine spare.
package rng

import "math"

const (
	// boxMullerRadiusScale is the -2 multiplier in the radial logarithm transform.
	boxMullerRadiusScale = -2.0

	// boxMullerAngleScale is the full-circle multiplier for the second uniform.
	boxMullerAngleScale = 2 * math.Pi
)

// Normal draws a standard-normal variate using the cosine-first Box-Muller map.
// The second (sine) variate remains cached across intervening raw/uniform calls.
//
// Implementation:
//   - Stage 1: when hasSpare is true, clear the flag and return spare with zero draws.
//   - Stage 2: otherwise draw two uniforms and compute radius and full-circle angle.
//   - Stage 3: cache radius*sin(angle), set hasSpare, return radius*cos(angle).
//
// Behavior highlights:
//   - Cache miss consumes exactly two raw words; cache hit consumes exactly zero.
//   - Both Gaussian values are finite because the radius input is in (0,1).
//
// Inputs:
//   - None: two consecutive open uniforms are drawn only on a cache miss.
//
// Returns:
//   - float64: one finite standard-normal pseudorandom variate.
//
// Errors:
//   - None: factory-produced streams never supply zero or one for a logarithm.
//
// Determinism:
//   - Cosine is returned first; sine is cached. Raw/uniform interleaving preserves it.
//   - Cross-platform last-bit parity of transcendental functions is not promised.
//
// Complexity:
//   - Time O(1) per call, amortized one uniform per normal, Space O(1).
//
// Notes:
//   - A stream must not be called concurrently by two goroutines.
//   - This is Gaussian sampling for numerical experiments, not cryptography.
//
// AI-Hints:
//   - Never invalidate hasSpare from Uint64 or UniformOpen.
//   - Do not replace this transform with Ziggurat without changing Version.
func (p *pcgStream) Normal() float64 {
	// Serving the cached second variate must not touch the PCG state.
	if p.hasSpare {
		p.hasSpare = false
		return p.spare
	}

	// The radial logarithm remains finite because UniformOpen excludes both endpoints.
	u1 := p.UniformOpen()
	u2 := p.UniformOpen()
	radius := math.Sqrt(boxMullerRadiusScale * math.Log(u1))
	angle := boxMullerAngleScale * u2

	// Save the correlated pair member before returning cosine; subsequent calls
	// to raw/uniform draws intentionally leave these cache fields untouched.
	p.spare = radius * math.Sin(angle)
	p.hasSpare = true
	return radius * math.Cos(angle)
}
