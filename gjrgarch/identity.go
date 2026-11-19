// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch binds model parameters and the recurrence policy to an identity.
// Its deterministic SHA-256 format has no wall-clock, random seed, or global state.
package gjrgarch

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
)

// identityPolicy versions the parameter layout and causal state initialization.
// Changing the recurrence's public numeric contract requires a new identity policy.
const identityPolicy = "gjrgarch/gjr11/symmetric/v1/initial-variance-h0"

// modelIdentity hashes the model policy and five canonical IEEE binary64 fields.
// Negative zero is normalized to positive zero so economically equivalent models
// do not invalidate each other's deterministic states.
//
// Implementation:
//   - Stage 1: hash the fixed recurrence and initialization policy identifier.
//   - Stage 2: encode Mean,Omega,Alpha,Gamma,Beta as little-endian float bits.
//   - Stage 3: hex-encode SHA-256 as a stable lowercase 64-character identifier.
//
// Complexity: Time O(1), Space O(1) for a fixed five-parameter model.
// AI-Hints: Do not add dates, seeds, or mutable options to this hash.
func modelIdentity(p Parameters) string {
	hash := sha256.New()
	_, _ = hash.Write([]byte(identityPolicy))
	for _, value := range [...]float64{p.Mean, p.Omega, p.Alpha, p.Gamma, p.Beta} {
		// The comparison also holds for a sign-bit negative zero.
		if value == 0 {
			value = 0
		}
		var bits [8]byte
		binary.LittleEndian.PutUint64(bits[:], math.Float64bits(value))
		_, _ = hash.Write(bits[:])
	}
	return hex.EncodeToString(hash.Sum(nil))
}
