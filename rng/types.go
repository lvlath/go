// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng defines stream ownership and the immutable factory's public types.
// This file records which state is mutable and which interfaces remain ABI-stable.
package rng

// RNG is the minimal, versioned protocol of one mutable pseudo-random stream.
// Each successful Factory.Stream call returns a new instance owned by one goroutine.
//
// Implementation:
//   - Stage 1: Uint64 advances the 128-bit PCG state and mixes the NEW state.
//   - Stage 2: UniformOpen consumes one Uint64 and maps its top 52 bits to (0,1).
//   - Stage 3: Normal consumes two open uniforms per uncached Box-Muller pair.
//
// Behavior highlights:
//   - Box-Muller retains the sine variate until the next Normal call.
//   - Uint64/UniformOpen calls do not invalidate an existing Gaussian spare.
//
// Inputs:
//   - No call-time arguments: the seed/pathID are fixed when constructing the stream.
//
// Returns:
//   - Each method returns a numeric draw and advances only its own stream state.
//
// Errors:
//   - No method returns errors; factory-produced streams accept the full raw domain.
//
// Determinism:
//   - The exact method-call order and original seed/pathID determine the sequence.
//
// Complexity:
//   - O(1) time and O(1) memory per draw; Normal is amortized O(1).
//
// Notes:
//   - Not safe for concurrent calls on one instance, even when methods differ.
//   - Deliberately contains only three methods for custom implementation compatibility.
//   - External RNG implementations are not certified to meet numeric guarantees.
//
// AI-Hints:
//   - Do not add IsNil, Uint64N or batch methods to this interface: doing so breaks
//     existing user-defined RNG providers and changes the accepted public contract.
//   - Allocate a stream by logical path, not by physical worker, and do not share it.
type RNG interface {
	// Uint64 advances the PCG recurrence once and returns one raw 64-bit word.
	// It never invalidates the pending Gaussian spare.
	Uint64() uint64

	// UniformOpen consumes exactly one raw word and returns a strict (0,1) value.
	// Its midpoint projection uses 52 high bits, not the unsafe 53-bit variant.
	UniformOpen() float64

	// Normal returns one standard-normal Box-Muller sample.
	// Empty cache consumes two raw words; the next cached call consumes zero.
	Normal() float64
}

// Factory is a stateless, immutable owner of the stable stream creation policy.
// The zero value is usable; calls on a nil *Factory receiver are also supported.
//
// Implementation:
//   - Stage 1: accept any seed/pathID and fixed algorithm revision.
//   - Stage 2: derive fresh private state on every Stream call.
//   - Stage 3: expose the state only behind the RNG protocol.
//
// Behavior highlights:
//   - Concurrent Factory method calls are safe because there are no mutable fields.
//
// Inputs:
//   - Stream receives the complete uint64 domain for both identifiers.
//
// Returns:
//   - Stream returns distinct mutable objects; Version returns immutable metadata.
//
// Errors:
//   - None; Factory has no invalid configuration state.
//
// Determinism:
//   - Identical parameters and version yield identical draw sequences.
//
// Complexity:
//   - Time O(1), Space O(1) for the factory itself.
//
// Notes:
//   - IsNil reports pointer presence and does not imply that nil receivers are unusable.
//
// AI-Hints:
//   - Do not attach a generator, mutex, or worker counter to Factory.
type Factory struct{}

// pcgStream stores exactly one mutable PCG state and one pending Gaussian pair half.
// It is private to prevent callers from reseeding, serializing, or aliasing the kernel.
//
// Fields:
//   - hi/lo: unsigned 128-bit state, updated before every DXSM output.
//   - spare: sine half of the most recently generated Box-Muller pair.
//   - hasSpare: indicates whether spare is available for a zero-draw Normal call.
//
// Concurrency:
//   - One goroutine owns this state. No synchronization is performed.
//
// Complexity:
//   - O(1) mutable state, independent of number of draws.
type pcgStream struct {
	hi, lo   uint64
	spare    float64
	hasSpare bool
}

// IsNil classifies the underlying stream pointer when queried through a nilability interface.
// It does not enlarge the minimal RNG interface or expose a resettably mutable state.
//
// Determinism:
//   - Receiver-only pointer comparison, without raw draws or cache modifications.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - A Factory never returns a nil stream, but an optional nilability assertion is safe.
func (p *pcgStream) IsNil() bool {
	return p == nil
}

var _ RNG = (*pcgStream)(nil)
var _ interface{ IsNil() bool } = (*pcgStream)(nil)
var _ interface{ IsNil() bool } = (*Factory)(nil)
