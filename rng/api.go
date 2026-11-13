// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng defines the public factory, replay, and buffer-filling boundary.
// This file delegates every random draw to one version-pinned stream kernel.
package rng

// NewFactory constructs a stateless, immutable source of deterministic streams.
// It is the reusable dependency to share across independently scheduled workers.
//
// Implementation:
//   - Stage 1: select the sole pinned generator revision; no options are required.
//   - Stage 2: create an empty immutable factory without entropy acquisition.
//   - Stage 3: return the factory, which can construct arbitrarily many streams.
//
// Behavior highlights:
//   - A zero-value Factory and a nil *Factory receiver have identical stream behavior.
//
// Inputs:
//   - None: there is no implicit seed, clock input, or environment configuration.
//
// Returns:
//   - *Factory: non-nil, stateless object safe for simultaneous method calls.
//
// Errors:
//   - None: construction has no invalid input or fallible computation.
//
// Determinism:
//   - Factory creation never draws a word or changes any stream sequence.
//
// Complexity:
//   - Time O(1), Space O(1); the factory contains no mutable generator state.
//
// Notes:
//   - A Factory is reusable across goroutines; individual RNG streams are not.
//
// AI-Hints:
//   - Do not insert a package-wide default RNG or an instance cache into Factory.
//   - Persist Factory.Version(), seed, and logical path IDs with replay metadata.
func NewFactory() *Factory {
	return &Factory{}
}

// NewStream constructs one independently mutable stream without retaining a factory.
// It is a convenience wrapper for NewFactory().Stream(seed, pathID), not a new RNG.
//
// Implementation:
//   - Stage 1: accept two complete uint64 identifiers, including zero.
//   - Stage 2: delegate state derivation to the canonical Factory.Stream method.
//   - Stage 3: return the new single-owner RNG at its initial word position.
//
// Behavior highlights:
//   - Repeating both arguments returns a fresh stream with the same initial state.
//
// Inputs:
//   - seed: experiment identifier, any uint64; zero is an ordinary seed.
//   - pathID: logical trajectory or replication index, not a worker identifier.
//
// Returns:
//   - RNG: a distinct mutable stream with no cached Gaussian value.
//
// Errors:
//   - None: the entire unsigned argument domain is valid.
//
// Determinism:
//   - Exactly the state and draw sequence of (*Factory).Stream(seed, pathID).
//
// Complexity:
//   - Time O(1), Space O(1) per returned stream (one state allocation).
//
// Notes:
//   - Do not recreate this stream on every draw; hold it for the whole logical path.
//
// AI-Hints:
//   - This wrapper is for one owned stream; sharing it across goroutines is unsafe.
func NewStream(seed, pathID uint64) RNG {
	return NewFactory().Stream(seed, pathID)
}

// Stream constructs a fresh mutable PCG-DXSM generator for one logical path.
// The stream is derived only from the seed, path ID, and fixed algorithm version.
//
// Implementation:
//   - Stage 1: encode the fixed domain and two uint64 inputs in little-endian order.
//   - Stage 2: derive a 128-bit starting state using SHA-256 in deriveState.
//   - Stage 3: return a private stream with zeroed Box-Muller spare/cache state.
//
// Behavior highlights:
//   - No global PRNG, clock, pool, lock, shared counter, or worker ID participates.
//   - Two calls with identical parameters return non-aliased but replayable objects.
//
// Inputs:
//   - seed: caller-chosen unsigned experiment seed, including seed 0.
//   - pathID: caller-chosen logical sample ID, independent of goroutine scheduling.
//
// Returns:
//   - RNG: fresh mutable, single-goroutine-owned stream; it is never nil.
//
// Errors:
//   - None: all 2^64 possible values of both parameters are accepted.
//
// Determinism:
//   - SHA-256 maps domain || LE64(seed) || LE64(pathID) to state (hi, lo).
//   - The first Uint64 advances the state before applying the DXSM output map.
//
// Complexity:
//   - Time O(1) for hashing a fixed 38-byte message, Space O(1) per stream.
//
// Notes:
//   - Both nil and zero-value Factory receivers are valid; no fields are accessed.
//   - Distinct path IDs are deterministically separated, not proven non-overlapping.
//
// AI-Hints:
//   - Assign pathID from the logical trial/row number, never the worker slot.
//   - Use a stable indexed reduction for deterministic parallel float64 totals.
func (f *Factory) Stream(seed, pathID uint64) RNG {
	hi, lo := deriveState(seed, pathID)
	return &pcgStream{hi: hi, lo: lo}
}

// Version identifies the stable raw, uniform, and Gaussian draw protocol.
// Its exact string is part of the reproducible-experiment identity.
//
// Implementation:
//   - Stage 1: use the fixed local algorithm revision.
//   - Stage 2: return the canonical constant without consulting environment state.
//
// Behavior highlights:
//   - The method is nil-receiver safe and never changes during execution.
//
// Inputs:
//   - None: even a nil Factory receiver has a defined version.
//
// Returns:
//   - string: pcg-dxsm128-sha256le-u52-boxmuller-v1.
//
// Errors:
//   - None: version lookup cannot fail.
//
// Determinism:
//   - All receivers return the identical compile-time constant.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - An incompatible change to seed encoding or consumption requires a new version.
//
// AI-Hints:
//   - Never repurpose this identifier for a changed PCG or Box-Muller kernel.
func (f *Factory) Version() string {
	return algorithmVersion
}

// IsNil reports whether the Factory pointer itself is nil.
// This optional nilability method does not change nil-safe Stream/Version behavior.
//
// Implementation:
//   - Stage 1: compare the receiver pointer with nil.
//   - Stage 2: publish pointer presence without constructing a stream.
//
// Behavior highlights:
//   - A nil factory can still create streams; IsNil classifies identity, not utility.
//
// Returns:
//   - bool: true only for a nil *Factory pointer.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure receiver-identity comparison; no state or word consumption.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Structurally implements the core.Nilable convention without importing core.
//
// AI-Hints:
//   - Do not use IsNil to reject a factory: nil receivers are explicitly supported.
func (f *Factory) IsNil() bool {
	return f == nil
}

// FillUint64 publishes a fresh path's raw draws into a caller-owned buffer.
// It shortens batch initialization without adding a second generator kernel.
//
// Implementation:
//   - Stage 1: treat an empty (including nil) destination as a no-op.
//   - Stage 2: initialize exactly one Stream(seed, pathID) for nonempty output.
//   - Stage 3: overwrite dst in ascending index order with consecutive Uint64 calls.
//
// Behavior highlights:
//   - Each call restarts the stream; it does not continue a previous Fill call.
//
// Inputs:
//   - seed: any uint64 experiment identifier; zero is valid.
//   - pathID: logical trajectory identifier, not worker scheduling metadata.
//   - dst: caller-owned writable []uint64; nil and length zero are accepted.
//
// Returns:
//   - None: dst is filled in-place; no destination buffer is retained.
//
// Errors:
//   - None: all identifiers and destination lengths are accepted.
//
// Determinism:
//   - dst[i] equals the (i+1)th Uint64 from a fresh Stream(seed,pathID).
//
// Complexity:
//   - Time O(n+1), Space O(1) extra for n=len(dst); setup hashes 38 bytes.
//
// Notes:
//   - Do not mutate dst concurrently; unlike Stream, this wrapper owns no returned state.
//
// AI-Hints:
//   - For incremental/chunked generation retain one Stream and draw into successive chunks.
func (f *Factory) FillUint64(seed, pathID uint64, dst []uint64) {
	if len(dst) == 0 {
		return
	}
	stream := f.Stream(seed, pathID)
	for i := range dst {
		dst[i] = stream.Uint64()
	}
}

// FillUniformOpen fills a caller-owned buffer with strictly open uniform values.
// It is a replayable batch projection of the existing Stream.UniformOpen method.
//
// Implementation:
//   - Stage 1: return immediately for an empty or nil destination.
//   - Stage 2: create one canonical stream from seed and logical pathID.
//   - Stage 3: write one uniform draw per destination element, ascending by index.
//
// Behavior highlights:
//   - Every value is in [2^-53,1-2^-53], hence strictly inside (0,1).
//   - Exactly n raw words are consumed in a conceptual equivalent Stream traversal.
//
// Inputs:
//   - seed: any uint64, including zero.
//   - pathID: logical simulation-path identifier, not a goroutine number.
//   - dst: writable caller-owned []float64; nil and empty are accepted.
//
// Returns:
//   - None: dst is overwritten in place, with no retained alias.
//
// Errors:
//   - None: there is no invalid argument combination.
//
// Determinism:
//   - dst[i] equals draw i from fresh Stream(seed,pathID).UniformOpen().
//
// Complexity:
//   - Time O(n+1), Space O(1) extra for n=len(dst).
//
// Notes:
//   - To preserve a stream's position across batches, call UniformOpen on that stream.
//
// AI-Hints:
//   - Never use FillUniformOpen as an integer index sampler via naive modulo mapping.
func (f *Factory) FillUniformOpen(seed, pathID uint64, dst []float64) {
	if len(dst) == 0 {
		return
	}
	stream := f.Stream(seed, pathID)
	for i := range dst {
		dst[i] = stream.UniformOpen()
	}
}

// FillNormal fills a caller-owned buffer with standard-normal Box-Muller draws.
// The batch uses the exact cached-pair consumption law of Stream.Normal.
//
// Implementation:
//   - Stage 1: return for an empty (including nil) destination.
//   - Stage 2: create one private path stream with an initially empty spare cache.
//   - Stage 3: iterate dst ascending; each Normal call consumes 2 or 0 raw words.
//
// Behavior highlights:
//   - An odd destination length discards the unused spare when this temporary stream ends.
//   - For n values, the conceptual raw consumption is 2*ceil(n/2) words.
//
// Inputs:
//   - seed: uint64 experiment seed, zero included.
//   - pathID: logical path identifier; distinct from a worker ID.
//   - dst: writable caller-owned []float64, nil or empty allowed.
//
// Returns:
//   - None: consecutive Gaussian values are published in dst without retaining it.
//
// Errors:
//   - None: the standard-normal transform is finite for every raw word.
//
// Determinism:
//   - dst[i] equals the (i+1)th Normal from a fresh Stream(seed,pathID).
//
// Complexity:
//   - Time O(n+1), Space O(1) extra for n=len(dst).
//
// Notes:
//   - Callers needing interleaved raw draws or preserved spare state must retain RNG.
//
// AI-Hints:
//   - Never reset the underlying Box-Muller spare between elements of one Fill.
//   - Do not use batch calls to continue the same stream after an odd-length fill.
func (f *Factory) FillNormal(seed, pathID uint64, dst []float64) {
	if len(dst) == 0 {
		return
	}
	stream := f.Stream(seed, pathID)
	for i := range dst {
		dst[i] = stream.Normal()
	}
}

// UniformOpenFromUint64 maps one supplied raw word to a strictly open float64.
// It exposes the pure, version-pinned midpoint conversion without advancing an RNG.
//
// Implementation:
//   - Stage 1: retain the high 52 bits of raw without sign conversion.
//   - Stage 2: add half a unit and scale exactly by 2^-52.
//   - Stage 3: return a value in [2^-53,1-2^-53] with no hidden draw.
//
// Behavior highlights:
//   - Exactly matches the conversion performed by Stream.UniformOpen.
//
// Inputs:
//   - raw: any uint64 word, including zero and MaxUint64.
//
// Returns:
//   - float64: strictly inside (0,1); both endpoints are impossible.
//
// Errors:
//   - None: each raw input maps to exactly one bin midpoint.
//
// Determinism:
//   - Pure, bit-reproducible IEEE-754 binary64 operation; no RNG is consulted.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This function does not provide a second uniform distribution kernel.
//
// AI-Hints:
//   - Use to test x=0 and x=MaxUint64; never widen the extraction to 53 bits.
//   - It cannot replace cryptographic random sampling.
func UniformOpenFromUint64(raw uint64) float64 {
	return openFromRaw(raw)
}
