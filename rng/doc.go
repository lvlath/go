// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng provides version-pinned pseudo-random streams for reproducible
// numerical experiments, Monte Carlo simulation, and dependent-data resampling.
// It is the stochastic infrastructure layer, not a cryptographic entropy source.
//
// Domain / contract:
//   - RNG is the stable three-method mutable protocol: Uint64, UniformOpen, Normal.
//   - Factory is stateless and immutable; zero and nil pointer receivers are usable.
//   - Each Stream(seed,pathID) returns a new, privately owned mutable instance.
//   - NewStream is a shorthand; FillUint64, FillUniformOpen, and FillNormal fill
//     caller-owned buffers from one fresh stream in ascending index order.
//   - UniformOpenFromUint64 is the pure raw-word conversion used for exact endpoint
//     fixtures; it does not consume a stream or maintain any hidden state.
//
// Mathematical laws:
//   - State S=(hi<<64)|lo evolves as (S*M+C) mod 2^128, with M/C fixed in pcg.go.
//   - DXSM mixes the NEW state, never the state before the recurrence.
//   - SHA-256 input is the 22-byte ASCII domain "lvlath/rng/pcg-dxsm-v1", followed
//     by LE64(seed) then LE64(pathID); digest[0:8] -> hi and [8:16] -> lo as LE64.
//   - UniformOpen projects one raw word using (float64(x>>12)+0.5)*2^-52, whose
//     exact range is [2^-53,1-2^-53] and is strictly inside (0,1).
//   - Normal returns sqrt(-2 ln u1)*cos(2*pi*u2), caching the corresponding sine
//     member. A miss consumes two raw words; a hit consumes zero.
//   - Uint64 and UniformOpen never invalidate the pending sine spare.
//
// Version / determinism:
//   - Factory.Version() == "pcg-dxsm128-sha256le-u52-boxmuller-v1".
//   - Seed 0 is valid, as is every uint64 pathID including MaxUint64.
//   - Logical path IDs identify simulations/replicates, never physical workers.
//   - Bitwise raw-word replay is guaranteed for the pinned kernel/seed encoding.
//   - Floating transcendental operations may differ in their last bits across
//     platforms; no universal cross-platform bitwise Normal parity is claimed.
//   - An exact parallel aggregate also requires deterministic index-ordered
//     floating-point reduction; scheduling-independent paths alone are not enough.
//   - Hash separation is not proof of statistically independent/disjoint streams.
//
// Ownership / concurrency / nilability:
//   - Factory and nil/zero Factory receivers are safe for concurrent methods.
//   - A stream is single-goroutine-owned; concurrent mutable calls are unsupported.
//   - No locks, shared streams, global PRNG state, wall-clock seeds or pools exist.
//   - Factory.IsNil reports pointer presence; a nil Factory still works.
//   - The returned private stream supports optional IsNil structurally without
//     enlarging RNG, preserving compatibility with custom RNG implementations.
//   - Batch outputs belong to callers and are not retained. Nil/empty buffers
//     are accepted and left untouched; batch wrappers restart from the path seed.
//   - The package does not return Result artifacts or expose partial results.
//
// Numeric/error laws:
//   - All unsigned inputs are valid; no Options, context, or sentinel errors
//     are needed for the fixed stream surface.
//   - The factory-produced UniformOpen and Normal values are finite.
//   - Custom RNG interface implementations may violate these numeric laws;
//     consumers must validate external data at their own mathematical boundary.
//
// Complexity:
//   - Stream construction: O(1) time/space, one fixed-length SHA-256 digest.
//   - Uint64/UniformOpen: O(1) time, O(1) mutable generator state, 0 hot-path
//     allocations; Normal: amortized O(1) time with one cached scalar.
//   - Fill* over n cells: O(n+1) time, O(1) auxiliary space, writes in order.
//
// Non-goals:
//   - Cryptographically secure randomness, arbitrary stream jumping, stream
//     checkpointing, model-specific distributions, sampling policy, statistical
//     validation, automatic worker allocation, and implicit consumer migration.
//   - A snapshot API, if introduced later, must be separately versioned and
//     include BOTH spare and hasSpare to reproduce future normal draws.
//
// AI-Hints:
//   - Never use time.Now, global math/rand, worker IDs, or shared *rand.Rand.
//   - Never change the PCG/DXSM recurrence or the 52-bit midpoint mapping without
//     issuing a new Version and independent golden-vector tests.
//   - Do not insert Result/Options solely to mimic estimator packages: RNG is a
//     stateful stream protocol, not a failed/successful numerical fit artifact.
//   - Never use modulo sampling for arbitrary bounded integers without proving
//     the bound divides 2^64 or implementing unbiased rejection separately.
//   - For continuing batches, retain one RNG instead of calling Fill* repeatedly.
package rng
