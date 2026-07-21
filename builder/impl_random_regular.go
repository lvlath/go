// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package: lvlath/builder
//
// impl_random_regular.go - implementation of RandomRegular(n, d) constructor.
//
// Canonical model (Ta-builder V1):
//   - Undirected d-regular simple graph via stub-matching with bounded retries.
//   - Pairs stubs after a deterministic shuffle (per seed). Validates a pairing
//     against graph-mode constraints (no loops if !Looped, no multiedges if !Multigraph)
//     before mutating the graph; on invalid pairing, reshuffles up to a small limit.
//
// Contract:
//   - Only UNDIRECTED graphs are supported (else ErrUnsupportedGraphMode).
//   - n ≥ 1; 0 ≤ d < n; (n*d) must be even (else ErrTooFewVertices).
//   - cfg.rng must be non-nil (else ErrNeedRandSource).
//   - Adds vertices via cfg.idFn in ascending index order (0..n-1).
//   - Weight policy: if g.Weighted() then cfg.weightFn(cfg.rng) else 0.
//   - Returns only sentinel errors; never panics at runtime.
//
// Complexity:
//   - Per attempt ~O(n*d) time to validate + apply; O(n*d) temporary space for stubs.
//   - Attempts are constant-bounded → overall expected ~O(n*d).
//
// Determinism:
//   - Fixed attempt limit and fixed trial order → identical outcomes for same seed.
//   - Either a valid realization is produced, or ErrConstructFailed after N attempts.

package builder

import (
	"fmt" // For contextual %w wrapping.

	"github.com/lvlath/go/core"
)

// File-local constants (no magic numbers/strings; stable method tags).
const (
	methodRandomRegular     = "RandomRegular"
	minRRVertices           = 1
	maxStubMatchingAttempts = 3 // bounded retries; keep small and documented
)

// RandomRegular returns a Constructor that builds an undirected d-regular graph
// using the classic stub-matching (pairing) strategy with bounded retries.
func RandomRegular(n, d int) Constructor {
	// The closure captures (n, d); BuildGraph supplies (g, cfg).
	return func(g *core.Graph, cfg builderConfig) error {
		// 1) Mode gate: only UNDIRECTED graphs are supported by this constructor.
		if g.Directed() {
			return fmt.Errorf("%s: only undirected graphs are supported: %w",
				methodRandomRegular, ErrUnsupportedGraphMode)
		}

		// 2) Parameter validation (fail fast; zero side-effects on invalid input).
		//    Domain: n≥1, 0≤d<n, parity: (n*d) even.
		if n < minRRVertices {
			return fmt.Errorf("%s: n=%d < min=%d: %w",
				methodRandomRegular, n, minRRVertices, ErrTooFewVertices)
		}
		if d < 0 || d >= n {
			return fmt.Errorf("%s: degree must be in [0,%d), got %d: %w",
				methodRandomRegular, n, d, ErrTooFewVertices)
		}
		if (n*d)%2 != 0 {
			return fmt.Errorf("%s: n*d must be even (n=%d, d=%d): %w",
				methodRandomRegular, n, d, ErrTooFewVertices)
		}

		// 3) RNG is mandatory for stub shuffling (determinism + stochasticity).
		if cfg.rng == nil {
			return fmt.Errorf("%s: rng is required: %w", methodRandomRegular, ErrNeedRandSource)
		}

		// 4) Add all vertices deterministically via cfg.idFn (IDs 0..n-1).
		for i := 0; i < n; i++ {
			id := cfg.idFn(i) // compute vertex ID for index i
			if err := g.AddVertex(id); err != nil {
				return fmt.Errorf("%s: AddVertex(%s): %w", methodRandomRegular, id, err)
			}
		}

		// 5) Prepare stub list of length n*d (each vertex index i repeated d times).
		stubCount := n * d              // total number of stubs (must be even by parity check)
		stubs := make([]int, stubCount) // O(n*d) temporary array
		if stubCount == 0 {             // trivial case d=0 → isolated vertices only
			return nil // success (nothing to connect)
		}
		// Fill the stubs deterministically (i asc, each repeated d times).
		for i, pos := 0, 0; i < n; i++ { // i: vertex index
			for k := 0; k < d; k++ { // repeat d times
				stubs[pos] = i // place vertex index i
				pos++
			}
		}

		// 6) Cache mode flags (single-branch logic) and weight policy once.
		useWeight := g.Weighted()    // whether weights matter
		allowLoops := g.Looped()     // allow u==v if true
		allowMulti := g.Multigraph() // allow multiple edges between same pair if true
		rng := cfg.rng               // local alias (already validated non-nil)

		var (
			attempt, i int     // loop iterators
			w          float64 // decide weight once per cross pair.
			uIdx, vIdx int     // endpoint indexes
		)

		// 7) Attempt bounded reshuffles until we get a valid pairing or give up.
		for attempt = 1; attempt <= maxStubMatchingAttempts; attempt++ {
			// 7.1) Shuffle stubs in-place using the provided RNG (deterministic per seed).
			rng.Shuffle(stubCount, func(i, j int) { stubs[i], stubs[j] = stubs[j], stubs[i] })

			// 7.2) Validate the pairing WITHOUT mutating the graph.
			//      We check every consecutive pair (stubs[2k], stubs[2k+1]).
			valid := true                // optimistic; falsified on first violation
			var seen map[[2]int]struct{} // lazily allocate if !allowMulti
			if !allowMulti {             // track existing simple pairs when multiedges are forbidden
				seen = make(map[[2]int]struct{}, stubCount/2) // capacity hint
			}
			for i = 0; i < stubCount; i += 2 {
				uIdx = stubs[i]   // left endpoint index
				vIdx = stubs[i+1] // right endpoint index

				// Disallow self-loops if loops are not allowed by mode.
				if !allowLoops && uIdx == vIdx {
					valid = false
					break
				}

				// Disallow duplicate pairs if multiedges are not allowed.
				if !allowMulti {
					// Normalize unordered pair key (min,max) for undirected simple graph.
					if uIdx > vIdx {
						uIdx, vIdx = vIdx, uIdx
					}
					key := [2]int{uIdx, vIdx}
					if _, dup := seen[key]; dup {
						valid = false
						break
					}
					seen[key] = struct{}{}
				}
			}

			// 7.3) If invalid, retry with another shuffle (bounded by max attempts).
			if !valid {
				continue // next attempt
			}

			// 7.4) Pairing is valid under current mode constraints → apply edges.
			for i = 0; i < stubCount; i += 2 {
				uIdx = stubs[i]     // left endpoint index
				vIdx = stubs[i+1]   // right endpoint index
				u := cfg.idFn(uIdx) // map to vertex ID via cfg.idFn
				v := cfg.idFn(vIdx) // map to vertex ID via cfg.idFn

				if useWeight {
					w = cfg.weightFn(rng)
				} else {
					w = 0
				}

				// Add edge u—v (undirected semantics handled by core).
				if _, err := g.AddEdge(u, v, w); err != nil {
					return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w",
						methodRandomRegular, u, v, w, err)
				}
			}

			// 7.5) Success on this attempt; return the constructed graph.
			return nil
		}

		// 8) All attempts failed to satisfy mode constraints → construction failure.
		return fmt.Errorf("%s: failed to construct after %d attempts: %w",
			methodRandomRegular, maxStubMatchingAttempts, ErrConstructFailed)
	}
}
