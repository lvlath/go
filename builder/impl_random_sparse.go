// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package: lvlath/builder
//
// impl_random_sparse.go - implementation of RandomSparse(n, p) constructor.
//
// Canonical model (approved in Ta-builder V1):
//   - Erdős–Rényi-like generator: include each admissible edge independently with prob p.
//   - Undirected: iterate unordered pairs {i,j} with i<j.
//   - Directed: iterate ordered pairs (i,j); allow self-loops iff g.Looped()==true.
//
// Contract:
//   - n ≥ 1 (else ErrTooFewVertices).
//   - 0 ≤ p ≤ 1 (else ErrInvalidProbability).
//   - cfg.rng must be non-nil (else ErrNeedRandSource). Even for p∈{0,1}, contract requires RNG.
//   - Adds vertices via cfg.idFn in ascending index order (0..n-1).
//   - Weight policy: if g.Weighted() then cfg.weightFn(cfg.rng) else 0.
//   - Honors core flags (Directed/Weighted/Loops/Multigraph) without silent degrade.
//   - Returns only sentinel errors; never panics at runtime.
//
// Complexity:
//   - Time: O(n) vertices + O(n²) Bernoulli trials / edge checks.
//   - Space: O(1) extra (no global buffers).
//
// Determinism:
//   - Stable vertex order: i asc.
//   - Stable edge-trial order: for each i asc, j asc (undirected uses j>i).
//   - Deterministic outcomes for fixed seed/options due to fixed trial order.

package builder

import (
	"fmt"

	"github.com/lvlath/go/core"
)

// File-local constants (no magic literals; stable method tag and domains).
const (
	methodRandomSparse      = "RandomSparse"
	minRandomSparseVertices = 1
	probMin                 = 0.0
	probMax                 = 1.0
)

// RandomSparse returns a Constructor that samples an Erdős–Rényi-like graph
// over n vertices with independent edge probability p.
func RandomSparse(n int, p float64) Constructor {
	// The returned closure captures (n, p); BuildGraph supplies (g, cfg).
	return func(g *core.Graph, cfg builderConfig) error {
		// 1) Validate parameters early (fail fast, zero side-effects on invalid input).

		// Validate vertex count domain: n must be at least 1.
		if n < minRandomSparseVertices {
			return fmt.Errorf("%s: n=%d < min=%d: %w",
				methodRandomSparse, n, minRandomSparseVertices, ErrTooFewVertices)
		}

		// Validate probability: must lie in the closed interval [0,1].
		if p < probMin || p > probMax {
			return fmt.Errorf("%s: p=%.6f not in [%.1f,%.1f]: %w",
				methodRandomSparse, p, probMin, probMax, ErrInvalidProbability)
		}

		// RNG is only required when 0 < p < 1 (true stochastic sampling).
		if cfg.rng == nil && p > 0.0 && p < 1.0 {
			return fmt.Errorf("%s: rng is required: %w", methodRandomSparse, ErrNeedRandSource)
		}

		// 2) Add all vertices deterministically via cfg.idFn (IDs 0..n-1).
		for i := 0; i < n; i++ {
			idFunc := cfg.idFn(i)          // compute deterministic vertex ID
			if err := g.AddVertex(idFunc); // insert vertex; core enforces mode rules
			err != nil {
				return fmt.Errorf("%s: AddVertex(%s): %w", methodRandomSparse, idFunc, err)
			}
		}

		// 3) Cache mode flags for single-branch logic and dereference RNG once.
		useWeight := g.Weighted() // whether weights are observed by the core graph
		rng := cfg.rng            // local alias for RNG (nil already rejected)
		loops := g.Looped()       // whether self-loops are allowed
		directed := g.Directed()  // directed vs. undirected logic branch

		var (
			i, j int     // loop iterators
			w    float64 // decide weight once per cross pair.
			u, v string  // edges key
		)
		// 4) Sample edges per graph directedness with a stable, documented order.

		if directed {
			// Directed case: consider all ordered pairs (i,j).
			for i = 0; i < n; i++ { // stable outer loop: i asc
				u = cfg.idFn(i)         // left endpoint ID
				for j = 0; j < n; j++ { // inner loop: j asc
					// Disallow self-loops unless explicitly permitted by mode flags.
					if i == j && !loops {
						continue
					}
					// Bernoulli trial: include edge with probability p.

					if rng == nil {
						// Deterministic edge set for p ∈ {0,1}
						if p == 1.0 {
							v = cfg.idFn(j)
							if useWeight {
								w = cfg.weightFn(nil)
							} else {
								w = 0
							}
							if _, err := g.AddEdge(u, v, w); err != nil {
								return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodRandomSparse, u, v, w, err)
							}
						}
						continue
					}
					//if rng.Float64() <= p {

					if rng.Float64() <= p {
						v = cfg.idFn(j) // right endpoint ID

						if useWeight {
							w = cfg.weightFn(rng)
						} else {
							w = 0
						}

						// Add directed edge u→v; core handles multigraph/parallel policies.
						if _, err := g.AddEdge(u, v, w); err != nil {
							return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w",
								methodRandomSparse, u, v, w, err)
						}
					}
				}
			}
		} else {
			// Undirected case: consider unordered pairs {i,j} with i<j (no duplicates).
			for i = 0; i < n; i++ { // stable i asc
				u = cfg.idFn(i)             // left endpoint ID
				for j = i + 1; j < n; j++ { // j strictly greater than i
					// Bernoulli trial: include edge with probability p.

					if rng == nil {
						if p == 1.0 {
							v = cfg.idFn(j)
							if useWeight {
								w = cfg.weightFn(nil)
							} else {
								w = 0
							}
							if _, err := g.AddEdge(u, v, w); err != nil {
								return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodRandomSparse, u, v, w, err)
							}
						}
						continue
					}
					if rng.Float64() <= p {

						//}
						//if rng.Float64() <= p {
						v = cfg.idFn(j) // right endpoint ID

						if useWeight {
							w = cfg.weightFn(rng)
						} else {
							w = 0
						}

						// Add edge u—v; core interprets it per undirected semantics.
						if _, err := g.AddEdge(u, v, w); err != nil {
							return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w",
								methodRandomSparse, u, v, w, err)
						}
					}
				}
			}
		}

		// 5) Success: random sparse graph sampled deterministically for a fixed seed.
		return nil
	}
}
