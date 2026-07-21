// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package: lvlath/builder
//
// impl_complete.go - implementation of Complete(n) constructor.
//
// Contract:
//   - n ≥ 1 (else ErrTooFewVertices).
//   - Adds vertices via cfg.idFn in ascending index order (0..n-1).
//   - Emits each unordered pair {i,j} with i<j exactly once,
//     and mirrors to j→i only if g.Directed() is true.
//   - Weight policy: if g.Weighted() then cfg.weightFn(cfg.rng) else 0.
//   - Honors core mode flags (Directed/Loops/Multigraph) without silent degrade.
//   - Returns only sentinel errors; never panics at runtime.
//
// Complexity:
//   - Time: O(n) vertices + O(n²) edges emission.
//   - Space: O(n) extra for the precomputed ID slice.
//
// Determinism:
//   - Deterministic IDs via cfg.idFn.
//   - Deterministic pair order: lexicographic by (i,j), i<j.
//   - Deterministic weights for a fixed cfg.rng/weightFn.

package builder

import (
	"fmt"

	"github.com/lvlath/go/core"
)

// File-local constants for method tagging and parameter minima (no magic numbers).
const (
	methodComplete   = "Complete"
	minCompleteNodes = 1
)

// Complete returns a Constructor that builds the complete simple graph K_n.
func Complete(n int) Constructor {
	// The returned closure captures n; BuildGraph supplies (g,cfg).
	return func(g *core.Graph, cfg builderConfig) error {
		// Early parameter validation: K_n is defined for n≥1.
		if n < minCompleteNodes {
			return fmt.Errorf("%s: n=%d < min=%d: %w", methodComplete, n, minCompleteNodes, ErrTooFewVertices)
		}

		// Preallocate and fill the vertex ID slice in deterministic index order.
		ids := make([]string, n) // O(n) space for stable reuse below
		for i := 0; i < n; i++ { // O(n) time to compute and insert vertices
			// Compute vertex ID for index i via the configured scheme.
			ids[i] = cfg.idFn(i)
			// Insert vertex into the core graph; core enforces mode invariants.
			if err := g.AddVertex(ids[i]); err != nil {
				return fmt.Errorf("%s: AddVertex(%s): %w", methodComplete, ids[i], err)
			}
		}

		// Cache whether weights are observed by the core graph for single-branch logic.
		useWeight := g.Weighted()
		var (
			i, j int     // loop iterators
			w    float64 // decide edge weight once per pair (deterministic for fixed RNG).
			u, v string  // edges key
		)

		// Emit each unordered pair {i,j} with i<j in stable lexicographic order.
		for i = 0; i < n; i++ { // outer endpoint index
			u = ids[i]                  // stable left endpoint ID
			for j = i + 1; j < n; j++ { // right endpoint index (strictly greater)
				v = ids[j] // stable right endpoint ID

				if useWeight {
					w = cfg.weightFn(cfg.rng)
				} else {
					w = 0
				}

				// Add u→v (core handles undirected/parallel/loop policies).
				if _, err := g.AddEdge(u, v, w); err != nil {
					return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodComplete, u, v, w, err)
				}

				// If the graph is directed, also add v→u for symmetry of K_n.
				if g.Directed() {
					if _, err := g.AddEdge(v, u, w); err != nil {
						return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodComplete, v, u, w, err)
					}
				}
			}
		}

		// Success: complete graph constructed deterministically.
		return nil
	}
}
