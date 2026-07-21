// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package: lvlath/builder
//
// impl_star.go - implementation of Star(n) constructor.
//
// Contract:
//   - n ≥ 2 (else ErrTooFewVertices).
//   - Adds hub vertex with fixed ID "Center" (documented design choice).
//   - Adds leaves via cfg.idFn in ascending index order for i = 1..n-1.
//   - Emits spokes in stable order Center → leaf[i]. For directed graphs,
//     also emits leaf[i] → Center to preserve spoke symmetry.
//   - Weight policy: if g.Weighted() then cfg.weightFn(cfg.rng) else 0.
//   - Honors core mode flags (Directed/Loops/Multigraph) without silent degrade.
//   - Returns only sentinel errors; never panics at runtime.
//
// Complexity:
//   - Time: O(n) vertices + O(n-1) edges (undirected) or O(2n-2) (directed).
//   - Space: O(1) extra.
//
// Determinism:
//   - Deterministic IDs via cfg.idFn and fixed hub ID.
//   - Deterministic edge emission order by increasing leaf index.
//   - Deterministic weights for fixed cfg.rng/weightFn.

package builder

import (
	"fmt"

	"github.com/lvlath/go/core"
)

// File-local constants (no magic numbers/strings; stable method tags).
const (
	methodStar   = "Star"
	minStarNodes = 2
)

// Star returns a Constructor that builds a star topology with n vertices:
// one hub "Center" and n-1 leaves.
func Star(n int) Constructor {
	// The returned closure captures n and receives (g,cfg) from BuildGraph.
	return func(g *core.Graph, cfg builderConfig) error {
		// Validate the parameter domain early to avoid partial work.
		if n < minStarNodes {
			return fmt.Errorf("%s: n=%d < min=%d: %w", methodStar, n, minStarNodes, ErrTooFewVertices)
		}

		// Insert the hub vertex with a fixed, documented ID.
		if err := g.AddVertex(centerVertexID); err != nil {
			return fmt.Errorf("%s: AddVertex(%s): %w", methodStar, centerVertexID, err)
		}

		// Precompute whether weights are observed by the graph.
		useWeight := g.Weighted()

		var (
			i      int     // loop iterators
			w      float64 // decide edge weight once per spoke.
			leafID string  // edge key
		)
		// Add leaves in deterministic order and connect spokes.
		for i = 1; i < n; i++ {
			// Compute deterministic leaf ID for index i.
			leafID = cfg.idFn(i)

			// Add leaf vertex to the graph.
			if err := g.AddVertex(leafID); err != nil {
				return fmt.Errorf("%s: AddVertex(%s): %w", methodStar, leafID, err)
			}

			if useWeight {
				w = cfg.weightFn(cfg.rng)
			} else {
				w = 0
			}

			// Add Center → leaf spoke (core decides directed/undirected semantics).
			if _, err := g.AddEdge(centerVertexID, leafID, w); err != nil {
				return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodStar, centerVertexID, leafID, w, err)
			}

			// For directed graphs, add the reverse spoke to keep symmetry explicit.
			if g.Directed() {
				if _, err := g.AddEdge(leafID, centerVertexID, w); err != nil {
					return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodStar, leafID, centerVertexID, w, err)
				}
			}
		}

		// Success: star fully constructed.
		return nil
	}
}
