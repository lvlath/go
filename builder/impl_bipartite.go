// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package: lvlath/builder
//
// impl_bipartite.go - implementation of CompleteBipartite(n1,n2) constructor.
//
// Contract:
//   - n1 ≥ 1 and n2 ≥ 1 (else ErrTooFewVertices).
//   - Adds left partition IDs as "{leftPrefix}{i}", i=0..n1-1.
//   - Adds right partition IDs as "{rightPrefix}{j}", j=0..n2-1.
//     (Prefixes are resolved deterministically in newBuilderConfig; empty → defaults "L"/"R".)
//   - Emits every cross-pair L_i → R_j; mirrors R_j → L_i only if g.Directed().
//   - Weight policy: if g.Weighted() then cfg.weightFn(cfg.rng) else 0.
//   - Honors core mode flags without silent degrade.
//   - Returns only sentinel errors; never panics at runtime.
//
// Complexity:
//   - Time: O(n1 + n2) vertices + O(n1*n2) edges emission.
//   - Space: O(n1 + n2) extra for ID slices.
//
// Determinism:
//   - Deterministic IDs via (prefix, index) with stable prefixes from cfg.
//   - Deterministic edge emission order: i asc over L, inner j asc over R.
//   - Deterministic weights for a fixed cfg.rng/weightFn.

package builder

import (
	"fmt"

	"github.com/lvlath/go/core"
)

// File-local constants for method tag and minima (no magic numbers).
const (
	methodCompleteBipartite = "CompleteBipartite"
	minPartitionSize        = 1
)

// CompleteBipartite returns a Constructor for the complete bipartite graph K_{n1,n2}.
func CompleteBipartite(n1, n2 int) Constructor {
	// The closure captures (n1,n2); BuildGraph supplies (g,cfg).
	return func(g *core.Graph, cfg builderConfig) error {
		// Early validation: both partitions must be non-empty.
		if n1 < minPartitionSize || n2 < minPartitionSize {
			return fmt.Errorf("%s: n1=%d, n2=%d (each must be ≥ %d): %w",
				methodCompleteBipartite, n1, n2, minPartitionSize, ErrTooFewVertices)
		}

		// Resolve partition prefixes (already defaulted by newBuilderConfig).
		lp, rp := cfg.leftPrefix, cfg.rightPrefix

		// Preallocate and fill left partition IDs in ascending index order.
		leftIDs := make([]string, n1) // O(n1) space
		for i := 0; i < n1; i++ {     // O(n1) time
			// Compose deterministic left ID as "<lp><i>" (e.g., "L0").
			id := fmt.Sprintf("%s%d", lp, i)
			leftIDs[i] = id
			// Insert vertex into the graph.
			if err := g.AddVertex(id); err != nil {
				return fmt.Errorf("%s: AddVertex(%s): %w", methodCompleteBipartite, id, err)
			}
		}

		// Preallocate and fill right partition IDs in ascending index order.
		rightIDs := make([]string, n2) // O(n2) space
		for j := 0; j < n2; j++ {      // O(n2) time
			// Compose deterministic right ID as "<rp><j>" (e.g., "R0").
			id := fmt.Sprintf("%s%d", rp, j)
			rightIDs[j] = id
			// Insert vertex into the graph.
			if err := g.AddVertex(id); err != nil {
				return fmt.Errorf("%s: AddVertex(%s): %w", methodCompleteBipartite, id, err)
			}
		}

		// Cache whether weights are observed by the core graph.
		useWeight := g.Weighted()
		var (
			i, j int     // loop iterators
			w    float64 // decide weight once per cross pair.
			u, v string  // edges key
		)

		// Emit all cross edges in stable (i over left, j over right) order.
		for i = 0; i < n1; i++ { // iterate left side first
			u = leftIDs[i]           // left endpoint ID
			for j = 0; j < n2; j++ { // then each right endpoint
				v = rightIDs[j] // right endpoint ID

				if useWeight {
					w = cfg.weightFn(cfg.rng)
				} else {
					w = 0
				}

				// Add u→v edge.
				if _, err := g.AddEdge(u, v, w); err != nil {
					return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodCompleteBipartite, u, v, w, err)
				}

				// If directed, also add v→u for full bipartite symmetry in digraph mode.
				if g.Directed() {
					if _, err := g.AddEdge(v, u, w); err != nil {
						return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodCompleteBipartite, v, u, w, err)
					}
				}
			}
		}

		// Success: K_{n1,n2} constructed deterministically.
		return nil
	}
}
