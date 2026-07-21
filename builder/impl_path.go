// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package: lvlath/builder
//
// impl_path.go - implementation of Path(n) constructor.
//
// Contract:
//   - n ≥ 2 (else ErrTooFewVertices).
//   - Adds vertices via cfg.idFn in ascending index order (0..n-1).
//   - Emits edges (i-1) -> i for i=1..n-1 in stable increasing order.
//   - Weight policy: if g.Weighted() then cfg.weightFn(cfg.rng) else 0.
//   - Honors core mode flags (Directed/Loops/Multigraph) without silent degrade.
//   - Returns only sentinel errors; never panics at runtime.
//
// Complexity:
//   - Time: O(n) vertices + O(n-1) edges.
//   - Space: O(1) extra.
//
// Determinism:
//   - Deterministic IDs via cfg.idFn.
//   - Deterministic edge emission order by increasing i.
//   - Deterministic weights given fixed cfg.rng/weightFn.

package builder

import (
	"fmt"

	"github.com/lvlath/go/core"
)

// File-local constants for method tagging and parameter minima.
const (
	methodPath   = "Path"
	minPathNodes = 2
)

// Path returns a Constructor that builds a simple path P_n.
func Path(n int) Constructor {
	// Return a closure capturing n; BuildGraph supplies (g,cfg).
	return func(g *core.Graph, cfg builderConfig) error {
		// Validate parameter domain early.
		if n < minPathNodes {
			// Preserve sentinel semantics with deterministic context message.
			return fmt.Errorf("%s: n=%d < min=%d: %w", methodPath, n, minPathNodes, ErrTooFewVertices)
		}

		// Add n vertices with deterministic IDs produced by cfg.idFn.
		for i := 0; i < n; i++ {
			// Compute vertex ID for index i.
			id := cfg.idFn(i)
			// Insert vertex into the core graph.
			if err := g.AddVertex(id); err != nil {
				// Wrap and return on the first insertion failure.
				return fmt.Errorf("%s: AddVertex(%s): %w", methodPath, id, err)
			}
		}

		// Precompute whether weights are observed by the core graph.
		useWeight := g.Weighted()

		var (
			i        int     // loop iterator
			w        float64 // choose edge weight based on graph weighting policy..
			uID, vID string  // edges key
		)
		// Emit path edges from 0->1->2->...->(n-1) in stable order.
		for i = 1; i < n; i++ {
			// Determine endpoints for the current segment.
			uID = cfg.idFn(i - 1)
			vID = cfg.idFn(i)

			if useWeight {
				// Deterministic given cfg.rng seed.
				w = cfg.weightFn(cfg.rng)
			} else {
				// Unweighted policy → zero weight.
				w = 0
			}

			// Add the path edge; core handles directedness.
			if _, err := g.AddEdge(uID, vID, w); err != nil {
				// Wrap context and surface the error.
				return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodPath, uID, vID, w, err)
			}
		}

		// Success: path fully constructed.
		return nil
	}
}
