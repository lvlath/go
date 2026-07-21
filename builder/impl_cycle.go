// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package: lvlath/builder
//
// impl_cycle.go - implementation of Cycle(n) constructor.
//
// Contract:
//   - n ≥ 3 (else ErrTooFewVertices).
//   - Adds vertices via cfg.idFn in ascending index order (0..n-1).
//   - Emits edges in stable order i -> (i+1)%n for i=0..n-1.
//   - Weight policy: if g.Weighted() then cfg.weightFn(cfg.rng) else 0.
//   - Honors core mode flags (Directed/Loops/Multigraph) without silent degrade.
//   - Returns only sentinel errors; never panics at runtime.
//
// Complexity:
//   - Time: O(n) vertices + O(n) edges.
//   - Space: O(1) extra (iter vars only).
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

// File-local constants (no magic numbers; stable method tags for context).
const (
	methodCycle   = "Cycle"
	minCycleNodes = 3
)

// Cycle returns a Constructor that builds an n-vertex simple cycle C_n.
func Cycle(n int) Constructor {
	// Return a closure capturing n; BuildGraph will pass (g,cfg).
	return func(g *core.Graph, cfg builderConfig) error {
		// Validate parameter domain early (fail fast, no work on invalid input).
		if n < minCycleNodes {
			// Provide deterministic context while preserving sentinel semantics.
			return fmt.Errorf("%s: n=%d < min=%d: %w", methodCycle, n, minCycleNodes, ErrTooFewVertices)
		}

		// Add n vertices with deterministic IDs produced by cfg.idFn.
		for i := 0; i < n; i++ {
			// Compute vertex ID for index i.
			id := cfg.idFn(i)
			// Insert vertex into the core graph (core enforces mode invariants).
			if err := g.AddVertex(id); err != nil {
				// Wrap the core error with method context and return.
				return fmt.Errorf("%s: AddVertex(%s): %w", methodCycle, id, err)
			}
		}

		// Precompute whether weights are observed by the core graph.
		useWeight := g.Weighted()

		var (
			i        int     // loop iterator
			w        float64 // choose edge weight based on graph weighting policy.
			uID, vID string  // edges key
		)
		// Emit edges in ascending i; for i==n-1, connect to 0 to close the ring.
		for i = 0; i < n; i++ {
			// Compute ordered pair (u,v) for the ring step.
			uID = cfg.idFn(i)
			vID = cfg.idFn((i + 1) % n)

			if useWeight {
				// Call configured generator; determinism depends on rng seed.
				w = cfg.weightFn(cfg.rng)
			} else {
				// Unweighted policy → zero weight (ignored by core).
				w = 0
			}

			// Add the ring edge; core handles directed/undirected per its flags.
			if _, err := g.AddEdge(uID, vID, w); err != nil {
				// Wrap and return immediately on first failure (no partial cleanup).
				return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodCycle, uID, vID, w, err)
			}
		}

		// Success: cycle fully constructed.
		return nil
	}
}
