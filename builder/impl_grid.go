// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package: lvlath/builder
//
// impl_grid.go - implementation of Grid(rows, cols) constructor.
//
// Canonical model (approved in Ta-builder V1):
//   - 2D orthogonal grid with 4-neighborhood (right & bottom neighbors per cell).
//   - Vertex IDs use a fixed, documented scheme "r,c" (row-major order).
//     This is a deliberate exception to cfg.idFn to keep coordinates explicit.
//
// Contract:
//   - rows ≥ 1 and cols ≥ 1 (else ErrTooFewVertices).
//   - Adds vertices in row-major order with IDs "r,c" for r∈[0..rows-1], c∈[0..cols-1].
//   - Adds edges to right (r,c+1) and bottom (r+1,c) neighbors where they exist.
//     In directed graphs, also emits the reverse arc for symmetry.
//   - Weight policy: if g.Weighted() then cfg.weightFn(cfg.rng) else 0.
//   - Honors core mode flags (Directed/Loops/Multigraph) without silent degrade.
//   - Returns only sentinel errors; never panics at runtime.
//
// Complexity:
//   - Time: O(rows*cols) vertices + O(rows*cols) edges emission (linear in grid size).
//   - Space: O(1) extra (IDs are composed on the fly; no full grid storage).
//
// Determinism:
//   - Stable vertex order: row-major (r asc, then c asc).
//   - Stable edge order: for each (r,c) emit Right then Bottom if present.
//   - Deterministic weights for a fixed cfg.rng/weightFn.

package builder

import (
	"fmt"

	"github.com/lvlath/go/core"
)

// File-local constants: method tag, minima, and ID format (no magic literals).
const (
	methodGrid = "Grid"
	minGridDim = 1
	gridIDFmt  = "%d,%d" // "r,c" - fixed, documented coordinate ID scheme
)

// Grid returns a Constructor that builds a rows×cols orthogonal grid.
func Grid(rows, cols int) Constructor {
	// The returned closure captures (rows, cols); BuildGraph supplies (g, cfg).
	return func(g *core.Graph, cfg builderConfig) error {
		// 1) Validate parameters early (fail fast; no partial work).
		if rows < minGridDim || cols < minGridDim {
			return fmt.Errorf("%s: rows=%d, cols=%d (each must be ≥ %d): %w",
				methodGrid, rows, cols, minGridDim, ErrTooFewVertices)
		}

		// 2) Add all vertices in deterministic row-major order with IDs "r,c".
		for r := 0; r < rows; r++ { // iterate rows ascending
			for c := 0; c < cols; c++ { // iterate cols ascending
				// Compose the coordinate ID; fixed scheme by design.
				id := fmt.Sprintf(gridIDFmt, r, c)
				// Insert vertex; core enforces mode invariants.
				if err := g.AddVertex(id); err != nil {
					return fmt.Errorf("%s: AddVertex(%s): %w", methodGrid, id, err)
				}
			}
		}

		// 3) Prepare weight observation flag once (single-branch logic).
		useWeight := g.Weighted()

		var (
			r, c int
			w    float64 // decide weight once per edge; deterministic for fixed rng.
			u, v string  // edges key
		)
		// 4) Emit edges: for each (r,c), connect to Right and Bottom neighbors if they exist.
		for r = 0; r < rows; r++ {
			for c = 0; c < cols; c++ {
				u = fmt.Sprintf(gridIDFmt, r, c) // current cell ID

				// 4a) Right neighbor (r, c+1).
				if c+1 < cols {
					v = fmt.Sprintf(gridIDFmt, r, c+1) // right neighbor ID

					if useWeight {
						w = cfg.weightFn(cfg.rng)
					} else {
						w = 0
					}

					// Add u→v; core handles undirected/parallel constraints.
					if _, err := g.AddEdge(u, v, w); err != nil {
						return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodGrid, u, v, w, err)
					}
					// Mirror for directed graphs to preserve symmetric neighborhood.
					if g.Directed() {
						if _, err := g.AddEdge(v, u, w); err != nil {
							return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodGrid, v, u, w, err)
						}
					}
				}

				// 4b) Bottom neighbor (r+1, c).
				if r+1 < rows {
					v = fmt.Sprintf(gridIDFmt, r+1, c) // bottom neighbor ID

					if useWeight {
						w = cfg.weightFn(cfg.rng)
					} else {
						w = 0
					}

					// Add u→v.
					if _, err := g.AddEdge(u, v, w); err != nil {
						return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodGrid, u, v, w, err)
					}
					// Mirror for directed graphs.
					if g.Directed() {
						if _, err := g.AddEdge(v, u, w); err != nil {
							return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodGrid, v, u, w, err)
						}
					}
				}
			}
		}

		// Success: grid constructed deterministically with stable IDs and order.
		return nil
	}
}
