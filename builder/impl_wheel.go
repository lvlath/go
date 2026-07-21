// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package: lvlath/builder
//
// impl_wheel.go - implementation of Wheel(n) constructor.
//
// Canonical definition:
//   - Wₙ = Cₙ + "Center": a cycle of size n plus a hub vertex.
//   - Therefore, n ≥ 3.
// Contract:
//   - n ≥ 4 (else ErrTooFewVertices).
//   - Builds the outer cycle using Cycle(n) with the same cfg semantics.
//   - Adds hub vertex with fixed ID "Center".
//   - Emits spokes from "Center" to each cycle vertex in index order.
//     For directed graphs, also emits the reverse arc for symmetry.
//   - Weight policy: if g.Weighted() then cfg.weightFn(cfg.rng) else 0.
//   - Honors core mode flags without silent degrade.
//   - Returns only sentinel errors; never panics at runtime.
//
// Complexity:
//   - Time: O(n) vertices + O(n + (n)) edges ≈ O(n) (undirected),
//            or O(n) vertices + O(n + 2(n)) edges for directed (still O(n)).
//   - Space: O(1) extra.
//
// Determinism:
//   - Deterministic cycle IDs via cfg.idFn (0..n-2) and fixed hub ID.
//   - Deterministic spoke emission order by increasing ring index.
//   - Deterministic weights for fixed cfg.rng/weightFn.

package builder

import (
	"fmt"

	"github.com/lvlath/go/core"
)

// File-local constants for method tag and minima.
const (
	methodWheel   = "Wheel"
	minWheelNodes = 3 // because outer cycle has size (n) which must be ≥ 3
)

// Wheel returns a Constructor that builds a wheel Wₙ = Cₙ₋₁ + "Center".
func Wheel(n int) Constructor {
	// The closure captures n and uses the same option-resolved cfg as Cycle.
	return func(g *core.Graph, cfg builderConfig) error {
		// Early validation (no work on invalid input).
		if n < minWheelNodes {
			return fmt.Errorf("%s: n=%d < min=%d: %w", methodWheel, n, minWheelNodes, ErrTooFewVertices)
		}

		// Build the outer cycle of size (n) using the same (g,cfg).
		// Note: Cycle uses cfg.idFn(i) for i=0..n-2, matching our spoke iteration below.
		if err := Cycle(n)(g, cfg); err != nil {
			// Surface Cycle's sentinel with our method context.
			return fmt.Errorf("%s: base cycle C_%d: %w", methodWheel, n, err)
		}

		// Add the hub vertex with a fixed, documented ID.
		if err := g.AddVertex(centerVertexID); err != nil {
			return fmt.Errorf("%s: AddVertex(%s): %w", methodWheel, centerVertexID, err)
		}

		// Precompute whether the graph is weighted for single-branch logic.
		useWeight := g.Weighted()

		var (
			i     int     // loop iterators
			w     float64 // decide spoke weight once; determinism hinges on cfg.rng seed.
			rimID string  // edges key
		)
		// Connect spokes between the hub and each cycle vertex in stable order.
		for i = 0; i < n; i++ {
			// Deterministic cycle vertex ID from the same idFn domain.
			rimID = cfg.idFn(i)

			if useWeight {
				w = cfg.weightFn(cfg.rng)
			} else {
				w = 0
			}

			// Add Center → rim spoke.
			if _, err := g.AddEdge(centerVertexID, rimID, w); err != nil {
				return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodWheel, centerVertexID, rimID, w, err)
			}

			// For directed graphs, also add rim → Center for symmetric spokes.
			if g.Directed() {
				if _, err := g.AddEdge(rimID, centerVertexID, w); err != nil {
					return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodWheel, rimID, centerVertexID, w, err)
				}
			}
		}

		// Success: wheel fully constructed following the canonical Wₙ definition.
		return nil
	}
}
