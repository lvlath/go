// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package: lvlath/builder
//
// impl_platonic.go - implementation of PlatonicSolid(name, withCenter) constructor.
//
// Canonical model (Ta-builder V1):
//   - Build one of the five Platonic solids using a canonical, deterministic edge set.
//   - Optionally add a central hub "Center" with spokes to all shell vertices.
//
// Contract:
//   - name ∈ {Tetrahedron, Cube, Octahedron, Dodecahedron, Icosahedron}.
//   - Unknown name → ErrOptionViolation (invalid parameter).
//   - Adds vertices via cfg.idFn in ascending index order (0..n-1).
//   - Emits shell edges in stable order (pre-sorted in variants_platonic.go).
//   - For directed graphs, mirrors each shell edge to preserve symmetry.
//   - If withCenter == true, adds fixed hub ID "Center" and spokes (mirrored if directed).
//   - Weight policy: if g.Weighted() then cfg.weightFn(cfg.rng) else 0.
//   - Returns only sentinel errors; never panics at runtime.
//
// Complexity:
//   - Time: O(V+E) for the selected solid (constants: V≤20, E≤30).
//   - Space: O(1) extra beyond small loop locals.
//
// Determinism:
//   - Vertex IDs are deterministic via cfg.idFn.
//   - Edge emission order is deterministic (pre-sorted (u,v) pairs).
//   - Spoke order uses ascending vertex indices.

package builder

import (
	"fmt"

	"github.com/lvlath/go/core"
)

// File-local constants (no magic strings; stable method tag and hub ID).
const (
	methodPlatonicSolid = "PlatonicSolid" // context tag for error wrapping
	//centerVertexID      = "Center"        // fixed hub ID (explicit design choice)
)

// PlatonicSolid returns a Constructor that builds the chosen Platonic shell,
// optionally stellated with a central hub connected by spokes.
//
// Parameters:
//   - name: platonic solid identifier (see variants_platonic.go).
//   - withCenter: whether to add a hub "Center" with spokes to all shell vertices.
//
// Behavior:
//   - Honors core flags: Directed() → mirror edges; Weighted() → observe cfg.weightFn;
//     Looped()/Multigraph() are irrelevant for these canonical shells (no loops/parallel edges).
func PlatonicSolid(name PlatonicName, withCenter bool) Constructor {
	// Capture (name, withCenter); BuildGraph supplies (g, cfg).
	return func(g *core.Graph, cfg builderConfig) error {
		// 1) Lookup canonical vertex count for the selected solid (O(1) map lookup).
		n, ok := platonicVertexCounts[name] // how many base vertices to allocate
		if !ok {
			// Unknown variant → surface as option violation (invalid parameter domain).
			return fmt.Errorf("%s: unknown solid %q: %w", methodPlatonicSolid, name, ErrOptionViolation)
		}

		// 2) Add all shell vertices using the deterministic ID scheme (0..n-1).
		for i := 0; i < n; i++ { // indices in ascending order → stable vertex order
			id := cfg.idFn(i)                       // compute deterministic vertex ID
			if err := g.AddVertex(id); err != nil { // delegate mode invariants to core
				return fmt.Errorf("%s: AddVertex(%s): %w", methodPlatonicSolid, id, err)
			}
		}

		// 3) Cache mode/weight flags once for single-branch logic in tight loops.
		useWeight := g.Weighted() // whether weights are observed by the core graph
		directed := g.Directed()  // whether to mirror edges explicitly

		var (
			w        float64 // decide weight exactly once per realized edge (deterministic per rng state).
			uID, vID string  // edges key
		)
		// 4) Fetch the pre-sorted canonical shell edges and emit them deterministically.
		edges, ok := platonicEdgeSets[name] // O(1) retrieval (predefined at init)
		if !ok {
			// Defensive guard: dataset missing (should never happen given counts).
			return fmt.Errorf("%s: missing edge set for %q: %w", methodPlatonicSolid, name, ErrConstructFailed)
		}
		for _, ch := range edges { // deterministic iteration order guaranteed by pre-sort
			uID = cfg.idFn(ch.U) // left endpoint ID via cfg.idFn
			vID = cfg.idFn(ch.V) // right endpoint ID via cfg.idFn

			if useWeight {
				w = cfg.weightFn(cfg.rng)
			} else {
				w = 0
			}

			// Add shell edge u—v; core interprets directedness/multigraph policy.
			if _, err := g.AddEdge(uID, vID, w); err != nil {
				return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodPlatonicSolid, uID, vID, w, err)
			}
			// Mirror for directed graphs to preserve shell symmetry explicitly.
			if directed {
				if _, err := g.AddEdge(vID, uID, w); err != nil {
					return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodPlatonicSolid, vID, uID, w, err)
				}
			}
		}

		// 5) Optional stellation: add hub "Center" and connect spokes in index order.
		if withCenter {
			// Add the central hub vertex with a fixed, documented ID (deterministic).
			if err := g.AddVertex(centerVertexID); err != nil {
				return fmt.Errorf("%s: AddVertex(%s): %w", methodPlatonicSolid, centerVertexID, err)
			}
			// Connect hub to every shell vertex idFn(i) for i=0..n-1 (stable order).
			for i := 0; i < n; i++ {
				vID = cfg.idFn(i) // shell vertex ID

				if useWeight {
					w = cfg.weightFn(cfg.rng)
				} else {
					w = 0
				}

				// Add Center → vID spoke (core handles semantics).
				if _, err := g.AddEdge(centerVertexID, vID, w); err != nil {
					return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodPlatonicSolid, centerVertexID, vID, w, err)
				}
				// Mirror for directed graphs (explicit symmetric spokes).
				if directed {
					if _, err := g.AddEdge(vID, centerVertexID, w); err != nil {
						return fmt.Errorf("%s: AddEdge(%s→%s, w=%g): %w", methodPlatonicSolid, vID, centerVertexID, w, err)
					}
				}
			}
		}

		// 6) Success: canonical Platonic shell (+ optional hub) constructed deterministically.
		return nil
	}
}
