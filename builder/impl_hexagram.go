// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package: lvlath/builder
//
// impl_hexagram.go - Star-of-David (hexagram) patterns as deterministic builders.
//
// Purpose:
//   - Provide several hexagram variants by overlaying chord sets over a base cycle/wheel.
//   - Keep full determinism, idempotency, and respect for core flags.
//   - Weight policy: if g.Weighted() → cfg.weightFn(cfg.rng); else weight=0.
//
// Contract:
//   - Hexagram(variant) returns a Constructor (no panics, sentinel errors only).
//   - Base ring vertices are assumed to be indexed with cfg.idFn(0..n-1).
//   - For Wheel-based variants, we do not touch the hub ("Center"); chords connect ring vertices.
//   - Directed graphs: we explicitly mirror semantic undirected edges.
//
// Complexity:
//   - O(n + |chords|) time, O(1) extra space; n is the base ring size per variant.
//
// Design contract:
//  – Ring IDs follow cfg.idFn(0..n-1); wheel hub is literal "Center" (document explicitly).
//  - Idempotency: chords are added only if edge absent (both orientations for undirected).
//  - Directed mirroring: always mirror to preserve undirected semantics.
//
// AI-Hints:
//  - To add a new variant, extend hexRingSize and hexChords only; do not touch builder logic.
//  - Keep chord emission order stable to preserve determinism (affects weight RNG sequence).
//  - If you need weighted chords with a different distribution than the base ring,
//    inject a local weightFn via a dedicated option and read it in the constructor (backward-compatible).

package builder

import (
	"fmt"

	"github.com/lvlath/go/core"
)

// ----------------------------
// Context tag for error wraps.
// ----------------------------

const methodHexagram = "Hexagram" // used in error messages

// --------------------------------------------
// Variants (kept small and deterministic).
// --------------------------------------------

type HexagramVariant int

const (
	// HexDefault : classic 6-vertex hexagram (two interlocking triangles).
	HexDefault HexagramVariant = iota
	// HexMedium : 8-vertex variant with two interlocking quadrilaterals.
	HexMedium
	// HexBig : 12-vertex variant with long chords (outer triangles).
	HexBig
	// HexHuge : 12-vertex variant with outer triangles + two inner triangles.
	HexHuge
)

// ring sizes per variant (stable, tiny).
var hexRingSize = map[HexagramVariant]int{
	HexDefault: 6,
	HexMedium:  8,
	HexBig:     12,
	HexHuge:    12,
}

// chord sets per variant (stable emission order).
var hexChords = map[HexagramVariant][]chord{
	// Default: two triangles skipping one vertex each time.
	HexDefault: {
		{0, 2}, {2, 4}, {4, 0}, // triangle 0-2-4
		{1, 3}, {3, 5}, {5, 1}, // triangle 1-3-5
	},
	// Medium: two interlocking quads on ring 0..7 (ordered sensibly).
	HexMedium: {
		{0, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 0},
		{1, 2}, {2, 4}, {4, 6}, {6, 7}, {7, 0}, {0, 1},
	},
	// Big: two outer triangles over 12-vertex ring.
	HexBig: {
		{0, 1}, {1, 3}, {3, 4}, {4, 5}, {5, 7}, {7, 8}, {8, 9}, {9, 11}, {11, 0},
		{2, 3}, {3, 5}, {5, 6}, {6, 7}, {7, 9}, {9, 10}, {10, 11}, {11, 1}, {1, 2},
	},
	// Huge: Big + two inner triangles (stellation).
	HexHuge: {
		// outer
		{0, 1}, {1, 3}, {3, 4}, {4, 5}, {5, 7}, {7, 8}, {8, 9}, {9, 11}, {11, 0},
		{2, 3}, {3, 5}, {5, 6}, {6, 7}, {7, 9}, {9, 10}, {10, 11}, {11, 1}, {1, 2},
		// inner
		{1, 5}, {5, 9}, {9, 1},
		{3, 7}, {7, 11}, {11, 3},
	},
}

// ----------------------------------------
// Local hasEdge consistent with semantics.
// ----------------------------------------

func hasEdgeUndirAware(g *core.Graph, u, v string) bool {
	if g.Directed() {
		return g.HasEdge(u, v)
	}
	return g.HasEdge(u, v) || g.HasEdge(v, u)
}

// ----------------------------------------
// Public constructor.
// ----------------------------------------

// Hexagram builds a Star-of-David pattern for the given variant.
//
// Base:
//   - HexDefault/HexMedium → base ring via Cycle(n).
//   - HexBig/HexHuge       → base ring + hub via Wheel(n) (hub is "Center").
//
// Chords:
//   - Overlayed between ring vertices only (no hub involvement).
//
// Directed policy:
//   - We preserve undirected semantics by mirroring (u→v) and (v→u).
//
// Weight policy:
//   - If g.Weighted(): weight := cfg.weightFn(cfg.rng); else 0.
//
// Returns:
//
//	– GraphConstructor.
//
// Errors:
//   - builderErrorf(MethodHexagram, "unknown variant").
//   - builderErrorf(MethodHexagram, "no chords defined").
//   - core errors via builderErrorf.
//
// Complexity:
//   - O(hSize + |chords|) time, O(1) space.
func Hexagram(variant HexagramVariant) Constructor {
	return func(g *core.Graph, cfg builderConfig) error {
		// Resolve ring size.
		n, ok := hexRingSize[variant]
		if !ok || n <= 0 {
			return fmt.Errorf("%s: unknown variant %v: %w", methodHexagram, variant, ErrOptionViolation)
		}

		// Build base structure deterministically.
		switch variant {
		case HexDefault, HexMedium:
			if err := Cycle(n)(g, cfg); err != nil {
				return fmt.Errorf("%s: base cycle: %w", methodHexagram, err)
			}
		case HexBig, HexHuge:
			if err := Wheel(n)(g, cfg); err != nil {
				return fmt.Errorf("%s: base wheel: %w", methodHexagram, err)
			}
		default:
			return fmt.Errorf("%s: unhandled variant %v: %w", methodHexagram, variant, ErrConstructFailed)
		}

		// Fetch chords to overlay (stable emission order).
		set, ok := hexChords[variant]
		if !ok || len(set) == 0 {
			return fmt.Errorf("%s: missing chords for %v: %w", methodHexagram, variant, ErrConstructFailed)
		}

		var (
			w    float64 // Weight resolution once per edge (deterministic for same rng state).
			u, v string  // edges key
		)
		nextWeight := func() float64 {
			if g.Weighted() {
				return cfg.weightFn(cfg.rng)
			}
			return 0
		}

		// Emit chords with idempotency and directed mirroring.
		for _, ch := range set {
			// Derive ring endpoint IDs via cfg.idFn (consistent with Cycle/Wheel).
			u = cfg.idFn(ch.U)
			v = cfg.idFn(ch.V)

			// Skip invalid self-chords if loops are not allowed.
			if u == v && !g.Looped() {
				continue
			}

			// Add (u,v) once if not present.
			if !hasEdgeUndirAware(g, u, v) {
				w = nextWeight()
				if _, err := g.AddEdge(u, v, w); err != nil {
					return fmt.Errorf("%s: AddEdge(%s→%s,w=%g): %w", methodHexagram, u, v, w, err)
				}
			}

			// Mirror explicitly for directed graphs to preserve undirected intent.
			if g.Directed() && !hasEdgeUndirAware(g, v, u) {
				w = nextWeight()
				if _, err := g.AddEdge(v, u, w); err != nil {
					return fmt.Errorf("%s: AddEdge(%s→%s,w=%g): %w", methodHexagram, v, u, w, err)
				}
			}
		}

		return nil
	}
}
