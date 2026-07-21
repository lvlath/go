// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package: lvlath/builder
//
// impl_letters.go - builders for letters, words, digits, and numbers.
//
// Purpose:
//   - Build glyph skeletons (letters A..Z/a..z, digits 0..9) using the canonical
//     datasets from letters_spec.go (data-only registry).
//   - Preserve determinism and idempotency: we check HasVertex/HasEdge before
//     any addition to avoid duplicates.
//   - Respect core flags: Directed() → mirror edges; Weighted() → default
//     constant weight 1; Looped()/Multigraph() → enforced when emitting edges.
//
// Contract:
//   - Public entry points return Constructor (same pattern as other builders).
//   - Vertex IDs come from the canonical spec. For multi-glyph inputs that
//     contain repeated glyphs (e.g., "AA"), you have two options:
//       – Pass an explicit non-empty `scope` to namespace IDs like
//         "<scope>::<idx>::<CanonicalID>" (collision-free).
//       – Or keep `scope == ""` (pure canonical IDs) but avoid repeated glyphs,
//         otherwise an ErrOptionViolation is returned.
//   - For non-loop-capable graphs (g.Looped()==false) self-loop edges from the
//     dataset are safely skipped (no error).
//   - Weight policy: if g.Weighted() => weight=1 (constant); else weight=0.
//   - Edge idempotency: we do not add an edge if it already exists
//     (both orientations are checked for undirected graphs).
//
// Complexity:
//   - Per glyph: O(V+E). Word/number of length k: O(k*(V+E)).
//   - Only tiny constants (5×7 grid skeletons).
//
// Determinism:
//   - Emission strictly follows the stable order encoded in letterSpec.Edges.
//   - Vertex-add order uses letterSpec.IDs (pre-sorted in letters_spec.go).
//
// AI-Hints:
//   - If you need decimal dot or extra symbols, extend letters_spec.go with a
//     data-only entry (do not hardcode shapes here).
//   - Keep `scope` stable in tests to ensure reproducible vertex namespaces.
//
// Notes:
//   - This file intentionally contains only building logic; all glyph geometry
//     (IDs/edges) stays in letters_spec.go to keep responsibilities clean.

package builder

import (
	"fmt"
	"strconv"
	"unicode"

	"github.com/lvlath/go/core"
)

// -----------------------------------------------------------------------------
// File-local constants (no magic strings)
// -----------------------------------------------------------------------------

const (
	methodBuildLetters = "BuildLetters" // context tag for error wrapping
	methodBuildDigit   = "BuildDigit"   // context tag for error wrapping
	methodBuildNumber  = "BuildNumber"  // context tag for error wrapping

	nsSep      = "::" // stable namespace separator when scope is used
	weightOne  = 1.0
	weightZero = 0.0
)

// -----------------------------------------------------------------------------
// Public constructors
// -----------------------------------------------------------------------------

// Letters returns a Constructor that builds all glyphs from `text`.
// Scope rules:
//   - scope == ""  → use pure canonical IDs from the spec; repeated glyphs are
//     forbidden (to avoid ID collisions) → ErrOptionViolation.
//   - scope != ""  → each glyph is namespaced as "<scope>::<pos>::<CanonicalID>".
//
// Idempotency:
//   - We never re-add existing vertices/edges (HasVertex/HasEdge checks).
func Letters(text string, scope string) Constructor {
	return func(g *core.Graph, _ builderConfig) error {
		// Guard: empty input is not useful → option violation.
		if text == "" {
			return fmt.Errorf("%s: empty input text: %w", methodBuildLetters, ErrOptionViolation)
		}

		// Check for repeated glyphs when scope is empty (canonical IDs would collide).
		if scope == "" {
			seen := make(map[rune]bool, len(text))
			for _, r := range text {
				if r == ' ' { // Allow whitespace to be ignored (no geometry).
					continue
				}
				if seen[r] {
					return fmt.Errorf("%s: repeated glyph %q with empty scope would collide: %w",
						methodBuildLetters, r, ErrOptionViolation)
				}
				seen[r] = true
			}
		}

		// Determine fixed edge weight according to weight policy.
		var w float64
		if g.Weighted() { // Observe weight when the graph supports weights.
			w = weightOne // Default constant weight = 1 (as required).
		} else {
			w = weightZero // Unweighted graph → force zero weights.
		}

		// Emit every rune in order; each becomes an independent component.
		// Complexity: O(len(text) * (V+E)).
		pos := 0 // positional index for stable namespacing
		for _, r := range text {
			// Skip whitespace runes gracefully (no-ops).
			if unicode.IsSpace(r) {
				pos++
				continue
			}
			// Resolve canonical spec from the registry (letters, digits).
			spec, found := resolveSpec(r)
			if !found {
				return fmt.Errorf("%s: unknown glyph %q: %w", methodBuildLetters, r, ErrOptionViolation)
			}

			// Compute the namespace prefix once for this glyph.
			nsPrefix := makeNamespace(scope, pos)

			// Emit vertices and edges with idempotency and mode guards.
			if err := emitGlyph(g, spec, nsPrefix, w); err != nil {
				return fmt.Errorf("%s: glyph %q at pos %d: %w", methodBuildLetters, r, pos, err)
			}
			pos++
		}
		return nil
	}
}

// Word is a thin alias for BuildLetters to match the requested naming.
func Word(word string, scope string) Constructor {
	return Letters(word, scope)
}

// Digit returns a Constructor that builds a single decimal digit (0..9).
// Scope rules are identical to BuildLetters (see above).
func Digit(digit int, scope string) Constructor {
	return func(g *core.Graph, _ builderConfig) error {
		// Validate digit domain.
		if digit < 0 || digit > 9 {
			return fmt.Errorf("%s: digit out of range [0..9]: %d: %w", methodBuildDigit, digit, ErrOptionViolation)
		}
		// Resolve spec using the 'rune' key.
		r := rune('0' + digit)
		spec, found := numberSpec[r]
		if !found {
			// Defensive: should never happen if numberSpec is complete.
			return fmt.Errorf("%s: missing spec for %q: %w", methodBuildDigit, r, ErrConstructFailed)
		}

		// Weight policy.
		var w float64
		if g.Weighted() {
			w = weightOne
		} else {
			w = weightZero
		}

		// If scope is empty, canonical IDs are used; safe for a single glyph.
		nsPrefix := makeNamespace(scope, 0)
		if err := emitGlyph(g, spec, nsPrefix, w); err != nil {
			return fmt.Errorf("%s: %q: %w", methodBuildDigit, r, err)
		}
		return nil
	}
}

// Number returns a Constructor that builds the digits composing `number`.
// Behavior:
//   - If decimal == false → only the integer part is built (no sign).
//   - If decimal == true  → both integer and fractional digits are built;
//     non-digit chars (e.g., '.', '-') are ignored unless you add glyph specs.
//   - Scope rules are identical to BuildLetters. If scope == "" and a digit
//     repeats (common for numbers), we return ErrOptionViolation.
//
// Note:
//   - If you want to visualize the decimal point itself, add a data-only entry
//     for '.' into letters_spec.go (then it will be picked automatically).
func Number(number float64, decimal bool, scope string) Constructor {
	return func(g *core.Graph, _ builderConfig) error {
		// Format the number as a string once.
		var s string
		if decimal {
			// Use a stable fixed formatting that preserves decimals (no exponent).
			s = strconv.FormatFloat(number, 'f', -1, 64)
		} else {
			// Only integer part (truncate toward zero).
			s = strconv.FormatInt(int64(number), 10)
		}
		if s == "" {
			return fmt.Errorf("%s: empty formatted number: %w", methodBuildNumber, ErrOptionViolation)
		}

		// If scope is empty, detect repeated digits to avoid collisions.
		if scope == "" {
			seen := make(map[rune]bool, len(s))
			for _, r := range s {
				if r < '0' || r > '9' { // ignore non-digit symbols ('.', '-', etc.)
					continue
				}
				if seen[r] {
					return fmt.Errorf("%s: repeated digit %q with empty scope would collide: %w",
						methodBuildNumber, r, ErrOptionViolation)
				}
				seen[r] = true
			}
		}

		// Weight policy.
		var w float64
		if g.Weighted() {
			w = weightOne
		} else {
			w = weightZero
		}

		// Emit only digits; other runes are ignored unless the spec registry supplies them.
		pos := 0
		for _, r := range s {
			if r < '0' || r > '9' {
				pos++
				continue
			}
			spec, ok := numberSpec[r]
			if !ok {
				return fmt.Errorf("%s: missing spec for digit %q: %w", methodBuildNumber, r, ErrConstructFailed)
			}
			nsPrefix := makeNamespace(scope, pos)
			if err := emitGlyph(g, spec, nsPrefix, w); err != nil {
				return fmt.Errorf("%s: digit %q at pos %d: %w", methodBuildNumber, r, pos, err)
			}
			pos++
		}
		return nil
	}
}

// -----------------------------------------------------------------------------
// Internal helpers (focused, side-effect free except for graph mutations)
// -----------------------------------------------------------------------------

// resolveSpec first looks in letterSpecs (letters), then in numberSpec (digits).
// This allows BuildLetters to handle mixed inputs (letters+digits) seamlessly.
func resolveSpec(r rune) (letterSpec, bool) {
	if sp, ok := letterSpecs[r]; ok {
		return sp, true
	}
	if sp, ok := numberSpec[r]; ok {
		return sp, true
	}
	return letterSpec{}, false
}

// makeNamespace composes a stable per-glyph namespace prefix.
// Empty scope → empty prefix; Non-empty scope → "<scope>::<pos>::".
func makeNamespace(scope string, pos int) string {
	if scope == "" {
		return ""
	}
	return scope + nsSep + strconv.Itoa(pos) + nsSep
}

// qualify transforms a canonical vertex ID into a possibly-namespaced one.
// If prefix is empty, returns the canonical ID unchanged.
func qualify(prefix, canonical string) string {
	if prefix == "" {
		return canonical
	}
	return prefix + canonical
}

// emitGlyph materializes one glyph into the graph:
//   - Adds vertices in the canonical order (spec.IDs) with idempotency.
//   - Emits edges (spec.Edges) with weight policy and directed mirroring.
//   - Skips self-loops when g.Looped()==false.
//
// Complexity: O(V+E) with tiny constants.
func emitGlyph(g *core.Graph, spec letterSpec, nsPrefix string, w float64) error {
	// 1) Add all vertices in deterministic order.
	for _, id := range spec.IDs {
		qid := qualify(nsPrefix, id) // possibly namespaced ID
		if g.HasVertex(qid) {        // idempotency: skip if already present
			continue
		}
		if err := g.AddVertex(qid); err != nil { // delegate mode validation to core
			return fmt.Errorf("AddVertex(%s): %w", qid, err)
		}
	}

	// 2) Emit all edges in declared emission order.
	var u, v string
	var exists bool
	for _, ep := range spec.Edges {
		u = qualify(nsPrefix, ep.U) // namespace both endpoints
		v = qualify(nsPrefix, ep.V)

		// Respect Looped(): if self-loop and loops are disallowed → skip.
		if u == v && !g.Looped() {
			continue // deterministic and safe no-op
		}

		// Add (u,v) once if it doesn't exist.
		exists = g.HasEdge(u, v) || (!g.Directed() && g.HasEdge(v, u))
		if !exists {
			if _, err := g.AddEdge(u, v, w); err != nil {
				return fmt.Errorf("AddEdge(%s→%s, w=%g): %w", u, v, w, err)
			}
		}

		// Mirror for directed graphs to preserve undirected semantics explicitly.
		if g.Directed() && !g.HasEdge(v, u) {
			if _, err := g.AddEdge(v, u, w); err != nil {
				return fmt.Errorf("AddEdge(%s→%s, w=%g): %w", v, u, w, err)
			}
		}
	}

	// 3) (Optional edges) - emit if present in spec.OptionalEdges (none by default in your dataset).
	for _, ep := range spec.OptionalEdges {
		u = qualify(nsPrefix, ep.U)
		v = qualify(nsPrefix, ep.V)
		if u == v && !g.Looped() {
			continue
		}

		exists = g.HasEdge(u, v) || (!g.Directed() && g.HasEdge(v, u))
		if !exists {
			if _, err := g.AddEdge(u, v, w); err != nil {
				return fmt.Errorf("AddEdge(optional %s→%s, w=%g): %w", u, v, w, err)
			}
		}
		if g.Directed() && !g.HasEdge(v, u) {
			if _, err := g.AddEdge(v, u, w); err != nil {
				return fmt.Errorf("AddEdge(optional %s→%s, w=%g): %w", v, u, w, err)
			}
		}
	}

	return nil
}
