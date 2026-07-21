// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
//
// Package: lvlath/builder
//
// api.go - thin public entry-points for the builder package.
//
// Design contract (strict):
//   - One orchestrator: BuildGraph(gopts, bopts, cons...). Creates g, resolves cfg, runs cons in order.
//   - All public factories are declared here, implemented in impl_*.go (single place to read docs).
//   - Functional options (Option) resolve into an immutable builderConfig (no global state).
//   - Determinism: same inputs/options/seed and constructor order ⇒ identical graphs/series.
//   - Safety: never panic; return sentinel errors from constructors; data helpers return nil on invalid input.
//
// AI-Hints (practical):
//   - Compose multiple constructors in BuildGraph to assemble complex fixtures deterministically.
//   - Use WithSeed(...) to freeze stochastic paths (RandomSparse/Regular, sequence RNG via cfg.rng).
//   - WithIDScheme(...) for human-readable vertex IDs (graphs only).
//   - WithPartitionPrefix(left,right) for bipartite graphs (empty ⇒ defaults).
//   - Sequences (impl_pulse.go / impl_chirp.go / impl_ohlc.go): options are resolved via newBuilderConfig(opts...).
//     If you add knobs (Noise/Trend/Frequency/etc.), wire them in extract*Params and keep determinism stable.

package builder

import (
	"fmt"

	"github.com/lvlath/go/core"
)

// Constructor applies a deterministic graph mutation using the resolved
// builderConfig. Constructors MUST:
//   - Validate parameters early and return sentinel errors (no panics).
//   - Respect core graph mode flags (directed/loops/multigraph/weighted).
//   - Preserve determinism for the same config and call order.
//
// Rationale: isolates topology logic behind a uniform function type.
// Complexity (this type): O(1) to pass; actual cost is in the closure body.
type Constructor func(g *core.Graph, cfg builderConfig) error

// BuildGraph creates a new core.Graph with graph options gopts, resolves the
// builder configuration from bopts, and applies all constructors in order.
// Any constructor error is wrapped with the context "BuildGraph: %w" and
// returned immediately; no partial cleanup is attempted by design.
//
// Rationale:
//   - Single public entry-point ensures consistent option resolution & error wrapping.
//   - Enforces deterministic composition order of constructors.
//
// Complexity:
//   - Resolving options: O(len(bopts)) time, O(1) space.
//   - Applying K constructors: Σ cost of each constructor; wrapper overhead O(K).
//
// Concurrency:
//   - The function is not concurrent by itself; it invokes core which manages locks.
//
// Errors:
//   - Wraps constructor errors via %w; callers should branch with errors.Is
//     against builder sentinels (ErrTooFewVertices, ErrInvalidProbability, ...).
func BuildGraph(gopts []core.GraphOption, bopts []Option, cons ...Constructor) (*core.Graph, error) {
	// Create a new graph using the provided core graph options (O(1) here).
	g, err := core.NewGraph(gopts...)
	if err != nil {
		return nil, err
	}

	// Resolve deterministic builder configuration from functional options (O(len(bopts))).
	cfg := newBuilderConfig(bopts...)

	// Apply each constructor sequentially to preserve deterministic order & effects.
	for i, fn := range cons {
		// Defensive: reject a nil constructor to avoid a panic later (programmer error).
		if fn == nil {
			// Use a sentinel that communicates construction failure; keep %w for Is().
			return nil, fmt.Errorf("BuildGraph: nil constructor at index %d: %w", i, ErrConstructFailed)
		}
		// Execute the constructor. Implementations must not panic; they must return errors.
		if err := fn(g, cfg); err != nil {
			// Wrap once at the API boundary; inner layers may have already wrapped with context.
			return nil, fmt.Errorf("BuildGraph: %w", err)
		}
	}

	// Success: return the fully constructed graph (deterministic for equal inputs).
	return g, nil
}

// =============================================================================
// Topology factories (declarations) - implemented in impl_*.go
// =============================================================================
//
// Each factory returns a Constructor closure. The closure MUST:
//   - Add vertices via cfg.idFn (except documented fixed IDs like "Center").
//   - Emit edges in a stable, documented order.
//   - Honor core flags (Directed/Weighted/Loops/Multigraph) without silent degrade.
//   - Return only sentinel errors; NEVER panic at runtime.

// Cycle builds an n-vertex simple cycle C_n (n ≥ 3).
// Complexity: O(n) vertices + O(n) edges; O(1) extra space.
//func Cycle(n int) Constructor

// Path builds a simple path P_n (n ≥ 2).
// Complexity: O(n) vertices + O(n-1) edges; O(1) extra space.
//func Path(n int) Constructor

// Star builds a star with center "Center" and n-1 leaves (n ≥ 2).
// Complexity: O(n) vertices + O(n-1) edges; O(1) extra space.
//func Star(n int) Constructor

// Wheel builds a wheel W_n = C_{n-1} + center "Center" (n ≥ 4).
// Complexity: O(n) vertices + O(2n-2) edges; O(1) extra space.
//func Wheel(n int) Constructor

// Complete builds the complete simple graph K_n (n ≥ 1).
// Complexity: O(n) vertices + O(n^2) edges; O(1) extra space.
//func Complete(n int) Constructor

// CompleteBipartite builds simple K_{n1,n2} using cfg.leftPrefix/cfg.rightPrefix.
// Complexity: O(n1+n2) vertices + O(n1*n2) edges; O(1) extra space.
//func CompleteBipartite(n1, n2 int) Constructor

// Grid builds an R×C 4-neighborhood grid with IDs "r,c" (row-major).
// Complexity: O(R*C) vertices + O(R*C) edges; O(1) extra space.
//func Grid(rows, cols int) Constructor

// RandomSparse builds an Erdős–Rényi-like sparse graph.
// Requires cfg.rng != nil and 0 ≤ p ≤ 1.
// Complexity: undirected O(n^2) pair checks; directed O(n^2) ordered pairs.
// Deterministic for fixed seed and options.
//func RandomSparse(n int, p float64) Constructor

// RandomRegular builds a d-regular simple graph via stub-matching with bounded retries.
// Only for undirected simple graphs; requires cfg.rng != nil.
// Complexity: ~O(n*d) per attempt; attempts are constant-bounded. Deterministic per seed.
//func RandomRegular(n, d int) Constructor

// PlatonicSolid builds a fixed Platonic topology; optionally adds a "Center" with spokes.
// Complexity: O(V+E) for the chosen solid; stable emission order.
//func PlatonicSolid(name PlatonicName, withCenter bool) Constructor

// Hexagram overlays variant-specific chord sets over a base Cycle/Wheel ring.
// Complexity: O(n + |chords|) where n is the ring size of the chosen variant.
//func Hexagram(variant HexagramVariant) Constructor

// =============================================================================
// Letters/Word and Numbers/Digits (constructors + thin wrappers) - impl in impl_letters.go
// =============================================================================

// Letters adds per-letter subgraphs using the canonical ID scheme
// documented in letters_spec.go. Unknown runes → ErrUnknownLetter.
// Complexity: O(total nodes/edges across letters). Deterministic per input order.
//func Letters(text string, scope string) Constructor

// Word composes Letters over runes(word) in input order.
// Complexity: O(len(word)) constructors + total glyph size. Deterministic per input.
//func Word(word string, scope string) Constructor

// BuildLetters is a thin helper: resolve cfg and run Letters(...) against an existing g.
// It returns sentinel errors; it never panics.
// Complexity: O(len(opts)) + cost of Letters constructor.
func BuildLetters(g *core.Graph, text, scope string, opts ...Option) error {
	// Resolve configuration once for this call.
	cfg := newBuilderConfig(opts...)

	// Defensive: ensure we have a target graph to mutate.
	if g == nil {
		return fmt.Errorf("BuildLetters: nil graph: %w", ErrConstructFailed)
	}

	// Delegate to the constructor; Letters(...) MUST be implemented in impl_letters.go.
	return Letters(text, scope)(g, cfg)
}

// BuildWord is a thin helper: resolve cfg and run Word(...) against an existing g.
// It returns sentinel errors; it never panics.
// Complexity: O(len(opts)) + cost of Word constructor.
func BuildWord(g *core.Graph, word, scope string, opts ...Option) error {
	// Resolve configuration for deterministic behavior.
	cfg := newBuilderConfig(opts...)

	// Defensive: avoid nil graph dereference.
	if g == nil {
		return fmt.Errorf("BuildWord: nil graph: %w", ErrConstructFailed)
	}

	// Delegate to the constructor; Word(...) MUST be implemented in impl_letters.go.
	return Word(word, scope)(g, cfg)
}

func BuildDigit(g *core.Graph, digit int, scope string, opts ...Option) error {
	cfg := newBuilderConfig(opts...)
	if g == nil {
		return fmt.Errorf("BuildDigit: nil graph: %w", ErrConstructFailed)
	}
	return Digit(digit, scope)(g, cfg)
}

func BuildNumber(g *core.Graph, number float64, decimal bool, scope string, opts ...Option) error {
	cfg := newBuilderConfig(opts...)
	if g == nil {
		return fmt.Errorf("BuildNumber: nil graph: %w", ErrConstructFailed)
	}
	return Number(number, decimal, scope)(g, cfg)
}

// =============================================================================
// Sequence datasets - impl in impl_pulse.go, impl_chirp.go and impl_ohlc.go
// =============================================================================
//
// Determinism policy (sequences):
//   - RNG selection uses rngFrom(cfg, seed): if cfg.rng != nil → use shared stream; else a local rand.New(rand.NewSource(seed)).
//   - Options (A/f0/duty/triangular/sigma/trend/GBM params) are resolved via newBuilderConfig(opts...) → extract*Params.
//   - No NaN/Inf; OHLC invariants are guaranteed by implementation.
//
// BuildPulse returns a deterministic pulse series of length n.
// Validation: n ≥ 1 (else returns nil). Determinism per (n, seed, opts...).
// Complexity: O(n).
//func BuildPulse(n int, seed int64, opts ...Option) []float64
//
// BuildAudioChirp returns a deterministic linear chirp series of length n.
// Validation: n ≥ 1 (else returns nil). Determinism per (n, seed, opts...).
// Complexity: O(n).
//func BuildAudioChirp(n int, seed int64, opts ...Option) []float64
//
// BuildOHLCSeries returns deterministic OHLC arrays for the given number of days.
// Validation: days ≥ 1 (else returns nils). Invariants: low≤min(open,close), high≥max(...).
// Complexity: O(days * steps) where steps is a small constant.
//func BuildOHLCSeries(days int, seed int64, opts ...Option) (open, high, low, close []float64)
