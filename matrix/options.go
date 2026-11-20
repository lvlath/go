// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix: functional configuration for graph→matrix adapters and
// numeric policy. This file defines:
//   - Option / Options (functional options with internal state),
//   - documented defaults (constants),
//   - WithX constructors with strong error-first validation,
//   - gatherOptions helper (internal) that enforces invariants.
//
// Design goals:
//   - Deterministic behavior: no global state, no implicit randomness.
//   - No dead switches: each flag impacts behavior and is covered by tests.
//   - Safe by construction: ordinary invalid public input returns errors, not panics.
//   - Reusability: Options fields are unexported (is internal); public APIs consume ...Option.
//
// Notes:
//   - Core vs Matrix defaults:
//   - core defaults are: undirected, unweighted, loops=false, multi-edges=false.
//   - matrix adapter defaults mirror that spirit, with one intentional difference:
//     DefaultAllowMulti=true so adapters can faithfully represent multi-edge graphs
//     when the source graph was explicitly built with WithMultiEdges().
//   - Directedness mapping (core → matrix):
//   - core can be uniform-directed or mixed-mode per-edge (Edge.Directed).
//   - adapters must remain deterministic: vertex order is stable (ID asc), and
//     edge iteration must be stable (Edge.ID asc) before writing into matrices.
//   - Multi-edge representation:
//   - Dense adjacency has one cell per (u,v); it cannot losslessly represent
//     parallel edges. When AllowMulti=true, a deterministic overwrite policy
//     must be documented (e.g., last-write-wins under stable edge order).
//   - Incidence matrices can represent multi-edges naturally (one column per edge).
//   - Weighted semantics:
//   - core enforces weight==0 unless WithWeighted() was enabled.
//   - matrix.Weighted controls whether adapters export numeric weights or a binary structure.
//   - incidence sign is structural; numeric weights are intentionally ignored there.
//   - Numeric policy is orthogonal and explicit:
//   - validateNaNInf controls whether Set()/ingestion rejects NaN/Inf at all.
//   - allowInfDistances is a narrow exception for +Inf as “no path” in distance matrices.
//     Under validation, NaN and -Inf remain rejected even when allowInfDistances=true.
//   - MetricClosure / APSP:
//   - distance-policy builders require +Inf off-diagonal for “no path”.
//   - those builders must allocate Dense with allowInfDistances=true to make Set(+Inf) legal.
package matrix

import "fmt"

// ---------- Defaults (single source of truth) ----------

// Numeric policy.
const (
	// DefaultEpsilon defines the non-negative tolerance used by structural checks.
	DefaultEpsilon = 1e-9

	// DefaultValidateNaNInf toggles strict finite-value validation on ingestion and Set.
	DefaultValidateNaNInf = true

	// DefaultAllowInfDistances permits +Inf values to represent “no path” in
	// APSP/MetricClosure distance-policy matrices.
	//
	// IMPORTANT:
	//   - This is NOT a “dirty-data” mode.
	//   - When ValidateNaNInf is enabled, NaN and -Inf are still rejected; only +Inf
	//     is allowed by this mode.
	DefaultAllowInfDistances = false
)

// DEFAULTS - single source of truth for zero-value behavior.
// These constants MUST reflect the intended defaults in defaultOptions.
const (
	// DefaultDirected controls whether edges are treated as directed.
	// false ⇒ undirected (mirror [u,v] into [v,u], except loops).
	DefaultDirected = false

	// DefaultWeighted controls whether actual edge weights are preserved.
	// false ⇒ build binary adjacency/incidence with unit entries.
	DefaultWeighted = false

	// DefaultAllowMulti allows multiple parallel edges between the same endpoints.
	// true ⇒ include all by default; row/col reflects the last write in a dense
	// adjacency cell (structural limitation). When false, the policy is
	// first-edge-wins; see comments in builders.
	DefaultAllowMulti = true

	// DefaultAllowLoops includes self-loops when true.
	DefaultAllowLoops = false

	// DefaultMetricClosure converts adjacency to all-pairs shortest-path distances
	// (Floyd–Warshall) when true: diag=0, off-diag=+Inf if no path.
	// IMPORTANT: ToGraph is unsupported under MetricClosure (ErrMatrixNotImplemented).
	DefaultMetricClosure = false

	// DefaultAllowZeroShape documents the package-wide zero-shape law.
	//
	// Contract:
	//   - 0×0, 0×N, and N×0 matrices are valid structural matrices.
	//   - Operations that have no elements to scan should return deterministic no-op
	//     results where mathematically meaningful.
	//   - Operations requiring sample degrees of freedom may still reject degenerate
	//     observation counts explicitly.
	//
	// Why constant instead of option:
	//   - Zero-shape validity is a package invariant, not a user preference.
	//   - Making it configurable would reintroduce split behavior across builders,
	//     element-wise kernels, and statistics.
	//
	// AI-Hints:
	//   - Do not add WithAllowZeroShape/WithDisallowZeroShape.
	//   - Use validators to express stronger preconditions locally.
	DefaultAllowZeroShape = true
)

// Export policy (AdjacencyMatrix.ToGraph).
const (
	// DefaultEdgeThreshold a[i,j] > threshold ⇒ edge.
	DefaultEdgeThreshold = 0.5

	// DefaultKeepWeights if true, weight=a[i,j]; else weight=1.
	DefaultKeepWeights = true

	// DefaultBinaryWeights if true, exported edges get weight=1 (ignores a[i,j]).
	DefaultBinaryWeights = false

	// DefaultPreserveZeroWeights controls whether all-zero weighted input is preserved
	// as real zero-weight edges instead of being auto-degraded to binary adjacency.
	//
	// false preserves the historical adapter behavior:
	// weighted-looking input with only zero weights is treated as effectively unweighted.
	// true forces +Inf-as-no-edge encoding so finite 0 remains a real edge weight.
	DefaultPreserveZeroWeights = false
)

// ---------- Public option type (functional) ----------

// Option mutates matrix Options during deterministic option assembly.
//
// What:
// - Option is the public configuration setter type for matrix facades.
//
// Why:
// - It makes contract-changing behavior explicit and testable.
//
// Implementation:
// - gatherOptions applies options left-to-right.
// - Each Option validates its own public input and returns a sentinel-preserving error.
// - Ordinary invalid public inputs must return errors, not panic.
//
// Inputs:
// - *Options: non-nil policy object owned by gatherOptions.
//
// Returns:
// - nil on success.
// - Sentinel-preserving error on invalid option input.
//
// Errors:
// - ErrNaNInf: NaN or forbidden infinity in numeric option arguments.
// - ErrOutOfRange: negative epsilon/threshold or other invalid numeric range.
// - ErrNilOption: returned by gatherOptions when an option slot is nil.
//
// Determinism:
// - Option application order is stable and equals call order.
//
// Complexity:
// - O(1) per option.
//
// AI-Hints:
// - Never reintroduce panic-based option validation.
// - Keep options local: mutate only the provided *Options.
type Option func(*Options) error

// Options stores matrix construction, graph-adapter, numeric, and export policy.
//
// What:
//   - Options is the frozen policy object produced by gatherOptions/NewMatrixOptions.
//   - It controls numeric validation, graph directionality, edge handling,
//     weight encoding, metric-closure behavior, and threshold-based graph export.
//
// Why:
// - Matrix builders and adapters need one deterministic source of policy.
// - Public facades must not reinvent option semantics locally.
//
// Implementation:
// - Options values are assembled by applying Option values left-to-right.
// - finalizeOptions resolves derived invariants once after all user options apply.
//
// Behavior highlights:
// - Zero value is not the public default contract; use gatherOptions/NewMatrixOptions.
// - Fields are intentionally unexported to prevent partial, unfinalized policies.
// - metricClose forces allowInfDistances because +Inf is required for "no path".
// - binaryWeights and keepWeights are mutually exclusive after finalization.
//
// Numeric policy:
//   - validateNaNInf controls ordinary Dense value validation.
//   - allowInfDistances allows +Inf only in distance/metric-closure contexts;
//     it does not authorize NaN or -Inf as valid numeric payloads.
//
// Weight policy:
// - keepWeights=true means graph weights are preserved where the builder contract allows it.
// - binaryWeights=true means topology is encoded as binary presence.
// - finalizeOptions preserves last-writer-wins between WithKeepWeights and WithBinaryWeights.
//
// AI-Hints:
// - Do not construct Options manually in production code.
// - Do not bypass finalizeOptions in tests; use gatherOptions or NewMatrixOptions.
// - Do not use panic for option validation.
type Options struct {
	// numeric policy
	eps               float64 // >= 0; DefaultEpsilon
	validateNaNInf    bool    // DefaultValidateNaNInf
	allowInfDistances bool    // DefaultAllowInfDistances (+Inf as “no path”)

	// adjacency/incidence build policy
	directed    bool // DefaultDirected
	allowMulti  bool // DefaultAllowMulti
	allowLoops  bool // DefaultAllowLoops
	weighted    bool // DefaultWeighted
	metricClose bool // DefaultMetricClosure

	// preserveZeroWeights forces weighted adjacency to preserve finite zero weights.
	// When true, builders use +Inf as the no-edge sentinel whenever zero weights
	// would otherwise be ambiguous.
	preserveZeroWeights bool

	// export policy (ToGraph)
	edgeThreshold    float64 // DefaultEdgeThreshold
	edgeThresholdSet bool    // true when WithEdgeThreshold was explicitly provided
	keepWeights      bool    // DefaultKeepWeights
	binaryWeights    bool    // DefaultBinaryWeights
}

// ---------- Constructors (WithX) ----------

// WithEpsilon sets the numeric tolerance eps used by structural checks.
// Implementation:
//   - Stage 1: validate eps is finite and ≥ 0.
//   - Stage 2: return a setter that writes eps into Options.
//
// Behavior highlights:
//   - Error-first validation; nonsensical values are rejected by the returned Option.
//
// Inputs:
//   - eps: non-negative finite tolerance.
//
// Returns:
//   - Option: functional setter.
//
// Errors:
//   - ErrNaNInf: eps is NaN or ±Inf.
//   - ErrOutOfRange: eps < 0.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Applies to structural numeric checks (e.g., AllClose-like, symmetry).
//     Larger eps relaxes equality checks; use judiciously.
//
// AI-Hints:
//   - Prefer small positive eps (e.g., 1e-9) for double-precision data or unless dealing with noisy data.
//   - Zero epsilon is allowed and means exact comparison where the consuming
//     operation supports tolerance.
func WithEpsilon(eps float64) Option {
	return func(o *Options) error {
		if isNonFinite(eps) {
			return fmt.Errorf("matrix: WithEpsilon(%v): %w", eps, ErrNaNInf)
		}
		if eps < 0 {
			return fmt.Errorf("matrix: WithEpsilon(%v): %w", eps, ErrOutOfRange)
		}

		// Assign validated epsilon
		o.eps = eps

		return nil
	}
}

// WithValidateNaNInf enables strict finite-value validation.
// Implementation:
//   - Stage 1: set validateNaNInf=true.
//
// Behavior highlights:
//   - When enabled, NaN and -Inf are always rejected.
//   - +Inf is rejected unless AllowInfDistances is enabled.
//
// Returns:
//   - Option: functional setter.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Affects newly created matrices via builders; existing matrices keep their policy.
//   - This is the default; use WithNoValidateNaNInf to relax.
//
// AI-Hints:
//   - Keep this enabled in data-clean pipelines; disable only when ingesting
//     external data with known ±Inf placeholders and sanitizing later or in controlled experiments.
//   - Combine with WithAllowInfDistances for APSP.
//   - Do not use this option to validate distance matrices alone; distance
//     semantics also require shape/diagonal checks.
func WithValidateNaNInf() Option {
	return func(o *Options) error {
		o.validateNaNInf = true
		return nil
	}
}

// WithNoValidateNaNInf disables NaN/Inf validation (use with care).
// Implementation:
//   - Stage 1: set validateNaNInf=false.
//
// Behavior highlights:
//   - Allows ±Inf/NaN to pass through on newly created matrices.
//   - This is an explicit expert mode.
//   - It should be reserved for trusted numeric pipelines or external validation.
//
// Returns:
//   - Option: functional setter.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This flag propagates only on creation; existing matrices are unaffected.
//
// AI-Hints:
//   - Combine with data sanitization ops (ReplaceInfNaN, Clip) if you disable checks.
//   - Do not use this as a shortcut for distance matrices; prefer
//     WithAllowInfDistances/WithMetricClosure when +Inf is semantically required.
func WithNoValidateNaNInf() Option {
	return func(o *Options) error {
		o.validateNaNInf = false
		return nil
	}
}

// WithAllowInfDistances permits +Inf entries to represent “no path” in distance-policy matrices.
// Implementation:
//   - Stage 1: set allowInfDistances=true.
//
// Behavior highlights:
//   - Allows +Inf where the Dense value policy supports distance matrices.
//   - Does not make NaN valid.
//   - Does not make -Inf valid.
//
// Returns:
//   - Option: functional setter.
//
// Errors:
//   - none (pure setter).
//
// Determinism:
//   - N/A.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Intended for APSP/MetricClosure where +Inf is a semantic sentinel.
//
// AI-Hints:
//   - Use together with WithMetricClosure or APSP builders.
//   - Use for shortest-path/distance matrices, not for arbitrary Dense payloads.
func WithAllowInfDistances() Option {
	return func(o *Options) error {
		o.allowInfDistances = true
		return nil
	}
}

// WithDisallowInfDistances disables +Inf-permission mode (default).
// Implementation:
//   - Stage 1: set allowInfDistances=false.
//
// Behavior highlights:
//   - This is the default.
//   - If WithMetricClosure is also present, finalizeOptions re-enables +Inf
//     because metric closure needs +Inf for missing paths.
//
// Returns:
//   - Option: functional setter.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Do not expect this option to override WithMetricClosure; metric closure has
//     a stronger derived invariant.
func WithDisallowInfDistances() Option {
	return func(o *Options) error {
		o.allowInfDistances = false
		return nil
	}
}

// WithDirected builds directed adjacency/incidence (no mirroring).
// Implementation:
//   - Stage 1: set directed=true; export undirected is derived later.
//
// Behavior highlights:
//   - Export 'undirected' is computed as !directed in gatherOptions.
//
// Returns:
//   - Option: functional setter.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Incidence sign conventions depend on this flag.
//   - For export, undirected is inferred; no separate public toggle required.
//
// AI-Hints:
//   - This controls adapter interpretation, not Dense storage layout.
//   - Use WithUndirected for mirrored graphs without asymmetric weights.
//   - Combine with AllowMulti/AllowLoops explicitly for reproducibility.
func WithDirected() Option {
	return func(o *Options) error {
		o.directed = true
		return nil
	}
}

// WithUndirected builds undirected adjacency/incidence (mirror [u,v]→[v,u], except loops).
// Implementation:
//   - Stage 1: set directed=false.
//
// Behavior highlights:
//   - Export undirected field will be set to true by gatherOptions.
//
// Returns:
//   - Option: functional setter.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Loops remain on the diagonal and are not mirrored.
//
// AI-Hints:
//   - Prefer undirected for symmetric datasets; pair with Weighted as needed.
//   - In undirected mode, multi-edge deduplication normalizes pair keys.
//   - Undirected builders may mirror adjacency entries where the source edge
//     policy permits it.
func WithUndirected() Option {
	return func(o *Options) error {
		o.directed = false
		return nil
	}
}

// WithAllowMulti includes parallel edges in the ingestion phase.
// Implementation:
//   - Stage 1: set allowMulti=true.
//
// Behavior highlights:
//   - Adjacency cells may be updated multiple times; last-write-wins on Dense.
//
// Returns:
//   - Option: functional setter.
//
// Determinism:
//   - Deterministic if edge iteration order is stable (builders enforce this).
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Undirected mode uses normalized keys {min,max} for de-duplication.
//
// AI-Hints:
//   - Prefer disallowing multi unless your pipeline requires parallel edges.
//   - If you need strict first-edge semantics, use WithDisallowMulti.
//   - The exact merge/overwrite semantics belong to the consuming builder/exporter;
//     this option only enables the domain policy.
func WithAllowMulti() Option {
	return func(o *Options) error {
		o.allowMulti = true
		return nil
	}
}

// WithDisallowMulti enforces "first-edge-wins" de-duplication.
// Implementation:
//   - Stage 1: set allowMulti=false.
//
// Behavior highlights:
//   - Directed: key uses ordered pair (u,v); Undirected: unordered {min,max}.
//
// Returns:
//   - Option: functional setter.
//
// Determinism:
//   - Requires stable vertex indexing (builders guarantee lexicographic order).
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Prevents late edges from overwriting earlier weights.
//
// AI-Hints:
//   - Prefer this when ingest order is meaningful (ETL pipelines).
//   - Keep this default for deterministic simple-graph exports.
func WithDisallowMulti() Option {
	return func(o *Options) error {
		o.allowMulti = false
		return nil
	}
}

// WithAllowLoops includes self-loops (u==v) during ingestion.
// Implementation:
//   - Stage 1: set allowLoops=true.
//
// Behavior highlights:
//   - Loops affect diagonal entries only.
//
// Returns:
//   - Option: functional setter.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - For incidence, undirected loop contributes +2 in a single row.
//
// AI-Hints:
//   - Enable when loops carry semantic meaning (e.g., self-transitions).
func WithAllowLoops() Option {
	return func(o *Options) error {
		o.allowLoops = true
		return nil
	}
}

// WithDisallowLoops ignores self-loops during ingestion.
// Implementation:
//   - Stage 1: set allowLoops=false.
//
// Behavior highlights:
//   - Diagonal contributions are suppressed.
//
// Returns:
//   - Option: functional setter.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Use to simplify structures where loops are artifacts.
//
// AI-Hints:
//   - Pairs well with binary adjacency for structural analyses.
func WithDisallowLoops() Option {
	return func(o *Options) error {
		o.allowLoops = false
		return nil
	}
}

// WithWeighted preserves actual edge weights if the input graph is weighted.
// Implementation:
//   - Stage 1: set weighted=true.
//
// Behavior highlights:
//   - Adjacency: stores edge weights (or 1 if graph is effectively unweighted).
//   - Incidence: structure-only; weights are ignored by design.
//
// Returns:
//   - Option: functional setter.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - In auto mode, all-zero weighted-looking input may degrade to binary adjacency
//     to preserve historical unweighted core-graph adapter behavior.
//   - Use WithPreserveZeroWeights when zero is a meaningful edge weight and must
//     survive adjacency construction/export.
//   - Incidence ignoring weights is intentional and documented; see file header.
//
// AI-Hints:
//   - Mixed zero/non-zero weighted input is preserved automatically by +Inf no-edge encoding.
//   - All-zero weighted graphs require WithPreserveZeroWeights to avoid auto-degrade.
//   - Combine with MetricClosure to obtain distance matrices (APSP) from weights.
//   - Prefer WithKeepWeights in code that talks specifically about adjacency
//     encoding; prefer WithWeighted in graph-domain examples.
func WithWeighted() Option {
	return func(o *Options) error {
		o.weighted = true

		return nil
	}
}

// WithUnweighted forces binary adjacency/incidence (unit weights).
// Implementation:
//   - Stage 1: no validation needed.
//   - Stage 2: set weighted=false.
//
// Behavior highlights:
//   - Adjacency becomes {0,1}; incidence remains sign-based.
//
// Returns:
//   - Option: functional setter.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Useful for topology-only analyses.
//
// AI-Hints:
//   - Pair with DisallowMulti for clean simple graphs.
//   - This is a policy choice, not a fallback from invalid weights.
func WithUnweighted() Option {
	return func(o *Options) error {
		o.weighted = false
		o.preserveZeroWeights = false

		return nil
	}
}

// WithPreserveZeroWeights forces weighted adjacency builders to preserve zero-weight edges.
//
// Implementation:
//   - Stage 1: set weighted=true because zero-weight preservation is meaningful
//     only in weighted adjacency.
//   - Stage 2: set preserveZeroWeights=true.
//
// Behavior highlights:
//   - Mixed zero/non-zero weighted input is already preserved automatically.
//   - This option is primarily needed for all-zero weighted graphs, where auto mode
//     would otherwise degrade to binary adjacency.
//   - Builders use +Inf as the no-edge sentinel so finite 0 remains an edge weight.
//
// Returns:
//   - Option: functional setter.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Incidence matrices ignore weights by design; this option affects adjacency builders.
//
// AI-Hints:
//   - Use this when 0 is a real domain weight, not an absent edge.
//   - Do not use NaN/-Inf sentinels to preserve zero weights.
func WithPreserveZeroWeights() Option {
	return func(o *Options) error {
		o.weighted = true
		o.preserveZeroWeights = true
		return nil
	}
}

// WithAutoZeroWeights restores the default all-zero auto-degrade behavior.
//
// Implementation:
//   - Stage 1: set preserveZeroWeights=false.
//   - Stage 2: leave weighted unchanged; callers may still choose weighted/unweighted separately.
//
// Behavior highlights:
//   - In weighted adjacency, all-zero edge lists may be treated as effectively unweighted.
//   - Mixed zero/non-zero input remains preserved automatically.
//
// Returns:
//   - Option: functional setter.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - This is the compatibility mode. Use WithPreserveZeroWeights for exact
//     all-zero weighted graphs.
func WithAutoZeroWeights() Option {
	return func(o *Options) error {
		o.preserveZeroWeights = false
		return nil
	}
}

// WithMetricClosure converts adjacency into all-pairs shortest-path distances via Floyd–Warshall (APSP).
// Implementation:
//   - Stage 1: no validation needed.
//   - Stage 2: set metricClose=true (APSP later performed in builders).
//
// Behavior highlights:
//   - Produces distance-policy matrices: diag=0, +Inf denotes no path.
//   - ToGraph is intentionally unsupported and must return ErrMatrixNotImplemented.
//
// Returns:
//   - Option: functional setter.
//
// Errors:
//   - none here; APSP may surface errors during execution.
//
// Determinism:
//   - Floyd–Warshall uses a fixed k→i→j loop order.
//
// Complexity:
//   - Time O(1) to set; APSP later is O(n^3).
//
// Notes:
//   - Input adjacency is not mutated when using non-in-place facade.
//
// AI-Hints:
//   - Prefer APSPFromAdjacency for convenience if available at API level.
func WithMetricClosure() Option {
	return func(o *Options) error {
		o.metricClose = true
		return nil
	}
}

// WithEdgeThreshold sets threshold for ToGraph export (a[i,j] > t ⇒ edge).
// Implementation:
//   - Stage 1: set edgeThreshold=t.
//
// Behavior highlights:
//   - Affects only export; has no impact on in-memory computations.
//
// Inputs:
//   - t: finite threshold (NaN/±Inf disallowed).
//
// Returns:
//   - Option: functional setter.
//
// Errors:
//   - ErrNaNInf: t is NaN or ±Inf.
//   - ErrOutOfRange: t < 0.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - In classic 0-as-no-edge adjacency, threshold is always applied.
//   - In +Inf-as-no-edge weighted adjacency, the default threshold is not used
//     to erase finite zero-weight edges. Threshold filtering is applied only
//     when the caller explicitly provides WithEdgeThreshold.
//
// AI-Hints:
//   - For weighted graphs, pick a domain-relevant cutoff.
//   - Threshold controls export/presence decisions; it is not a numeric sanitizer.
func WithEdgeThreshold(t float64) Option {
	return func(o *Options) error {
		if isNonFinite(t) {
			return fmt.Errorf("matrix: WithEdgeThreshold(%v): %w", t, ErrNaNInf)
		}
		if t < 0 {
			return fmt.Errorf("matrix: WithEdgeThreshold(%v): %w", t, ErrOutOfRange)
		}
		o.edgeThreshold = t
		o.edgeThresholdSet = true

		return nil
	}
}

// WithKeepWeights keeps numeric weights on ToGraph export (weight=a[i,j]).
// Implementation:
//   - Stage 1: set keepWeights=true and binaryWeights=flase.
//
// Behavior highlights:
//   - Explicitly disables binary weight encoding.
//   - Last-writer-wins against WithBinaryWeights.
//
// Returns:
//   - Option: functional setter.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Combine with Weighted adjacency for meaningful magnitudes.
//
// AI-Hints:
//   - Set an appropriate EdgeThreshold to filter weak links.
//   - Do not remove the explicit binaryWeights=false assignment; it is what makes
//     last-writer-wins deterministic before finalizeOptions.
func WithKeepWeights() Option {
	return func(o *Options) error {
		o.keepWeights = true
		o.binaryWeights = false

		return nil
	}
}

// WithBinaryWeights forces unit weights on ToGraph export (weight=1).
// Implementation:
//   - Stage 1: no validation needed.
//   - Stage 2: set keepWeights=false and binaryWeights=true.
//
// Behavior highlights:
//   - Edges become unweighted on export, irrespective of matrix entries.
//   - Explicitly disables weight preservation.
//   - Last-writer-wins against WithKeepWeights.
//
// Returns:
//   - Option: functional setter.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Useful for producing topological graphs from weighted matrices.
//
// AI-Hints:
//   - Pair with a threshold to sparsify exports.
//   - This is not a fallback for invalid weighted data; it is an explicit topology
//     encoding mode.
func WithBinaryWeights() Option {
	return func(o *Options) error {
		o.binaryWeights = true
		o.keepWeights = false

		return nil
	}
}

// --------------------------- Deprecated Aliases ---------------------------

// DisableValidateNaNInf disables NaN/Inf validation.
// Deprecated: Use WithNoValidateNaNInf.
// NOTE: This alias is kept for backwards compatibility and may be removed
// in a future major release (see CHANGELOG).
func DisableValidateNaNInf() Option { return WithNoValidateNaNInf() }

// WithThreshold sets threshold for ToGraph export.
// Deprecated: Use WithEdgeThreshold.
func WithThreshold(t float64) Option { return WithEdgeThreshold(t) }

// WithBinary forces unit weights on export.
// Deprecated: Use WithBinaryWeights.
func WithBinary() Option { return WithBinaryWeights() }

// --------------------------- Option Resolution ---------------------------

// NewMatrixOptions resolves option setters against documented defaults.
// Implementation:
// - Delegates to gatherOptions.
// - Does not add alternative defaults or conflict rules.
//
// Inputs:
// - opts: zero or more Option values.
//
// Returns:
// - Options: finalized policy on success.
// - error: nil on success, otherwise sentinel-preserving failure.
//
// Errors:
// - ErrNilOption: nil option slot.
// - ErrNaNInf / ErrOutOfRange: invalid numeric option arguments.
//
// Determinism:
// - Same as gatherOptions.
//
// Complexity:
// - Time O(len(opts)), Space O(1).
//
// Notes:
//   - Most public entry points accept ...Option and call gatherOptions.
//   - The resulting Options is intended for internal consumption by builders.
//
// AI-Hints:
//   - Callers must check the returned error; no panic/recover workflow is part
//     of the public contract.
//   - Compose options close to the adapter call-site for clarity.
//   - Group related flags together (e.g., Directed + DisallowMulti).
func NewMatrixOptions(opts ...Option) (Options, error) {
	return gatherOptions(opts...)
}

// gatherOptions applies user-provided Option setters on top of defaults and
// finalizes derived invariants (e.g., export 'undirected' = !build.directed).
// This is the canonical internal entry in api/impl layers.
//   - Builds the canonical runtime policy used by matrix facades and kernels.
//   - Prevents duplicated option defaults, conflict rules, and validation branches.
//
// Implementation:
//   - Stage 1: Install default policy.
//   - Stage 2: Reject nil Option values before invocation.
//   - Stage 3: Apply all options left-to-right.
//   - Stage 4: Finalize derived invariants once.
//   - Stage 5: Return the frozen policy.
//
// Behavior highlights:
//   - No panic path for ordinary public input.
//   - The first invalid option aborts assembly.
//   - On error, the returned Options value must be ignored.
//
// Inputs:
//   - user: zero or more public Option values.
//
// Returns:
//   - Options: finalized policy on success.
//   - error: nil on success, otherwise sentinel-preserving failure.
//
// Errors:
//   - ErrNilOption: if any option value is nil.
//   - ErrNaNInf / ErrOutOfRange: from numeric option setters.
//
// Determinism:
//   - Stable left-to-right application.
//   - Stable last-writer-wins for directly conflicting setters, except derived
//     invariants enforced by finalizeOptions.
//
// Complexity:
//   - Time O(len(user)), Space O(1).
//
// AI-Hints:
//   - All public matrix functions accepting opts ...Option should call gatherOptions
//     exactly once near the top of the facade.
func gatherOptions(user ...Option) (Options, error) {
	o := Options{
		// numeric policy
		eps:               DefaultEpsilon,
		validateNaNInf:    DefaultValidateNaNInf,
		allowInfDistances: DefaultAllowInfDistances,

		// build policy
		directed:    DefaultDirected,
		allowMulti:  DefaultAllowMulti,
		allowLoops:  DefaultAllowLoops,
		weighted:    DefaultWeighted,
		metricClose: DefaultMetricClosure,

		preserveZeroWeights: DefaultPreserveZeroWeights,

		// export policy
		edgeThreshold: DefaultEdgeThreshold,
		keepWeights:   DefaultKeepWeights,
		binaryWeights: DefaultBinaryWeights,
	}

	for i, opt := range user {
		if opt == nil {
			return Options{}, fmt.Errorf("matrix: option #%d: %w", i, ErrNilOption)
		}
		if err := opt(&o); err != nil { // apply in order; last-writer-wins semantics
			return Options{}, err
		}
	}

	finalizeOptions(&o)
	return o, nil
}

// finalizeOptions enforces derived invariants in exactly one place.
//
// Implementation:
//   - Metric closure requires +Inf as "no path", so it forces allowInfDistances.
//   - binaryWeights and keepWeights are mutually exclusive.
//   - Last explicit writer is preserved by the setters themselves; this function
//     only enforces the final invariant.
//
// Behavior highlights:
//   - No validation.
//   - No allocation.
//   - No panic.
//   - Does not silently enable metric closure or binary mode.
//
// Inputs:
//   - o: pointer to Options to normalize.
//
// Returns:
//   - None (mutates *o).
//
// Errors:
//   - None (invariants are internal; option constructors handle validation/panics).
//
// Determinism:
//   - Deterministic for a fixed o state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This function MUST be called after applying all Option setters.
//
// AI-Hints:
//   - If you add a new option that implies others, encode that implication here (single source of truth).
//   - Do not duplicate this logic inside builders.
//   - Do not remove the keepWeights/binaryWeights reconciliation branch.
func finalizeOptions(o *Options) {
	// If metric closure is requested, the distance policy must allow +Inf sentinels.
	if o.metricClose {
		o.allowInfDistances = true
	}

	// Zero-weight preservation is only meaningful in weighted adjacency.
	// If a later option forced unweighted mode, clear the preservation flag.
	if !o.weighted {
		o.preserveZeroWeights = false
	}

	// Export weight mode must be internally consistent.
	if o.binaryWeights {
		o.keepWeights = false
	} else if o.keepWeights {
		o.binaryWeights = false
	} else {
		// Defensive normalization: ensure a stable default.
		o.keepWeights = true
		o.binaryWeights = false
	}
}

// SVDOptions controls cyclic one-sided Jacobi convergence for economy SVD.
// A zero-valued options instance selects 100 full sweeps and correlation
// tolerance 8*2^-52; both parameters affect iteration, never numeric rank.
//
// Implementation:
//   - Stage 1: validate explicit values before copying the input matrix.
//   - Stage 2: resolve zero values to their documented numeric defaults.
//   - Stage 3: pass a frozen value-copy policy to the single Jacobi kernel.
//
// Behavior highlights:
//   - MaxIterations limits *completed full sweeps*, not rotations.
//   - Tolerance is a relative normalized column correlation, not a spectral cutoff.
//
// Inputs: Both fields are caller-owned value types, never retained by reference.
// Returns: Canonical SVD copies and finalizes the policy for each invocation.
// Errors: ErrInvalidOption on negative budget or tolerance outside [0,1).
// Determinism: Default normalization and pair order depend only on field values.
// Complexity: Time O(1), Space O(1).
// Notes: Zero budget input means the default 100, not an explicit no-work mode.
// AI-Hints: Do not use Tolerance to truncate singular values or alter Rank;
// callers wanting a different rank policy should inspect SingularValues separately.
type SVDOptions struct {
	// MaxIterations is the maximum full sweep count. Zero selects 100 sweeps;
	// a negative value is invalid. One sweep checks every unordered pair once.
	MaxIterations int

	// Tolerance bounds absolute normalized cross-column correlations after a
	// sweep. Zero selects 8*2^-52; positive values must be finite and <1.
	Tolerance float64
}

// FactorOptions configures numerical rank selection and condition rejection.
// Its zero value uses a dimension-scaled machine-epsilon rank tolerance without
// imposing a hard condition-number limit on otherwise full-rank inputs.
// Implementation:
//   - RankTolerance is validated before any matrix input is inspected.
//   - ConditionLimit is validated and applied to a measured 1-norm condition estimate.
//
// Behavior highlights:
//   - RankTolerance=0 selects 2^-52*max(rows,cols); positive values are relative.
//   - ConditionLimit=0 disables rejection; values >=1 reject estimates strictly above the limit.
//
// Inputs:
//   - RankTolerance: finite, nonnegative, dimensionless relative diagonal cutoff.
//   - ConditionLimit: finite 0 or >=1, dimensionless diagnostic threshold.
//
// Returns: A value-only immutable policy snapshot stored by each factor.
// Errors: ErrInvalidOption for NaN, infinity, negative tolerance, or a limit in (0,1).
// Determinism: No implicit state, randomness, or environment-dependent options.
// Complexity: O(1) validation time and storage.
// Notes: Numerical rank is a scale-relative heuristic, not algebraic rank certification.
// AI-Hints: Never interpret ConditionLimit as an independently certified bound on kappa_2.
type FactorOptions struct {
	// RankTolerance is a relative cutoff for the largest absolute diagonal
	// of the pivoted U or R. Equality to the cutoff counts as rank loss.
	RankTolerance float64

	// ConditionLimit optionally rejects the measured 1-norm condition number.
	// The comparison is strictly greater, so equality remains admissible.
	ConditionLimit float64
}
