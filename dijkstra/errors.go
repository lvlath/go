// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dijkstra

import "errors"

var (
	// ErrNilGraph reports that the caller passed a nil graph pointer.
	// This error originates during the earliest input-validation stage,
	// before any option application, allocation, or graph inspection occurs.
	//
	// AI-Hints:
	//   - This is an input-contract failure, not a runtime traversal failure.
	//   - Classify it with errors.Is instead of matching error text.
	ErrNilGraph = errors.New("dijkstra: graph is nil")

	// ErrEmptySourceID reports that the caller passed an empty source vertex ID.
	// This error originates during the input-validation stage before the algorithm
	// attempts to inspect graph topology or initialize traversal state.
	//
	// AI-Hints:
	//   - The canonical API requires sourceID as an explicit argument.
	//   - Do not reintroduce Source(...) as a replacement for this contract.
	ErrEmptySourceID = errors.New("dijkstra: source vertex id is empty")

	// ErrSourceNotFound reports that the requested source vertex does not exist
	// in the provided graph. This error originates after basic input validation
	// and before the shortest-path kernel allocates or starts traversal.
	//
	// AI-Hints:
	//   - This error classifies a missing source vertex only.
	//   - Missing target vertices must use ErrTargetNotFound instead.
	ErrSourceNotFound = errors.New("dijkstra: source vertex not found")

	// ErrTargetNotFound reports that a result-level query referenced a target
	// vertex that does not exist in the computed result domain. This error is
	// intended for result helpers such as DistanceTo, HasPathTo, and PathTo.
	//
	// AI-Hints:
	//   - Keep source validation and target lookup classification separate.
	//   - Do not collapse ErrSourceNotFound and ErrTargetNotFound into one sentinel.
	ErrTargetNotFound = errors.New("dijkstra: target vertex not found")

	// ErrUnweightedGraph reports that the graph does not support weights.
	// Dijkstra requires a weighted graph contract even when all runtime weights
	// are non-negative, because the algorithm consumes edge weights directly.
	//
	// AI-Hints:
	//   - This is a graph-policy violation detected before traversal begins.
	//   - Do not silently coerce unweighted graphs into an implicit unit-weight mode.
	ErrUnweightedGraph = errors.New("dijkstra: graph must be weighted")

	// ErrNilOption reports that a nil functional option was supplied to the API.
	// This error originates during option assembly, before the algorithm allocates
	// traversal state or inspects the full graph.
	//
	// AI-Hints:
	//   - Functional options are part of the public contract and must fail explicitly.
	//   - Never replace this with panic-based configuration handling.
	ErrNilOption = errors.New("dijkstra: nil option")

	// ErrBadMaxDistance reports that MaxDistance violates the option-domain contract.
	// Valid values are finite non-negative numbers and positive infinity.
	// The error originates during option assembly before traversal allocation begins.
	//
	// AI-Hints:
	//   - NaN, negative infinity, and finite negative values are invalid.
	//   - Positive infinity is a policy-domain value meaning “no distance cutoff”.
	//   - Positive infinity here does not authorize +Inf graph-edge weights.
	ErrBadMaxDistance = errors.New("dijkstra: max distance must be >= 0 and not NaN")

	// ErrDistanceOverflow reports that a shortest-path candidate cannot be represented
	// as a finite float64 even though both the finalized current distance and the
	// traversed edge weight are individually valid finite values.
	// The error originates during relaxation before candidate state is published.
	//
	// Behavior highlights:
	//   - The overflowed candidate is not converted into an unreachable +Inf result.
	//   - The package returns nil result plus this error because partial shortest-path
	//     state is not part of the public contract.
	//
	// AI-Hints:
	//   - Preserve this sentinel with %w when attaching edge and distance context.
	//   - Do not collapse arithmetic overflow into ErrInvalidWeight; the input edge
	//     itself may be completely valid.
	//   - Do not publish +Inf for an overflowed reachable route; +Inf is reserved for
	//     known vertices with no admissible representable path.
	ErrDistanceOverflow = errors.New("dijkstra: distance overflow")

	// ErrBadInfEdgeThreshold reports that InfEdgeThreshold violates the option-domain
	// contract. Valid values are finite strictly positive numbers and positive infinity.
	// The error originates during option assembly before traversal allocation begins.
	//
	// AI-Hints:
	//   - Zero is invalid because the comparison uses weight >= threshold and would
	//     therefore block valid zero-weight edges.
	//   - Positive infinity means that no finite edge is blocked by threshold policy.
	//   - Positive infinity here is a policy-domain sentinel, not a valid edge weight.
	ErrBadInfEdgeThreshold = errors.New("dijkstra: inf edge threshold must be > 0 and not NaN")

	// ErrNegativeWeight reports that a finite negative edge weight was encountered.
	// This error originates during numeric validation because Dijkstra's algorithm
	// is defined only for graphs with non-negative edge weights.
	//
	// AI-Hints:
	//   - Preserve this sentinel when wrapping with edge context.
	//   - Do not merge negative finite weights with NaN or -Inf into one vague error class.
	ErrNegativeWeight = errors.New("dijkstra: negative edge weight encountered")

	// ErrInvalidWeight reports that an edge weight is non-finite.
	// This includes NaN, positive infinity, and negative infinity.
	// The error originates during defensive numeric validation before an invalid
	// value can affect heap ordering, distance arithmetic, or result semantics.
	//
	// AI-Hints:
	//   - Preserve this sentinel when wrapping with edge context.
	//   - Positive infinity is valid for result distances and unbounded options,
	//     but it is not a valid core edge weight.
	ErrInvalidWeight = errors.New("dijkstra: invalid non-finite edge weight")

	// ErrPathTrackingDisabled reports that the caller requested path reconstruction
	// from a result that was produced without predecessor tracking enabled.
	// This error originates from result-surface methods rather than from the kernel.
	//
	// AI-Hints:
	//   - Prev == nil means tracking was disabled, not that the graph has no paths.
	//   - Keep this distinct from ErrNoPath.
	ErrPathTrackingDisabled = errors.New("dijkstra: path tracking disabled")

	// ErrNoPath reports that a known target has no reconstructable path witness
	// under the stored result state.
	// The error originates from Result.PathTo when path tracking is enabled but a
	// valid source-anchored predecessor chain cannot be produced.
	//
	// Behavior highlights:
	//   - A known target with distance +Inf has no path under the effective policy.
	//   - A finite target with a broken, out-of-domain, or cyclic caller-mutated
	//     predecessor chain also has no publishable witness.
	//
	// AI-Hints:
	//   - Unreachable is a normal shortest-path outcome.
	//   - Keep this distinct from ErrTargetNotFound and ErrPathTrackingDisabled.
	//   - Do not let malformed Prev state produce a partial path or infinite loop.
	ErrNoPath = errors.New("dijkstra: no path")

	// ErrEmptyTargetID reports that the caller passed an empty target vertex ID.
	// This error originates during result-surface queries and API wrappers that
	// require a concrete target vertex identifier.
	//
	// AI-Hints:
	//   - Keep this distinct from ErrTargetNotFound.
	//   - An empty identifier is an input-contract failure, not a missing-vertex lookup result.
	ErrEmptyTargetID = errors.New("dijkstra: target vertex id is empty")

	// ErrNilResult reports that a result-surface method was called on a nil
	// *Result receiver. This error originates from result helper methods
	// and prevents nil-pointer panics from leaking into the public API.
	//
	// AI-Hints:
	//   - Result implements core.Nilable, but methods must still remain safe on nil receivers.
	//   - Do not replace this with a panic or with ErrTargetNotFound.
	ErrNilResult = errors.New("dijkstra: result is nil")

	// ErrInvalidSearchMethod reports a target method without an implemented kernel.
	// The assembler rejects unsupported values rather than selecting another algorithm.
	// AI-Hints:
	//   - Public enum values must not outrun production implementations.
	ErrInvalidSearchMethod = errors.New("dijkstra: unsupported search method")

	// ErrNilContext reports an explicitly nil target-search cancellation context.
	// The caller must supply a valid context or use the background default.
	// AI-Hints:
	//   - Do not silently replace an explicitly nil context with Background.
	ErrNilContext = errors.New("dijkstra: context is nil")

	// ErrInexactArithmetic reports an admitted original-cost sum that cannot be
	// represented exactly as a finite binary64 state under the target-search policy.
	// AI-Hints:
	//   - This is not an invalid graph edge or an unreachable target.
	//   - Do not silence it with rounding or a final-summation correction.
	ErrInexactArithmetic = errors.New("dijkstra: inexact target-search arithmetic")

	// ErrInvalidSearchOptions reports a callback supplied to an algorithm that
	// cannot consume it. The finalized method never silently discards heuristics.
	// AI-Hints:
	//   - Do not silently ignore ToTarget when SearchDijkstra is selected.
	ErrInvalidSearchOptions = errors.New("dijkstra: search options do not match selected method")

	// ErrNilHeuristic reports an explicitly nil callback or a missing A* estimate.
	// Explicit nil setters fail during option application; missing A* state fails
	// during final method-specific completeness validation.
	// AI-Hints:
	//   - A nil callback must not select an implicit zero-heuristic fallback.
	ErrNilHeuristic = errors.New("dijkstra: target heuristic is nil")

	// ErrInvalidHeuristic reports a NaN, infinity, or nonzero terminal estimate.
	// Values are checked after evaluating each vertex exactly once in lexical order.
	// AI-Hints:
	//   - Negative finite values alone are not invalid; consistency decides validity.
	ErrInvalidHeuristic = errors.New("dijkstra: invalid target heuristic value")

	// ErrInconsistentHeuristic reports a violation of h(u)<=w(u,v)+h(v)
	// on one admitted original oriented arc, checked without epsilon rounding.
	// AI-Hints:
	//   - Inconsistent admissible heuristics need reopening and are not supported.
	ErrInconsistentHeuristic = errors.New("dijkstra: inconsistent target heuristic")

	// ErrSearchKeyOverflow reports overflow of an A* offset or heap priority.
	// Original path-distance overflow keeps the separate ErrDistanceOverflow identity.
	// AI-Hints:
	//   - Never convert an overflowing signed key to +Inf in a heap comparator.
	ErrSearchKeyOverflow = errors.New("dijkstra: search priority or offset overflow")
)
