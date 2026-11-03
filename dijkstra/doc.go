// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dijkstra implements deterministic non-negative weighted shortest paths.
// It offers full single-source Dijkstra and a family of target-directed Dijkstra/A* methods.
//
// -----------------------------------------------------------------------------
// -- WHAT ---------------------------------------------------------------------
//
// dijkstra provides two deliberately distinct result domains:
// full single-source Dijkstra and certified source-to-target route queries.
// The single-source family includes:
//
//   - Dijkstra(g, sourceID, opts...)
//     Runs deterministic single-source shortest paths and returns a detached
//     Result.
//
//   - Distances(g, sourceID, opts...)
//     Convenience wrapper that returns a detached distance map only.
//
//   - DistanceTo(g, sourceID, targetID, opts...)
//     Convenience wrapper that runs Dijkstra and resolves one target distance.
//
//   - ShortestPathTo(g, sourceID, targetID, opts...)
//     Convenience wrapper that runs Dijkstra with path tracking enabled and
//     returns one deterministic shortest-path witness plus its distance.
//
// The target-directed family shares one Search facade and three wrappers:
//
//   - Search(g, sourceID, targetID, opts...)
//     Runs a selected, implemented SearchMethod and returns a detached PathResult.
//
//   - Bidirectional(g, sourceID, targetID, opts...)
//     Runs two-frontier Dijkstra with an exact stopping certificate.
//
//   - AStar(g, sourceID, targetID, toTarget, opts...)
//     Runs consistent-heuristic A* using the canonical forward label-setting loop.
//
//   - BidirectionalAStar(g, sourceID, targetID, toTarget, fromSource, opts...)
//     Runs balanced-potential A* using the canonical bidirectional engine.
//
// SearchMethod selects exactly one implemented kernel: SearchDijkstra,
// SearchBidirectional, SearchAStar, or SearchBidirectionalAStar.
// SearchOptions is intentionally separate from legacy Options; SearchOption
// assembly requires heuristic completeness for the selected method.
//
// Result is the full-SSSP result artifact. It exposes:
//
//   - SourceID  - the source vertex identifier used for the run.
//   - Distances - finalized shortest-path distances for the known result domain.
//   - Prev      - optional predecessor map used for path reconstruction.
//
// PathResult is the target-only artifact. It exposes SourceID, TargetID,
// Reachable, finite original-cost Distance (or +Inf when unreachable), Method,
// policy metadata, and ordered original Vertices/EdgeIDs of exactly one witness.
// Reachable target paths always include their witness; there is no TrackPaths
// option in SearchOptions. PathResult.Path and Clone return detached copies.
//
// The package is designed as a reusable weighted shortest-path kernel for
// downstream algorithms that require deterministic behavior, explicit numeric
// policy, errors.Is-compatible failure matching, and detached caller-owned
// result data.
//
// -----------------------------------------------------------------------------
// -- WHY ----------------------------------------------------------------------
//
//   - Determinism First:
//     Legacy Dijkstra follows core.Neighbors ordering and breaks equal distance
//     heap keys by vertex ID. Target Search enumerates sorted original edge IDs
//     and orders heap entries by (priority, distance, vertex index, sequence).
//
//   - Correct Weighted Semantics:
//     Distances are true weighted shortest-path costs over non-negative edges.
//     The package intentionally rejects invalid numeric input and negative costs.
//
//   - Explicit Result Surface:
//     Distances, reachability, and one shortest-path witness are exposed through
//     Result instead of ad-hoc parallel maps returned from the kernel.
//
//   - Policy Without Topology Mutation:
//     Runtime limits never rewrite caller topology. Target Search clones the
//     graph once before preparing admitted original oriented arcs; legacy SSSP
//     traverses its normal core graph surfaces directly.
//
//   - Detached Ownership:
//     Published results are detached from the graph and belong to the caller
//     after return.
//
// -----------------------------------------------------------------------------
// -- DOMAIN SCOPE -------------------------------------------------------------
//
// dijkstra is intentionally scoped to:
//
//   - deterministic full single-source and one-target shortest paths,
//   - one-directional A* with a consistent target heuristic,
//   - balanced two-frontier A* with consistent directional heuristics,
//   - weighted graphs with non-negative finite edge costs,
//   - deterministic path reconstruction of one shortest-path witness when
//     path tracking is enabled,
//   - optional traversal policy through distance cutoff and wall threshold.
//
// The package answers questions such as:
//
//   - “What is the minimum total cost from this source to every known vertex?”
//   - “What is the cost to this specific target?”
//   - “Is this known target reachable under the effective traversal policy?”
//   - “What is one deterministic shortest-path witness to this target?”
//
// -----------------------------------------------------------------------------
// -- NON-GOALS ----------------------------------------------------------------
//
// dijkstra is intentionally NOT:
//
//   - a negative-weight shortest-path package,
//   - an all-pairs shortest-path package,
//   - a guarantee that legacy SSSP observes one atomic topology snapshot,
//   - a guarantee for callers mutating live Edge pointers without synchronization,
//   - an exhaustive all-shortest-path reconstruction package.
//
// Consequences:
//
//   - If any traversed edge may have a negative finite weight, Dijkstra is
//     mathematically the wrong tool and the package rejects that input.
//   - If you need all-pairs shortest paths, this package does not provide
//     Floyd-Warshall-, Johnson-, or repeated-source orchestration.
//   - Target Search obtains one core.Clone before validation, policy admission,
//     heuristic evaluation and traversal; legacy SSSP does not use that snapshot.
//     Synchronize caller-owned direct Edge pointer mutations in either case.
//   - If you need every shortest witness rather than one deterministic witness,
//     Prev and PathTo are insufficient by design.
//
// -----------------------------------------------------------------------------
// -- NUMERIC POLICY -----------------------------------------------------------
//
// Numeric semantics are separated by domain and form part of the package contract.
//
//  1. Graph-edge input domain
//
//     Every consumed Edge.Weight must be a finite float64.
//
//     Accepted:
//
//     - finite w >= 0
//
//     Rejected:
//
//     - math.NaN()
//     - math.Inf(1)
//     - math.Inf(-1)
//     - every finite w < 0
//
//  2. Runtime-policy domain
//
//     Positive infinity is valid for:
//
//     - MaxDistance, where +Inf means “no accumulated-distance cutoff”;
//     - InfEdgeThreshold, where +Inf means “no finite edge is blocked”.
//
//     These option-domain meanings do not authorize +Inf graph-edge weights.
//
//  3. Published result domain
//
//     Result.Distances contains:
//
//     - finite non-negative shortest-path distances for reachable vertices;
//     - math.Inf(1) for known vertices that are unreachable under the effective
//     graph and traversal policy.
//
//     A missing target is not represented numerically. It is classified through
//     ErrTargetNotFound.
//
// Edge-weight classification law:
//
//   - math.IsNaN(w)     -> ErrInvalidWeight
//   - math.IsInf(w, 0)  -> ErrInvalidWeight
//   - finite w < 0      -> ErrNegativeWeight
//   - finite w >= 0     -> accepted
//
// Arithmetic law (method-specific):
//
//   - Legacy SSSP uses its established rounded float64 relaxation and cutoff
//     semantics. A candidate that overflows its admitted domain fails with
//     ErrDistanceOverflow rather than becoming unreachable.
//   - Target Search compares exact represented-binary sums against a finite
//     MaxDistance. Candidates above that cutoff are excluded before required
//     storage checks. Admitted costs must be exactly representable as float64:
//     a rounded finite but inexact cost fails with ErrInexactArithmetic.
//   - Target A* and balanced A* additionally require exact finite normalized
//     heap priorities; a nonrepresentable key fails with ErrInexactArithmetic
//     or ErrSearchKeyOverflow. Original path costs remain distinct from keys.
//   - All numeric failures return nil result plus error, not +Inf or a partial path.
//
// Published-distance law:
//
//   - A known but unreachable vertex remains present in Distances with +Inf.
//   - A known reachable vertex has a finite non-negative distance.
//   - An unknown target is classified through ErrTargetNotFound.
//   - +Inf in Distances is package-owned result state; it is not copied from a
//     graph-edge weight.
//   - Target PathResult.Distance uses the same unreachable +Inf meaning;
//     its public witness is nil when Reachable is false.
//
// -----------------------------------------------------------------------------
// -- GRAPH POLICY -------------------------------------------------------------
//
// dijkstra runs over *core.Graph and requires weighted graph semantics.
//
// Graph policy:
//
//   - g must be non-nil.
//   - g must be weighted.
//   - directed graphs are supported.
//   - undirected graphs are supported.
//   - mixed-edge graphs are supported.
//
// Legacy SSSP takes live graph relations from g.Neighbors(u). Target Search
// instead builds one isolated searchGraph from core.Clone, preserving every
// original Edge.ID and creating oriented arcs only for admitted finite edges.
// Its out/in indexes are shared by forward and backward search; a backward
// search never manufactures a reversed original directed edge.
//
// Endpoint interpretation law:
//
//   - dijkstra does not assume that the traversable neighbor is always edge.To.
//   - For undirected and mixed relations, the effective opposite endpoint is
//     resolved relative to the current vertex.
//   - Directed reverse traversal is forbidden unless the original arc permits it.
//   - On target Search, InfEdgeThreshold is applied after global weight
//     validation; every blocked edge is still checked for invalid numbers.
//
// -----------------------------------------------------------------------------
// -- DETERMINISM LAW ----------------------------------------------------------
//
// Determinism is a package-level contract. The rules below describe legacy
// full SSSP; target Search has a different, explicit prepared-order contract.
//
//  1. Vertex-domain initialization
//     The kernel initializes the known result domain by iterating:
//
//     g.Vertices()
//
//     as surfaced by core.
//
//  2. Relation enumeration
//     Relaxation processes candidate relations in the exact order returned by:
//
//     g.Neighbors(u)
//
//     dijkstra does not inject a second hidden sorting layer for neighbors.
//
//  3. Heap ordering
//     Equal-distance frontier items are ordered by:
//
//     (candidateDistance, vertexID)
//
//     so equal candidate costs are broken by lexicographically smaller vertex IDs.
//
//  4. Predecessor updates
//     Relaxation uses strict improvement only.
//
//     - candidate < currentDistance  -> update
//     - candidate == currentDistance -> do not overwrite predecessor state
//
// Together, these rules make published distances, predecessor selection, and
// reconstructed path witnesses stable for the same graph state, sourceID, and
// runtime policy.
//
// Target Search determinism:
//
//   - One snapshot; vertex IDs and original edges are sorted lexicographically.
//   - Undirected arcs preserve one Edge.ID in both permitted orientations.
//   - Heap order is (priority, original distance, dense lexical vertex, sequence).
//   - All parent links update on strict improvement only; ties keep the first.
//   - Bidirectional ties between the two clean minimum keys expand forward first.
//   - Frozen heuristic callbacks run once per vertex in lexical ID order:
//     for balanced A*, all to-target callbacks precede all from-source callbacks.
//   - Selected PathResult.Method reports the executed algorithm, never a fallback.
//
// -----------------------------------------------------------------------------
// -- RESULT CONTRACT ----------------------------------------------------------
//
// Result is the detached public full-SSSP artifact.
//
// Public result semantics:
//
//   - SourceID
//     The explicit source vertex identifier used for the run.
//
//   - Distances
//     Finalized shortest-path distances over the known result domain.
//     Known but unreachable vertices remain present with value +Inf.
//
//   - Prev
//     Optional predecessor map used for path reconstruction.
//     Prev == nil means path tracking was disabled, not “no path exists”.
//
// Result-surface laws:
//
//   - DistanceTo(targetID)
//     Returns the stored distance when the target is known.
//     Returns ErrTargetNotFound only when the target is absent from the result domain.
//
//   - HasPathTo(targetID)
//     Returns false with nil error for a known target whose distance is +Inf.
//
//   - PathTo(targetID)
//     Requires path tracking to have been enabled.
//     Returns one deterministic shortest-path witness.
//     Validates every predecessor against the result domain.
//     Rejects broken, cyclic, and out-of-domain predecessor state with ErrNoPath.
//     Never returns a partial witness and never loops indefinitely on malformed Prev.
//     Does not enumerate all shortest paths.
//
// Target PathResult laws:
//
//   - Reachable=true: Vertices[0]=SourceID, last vertex=TargetID,
//     len(EdgeIDs)+1=len(Vertices), finite Distance, and original directed
//     or undirected edges matching every source-to-target transition.
//   - Reachable=false: Distance=+Inf and both witness slices nil, with nil error.
//   - Source==target: one vertex, zero distance, nil EdgeIDs after validation.
//   - A missing target is ErrTargetNotFound; unreachable known target is not an error.
//   - PathResult.Path returns independent vertex and edge ID slices.
//   - PathResult.Clone deep-copies both public slices and preserves nil slices.
//   - Every error returns nil *PathResult, even if a tentative route exists.
//
// -----------------------------------------------------------------------------
// -- OPTIONS GOVERNANCE -------------------------------------------------------
//
// Options are explicit runtime policy inputs, not hidden mutable state.
//
// Dijkstra option law:
//
//   - options are applied sequentially,
//   - last-writer-wins per field,
//   - nil options are invalid,
//   - invalid explicit option input fails before traversal begins.
//
// Public runtime policy includes:
//
//   - WithPathTracking()
//     Enables predecessor tracking for later path reconstruction.
//
//   - WithMaxDistance(max)
//     Limits exploration to shortest paths whose distance does not exceed max.
//
//   - WithInfEdgeThreshold(threshold)
//     Treats edges with weight >= threshold as impassable walls.
//
// Baseline default policy:
//
//   - TrackPaths       = false
//   - MaxDistance      = +Inf
//   - InfEdgeThreshold = +Inf
//
// Target SearchOptions are a separate options family:
//
//   - WithSearchMethod(SearchDijkstra | SearchBidirectional | SearchAStar |
//     SearchBidirectionalAStar) selects only an implemented kernel.
//   - WithSearchContext(ctx) supplies a non-nil cancellation context.
//   - WithSearchMaxDistance(max) and WithSearchInfEdgeThreshold(wall) retain
//     the same policy domains but exact target-search cutoff semantics.
//   - WithTraversalPolicy(Options) copies the two traversal scalars only;
//     it does not copy TrackPaths, because target witnesses are mandatory.
//   - WithHeuristic(toTarget) provides one A* callback without selecting a method.
//   - WithBalancedHeuristics(toTarget, fromSource) sets both directional callbacks.
//   - DefaultSearchOptions selects SearchDijkstra with no finite cutoff or wall.
//   - Explicit nil callback setters fail immediately. Method-specific completeness
//     and absence of unused heuristics are verified after all valid options.
//
// Important separation:
//
//   - sourceID is not an option.
//   - sourceID is an explicit required input to the public API.
//   - Legacy Option and target SearchOption are not interchangeable.
//
// -----------------------------------------------------------------------------
// -- WALL AND CUTOFF POLICY ---------------------------------------------------
//
// dijkstra exposes two traversal-policy gates that are distinct from graph
// validity and distinct from each other.
//
//  1. InfEdgeThreshold
//     Edges with weight >= InfEdgeThreshold are skipped as impassable walls.
//
//     Consequence:
//     a target may remain known but unreachable because all admissible routes
//     to it are blocked by threshold policy.
//
//  2. MaxDistance
//     Candidate paths with distance > MaxDistance are not explored.
//
//     Consequence:
//     a graph path may exist mathematically while the published result still
//     contains +Inf for that target because traversal policy disallowed it.
//
// These are traversal policies, not numeric-validation failures.
// Target Search additionally checks the exact binary sum before admitting a
// candidate under a finite MaxDistance. In contrast, the legacy SSSP kernel
// retains its rounded-candidate cutoff policy for compatibility.
//
// -----------------------------------------------------------------------------
// -- ERROR LAW ----------------------------------------------------------------
//
// Exported package-level sentinels are the single source of truth for protocol
// matching. Callers must compare via errors.Is, never by string.
//
// Primary sentinels include:
//
//   - ErrNilGraph
//   - ErrEmptySourceID
//   - ErrSourceNotFound
//   - ErrTargetNotFound
//   - ErrUnweightedGraph
//   - ErrNilOption
//   - ErrBadMaxDistance
//   - ErrBadInfEdgeThreshold
//   - ErrNegativeWeight
//   - ErrInvalidWeight
//   - ErrDistanceOverflow
//   - ErrPathTrackingDisabled
//   - ErrNoPath
//   - ErrEmptyTargetID
//   - ErrNilResult
//   - ErrInvalidSearchMethod
//   - ErrNilContext
//   - ErrInexactArithmetic
//   - ErrInvalidSearchOptions
//   - ErrNilHeuristic
//   - ErrInvalidHeuristic
//   - ErrInconsistentHeuristic
//   - ErrSearchKeyOverflow
//
// The Search error classes distinguish invalid original weights, unrepresentable
// original costs, inexact/overflowing heuristic keys, inconsistent potentials,
// invalid option combinations, and canceled contexts with preserved causes.
//
// Wrapping law:
//
//   - When runtime numeric validation adds edge context, the sentinel must be
//     preserved with %w so callers can still match via errors.Is.
//   - Graph-surface runtime failures are wrapped with context and preserved as
//     ordinary Go wrapped errors.
//
// Panic policy:
//
//   - Public option validation is error-returning.
//   - Panic-based option governance is forbidden.
//
// -----------------------------------------------------------------------------
// -- OWNERSHIP LAW ------------------------------------------------------------
//
// Published results are detached and caller-owned.
//
// Ownership contract:
//
//   - The package does not retain a live mutable link from Result back
//     to the graph.
//   - After return, callers may read, clone, and transform published results.
//   - Convenience wrappers that return maps publish detached caller-owned copies.
//
// Caller-mutation consequence:
//
//   - Distances and Prev are caller-owned and therefore mutable after return.
//   - Mutating them changes the meaning of that Result instance.
//   - PathTo validates witness-chain safety and fails with ErrNoPath when mutation
//     leaves the predecessor relation broken, cyclic, or outside Distances.
//   - Clone creates independent maps but does not validate semantic correctness.
//   - PathResult owns its witness and carries no live graph or heuristic callback.
//     The caller may mutate its exported slices, potentially invalidating them.
//
// Clone law:
//
//   - Result.Clone returns a deep copy.
//   - Prev == nil is preserved exactly and is not rewritten into an empty map.
//
// -----------------------------------------------------------------------------
// -- PARTIAL RESULT LAW -------------------------------------------------------
//
// On failure, neither the full-SSSP Result nor target PathResult is published.
//
// Publication law:
//
//   - validation failures return nil result plus error,
//   - runtime failures during kernel execution return nil result plus error,
//   - detached results are published only after the run completes successfully.
//
// This prevents callers from accidentally depending on incomplete shortest-path
// state as if it were finalized output.
//
// -----------------------------------------------------------------------------
// -- CONCURRENCY LAW ----------------------------------------------------------
//
// Legacy SSSP uses the graph's live public surfaces; callers should keep
// topology stable during that run to preserve a coherent problem instance.
//
// Target Search calls core.Clone once, then prepares all admitted vertex/edge
// and oriented-arc state from that clone. It does not read the original graph
// again during any shortest-path kernel. Core-owned mutation through its proper
// synchronized methods may overlap with Clone according to core's lock model;
// direct unsynchronized writes through retained *core.Edge pointers are not safe.
//
// Context cancellation is checked during search preparation, heuristic callbacks,
// arc expansion and witness publication. Synchronous Clone or a blocking user
// callback cannot be forcibly interrupted by the context.
//
// -----------------------------------------------------------------------------
// -- COMPLEXITY SUMMARY -------------------------------------------------------
//
// Effective complexity depends on the graph surfaces provided by core.
//
// Public summary:
//
//   - Dijkstra / Distances / DistanceTo / ShortestPathTo:
//     Effective time O(V log V + E log V + graph-surface-enumeration cost).
//     Effective space O(V + E_heap), excluding graph-owned storage.
//   - Search / Bidirectional / AStar / BidirectionalAStar:
//     Snapshot/sort O(V log V+E log E), arc preparation O(V+A), and target
//     kernels O(V+(V+A)log(A+1)), with A<=2E oriented admitted arcs.
//     A* preparation adds O(V*HT+A), balanced A* adds O(V*(HT+HS)+A), where
//     HT/HS are callback costs. Space O(V+E+A) including one clone, arcs,
//     up to two lazy frontiers, frozen heuristics, and detached witness.
//     This is an end-to-end upper bound, not a promised speedup over Dijkstra.
//
// Result-surface summary:
//
//   - PathResult.Path / Clone:
//     Time O(k), Space O(k) over a k-vertex witness.
//
//   - DistanceTo / HasPathTo
//     Time O(1), Space O(1).
//
//   - PathTo
//     Time O(k), Space O(k), where k is the number of vertices on the returned path.
//
//   - Clone
//     Time O(V), Space O(V) over copied result entries.
//
// -----------------------------------------------------------------------------
// -- AI-HINT (LLM/Copilot/ChatGPT/Claude/Gemini/Qwen guidance) ----------------
//
//   - Error Matching:
//     Never parse error strings. Use errors.Is with exported sentinels only.
//
//   - Numeric Domain Separation:
//     Graph-edge weights must remain finite.
//     +Inf is valid only in the result domain for unreachability and in the
//     option domain for unbounded policy.
//     Do not transfer +Inf semantics implicitly between these domains.
//
//   - Arithmetic Overflow:
//     Do not publish an overflowed candidate as unreachable +Inf.
//     Preserve ErrDistanceOverflow and suppress partial result publication.
//
//   - Endpoint Law:
//     Never simplify mixed or undirected traversal to edge.To-only logic.
//     The effective opposite endpoint must be resolved relative to the current vertex.
//
//   - Determinism:
//     Do not remove the heap tie-break on vertex ID.
//     Equal-distance ordering affects predecessor stability and user-visible paths.
//
//   - Path Tracking:
//     Prev == nil means path tracking was disabled.
//     It does not mean that the graph has no reachable paths.
//
//   - Path Witness Integrity:
//     Result.Prev is caller-owned after publication.
//     Never follow predecessor links without cycle and domain guards.
//     Broken, cyclic, or out-of-domain witness state must fail with ErrNoPath.
//
//   - Numeric Domain Separation:
//     Graph-edge weights must remain finite.
//     +Inf is valid only in result and policy domains.
//
//   - Custom Options:
//     Option is publicly constructible.
//     Canonical assembly must revalidate finalized state after every option.
//
//   - Options:
//     Do not move sourceID back into options.
//     Do not reintroduce panic-based option validation.
//
//   - Result Publication:
//     Do not publish partial shortest-path state on runtime failure.
//
//   - Concurrency:
//     Legacy Dijkstra requires stable topology; target Search clones once and
//     must not be confused with direct unsynchronized Edge pointer mutation.
//
//   - A* Consistency:
//     hT(t)=0 and hT(u)<=w(u,v)+hT(v) on every effective arc are mandatory.
//     Balanced search also requires hS(s)=0 and hS(v)<=w(u,v)+hS(u).
//     Admissibility alone does not authorize a no-reopening solver.
//
//   - Balanced Normalization:
//     p(v)=(hT(v)-hS(v))/2; forward keys are dF(v)+p(v)-p(s), backward
//     keys are dB(v)+p(t)-p(v). Stop only when clean key minima satisfy
//     minKeyF+minKeyB >= mu+p(t)-p(s). mu and PathResult.Distance remain
//     original costs, never potentials or rounded reduced weights.
//
//   - Target Arithmetic:
//     Target paths require exact binary64 cost storage. A successful original
//     edge weight can still lead to ErrInexactArithmetic or ErrSearchKeyOverflow.
//     Never replace their sentinels with unreachable +Inf or epsilon comparisons.
//
//   - Scenario Executability:
//     Three standalone examples in examples/dijkstra_scenario_*.go are launched
//     by dijkstra/scenario_test.go; benchmarks in search_bench_test.go compare all four
//     target methods on one fixed directed grid and cutoff-sensitive regimes.
//
// -----------------------------------------------------------------------------
// -- SEE ALSO -----------------------------------------------------------------
//
//   - docs/DIJKSTRA.md for repository-level tutorial, formulas, diagrams,
//     algorithm notes, and operational examples.
//   - package GoDoc on Dijkstra, Result, Options, Option helpers,
//     and wrapper APIs for per-symbol contract details.
package dijkstra
