// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst builds minimum spanning structures and terminal-covering networks
// on weighted, undirected core.Graph values, and covers validated trees with walks.
//
// Domain scope:
//
//   - Strict MST: connect every vertex with exactly |V|-1 selected edges.
//   - Explicit MSF: return one minimum spanning tree per connected component.
//   - Algorithms: Kruskal and Prim for spanning all graph vertices.
//   - Graph Steiner: Kou-Markowsky-Berman (KMB) approximation for required terminals.
//   - Tree covering: closed or fixed-end minimum-distance edge-covering walk.
//   - Input model: weighted, undirected *core.Graph for MST and SteinerTree;
//     caller-owned SteinerResult for SteinerWalk.
//   - Results: Result (MST/MSF), SteinerResult (unique network), WalkResult (steps).
//
// What MST solves:
//
//   - Given an undirected weighted graph G=(V,E), a minimum spanning tree is a
//     subset T of edges that connects all vertices with minimum total finite cost:
//
//     minimize:   sum w(e), for every e in T
//     subject to: T spans V, T is acyclic, and |T| = |V| - 1
//
//     Equivalent compact form:
//
//     min_{T ⊆ E} Σ_{e ∈ T} w(e),   |T| = |V| - 1
//
//     CodeCogs view:
//     https://latex.codecogs.com/svg.image?\min_{T\subseteq%20E}\sum_{e\in%20T}w(e),\quad|T|=|V|-1
//
//   - For a disconnected graph, explicit forest mode applies the same objective
//     independently to each connected component:
//
//     MSF(G) = union of MST(C_i) for every connected component C_i.
//
//   - Graph Steiner network design requires only specified vertices T (terminals),
//     not every original graph vertex. Additional original vertices may connect
//     them more cheaply; no geometric points or synthetic edges are created:
//
//     minimize sum w(e) over connected acyclic original-edge H with T subset V(H).
//
//     KMB computes a deterministic polynomial-time approximation to this NP-hard
//     objective. In the exact nonnegative-weight model its conservative bound for
//     k=|T|>1 is cost(KMB) <= 2(1-1/k)*OPT. This is not an evaluated ratio,
//     exact optimum certificate, or universal finite-precision float64 guarantee.
//
//   - A covering walk on a validated tree must traverse every tree edge. For
//     a closed start=end walk every edge is traversed twice. For distinct fixed
//     endpoints s and t, spine edges are traversed once and other edges twice:
//
//     closed length = 2W; fixed-end length = 2W - d_tree(s,t).
//
//     Here W is the unique-edge network cost; these are exact-model identities.
//     Runtime Distance instead adds the cost of each real passage, avoiding
//     unnecessary intermediate overflow in the formula. The resulting walk is
//     optimal for covering that tree, not globally optimal in the original graph.
//
// Why MST matters:
//
//   - Network design:
//     MST gives the cheapest backbone that still connects every required site.
//     Examples include fiber rings, emergency radio relays, satellite mesh control
//     links, industrial sensor networks, and campus / data-center cabling.
//
//   - VLSI and physical design:
//     MST-style topologies are useful as a deterministic baseline for global
//     routing, clock distribution sketches, pin clustering, and low-cost net
//     connection before more specialized Steiner-tree or timing-aware refinements.
//
//   - Smart grids and green energy:
//     MST helps reason about minimum-cost connectivity among substations,
//     batteries, renewable sources, and critical loads while keeping disconnected
//     islanded microgrids explicit through forest mode.
//
//   - Big data and machine learning:
//     MST is a standard primitive for clustering, outlier analysis, graph-based
//     manifold approximations, and “cut the largest tree edges” segmentation.
//
//   - Algorithmic subroutines:
//     MST appears inside approximation algorithms, clustering pipelines,
//     graph partitioning, network sparsification, and topology preconditioners.
//
// Non-goals:
//
//   - Directed optimum branching/arborescence is not MST and is intentionally rejected.
//   - Shortest path computation is not MST; use a shortest-path package for route queries.
//   - Geometric Steiner-point placement and exact general Steiner optimization are
//     not implemented; SteinerTree uses only original vertices and edges.
//   - Steiner forest, degree/capacity-constrained spanning design, directed
//     arborescence, and globally optimal original-graph edge-covering routes
//     are not implemented by these facades.
//   - Dynamic MST maintenance is out of scope; no incremental update API exists.
//   - The package does not mutate the input graph.
//   - Matrix-backed APIs must not be documented here unless their exact public
//     signatures and sentinel laws exist in the package.
//
// Public API:
//
//   - MinimumSpanningTree(graph *core.Graph, opts ...Option) (*Result, error)
//     Canonical facade. It assembles options, validates graph policy, snapshots graph
//     data, dispatches to one algorithm, and returns a detached result artifact.
//
//   - Kruskal(graph *core.Graph) (*Result, error)
//     Focused strict-tree wrapper selecting AlgorithmKruskal.
//
//   - Prim(graph *core.Graph, root string) (*Result, error)
//     Focused strict-tree wrapper selecting AlgorithmPrim with an explicit root.
//
//   - Result
//     Canonical result artifact containing Algorithm, Mode, Root, Edges,
//     TotalWeight, VertexCount, ComponentCount, and ComponentRoots.
//
//   - SteinerTree(graph *core.Graph, terminals []string, opts ...SteinerOption)
//     (*SteinerResult, error)
//     Validates the entire nonnegative weighted input, clones topology once,
//     computes terminal-pair SSSP closure, selects its MST, expands selected
//     links to original edges, selects a second MST, prunes nonterminal leaves,
//     verifies the final tree, and publishes a checked unique-edge total.
//
//   - DefaultSteinerOptions, WithSteinerContext, WithSteinerPathStorage,
//     WithSteinerMaxVertices, WithSteinerMaxEdges, WithSteinerMaxTerminals
//     Configure cooperative cancellation, recompute/retain predecessor memory,
//     and optional original-input element-count admission caps.
//
//   - SteinerWalk(tree *SteinerResult, startID, endID string) (*WalkResult, error)
//     Validates caller-owned tree structure and canonical cost independently;
//     emits every actual original Edge.ID passage in a deterministic itinerary.
//     Equal endpoints choose the closed regime; different endpoints choose the
//     fixed-end regime. Both explicit nonempty endpoint IDs must be tree vertices.
//
// Result interpretation:
//
//   - Algorithm reports the kernel selected by the facade.
//   - Mode reports strict tree or explicit forest semantics.
//   - Root is meaningful for Prim; in forest mode it can describe the first
//     deterministic component root when no explicit root is supplied.
//   - Edges contains detached core.Edge values selected by the MST/MSF.
//   - TotalWeight is the sum of selected edge weights.
//   - VertexCount is the number of vertices in the validated snapshot.
//   - ComponentCount is 1 for a successful strict MST and can be greater than 1
//     for explicit forest mode.
//   - ComponentRoots gives deterministic public component roots.
//   - SteinerResult.Terminals is a sorted unique copy of the request;
//     Vertices includes only the chosen original network vertices; Edges contains
//     distinct original core.Edge values sorted by ID; TotalWeight is the checked
//     Edge.ID-ordered float64 left-fold of unique final selected weights.
//   - WalkResult.StartID/EndID identify mandatory endpoints; Vertices and
//     EdgeIDs are passage-aligned (len(Vertices)=len(EdgeIDs)+1). Repeated IDs
//     are valid and required when edges are traversed in both directions.
//     NetworkWeight describes unique edges; Distance sums actual walk passages.
//
// Options and regimes:
//
//   - DefaultOptions returns AlgorithmKruskal + ModeStrictTree.
//   - WithAlgorithm selects AlgorithmKruskal or AlgorithmPrim.
//   - WithRoot supplies the explicit Prim root.
//   - WithForest enables explicit minimum spanning forest mode.
//   - WithStrictTree restores strict spanning tree mode after WithForest.
//   - Option application is deterministic and follows caller-provided order.
//   - The legacy ApplyOptions validates final configuration after callbacks;
//     SteinerOption is an independent policy assembled with scalar validation
//     after each callback. A nil callback is ErrNilOption in both families.
//   - SteinerPathStorage defaults to SteinerRecomputePaths. It runs k-1
//     distance-only SSSP queries for k>=3, then source-grouped tracked reruns
//     for needed expansion paths (at most k-1 more). SteinerRetainPaths
//     instead keeps k-1 predecessor maps, O(kV) live history, and avoids
//     those extra queries. For k=2 both use one tracked SSSP.
//   - Three positive Max* count limits apply to the original graph/terminals;
//     zero means no caller-imposed cap. They do not bound RSS. A nil context,
//     negative cap, or unknown path-storage mode is ErrInvalidOption.
//
// Graph policy:
//
//   - graph must be non-nil.
//   - graph must be weighted.
//   - graph must be undirected at graph level.
//   - graph must not contain directed edge-level overrides.
//   - Self-loops are ignored because they cannot connect two components.
//   - Parallel edges are allowed; the lighter useful candidate may be selected.
//   - Negative finite weights are allowed and mathematically valid for MST.
//   - NaN and ±Inf weights are rejected before sorting or heap operations.
//   - Ordinary MST/MSF accepts any finite negative edge weight; SteinerTree
//     rejects every negative weight, including on loops and unrelated islands.
//     All original Steiner edges must be finite and nonnegative; loops are
//     validated but excluded from network candidates, and parallel IDs remain
//     distinguishable after source snapshot preparation.
//   - SteinerWalk accepts a structurally valid nonnegative undirected tree,
//     including nonterminal leaves, without consulting the original graph.
//     It cannot certify historical edge provenance, KMB construction, or OPT.
//
// Connectivity policy:
//
//   - ModeStrictTree is the default and returns ErrDisconnected when the graph
//     cannot be spanned by one tree.
//   - ModeForest is explicit opt-in via WithForest and returns a minimum spanning
//     forest instead of silently downgrading strict tree semantics.
//   - Empty graphs are classified as disconnected and empty through joined sentinels
//     where the validator exposes both conditions.
//   - SteinerTree requires all terminals in one original component and rejects
//     empty, repeated, empty-ID, and unknown terminal identifiers. Unrelated
//     disconnected nonterminal islands do not invalidate a connected terminal set.
//     k=1 returns a checked zero-edge singleton after global validation;
//     k=2 returns one tracked shortest path; k>=3 runs full KMB.
//   - SteinerWalk requires exactly one connected acyclic tree and complete
//     coverage of its caller-owned terminal set; it does not prune leaves.
//
// Determinism:
//
//   - Vertex order is inherited from core.Vertices().
//   - Edge order is inherited from core.Edges().
//   - Kruskal orders candidates by (Weight, Edge.ID), matching the former
//     stable weight sort on core.Edges()'s already sorted ID catalog.
//   - Prim's heap orders by finite Weight, then unique Edge.ID.
//   - Kruskal ComponentRoots are lexicographic component minima; Prim preserves
//     its explicit first root and selects later forest roots in sorted vertex order.
//   - Examples print invariant outputs unless edge order itself is the demonstrated contract.
//   - KMB closure uses one distance per unordered pair from lower-index source;
//     it does not average directionally rounded float64 distances. Original
//     parallel-edge recovery uses minimum (Weight, Edge.ID) and verifies SSSP
//     distance prefixes; original EdgeIDs survive both shared Kruskal phases.
//   - SteinerTree publishes sorted original Edge.ID values, never synthetic
//     terminal-closure edges. SteinerWalk sorts detached input copies and
//     explores neighbors by (neighbor ID, Edge.ID), regardless of input order.
//
// Error law:
//
//   - Use errors.Is for classification.
//   - Invalid graph failures preserve ErrInvalidGraph plus precise sentinels where applicable.
//   - ErrNilGraph reports nil graph input.
//   - ErrUnweightedGraph reports missing weighted semantics.
//   - ErrDirectedGraph reports directed graph policy.
//   - ErrDirectedEdge reports directed edge-level overrides.
//   - ErrEmptyGraph reports an empty graph.
//   - ErrDisconnected reports strict tree mode failure on empty or disconnected graphs.
//   - ErrNaNInfWeight reports non-finite edge weights.
//   - ErrEmptyRoot reports missing Prim root in strict mode.
//   - core.ErrVertexNotFound reports an absent non-empty Prim root.
//   - ErrNilOption reports nil Option values.
//   - ErrUnsupportedAlgorithm reports unsupported algorithm policy.
//   - ErrInvalidOption reports invalid option state.
//   - ErrNilResult reports invalid nil result helper access.
//   - ErrEmptyTerminals, ErrDuplicateTerminal, and core.ErrEmptyVertexID/
//     core.ErrVertexNotFound classify invalid required vertex identifiers.
//   - ErrDisconnectedTerminals means the required terminals cannot connect;
//     overflow from Dijkstra remains its original numeric error.
//   - ErrNegativeWeight is specific to the Steiner nonnegative input domain.
//   - ErrInvalidWitness classifies an original-edge path/identity mismatch;
//     ErrInvalidSteinerResult classifies invalid tree shape, cycles, coverage,
//     directed edges, or stale caller-owned total in a walk request.
//   - ErrWeightOverflow rejects a nonfinite left-fold prefix for a published
//     network/walk cost. ErrResourceLimit rejects a positive logical cap;
//     ErrSizeOverflow rejects unrepresentable count/allocation arithmetic.
//   - The last-failure cause remains discoverable by errors.Is where wrapped.
//     All three public kernels return nil on error; neither a partial tree
//     nor a partial walk is published. Context cancellation propagates as-is
//     when checked by SteinerTree between its synchronous SSSP phases.
//
// Complexity:
//
//   - Kruskal: O(E log E + E·α(V)) time, O(E + V) space.
//     Sorting dominates; DSU operations are effectively near-constant amortized.
//
//   - Prim: O(E log E) time, O(E + V) space for the current edge-frontier heap.
//     This package does not currently use a decrease-key vertex heap; therefore
//     O(E log V) must not be claimed for the current implementation.
//
//   - Result clone helpers are O(E + C), where C is the component count.
//
//   - SteinerTree: clone/admission including sorted core catalogs, followed by
//     (k-1+g) full SSSP queries for k>=3 (g<=k-1 in recompute mode; g=0
//     in retain mode), O(k^2 log k) closure selection, and O(Eu log Eu)
//     union selection plus linear pruning. The retained history costs O(kV);
//     recompute maintains O(V) live SSSP history plus O(k^2) closure data.
//     V/E refer to original graph, Eu to original-edge path union.
//
//   - SteinerWalk: O(Vs log Vs + Es log Es + sum deg log(deg+1) + S)
//     time, O(Vs+Es+S) space, where S<=2Es is published passage count.
//     It uses iterative BFS/DFS, not recursion, on arbitrarily deep tree chains.
//
//   - Checked cost publication uses float64 left folds in a fixed order.
//     Closure MST and union MST intermediate scalar totals are never computed.
//     Nonfinite final prefixes fail, but finite signed sums in ordinary MST
//     and fractional sums in Steiner are not exact-real arithmetic.
//
// Result ownership:
//
//   - Result contains detached core.Edge values.
//   - No live *core.Edge pointers are retained in public results.
//   - Result.Clone returns detached slice fields.
//   - Result.EdgeValues returns a caller-owned edge slice.
//   - SteinerResult and WalkResult own exported slices. Clone deep-copies
//     their mutable slices; IsNil is nil-safe. SteinerResult.EdgeValues
//     returns an independent slice, or ErrNilResult for a nil receiver.
//   - SteinerWalk validates a local canonical copy of the mutable input tree,
//     including the Edge.ID-ordered TotalWeight; it never repairs or mutates
//     the caller's tree and does not require KMB provenance.
//
// Concurrency:
//
//   - Ordinary MST adapters copy vertex/edge values before their kernels execute;
//     SteinerTree instead consumes one coherent core.Clone throughout KMB.
//   - Concurrent mutation of the same graph during construction is unsupported;
//     callers must synchronize graph writes.
//   - SteinerWalk is synchronous and has no context argument. Its caller-owned
//     result must not be concurrently mutated during validation/traversal.
//   - SteinerTree observes context cancellation in its own phases; legacy
//     full-SSSP Dijkstra calls do not offer an interruption context. No
//     detached goroutine continues computation after the call returns.
//
// AI-Hints:
//
//   - Do not reintroduce tuple-return public APIs; Result is the canonical result.
//   - Do not use epsilon inside sort or heap comparators; reject NaN/Inf and compare finite weights directly.
//   - Do not use edge.To as the next endpoint for undirected traversal; resolve endpoints relative to the source vertex.
//   - Do not silently return a forest from strict tree mode; require WithForest.
//   - Do not document matrix-backed APIs unless the exact public signatures exist.
//   - Do not claim Prim is O(E log V) unless the implementation changes to a vertex-key decrease-key policy.
//   - Use SteinerTree for terminal-only connectivity, not Kruskal over all V;
//     KMB is an approximation and must not be labeled Optimal.
//   - Choose SteinerRetainPaths only when its additional O(kV) history is
//     justified; full-SSSP numeric rules differ from Dijkstra target Search.
//   - Never calculate a closure/union MST scalar total when only its edges
//     are needed; an overflowing intermediate can hide a finite final tree.
//   - Never infer original-graph route optimality from SteinerWalk: the walk
//     is optimal only for covering every edge of its supplied tree.
//   - Do not deduplicate repeated walk EdgeIDs or use a final 2W-d scalar
//     in place of pricing the actual recorded passage sequence.
package mst
