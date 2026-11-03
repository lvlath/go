// Package lvlath is a deterministic, contract-driven Go toolkit for graph
// algorithms, dense graph algebra, routing, network optimization, time-series
// alignment, tour planning, and reproducible algorithm fixtures.
//
// lvlath is pure Go: no cgo and no external runtime dependencies. The repository
// is organized as small, composable subpackages. Each package owns a narrow
// mathematical contract and publishes result artifacts that callers can inspect,
// test, serialize, and reason about.
//
// ─────────────────────────────────────────────────────────────────────────────
// What lvlath is
// ─────────────────────────────────────────────────────────────────────────────
//
// lvlath is a package ecosystem, not a monolithic solver:
//
//   - core      - deterministic, capability-driven graph substrate.
//   - bfs       - unweighted hop-distance traversal and weak components.
//   - dfs       - finish-order traversal, cycle witnesses, and topological sort.
//   - dijkstra  - non-negative weighted single-source shortest paths.
//   - mst       - strict MST and explicit minimum spanning forest via Kruskal/Prim.
//   - flow      - max-flow / min-cut algorithms with residual graph artifacts.
//   - matrix    - dense row-major graph algebra, APSP, statistics, sanitation.
//   - dtw       - deterministic Dynamic Time Warping for scalar, cost-matrix,
//     and multivariate sequence alignment.
//   - tsp       - matrix-backed TSP/ATSP solvers: exact, approximate, matching,
//     local-search, and time-governed regimes.
//   - gridgraph - generated 2D lattice topology for pathfinding-style workflows.
//   - builder   - deterministic fixtures for examples, tests, and benchmarks.
//
// The main architectural rule is:
//
//	core owns topology;
//	matrix owns dense numeric artifacts;
//	algorithm packages own explicit result contracts.
//
// Each package documents implemented API behavior in its own doc.go. Repository
// theory, diagrams, pitfalls, and operational recipes live in docs/{PACKAGE}.md.
// Start with docs/TUTORIAL.md for the learning path and docs/lvlath_UES.md for
// the project engineering standard.
//
// ─────────────────────────────────────────────────────────────────────────────
// Error model and invariants
// ─────────────────────────────────────────────────────────────────────────────
//
// All packages use a shared set of sentinel errors (test-friendly via errors.Is):
//
//	ErrNonSquare, ErrNonZeroDiagonal, ErrAsymmetry,
//	ErrNegativeWeight, ErrIncompleteGraph, ErrDimensionMismatch,
//	ErrTimeLimit, ErrATSPNotSupportedByAlgo, ErrMatchingNotImplemented.
//
// Policy:
//   - Validate inputs early and explicitly (shape, bounds, NaN, ±Inf).
//   - Make ambiguous conditions explicit through options (e.g., symmetric vs ATSP).
//   - Disallow unsupported combinations in dispatchers (e.g., Christofides with ATSP).
//
// ─────────────────────────────────────────────────────────────────────────────
// Determinism and numerics
// ─────────────────────────────────────────────────────────────────────────────
//
//   - No global state, no implicit RNG. Algorithms are deterministic for the
//     same inputs and options.
//   - Where randomized neighborhood order is useful (local search), it is fully
//     controlled by Options.Seed and can be disabled; results remain stable under
//     the same seed.
//   - Floating-point comparisons honor a clear epsilon policy (documented per
//     package). Tests stabilize comparisons via consistent rounding to avoid
//     platform drift.
//
// ─────────────────────────────────────────────────────────────────────────────
// Package overview (balanced highlights)
// ─────────────────────────────────────────────────────────────────────────────
//
// core
//
//	Provides the deterministic graph substrate used by downstream packages.
//	Graph capabilities such as directedness, weights, loops, multi-edges, and
//	mixed edges are explicit model choices. Stable topology surfaces make tests,
//	witnesses, examples, and algorithm results reproducible.
//
// bfs
//
//	Computes unweighted hop-distance traversals and weak components. BFS result
//	fields separate discovered, queued, processed, depth, and parent semantics.
//	Partial results are meaningful under cancellation or hook failure according
//	to the package contract.
//
// dfs
//
//	Provides deterministic depth-first traversal, finish-order output, cycle
//	witnesses, and topological sorting. dfs.Result.Order is finish order, not
//	discovery order. Cycle detection publishes deterministic witnesses rather
//	than pretending to enumerate every simple cycle.
//
// dijkstra
//
//	Computes single-source shortest paths on non-negative weighted graphs.
//	The package separates unknown targets, known-but-unreachable targets,
//	disabled path tracking, and no-path states. +Inf is the canonical distance
//	for known unreachable vertices under the active policy.
//
// mst
//
//	Computes minimum spanning trees and explicit minimum spanning forests over
//	weighted, undirected graphs. MinimumSpanningTree is the canonical facade;
//	Kruskal and Prim are focused wrappers. mst.Result reports algorithm, mode,
//	root, detached selected edges, total weight, vertex count, component count,
//	and deterministic component roots.
//
// flow
//
//	Provides max-flow / min-cut workflows through Ford-Fulkerson,
//	Edmonds-Karp, and Dinic-style algorithms. Weights represent capacities in
//	this package. Results expose residual-network state, which is part of the
//	algorithm artifact rather than incidental debug data.
//
// matrix
//
//	Implements deterministic row-major dense matrices, graph-to-matrix adapters,
//	metric closure, Floyd-Warshall, algebra facades, LU/QR/inverse/eigen helpers,
//	statistical transforms, sanitation, and tolerant comparison. Matrix policy
//	preserves important graph meanings such as zero-weight edges, +Inf absence,
//	incidence signs, metric-closure distance surfaces, and zero-shape matrices.
//
// dtw
//
//	Implements Dynamic Time Warping through canonical Result-returning facades:
//	Align for scalar sequences, AlignCostMatrix for precomputed local-cost
//	surfaces, and AlignMatrix for multivariate time-step matrices. Window,
//	slope-penalty, memory mode, path tracking, accumulated matrix output, and
//	local-cost artifacts are explicit policy choices.
//
// tsp
//
//	Provides deterministic matrix-backed Traveling Salesman Problem solvers.
//	SolveMatrix is the canonical matrix facade; SolveGraph adapts core.Graph at
//	the boundary. ExactHeldKarp and completed BranchAndBound are exact regimes;
//	Christofides is a symmetric metric approximation pipeline; BlossomMatch is
//	the exact matching mode required for the 1.5 ratio; GreedyMatch is explicit
//	and weaker; TwoOptOnly and ThreeOptOnly are local-search regimes. Result
//	publishes tour, cost, algorithm, exactness, optimality, timeout state,
//	approximation ratio, metric-closure flag, and detached IDs.
//
// gridgraph
//
//	Generates grid/lattice topology for pathfinding, teaching, and benchmark
//	fixtures. Use it when a map-like graph should be generated from coordinate
//	rules instead of manually wiring hundreds of edges.
//
// builder
//
//	Builds deterministic graph and data fixtures for examples, tests, and
//	benchmarks. Use it to avoid random, unstable, or hand-wired setup when the
//	goal is reproducible algorithm behavior.
//
// ─────────────────────────────────────────────────────────────────────────────
// Performance and trade-offs (concise, per area)
// ─────────────────────────────────────────────────────────────────────────────
//
//   - core: deterministic graph surfaces and capability checks are favored over
//     raw map-only shortcuts. Algorithms rely on this stability.
//   - bfs / dfs: traversal is O(V+E). Hooks and filters add per-event overhead;
//     cancellation and partial-result behavior are contract-visible.
//   - dijkstra: O((V+E) log V) style priority-queue routing over non-negative
//     weights. Runtime walls and max-distance cutoffs are policy gates, not
//     graph mutations.
//   - mst: Kruskal is O(E log E + E·α(V)); Prim is O(E log E) for the current
//     edge-frontier heap implementation. Do not document Prim as O(E log V)
//     unless the implementation changes to a vertex-key decrease-key heap.
//   - flow: Edmonds-Karp is simpler and easier to audit; Dinic is typically
//     faster on larger layered networks. Residual graph construction is part of
//     the real cost.
//   - matrix: dense algorithms trade memory O(R*C) for predictable row-major
//     kernels and graph-algebra convenience. Zero-shape matrices are valid
//     structural results.
//   - dtw: scalar Align is O(n*m); AlignMatrix adds O(n*m*d) local-cost
//     construction. Rolling-row distance mode uses O(m) DP memory; path or
//     accumulated-matrix output requires O(n*m) memory.
//   - tsp: Held-Karp is exponential O(n²·2^n) and size-guarded. Branch-and-Bound
//     is exact only when completed. Christofides requires symmetric complete
//     metric input and exact Blossom matching to publish the 1.5 ratio. Local
//     search improves tours but does not certify global optimality.
//   - builder / gridgraph: generated fixtures reduce setup mistakes and improve
//     benchmark reproducibility; they should not hide the topology being tested.
//
// ─────────────────────────────────────────────────────────────────────────────
// Scope boundaries and known limitations
// ─────────────────────────────────────────────────────────────────────────────
//
//   - core: no persistent graph database, no distributed graph storage, and no
//     hidden algorithm-specific topology mutation.
//   - bfs: unweighted hop-distance traversal only. It is not a weighted routing
//     algorithm.
//   - dfs: cycle detection returns deterministic witnesses, not exhaustive
//     simple-cycle enumeration.
//   - dijkstra: finite negative weights are rejected by design. Use a different
//     shortest-path algorithm for negative-weight domains.
//   - mst: directed optimum branching/arborescence and Steiner tree optimization
//     are out of scope. Strict MST does not silently downgrade to forest mode;
//     callers must request forest mode explicitly.
//   - flow: capacities are finite non-negative values. Generalized flows,
//     min-cost flow, and multi-commodity flow are out of scope.
//   - matrix: dense storage is not a sparse-matrix engine. Metric closure is a
//     distance artifact and must not be exported as original topology.
//   - dtw: exact DTW is not generally a metric and does not imply triangle
//     inequality. Path recovery requires full accumulated storage. AlignMatrix
//     uses squared L2 row costs, so feature scale is caller responsibility.
//   - tsp: solver kernels consume matrix-backed complete cost models. Sparse
//     graph-native TSP kernels and Concorde-style branch-and-cut are out of
//     scope. Branch-and-Bound with a time limit may return a feasible incumbent,
//     but that result is not certified optimal. Greedy matching does not publish
//     the Christofides 1.5 ratio.
//   - builder / gridgraph: fixture generation is deterministic infrastructure,
//     not proof that an algorithm is correct for all possible graph families.
//
// ─────────────────────────────────────────────────────────────────────────────
// Where to go next
// ─────────────────────────────────────────────────────────────────────────────
//
//   - docs/TUTORIAL.md - start here for an end-to-end tour, selection matrix,
//     and practical guidance on determinism and numeric stability.
//   - {package}/doc.go - the formal API contracts, options, and edge cases.
//   - docs/{ALGORITHM}.md - compact backgrounders with formulas, diagrams,
//     and runnable examples for each algorithm.
//
// lvlath targets modern Go, is entirely in Go, and has no external dependencies.
package main
