// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp provides deterministic Traveling Salesman Problem solvers over
// matrix.Matrix distance models.
//
// The package is contract-first: solver selection, numeric rules, ownership
// rules, matching policy, partial-result semantics, and approximation claims are
// part of the public API. Documentation, examples, tests, and benchmarks must
// describe the same behavior as the implementation.
//
// # Why TSP matters
//
// TSP models a closed route that visits every required stop exactly once and
// returns to a fixed start. The abstract formulation appears in logistics,
// robotics, manufacturing, inspection, drilling, warehouse picking, DNA/PCB
// sequencing, fleet dispatch, and route-risk minimization.
//
// In production systems the "cost" does not have to be geographic distance.
// It may represent time, fuel, energy, mechanical latency, security exposure,
// spoilage risk, or a weighted business penalty. The package therefore accepts
// matrix.Matrix as the canonical distance source and lets callers define the
// meaning of each directed or undirected arc cost.
//
// TSP is NP-hard. The package exposes several explicit regimes because exactness,
// runtime, approximation guarantees, and local-search behavior are different
// engineering contracts.
//
// # Mathematical model
//
// For symmetric TSP, the solver searches for a Hamiltonian cycle over vertices
// V={0,...,n-1}. A tour π fixes a cyclic order, and the objective is:
//
//	min_{π} ∑_{i=0}^{n-1} c(π_i, π_{i+1}),   with π_n = π_0
//
// For asymmetric TSP, arc direction matters:
//
//	c(u,v) may be different from c(v,u).
//
// A valid closed tour has length n+1, starts at StartVertex, ends at the same
// vertex, and contains each vertex exactly once before the closing vertex.
//
// # Domain Scope & Goals
//
// The package is designed for matrix-backed complete TSP/ATSP solver kernels:
//
//   - SolveMatrix is the canonical matrix facade.
//   - SolveGraph adapts core.Graph through matrix preparation at the facade boundary.
//   - ExactHeldKarp provides exact dynamic programming for guarded small instances.
//   - BranchAndBound provides exact search when the search completes.
//   - Christofides provides a symmetric metric approximation pipeline.
//   - BlossomMatch delegates exact MWPM to the independent match.Perfect package.
//   - GreedyMatch delegates an explicit weaker matching policy with no formal ratio.
//   - TwoOptOnly and ThreeOptOnly provide deterministic local-search regimes.
//   - Optional metric closure resolves +Inf missing edges before final kernels.
//
// The matrix.Matrix value is the only distance source consumed by solver kernels.
// core.Graph is not consumed directly by Held-Karp, Branch-and-Bound,
// Christofides, matching, Eulerian, or local-search kernels.
//
// # Non-Goals
//
// The package intentionally does not implement:
//
//   - Concorde-style branch-and-cut.
//   - Hidden exact-to-heuristic fallback.
//   - Hidden heuristic-to-exact fallback.
//   - Silent GreedyMatch substitution when BlossomMatch fails.
//   - Final solving over matrices that still contain +Inf missing edges.
//   - Full arbitrary-reconnection ATSP 3-opt.
//   - Sparse graph-native TSP kernels.
//
// If a caller chooses a mathematically stronger policy, failure remains visible
// through errors instead of being silently replaced by weaker mathematics.
//
// # Matrix Law
//
// Public matrix input is matrix.Matrix. Before a solver kernel starts, the
// package validates the matrix against TSP-domain constraints.
//
// Final solver kernels require:
//
//   - square n×n shape;
//   - n >= 2 for public TSP solving;
//   - diagonal equal to zero within the package tolerance;
//   - no NaN;
//   - no -Inf;
//   - no negative finite weights;
//   - no +Inf after optional metric closure;
//   - symmetry when the selected algorithm requires symmetric input.
//
// +Inf may appear only as a missing-edge sentinel before metric closure. After
// metric closure, every final solver kernel receives a complete finite matrix.
//
// # Numeric Law
//
// NaN and infinities are sentinel-classified validation failures. Negative
// finite weights are rejected. Published costs are stabilized by the package
// rounding policy to prevent meaningless floating-point drift from changing
// observable output.
//
// Options.Eps controls strict-improvement and equality decisions in algorithms
// that use a tolerance. Invalid tolerance values are rejected; they are not
// silently replaced.
//
// # Algorithm Law
//
// ExactHeldKarp is exact and exponential:
//
//	time O(n^2·2^n), memory O(n*2^n).
//
// It is size-guarded and must not be treated as an unbounded production fallback.
//
// BranchAndBound is exact only when the search completes. With a time limit, it
// may return a feasible incumbent together with ErrTimeLimit. In that case,
// TimedOut is true and Optimal is false.
//
// Christofides requires symmetric complete metric input. Its constructive stages
// are:
//
//   - build a minimum spanning tree;
//   - collect odd-degree tree vertices;
//   - compute a minimum-weight perfect matching on those odd vertices;
//   - build an Eulerian multigraph;
//   - shortcut the Eulerian circuit into a Hamiltonian cycle.
//
// The formal approximation guarantee is:
//
//	cost(Christofides) <= 1.5 * cost(OPT)
//
// This 1.5 ratio is published only when MatchingAlgo==BlossomMatch, because the
// proof requires exact minimum-weight perfect matching.
//
// MatchingAlgo==GreedyMatch is an explicit weaker mode. It is deterministic and
// useful when callers intentionally trade proof strength for speed, but it does
// not publish the Christofides 1.5 approximation ratio.
//
// TwoOptOnly and ThreeOptOnly are local-search heuristics. They improve a
// current tour inside their implemented neighborhoods, but they do not certify
// global optimality.
//
// In the asymmetric 2-opt regime, an accepted move reverses a contiguous tour
// segment AND accounts for every internally reversed directed arc. Reconnecting
// the two boundary arcs without that reversal would split a single directed
// Hamiltonian cycle into two disjoint cycles. Directed 3-opt remains a more
// restricted, orientation-preserving neighborhood and is not full ATSP 3-opt.
//
// The final cost of an improved tour is recomputed from the published tour
// rather than inferred from accumulated floating-point move deltas. An
// asymmetric result is not reversed for symmetric-style canonical appearance.
//
// # Independent matching boundary
//
// The package github.com/lvlath/go/match independently owns the weighted
// Blossom numerical engine, Greedy pivot policy, ordered subset mapping and
// complete detached pairing result. tsp imports match, never vice versa.
// applyChristofidesMatching translates tsp.MatchingAlgo to match.Algorithm;
// the adapter validates all returned pairs before appending reciprocal
// undirected edges to the Christofides-local Eulerian adjacency. There is no
// second Blossom implementation in tsp and no hidden Greedy fallback.
// DefaultOptions currently selects BlossomMatch; choosing GreedyMatch is an
// explicit weakening of the approximation guarantee.

// ATSP 3-opt mode is restricted 3-opt*: it preserves segment orientation and
// does not implement full arbitrary directed 3-opt reconnection.
//
// # Options Law
//
// Options is an explicit solver policy, not a bag of hints.
//
// Important fields:
//
//   - Algo selects the solver family.
//   - Symmetric declares whether the input should be treated as symmetric TSP.
//   - StartVertex fixes tour start and closure.
//   - MatchingAlgo selects BlossomMatch or GreedyMatch for Christofides.
//   - BoundAlgo selects Branch-and-Bound lower-bound policy.
//   - EnableLocalSearch enables dispatcher-managed local-search post-processing.
//   - BestImprovement selects local-search neighborhood policy where supported.
//   - RunMetricClosure allows +Inf missing edges to be resolved before final solving.
//   - Eps controls numeric strictness.
//   - TimeLimit governs soft time-budget behavior in supported algorithms.
//   - Seed controls documented deterministic randomized neighborhood enumeration.
//
// Invalid options are rejected. The package does not infer hidden policy changes
// from matrix size, runtime pressure, or algorithm failure.
//
// # Result Law
//
// Result is the canonical public result. It owns detached slices and publishes
// solver facts instead of requiring callers to infer behavior from Options alone.
//
// Result fields communicate:
//
//   - Tour: closed vertex-index cycle.
//   - Cost: stabilized directed cycle cost.
//   - Algorithm: selected algorithm after option finalization.
//   - Exact: whether the algorithm is exact by design.
//   - Optimal: whether the returned result is certified optimal.
//   - TimedOut: whether a governed time limit stopped search.
//   - ApproximationRatio: formal ratio when one is valid, otherwise 0.
//   - MetricClosureApplied: whether +Inf closure preparation was used.
//   - IDs: optional detached labels aligned with matrix order.
//
// BranchAndBound is the governed partial-result exception. Callers using time
// limits must inspect both error and result.
//
// # Matching Law
//
// Christofides depends on a matching engine over odd MST vertices.
//
// BlossomMatch means exact minimum-weight perfect matching. It is the only mode
// that supports publishing the Christofides 1.5 ratio.
//
// GreedyMatch is deterministic but weaker. It is allowed only when explicitly
// selected and must publish NoApproximationRatio.
//
// Matching errors are not hidden by replacing the selected matching policy with
// another one.
//
// # Determinism & Stability Law
//
// For fixed matrix data, IDs, Options, and package version, successful results
// are deterministic. Stable ordering comes from matrix order, vertex index order,
// dense edge IDs, queue order, and documented algorithm-specific tie-breaks.
//
// Randomized local-search neighborhood enumeration is deterministic for a fixed
// seed. The seed must not alter validation, sentinel classification, or solver
// selection.
//
// # Ownership Law
//
// Public results own their Tour and IDs. Mutating caller-owned matrices, graphs,
// or ID slices after solving must not mutate the published result.
//
// matrix.Matrix values are treated as immutable by convention during solving.
// When metric closure is requested, closure runs on detached storage.
//
// # Error Law
//
// Errors are classified through sentinel values and errors.Is. Error strings are
// diagnostics for humans and must not be used as machine protocol.
//
// The package preserves sentinel classification for validation failures,
// unsupported policies, numeric failures, incomplete graphs, asymmetry, invalid
// IDs, time limits, and invalid matching states.
//
// # Concurrency Law
//
// Independent calls may run concurrently when callers do not concurrently mutate
// the same input matrix, graph, or ID storage. The package does not use mutable
// package-level solver state in normal solving.
//
// External concurrent mutation of caller-owned input is outside the package
// contract.
//
// # Typical usage
//
// For a symmetric metric logistics instance:
//
//	opts := tsp.DefaultOptions()
//	opts.Algo = tsp.Christofides
//	opts.Symmetric = true
//	opts.MatchingAlgo = tsp.BlossomMatch
//	result, err := tsp.SolveMatrix(dist, ids, opts)
//
// For an exact small manufacturing layout:
//
//	opts := tsp.DefaultOptions()
//	opts.Algo = tsp.BranchAndBound
//	opts.Symmetric = true
//	opts.BoundAlgo = tsp.OneTreeBound
//	result, err := tsp.SolveMatrix(dist, ids, opts)
//
// For directed local search:
//
//	opts := tsp.DefaultOptions()
//	opts.Algo = tsp.TwoOptOnly
//	opts.Symmetric = false
//	opts.EnableLocalSearch = true
//	result, err := tsp.SolveMatrix(dist, ids, opts)
//
// # AI-Hints
//
//   - Use SolveMatrix for new matrix-based examples and integrations.
//   - Use SolveGraph only when the source domain is core.Graph.
//   - Do not send core.Graph into solver kernels.
//   - Do not describe GreedyMatch as preserving the Christofides 1.5 proof.
//   - Do not hide exact-to-heuristic fallback behind Auto.
//   - Do not allow +Inf into final solver kernels.
//   - Do not compare error strings; use errors.Is.
//   - Do not describe ATSP 3-opt* as full ATSP 3-opt.
//   - Do not infer optimality from a low cost; read Result.Optimal.
package tsp
