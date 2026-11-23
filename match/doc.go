// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package match solves minimum-weight perfect matching on an explicitly
// selected even subset of a complete undirected weighted graph.
// The canonical Perfect facade accepts matrix.Matrix and publishes detached
// original-index pairs; it never reads or modifies TSP adjacency.
//
// # The mathematical contract
//
// For an ordered subset S of distinct original matrix vertices, a perfect
// matching partitions S into |S|/2 disjoint unordered pairs. Blossom minimizes
//
//	sum_{(u,v) in matching} dist[u,v]
//
// among all such partitions. The input must be a complete, finite,
// nonnegative, approximately symmetric square matrix; its diagonal is
// approximately zero. Triangle inequality is NOT required for matching.
//
// # Public entrypoints
//
//   - Perfect(dist, vertices, options...) selects Blossom unless explicitly
//     configured otherwise. It validates each option before reading a matrix.
//   - PerfectWithOptions(dist, vertices, policy) consumes a complete Options
//     value. Options{} is not a valid policy: Eps must be finite and positive.
//   - GreedyPerfect(dist, vertices) explicitly selects the deterministic greedy
//     heuristic without silently invoking Blossom.
//   - WithAlgorithm and WithEpsilon configure the canonical facade.
//   - DefaultOptions provides a validated starting configuration.
//
// The selection's order is semantically important: both the Blossom engine's
// local IDs and Greedy's last-remaining-pivot tie-breaking follow this order.
// Result.Pairs contain ORIGINAL matrix indices, not local subset positions.
// Pair endpoints are normalized U<V, and Pairs are sorted by ascending U only
// after the selected numerical kernel has finished.
//
// # Algorithm implementations
//
// The Blossom engine is the existing weighted Edmonds-style implementation:
// it searches a tight-edge alternating forest, contracts odd cycles, updates
// dual variables, expands and lifts nested blossoms, augments matching paths,
// and verifies reciprocal mates and dual consistency before publication.
// Blossom is exact-or-error: an unsuccessful search NEVER falls back to Greedy.
// Result.Optimal is true on successful verified Blossom completion; it is not
// an independently serialized dual proof or a bound on operating time.
//
// Greedy takes the LAST remaining vertex as pivot, scans partner candidates
// in remaining-slice order, and removes the chosen partner by swap-delete.
// It is not a global-cheapest-edge algorithm. Greedy can be much less accurate
// than Blossom and always publishes Optimal=false, including on tied optima.
//
// # Validation, numerical and error law
//
// A zero-length subset produces a non-nil empty result with cost zero WITHOUT
// accessing the source matrix. An odd-length subset returns ErrOddVertices
// before matrix access. Every nonempty request checks the entire matrix,
// including rows outside the requested subset, before indexing the subset.
// Incorrect indices, duplicate vertices, malformed weights, invalid symmetry,
// allocation-size overflow, failed matching, and numeric overflow are rejected
// with errors.Is-compatible sentinels; errors from matrix.At are preserved.
// Every failing Perfect call publishes (nil, error), never half a matching.
//
// Result.Cost is the checked UNROUNDED sum of original selected edge weights.
// It is not the Blossom transformed-profit objective and is not rounded to
// the nine-decimal TSP tour presentation convention. Asymmetric directions
// within the accepted structural tolerance are not averaged or normalized.
//
// # Ownership and composition
//
// Neither dist nor vertices is modified or retained. Result owns the pair
// slice; Clone detaches it, and a nil *Result implements IsNil() == true.
// Multi-pass reads assume callers do not mutate matrix storage concurrently;
// this is NOT an atomic snapshot of an arbitrary matrix.Matrix backend.
//
// TSP imports match, not conversely. Christofides creates a minimum spanning
// tree, computes odd-degree vertices, asks match for complete detached pairs,
// and validates the FULL matching before adding any edge to its local
// Eulerian multigraph. In the metric symmetric case exact Blossom matching
// is needed for the formal 3/2 approximation bound. Greedy does not justify
// such a bound. Neither algorithm checks the metric triangle inequality.
//
// # Complexity and resource limits
//
// Whole-matrix admission and copying take O(n^2) time and memory. The local
// Blossom problem and dense edge storage use O(k^2) memory for k selected
// vertices; Greedy partner scanning is O(k^2) after admission. Sorting the
// finished pairs costs O(k log k). Although classical weighted Blossom can
// have a polynomial bound, this existing implementation's nested path-lifting
// explores multiple candidate paths, so no global O(k^3) or O(k^4) runtime
// bound is asserted for this exact code. Source-size checks precede dense
// allocations; Go runtime out-of-memory termination cannot be recovered here.
//
// # Usage
//
//	exact, err := match.Perfect(dist, selected)
//	approximate, err := match.GreedyPerfect(dist, selected)
//
// In the example above dist denotes a stable matrix.Matrix and selected is an
// even ordered subset of original indices. Check each error before dereferencing
// a result; names are illustrative rather than standalone Go identifiers.
// See docs/MATCH.md and match/example_test.go for fully executable scenarios.
//
// # AI-Hints
//
//   - Never sort vertices before invoking Perfect: equal-cost tie behavior can
//     change even when the total optimal weight remains the same.
//   - Never interpret a public Pair index as a local 0..k-1 engine index.
//   - Do not substitute a greedy result after Blossom failure or infer a
//     Christofides bound from explicit Greedy selection.
//   - Do not treat finite high penalties as missing edges: +Inf is invalid
//     for complete matching, and constraint satisfaction is an application law.
//   - Distinguish raw matching Cost from rounded TSP tour Cost; they serve
//     different mathematical objects.
package match
