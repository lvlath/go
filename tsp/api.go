// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp exposes canonical public facades for Traveling Salesman Problem solvers.
// The public surface publishes Result only, so exactness, optimality, timeout state,
// approximation metadata, and ownership rules remain available to callers.
package tsp

import (
	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

// SolveMatrix solves a TSP/ATSP instance represented by a distance matrix.
// It is the canonical matrix facade and publishes the full Result contract.
//
// Implementation:
//   - Stage 1: Prepare direct matrix input, including optional metric closure.
//   - Stage 2: Validate optional IDs against final matrix order.
//   - Stage 3: Delegate to the result-native solver dispatcher without alternative mathematics.
//   - Stage 4: Return the detached canonical Result.
//
// Behavior highlights:
//   - Does not mutate caller-owned direct matrices.
//   - Uses matrix.APSPInPlace for metric closure.
//   - Publishes the full canonical result contract.
//   - Does not silently change algorithms beyond explicit options.
//
// Inputs:
//   - dist: square distance matrix; +Inf may appear only before metric closure.
//   - ids: optional row/column labels; when non-nil, len(ids)==n and all IDs are non-empty/unique.
//   - opts: explicit solver policy; use DefaultOptions and override fields.
//
// Returns:
//   - *Result: detached canonical result on success.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - ErrInvalidOptions, ErrUnsupportedAlgorithm, ErrATSPNotSupportedByAlgo.
//   - ErrNilDistanceMatrix, ErrNonSquare, ErrDimensionMismatch.
//   - ErrNaNInf, ErrNonZeroDiagonal, ErrNegativeWeight, ErrIncompleteGraph, ErrAsymmetry.
//   - ErrInvalidIDs.
//   - Algorithm-specific sentinels such as ErrSizeTooLarge or ErrTimeLimit.
//
// Determinism:
//   - Matrix row/column order is the vertex order.
//   - Kernel tie-breaks and canonicalization define Tour order.
//   - IDs are copied without sorting.
//
// Complexity:
//   - Validation O(n^2).
//   - Metric closure O(n^3) when enabled.
//   - Solver complexity depends on opts.Algo.
//
// Notes:
//   - Branch-and-Bound timeout incumbents remain visible through Result.
//
// AI-Hints:
//   - Do not bypass this facade in examples unless testing a specific kernel.
//   - Do not publish a reduced result from this canonical facade.
func SolveMatrix(dist matrix.Matrix, ids []string, opts Options) (*Result, error) {
	if err := validateOptionsStandalone(opts); err != nil {
		return nil, err
	}

	if result, handled, err := solveDegenerateIfAny(dist, ids, opts); handled {
		return result, err
	}

	prepared, err := prepareSolverDistanceMatrix(dist, opts)
	if err != nil {
		return nil, err
	}

	finalOptions := opts
	finalOptions.RunMetricClosure = false

	finalOptions, err = finalizeSolverOptions(prepared.n, finalOptions)
	if err != nil {
		return nil, err
	}

	if ids != nil {
		if err = validateIDs(ids, prepared.n); err != nil {
			return nil, err
		}
	}

	result, err := solvePreparedMatrix(prepared, ids, finalOptions)
	if result == nil {
		return nil, err
	}

	return result, err
}

// SolveGraph solves a TSP instance by adapting a core.Graph into a matrix distance model.
// It is the canonical graph facade and publishes the full Result contract.
//
// Implementation:
//   - Stage 1: Reject nil and currently unsupported mixed-direction graphs.
//   - Stage 2: Build matrix adapter options from core.Graph flags.
//   - Stage 3: Build adjacency/distance matrix through matrix.NewAdjacencyMatrix.
//   - Stage 4: Recover stable row-index IDs through the matrix adapter accessor.
//   - Stage 5: Delegate to SolveMatrix without re-running graph metric closure.
//
// Behavior highlights:
//   - Adapter policy is explicit and deterministic.
//   - Mixed directedness is rejected until matrix exposes a lossless mixed-edge adapter.
//   - Vertex IDs are recovered by index, not map iteration order.
//
// Inputs:
//   - g: source graph.
//   - opts: solver policy.
//
// Returns:
//   - *Result: detached canonical result.
//   - error: sentinel-classified failure.
//
// Errors:
//   - ErrNilGraph for nil graph.
//   - ErrMixedGraphNotRepresentable for unsupported mixed directedness.
//   - matrix adapter errors from matrix.NewMatrixOptions / matrix.NewAdjacencyMatrix.
//   - SolveMatrix errors after adaptation.
//
// Determinism:
//   - Core graph order is inherited through matrix adapter vertex indices.
//   - ID recovery writes ids[index]=id, so map iteration does not affect output.
//
// Complexity:
//   - Adapter build O(V^2+E).
//   - Optional metric closure O(V^3).
//   - Solver complexity depends on opts.Algo.
//
// AI-Hints:
//   - Do not flatten mixed directed/undirected graphs into a global directed matrix.
//   - Do not recover IDs by appending over map iteration.
func SolveGraph(g *core.Graph, opts Options) (*Result, error) {
	// Nil graph => invalid shape for building matrices.
	if g == nil {
		return nil, ErrNilGraph
	}

	graphStats := g.Stats()
	if graphStats.MixedMode || (graphStats.DirectedEdgeCount > 0 && graphStats.UndirectedEdgeCount > 0) {
		return nil, ErrMixedGraphNotRepresentable
	}

	matrixOptions := make([]matrix.Option, 0, 6)

	// Map core → matrix options (explicit, no booleans into WithX)
	// Orientation: mixed OR directed-default ⇒ directed matrix (can encode both types of edges).
	if graphStats.DirectedDefault || graphStats.DirectedEdgeCount > 0 {
		matrixOptions = append(matrixOptions, matrix.WithDirected())
	} else {
		matrixOptions = append(matrixOptions, matrix.WithUndirected())
	}

	// Multi-edges: matrix default = allowMulti=true; respect core policy explicitly.
	if graphStats.AllowsMulti {
		matrixOptions = append(matrixOptions, matrix.WithAllowMulti())
	} else {
		matrixOptions = append(matrixOptions, matrix.WithDisallowMulti())
	}

	// Loops: matrix default = allowLoops=false.
	if graphStats.AllowsLoops {
		matrixOptions = append(matrixOptions, matrix.WithAllowLoops())
	} else {
		matrixOptions = append(matrixOptions, matrix.WithDisallowLoops())
	}

	// Weights: matrix default = unweighted(false).
	if graphStats.Weighted {
		matrixOptions = append(matrixOptions, matrix.WithWeighted())
	} else {
		matrixOptions = append(matrixOptions, matrix.WithUnweighted())
	}

	// Metric closure (APSP).
	if opts.RunMetricClosure {
		matrixOptions = append(matrixOptions, matrix.WithMetricClosure())
	}

	// Build matrix options from graph flags + dispatcher policy; «single source of truth».
	frozenMatrixOptions, err := matrix.NewMatrixOptions(matrixOptions...)
	if err != nil {
		return nil, err
	}

	adjacency, err := matrix.NewAdjacencyMatrix(g, frozenMatrixOptions)
	if err != nil {
		// NewAdjacencyMatrix returns matrix-level errors; forward them as-is.
		// Matrix facade validation will surface tsp sentinels after adaptation.
		return nil, err
	}

	// Recover stable vertex ordering through the matrix adapter accessor.
	// The accessor returns a detached index-to-ID slice in canonical matrix order.
	ids, err := adjacency.VertexIDs()
	if err != nil {
		return nil, err
	}

	delegateOptions := opts
	if opts.RunMetricClosure {
		delegateOptions.RunMetricClosure = false
	}

	// Delegate to matrix dispatcher (unified validation is done there).
	result, err := SolveMatrix(adjacency.Mat, ids, delegateOptions)
	if err != nil {
		return nil, err
	}

	if opts.RunMetricClosure {
		result.MetricClosureApplied = true
	}

	return result, nil
}

// HeldKarp solves a complete TSP or ATSP instance with the Held-Karp dynamic program.
// It is the direct exact dynamic-programming entrypoint and always publishes
// Exact=true and Optimal=true on successful completion.
//
// Implementation:
//   - Stage 1: Force the direct solver policy to ExactHeldKarp.
//   - Stage 2: Run the private Held-Karp kernel over the validated final matrix.
//   - Stage 3: Publish a detached Result with exact optimal metadata.
//
// Behavior highlights:
//   - Supports asymmetric distances.
//   - Does not perform metric closure.
//   - Returns no partial result on timeout.
//   - Does not depend on opts.Algo supplied by the caller.
//
// Inputs:
//   - dist: complete finite square distance matrix.
//   - opts: solver policy; StartVertex, MaxExactN, Eps, and TimeLimit are honored.
//
// Returns:
//   - *Result: exact optimal tour on success.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - ErrInvalidOptions from option validation.
//   - ErrNilDistanceMatrix, ErrNonSquare, ErrDimensionMismatch from matrix validation.
//   - ErrNaNInf, ErrNegativeWeight, ErrIncompleteGraph from final matrix checks.
//   - ErrStartOutOfRange for invalid StartVertex.
//   - ErrSizeTooLarge when MaxExactN is exceeded.
//   - ErrTimeLimit when the configured budget is exhausted before completion.
//
// Determinism:
//   - Fixed subset-size, mask, endpoint, and predecessor scan order.
//   - Equal-cost ties follow the kernel predecessor policy.
//
// Complexity:
//   - Time O(n^2 * 2^n), Space O(n * 2^n).
//
// Notes:
//   - Use SolveMatrix when IDs or metric closure are required.
//   - This wrapper is not a dispatcher; it always selects ExactHeldKarp.
//
// AI-Hints:
//   - Always force solverOptions.Algo=ExactHeldKarp before validation.
//   - Do not reject ATSP because a caller accidentally passed opts.Algo=Christofides.
func HeldKarp(dist matrix.Matrix, opts Options) (*Result, error) {
	opts.Algo = ExactHeldKarp

	return heldKarp(dist, opts)
}

// ChristofidesSolve runs the symmetric Christofides construction directly.
// The formal 1.5 ratio is published only when the selected matching policy proves
// exact minimum-weight perfect matching.
//
// Implementation:
//   - Stage 1: Force the direct solver policy to Christofides.
//   - Stage 2: Validate symmetric complete metric input.
//   - Stage 3: Run MST, odd-degree matching, Eulerian circuit, and shortcutting.
//   - Stage 4: Publish a detached Result with approximation proof metadata.
//
// Behavior highlights:
//   - Requires symmetric final matrix input.
//   - GreedyMatch is explicit weaker behavior and publishes NoApproximationRatio.
//   - BlossomMatch is exact-or-error for the currently supported matching regime.
//   - Does not perform metric closure.
//
// Inputs:
//   - dist: complete finite symmetric distance matrix.
//   - opts: solver policy; StartVertex and MatchingAlgo are honored.
//
// Returns:
//   - *Result: canonical heuristic/approximation result.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - ErrInvalidOptions from option validation.
//   - ErrATSPNotSupportedByAlgo when Symmetric=false.
//   - ErrNonEulerian, ErrInvalidTour from pipeline stages.
//   - ErrInvalidMatching for malformed matching state.
//   - ErrIncompleteGraph when a required perfect matching cannot be formed.
//   - Matrix validation sentinels from final matrix checks.
//
// Determinism:
//   - MST scan, odd-vertex collection, matching, Eulerian traversal, and shortcutting use fixed order.
//
// Complexity:
//   - Time O(n^2) plus matching complexity.
//   - Space O(n^2) for dense graph and matching-local buffers.
//
// Notes:
//   - Use SolveMatrix when IDs, metric closure, or dispatcher local-search policy are required.
//
// AI-Hints:
//   - Do not infer ApproximationRatio from Options.MatchingAlgo alone.
//   - Do not convert exact matching failures into heuristic output.
func ChristofidesSolve(dist matrix.Matrix, opts Options) (*Result, error) {
	opts.Algo = Christofides

	return christofides(dist, opts)
}

// BranchAndBoundSolve runs exact deterministic Branch-and-Bound search directly.
// It preserves timeout incumbent metadata when a feasible incumbent exists before
// the configured time budget is exhausted.
//
// Implementation:
//   - Stage 1: Force the direct solver policy to BranchAndBound.
//   - Stage 2: Copy the matrix into the solver weight buffer.
//   - Stage 3: Seed an incumbent when possible and run deterministic DFS search.
//   - Stage 4: Publish complete or partial Result metadata.
//
// Behavior highlights:
//   - Supports symmetric and asymmetric final matrices.
//   - Exact=true because the algorithm is exact, even when timeout prevents proof.
//   - Optimal=false on timeout incumbents.
//   - NodesExpanded records actual DFS node expansions.
//
// Inputs:
//   - dist: complete finite square distance matrix.
//   - opts: solver policy; StartVertex, BoundAlgo, Eps, and TimeLimit are honored.
//
// Returns:
//   - *Result: optimal result, timeout incumbent, or nil if no incumbent exists.
//   - error: nil on completion or ErrTimeLimit on timeout.
//
// Errors:
//   - ErrInvalidOptions.
//   - Matrix validation sentinels from weight copying.
//   - ErrTimeLimit when deadline expires.
//   - ErrIncompleteGraph when no tour can be constructed.
//
// Determinism:
//   - Branch order is sorted by cost and stable vertex tie-breaks.
//   - DFS state transitions are deterministic.
//
// Complexity:
//   - Worst-case Time O(n!), Space O(n^2) for buffers plus O(n) search state.
//
// Notes:
//   - Use SolveMatrix for facade-level IDs and metric closure.
//
// AI-Hints:
//   - Always force solverOptions.Algo=BranchAndBound before validation.
//   - Do not mark timeout incumbents as Optimal.
func BranchAndBoundSolve(dist matrix.Matrix, opts Options) (*Result, error) {
	opts.Algo = BranchAndBound

	return branchAndBound(dist, opts)
}

// TwoOptSearch runs 2-opt local search from a caller-provided initial tour.
// It publishes canonical result metadata and never claims exactness or global optimality.
//
// Implementation:
//   - Stage 1: Force the direct solver policy to TwoOptOnly.
//   - Stage 2: Validate the initial tour through the local-search kernel.
//   - Stage 3: Run deterministic 2-opt improvement.
//   - Stage 4: Publish a detached Result.
//
// Behavior highlights:
//   - Supports symmetric and asymmetric matrices according to the 2-opt kernel policy.
//   - Timeout may return the current feasible tour.
//   - ApproximationRatio is not claimed.
//
// Inputs:
//   - dist: complete finite final solver matrix.
//   - initTour: closed Hamiltonian cycle used as the starting solution.
//   - opts: local-search policy.
//
// Returns:
//   - *Result: improved tour result.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - ErrInvalidOptions.
//   - ErrInvalidTour.
//   - Matrix validation sentinels.
//   - ErrTimeLimit when timeout occurs after a feasible current tour exists.
//
// Determinism:
//   - Neighborhood scan order is fixed unless ShuffleNeighborhood is explicitly enabled.
//
// Complexity:
//   - Time O(iterations*n^2), Space O(n).
//
// Notes:
//   - This direct wrapper does not attach IDs.
//
// AI-Hints:
//   - Do not publish exact or optimal metadata from local search.
func TwoOptSearch(dist matrix.Matrix, initTour []int, opts Options) (*Result, error) {
	opts.Algo = TwoOptOnly

	return twoOptSearch(dist, initTour, opts)
}

// ThreeOptSearch runs 3-opt local search from a caller-provided initial tour.
// Symmetric mode uses the full symmetric 3-opt move set; asymmetric mode uses
// the package restricted orientation-preserving neighborhood.
//
// Implementation:
//   - Stage 1: Force the direct solver policy to ThreeOptOnly.
//   - Stage 2: Validate the initial tour and matrix policy.
//   - Stage 3: Run deterministic 3-opt improvement.
//   - Stage 4: Publish a detached Result.
//
// Behavior highlights:
//   - Does not claim global optimality.
//   - Timeout may return the current feasible tour.
//   - Asymmetric mode must not be described as full ATSP 3-opt.
//
// Inputs:
//   - dist: complete finite final solver matrix.
//   - initTour: closed Hamiltonian cycle used as the starting solution.
//   - opts: local-search policy.
//
// Returns:
//   - *Result: improved tour result.
//   - error: nil on success or sentinel-classified failure.
//
// Errors:
//   - ErrInvalidOptions.
//   - ErrInvalidTour.
//   - Matrix validation sentinels.
//   - ErrTimeLimit when timeout occurs after a feasible current tour exists.
//
// Determinism:
//   - Neighborhood scan order is fixed unless ShuffleNeighborhood is explicitly enabled.
//
// Complexity:
//   - Symmetric mode: Time O(iterations*n^3), Space O(n).
//   - Restricted asymmetric mode: depends on the implemented tail-swap scan, Space O(n).
//
// Notes:
//   - This direct wrapper does not attach IDs.
//
// AI-Hints:
//   - Do not call the asymmetric path full ATSP 3-opt.
//   - Do not publish exact or optimal metadata from local search.
func ThreeOptSearch(dist matrix.Matrix, initTour []int, opts Options) (*Result, error) {
	opts.Algo = ThreeOptOnly

	return threeOptSearch(dist, initTour, opts)
}
