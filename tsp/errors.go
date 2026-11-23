// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp defines sentinel errors for the Traveling Salesman Problem solvers.
// The file is the single extension point for safety and correctness error classes.
package tsp

import "errors"

var (
	// ErrNilDistanceMatrix reports a nil or typed-nil distance matrix input.
	//
	// AI-Hints:
	//   - Use this sentinel for nil matrix inputs, not ErrDimensionMismatch.
	//   - Preserve matrix-level nil sentinels with errors.Join where possible.
	ErrNilDistanceMatrix = errors.New("tsp: distance matrix is nil")

	// ErrNilGraph reports a nil graph input for graph-to-matrix adapters.
	//
	// AI-Hints:
	//   - Use this sentinel at the tsp adapter layer.
	//   - Preserve matrix/core adapter errors with errors.Join where applicable.
	ErrNilGraph = errors.New("tsp: graph is nil")

	// ErrNilTour reports a nil tour slice.
	//
	// AI-Hints:
	//   - Nil and structurally invalid tours are different classes.
	//   - Use ErrInvalidTour for malformed non-nil tours.
	ErrNilTour = errors.New("tsp: tour is nil")

	// ErrNilResult reports method calls on a nil *Result receiver.
	//
	// AI-Hints:
	//   - Result helper methods must classify nil access with this sentinel.
	//   - Do not panic on nil result receivers.
	ErrNilResult = errors.New("tsp: result is nil")

	// ErrInvalidOptions reports structurally invalid solver options.
	//
	// AI-Hints:
	//   - Negative durations, NaN/Inf epsilon, invalid iteration caps,
	//     and unknown enum values belong here.
	ErrInvalidOptions = errors.New("tsp: invalid options")

	// ErrInvalidIDs reports empty or duplicated vertex IDs.
	//
	// AI-Hints:
	//   - Length mismatch remains ErrDimensionMismatch.
	//   - Empty/duplicate IDs are semantic ID violations.
	ErrInvalidIDs = errors.New("tsp: invalid vertex ids")

	// ErrInvalidTour reports a malformed non-nil Hamiltonian cycle.
	//
	// AI-Hints:
	//   - Use for wrong length, missing closure, duplicate vertex, or out-of-range vertex.
	ErrInvalidTour = errors.New("tsp: invalid tour")

	// ErrInvalidMatching reports malformed matching input or infeasible matching state.
	//
	// AI-Hints:
	//   - Use when the odd-degree set cannot be paired safely.
	//   - Do not silently skip unmatched odd vertices in Christofides.
	ErrInvalidMatching = errors.New("tsp: invalid matching input")

	// ErrNaNInf reports forbidden NaN, -Inf, or contract-invalid Inf values.
	//
	// AI-Hints:
	//   - +Inf may be allowed only before metric closure.
	//   - Final solver input must not contain +Inf off the diagonal.
	ErrNaNInf = errors.New("tsp: NaN or Inf encountered")

	// ErrMixedGraphNotRepresentable reports a core.Graph with mixed edge directionality
	// that cannot be losslessly converted through the current plain adjacency adapter.
	//
	// AI-Hints:
	//   - Reject mixed graphs until matrix exposes a mixed-aware orientation adapter.
	//   - Do not flatten mixed graphs into a global directed/undirected matrix silently.
	ErrMixedGraphNotRepresentable = errors.New("tsp: mixed directedness graph cannot be converted to a TSP distance matrix safely")

	// ErrNonEulerian reports a malformed multigraph passed to Eulerian circuit construction.
	ErrNonEulerian = errors.New("tsp: graph is not Eulerian")

	// Validation / input-shape errors. Do not wrap with fmt.Errorf where a sentinel suffices.

	// ErrNonSquare indicates the distance matrix is not square.
	ErrNonSquare = errors.New("tsp: matrix is not square")

	// ErrNegativeWeight reports a negative distance in a TSP input.
	ErrNegativeWeight = errors.New("tsp: negative distance encountered")

	// ErrAsymmetry reports a symmetry violation (dist[i][j] != dist[j][i]) under symmetric TSP policy.
	ErrAsymmetry = errors.New("tsp: asymmetric distance matrix")

	// ErrNonZeroDiagonal reports a non-zero self-distance (dist[i][i] != 0).
	ErrNonZeroDiagonal = errors.New("tsp: non-zero self-distance")

	// ErrIncompleteGraph reports that final solver input still has missing edges.
	ErrIncompleteGraph = errors.New("tsp: incomplete distance matrix (no Hamiltonian cycle possible)")

	// ErrDimensionMismatch reports shape mismatch after a more specific shape sentinel is not available.
	ErrDimensionMismatch = errors.New("tsp: dimension mismatch")

	// ErrStartOutOfRange indicates Options.StartVertex is outside [0..n-1].
	ErrStartOutOfRange = errors.New("tsp: start vertex out of range")

	// ErrMissingVertexIDs reports a vertex-ID projection request on a result that
	// has no attached matrix-index-to-ID mapping.
	//
	// AI-Hints:
	//   - Use this from Result.VertexTour when IDs are absent.
	//   - Do not collapse this into ErrInvalidIDs; missing IDs and malformed IDs are different contracts.
	ErrMissingVertexIDs = errors.New("tsp: vertex ids are missing")

	// Planner/engine governance sentinels.

	// ErrUnsupportedAlgorithm reports an unknown or unavailable top-level algorithm(Options.Algo).
	ErrUnsupportedAlgorithm = errors.New("tsp: unsupported algorithm")

	// ErrTimeLimit reports exhausted wall-clock budget.
	ErrTimeLimit = errors.New("tsp: time limit exceeded")

	// ErrATSPNotSupportedByAlgo reports a symmetric-only algorithm used for ATSP.
	ErrATSPNotSupportedByAlgo = errors.New("tsp: algorithm does not support ATSP")

	// ErrNumericOverflow reports a cost sum that cannot be represented as a finite float64.
	// It does not describe a non-finite input weight or an incomplete route.
	ErrNumericOverflow = errors.New("tsp: route or matching cost overflows float64")

	// ErrSizeOverflow reports an unrepresentable dense allocation or matching capacity.
	// It is distinct from the solver's policy-driven ErrSizeTooLarge limit.
	ErrSizeOverflow = errors.New("tsp: allocation size overflows int")
)
