// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp validates solver policy, distance matrices, and auxiliary TSP inputs.
// Validation helpers are deterministic firewalls: they reject invalid public input
// before expensive solver allocation and preserve sentinel error identity for callers.
//
// Design principles:
//   - Deterministic, side-effect-free functions.
//   - No logging and no panics on user input; failures are classified by sentinels from errors.go.
//   - Matrix validation is O(n^2) worst-case and uses no hidden heap allocation.
package tsp

import (
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

// symTol is the structural tolerance for diagonal and symmetry checks in TSP distance matrices.
// It is intentionally stricter than matrix.DefaultEpsilon because TSP contracts usually consume
// already prepared costs, not noisy floating-point measurements from algebraic kernels.
const symTol = 1e-12

// validateOptionsStandalone checks solver policy without matrix-size-dependent finalization.
//
// Implementation:
//   - Stage 1: Validate numeric knobs.
//   - Stage 2: Validate enum domains.
//   - Stage 3: Validate cross-policy constraints that are already knowable.
//   - Stage 4: Accept Auto as an explicit unresolved algorithm mode.
//
// Behavior highlights:
//   - No allocations.
//   - Auto is valid here, but must be finalized after matrix order is known.
//   - No hidden matching-substitution policy is accepted.
//   - No silent normalization of invalid public options.
//
// Inputs:
//   - opts: solver configuration assembled by the caller.
//
// Returns:
//   - error: nil when options are structurally valid.
//
// Errors:
//   - ErrInvalidOptions for invalid numeric knobs or enum values.
//   - ErrATSPNotSupportedByAlgo for known symmetric-only modes.
//   - ErrUnsupportedAlgorithm for unknown Algorithm.
//
// Determinism:
//   - Pure O(1) validation; no randomness and no state mutation.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This function does not normalize options; it only rejects invalid public input.
//
// AI-Hints:
//   - Do not add hidden matching-substitution policy.
//   - Tests must classify invalid options with errors.Is(err, ErrInvalidOptions).
func validateOptionsStandalone(opts Options) error {
	// TimeLimit must be non-negative (negative durations are undefined).
	if opts.TimeLimit < 0 {
		return ErrInvalidOptions
	}
	// Eps is the acceptance tolerance for Δ<−Eps. A negative epsilon would invert
	// the acceptance logic and break optimality guarantees ⇒ reject.
	if math.IsNaN(opts.Eps) || math.IsInf(opts.Eps, 0) || opts.Eps < 0 {
		return ErrInvalidOptions
	}
	// TwoOpt iteration bound must be non-negative (0 ⇒ unlimited).
	if opts.TwoOptMaxIters < 0 {
		return ErrInvalidOptions
	}
	// ThreeOpt uses a larger neighborhood and therefore owns a separate move cap.
	if opts.ThreeOptMaxMoves < 0 {
		return ErrInvalidOptions
	}
	if opts.MaxExactN < 0 {
		return ErrInvalidOptions
	}

	switch opts.MatchingAlgo {
	case GreedyMatch, BlossomMatch:
	default:
		return ErrInvalidOptions
	}

	switch opts.BoundAlgo {
	case NoBound, SimpleBound, OneTreeBound:
	default:
		return ErrInvalidOptions
	}

	// Christofides requires a symmetric (metric) TSP instance.
	if opts.Algo == Christofides && !opts.Symmetric {
		return ErrATSPNotSupportedByAlgo
	}

	// OneTreeBound is symmetric-only (Held–Karp 1-tree).
	if opts.BoundAlgo == OneTreeBound && !opts.Symmetric {
		return ErrATSPNotSupportedByAlgo
	}

	// Accept only known algorithms; dispatcher may still return a runtime sentinel later.
	switch opts.Algo {
	case Auto, Christofides, ExactHeldKarp, TwoOptOnly, ThreeOptOnly, BranchAndBound:
		return nil
	default:
		return ErrUnsupportedAlgorithm
	}
}

// mustEnforceSymmetry reports whether the selected algorithm requires a symmetric
// distance matrix at validation time. It centralizes algorithm-specific symmetry policy
// so solver entry points do not drift apart.
//
// Implementation:
//   - Stage 1: Inspect opts.Algo and related algorithm flags.
//   - Stage 2: Return true for algorithms with a mathematical symmetry requirement.
//   - Stage 3: Return false for algorithms that support directed/asymmetric costs.
//
// Behavior highlights:
//   - Does not inspect matrix values.
//   - Does not validate metric triangle inequality.
//   - Keeps Christofides strictly symmetric.
//   - Keeps Held-Karp and local-search-only paths permissive when supported.
//
// Inputs:
//   - opts: solver options after defaulting.
//
// Returns:
//   - bool: true when matrix symmetry must be enforced.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure switch over options.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Christofides still needs metric assumptions for the 1.5 guarantee;
//     this helper only controls symmetry validation.
//
// AI-Hints:
//   - Do not weaken Christofides symmetry here.
//   - Do not add O(n^2) matrix checks to this predicate.
func mustEnforceSymmetry(opts Options) bool {
	if opts.Algo == Christofides {
		return true
	}

	return opts.Symmetric
}

// validateStartVertex verifies that start is a valid local vertex in [0,n).
// It is shared by solvers, tour normalization, and graph traversal entry points.
//
// Implementation:
//   - Stage 1: Validate positive dimension.
//   - Stage 2: Check lower and upper bounds for start.
//
// Behavior highlights:
//   - Does not inspect the distance matrix.
//   - Does not rewrite invalid starts.
//   - Preserves sentinel error semantics for public APIs.
//
// Inputs:
//   - n: number of local vertices.
//   - start: requested local start vertex.
//
// Returns:
//   - error: nil when start is in range.
//
// Errors:
//   - ErrDimensionMismatch for n <= 0.
//   - ErrInvalidVertex for start outside [0,n).
//
// Determinism:
//   - Pure bounds check.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Call after matrix dimension validation when possible.
//
// AI-Hints:
//   - Do not silently clamp start to zero.
func validateStartVertex(n int, start int) error {
	if start < 0 || start >= n {
		return ErrStartOutOfRange
	}

	return nil
}

// validateIDs verifies optional vertex IDs attached to matrix rows/columns.
// Implementation:
//   - Stage 1: verify exact shape match len(ids)==n.
//   - Stage 2: scan IDs once and reject empty or duplicate labels.
//
// Behavior highlights:
//   - Keeps index-to-ID mapping deterministic and lossless.
//   - Does not sort; input order is the matrix row/column order.
//
// Inputs:
//   - ids: optional matrix row/column labels.
//   - n: matrix order.
//
// Returns:
//   - error: nil when IDs are valid.
//
// Errors:
//   - ErrDimensionMismatch when len(ids)!=n.
//   - ErrInvalidIDs for empty or duplicated IDs.
//
// Determinism:
//   - Fixed increasing index scan.
//
// Complexity:
//   - Time O(n), Space O(n).
//
// AI-Hints:
//   - Do not collapse empty/duplicate IDs into ErrDimensionMismatch.
func validateIDs(ids []string, n int) error {
	if len(ids) != n {
		return ErrDimensionMismatch
	}

	seen := make(map[string]struct{}, n)

	var (
		i  int    // loop index
		id string // current ID under validation
		ok bool   // presence flag in the 'seen' set
	)
	for i = 0; i < n; i++ { // scan each ID
		id = ids[i] // read ID at position i
		// Empty or duplicate IDs violate the shape/uniqueness contract.
		if id == "" {
			return ErrInvalidIDs
		}
		if _, ok = seen[id]; ok {
			return ErrInvalidIDs
		}
		seen[id] = struct{}{} // mark ID as seen
	}

	return nil
}

// validateSolverDistanceMatrix validates the TSP-specific distance-matrix contract.
// It relies on matrix.ValidateSquare for canonical nil / typed-nil / square checks,
// then performs exactly the domain checks that are specific to TSP solvers.
//
// Implementation:
//   - Stage 1: Delegate nil, typed-nil, and square shape validation to matrix.ValidateSquare.
//   - Stage 2: Reject unsupported degenerate size n<2.
//   - Stage 3: Scan values in deterministic row-major order.
//   - Stage 4: Enforce TSP numeric policy: diagonal≈0, no NaN/-Inf, no negative weights.
//   - Stage 5: Enforce final completeness (+Inf forbidden) and optional symmetry.
//
// Behavior highlights:
//   - Does not duplicate matrix structural validation.
//   - Does not use matrix.ValidateDistanceMatrix as the full validator because APSP permits
//     finite negative off-diagonal distances, while TSP forbids them.
//   - Preserves matrix sentinel identity with errors.Join when failures originate in matrix.
//
// Inputs:
//   - dist: matrix.Matrix distance model consumed by TSP kernels.
//   - symmetric: whether dist[i][j] must match dist[j][i] within symTol.
//   - complete: whether +Inf off-diagonal values must be rejected as ErrIncompleteGraph.
//   - tol: non-negative structural tolerance for diagonal and symmetry checks.
//
// Returns:
//   - int: matrix order n on success.
//   - error: nil on success or a sentinel-classified validation failure.
//
// Errors:
//   - ErrNilDistanceMatrix joined with matrix.ErrNilMatrix from matrix.ValidateSquare.
//   - ErrNonSquare joined with matrix.ErrNonSquare from matrix.ValidateSquare.
//   - ErrDimensionMismatch when n<2.
//   - ErrNaNInf joined with matrix.ErrNaNInf for NaN or -Inf.
//   - ErrNonZeroDiagonal for non-zero self-distance.
//   - ErrNegativeWeight for negative finite off-diagonal weights.
//   - ErrIncompleteGraph for +Inf off-diagonal when complete=true.
//   - ErrAsymmetry for symmetry violations.
//
// Determinism:
//   - Fixed row-major value scan.
//   - Fixed upper-triangle symmetry scan.
//   - The first failing position is deterministic through the loop order.
//
// Complexity:
//   - Time O(n^2), Space O(1).
//
// Notes:
//   - complete=false is valid only before direct-matrix metric closure.
//   - complete=true is mandatory before any final TSP solver kernel.
//
// AI-Hints:
//   - Do not replace this with matrix.ValidateDistanceMatrix only; that would allow
//     negative APSP distances that TSP must reject.
//   - Do not map final +Inf to ErrNaNInf; in TSP it means the Hamiltonian instance is incomplete.
func validateSolverDistanceMatrix(
	dist matrix.Matrix,
	symmetric bool,
	complete bool,
	tol float64,
) (int, error) {
	if err := matrix.ValidateSquare(dist); err != nil {
		if errors.Is(err, matrix.ErrNilMatrix) {
			return 0, errors.Join(ErrNilDistanceMatrix, err)
		}
		if errors.Is(err, matrix.ErrNonSquare) {
			return 0, errors.Join(ErrNonSquare, err)
		}

		return 0, errors.Join(ErrDimensionMismatch, err)
	}

	n := dist.Rows()
	if n < 2 {
		return 0, ErrDimensionMismatch
	}

	var (
		row     int
		col     int
		value   float64
		mirror  float64
		readErr error
		absDiff float64
		absDiag float64
	)

	for row = 0; row < n; row++ {
		for col = 0; col < n; col++ {
			value, readErr = dist.At(row, col)
			if readErr != nil {
				return 0, errors.Join(ErrDimensionMismatch, readErr)
			}

			if math.IsNaN(value) || math.IsInf(value, -1) {
				return 0, errors.Join(ErrNaNInf, matrix.ErrNaNInf)
			}

			if row == col {
				if math.IsInf(value, 1) {
					return 0, ErrNonZeroDiagonal
				}

				absDiag = math.Abs(value)
				if absDiag > tol {
					return 0, ErrNonZeroDiagonal
				}

				continue
			}

			if value < 0 {
				return 0, ErrNegativeWeight
			}

			if complete && math.IsInf(value, 1) {
				return 0, ErrIncompleteGraph
			}
		}
	}

	if symmetric {
		for row = 0; row < n; row++ {
			for col = row + 1; col < n; col++ {
				value, readErr = dist.At(row, col)
				if readErr != nil {
					return 0, errors.Join(ErrDimensionMismatch, readErr)
				}

				mirror, readErr = dist.At(col, row)
				if readErr != nil {
					return 0, errors.Join(ErrDimensionMismatch, readErr)
				}

				absDiff = math.Abs(value - mirror)
				if absDiff > tol {
					return 0, ErrAsymmetry
				}
			}
		}
	}

	return n, nil
}
