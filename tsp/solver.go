// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp coordinates validated TSP solver dispatch.
// The solver layer prepares matrices, finalizes explicit options, dispatches one
// selected algorithm, and publishes detached canonical result metadata.
package tsp

import (
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

// preparedMatrix stores the final solver matrix and preparation metadata.
//
// Implementation:
//   - Stage 1: prepareSolverDistanceMatrix validates or copies the input matrix.
//   - Stage 2: Optional metric closure produces a detached dense matrix.
//   - Stage 3: The dispatcher consumes this immutable-by-convention preparation payload.
//
// Behavior highlights:
//   - dist is the final matrix consumed by solver kernels.
//   - n is cached to avoid repeated Rows calls across dispatcher policy.
//   - metricClosureApplied is published into Result.
//
// Inputs:
//   - Produced only by prepareSolverDistanceMatrix.
//
// Returns:
//   - Consumed by solvePreparedMatrix.
//
// Errors:
//   - None; construction errors are returned by prepareSolverDistanceMatrix.
//
// Determinism:
//   - Preserves matrix row/column order.
//
// Complexity:
//   - Storage O(1) plus referenced matrix memory.
//
// Notes:
//   - This type does not own direct caller matrices unless metric closure created a copy.
//
// AI-Hints:
//   - Do not store IDs here; ID validation belongs to the facade boundary.
type preparedMatrix struct {
	// dist is the final matrix consumed by solver kernels.
	// It is either the original caller matrix or a detached metric-closure dense matrix.
	dist matrix.Matrix

	// n is the validated square matrix order.
	// It is cached to avoid repeated Rows calls and argument drift.
	n int

	// metricClosureApplied records whether APSP closure produced dist.
	// The facade publishes this value into Result.
	metricClosureApplied bool
}

// publishKernelResult attaches facade-owned metadata to a result-native solver output.
// It is the only solver-level publisher used after kernels return canonical Result.
//
// Implementation:
//   - Stage 1: Return nil when the selected kernel returned nil.
//   - Stage 2: Clone the kernel result to preserve ownership boundaries.
//   - Stage 3: Attach stable matrix-index IDs.
//   - Stage 4: Attach finalized facade policy and metric-closure metadata.
//
// Behavior highlights:
//   - Does not recompute cost.
//   - Does not canonicalize tour.
//   - Preserves Exact, Optimal, TimedOut, ApproximationRatio, Iterations, and NodesExpanded.
//   - Overrides Algorithm, Symmetric, MetricClosureApplied, and IDs at the facade boundary.
//
// Inputs:
//   - result: result-native solver output.
//   - ids: optional stable matrix-index labels validated by SolveMatrix.
//   - opts: finalized solver options after Auto resolution.
//   - metricClosureApplied: true when APSP closure was applied before solving.
//
// Returns:
//   - *Result: detached canonical facade result.
//
// Errors:
//   - None. Inputs are assumed validated by caller-owned facade stages.
//
// Determinism:
//   - Preserves Tour order exactly.
//   - Preserves IDs order exactly.
//
// Complexity:
//   - Time O(len(result.Tour)+len(ids)), Space O(len(result.Tour)+len(ids)).
//
// Notes:
//   - This helper is a publishing boundary, not an algorithm.
//   - Kernel metadata remains authoritative for exactness, optimality, timeout, ratio, and counters.
//
// AI-Hints:
//   - Do not infer Optimal from Cost.
//   - Do not clear timeout or search telemetry fields.
//   - Do not attach live matrix, graph, or engine references to Result.
func publishKernelResult(result *Result, ids []string, opts Options, metricClosureApplied bool) *Result {
	if result == nil {
		return nil
	}

	published := result.Clone()
	published.Algorithm = opts.Algo
	published.Symmetric = opts.Symmetric
	published.MetricClosureApplied = metricClosureApplied

	if ids != nil {
		published.IDs = append([]string(nil), ids...)
	} else {
		published.IDs = nil
	}

	return published
}

// prepareSolverDistanceMatrix prepares direct matrix input for final TSP solver kernels.
// Implementation:
//   - Stage 1: Validate option policy before allocation.
//   - Stage 2: Validate pre-closure TSP distance semantics with complete=false.
//   - Stage 3: If metric closure is disabled, return the original matrix.
//   - Stage 4: If metric closure is enabled, copy values into a detached Dense matrix.
//   - Stage 5: Run matrix.APSPInPlace to compute Floyd-Warshall closure.
//   - Stage 6: Validate final TSP solver matrix with complete=true.
//
// Behavior highlights:
//   - Does not mutate caller-owned direct matrix inputs.
//   - Uses matrix.APSPInPlace as the single APSP kernel.
//   - Uses TSP validation only for TSP-specific laws: no negative weights and final completeness.
//
// Inputs:
//   - dist: direct distance matrix.
//   - opts: explicit solver policy.
//
// Returns:
//   - matrix.Matrix: original matrix or detached metric-closed matrix.
//   - bool: true when closure was applied.
//   - int: final matrix order.
//   - error: sentinel-classified failure.
//
// Errors:
//   - ErrInvalidOptions from option validation.
//   - ErrNilDistanceMatrix / ErrNonSquare / ErrDimensionMismatch from structure validation.
//   - ErrNaNInf, ErrNonZeroDiagonal, ErrNegativeWeight, ErrIncompleteGraph, ErrAsymmetry.
//   - ErrNegativeWeight joined with matrix.ErrNegativeCycle if APSP detects a negative cycle.
//
// Determinism:
//   - Copy is row-major.
//   - APSP uses matrix.FloydWarshall deterministic k→i→j order.
//   - Final validation uses deterministic row-major / upper-triangle scans.
//
// Complexity:
//   - Without closure: Time O(n^2), Space O(1).
//   - With closure: Time O(n^3), Space O(n^2).
//
// Notes:
//   - This is a policy-stage helper, not a second TSP algorithm.
//   - Graph inputs that already asked matrix.NewAdjacencyMatrix for metric closure should clear
//     RunMetricClosure before delegating to direct matrix solving to avoid double closure.
//
// AI-Hints:
//
//	  final TSP instances reach kernels.
//	- Do not call matrix.InitDistancesInPlace here; direct matrix input already uses
//	  +Inf as the missing-edge sentinel, not raw 0-as-no-edge adjacency.
func prepareSolverDistanceMatrix(dist matrix.Matrix, opts Options) (preparedMatrix, error) {
	if err := validateOptionsStandalone(opts); err != nil {
		return preparedMatrix{}, err
	}

	n, err := validateSolverDistanceMatrix(dist, mustEnforceSymmetry(opts), false, symTol)
	if err != nil {
		return preparedMatrix{}, err
	}

	if !opts.RunMetricClosure {
		n, err = validateSolverDistanceMatrix(dist, mustEnforceSymmetry(opts), true, symTol)
		if err != nil {
			return preparedMatrix{}, err
		}

		return preparedMatrix{dist, n, false}, nil
	}

	closed, err := matrix.NewPreparedDense(n, n, matrix.WithAllowInfDistances())
	if err != nil {
		return preparedMatrix{}, errors.Join(ErrDimensionMismatch, err)
	}

	var (
		row, col int
		value    float64
		readErr  error
	)
	for row = 0; row < n; row++ {
		for col = 0; col < n; col++ {
			value, readErr = dist.At(row, col)
			if readErr != nil {
				return preparedMatrix{}, errors.Join(ErrDimensionMismatch, readErr)
			}

			if row == col {
				value = 0
			}

			if err = closed.Set(row, col, value); err != nil {
				return preparedMatrix{}, errors.Join(ErrNaNInf, err)
			}
		}
	}

	if err = matrix.APSPInPlace(closed); err != nil {
		if errors.Is(err, matrix.ErrNegativeCycle) {
			return preparedMatrix{}, errors.Join(ErrNegativeWeight, err)
		}
		if errors.Is(err, matrix.ErrNaNInf) {
			return preparedMatrix{}, errors.Join(ErrNaNInf, err)
		}

		return preparedMatrix{}, err
	}

	n, err = validateSolverDistanceMatrix(closed, mustEnforceSymmetry(opts), true, symTol)
	if err != nil {
		return preparedMatrix{}, err
	}

	return preparedMatrix{closed, n, true}, nil
}

// solvePreparedMatrix routes an already validated final distance matrix to one solver branch.
// It is the result-native dispatcher used by canonical facades after matrix preparation,
// option finalization, and optional ID validation have already completed.
// Implementation:
//   - Stage 1: Dispatch by opts.Algo.
//   - Stage 2: Run the chosen kernel.
//   - Stage 3: Apply local-search post-pass where the selected policy permits it.
//   - Stage 4: Attach facade metadata to the result-native solver output.
//   - Stage 5: Publish a detached Result with canonical metadata.
//
// Behavior highlights:
//   - Assumes prepareSolverDistanceMatrix and validateIDs already ran.
//   - Does not run metric closure.
//   - Does not mutate caller-owned matrix data.
//   - Does not expose reduced result projections on the internal dispatcher boundary.
//
// Inputs:
//   - dist: final complete solver matrix.
//   - ids: optional matrix-index ordered labels, already validated by SolveMatrix.
//   - opts: final options with RunMetricClosure=false.
//   - n: matrix order.
//   - metricClosureApplied: true when the facade or adapter applied APSP closure.
//
// Returns:
//   - *Result: canonical successful or partial solver result.
//   - error: sentinel-classified failure.
//
// Errors:
//   - Kernel-specific sentinels from Christofides, Held-Karp, local search, and Branch-and-Bound.
//   - ErrUnsupportedAlgorithm for unknown algorithm.
//
// Determinism:
//   - Fixed switch dispatch and kernel-level tie-breaks.
//   - Tour canonicalization is applied before return.
//
// Complexity:
//   - Depends on selected algorithm.
//
// AI-Hints:
//   - Do not call this from public code.
//   - Do not add validation here except final invariant checks; public facades own validation.
//   - High complexity is due to strict option branching and pipeline orchestration.
//   - Splitting this routing logic would make tracking the overall execution flow harder.
//
// nolint:gocyclo
func solvePreparedMatrix(prepared preparedMatrix, ids []string, opts Options) (*Result, error) {
	switch opts.Algo {
	case Christofides:
		result, err := christofides(prepared.dist, opts)
		if result == nil {
			return nil, err
		}

		if opts.EnableLocalSearch && prepared.n >= 4 {
			local, localErr := twoOptKernel(prepared.dist, result.Tour, opts)
			if localErr != nil && !errors.Is(localErr, ErrTimeLimit) {
				return nil, localErr
			}
			attachLocalSearchProgress(result, local)
			if errors.Is(localErr, ErrTimeLimit) {
				return publishKernelResult(result, ids, opts, prepared.metricClosureApplied), ErrTimeLimit
			}

			if opts.BestImprovement {
				local, localErr = threeOptKernel(prepared.dist, result.Tour, opts, true)
				if localErr != nil && !errors.Is(localErr, ErrTimeLimit) {
					return nil, localErr
				}
				attachLocalSearchProgress(result, local)
				if errors.Is(localErr, ErrTimeLimit) {
					return publishKernelResult(result, ids, opts, prepared.metricClosureApplied), ErrTimeLimit
				}

				local, localErr = twoOptKernel(prepared.dist, result.Tour, opts)
				if localErr != nil && !errors.Is(localErr, ErrTimeLimit) {
					return nil, localErr
				}
				attachLocalSearchProgress(result, local)
				if errors.Is(localErr, ErrTimeLimit) {
					return publishKernelResult(result, ids, opts, prepared.metricClosureApplied), ErrTimeLimit
				}
			}
		}

		return publishKernelResult(result, ids, opts, prepared.metricClosureApplied), nil

	case ExactHeldKarp:
		result, err := heldKarp(prepared.dist, opts)
		if result == nil {
			return nil, err
		}

		return publishKernelResult(result, ids, opts, prepared.metricClosureApplied), err

	case TwoOptOnly:
		base, err := trivialRing(prepared.n, opts.StartVertex)
		if err != nil {
			return nil, err
		}

		result, err := twoOptSearch(prepared.dist, base, opts)
		if result == nil {
			return nil, err
		}

		return publishKernelResult(result, ids, opts, prepared.metricClosureApplied), err

	case ThreeOptOnly:
		base, err := trivialRing(prepared.n, opts.StartVertex)
		if err != nil {
			return nil, err
		}

		if opts.EnableLocalSearch && prepared.n >= 4 {
			local, localErr := twoOptKernel(prepared.dist, base, opts)
			if localErr != nil && !errors.Is(localErr, ErrTimeLimit) {
				return nil, localErr
			}
			if local.hasTour() {
				base = local.tour
			}
			if errors.Is(localErr, ErrTimeLimit) {
				result := publishLocalSearchResult(local, ids, opts, ThreeOptOnly, prepared.metricClosureApplied)
				return result, ErrTimeLimit
			}
		}

		result, err := threeOptSearch(prepared.dist, base, opts)
		if result == nil {
			return nil, err
		}

		if opts.EnableLocalSearch && prepared.n >= 4 && !errors.Is(err, ErrTimeLimit) {
			local, localErr := twoOptKernel(prepared.dist, result.Tour, opts)
			if localErr != nil && !errors.Is(localErr, ErrTimeLimit) {
				return nil, localErr
			}

			attachLocalSearchProgress(result, local)
			if errors.Is(localErr, ErrTimeLimit) {
				return publishKernelResult(result, ids, opts, prepared.metricClosureApplied), ErrTimeLimit
			}
		}

		return publishKernelResult(result, ids, opts, prepared.metricClosureApplied), err

	case BranchAndBound:
		result, err := branchAndBound(prepared.dist, opts)
		if result == nil {
			return nil, err
		}

		return publishKernelResult(result, ids, opts, prepared.metricClosureApplied), err

	default:
		return nil, ErrUnsupportedAlgorithm
	}
}

// finalizeSolverOptions resolves size-dependent solver policy after matrix validation.
// Implementation:
//   - Stage 1: Normalize MaxExactN zero value to DefaultMaxExactN.
//   - Stage 2: Resolve Auto into a concrete Algorithm.
//   - Stage 3: Re-run option validation on the concrete policy.
//
// Behavior highlights:
//   - Auto is explicit and opt-in.
//   - Exact algorithms still keep their resource guards.
//   - No hidden algorithm substitution happens when opts.Algo is not Auto.
//
// Inputs:
//   - n: validated matrix order.
//   - opts: user policy after metric-closure flag has been cleared for final solver input.
//
// Returns:
//   - Options: finalized solver policy.
//   - error: sentinel-classified invalid policy.
//
// Errors:
//   - ErrInvalidOptions for invalid MaxExactN.
//   - ErrUnsupportedAlgorithm / ErrATSPNotSupportedByAlgo from final validation.
//
// Determinism:
//   - Pure function of n and opts.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Do not change DefaultOptions to Auto without a compatibility decision.
//   - Do not silently choose heuristics unless opts.Algo == Auto.
func finalizeSolverOptions(n int, opts Options) (Options, error) {
	if opts.MaxExactN < 0 {
		return Options{}, ErrInvalidOptions
	}
	if opts.MaxExactN == 0 {
		opts.MaxExactN = DefaultMaxExactN
	}

	if opts.Algo == Auto {
		opts.Algo = chooseAlgorithm(n, opts)
	}

	if err := validateOptionsStandalone(opts); err != nil {
		return Options{}, err
	}

	return opts, nil
}

// chooseAlgorithm selects a concrete solver for explicit Auto mode.
// Implementation:
//   - Stage 1: Prefer exact Held-Karp when n fits MaxExactN.
//   - Stage 2: Prefer Christofides for symmetric larger instances.
//   - Stage 3: Use TwoOptOnly for asymmetric larger instances.
//
// Behavior highlights:
//   - Deterministic and documented.
//   - Does not override explicit non-Auto algorithms.
//
// Complexity:
//   - Time O(1), Space O(1).
func chooseAlgorithm(n int, opts Options) Algorithm {
	if n <= opts.MaxExactN {
		return ExactHeldKarp
	}
	if opts.Symmetric {
		return Christofides
	}

	return TwoOptOnly
}

// solveDegenerateIfAny handles mathematically valid degenerate TSP instances.
// A one-vertex distance matrix represents the Hamiltonian cycle [0,0] with zero
// total cost; empty matrices remain invalid because they contain no vertex to visit.
//
// Implementation:
//   - Stage 1: Delegate nil, typed-nil, and square shape checks to matrix.ValidateSquare.
//   - Stage 2: Return handled=false for normal n>=2 instances.
//   - Stage 3: Validate the single diagonal value and optional ID mapping.
//   - Stage 4: Publish an exact canonical Result with detached slices.
//
// Behavior highlights:
//   - Does not run metric closure; n==1 closure is a no-op.
//   - Does not allocate algorithm DP/local-search buffers.
//   - Keeps Auto resolution deterministic through finalizeSolverOptions.
//
// Inputs:
//   - dist: direct distance matrix.
//   - ids: optional one-element matrix-index ID mapping.
//   - opts: caller policy already checked by validateOptionsStandalone.
//
// Returns:
//   - *Result: degenerate exact result when handled.
//   - bool: true when n==1 or when shape validation produced a terminal error.
//   - error: nil on valid n==1 or sentinel-classified failure.
//
// Errors:
//   - ErrNilDistanceMatrix joined with matrix.ErrNilMatrix.
//   - ErrNonSquare joined with matrix.ErrNonSquare.
//   - ErrDimensionMismatch for empty matrix or ID length mismatch.
//   - ErrInvalidIDs for empty/duplicate IDs.
//   - ErrStartOutOfRange when StartVertex != 0.
//   - ErrNaNInf joined with matrix.ErrNaNInf for NaN or Inf diagonal.
//   - ErrNonZeroDiagonal when dist[0][0] is not structurally zero.
//
// Determinism:
//   - Fixed single-cell read and fixed `[0,0]` tour publication.
//
// Complexity:
//   - Time O(1), Space O(1) excluding detached result slices.
//
// Notes:
//   - This helper is a facade-level contract guard, not a general solver kernel.
//   - n==2 remains a normal solver instance and passes through regular kernels.
//
// AI-Hints:
//   - Do not route n==1 through Held-Karp or local search.
//   - Do not treat empty matrix as a zero-cost tour.
func solveDegenerateIfAny(dist matrix.Matrix, ids []string, opts Options) (*Result, bool, error) {
	if err := matrix.ValidateSquare(dist); err != nil {
		if errors.Is(err, matrix.ErrNilMatrix) {
			return nil, true, errors.Join(ErrNilDistanceMatrix, err)
		}
		if errors.Is(err, matrix.ErrNonSquare) {
			return nil, true, errors.Join(ErrNonSquare, err)
		}

		return nil, true, errors.Join(ErrDimensionMismatch, err)
	}

	n := dist.Rows()
	if n == 0 {
		return nil, true, ErrDimensionMismatch
	}
	if n != 1 {
		return nil, false, nil
	}
	if opts.StartVertex != 0 {
		return nil, true, ErrStartOutOfRange
	}
	if ids != nil {
		if err := validateIDs(ids, n); err != nil {
			return nil, true, err
		}
	}

	value, err := dist.At(0, 0)
	if err != nil {
		return nil, true, errors.Join(ErrDimensionMismatch, err)
	}
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return nil, true, errors.Join(ErrNaNInf, matrix.ErrNaNInf)
	}
	if math.Abs(value) > symTol {
		return nil, true, ErrNonZeroDiagonal
	}

	finalOptions := opts
	finalOptions.RunMetricClosure = false

	finalOptions, err = finalizeSolverOptions(n, finalOptions)
	if err != nil {
		return nil, true, err
	}

	return &Result{
		Tour:                 []int{0, 0},
		Cost:                 0,
		IDs:                  append([]string(nil), ids...),
		Algorithm:            finalOptions.Algo,
		Exact:                true,
		Optimal:              true,
		TimedOut:             false,
		MetricClosureApplied: false,
		Symmetric:            finalOptions.Symmetric,
		ApproximationRatio:   NoApproximationRatio,
	}, true, nil
}

// trivialRing returns the canonical Hamiltonian cycle that walks vertices in increasing
// modular order from start and appends the closing start vertex. It is used for tiny
// or degenerate solver paths that do not need matrix inspection.
//
// Implementation:
//   - Stage 1: Validate n and start bounds.
//   - Stage 2: Allocate exactly n+1 integers.
//   - Stage 3: Fill vertices start,start+1,...,n-1,0,...,start-1.
//   - Stage 4: Append start as the closing vertex.
//
// Behavior highlights:
//   - Performs no matrix lookups.
//   - Does not compute cost.
//   - Produces deterministic output for a fixed n/start.
//
// Inputs:
//   - n: number of local vertices, requiring n >= 2.
//   - start: local start vertex, requiring 0 <= start < n.
//
// Returns:
//   - []int: closed Hamiltonian cycle of length n+1.
//   - error: nil when inputs are valid.
//
// Errors:
//   - ErrDimensionMismatch for n < 2.
//   - ErrInvalidVertex for invalid start.
//
// Determinism:
//   - Fixed increasing modular order.
//
// Complexity:
//   - Time O(n), Space O(n).
//
// Notes:
//   - Caller must compute cost separately if needed.
//
// AI-Hints:
//   - Do not inspect the distance matrix here; this is a structural helper only.
func trivialRing(n int, start int) ([]int, error) {
	if n < 2 {
		return nil, ErrDimensionMismatch
	}
	if start < 0 || start >= n {
		return nil, ErrStartOutOfRange
	}
	out := make([]int, n+1)

	var (
		i   int // loop iterator
		pos = 0 // independent index of the entry into the resulting slice.
	)

	// Fill from start to n-1.
	for i = start; i < n; i++ {
		out[pos] = i
		pos++
	}
	// Then wrap from 0 to start-1.
	for i = 0; i < start; i++ {
		out[pos] = i
		pos++
	}

	// Close the cycle by returning to start.
	out[n] = start

	return out, nil
}
