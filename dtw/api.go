// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw exposes public Dynamic Time Warping facades.
// Facades assemble policy, validate adapter-level inputs, and delegate to canonical kernels.
package dtw

import (
	"github.com/lvlath/go/matrix"
)

// Align computes Dynamic Time Warping for two scalar sequences.
//
// What:
//   - Align is the canonical scalar DTW facade.
//   - It returns a Result artifact carrying distance, reachability, path state,
//     and optional matrix artifacts.
//
// Implementation:
//   - Stage 1: assemble and finalize functional options.
//   - Stage 2: validate scalar inputs and numeric policy inside compute.
//   - Stage 3: compute the row-major DTW recurrence.
//   - Stage 4: classify no-path before backtracking.
//   - Stage 5: publish a detached Result artifact.
//
// Behavior highlights:
//   - Window == -1 disables the Sakoe-Chiba band.
//   - Window >= 0 enforces |i-j| <= Window.
//   - Window == 0 is strict diagonal alignment.
//   - +Inf distance with Reachable=false means no admissible path.
//   - WithReturnPath(true) automatically uses FullMatrix storage.
//
// Inputs:
//   - a: first scalar sequence; must be non-nil and non-empty.
//   - b: second scalar sequence; must be non-nil and non-empty.
//   - opts: explicit policy switches; nil options are rejected.
//
// Returns:
//   - *Result: canonical detached result artifact.
//
// Errors:
//   - ErrNilOption from option assembly.
//   - ErrInvalidWindow / ErrInvalidPenalty joined with ErrBadInput.
//   - ErrNilInput / ErrEmptyInput from sequence validation.
//   - ErrNaNInf from finite-input or local-cost validation.
//   - ErrNoPath when path tracking is requested but no admissible path exists.
//
// Determinism:
//   - DP traversal is row-major: i ascending, then j ascending.
//   - Path tie-break is diagonal first, then vertical/up, then horizontal/left.
//
// Complexity:
//   - Time O(n*m), where n=len(a), m=len(b).
//   - Space O(m) for distance-only rolling rows.
//   - Space O(n*m) when path or accumulated matrix storage is required.
//
// Notes:
//   - Align does not mutate input slices.
//   - Concurrent mutation of input slices during Align is unsupported.
//
// AI-Hints:
//   - Do not reintroduce pointer-based options into the canonical facade.
//   - Do not treat nil Result and unreachable Result as the same state.
func Align(a, b []float64, opts ...Option) (*Result, error) {
	cfg, err := applyOptions(opts...)
	if err != nil {
		return nil, err
	}

	return compute(a, b, cfg)
}

// AlignCostMatrix computes DTW over a precomputed local-cost matrix.
//
// What:
//   - AlignCostMatrix treats local.Rows() as the number of steps in sequence A.
//   - It treats local.Cols() as the number of steps in sequence B.
//   - local.At(i,j) is the finite non-negative local cost for aligning A[i] with B[j].
//
// Implementation:
//   - Stage 1: assemble and finalize DTW options.
//   - Stage 2: validate the local-cost matrix shape and numeric contract.
//   - Stage 3: materialize local costs into a row-major []float64 for hot-loop purity.
//   - Stage 4: delegate to computeFromFlatLocal.
//   - Stage 5: optionally publish a detached local-cost Dense artifact.
//
// Behavior highlights:
//   - This facade is the adapter for domain-specific or externally computed costs.
//   - SlopePenalty, Window, ReturnPath, and matrix artifacts work exactly as in Align.
//   - CostFunc is ignored because local costs are already supplied.
//
// Inputs:
//   - local: matrix.Matrix with rows>0, cols>0, finite non-negative values.
//   - opts: explicit DTW policy switches.
//
// Returns:
//   - *Result: canonical detached result artifact.
//
// Errors:
//   - ErrNilOption from option assembly.
//   - ErrBadInput joined with matrix validation/access errors.
//   - ErrEmptyInput when local has zero rows or zero columns.
//   - ErrNaNInf when any local cost is NaN or ±Inf.
//   - ErrNegativeCost when any local cost is negative.
//   - ErrNoPath when path tracking is requested but no admissible path exists.
//
// Determinism:
//   - Matrix values are read in row-major order.
//   - DP traversal is row-major.
//   - Path tie-break is diagonal first, then vertical/up, then horizontal/left.
//
// Complexity:
//   - Validation/materialization Time O(n*m), Space O(n*m).
//   - DP Time O(n*m), Space O(m) or O(n*m) depending on options.
//
// Notes:
//   - local is never mutated.
//   - Result.LocalCost is a detached Dense copy only when WithReturnLocalCost(true) is set.
//
// AI-Hints:
//   - Do not call CostFunc in this facade; the matrix already is the local-cost source.
//   - Do not keep aliases to caller-owned matrix storage in Result.
func AlignCostMatrix(local matrix.Matrix, opts ...Option) (*Result, error) {
	cfg, err := applyOptions(opts...)
	if err != nil {
		return nil, err
	}

	costs, rows, cols, err := materializeLocalCostMatrix(local)
	if err != nil {
		return nil, err
	}

	res, err := computeFromFlatLocal(costs, rows, cols, cfg)
	if res != nil && cfg.returnLocalCost {
		localCopy, cloneErr := denseCopyOfMatrix(local)
		if cloneErr != nil {
			return res, cloneErr
		}
		res.LocalCost = localCopy
	}

	return res, err
}

// AlignMatrix computes multivariate DTW using squared L2 row-to-row local costs.
//
// What:
//   - AlignMatrix treats each row of x and y as one time step.
//   - Columns are feature dimensions and must match.
//   - The local cost is squared Euclidean distance between rows.
//
// Implementation:
//   - Stage 1: assemble and finalize DTW options.
//   - Stage 2: compute a matrix.Dense local-cost matrix using matrix operations.
//   - Stage 3: delegate to AlignCostMatrix-compatible flat local-cost DP.
//   - Stage 4: publish the local-cost artifact when requested.
//
// Behavior highlights:
//   - This is the canonical matrix-backed multivariate DTW facade.
//   - It reuses matrix validation, multiplication, elementwise operations, and row sums.
//   - It clamps tiny negative squared distances caused by floating-point roundoff to zero.
//
// Inputs:
//   - x: matrix where rows are time steps and columns are features.
//   - y: matrix where rows are time steps and columns are features.
//   - opts: explicit DTW policy switches.
//
// Returns:
//   - *Result: canonical detached result artifact.
//
// Errors:
//   - ErrBadInput joined with matrix errors for nil, shape, access, or operation failures.
//   - ErrNaNInf joined with matrix finite-validation failures.
//   - matrix.ErrDimensionMismatch is preserved when feature dimensions differ.
//   - ErrNoPath when path tracking is requested but no admissible path exists.
//
// Determinism:
//   - Matrix local costs are generated in row-major order.
//   - DP traversal and backtracking use the same laws as Align.
//
// Complexity:
//   - Local-cost construction is O(nx*ny*d) conceptually, where d is feature count.
//   - DP is O(nx*ny).
//   - Space is O(nx*ny) for the local-cost matrix plus optional accumulated storage.
//
// Notes:
//   - AlignMatrix uses squared L2 costs, not Euclidean L2 with sqrt.
//   - Use matrix preprocessing functions before calling AlignMatrix when normalization is needed.
//
// AI-Hints:
//   - Do not mention AlignMatrix in docs/examples unless this facade exists.
//   - Do not silently normalize inputs here; preprocessing belongs to callers.
func AlignMatrix(x, y matrix.Matrix, opts ...Option) (*Result, error) {
	cfg, err := applyOptions(opts...)
	if err != nil {
		return nil, err
	}

	local, err := localSquaredL2Matrix(x, y)
	if err != nil {
		return nil, err
	}

	costs, rows, cols, err := materializeLocalCostMatrix(local)
	if err != nil {
		return nil, err
	}

	res, err := computeFromFlatLocal(costs, rows, cols, cfg)
	if res != nil && cfg.returnLocalCost {
		res.LocalCost = local
	}

	return res, err
}

// DTW computes DTW distance through the legacy pointer-options API.
//
// What:
//   - DTW preserves backward compatibility with the previous public signature.
//   - New code should prefer Align and the Result artifact.
//
// Implementation:
//   - Stage 1: convert legacy *Options into finalized internal options.
//   - Stage 2: delegate to the same scalar kernel used by Align.
//   - Stage 3: project Result into the legacy typed-value return shape.
//
// Behavior highlights:
//   - nil legacy options mean DefaultOptions.
//   - ReturnPath still requires MemoryMode == FullMatrix in the legacy API.
//   - The wrapper does not introduce alternative mathematics.
//
// Inputs:
//   - a: first scalar sequence.
//   - b: second scalar sequence.
//   - legacy: legacy options pointer; nil means defaults.
//
// Returns:
//   - dist: Result.Distance.
//   - path: detached path only when path tracking is enabled.
//   - error: sentinel-classified failure.
//
// Errors:
//   - ErrPathNeedsMatrix for legacy ReturnPath without FullMatrix.
//   - ErrNilInput / ErrEmptyInput / ErrNaNInf from the shared kernel.
//   - ErrNoPath may be returned with dist=+Inf when path was requested.
//
// Determinism:
//   - Same as Align because this wrapper delegates to the same kernel.
//
// Complexity:
//   - Same as Align for the finalized storage mode.
//
// Notes:
//   - Deprecated: use Align with WithXxx options and Result helper methods.
//   - Compatibility requires nil legacy options to remain accepted as defaults.
//
// AI-Hints:
//   - Do not add special-case math here; this wrapper must stay a projection of Align.
func DTW(a, b []float64, legacy *Options) (dist float64, path []Coord, err error) {
	cfg, err := optionsFromLegacy(legacy)
	if err != nil {
		return 0, nil, err
	}

	res, err := compute(a, b, cfg)
	if res == nil {
		return 0, nil, err
	}

	if res.PathTracked && res.Path != nil {
		path = make([]Coord, len(res.Path))
		copy(path, res.Path)
	}

	return res.Distance, path, err
}
