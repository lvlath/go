// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw implements classified accumulated-cost states and deterministic
// predecessor tracking. This file owns the common DTW recurrence for all input adapters.
package dtw

import (
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/matrix"
)

// maxIntValue bounds row-major size arithmetic before any memory allocation.
// It depends only on the machine word size and is not a configurable limit.
const maxIntValue = int(^uint(0) >> 1)

// dpClass separates structural absence from an attainable cost outside float64.
// A finite candidate always dominates an overflow candidate; neither represents absence.
type dpClass uint8

const (
	// dpUnreachable means there is no legal prefix path to this cell.
	dpUnreachable dpClass = iota
	// dpFinite means the optimal prefix cost is finite and non-negative.
	dpFinite
	// dpOverflow means a legal prefix exists but its optimum is unrepresentable.
	dpOverflow
)

// predecessorCode records the chosen incoming transition at the time of comparison.
// Codes are private because a path is the only public witness of the recurrence.
type predecessorCode uint8

const (
	// predecessorMissing is stored for structurally absent or not-yet-computed cells.
	predecessorMissing predecessorCode = iota
	// predecessorOrigin marks the first consumed pair without an incoming step.
	predecessorOrigin
	// predecessorDiagonal consumes one sample from both sequences.
	predecessorDiagonal
	// predecessorUp advances sequence A and incurs SlopePenalty.
	predecessorUp
	// predecessorLeft advances sequence B and incurs SlopePenalty.
	predecessorLeft
)

// dpCell carries a classified DP state without conflating +Inf and missing paths.
// cost is meaningful only for dpFinite, where it is finite and non-negative.
type dpCell struct {
	// cost stores an ordinary rounded binary64 accumulated cost only for dpFinite.
	cost float64
	// class records reachability/range independently of numeric representation.
	class dpClass
}

// costSource provides one finite, non-negative local cost for a zero-based cell.
// A scalar source invokes CostFunc; a flat-matrix source indexes validated storage.
type costSource func(i, j int) (float64, error)

// checkedNonnegativeAdd adds two finite, non-negative float64 costs.
// It distinguishes an exact sum above MaxFloat64 from a finite representable sum,
// including the boundary where rounding would otherwise conceal overflow.
//
// Implementation:
//   - Stage 1: perform the ordinary IEEE-754 binary64 addition, preserving its rounding.
//   - Stage 2: classify infinity as overflow, never as structural unreachability.
//   - Stage 3: at MaxFloat64, use the error-free TwoSum residual to detect a real
//     exact sum larger than the largest finite binary64 number.
//
// Behavior highlights:
//   - The finite result follows Go float64 addition, not exact-real path arithmetic.
//   - The boundary test does not depend on a rounded MaxFloat64-a subtraction.
//
// Inputs:
//   - a, b: finite, non-negative accumulated or increment costs in cost units.
//
// Returns:
//   - float64: ordinarily rounded sum if representable.
//   - bool: true only if the exact sum exceeds the finite float64 range.
//
// Errors:
//   - None; callers must establish the finite, non-negative precondition.
//
// Determinism:
//   - Fixed addition and TwoSum evaluation order, no tolerance-based comparison.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - TwoSum is exact for finite IEEE-754 operations with gradual underflow.
//   - Checked accumulation is not an arbitrary-precision shortest-path mode.
//
// AI-Hints:
//   - Testing only math.IsInf(a+b) misses exact overflow rounded to MaxFloat64.
//   - Do not replace the boundary check with b > MaxFloat64-a; that subtraction rounds.
func checkedNonnegativeAdd(a, b float64) (float64, bool) {
	sum := a + b
	if math.IsInf(sum, 1) {
		return 0, true
	}
	if sum != math.MaxFloat64 || a == 0 || b == 0 {
		return sum, false
	}

	// Recover the exact rounding residual without introducing a high-precision
	// allocation in the normal DP hot path.
	intermediate := sum - a
	residual := (a - (sum - intermediate)) + (b - intermediate)
	return sum, residual > 0
}

// addCost extends a reachable DP prefix by one non-negative finite cost.
// The classification is monotone because adding non-negative costs cannot make
// an unrepresentable prefix finite again.
//
// Implementation:
//   - Stage 1: preserve unreachable and overflow states without arithmetic.
//   - Stage 2: add the increment to a finite prefix with checkedNonnegativeAdd.
//   - Stage 3: publish the finite or overflow class.
//
// Behavior highlights:
//   - No +Inf value is stored in the cost field of a finite state.
//
// Inputs:
//   - increment: finite non-negative cost, with the same units as cell.cost.
//
// Returns:
//   - dpCell: classified, non-aliased extended state.
//
// Errors:
//   - None; local costs and penalty are validated before entering this method.
//
// Determinism:
//   - Exactly one checked addition when the input cell is finite.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This method implements saturation into a class, not a finite clamp.
//
// AI-Hints:
//   - An overflow prefix is reachable; never silently change it to dpUnreachable.
//   - Only the selected final overflow warrants ErrOverflow for the alignment.
func (cell dpCell) addCost(increment float64) dpCell {
	if cell.class != dpFinite {
		return cell
	}

	sum, overflow := checkedNonnegativeAdd(cell.cost, increment)
	if overflow {
		return dpCell{class: dpOverflow}
	}
	return dpCell{cost: sum, class: dpFinite}
}

// candidateBeats reports whether a new predecessor is strictly better than the current one.
// A finite cost wins over overflow, while equal candidates keep their earlier
// diagonal-to-up-to-left selection without epsilon relaxation.
//
// Implementation:
//   - Stage 1: discard structurally unreachable candidates.
//   - Stage 2: compare finite and overflow classes before numeric values.
//   - Stage 3: compare finite values with strict less-than only.
//
// Behavior highlights:
//   - Equal finite values and two overflow values never replace the earlier choice.
//
// Inputs:
//   - candidate: the newly considered transition's classified cost.
//   - current: the best transition seen so far.
//
// Returns:
//   - bool: true if and only if candidate strictly improves current.
//
// Errors:
//   - None; candidate construction establishes the state invariants.
//
// Determinism:
//   - Numeric equality keeps the earlier predecessor.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Class priority is finite, then overflow, then unreachable.
//
// AI-Hints:
//   - Never use tieEpsilon to select a strictly more expensive predecessor.
//   - An unreachable state is not an overflow state.
func candidateBeats(candidate, current dpCell) bool {
	if candidate.class == dpUnreachable {
		return false
	}
	if current.class == dpUnreachable {
		return true
	}
	if candidate.class == dpFinite {
		return current.class == dpOverflow || (current.class == dpFinite && candidate.cost < current.cost)
	}
	return false
}

// choosePredecessor evaluates all three legal transitions for one admitted cell.
// It preserves the recurrence's operation order: penalty is added to up/left
// predecessor first, then the selected predecessor receives the local cost.
//
// Implementation:
//   - Stage 1: consider diagonal without slope penalty.
//   - Stage 2: consider up plus penalty, updating only on strict improvement.
//   - Stage 3: consider left plus penalty, updating only on strict improvement.
//
// Behavior highlights:
//   - A discarded overflowing transition never fails an otherwise finite optimum.
//
// Inputs:
//   - diagonal, up, left: classified accumulated prefix costs.
//   - penalty: finite non-negative cost applied only to up and left.
//
// Returns:
//   - dpCell: selected predecessor including its transition penalty.
//   - predecessorCode: selected transition, or predecessorMissing if none exists.
//
// Errors:
//   - None; invalid options are rejected before the kernel.
//
// Determinism:
//   - Strict diagonal, up, left order; exact finite ties keep the first.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The local cost is intentionally added by the caller after selection.
//
// AI-Hints:
//   - Do not fail at a discarded overflow branch.
//   - Do not use approximate equality for predecessor selection.
func choosePredecessor(diagonal, up, left dpCell, penalty float64) (dpCell, predecessorCode) {
	best := diagonal
	code := predecessorDiagonal
	if best.class == dpUnreachable {
		code = predecessorMissing
	}

	up = up.addCost(penalty)
	if candidateBeats(up, best) {
		best, code = up, predecessorUp
	}
	left = left.addCost(penalty)
	if candidateBeats(left, best) {
		best, code = left, predecessorLeft
	}
	return best, code
}

// backtrackPredecessors materializes the exact forward-DP path witness.
// No accumulated-cost subtraction or user callback is performed during backtrack,
// so cancellation and approximate equality cannot rewrite the selected optimum.
//
// Implementation:
//   - Stage 1: start at the terminal zero-based cell and inspect saved codes.
//   - Stage 2: append each selected cell, moving backward by the recorded step.
//   - Stage 3: reverse once to return the start-to-end coordinate order.
//
// Behavior highlights:
//   - The origin is a distinct code and is consumed exactly once.
//
// Inputs:
//   - predecessors: row-major private codes for a rows-by-cols DP grid.
//   - rows, cols: positive sequence lengths, with rows*cols allocated codes.
//
// Returns:
//   - Path: caller-owned monotone start-to-end witness.
//   - error: ErrIncompletePath on a broken private predecessor chain.
//
// Errors:
//   - ErrIncompletePath: missing/invalid code or a step leaving valid coordinates.
//
// Determinism:
//   - Follows previously chosen codes; no tie-breaking during reconstruction.
//
// Complexity:
//   - Time O(rows+cols), Space O(rows+cols) for the returned path.
//
// Notes:
//   - The predecessor buffer is private and never aliases Result.Path.
//
// AI-Hints:
//   - Never infer a predecessor from accumulated[current]-localCost.
//   - Never invoke CostFunc during backtracking.
func backtrackPredecessors(predecessors []predecessorCode, rows, cols int) (Path, error) {
	if rows <= 0 || cols <= 0 || len(predecessors) != rows*cols {
		return nil, ErrIncompletePath
	}
	i, j := rows-1, cols-1
	path := make(Path, 0, rows+cols-1)

	for i >= 0 && j >= 0 {
		path = append(path, Coord{I: i, J: j})
		switch predecessors[i*cols+j] {
		case predecessorOrigin:
			if i != 0 || j != 0 {
				return nil, ErrIncompletePath
			}
			reversePath(path)
			return path, nil
		case predecessorDiagonal:
			i--
			j--
		case predecessorUp:
			i--
		case predecessorLeft:
			j--
		default:
			return nil, ErrIncompletePath
		}
	}
	return nil, ErrIncompletePath
}

// computeFromCostSource runs the canonical classified DTW recurrence for any local-cost adapter.
// It evaluates scalar callbacks only for admitted cells unless the full local
// artifact was explicitly requested, in which case every cell is evaluated once.
//
// Implementation:
//   - Stage 1: allocate rolling classified states, optional dense artifacts, and
//     optional predecessor codes; materialize the complete local artifact if requested.
//   - Stage 2: traverse admitted cells in row-major order, compare classified
//     transitions strictly, and save the chosen incoming step during forward DP.
//   - Stage 3: classify terminal reachability versus numeric overflow, then
//     publish detached artifacts and optionally reconstruct the saved witness.
//
// Behavior highlights:
//   - Origin (0,0) contributes exactly one local cost, with no penalty.
//   - Out-of-window cells are structurally unreachable; overflow is a separate class.
//   - Full accumulated artifacts store +Inf for both structural absence and
//     unrepresentable accumulated values; their numeric cells alone cannot
//     distinguish these two internal classes.
//
// Inputs:
//   - rows, cols: positive counts of time steps, already shape-validated.
//   - cfg: finalized finite/non-negative policy, including window and penalty.
//   - local: pure deterministic source of finite non-negative local costs.
//
// Returns:
//   - *Result: detached finite result or legacy structural no-path result.
//   - error: sentinel-classified failure; numeric final overflow returns nil result.
//
// Errors:
//   - ErrNaNInf / ErrNegativeCost: invalid returned local cost or matrix artifact.
//   - ErrOverflow joined with ErrNaNInf: final minimum cannot fit float64.
//   - ErrNoPath: no admissible final cell and path tracking was requested.
//   - ErrBadInput: size or matrix storage failure.
//   - ErrIncompletePath: invalid private witness chain (internal invariant failure).
//
// Determinism:
//   - Input local artifact is evaluated row-major; DP is row-major.
//   - Strict finite cost ties choose diagonal, then up, then left.
//
// Complexity:
//   - Time O(rows*cols) even with a narrow window (full rectangular scan).
//   - Rolling rows O(cols); predecessor codes O(rows*cols) only with paths.
//   - Accumulated/local artifacts each require O(rows*cols) when requested.
//
// Notes:
//   - Source inputs are read-only; concurrent mutation is unsupported.
//   - The source must not use callback call order as business state.
//
// AI-Hints:
//   - Preserve option validation order in public facades rather than in this kernel.
//   - Do not introduce a dense local matrix into the distance-only scalar path.
//
// nolint:gocyclo
func computeFromCostSource(rows, cols int, cfg options, local costSource) (*Result, error) {
	if rows <= 0 || cols <= 0 {
		return nil, ErrEmptyInput
	}
	if cols == maxIntValue || rows == maxIntValue || rows > maxIntValue/cols {
		return nil, ErrBadInput
	}

	prevRow := make([]dpCell, cols+1)
	currRow := make([]dpCell, cols+1)
	prevRow[0] = dpCell{class: dpFinite}

	needAccumulated := cfg.memoryMode == FullMatrix || cfg.returnAccumulated || cfg.returnPath
	if needAccumulated && (rows+1 > maxIntValue/(cols+1)) {
		return nil, ErrBadInput
	}
	var accumulated *matrix.Dense
	var err error
	if needAccumulated {
		accumulated, err = newAccumulatedCostMatrix(rows+1, cols+1)
		if err != nil {
			return nil, err
		}
	}

	var predecessors []predecessorCode
	if cfg.returnPath {
		predecessors = make([]predecessorCode, rows*cols)
	}

	var localArtifact *matrix.Dense
	if cfg.returnLocalCost {
		localArtifact, err = matrix.NewPreparedDense(rows, cols)
		if err != nil {
			return nil, errors.Join(ErrBadInput, err)
		}

		// Every advertised cell must have a genuine local cost, including off-band cells.
		// Materializing before DP also guarantees a single callback invocation per cell.
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				value, costErr := local(i, j)
				if costErr != nil {
					return nil, costErr
				}
				if costErr = validateLocalCost(value); costErr != nil {
					return nil, costErr
				}
				if costErr = localArtifact.Set(i, j, value); costErr != nil {
					return nil, errors.Join(ErrNaNInf, costErr)
				}
			}
		}
	}

	for i := 1; i <= rows; i++ {
		currRow[0] = dpCell{} // Empty B cannot align to a non-empty A prefix.
		for j := 1; j <= cols; j++ {
			if cfg.window >= 0 && absInt(i-j) > cfg.window {
				currRow[j] = dpCell{} // This cell is absent from the admissible band.
				continue
			}

			var value float64
			if localArtifact != nil {
				value, err = localArtifact.At(i-1, j-1)
			} else {
				value, err = local(i-1, j-1)
			}
			if err != nil {
				return nil, err
			}
			if err = validateLocalCost(value); err != nil {
				return nil, err
			}

			best, predecessor := choosePredecessor(prevRow[j-1], prevRow[j], currRow[j-1], cfg.slopePenalty)
			if best.class != dpUnreachable {
				// All candidates share this local increment, so comparing predecessors
				// first preserves the minimizer under monotone non-negative addition.
				currRow[j] = best.addCost(value)
			} else {
				currRow[j] = dpCell{}
			}

			if predecessors != nil {
				if i == 1 && j == 1 {
					predecessor = predecessorOrigin
				}
				predecessors[(i-1)*cols+(j-1)] = predecessor
			}

			if accumulated != nil {
				stored := math.Inf(1)
				if currRow[j].class == dpFinite {
					stored = currRow[j].cost
				}
				if err = accumulated.Set(i, j, stored); err != nil {
					return nil, errors.Join(ErrNaNInf, err)
				}
			}
		}
		prevRow, currRow = currRow, prevRow
	}

	final := prevRow[cols]
	if final.class == dpOverflow {
		return nil, fmt.Errorf("dtw: terminal optimal accumulated cost: %w", errors.Join(ErrOverflow, ErrNaNInf))
	}

	result := &Result{
		Distance:     math.Inf(1),
		Reachable:    final.class == dpFinite,
		PathTracked:  cfg.returnPath,
		Window:       cfg.window,
		SlopePenalty: cfg.slopePenalty,
		MemoryMode:   cfg.memoryMode,
		LocalCost:    localArtifact,
	}
	if final.class == dpFinite {
		result.Distance = final.cost
	}
	if cfg.returnAccumulated {
		result.Accumulated = accumulated
	}
	if !result.Reachable {
		if cfg.returnPath {
			return result, ErrNoPath
		}
		return result, nil
	}
	if cfg.returnPath {
		result.Path, err = backtrackPredecessors(predecessors, rows, cols)
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}
