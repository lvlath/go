// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test verifies the public DTW contract.
// These tests protect mathematical behavior rather than implementation accidents.
package dtw_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/dtw"
)

func TestDTWLegacyNilOptionsUsesDefaults(t *testing.T) {
	dist, path, err := dtw.DTW([]float64{1, 2, 3}, []float64{1, 2, 3}, nil)

	mustNoError(t, err, "DTW legacy nil options")
	mustFloatEqual(t, dist, 0, 0, "DTW legacy nil distance")

	if path != nil {
		t.Fatalf("DTW legacy nil options path: got %+v, want nil", path)
	}
}

func TestOptionsValidateNilReceiverReturnsErrNilOptions(t *testing.T) {
	var opts *dtw.Options

	err := opts.Validate()

	mustErrorIs(t, err, dtw.ErrNilOptions, "Options.Validate nil receiver")
}

func TestAlignRejectsNilInput(t *testing.T) {
	res, err := dtw.Align(nil, []float64{1})

	mustErrorIs(t, err, dtw.ErrNilInput, "Align nil input")
	mustEqualBool(t, res == nil, true, "Align nil input result")
}

func TestAlignRejectsEmptyInput(t *testing.T) {
	res, err := dtw.Align([]float64{}, []float64{1})

	mustErrorIs(t, err, dtw.ErrEmptyInput, "Align empty input")
	mustEqualBool(t, res == nil, true, "Align empty input result")
}

func TestAlignRejectsNaNInputByDefault(t *testing.T) {
	res, err := dtw.Align([]float64{1, math.NaN()}, []float64{1, 2})

	mustErrorIs(t, err, dtw.ErrNaNInf, "Align NaN input")
	mustEqualBool(t, res == nil, true, "Align NaN result")
}

func TestAlignRejectsPositiveInfinityInputByDefault(t *testing.T) {
	res, err := dtw.Align([]float64{1, math.Inf(1)}, []float64{1, 2})

	mustErrorIs(t, err, dtw.ErrNaNInf, "Align +Inf input")
	mustEqualBool(t, res == nil, true, "Align +Inf result")
}

func TestAlignRejectsNaNPenalty(t *testing.T) {
	res, err := dtw.Align(
		[]float64{1},
		[]float64{1},
		dtw.WithSlopePenalty(math.NaN()),
	)

	mustErrorIs(t, err, dtw.ErrInvalidPenalty, "WithSlopePenalty(NaN)")
	mustErrorIs(t, err, dtw.ErrBadInput, "WithSlopePenalty(NaN) preserves ErrBadInput")
	mustEqualBool(t, res == nil, true, "WithSlopePenalty(NaN) result")
}

func TestAlignRejectsInvalidWindow(t *testing.T) {
	res, err := dtw.Align(
		[]float64{1},
		[]float64{1},
		dtw.WithWindow(-2),
	)

	mustErrorIs(t, err, dtw.ErrInvalidWindow, "WithWindow(-2)")
	mustErrorIs(t, err, dtw.ErrBadInput, "WithWindow(-2) preserves ErrBadInput")
	mustEqualBool(t, res == nil, true, "WithWindow(-2) result")
}

func TestAlignRejectsNilOption(t *testing.T) {
	res, err := dtw.Align([]float64{1}, []float64{1}, nil)

	mustErrorIs(t, err, dtw.ErrNilOption, "Align nil option")
	mustEqualBool(t, res == nil, true, "Align nil option result")
}

func TestAlignRejectsNilCostFunc(t *testing.T) {
	res, err := dtw.Align(
		[]float64{1},
		[]float64{1},
		dtw.WithCostFunc(nil),
	)

	mustErrorIs(t, err, dtw.ErrNilCostFunc, "WithCostFunc(nil)")
	mustEqualBool(t, res == nil, true, "WithCostFunc(nil) result")
}

func TestAlignRejectsNaNLocalCost(t *testing.T) {
	res, err := dtw.Align(
		[]float64{1},
		[]float64{1},
		dtw.WithCostFunc(func(_, _ float64) (float64, error) {
			return math.NaN(), nil
		}),
	)

	mustErrorIs(t, err, dtw.ErrNaNInf, "CostFunc NaN")
	mustEqualBool(t, res == nil, true, "CostFunc NaN result")
}

func TestAlignRejectsNegativeLocalCost(t *testing.T) {
	res, err := dtw.Align(
		[]float64{1},
		[]float64{1},
		dtw.WithCostFunc(func(_, _ float64) (float64, error) {
			return -1, nil
		}),
	)

	mustErrorIs(t, err, dtw.ErrNegativeCost, "CostFunc negative")
	mustEqualBool(t, res == nil, true, "CostFunc negative result")
}

func TestWithCostFuncErrorPropagates(t *testing.T) {
	errUserCost := errors.New("user cost failed")

	res, err := dtw.Align(
		[]float64{1},
		[]float64{1},
		dtw.WithCostFunc(func(_, _ float64) (float64, error) {
			return 0, errUserCost
		}),
	)

	mustErrorIs(t, err, errUserCost, "user cost error")
	mustEqualBool(t, res == nil, true, "user cost error result")
}

func TestLegacyReturnPathRequiresFullMatrix(t *testing.T) {
	opts := dtw.DefaultOptions()
	opts.ReturnPath = true
	opts.MemoryMode = dtw.TwoRows

	_, _, err := dtw.DTW([]float64{1}, []float64{1}, &opts)

	mustErrorIs(t, err, dtw.ErrPathNeedsMatrix, "legacy ReturnPath without FullMatrix")
}

func TestWindowLawMinusOneUnconstrainedZeroStrictDiagonal(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{1, 2, 3, 4}

	res, err := dtw.Align(a, b, dtw.WithWindow(-1))
	mustNoError(t, err, "Align Window=-1")
	mustFinite(t, res.Distance, "Window=-1 distance")
	mustEqualBool(t, res.Reachable, true, "Window=-1 reachable")

	res, err = dtw.Align(a, b, dtw.WithWindow(0))
	mustNoError(t, err, "Align Window=0")
	mustInf(t, res.Distance, "Window=0 distance")
	mustEqualBool(t, res.Reachable, false, "Window=0 reachable")
}

func TestBacktrackFullMatrixBoundaryWithPositivePenalty(t *testing.T) {
	a := []float64{1}
	b := []float64{1, 1}

	res, err := dtw.Align(
		a,
		b,
		dtw.WithSlopePenalty(1),
		dtw.WithReturnPath(true),
	)

	mustNoError(t, err, "Align path with positive penalty")
	mustFloatEqual(t, res.Distance, 1, 0, "positive penalty distance")
	mustEqualBool(t, res.PathTracked, true, "path tracked")
	mustPathValid(t, res.Path, len(a), len(b), "positive penalty path")
	mustDistanceMatchesPath(t, a, b, res.Path, 1, res.Distance, "positive penalty path cost")
}

func TestReturnPathNoPathClassifiedBeforeBacktracking(t *testing.T) {
	res, err := dtw.Align(
		[]float64{1, 2, 3},
		[]float64{1, 2, 3, 4},
		dtw.WithWindow(0),
		dtw.WithReturnPath(true),
	)

	mustErrorIs(t, err, dtw.ErrNoPath, "ReturnPath no admissible path")

	if res == nil {
		t.Fatalf("ReturnPath no path: got nil result, want partial result")
	}

	mustInf(t, res.Distance, "ReturnPath no path distance")
	mustEqualBool(t, res.Reachable, false, "ReturnPath no path reachable")
}

func TestPathTieBreakDiagonalFirst(t *testing.T) {
	res, err := dtw.Align(
		[]float64{1, 1},
		[]float64{1, 1},
		dtw.WithReturnPath(true),
	)

	mustNoError(t, err, "tie-break path")
	mustEqualInt(t, len(res.Path), 2, "tie-break path length")

	want := dtw.Path{{I: 0, J: 0}, {I: 1, J: 1}}
	for idx := range want {
		if res.Path[idx] != want[idx] {
			t.Fatalf("tie-break path[%d]: got %+v, want %+v; full=%+v", idx, res.Path[idx], want[idx], res.Path)
		}
	}
}

func TestResultPathOrErrorStates(t *testing.T) {
	var nilResult *dtw.Result

	_, err := nilResult.PathOrError()
	mustErrorIs(t, err, dtw.ErrNilResult, "nil Result PathOrError")

	res, err := dtw.Align([]float64{1}, []float64{1})
	mustNoError(t, err, "Align without path")

	_, err = res.PathOrError()
	mustErrorIs(t, err, dtw.ErrPathNotTracked, "PathOrError without tracking")

	res, err = dtw.Align([]float64{1}, []float64{1}, dtw.WithReturnPath(true))
	mustNoError(t, err, "Align with path")

	path, err := res.PathOrError()
	mustNoError(t, err, "PathOrError with path")
	mustEqualInt(t, len(path), 1, "PathOrError path length")

	path[0] = dtw.Coord{I: 99, J: 99}
	mustEqualBool(t, res.Path[0] == (dtw.Coord{I: 0, J: 0}), true, "PathOrError returns detached path")
}

func TestResultCloneDetachesPath(t *testing.T) {
	res, err := dtw.Align(
		[]float64{1, 2},
		[]float64{1, 2},
		dtw.WithReturnPath(true),
	)

	mustNoError(t, err, "Align for Clone")

	cloned := res.Clone()
	if cloned == nil {
		t.Fatalf("Result.Clone: got nil clone")
	}

	cloned.Path[0] = dtw.Coord{I: 9, J: 9}

	mustEqualBool(t, res.Path[0] == (dtw.Coord{I: 0, J: 0}), true, "Clone detaches path")
}
