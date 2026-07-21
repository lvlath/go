// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test verifies matrix-backed DTW adapters.
// These tests defend adapter policy, shape validation, and artifact ownership.
package dtw_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/matrix"
)

func TestAlignCostMatrixComputesVoiceCommandPath(t *testing.T) {
	local, err := matrix.NewPreparedDense(5, 7)
	mustNoError(t, err, "NewPreparedDense local cost")

	err = local.Fill([]float64{
		0.0, 0.1, 2.0, 3.0, 4.0, 5.0, 6.0,
		1.0, 0.0, 0.1, 2.0, 3.0, 4.0, 5.0,
		2.0, 1.0, 0.0, 0.1, 2.0, 3.0, 4.0,
		3.0, 2.0, 1.0, 0.0, 0.1, 2.0, 3.0,
		4.0, 3.0, 2.0, 1.0, 0.0, 0.1, 0.0,
	})
	mustNoError(t, err, "Fill local cost")

	res, err := dtw.AlignCostMatrix(
		local,
		dtw.WithWindow(2),
		dtw.WithSlopePenalty(0.2),
		dtw.WithReturnPath(true),
		dtw.WithReturnLocalCost(true),
	)
	mustNoError(t, err, "AlignCostMatrix")

	mustFloatEqual(t, res.Distance, 0.5, 1e-9, "AlignCostMatrix distance")
	mustPathEqual(t, res.Path, dtw.Path{
		{I: 0, J: 0},
		{I: 1, J: 1},
		{I: 2, J: 2},
		{I: 3, J: 3},
		{I: 4, J: 4},
		{I: 4, J: 5},
		{I: 4, J: 6},
	}, "AlignCostMatrix path")
	mustMatrixShape(t, res.LocalCost, 5, 7, "AlignCostMatrix local artifact")
}

func TestAlignCostMatrixRejectsNegativeCost(t *testing.T) {
	local, err := matrix.NewPreparedDense(1, 1)
	mustNoError(t, err, "NewPreparedDense local cost")

	err = local.Set(0, 0, -1)
	mustNoError(t, err, "Set negative local cost fixture")

	_, err = dtw.AlignCostMatrix(local)

	mustErrorIs(t, err, dtw.ErrNegativeCost, "AlignCostMatrix negative local cost")
}

func TestAlignCostMatrixRejectsNaNCost(t *testing.T) {
	local, err := matrix.NewPreparedDense(1, 1, matrix.WithNoValidateNaNInf())
	mustNoError(t, err, "NewPreparedDense raw local cost")

	err = local.Set(0, 0, math.NaN())
	mustNoError(t, err, "Set NaN local cost")

	_, err = dtw.AlignCostMatrix(local)

	mustErrorIs(t, err, dtw.ErrNaNInf, "AlignCostMatrix NaN local cost")
}

func TestAlignCostMatrixRejectsEmptyShape(t *testing.T) {
	local, err := matrix.NewPreparedDense(0, 1)
	mustNoError(t, err, "NewPreparedDense empty local cost")

	_, err = dtw.AlignCostMatrix(local)

	mustErrorIs(t, err, dtw.ErrEmptyInput, "AlignCostMatrix empty rows")
}

func TestAlignMatrixComputesVibrationSignaturePath(t *testing.T) {
	reference, err := matrix.NewPreparedDense(6, 2)
	mustNoError(t, err, "NewPreparedDense reference")

	err = reference.Fill([]float64{
		0.0, 0.0,
		0.3, 0.1,
		0.8, 0.2,
		1.0, 0.1,
		0.6, -0.1,
		0.1, -0.2,
	})
	mustNoError(t, err, "Fill reference")

	observed, err := matrix.NewPreparedDense(7, 2)
	mustNoError(t, err, "NewPreparedDense observed")

	err = observed.Fill([]float64{
		0.0, 0.0,
		0.2, 0.1,
		0.45, 0.1,
		0.85, 0.2,
		1.0, 0.0,
		0.65, -0.1,
		0.1, -0.2,
	})
	mustNoError(t, err, "Fill observed")

	res, err := dtw.AlignMatrix(
		reference,
		observed,
		dtw.WithWindow(2),
		dtw.WithSlopePenalty(0.02),
		dtw.WithReturnPath(true),
		dtw.WithReturnLocalCost(true),
	)
	mustNoError(t, err, "AlignMatrix")

	mustFloatEqual(t, res.Distance, 0.0675, 1e-12, "AlignMatrix distance")
	mustPathEqual(t, res.Path, dtw.Path{
		{I: 0, J: 0},
		{I: 1, J: 1},
		{I: 1, J: 2},
		{I: 2, J: 3},
		{I: 3, J: 4},
		{I: 4, J: 5},
		{I: 5, J: 6},
	}, "AlignMatrix path")
	mustMatrixShape(t, res.LocalCost, 6, 7, "AlignMatrix local artifact")
}

func TestAlignMatrixRejectsFeatureDimensionMismatch(t *testing.T) {
	x, err := matrix.NewPreparedDense(2, 2)
	mustNoError(t, err, "NewPreparedDense x")

	y, err := matrix.NewPreparedDense(2, 3)
	mustNoError(t, err, "NewPreparedDense y")

	_, err = dtw.AlignMatrix(x, y)

	mustErrorIs(t, err, matrix.ErrDimensionMismatch, "AlignMatrix dimension mismatch")
	mustErrorIs(t, err, dtw.ErrBadInput, "AlignMatrix preserves dtw ErrBadInput")
}

func TestAlignMatrixRejectsNaNInput(t *testing.T) {
	x, err := matrix.NewPreparedDense(1, 1, matrix.WithNoValidateNaNInf())
	mustNoError(t, err, "NewPreparedDense raw x")

	err = x.Set(0, 0, math.NaN())
	mustNoError(t, err, "Set NaN x")

	y, err := matrix.NewPreparedDense(1, 1)
	mustNoError(t, err, "NewPreparedDense y")

	err = y.Set(0, 0, 0)
	mustNoError(t, err, "Set y")

	_, err = dtw.AlignMatrix(x, y)

	mustErrorIs(t, err, dtw.ErrNaNInf, "AlignMatrix NaN input")
}

func TestAlignCostMatrixLocalArtifactIsDetached(t *testing.T) {
	local, err := matrix.NewPreparedDense(1, 1)
	mustNoError(t, err, "NewPreparedDense local")

	err = local.Set(0, 0, 0)
	mustNoError(t, err, "Set local")

	res, err := dtw.AlignCostMatrix(local, dtw.WithReturnLocalCost(true))
	mustNoError(t, err, "AlignCostMatrix")

	err = res.LocalCost.Set(0, 0, 99)
	mustNoError(t, err, "mutate result local artifact")

	original, err := local.At(0, 0)
	mustNoError(t, err, "read original local")

	mustFloatEqual(t, original, 0, 0, "original local matrix must remain unchanged")
}
