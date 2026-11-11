// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols_test verifies error precedence and no-partial-result invariants.
// Invalid Matrix implementations are supplied through a real external interface.
package ols_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ols"
)

func TestFit_ErrorPrecedenceAndShapes(t *testing.T) {
	ctx := context.Background()
	good := mustDense(t, [][]float64{{1}, {2}, {3}})
	values := []float64{1, 2, 3}
	var typedNilDense *matrix.Dense
	var typedNilView *matrix.View
	var typedNilCustom *fixtureMatrix
	for _, tt := range []struct {
		name string
		ctx  context.Context
		x    matrix.Matrix
		y    []float64
		opts ols.Options
		want error
	}{
		{"nil_context_precedes_invalid_option", nil, good, values, ols.Options{RankTolerance: -1}, ols.ErrInvalidOption},
		{"invalid_option_precedes_nil", ctx, nil, nil, ols.Options{RankTolerance: -1}, ols.ErrInvalidOption},
		{"negative_tolerance", ctx, good, values, ols.Options{RankTolerance: -0.1}, ols.ErrInvalidOption},
		{"nan_tolerance", ctx, good, values, ols.Options{RankTolerance: math.NaN()}, ols.ErrInvalidOption},
		{"posinf_tolerance", ctx, good, values, ols.Options{RankTolerance: math.Inf(1)}, ols.ErrInvalidOption},
		{"neginf_tolerance", ctx, good, values, ols.Options{RankTolerance: math.Inf(-1)}, ols.ErrInvalidOption},
		{"nil_matrix", ctx, nil, values, ols.Options{}, ols.ErrNilInput},
		{"typed_nil_dense", ctx, typedNilDense, values, ols.Options{}, ols.ErrNilInput},
		{"typed_nil_view", ctx, typedNilView, values, ols.Options{}, ols.ErrNilInput},
		{"typed_nil_custom_nilable", ctx, typedNilCustom, values, ols.Options{}, ols.ErrNilInput},
		{"nil_response", ctx, good, nil, ols.Options{}, ols.ErrNilInput},
		{"zero_rows", ctx, &fixtureMatrix{rows: 0, cols: 1}, values, ols.Options{}, ols.ErrEmptyInput},
		{"zero_columns", ctx, &fixtureMatrix{rows: 3, cols: 0}, values, ols.Options{}, ols.ErrEmptyInput},
		{"empty_nonnil_response", ctx, good, []float64{}, ols.Options{}, ols.ErrEmptyInput},
		{"incorrect_response_length", ctx, good, []float64{1, 2}, ols.Options{}, ols.ErrShape},
		{"n_less_than_q", ctx, mustDense(t, [][]float64{{1, 3}, {2, 4}}), []float64{1, 2}, ols.Options{}, ols.ErrShape},
		{"negative_shape", ctx, &fixtureMatrix{rows: -1, cols: 1}, values, ols.Options{}, ols.ErrShape},
		{"huge_element_count", ctx, &fixtureMatrix{rows: int(^uint(0) >> 1), cols: 8}, values, ols.Options{}, ols.ErrShape},
		{"nan_y", ctx, good, []float64{1, math.NaN(), 3}, ols.Options{}, ols.ErrNonFinite},
		{"positive_infinite_y", ctx, good, []float64{1, math.Inf(1), 3}, ols.Options{}, ols.ErrNonFinite},
		{"negative_infinite_y", ctx, good, []float64{1, math.Inf(-1), 3}, ols.Options{}, ols.ErrNonFinite},
		{"nan_x", ctx, &fixtureMatrix{rows: 3, cols: 1, values: []float64{1, math.NaN(), 3}}, values, ols.Options{}, ols.ErrNonFinite},
		{"infinite_x", ctx, &fixtureMatrix{rows: 3, cols: 1, values: []float64{1, math.Inf(-1), 3}}, values, ols.Options{}, ols.ErrNonFinite},
	} {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ols.Fit(tt.ctx, tt.x, tt.y, tt.opts)
			if result != nil {
				t.Fatalf("partial FitResult on error %v", err)
			}
			mustErrorIs(t, err, tt.want)
		})
	}
}

func TestFit_CancellationPrecedenceAndMatrixError(t *testing.T) {
	x := mustDense(t, [][]float64{{1}, {2}, {3}})
	y := []float64{1, 2, 3}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, tt := range []struct {
		name string
		opts ols.Options
	}{
		{"valid_options", ols.Options{}},
		{"invalid_options", ols.Options{RankTolerance: -1}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ols.Fit(ctx, x, y, tt.opts)
			if result != nil {
				t.Fatal("cancellation published result")
			}
			mustErrorIs(t, err, context.Canceled)
		})
	}
	root := errors.New("upstream matrix read refused")
	broken := &fixtureMatrix{rows: 3, cols: 1, values: []float64{1, 2, 3}, readErr: root, errRow: 2, errCol: 0}
	result, err := ols.Fit(context.Background(), broken, y, ols.Options{})
	if result != nil {
		t.Fatal("Fit published result after Matrix.At error")
	}
	mustErrorIs(t, err, root)
}

func TestFit_CancellationPolledDuringValidation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	const observations = 260
	input := &fixtureMatrix{rows: observations, cols: 1, values: make([]float64, observations)}
	for i := range input.values {
		input.values[i] = float64(i)
	}
	input.onRead = func(row, col int) {
		if row == 1 && col == 0 {
			cancel()
		}
	}
	y := make([]float64, observations)
	result, err := ols.Fit(ctx, input, y, ols.Options{})
	if result != nil {
		t.Fatal("Fit returned partial result on cancellation")
	}
	mustErrorIs(t, err, context.Canceled)
}

func TestPredict_ValidationAndSpecialErrors(t *testing.T) {
	modelFit, err := ols.FitThroughOrigin(context.Background(), mustDense(t,
		[][]float64{{1}, {2}, {3}, {4}}), []float64{10, 20, 30, 40})
	if err != nil {
		t.Fatal(err)
	}
	model := modelFit.Model
	var absent *ols.Model
	var typedNil *matrix.Dense
	var typedNilView *matrix.View
	root := errors.New("inference matrix unreadable")
	for _, tt := range []struct {
		name  string
		model *ols.Model
		ctx   context.Context
		x     matrix.Matrix
		want  error
	}{
		{"nil_context", model, nil, mustDense(t, [][]float64{{1}}), ols.ErrInvalidOption},
		{"nil_model", absent, context.Background(), mustDense(t, [][]float64{{1}}), ols.ErrNilInput},
		{"untrained_model", &ols.Model{}, context.Background(), mustDense(t, [][]float64{{1}}), ols.ErrNilInput},
		{"nil_x", model, context.Background(), nil, ols.ErrNilInput},
		{"typed_nil_dense", model, context.Background(), typedNil, ols.ErrNilInput},
		{"typed_nil_view", model, context.Background(), typedNilView, ols.ErrNilInput},
		{"wrong_p", model, context.Background(), mustDense(t, [][]float64{{1, 2}}), ols.ErrShape},
		{"negative_rows", model, context.Background(), &fixtureMatrix{rows: -1, cols: 1}, ols.ErrShape},
		{"nonfinite", model, context.Background(), &fixtureMatrix{rows: 1, cols: 1, values: []float64{math.NaN()}}, ols.ErrNonFinite},
		{"pos_inf", model, context.Background(), &fixtureMatrix{rows: 1, cols: 1, values: []float64{math.Inf(1)}}, ols.ErrNonFinite},
		{"neg_inf", model, context.Background(), &fixtureMatrix{rows: 1, cols: 1, values: []float64{math.Inf(-1)}}, ols.ErrNonFinite},
		{"read_error", model, context.Background(), &fixtureMatrix{rows: 1, cols: 1, values: []float64{1}, readErr: root}, root},
		{"arithmetic_overflow", model, context.Background(), mustDense(t, [][]float64{{1e308}}), ols.ErrOverflow},
	} {
		t.Run(tt.name, func(t *testing.T) {
			predicted, err := tt.model.Predict(tt.ctx, tt.x)
			if predicted != nil {
				t.Fatal("Predict published partial values on error")
			}
			mustErrorIs(t, err, tt.want)
		})
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	out, err := model.Predict(cancelled, nil)
	if out != nil {
		t.Fatal("cancellation published output")
	}
	mustErrorIs(t, err, context.Canceled)
}

func TestPredict_EmptyViewAndMatrixContract(t *testing.T) {
	ctx := context.Background()
	base := mustDense(t, [][]float64{{999, 999}, {1, 5}, {2, 7}, {3, 9}, {4, 11}, {999, 999}})
	view, err := base.View(1, 0, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	var input matrix.Matrix = view
	result, err := ols.FitDefault(ctx, input, []float64{11, 16, 21, 26})
	if err == nil {
		t.Fatal("an exactly collinear two-feature view plus intercept must be rejected")
	}
	if result != nil {
		t.Fatal("rank-deficient view produced model")
	}
	mustErrorIs(t, err, ols.ErrRankDeficient)

	other := mustDense(t, [][]float64{{10, 0}, {11, 2}, {13, 1}, {14, 5}, {16, 2}})
	win, err := other.View(0, 0, 5, 2)
	if err != nil {
		t.Fatal(err)
	}
	result, err = ols.FitDefault(ctx, win, []float64{21, 27, 28, 38, 36})
	if err != nil {
		t.Fatal(err)
	}
	empty, err := other.View(5, 0, 0, 2)
	if err != nil {
		t.Fatal(err)
	}
	pred, err := result.Model.Predict(ctx, empty)
	if err != nil || pred == nil || len(pred) != 0 {
		t.Fatalf("zero-row View inference: %v, %v", pred, err)
	}
	wrongCols, err := matrix.NewDense(0, 3)
	if err != nil {
		t.Fatal(err)
	}
	_, err = result.Model.Predict(ctx, wrongCols)
	mustErrorIs(t, err, ols.ErrShape)
}

// TestPredict_ValidatesAllFiniteDataBeforeArithmetic locks error precedence:
// an overflowing first prediction must not mask NaN in a later input row.
func TestPredict_ValidatesAllFiniteDataBeforeArithmetic(t *testing.T) {
	fit, err := ols.FitThroughOrigin(context.Background(), mustDense(t,
		[][]float64{{1}, {2}, {3}}), []float64{10, 20, 30})
	if err != nil {
		t.Fatal(err)
	}
	input := &fixtureMatrix{rows: 2, cols: 1, values: []float64{1e308, math.NaN()}}
	predictions, err := fit.Model.Predict(context.Background(), input)
	if predictions != nil {
		t.Fatal("nonfinite inference published output")
	}
	mustErrorIs(t, err, ols.ErrNonFinite)
}

// TestPredict_CancellationInsideInputScan checks bounded cancellation polling
// without sleeps, timing assumptions, or worker goroutines.
func TestPredict_CancellationInsideInputScan(t *testing.T) {
	fit, err := ols.FitDefault(context.Background(), mustDense(t,
		[][]float64{{1}, {2}, {3}}), []float64{4, 7, 10})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	input := &fixtureMatrix{rows: 260, cols: 1, values: make([]float64, 260)}
	input.onRead = func(row, col int) {
		if row == 1 && col == 0 {
			cancel()
		}
	}
	predictions, err := fit.Model.Predict(ctx, input)
	if predictions != nil {
		t.Fatal("cancelled inference published partial predictions")
	}
	mustErrorIs(t, err, context.Canceled)
}

// TestFit_CancellationAcrossScalingBoundaries verifies that OLS never publishes
// a model after cancellation between validation, the stats fit, and design build.
// The callback changes only context state; training matrix values stay fixed.
func TestFit_CancellationAcrossScalingBoundaries(t *testing.T) {
	for _, standardized := range []bool{false, true} {
		name := "unscaled_centering"
		if standardized {
			name = "population_stats_fit"
		}
		t.Run(name, func(t *testing.T) {
			const rows = 6
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			reads := 0
			input := &fixtureMatrix{rows: rows, cols: 1, values: []float64{1, 2, 3, 4, 5, 6}}
			input.onRead = func(_, _ int) {
				reads++
				// First six reads belong to validateFit; the seventh starts
				// either mean estimation or the accepted stats dependency.
				if reads == rows+1 {
					cancel()
				}
			}
			result, err := ols.Fit(ctx, input,
				[]float64{2, 4, 6, 8, 10, 12}, ols.Options{Standardize: standardized})
			if result != nil {
				t.Fatal("cancellation after validation published a fit")
			}
			mustErrorIs(t, err, context.Canceled)
		})
	}
}
