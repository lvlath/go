// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm_test anchors sentinel precedence, malicious matrix adapters and
// checked shape arithmetic using exclusively the public import/inference facade.
package lstm_test

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/lvlath/go/lstm"
	"github.com/lvlath/go/matrix"
)

var errAdapterRead = errors.New("matrix adapter: read failure")

// adapterMatrix is an independent Matrix implementation used to inject invalid
// numerical values and backing-store failures that Dense.Set normally rejects.
type adapterMatrix struct {
	rows, cols int
	values     []float64
	readError  error
}

func (a *adapterMatrix) Rows() int {
	if a == nil {
		return 0
	}
	return a.rows
}
func (a *adapterMatrix) Cols() int {
	if a == nil {
		return 0
	}
	return a.cols
}
func (a *adapterMatrix) IsNil() bool { return a == nil }
func (a *adapterMatrix) At(row, col int) (float64, error) {
	if a.readError != nil {
		return 0, a.readError
	}
	return a.values[row*a.cols+col], nil
}
func (a *adapterMatrix) Set(row, col int, value float64) error {
	a.values[row*a.cols+col] = value
	return nil
}
func (a *adapterMatrix) Clone() matrix.Matrix {
	if a == nil {
		return (*adapterMatrix)(nil)
	}
	values := append([]float64(nil), a.values...)
	return &adapterMatrix{rows: a.rows, cols: a.cols, values: values, readError: a.readError}
}

func TestImport_ValidationPrecedenceAndAdapterErrors(t *testing.T) {
	input := mustMatrix(t, 4, 2, nil)
	recurrent := mustMatrix(t, 4, 1, nil)
	base := lstm.Weights{Input: input, Recurrent: recurrent, Bias: make([]float64, 4), Hidden: 1}

	cases := []struct {
		name    string
		change  func(*lstm.Weights)
		wantErr error
	}{
		{"zero hidden", func(w *lstm.Weights) { w.Hidden = 0 }, lstm.ErrShape},
		{"hidden count overflow", func(w *lstm.Weights) { w.Hidden = maxTestInt()/4 + 1 }, lstm.ErrShape},
		{"nil input matrix", func(w *lstm.Weights) { w.Input = nil }, lstm.ErrNilInput},
		{"nil recurrent matrix", func(w *lstm.Weights) { w.Recurrent = nil }, lstm.ErrNilInput},
		{"typed nil input", func(w *lstm.Weights) { var value *matrix.Dense; w.Input = value }, lstm.ErrNilInput},
		{"nil combined bias", func(w *lstm.Weights) { w.Bias = nil }, lstm.ErrNilInput},
		{"empty nonnil bias", func(w *lstm.Weights) { w.Bias = []float64{} }, lstm.ErrShape},
		{"incorrect input rows", func(w *lstm.Weights) { w.Input = mustMatrix(t, 3, 2, nil) }, lstm.ErrShape},
		{"incorrect recurrent rows", func(w *lstm.Weights) { w.Recurrent = mustMatrix(t, 3, 1, nil) }, lstm.ErrShape},
		{"incorrect recurrent columns", func(w *lstm.Weights) { w.Recurrent = mustMatrix(t, 4, 2, nil) }, lstm.ErrShape},
		{"zero input width", func(w *lstm.Weights) { w.Input = mustMatrix(t, 4, 0, nil) }, lstm.ErrShape},
		{"input product overflow", func(w *lstm.Weights) { w.Input = &adapterMatrix{rows: 4, cols: maxTestInt()} }, lstm.ErrShape},
		{"nonfinite input matrix", func(w *lstm.Weights) {
			w.Input = &adapterMatrix{rows: 4, cols: 2, values: []float64{0, 0, 0, math.NaN(), 0, 0, 0, 0}}
		}, lstm.ErrNonFinite},
		{"nonfinite recurrent matrix", func(w *lstm.Weights) {
			w.Recurrent = &adapterMatrix{rows: 4, cols: 1, values: []float64{0, math.Inf(-1), 0, 0}}
		}, lstm.ErrNonFinite},
		{"matrix read failure", func(w *lstm.Weights) { w.Input = &adapterMatrix{rows: 4, cols: 2, readError: errAdapterRead} }, lstm.ErrShape},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			weights := base
			tc.change(&weights)
			model, err := lstm.Import(weights)
			if model != nil {
				t.Fatal("failed Import must not publish a model")
			}
			mustErrorIs(t, err, tc.wantErr)
			if tc.name == "matrix read failure" {
				mustErrorIs(t, err, errAdapterRead)
			}
		})
	}
}

func maxTestInt() int { return int(^uint(0) >> 1) }

func TestInfer_ErrorPrecedenceAndAtomicity(t *testing.T) {
	model := fixture(t, 1, 1, nil)
	good := lstm.Tensor3{Data: []float64{.1, .2}, Batch: 1, Time: 2, Features: 1}
	invalidShape := lstm.Tensor3{Data: []float64{.1}, Batch: 1, Time: -1, Features: 1}
	foreign := &lstm.State{ModelID: "foreign", Hidden: nil, Cell: nil}

	cases := []struct {
		name    string
		ctx     context.Context
		input   lstm.Tensor3
		lengths []int
		state   *lstm.State
		want    error
	}{
		{"nil context first", nil, invalidShape, []int{-1}, foreign, lstm.ErrInvalidOption},
		{"bad tensor dimensions", context.Background(), invalidShape, []int{-1}, foreign, lstm.ErrShape},
		{"nil nonempty data", context.Background(), lstm.Tensor3{Batch: 1, Time: 2, Features: 1}, []int{-1}, foreign, lstm.ErrNilInput},
		{"non-nil wrong data length", context.Background(), lstm.Tensor3{Data: []float64{}, Batch: 1, Time: 2, Features: 1}, []int{-1}, foreign, lstm.ErrShape},
		{"invalid masks before foreign state", context.Background(), good, []int{3}, foreign, lstm.ErrInvalidOption},
		{"invalid mask shape", context.Background(), good, []int{1, 1}, foreign, lstm.ErrInvalidOption},
		{"foreign state before NaN", context.Background(), lstm.Tensor3{Data: []float64{math.NaN(), 2}, Batch: 1, Time: 2, Features: 1}, nil, foreign, lstm.ErrStateVersion},
		{"nonfinite padding", context.Background(), lstm.Tensor3{Data: []float64{.1, math.Inf(1)}, Batch: 1, Time: 2, Features: 1}, []int{1}, nil, lstm.ErrNonFinite},
		{"negative local length", context.Background(), good, []int{-1}, nil, lstm.ErrInvalidOption},
		{"incorrect feature count", context.Background(), lstm.Tensor3{Data: []float64{0, 0}, Batch: 1, Time: 2, Features: 2}, nil, nil, lstm.ErrShape},
		{"checked tensor product", context.Background(), lstm.Tensor3{Batch: maxTestInt(), Time: 2, Features: 1}, nil, nil, lstm.ErrShape},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := model.Infer(tc.ctx, tc.input, tc.lengths, tc.state, lstm.Options{ReturnSequence: true})
			if result != nil {
				t.Fatal("invalid operation published partial result")
			}
			mustErrorIs(t, err, tc.want)
		})
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := model.Infer(ctx, invalidShape, nil, nil, lstm.Options{})
	if result != nil {
		t.Fatal("pre-canceled operation returned a result")
	}
	mustErrorIs(t, err, context.Canceled)

	var nilModel *lstm.Model
	if got := nilModel.ID(); got != "" {
		t.Fatalf("nil Model ID = %q", got)
	}
	result, err = nilModel.Infer(context.Background(), good, nil, nil, lstm.Options{})
	if result != nil {
		t.Fatal("nil model returned a result")
	}
	mustErrorIs(t, err, lstm.ErrNilInput)
}

func TestInfer_InitialStateValidation(t *testing.T) {
	model := fixture(t, 1, 1, nil)
	input := lstm.Tensor3{Data: []float64{0}, Batch: 1, Time: 1, Features: 1}
	cases := []struct {
		name  string
		state lstm.State
		want  error
	}{
		{"nil hidden", lstm.State{ModelID: model.ID(), Cell: mustMatrix(t, 1, 1, nil)}, lstm.ErrNilInput},
		{"typed nil cell", lstm.State{ModelID: model.ID(), Hidden: mustMatrix(t, 1, 1, nil), Cell: (*matrix.Dense)(nil)}, lstm.ErrNilInput},
		{"hidden shape", lstm.State{ModelID: model.ID(), Hidden: mustMatrix(t, 2, 1, nil), Cell: mustMatrix(t, 1, 1, nil)}, lstm.ErrShape},
		{"cell shape", lstm.State{ModelID: model.ID(), Hidden: mustMatrix(t, 1, 1, nil), Cell: mustMatrix(t, 1, 2, nil)}, lstm.ErrShape},
		{"hidden nan", lstm.State{ModelID: model.ID(), Hidden: &adapterMatrix{rows: 1, cols: 1, values: []float64{math.NaN()}}, Cell: mustMatrix(t, 1, 1, nil)}, lstm.ErrNonFinite},
		{"cell inf", lstm.State{ModelID: model.ID(), Hidden: mustMatrix(t, 1, 1, nil), Cell: &adapterMatrix{rows: 1, cols: 1, values: []float64{math.Inf(1)}}}, lstm.ErrNonFinite},
		{"matrix read error", lstm.State{ModelID: model.ID(), Hidden: &adapterMatrix{rows: 1, cols: 1, readError: errAdapterRead}, Cell: mustMatrix(t, 1, 1, nil)}, lstm.ErrShape},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := model.Infer(context.Background(), input, nil, &tc.state, lstm.Options{})
			if result != nil {
				t.Fatal("invalid initial state published result")
			}
			mustErrorIs(t, err, tc.want)
			if tc.name == "matrix read error" {
				mustErrorIs(t, err, errAdapterRead)
			}
		})
	}
}

// countCancelContext deterministically emulates cancellation during a long scan.
// No timers, goroutines or nondeterministic scheduling influence this regression.
type countCancelContext struct {
	reads, cancelAfter int
}

func (c *countCancelContext) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *countCancelContext) Done() <-chan struct{}       { return nil }
func (c *countCancelContext) Value(any) any               { return nil }
func (c *countCancelContext) Err() error {
	c.reads++
	if c.reads >= c.cancelAfter {
		return context.Canceled
	}
	return nil
}

func TestInfer_CancelDuringInputValidation(t *testing.T) {
	model := fixture(t, 1, 1, nil)
	ctx := &countCancelContext{cancelAfter: 5}
	input := lstm.Tensor3{Data: make([]float64, 4096), Batch: 1, Time: 4096, Features: 1}
	result, err := model.Infer(ctx, input, nil, nil, lstm.Options{ReturnSequence: true})
	if result != nil {
		t.Fatal("context cancellation published partial Result")
	}
	mustErrorIs(t, err, context.Canceled)
	if ctx.reads < ctx.cancelAfter {
		t.Fatal("test context did not actually reach its cancellation threshold")
	}
}

func TestImport_ModelID_CanonicalSignedZeroAndOrdering(t *testing.T) {
	negativeZero := math.Copysign(0, -1)
	makeWeights := func(signed bool) lstm.Weights {
		zero := 0.0
		if signed {
			zero = negativeZero
		}
		return lstm.Weights{
			Input:     &adapterMatrix{rows: 4, cols: 1, values: []float64{zero, .25, 0, 0}},
			Recurrent: &adapterMatrix{rows: 4, cols: 1, values: []float64{0, zero, 0, 0}},
			Bias:      []float64{zero, 0, 0, 0},
			Hidden:    1,
		}
	}
	positive, err := lstm.Import(makeWeights(false))
	if err != nil {
		t.Fatal(err)
	}
	negative, err := lstm.Import(makeWeights(true))
	if err != nil {
		t.Fatal(err)
	}
	mustEqualString(t, negative.ID(), positive.ID(), "signed zero canonical model ID")
	if len(positive.ID()) != 64 {
		t.Fatalf("unexpected canonical SHA-256 length: %d", len(positive.ID()))
	}

	changed := makeWeights(false)
	changed.Bias[1] = .01
	other, err := lstm.Import(changed)
	if err != nil {
		t.Fatal(err)
	}
	if other.ID() == positive.ID() {
		t.Fatal("changed weights must not share a canonical ModelID")
	}
}
