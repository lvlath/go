// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm_test anchors the analytical LSTM01-LSTM09 contracts, error
// classification and immutable model/result ownership through public entry points.
package lstm_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/lstm"
	"github.com/lvlath/go/matrix"
)

func TestLSTM01_ZeroWeightsDecay(t *testing.T) {
	m := fixture(t, 1, 1, nil)
	r, e := m.Infer(context.Background(), lstm.Tensor3{Data: []float64{1, 2}, Batch: 1, Time: 2, Features: 1}, nil, state(t, m, 2, 0), lstm.Options{ReturnSequence: true})
	if e != nil {
		t.Fatal(e)
	}
	mustApprox(t, at(t, r.Final.Cell, 0, 0), .5)
	mustApprox(t, r.Sequence.Data[0], .5*math.Tanh(1))
	mustApprox(t, r.Sequence.Data[1], .5*math.Tanh(.5))
}
func TestLSTM02_CandidateBias(t *testing.T) {
	m := fixture(t, 1, 1, []float64{0, 0, math.Atanh(.5), 0})
	r, e := m.Infer(context.Background(), lstm.Tensor3{Data: []float64{0, 0}, Batch: 1, Time: 2, Features: 1}, nil, nil, lstm.Options{ReturnSequence: true})
	if e != nil {
		t.Fatal(e)
	}
	mustApprox(t, at(t, r.Final.Cell, 0, 0), .375)
	mustApprox(t, r.Sequence.Data[0], .5*math.Tanh(.25))
}
func TestLSTM03_PaddingZero(t *testing.T) {
	m := fixture(t, 1, 1, nil)
	r, e := m.Infer(context.Background(), lstm.Tensor3{Data: []float64{1, 2, 3}, Batch: 1, Time: 3, Features: 1}, []int{1}, state(t, m, 2, 0), lstm.Options{ReturnSequence: true})
	if e != nil {
		t.Fatal(e)
	}
	mustApprox(t, at(t, r.Final.Cell, 0, 0), 1)
	if r.Sequence.Data[1] != 0 || r.Sequence.Data[2] != 0 {
		t.Fatal("padding must be zero")
	}
}
func TestLSTM04_ZeroLength(t *testing.T) {
	m := fixture(t, 1, 1, nil)
	r, e := m.Infer(context.Background(), lstm.Tensor3{Data: []float64{3, 4}, Batch: 1, Time: 2, Features: 1}, []int{0}, state(t, m, 2, 3), lstm.Options{ReturnSequence: true})
	if e != nil {
		t.Fatal(e)
	}
	if at(t, r.Final.Cell, 0, 0) != 2 || at(t, r.Final.Hidden, 0, 0) != 3 || r.Sequence.Data[0] != 0 || r.Sequence.Data[1] != 0 {
		t.Fatal("zero length advanced state")
	}
}
func TestLSTM05_ZeroTime(t *testing.T) {
	m := fixture(t, 1, 1, nil)
	r, e := m.Infer(context.Background(), lstm.Tensor3{Batch: 1, Time: 0, Features: 1}, nil, state(t, m, 2, 3), lstm.Options{ReturnSequence: true})
	if e != nil {
		t.Fatal(e)
	}
	if len(r.Sequence.Data) != 0 || at(t, r.Final.Cell, 0, 0) != 2 {
		t.Fatal("invalid empty time")
	}
}
func TestLSTM08_Validation(t *testing.T) {
	m := fixture(t, 1, 1, nil)
	in := lstm.Tensor3{Data: []float64{1, math.NaN()}, Batch: 1, Time: 2, Features: 1}
	r, e := m.Infer(context.Background(), in, []int{1}, nil, lstm.Options{})
	if r != nil {
		t.Fatal("partial result")
	}
	mustErrorIs(t, e, lstm.ErrNonFinite)
	r, e = m.Infer(context.Background(), lstm.Tensor3{Data: []float64{0}, Batch: 1, Time: 1, Features: 1}, nil, &lstm.State{ModelID: "wrong"}, lstm.Options{})
	if r != nil {
		t.Fatal("partial result")
	}
	mustErrorIs(t, e, lstm.ErrStateVersion)
	r, e = m.Infer(nil, in, nil, nil, lstm.Options{})
	if r != nil {
		t.Fatal("partial result")
	}
	mustErrorIs(t, e, lstm.ErrInvalidOption)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r, e = m.Infer(ctx, in, nil, nil, lstm.Options{})
	if r != nil {
		t.Fatal("partial result")
	}
	mustErrorIs(t, e, context.Canceled)
	r, e = m.Infer(context.Background(), lstm.Tensor3{Data: []float64{0}, Batch: 1, Time: 1, Features: 1}, nil, &lstm.State{ModelID: m.ID(), Hidden: mustMatrix(t, 2, 1, nil), Cell: mustMatrix(t, 1, 1, nil)}, lstm.Options{})
	if r != nil {
		t.Fatal("partial result")
	}
	mustErrorIs(t, e, lstm.ErrShape)
}
func TestLSTM09_ImmutableOwnership(t *testing.T) {
	wi := mustMatrix(t, 4, 1, nil)
	wr := mustMatrix(t, 4, 1, nil)
	bias := []float64{0, 0, math.Atanh(.5), 0}
	m, e := lstm.Import(lstm.Weights{Input: wi, Recurrent: wr, Bias: bias, Hidden: 1})
	if e != nil {
		t.Fatal(e)
	}
	id := m.ID()
	bias[2] = 0
	_ = wi.Set(2, 0, 100)
	r, e := m.Infer(context.Background(), lstm.Tensor3{Data: []float64{1}, Batch: 1, Time: 1, Features: 1}, nil, nil, lstm.Options{ReturnSequence: true})
	if e != nil {
		t.Fatal(e)
	}
	r.Sequence.Data[0] = 99
	_ = r.Final.Cell.Set(0, 0, 99)
	next, e := m.Infer(context.Background(), lstm.Tensor3{Data: []float64{1}, Batch: 1, Time: 1, Features: 1}, nil, nil, lstm.Options{ReturnSequence: true})
	if e != nil {
		t.Fatal(e)
	}
	if m.ID() != id {
		t.Fatal("model mutated")
	}
	mustApprox(t, next.Sequence.Data[0], .5*math.Tanh(.25))
}
func TestResult_IsNilInterface(t *testing.T) {
	var r *lstm.Result
	if !r.IsNil() {
		t.Fatal("nil receiver")
	}
	r = &lstm.Result{}
	if r.IsNil() {
		t.Fatal("non-nil")
	}
}
func TestImportFailures(t *testing.T) {
	_, e := lstm.Import(lstm.Weights{Hidden: 1})
	mustErrorIs(t, e, lstm.ErrNilInput)
	_, e = lstm.Import(lstm.Weights{Hidden: 0})
	mustErrorIs(t, e, lstm.ErrShape)
	_, e = lstm.Import(lstm.Weights{Input: mustMatrix(t, 4, 1, nil), Recurrent: mustMatrix(t, 4, 1, nil), Bias: []float64{0, 0, math.NaN(), 0}, Hidden: 1})
	mustErrorIs(t, e, lstm.ErrNonFinite)
	var nilDense *matrix.Dense
	_, e = lstm.Import(lstm.Weights{Input: nilDense, Hidden: 1})
	mustErrorIs(t, e, lstm.ErrNilInput)
}
func TestAffineOverflowAndRescue(t *testing.T) {
	wi := mustMatrix(t, 4, 2, []float64{1e308, -1e308, 0, 0, 0, 0, 0, 0})
	m, e := lstm.Import(lstm.Weights{Input: wi, Recurrent: mustMatrix(t, 4, 1, nil), Bias: make([]float64, 4), Hidden: 1})
	if e != nil {
		t.Fatal(e)
	}
	r, e := m.Infer(context.Background(), lstm.Tensor3{Data: []float64{2, 2}, Batch: 1, Time: 1, Features: 2}, nil, nil, lstm.Options{})
	if e != nil || r == nil {
		t.Fatalf("exact rescue failed: %v", e)
	}
	r, e = m.Infer(context.Background(), lstm.Tensor3{Data: []float64{2, 0}, Batch: 1, Time: 1, Features: 2}, nil, nil, lstm.Options{})
	if r != nil {
		t.Fatal("partial result")
	}
	mustErrorIs(t, e, lstm.ErrOverflow)
}
