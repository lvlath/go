// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm_test anchors the exact overflow law, IFGO gate mapping and stable
// nonlinearities using external analytical oracles rather than implementation helpers.
package lstm_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/lstm"
)

func TestAffineRegression_MaxFloatPlusOneMustOverflow(t *testing.T) {
	// Ordinary double arithmetic rounds MaxFloat64+1 to MaxFloat64, silently
	// erasing exact mathematical overflow unless the boundary is checked.
	model, err := lstm.Import(lstm.Weights{
		Input:     mustMatrix(t, 4, 1, []float64{math.MaxFloat64, 0, 0, 0}),
		Recurrent: mustMatrix(t, 4, 1, nil),
		Bias:      []float64{1, 0, 0, 0}, Hidden: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.InferSequence(context.Background(), lstm.Tensor3{Data: []float64{1}, Batch: 1, Time: 1, Features: 1}, nil, nil)
	if result != nil {
		t.Fatal("mathematically unrepresentable gate preactivation was published")
	}
	mustErrorIs(t, err, lstm.ErrOverflow)
}

func TestAffineRegression_FiniteExtremeResultDoesNotFalseOverflow(t *testing.T) {
	// A mathematically finite MaxFloat64-1 still rounds to MaxFloat64.
	// The exact fallback must classify range, not merely a boundary value.
	model, err := lstm.Import(lstm.Weights{
		Input:     mustMatrix(t, 4, 1, []float64{math.MaxFloat64, 0, 0, 0}),
		Recurrent: mustMatrix(t, 4, 1, nil),
		Bias:      []float64{-1, 0, math.Atanh(.5), 0}, Hidden: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.InferFinal(context.Background(), lstm.Tensor3{Data: []float64{1}, Batch: 1, Time: 1, Features: 1}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	mustApprox(t, at(t, result.Final.Cell, 0, 0), .5) // i=1, g=.5; output cell=.5
}

func TestAffineRegression_IntermediateOverflowCancelsExactly(t *testing.T) {
	model, err := lstm.Import(lstm.Weights{
		Input:     mustMatrix(t, 4, 2, []float64{1e308, -1e308, 0, 0, 0, 0, 0, 0}),
		Recurrent: mustMatrix(t, 4, 1, nil),
		Bias:      []float64{0, 0, math.Atanh(.5), 0}, Hidden: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := model.InferSequence(context.Background(), lstm.Tensor3{Data: []float64{2, 2}, Batch: 1, Time: 1, Features: 2}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	mustApprox(t, at(t, result.Final.Cell, 0, 0), .25)
	mustApprox(t, result.Sequence.Data[0], .5*math.Tanh(.25))
}

func TestCellRegression_MaxFloatPlusOneMustOverflow(t *testing.T) {
	// Forget=1 and i*g>0: an exact cell sum larger than MaxFloat64 must fail
	// even when float64 addition rounds back to the previous finite cell.
	model := fixture(t, 1, 1, []float64{800, 800, 800, 0})
	initial := state(t, model, math.MaxFloat64, 0)
	result, err := model.InferFinal(context.Background(), lstm.Tensor3{Data: []float64{0}, Batch: 1, Time: 1, Features: 1}, nil, initial)
	if result != nil {
		t.Fatal("cell overflow was hidden by binary64 rounding")
	}
	mustErrorIs(t, err, lstm.ErrOverflow)
}

func TestCellRegression_HugeFiniteDecay(t *testing.T) {
	model := fixture(t, 1, 1, nil)
	initial := state(t, model, math.MaxFloat64, 0)
	result, err := model.InferFinal(context.Background(), lstm.Tensor3{Data: []float64{0}, Batch: 1, Time: 1, Features: 1}, nil, initial)
	if err != nil {
		t.Fatal(err)
	}
	mustApprox(t, at(t, result.Final.Cell, 0, 0)/math.MaxFloat64, .5)
}

func TestIFGO_DistinctCandidateOutputBiasMapping(t *testing.T) {
	// g must be tanh, o must be sigmoid. Swapping g/o changes the expected cell.
	bias := []float64{.4, -.25, math.Atanh(.6), 1.2}
	model := fixture(t, 1, 1, bias)
	result, err := model.InferSequence(context.Background(), lstm.Tensor3{Data: []float64{0}, Batch: 1, Time: 1, Features: 1}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	i := 1 / (1 + math.Exp(-bias[0]))
	o := 1 / (1 + math.Exp(-bias[3]))
	wantCell := i * .6
	mustApprox(t, at(t, result.Final.Cell, 0, 0), wantCell)
	mustApprox(t, result.Sequence.Data[0], o*math.Tanh(wantCell))
}

func TestSigmoid_ExtremeFiniteAffineBranches(t *testing.T) {
	// Very large finite gates must saturate without overflowing math.Exp.
	for _, gateValue := range []float64{-800, 800} {
		model := fixture(t, 1, 1, []float64{gateValue, 0, math.Atanh(.5), 0})
		result, err := model.InferSequence(context.Background(), lstm.Tensor3{Data: []float64{0}, Batch: 1, Time: 1, Features: 1}, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		got := at(t, result.Final.Cell, 0, 0)
		want := 0.0
		if gateValue > 0 {
			want = .5
		}
		mustApprox(t, got, want)
		if math.IsInf(result.Sequence.Data[0], 0) || math.IsNaN(result.Sequence.Data[0]) {
			t.Fatal("finite saturated gate led to nonfinite result")
		}
	}
}

func TestMaskedOverflowingStepsAreNotEvaluated(t *testing.T) {
	// A valid but enormous coefficient must not overflow a step that lengths masks.
	model, err := lstm.Import(lstm.Weights{
		Input:     mustMatrix(t, 4, 1, []float64{1e308, 0, 0, 0}),
		Recurrent: mustMatrix(t, 4, 1, nil), Bias: []float64{0, 0, 0, 0}, Hidden: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	initial := state(t, model, 2, .4)
	result, err := model.InferSequence(context.Background(), lstm.Tensor3{Data: []float64{2, 3, 4}, Batch: 1, Time: 3, Features: 1}, []int{0}, initial)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range result.Sequence.Data {
		if math.Float64bits(value) != math.Float64bits(0) {
			t.Fatalf("padded sequence[%d] was not +0", i)
		}
	}
	mustApprox(t, at(t, result.Final.Cell, 0, 0), 2)
	mustApprox(t, at(t, result.Final.Hidden, 0, 0), .4)
}
