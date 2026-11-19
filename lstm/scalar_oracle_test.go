// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm_test provides an independent scalar recurrence oracle.
// It deliberately reads matrices with At and does not call the implementation kernel.
package lstm_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/lstm"
	"github.com/lvlath/go/matrix"
)

// scalarReference uses explicit per-gate matrix indexing instead of private model buffers.
// Its naïve float64 arithmetic is suitable for well-conditioned reference fixtures,
// not for the adversarial high-dynamic-range tests that exercise exact overflow rescue.
func scalarReference(t *testing.T, weights lstm.Weights, input lstm.Tensor3, lengths []int, initialHidden, initialCell []float64) ([]float64, []float64, []float64) {
	t.Helper()
	hiddenWidth := weights.Hidden
	hidden := append([]float64(nil), initialHidden...)
	cell := append([]float64(nil), initialCell...)
	sequence := make([]float64, input.Batch*input.Time*hiddenWidth)

	for batch := 0; batch < input.Batch; batch++ {
		limit := input.Time
		if lengths != nil {
			limit = lengths[batch]
		}
		for timestep := 0; timestep < limit; timestep++ {
			pre := make([]float64, 4*hiddenWidth)
			// The oracle builds the entire 4H preactivation vector before updating hidden.
			for gate := 0; gate < 4; gate++ {
				for unit := 0; unit < hiddenWidth; unit++ {
					row := gate*hiddenWidth + unit
					value := weights.Bias[row]
					for feature := 0; feature < input.Features; feature++ {
						weight, err := weights.Input.At(row, feature)
						if err != nil {
							t.Fatal(err)
						}
						value += weight * input.Data[(batch*input.Time+timestep)*input.Features+feature]
					}
					for prev := 0; prev < hiddenWidth; prev++ {
						weight, err := weights.Recurrent.At(row, prev)
						if err != nil {
							t.Fatal(err)
						}
						value += weight * hidden[batch*hiddenWidth+prev]
					}
					pre[row] = value
				}
			}

			for unit := 0; unit < hiddenWidth; unit++ {
				i := 1 / (1 + math.Exp(-pre[unit]))
				f := 1 / (1 + math.Exp(-pre[hiddenWidth+unit]))
				g := math.Tanh(pre[2*hiddenWidth+unit])
				o := 1 / (1 + math.Exp(-pre[3*hiddenWidth+unit]))
				cell[batch*hiddenWidth+unit] = f*cell[batch*hiddenWidth+unit] + i*g
				hidden[batch*hiddenWidth+unit] = o * math.Tanh(cell[batch*hiddenWidth+unit])
				sequence[(batch*input.Time+timestep)*hiddenWidth+unit] = hidden[batch*hiddenWidth+unit]
			}
		}
	}
	return sequence, hidden, cell
}

func TestScalarOracle_MultiFeatureGatesAndInitialState(t *testing.T) {
	const (
		B = 3
		T = 7
		D = 3
		H = 2
	)
	inputWeights := make([]float64, 4*H*D)
	recurrentWeights := make([]float64, 4*H*H)
	bias := make([]float64, 4*H)
	data := make([]float64, B*T*D)
	initialHidden := make([]float64, B*H)
	initialCell := make([]float64, B*H)

	for index := range inputWeights {
		inputWeights[index] = .07 * math.Sin(float64(index+2))
	}
	for index := range recurrentWeights {
		recurrentWeights[index] = .11 * math.Cos(float64(index+1))
	}
	for index := range bias {
		bias[index] = .05 * float64(index-3)
	}
	for index := range data {
		data[index] = math.Sin(float64(index)*.33) * .8
	}
	for index := range initialHidden {
		initialHidden[index] = .02 * float64(index-2)
		initialCell[index] = .17 * float64(3-index)
	}

	weights := lstm.Weights{Input: mustMatrix(t, 4*H, D, inputWeights), Recurrent: mustMatrix(t, 4*H, H, recurrentWeights), Bias: bias, Hidden: H}
	model, err := lstm.Import(weights)
	if err != nil {
		t.Fatal(err)
	}
	input := lstm.Tensor3{Data: data, Batch: B, Time: T, Features: D}
	lengths := []int{7, 4, 0}
	stateT := &lstm.State{Hidden: mustMatrix(t, B, H, initialHidden), Cell: mustMatrix(t, B, H, initialCell), ModelID: model.ID()}
	wantSequence, wantHidden, wantCell := scalarReference(t, weights, input, lengths, initialHidden, initialCell)

	result, err := model.InferSequence(context.Background(), input, lengths, stateT)
	if err != nil {
		t.Fatal(err)
	}
	for index, want := range wantSequence {
		if math.Abs(result.Sequence.Data[index]-want) > 1e-12 {
			t.Fatalf("oracle sequence[%d]: got %.17g want %.17g", index, result.Sequence.Data[index], want)
		}
	}
	for index, want := range wantHidden {
		got := at(t, result.Final.Hidden, index/H, index%H)
		if math.Abs(got-want) > 1e-12 {
			t.Fatalf("oracle hidden[%d]: got %.17g want %.17g", index, got, want)
		}
	}
	for index, want := range wantCell {
		got := at(t, result.Final.Cell, index/H, index%H)
		if math.Abs(got-want) > 1e-12 {
			t.Fatalf("oracle cell[%d]: got %.17g want %.17g", index, got, want)
		}
	}
}

func TestLSTM06_BatchPermutationNonVacuous(t *testing.T) {
	wi := mustMatrix(t, 4, 1, []float64{.4, -.3, .9, .2})
	wr := mustMatrix(t, 4, 1, []float64{.1, .07, -.2, .3})
	model, err := lstm.Import(lstm.Weights{Input: wi, Recurrent: wr, Bias: []float64{.1, -.2, .3, .4}, Hidden: 1})
	if err != nil {
		t.Fatal(err)
	}
	inA := lstm.Tensor3{Data: []float64{.1, .9, -.2, 1.1, -.5, -.7}, Batch: 2, Time: 3, Features: 1}
	inB := lstm.Tensor3{Data: []float64{1.1, -.5, -.7, .1, .9, -.2}, Batch: 2, Time: 3, Features: 1}
	a, err := model.InferSequence(context.Background(), inA, []int{2, 3}, nil)
	if err != nil {
		t.Fatal(err)
	}
	b, err := model.InferSequence(context.Background(), inB, []int{3, 2}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if a.Sequence.Data[0] == a.Sequence.Data[3] {
		t.Fatal("batch permutation test is vacuous: rows should differ")
	}
	for step := 0; step < 3; step++ {
		if math.Float64bits(a.Sequence.Data[step]) != math.Float64bits(b.Sequence.Data[3+step]) ||
			math.Float64bits(a.Sequence.Data[3+step]) != math.Float64bits(b.Sequence.Data[step]) {
			t.Fatalf("batch permutation mismatch at timestep %d", step)
		}
	}
	for _, value := range []struct{ first, second matrix.Matrix }{{a.Final.Hidden, b.Final.Hidden}, {a.Final.Cell, b.Final.Cell}} {
		if math.Float64bits(at(t, value.first, 0, 0)) != math.Float64bits(at(t, value.second, 1, 0)) ||
			math.Float64bits(at(t, value.first, 1, 0)) != math.Float64bits(at(t, value.second, 0, 0)) {
			t.Fatal("final state mixed independent batch rows")
		}
	}
}
