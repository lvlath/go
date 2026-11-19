// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm_test verifies state handoff with independent batch masks and
// bitwise identity of each valid and padded sequence coordinate across chunks.
package lstm_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/lstm"
)

func TestLSTM07_MultiBatchMaskedChunksBitwise(t *testing.T) {
	const (
		batch      = 3
		time       = 6
		width      = 2
		hiddenSize = 2
		split      = 2
	)
	inputWeights := make([]float64, 4*hiddenSize*width)
	recurrentWeights := make([]float64, 4*hiddenSize*hiddenSize)
	for index := range inputWeights {
		inputWeights[index] = math.Sin(float64(index+1)) * .09
	}
	for index := range recurrentWeights {
		recurrentWeights[index] = math.Cos(float64(index+3)) * .07
	}
	model, err := lstm.Import(lstm.Weights{
		Input:     mustMatrix(t, 8, width, inputWeights),
		Recurrent: mustMatrix(t, 8, hiddenSize, recurrentWeights),
		Bias:      []float64{.1, -.1, .2, -.2, .04, .15, -.07, .03}, Hidden: hiddenSize,
	})
	if err != nil {
		t.Fatal(err)
	}
	all := make([]float64, batch*time*width)
	for index := range all {
		all[index] = .3 * math.Sin(float64(index)*.19)
	}
	fullInput := lstm.Tensor3{Data: all, Batch: batch, Time: time, Features: width}
	fullLengths := []int{6, 3, 0}
	initialHidden := []float64{.1, -.1, -.05, .04, .2, .3}
	initialCell := []float64{.9, -.3, .6, -.4, .25, .15}
	initial := &lstm.State{Hidden: mustMatrix(t, batch, hiddenSize, initialHidden), Cell: mustMatrix(t, batch, hiddenSize, initialCell), ModelID: model.ID()}

	full, err := model.InferSequence(context.Background(), fullInput, fullLengths, initial)
	if err != nil {
		t.Fatal(err)
	}
	// Build batch-first chunks: direct slicing would be wrong for B>1.
	firstValues := make([]float64, batch*split*width)
	secondValues := make([]float64, batch*(time-split)*width)
	firstLengths := make([]int, batch)
	secondLengths := make([]int, batch)
	for row, length := range fullLengths {
		copy(firstValues[row*split*width:(row+1)*split*width], all[row*time*width:row*time*width+split*width])
		copy(secondValues[row*(time-split)*width:(row+1)*(time-split)*width], all[row*time*width+split*width:(row+1)*time*width])
		firstLengths[row] = min(length, split)
		secondLengths[row] = min(max(length-split, 0), time-split)
	}

	first, err := model.InferSequence(context.Background(), lstm.Tensor3{Data: firstValues, Batch: batch, Time: split, Features: width}, firstLengths, initial)
	if err != nil {
		t.Fatal(err)
	}
	second, err := model.InferChunk(context.Background(), lstm.Tensor3{Data: secondValues, Batch: batch, Time: time - split, Features: width}, secondLengths, first, lstm.Options{ReturnSequence: true})
	if err != nil {
		t.Fatal(err)
	}

	for row := 0; row < batch; row++ {
		for step := 0; step < time; step++ {
			for feature := 0; feature < hiddenSize; feature++ {
				index := (row*time+step)*hiddenSize + feature
				var got float64
				if step < split {
					got = first.Sequence.Data[(row*split+step)*hiddenSize+feature]
				} else {
					got = second.Sequence.Data[(row*(time-split)+step-split)*hiddenSize+feature]
				}
				if math.Float64bits(full.Sequence.Data[index]) != math.Float64bits(got) {
					t.Fatalf("row %d step %d feature %d: chunk sequence differs", row, step, feature)
				}
			}
		}
		for feature := 0; feature < hiddenSize; feature++ {
			if math.Float64bits(at(t, full.Final.Hidden, row, feature)) != math.Float64bits(at(t, second.Final.Hidden, row, feature)) ||
				math.Float64bits(at(t, full.Final.Cell, row, feature)) != math.Float64bits(at(t, second.Final.Cell, row, feature)) {
				t.Fatalf("row %d feature %d: streamed Final differs", row, feature)
			}
		}
	}
}

func TestWrapperOptionsAndNilPriorResult(t *testing.T) {
	model := fixture(t, 1, 1, []float64{.3, -.1, math.Atanh(.5), .7})
	input := lstm.Tensor3{Data: []float64{.1, .2, .3, .4}, Batch: 1, Time: 4, Features: 1}
	ctx := context.Background()
	defaultResult, err := model.InferDefault(ctx, input)
	if err != nil {
		t.Fatal(err)
	}
	if defaultResult.Sequence.Data != nil || defaultResult.Sequence.Batch != 0 {
		t.Fatal("default must not allocate a sequence")
	}
	finalResult, err := model.InferFinal(ctx, input, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	seqResult, err := model.InferSequence(ctx, input, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, result := range []*lstm.Result{finalResult, seqResult} {
		mustEqualString(t, result.Final.ModelID, model.ID(), "wrapper model version")
		if math.Float64bits(at(t, result.Final.Hidden, 0, 0)) != math.Float64bits(at(t, defaultResult.Final.Hidden, 0, 0)) ||
			math.Float64bits(at(t, result.Final.Cell, 0, 0)) != math.Float64bits(at(t, defaultResult.Final.Cell, 0, 0)) {
			t.Fatal("facade wrapper changed final arithmetic")
		}
	}
	mustEqualInt(t, len(seqResult.Sequence.Data), 4, "sequence wrapper output")

	var nilPrior *lstm.Result
	failed, err := model.InferChunk(ctx, input, nil, nilPrior, lstm.Options{})
	if failed != nil {
		t.Fatal("nil previous Result returned a value")
	}
	mustErrorIs(t, err, lstm.ErrNilInput)

	other := fixture(t, 1, 1, nil)
	failed, err = other.InferChunk(ctx, input, nil, seqResult, lstm.Options{})
	if failed != nil {
		t.Fatal("foreign previous model state returned a value")
	}
	mustErrorIs(t, err, lstm.ErrStateVersion)

	canceledCtx, cancel := context.WithCancel(ctx)
	cancel()
	failed, err = model.InferChunk(canceledCtx, input, nil, nil, lstm.Options{})
	if failed != nil {
		t.Fatal("canceled chunk published a value")
	}
	mustErrorIs(t, err, context.Canceled)
}

func TestStateAndResult_DetachedAcrossStreamingCalls(t *testing.T) {
	model := fixture(t, 1, 1, []float64{.2, .1, math.Atanh(.5), 0})
	input := lstm.Tensor3{Data: []float64{.2, -.3}, Batch: 1, Time: 2, Features: 1}
	initial := state(t, model, 2, .3)
	result, err := model.InferSequence(context.Background(), input, nil, initial)
	if err != nil {
		t.Fatal(err)
	}
	originalHidden := at(t, result.Final.Hidden, 0, 0)
	originalCell := at(t, result.Final.Cell, 0, 0)

	// Returned final matrices are detached from the caller's supplied initial State.
	if err = initial.Hidden.Set(0, 0, 50); err != nil {
		t.Fatal(err)
	}
	if err = initial.Cell.Set(0, 0, 50); err != nil {
		t.Fatal(err)
	}
	mustApprox(t, at(t, result.Final.Hidden, 0, 0), originalHidden)
	mustApprox(t, at(t, result.Final.Cell, 0, 0), originalCell)

	// A continued result is detached from an already-published previous result.
	next, err := model.InferChunk(context.Background(), input, nil, result, lstm.Options{})
	if err != nil {
		t.Fatal(err)
	}
	wantNext := at(t, next.Final.Cell, 0, 0)
	if err = result.Final.Hidden.Set(0, 0, -50); err != nil {
		t.Fatal(err)
	}
	if err = result.Final.Cell.Set(0, 0, -50); err != nil {
		t.Fatal(err)
	}
	result.Sequence.Data[0] = 123
	mustApprox(t, at(t, next.Final.Cell, 0, 0), wantNext)

	// Mutating any caller-visible Result also cannot alter the immutable Model.
	fresh, err := model.InferDefault(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	freshAgain, err := model.InferDefault(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if at(t, fresh.Final.Cell, 0, 0) != at(t, freshAgain.Final.Cell, 0, 0) {
		t.Fatal("model was unexpectedly mutated by earlier Results")
	}
}

func TestResultNilable_ThroughInterface(t *testing.T) {
	var nilResult *lstm.Result
	var nilable interface{ IsNil() bool } = nilResult
	mustNilState(t, nilable, true, "typed nil result")
	mustNilState(t, &lstm.Result{}, false, "allocated zero result")
	mustNilState(t, nil, true, "nil interface")
}
