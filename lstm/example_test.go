// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm_test demonstrates applied, deterministic LSTM inference pipelines.
// All examples use an explicitly synthetic model: they exercise data engineering
// workflows, not validated financial, industrial or clinical prediction accuracy.
package lstm_test

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/lstm"
	"github.com/lvlath/go/matrix"
)

// monitoringModel builds a fixed 3-input/2-hidden IFGO fixture with nonzero
// coefficients in every gate. In production, these coefficients must instead
// come from a separately trained and independently verified double-precision model.
func monitoringModel() (*lstm.Model, error) {
	const (
		inputWidth  = 3
		hiddenWidth = 2
	)
	inputCoefficients := [8][3]float64{
		{.16, -.09, .05}, {.02, .22, -.04}, // Input gates.
		{-.12, .08, .03}, {.06, -.11, .12}, // Forget gates.
		{.72, .15, -.30}, {-.18, .54, .21}, // Candidate cell gates.
		{.13, .09, -.05}, {-.08, .06, .14}, // Output gates.
	}
	recurrentCoefficients := [8][2]float64{
		{.02, -.01}, {.03, .04},
		{-.04, .02}, {.02, -.02},
		{.09, -.04}, {-.03, .08},
		{.01, .03}, {-.02, .01},
	}

	input, err := matrix.NewDense(4*hiddenWidth, inputWidth)
	if err != nil {
		return nil, err
	}
	recurrent, err := matrix.NewDense(4*hiddenWidth, hiddenWidth)
	if err != nil {
		return nil, err
	}
	for row, coefficients := range inputCoefficients {
		for col, value := range coefficients {
			if err := input.Set(row, col, value); err != nil {
				return nil, err
			}
		}
	}
	for row, coefficients := range recurrentCoefficients {
		for col, value := range coefficients {
			if err := recurrent.Set(row, col, value); err != nil {
				return nil, err
			}
		}
	}
	return lstm.Import(lstm.Weights{
		Input: input, Recurrent: recurrent,
		Bias: []float64{.1, -.05, .2, .15, .02, -.03, .05, .08}, Hidden: hiddenWidth,
	})
}

// ExampleModel_InferSequence_orderBookMicrostructure extracts a time-aligned
// latent representation from a realistic miniature exchange order-book feed.
//
// Scenario:
//   - A feed handler collects 24 successive snapshots of quoted spread (basis
//     points), near-touch depth imbalance, and signed aggressive-trade pressure.
//   - The caller normalizes raw exchange fields with fixed deployed scaling;
//     the LSTM produces temporal features for downstream replay and diagnostics.
//
// Process:
//   - Build a batch-first [1,24,3] tensor, infer all hidden vectors, and measure
//     the largest one-step feature displacement without inventing a trading signal.
//   - The summary is deterministic and suitable for monitoring feature continuity.
//
// AI-Hints:
//   - A synthetic LSTM embedding is not a probability of price movement or a
//     trading strategy; a separately validated trained head would be required.
func ExampleModel_InferSequence_orderBookMicrostructure() {
	model, err := monitoringModel()
	if err != nil {
		fmt.Println("model:", err)
		return
	}
	book := [][3]float64{
		{1.1, .12, -.05}, {.9, .18, .02}, {1.0, .16, .07}, {1.2, -.04, -.09},
		{1.4, -.12, -.18}, {1.2, -.20, -.25}, {1.0, -.08, -.05}, {.8, .05, .11},
		{.9, .14, .21}, {1.1, .27, .19}, {1.3, .32, .28}, {1.5, .41, .38},
		{1.4, .33, .25}, {1.2, .15, .09}, {1.0, .08, -.08}, {1.1, -.10, -.15},
		{1.3, -.25, -.31}, {1.6, -.39, -.42}, {1.5, -.34, -.23}, {1.3, -.21, -.08},
		{1.2, -.02, .02}, {1.1, .11, .12}, {.9, .19, .18}, {.8, .24, .26},
	}
	features := make([]float64, 0, len(book)*3)
	for _, quote := range book {
		features = append(features,
			(quote[0]-1.0)/2.0, // Spread in bps -> fixed dimensionless scale.
			quote[1],           // Depth imbalance already in [-1,1].
			quote[2],           // Signed aggressor-flow ratio in [-1,1].
		)
	}
	result, err := model.InferSequence(context.Background(), lstm.Tensor3{
		Data: features, Batch: 1, Time: len(book), Features: 3,
	}, nil, nil)
	if err != nil {
		fmt.Println("inference:", err)
		return
	}

	peakShift := 0.0
	for timestep := 1; timestep < len(book); timestep++ {
		left := result.Sequence.Data[timestep*2] - result.Sequence.Data[(timestep-1)*2]
		right := result.Sequence.Data[timestep*2+1] - result.Sequence.Data[(timestep-1)*2+1]
		if shift := math.Hypot(left, right); shift > peakShift {
			peakShift = shift
		}
	}
	fmt.Printf("order-book frames=%d latent=%d max-step-shift=%.4f\n", result.Sequence.Time, result.Sequence.Features, peakShift)
	// Output: order-book frames=24 latent=2 max-step-shift=0.0703
}

// ExampleModel_InferChunk_pumpVibrationStream tracks a pump's latent state across
// independently delivered sensor chunks without resetting memory at chunk edges.
//
// Scenario:
//   - A pump publishes temperature in degrees C, vibration RMS in g and current
//     in amperes in two successive twelve-sample network packets.
//   - Stored training-time scales convert sensor values to dimensionless features.
//
// Process:
//   - Run two chunks, pass the first Final to InferChunk, and compare against
//     one uninterrupted pass on all 24 observations, bit for bit.
//
// AI-Hints:
//   - The entire State (both Hidden and Cell) is needed for an exact restart.
//   - Feature scales must remain identical for every chunk of the same model.
func ExampleModel_InferChunk_pumpVibrationStream() {
	model, err := monitoringModel()
	if err != nil {
		fmt.Println("model:", err)
		return
	}
	all := make([]float64, 24*3)
	for sample := 0; sample < 24; sample++ {
		temperatureC := 64.0 + .12*float64(sample) + math.Sin(float64(sample)*.3)
		vibrationG := .28 + .013*float64(sample) + .03*math.Sin(float64(sample)*.6)
		currentA := 8.5 + .06*float64(sample) + .14*math.Cos(float64(sample)*.25)
		all[sample*3] = (temperatureC - 60) / 20
		all[sample*3+1] = (vibrationG - .25) / .50
		all[sample*3+2] = (currentA - 8) / 4
	}
	first, err := model.InferFinal(context.Background(), lstm.Tensor3{
		Data: all[:12*3], Batch: 1, Time: 12, Features: 3,
	}, nil, nil)
	if err != nil {
		fmt.Println("first chunk:", err)
		return
	}
	second, err := model.InferChunk(context.Background(), lstm.Tensor3{
		Data: all[12*3:], Batch: 1, Time: 12, Features: 3,
	}, nil, first, lstm.Options{})
	if err != nil {
		fmt.Println("second chunk:", err)
		return
	}
	full, err := model.InferDefault(context.Background(), lstm.Tensor3{
		Data: all, Batch: 1, Time: 24, Features: 3,
	})
	if err != nil {
		fmt.Println("full replay:", err)
		return
	}
	same := true
	for feature := 0; feature < 2; feature++ {
		streamH, errH := second.Final.Hidden.At(0, feature)
		streamC, errC := second.Final.Cell.At(0, feature)
		fullH, errFH := full.Final.Hidden.At(0, feature)
		fullC, errFC := full.Final.Cell.At(0, feature)
		if errH != nil || errC != nil || errFH != nil || errFC != nil {
			fmt.Println("invalid state access")
			return
		}
		same = same && math.Float64bits(streamH) == math.Float64bits(fullH) && math.Float64bits(streamC) == math.Float64bits(fullC)
	}
	fmt.Printf("pump samples=%d checkpoints=%d state-bitwise=%t\n", 24, 2, same)
	// Output: pump samples=24 checkpoints=2 state-bitwise=true
}

// ExampleModel_InferSequence_fleetTelemetryMask extracts comparable per-device
// features from variable-duration machine sessions without reading padded steps.
//
// Scenario:
//   - Four battery-backed edge devices upload sessions of 10, 25, 15 and 30 samples.
//   - All messages have 30 slots; padding is finite but must not affect state.
//
// Process:
//   - Build a [4,30,3] tensor of normalized temperature, vibration and voltage;
//     apply the explicit lengths mask and validate zero tails per hidden coordinate.
//
// AI-Hints:
//   - ReturnSequence padding must remain zero; final per-device embeddings live
//     in Final.Hidden, not in Sequence at t=29 for shorter sessions.
func ExampleModel_InferSequence_fleetTelemetryMask() {
	model, err := monitoringModel()
	if err != nil {
		fmt.Println("model:", err)
		return
	}
	lengths := []int{10, 25, 15, 30}
	data := make([]float64, 4*30*3)
	for device := 0; device < 4; device++ {
		for sample := 0; sample < 30; sample++ {
			index := (device*30 + sample) * 3
			data[index] = (34 + float64(device)*3 + .15*float64(sample) - 40) / 20
			data[index+1] = (.14 + .04*float64(device) + .003*float64(sample) - .2) / .3
			data[index+2] = (3.95 - .12*float64(device) - .002*float64(sample) - 3.7) / .5
		}
	}
	result, err := model.InferSequence(context.Background(), lstm.Tensor3{
		Data: data, Batch: 4, Time: 30, Features: 3,
	}, lengths, nil)
	if err != nil {
		fmt.Println("fleet:", err)
		return
	}

	validFrames, zeroPaddedFeatures := 0, 0
	for device, valid := range lengths {
		validFrames += valid
		for step := valid; step < 30; step++ {
			for feature := 0; feature < 2; feature++ {
				if result.Sequence.Data[(device*30+step)*2+feature] == 0 {
					zeroPaddedFeatures++
				}
			}
		}
	}
	fmt.Printf("fleet devices=%d valid=%d zero-padded-features=%d\n", result.Sequence.Batch, validFrames, zeroPaddedFeatures)
	// Output: fleet devices=4 valid=80 zero-padded-features=80
}

// ExampleModel_InferFinal_compressorComparison compares final feature vectors
// from two equal-duration compressor vibration traces for offline inspection.
//
// Scenario:
//   - Two compressors report 18 samples each of temperature, vibration and load.
//   - The second shows increasing vibration and heat under comparable electrical load.
//
// Process:
//   - Evaluate the [2,18,3] batch with no sequence allocation, then measure
//     Euclidean separation of its two final two-dimensional latent vectors.
//
// AI-Hints:
//   - Feature distance is a diagnostic primitive, not a calibrated failure
//     probability; calibration and maintenance decisions belong upstream.
func ExampleModel_InferFinal_compressorComparison() {
	model, err := monitoringModel()
	if err != nil {
		fmt.Println("model:", err)
		return
	}
	data := make([]float64, 2*18*3)
	for machine := 0; machine < 2; machine++ {
		for sample := 0; sample < 18; sample++ {
			index := (machine*18 + sample) * 3
			temperatureC := 58 + .12*float64(sample) + float64(machine)*.23*float64(sample)
			vibrationG := .2 + .004*float64(sample) + float64(machine)*.016*float64(sample)
			loadA := 9.1 + .06*math.Sin(float64(sample)*.25)
			data[index] = (temperatureC - 55) / 20
			data[index+1] = (vibrationG - .2) / .5
			data[index+2] = (loadA - 9) / 3
		}
	}
	result, err := model.InferFinal(context.Background(), lstm.Tensor3{
		Data: data, Batch: 2, Time: 18, Features: 3,
	}, nil, nil)
	if err != nil {
		fmt.Println("compressors:", err)
		return
	}
	a0, errA0 := result.Final.Hidden.At(0, 0)
	a1, errA1 := result.Final.Hidden.At(0, 1)
	b0, errB0 := result.Final.Hidden.At(1, 0)
	b1, errB1 := result.Final.Hidden.At(1, 1)
	if errA0 != nil || errA1 != nil || errB0 != nil || errB1 != nil {
		fmt.Println("invalid final state")
		return
	}
	fmt.Printf("compressors=%d final-feature-gap=%.4f sequence-allocated=%t\n", result.Final.Hidden.Rows(), math.Hypot(a0-b0, a1-b1), result.Sequence.Data != nil)
	// Output: compressors=2 final-feature-gap=0.1769 sequence-allocated=false
}

// ExampleImport_deploymentFingerprint shows safe model rollout with immutable
// copied weights and strict rejection of a checkpoint from a different version.
//
// Scenario:
//   - A deployment records a canonical model identity next to its stream states.
//   - If weights change, stale state is rejected rather than silently reused.
//
// Process:
//   - Import a fixed IFGO model, mutate caller weights after import, and try
//     replaying the previous state with a deliberately changed new model.
//
// AI-Hints:
//   - SHA-256 binds state to bytes, not to an authenticated trusted publisher.
func ExampleImport_deploymentFingerprint() {
	input, err := matrix.NewDense(4, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	recurrent, err := matrix.NewDense(4, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := input.Set(2, 0, .3); err != nil {
		fmt.Println(err)
		return
	}
	bias := []float64{0, 0, .1, 0}
	firstModel, err := lstm.Import(lstm.Weights{Input: input, Recurrent: recurrent, Bias: bias, Hidden: 1})
	if err != nil {
		fmt.Println(err)
		return
	}
	fingerprint := firstModel.ID()
	if err := input.Set(2, 0, .9); err != nil {
		fmt.Println(err)
		return
	}
	bias[2] = .2
	secondModel, err := lstm.Import(lstm.Weights{Input: input, Recurrent: recurrent, Bias: bias, Hidden: 1})
	if err != nil {
		fmt.Println(err)
		return
	}

	inputPacket := lstm.Tensor3{Data: []float64{.6, .7}, Batch: 1, Time: 2, Features: 1}
	first, err := firstModel.InferDefault(context.Background(), inputPacket)
	if err != nil {
		fmt.Println(err)
		return
	}
	second, err := secondModel.InferChunk(context.Background(), inputPacket, nil, first, lstm.Options{})
	fmt.Printf("model-id-stable=%t version-changed=%t stale-rejected=%t no-partial=%t\n",
		fingerprint == firstModel.ID(), firstModel.ID() != secondModel.ID(), errors.Is(err, lstm.ErrStateVersion), second == nil)
	// Output: model-id-stable=true version-changed=true stale-rejected=true no-partial=true
}

// ExampleResult_IsNil_quarantinedTelemetry demonstrates an atomic refusal of a
// malformed payload even when its nonfinite value appears only in masked padding.
//
// Scenario:
//   - A sensor claims two valid frames in a three-frame packet, but the last
//     supposedly ignored slot is NaN from a corrupted telemetry message.
//
// Process:
//   - Decode to a numerical tensor, check the error sentinel, and quarantine
//     the packet without handing a partial hidden state to downstream systems.
//
// AI-Hints:
//   - Masked bytes still belong to the public input payload and must be valid.
func ExampleResult_IsNil_quarantinedTelemetry() {
	model, err := monitoringModel()
	if err != nil {
		fmt.Println(err)
		return
	}
	input := lstm.Tensor3{Data: []float64{
		.3, .2, -.1, .4, .1, .2, math.NaN(), 0, 0,
	}, Batch: 1, Time: 3, Features: 3}
	result, err := model.InferSequence(context.Background(), input, []int{2}, nil)
	fmt.Printf("quarantine=%t nil-result=%t\n", errors.Is(err, lstm.ErrNonFinite), result.IsNil())
	// Output: quarantine=true nil-result=true
}
