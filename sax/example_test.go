// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// These five executable external examples demonstrate distinct operational
// contracts with synthetic, reproducible data. SAX provides shape signatures
// and Euclidean screening signals; it never certifies clinical decisions or DTW.
package sax_test

import (
	"fmt"
	"math"
	"sort"

	"github.com/lvlath/go/sax"
)

// ExampleEncode_ecgMorphology screens synthetic ECG morphology against a
// patient-independent template, preserving the distinction between a shape
// shortlist flag and a medically validated arrhythmia diagnosis.
//
// The samples represent a 64-point, 125 Hz synthetic beat window (512 ms).
// Per-beat Z scoring removes amplitude/offset; unexpected QRS inversion is
// screened by comparing SAX signatures, not by making a clinical diagnosis.
func ExampleEncode_ecgMorphology() {
	const samplesPerBeat = 64
	baseline := make([]float64, samplesPerBeat)
	stableRepeat := make([]float64, samplesPerBeat)
	atypicalBeat := make([]float64, samplesPerBeat)

	// Build a reproducible ECG-shaped synthetic waveform with P/QRS/T features.
	for i := range baseline {
		t := float64(i)
		p := 0.12 * math.Exp(-math.Pow((t-15)/4, 2))
		q := -0.25 * math.Exp(-math.Pow((t-29)/1.7, 2))
		r := 1.20 * math.Exp(-math.Pow((t-32)/1.8, 2))
		s := -0.32 * math.Exp(-math.Pow((t-35)/1.6, 2))
		tWave := 0.28 * math.Exp(-math.Pow((t-47)/6, 2))
		baseline[i] = p + q + r + s + tWave
		stableRepeat[i] = 1.05*baseline[i] + 0.03
		atypicalBeat[i] = baseline[i]
		if i >= 29 && i <= 35 {
			atypicalBeat[i] = -0.9 * baseline[i]
		}
	}

	opts := sax.DefaultOptions(16)
	opts.Alphabet = 8
	encoder, err := sax.NewEncoder(opts)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Reuse a single immutable model for all aligned beat windows.
	baselineWord, err := encoder.Encode(baseline)
	if err != nil {
		fmt.Println(err)
		return
	}
	stableWord, err := encoder.Encode(stableRepeat)
	if err != nil {
		fmt.Println(err)
		return
	}
	atypicalWord, err := encoder.Encode(atypicalBeat)
	if err != nil {
		fmt.Println(err)
		return
	}
	stableDistance, err := encoder.Mindist(baselineWord, stableWord)
	if err != nil {
		fmt.Println(err)
		return
	}
	atypicalDistance, err := encoder.Mindist(baselineWord, atypicalWord)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Operational interpretation: inspect the larger shape mismatch. A
	// distance threshold still needs validation against labeled ECG examples.
	fmt.Printf("ECG: samples=%d segments=%d stable=%.1f atypical-higher=%t\n",
		samplesPerBeat, len(baselineWord.Symbols), stableDistance, atypicalDistance > stableDistance)
	// Output: ECG: samples=64 segments=16 stable=0.0 atypical-higher=true
}

// ExampleEncodeRawWithBreakpoints_marketReturns describes minute-return regimes
// in basis points, not raw asset prices. Domain-calibrated bins retain the
// magnitude of a volatility shock that independent Z scoring would suppress.
//
// Both sequences are synthetic 60-minute illustrations, not live market data
// or an investment decision; a positive MINDIST is a screening signal only.
func ExampleEncodeRawWithBreakpoints_marketReturns() {
	const minutes = 60
	quietReturns := make([]float64, minutes)
	shockReturns := make([]float64, minutes)
	for i := 0; i < minutes; i++ {
		quietReturns[i] = 2.5 * math.Sin(float64(i)*0.41)
		shockReturns[i] = quietReturns[i]
		if i >= 40 && i < 50 {
			shockReturns[i] += 26 // synthetic 26-basis-point impulsive interval
		}
	}

	// Four breakpoints define five interpretable return-size regimes.
	breakpoints := []float64{-15, -5, 5, 15}
	baseline, err := sax.EncodeRawWithBreakpoints(quietReturns, 12, breakpoints, "returns-bp-v1")
	if err != nil {
		fmt.Println(err)
		return
	}
	opts := sax.Options{Segments: 12, Alphabet: 5, Normalization: sax.Raw,
		Breakpoints: breakpoints, BreakpointID: "returns-bp-v1"}
	encoder, err := sax.NewEncoder(opts)
	if err != nil {
		fmt.Println(err)
		return
	}
	shock, err := encoder.Encode(shockReturns)
	if err != nil {
		fmt.Println(err)
		return
	}
	distance, err := encoder.Mindist(baseline, shock)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Returns: minutes=%d shock-segments=%d positive-mismatch=%t\n",
		minutes, len(shock.Symbols), distance > 0)
	// Output: Returns: minutes=60 shock-segments=12 positive-mismatch=true
}

// ExampleEncoder_Mindist_clusterTelemetry ranks synthetic 48-minute worker
// profiles by CPU and memory SAX lower bounds in percent units. Two independent
// calibrated encoders prevent accidentally mixing unlike feature domains.
//
// The combined math.Hypot score is a bound for the concatenated CPU+RAM Raw
// vectors only because both streams share the same cadence and percent units.
func ExampleEncoder_Mindist_clusterTelemetry() {
	const minutes = 48
	cpuOpts := sax.Options{Segments: 8, Alphabet: 5, Normalization: sax.Raw,
		Breakpoints: []float64{20, 40, 60, 80}, BreakpointID: "cpu-percent"}
	ramOpts := sax.Options{Segments: 8, Alphabet: 5, Normalization: sax.Raw,
		Breakpoints: []float64{20, 40, 60, 80}, BreakpointID: "ram-percent"}
	cpuEncoder, err := sax.NewEncoder(cpuOpts)
	if err != nil {
		fmt.Println(err)
		return
	}
	ramEncoder, err := sax.NewEncoder(ramOpts)
	if err != nil {
		fmt.Println(err)
		return
	}

	templateCPU := make([]float64, minutes)
	templateRAM := make([]float64, minutes)
	for i := range templateCPU {
		templateCPU[i] = 48 + 3*math.Sin(float64(i)*0.19)
		templateRAM[i] = 32 + 2*math.Cos(float64(i)*0.13)
	}
	referenceCPU, err := cpuEncoder.Encode(templateCPU)
	if err != nil {
		fmt.Println(err)
		return
	}
	referenceRAM, err := ramEncoder.Encode(templateRAM)
	if err != nil {
		fmt.Println(err)
		return
	}

	type workload struct {
		name     string
		cpu, ram []float64
		score    float64
	}
	candidates := []workload{
		{name: "worker-healthy", cpu: make([]float64, minutes), ram: make([]float64, minutes)},
		{name: "worker-overloaded", cpu: make([]float64, minutes), ram: make([]float64, minutes)},
	}
	for i := range templateCPU {
		candidates[0].cpu[i] = templateCPU[i] + 2
		candidates[0].ram[i] = templateRAM[i] + 2
		candidates[1].cpu[i] = 91 + 3*math.Sin(float64(i)*0.19)
		candidates[1].ram[i] = 92 + 2*math.Cos(float64(i)*0.13)
	}

	// Compare both independent axes and rank deterministically by score, name.
	for i := range candidates {
		cpuWord, cpuErr := cpuEncoder.Encode(candidates[i].cpu)
		ramWord, ramErr := ramEncoder.Encode(candidates[i].ram)
		if cpuErr != nil || ramErr != nil {
			fmt.Println("invalid telemetry input")
			return
		}
		cpuGap, cpuErr := cpuEncoder.Mindist(referenceCPU, cpuWord)
		ramGap, ramErr := ramEncoder.Mindist(referenceRAM, ramWord)
		if cpuErr != nil || ramErr != nil {
			fmt.Println("incompatible telemetry encoding")
			return
		}
		candidates[i].score = math.Hypot(cpuGap, ramGap)
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].score == candidates[j].score {
			return candidates[i].name < candidates[j].name
		}
		return candidates[i].score < candidates[j].score
	})
	fmt.Printf("Telemetry: closest=%s furthest=%s suspect=%t\n",
		candidates[0].name, candidates[len(candidates)-1].name, candidates[1].score > 0)
	// Output: Telemetry: closest=worker-healthy furthest=worker-overloaded suspect=true
}

// ExampleMindistValues_gridFrequency encodes grid-frequency residuals in mHz
// for aligned synthetic 50 Hz measurement windows. A sustained under-frequency
// interval produces a nonzero Euclidean symbolic signal for further inspection.
//
// This is a monitoring example, not an automatic grid-protection trip rule;
// operational actions require additional physical and temporal constraints.
func ExampleMindistValues_gridFrequency() {
	const pointsPerWindow = 96
	quiet := make([]float64, pointsPerWindow)
	underFrequency := make([]float64, pointsPerWindow)
	for i := range quiet {
		quiet[i] = 6 * math.Sin(float64(i)*0.21) // millihertz offset from 50 Hz
		underFrequency[i] = quiet[i]
		if i >= 48 && i < 80 {
			underFrequency[i] -= 110
		}
	}
	policy := sax.Options{Segments: 12, Alphabet: 5, Normalization: sax.Raw,
		Breakpoints: []float64{-60, -15, 15, 60}, BreakpointID: "freq-residual-mHz-v1"}
	bound, err := sax.MindistValues(quiet, underFrequency, policy)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Grid: n=%d under-frequency-flag=%t\n", len(quiet), bound > 0)
	// Output: Grid: n=96 under-frequency-flag=true
}

// ExampleEncode_constantSensors shows why a constant ZPopulation word cannot
// preserve absolute sensor amplitude. Two stuck temperature channels at very
// different readings both produce the same normalized zero word under ZeroWord.
//
// A separate Raw domain-calibrated comparison catches the amplitude mismatch;
// no single symbolized signal replaces a dedicated sensor-health alarm.
func ExampleEncode_constantSensors() {
	const readings = 48
	stuckCool := make([]float64, readings)
	stuckHot := make([]float64, readings)
	for i := range stuckCool {
		stuckCool[i], stuckHot[i] = 44, 82 // degrees Celsius, synthetic
	}

	zPolicy := sax.DefaultOptions(8)
	zPolicy.ConstantPolicy = sax.ZeroWord
	zEncoder, err := sax.NewEncoder(zPolicy)
	if err != nil {
		fmt.Println(err)
		return
	}
	coolZ, err := zEncoder.Encode(stuckCool)
	if err != nil {
		fmt.Println(err)
		return
	}
	hotZ, err := zEncoder.Encode(stuckHot)
	if err != nil {
		fmt.Println(err)
		return
	}
	zDistance, err := zEncoder.Mindist(coolZ, hotZ)
	if err != nil {
		fmt.Println(err)
		return
	}

	rawPolicy := sax.Options{Segments: 8, Alphabet: 3, Normalization: sax.Raw,
		Breakpoints: []float64{50, 75}, BreakpointID: "temperature-celsius"}
	rawDistance, err := sax.MindistValues(stuckCool, stuckHot, rawPolicy)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Sensors: both-degenerate=%t Z-distance=%.0f raw-separates=%t\n",
		coolZ.Degenerate && hotZ.Degenerate, zDistance, rawDistance > 0)
	// Output: Sensors: both-degenerate=true Z-distance=0 raw-separates=true
}
