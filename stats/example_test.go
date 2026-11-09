// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats_test demonstrates five independent statistical operating modes.
// Each example uses the public API, consumes output, and has deterministic output.
// AI-Hints: production code must check returned errors, especially ErrOverflow.
package stats_test

import (
	"fmt"
	"math"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/stats"
)

// ExampleSummarize compares population and sample variance conventions.
// The two wrappers explicitly encode the denominator without changing the
// observed mean, extrema, or number of readings in an input batch.
func ExampleSummarize() {
	// Four finite measurements belong to one prepared acquisition window.
	readings := []float64{12, 14, 16, 18}
	population, err := stats.SummarizePopulation(readings)
	if err != nil {
		panic(err)
	}
	sample, err := stats.SummarizeSample(readings)
	if err != nil {
		panic(err)
	}

	fmt.Printf("count=%d mean=%.1f populationVar=%.1f sampleVar=%.3f\n",
		population.Count, population.Mean, population.Variance, sample.Variance)
	// Output: count=4 mean=15.0 populationVar=5.0 sampleVar=6.667
}

// ExampleMoments_Update demonstrates long-offset streaming observations.
// A tiny spread around a large baseline must not collapse to zero merely
// because the sample mean has a much larger absolute magnitude.
func ExampleMoments_Update() {
	measurements := []float64{1e12 + 1, 1e12 + 2, 1e12 + 3}
	var moments stats.Moments // The zero value is a valid empty accumulator.

	// Consume each observation once without buffering a second moment history.
	for _, measurement := range measurements {
		if err := moments.Update(measurement); err != nil {
			panic(err)
		}
	}
	summary, err := moments.Summary(1)
	if err != nil {
		panic(err)
	}
	fmt.Printf("mean=%.0f sampleVar=%.0f\n", summary.Mean, summary.Variance)
	// Output: mean=1000000000002 sampleVar=1
}

// ExampleMoments_Merge demonstrates a distributed full-range reduction.
// Two partitions combine a signed extreme dataset whose intermediate M2
// does not fit in float64, although the final sample variance does.
func ExampleMoments_Merge() {
	partitionA, partitionB := stats.NewMoments(), stats.NewMoments()
	for _, observation := range []float64{-1e154, 0} {
		if err := partitionA.Update(observation); err != nil {
			panic(err)
		}
	}
	if err := partitionB.Update(1e154); err != nil {
		panic(err)
	}

	// Merge only after each partition is complete and no longer mutating.
	if err := partitionA.Merge(partitionB); err != nil {
		panic(err)
	}
	summary, err := partitionA.Summary(1)
	if err != nil {
		panic(err)
	}
	fmt.Printf("count=%d mean=%.0f sampleVar=%.2e\n",
		summary.Count, summary.Mean, summary.Variance)
	// Output: count=3 mean=0 sampleVar=1.00e+308
}

// ExampleFitStandardizer demonstrates training/heldout feature separation.
// Column 0 varies, whereas column 1 was constant in training; the unit-scale
// policy preserves new holdout deviations instead of forcing them to zero.
func ExampleFitStandardizer() {
	train, err := matrix.NewDense(4, 2)
	if err != nil {
		panic(err)
	}
	training := [4][2]float64{{10, 5}, {12, 5}, {14, 5}, {16, 5}}
	for row, values := range training {
		for col, value := range values {
			if err := train.Set(row, col, value); err != nil {
				panic(err)
			}
		}
	}

	// Fit exactly once and explicitly accept zero-variance training columns.
	fit, err := stats.FitPopulationStandardizer(train, stats.UnitScaleConstant)
	if err != nil {
		panic(err)
	}
	holdout, err := matrix.NewDense(1, 2)
	if err != nil {
		panic(err)
	}
	if err = holdout.Set(0, 0, 18); err != nil {
		panic(err)
	}
	if err = holdout.Set(0, 1, 7); err != nil {
		panic(err)
	}

	z, err := fit.Model.Transform(holdout)
	if err != nil {
		panic(err)
	}
	varying, err := z.At(0, 0)
	if err != nil {
		panic(err)
	}
	constant, err := z.At(0, 1)
	if err != nil {
		panic(err)
	}
	fmt.Printf("varyingZ=%.3f constantDelta=%.1f\n", varying, constant)
	// Output: varyingZ=2.236 constantDelta=2.0
}

// ExampleStandardizer_InverseTransform demonstrates fused inverse arithmetic.
// A large negative scaled product may overflow by itself, but adding the
// trained mean with FMA can still produce a finite reconstructed value.
func ExampleStandardizer_InverseTransform() {
	train, err := matrix.NewDense(2, 1)
	if err != nil {
		panic(err)
	}
	if err = train.Set(0, 0, 0.65e308); err != nil {
		panic(err)
	}
	if err = train.Set(1, 0, 1.75e308); err != nil {
		panic(err)
	}
	fit, err := stats.FitStandardizer(train, stats.ScaleOptions{})
	if err != nil {
		panic(err)
	}
	z, err := matrix.NewDense(1, 1)
	if err != nil {
		panic(err)
	}
	if err = z.Set(0, 0, -4); err != nil {
		panic(err)
	}
	restored, err := fit.Model.InverseTransform(z)
	if err != nil {
		panic(err)
	}
	value, err := restored.At(0, 0)
	if err != nil {
		panic(err)
	}
	fmt.Printf("finite=%t negative=%t\n", !math.IsInf(value, 0), value < 0)
	// Output: finite=true negative=true
}
