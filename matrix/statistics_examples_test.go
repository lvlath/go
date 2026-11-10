// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test demonstrates stable finite-valued statistical pipelines.
// Every example uses the real public API, owns the source observations, and
// prints only deterministic caller-visible results.
package matrix_test

import (
	"fmt"
	"math"

	"github.com/lvlath/go/matrix"
)

// ExampleCenterColumns_sensorBaseline demonstrates centering three readings
// against their respective column baselines without mutating the input.
func ExampleCenterColumns_sensorBaseline() {
	readings, err := matrix.NewDense(4, 3)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	values := [4][3]float64{{100, 10, 5}, {102, 11, 5}, {98, 9, 5}, {100, 10, 5}}
	for i := range values {
		for j, value := range values[i] {
			if err = readings.Set(i, j, value); err != nil {
				fmt.Println("example error:", err)
				return
			}
		}
	}
	centered, means, err := matrix.CenterColumns(readings)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	deviation, err := centered.At(1, 0)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	fmt.Printf("baselines %.0f %.0f %.0f; excursion %.0f\n", means[0], means[1], means[2], deviation)
	// Output: baselines 100 10 5; excursion 2
}

// ExampleCovariance_sensorRelationship computes a two-feature sample
// covariance, preserving a negative relationship between two sensor channels.
func ExampleCovariance_sensorRelationship() {
	readings, err := matrix.NewDense(4, 2)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	for i, row := range [4][2]float64{{1, 4}, {2, 3}, {3, 2}, {4, 1}} {
		for j, value := range row {
			if err = readings.Set(i, j, value); err != nil {
				fmt.Println("example error:", err)
				return
			}
		}
	}
	covariance, _, err := matrix.Covariance(readings)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	cross, err := covariance.At(0, 1)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	fmt.Printf("inverse sensor covariance: %.3f\n", cross)
	// Output: inverse sensor covariance: -1.667
}

// ExampleCorrelation_invariantChannel shows two anticorrelated features and
// one constant channel, whose entire correlation row remains zero.
func ExampleCorrelation_invariantChannel() {
	input, err := matrix.NewDense(5, 3)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	for i := 0; i < 5; i++ {
		for j, value := range [3]float64{float64(i), float64(4 - i), 7} {
			if err = input.Set(i, j, value); err != nil {
				fmt.Println("example error:", err)
				return
			}
		}
	}
	correlations, _, deviations, err := matrix.Correlation(input)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	pair, err := correlations.At(0, 1)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	constantDiagonal, err := correlations.At(2, 2)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	fmt.Printf("pair %.0f; constant deviation %.0f; diagonal %.0f\n", pair, deviations[2], constantDiagonal)
	// Output: pair -1; constant deviation 0; diagonal 0
}

// ExampleNormalizeRowsL2_largeCoordinates normalizes a signal whose raw
// square would overflow float64, while publishing its true finite norm.
func ExampleNormalizeRowsL2_largeCoordinates() {
	input, err := matrix.NewDense(2, 1)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	if err = input.Set(0, 0, 1e308); err != nil {
		fmt.Println("example error:", err)
		return
	}
	normalized, norms, err := matrix.NormalizeRowsL2(input)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	first, err := normalized.At(0, 0)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	second, err := normalized.At(1, 0)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	fmt.Printf("norm %.0e, normalized %.0f, empty %.0f\n", norms[0], first, second)
	// Output: norm 1e+308, normalized 1, empty 0
}

// ExampleCovariance_largeCancellation uses a response and predictor whose
// enormous opposing values cancel; the finite cross-moment must survive.
func ExampleCovariance_largeCancellation() {
	input, err := matrix.NewDense(3, 2)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	for i, row := range [3][2]float64{{1e150, 1}, {-1e150, 1}, {1, -2}} {
		for j, value := range row {
			if err = input.Set(i, j, value); err != nil {
				fmt.Println("example error:", err)
				return
			}
		}
	}
	covariance, means, err := matrix.Covariance(input)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	cross, err := covariance.At(0, 1)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	fmt.Printf("mean %0.4f; covariance %0.0f; finite %t\n", means[0], cross, !math.IsInf(cross, 0))
	// Output: mean 0.3333; covariance -1; finite true
}
