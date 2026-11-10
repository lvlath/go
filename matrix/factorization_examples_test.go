// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test documents deterministic real-world factorization pipelines.
// The examples exercise only shipped public constructors, factors, and solves.
package matrix_test

import (
	"context"
	"errors"
	"fmt"

	"github.com/lvlath/go/matrix"
)

// ExampleFactorLU_powerDistribution models two coupled power-balance equations.
// Partial pivoting avoids failure when the first diagonal happens to be zero.
func ExampleFactorLU_powerDistribution() {
	ctx := context.Background()
	coefficients, err := matrix.NewDense(2, 2)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	if err = coefficients.Set(0, 1, 1); err != nil {
		fmt.Println("example error:", err)
		return
	}
	if err = coefficients.Set(1, 0, 1); err != nil {
		fmt.Println("example error:", err)
		return
	}
	loads, err := matrix.NewDense(2, 1)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	if err = loads.Set(0, 0, 2); err != nil {
		fmt.Println("example error:", err)
		return
	}
	if err = loads.Set(1, 0, 3); err != nil {
		fmt.Println("example error:", err)
		return
	}
	factor, err := matrix.FactorLU(ctx, coefficients, matrix.FactorOptions{})
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	result, err := factor.Solve(ctx, loads)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	first, err := result.Solution.At(0, 0)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	second, err := result.Solution.At(1, 0)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	fmt.Printf("flows %.0f %.0f, perm %v, residual %.0f\n", first, second, factor.Permutation(), result.ResidualNorms[0])
	// Output: flows 3 2, perm [1 0], residual 0
}

// ExampleFactorQR_sensorCalibration fits two device sensitivities to three
// measured calibration signals without forming the normal-equation Gram matrix.
func ExampleFactorQR_sensorCalibration() {
	ctx := context.Background()
	features, err := matrix.NewDense(3, 2)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	for i, row := range [3][2]float64{{1, 0}, {0, 1}, {1, 1}} {
		for j, value := range row {
			if err = features.Set(i, j, value); err != nil {
				fmt.Println("example error:", err)
				return
			}
		}
	}
	response, err := matrix.NewDense(3, 1)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	for i, value := range [3]float64{1, 2, 3} {
		if err = response.Set(i, 0, value); err != nil {
			fmt.Println("example error:", err)
			return
		}
	}
	factor, err := matrix.FactorQR(ctx, features, matrix.FactorOptions{})
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	fit, err := factor.Solve(ctx, response)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	alpha, err := fit.Solution.At(0, 0)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	beta, err := fit.Solution.At(1, 0)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	fmt.Printf("sensitivities %.0f %.0f, residual %.0f\n", alpha, beta, fit.ResidualNorms[0])
	// Output: sensitivities 1 2, residual 0
}

// ExampleFactorCholesky_sensorCovariance solves a small SPD covariance
// coupling system without modifying the source uncertainty matrix.
func ExampleFactorCholesky_sensorCovariance() {
	ctx := context.Background()
	covariance, err := matrix.NewDense(2, 2)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	for i, row := range [2][2]float64{{4, 2}, {2, 3}} {
		for j, value := range row {
			if err = covariance.Set(i, j, value); err != nil {
				fmt.Println("example error:", err)
				return
			}
		}
	}
	observed, err := matrix.NewDense(2, 1)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	if err = observed.Set(0, 0, 6); err != nil {
		fmt.Println("example error:", err)
		return
	}
	if err = observed.Set(1, 0, 5); err != nil {
		fmt.Println("example error:", err)
		return
	}
	factor, err := matrix.FactorCholesky(ctx, covariance, 0, 0)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	result, err := factor.Solve(ctx, observed)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	first, err := result.Solution.At(0, 0)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	second, err := result.Solution.At(1, 0)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	fmt.Printf("corrected %.0f %.0f; rank %d\n", first, second, result.Rank)
	// Output: corrected 1 1; rank 2
}

// ExampleSolveLeastSquares_multichannelCalibration demonstrates the one-call
// least-squares facade with two independently calibrated response channels.
func ExampleSolveLeastSquares_multichannelCalibration() {
	ctx := context.Background()
	features, err := matrix.NewDense(3, 2)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	for i, row := range [3][2]float64{{1, 0}, {0, 1}, {1, 1}} {
		for j, value := range row {
			if err = features.Set(i, j, value); err != nil {
				fmt.Println("example error:", err)
				return
			}
		}
	}
	responses, err := matrix.NewDense(3, 2)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	for i, row := range [3][2]float64{{1, 3}, {2, 4}, {3, 7}} {
		for j, value := range row {
			if err = responses.Set(i, j, value); err != nil {
				fmt.Println("example error:", err)
				return
			}
		}
	}
	result, err := matrix.SolveLeastSquares(ctx, features, responses, matrix.FactorOptions{})
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	x00, err := result.Solution.At(0, 0)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	x01, err := result.Solution.At(0, 1)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	fmt.Printf("first coefficients %.0f %.0f; rhs %d\n", x00, x01, len(result.ResidualNorms))
	// Output: first coefficients 1 3; rhs 2
}

// ExampleFactorQR_rankDeficiency shows the explicit full-rank requirement.
// Two duplicate predictors must be rejected rather than silently fit using
// a pseudo-inverse or a hidden Tikhonov regularization.
func ExampleFactorQR_rankDeficiency() {
	ctx := context.Background()
	duplicateColumns, err := matrix.NewDense(3, 2)
	if err != nil {
		fmt.Println("example error:", err)
		return
	}
	for i, row := range [3][2]float64{{1, 1}, {2, 2}, {3, 3}} {
		for j, value := range row {
			if err = duplicateColumns.Set(i, j, value); err != nil {
				fmt.Println("example error:", err)
				return
			}
		}
	}
	factor, err := matrix.FactorQR(ctx, duplicateColumns, matrix.FactorOptions{})
	fmt.Printf("factor nil: %t; rank deficient: %t\n", factor == nil, errors.Is(err, matrix.ErrRankDeficient))
	// Output: factor nil: true; rank deficient: true
}
