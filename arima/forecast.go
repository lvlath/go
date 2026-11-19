// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima performs causal, anchored multistep ARIMA inference.
// This file separates conditional level forecasts from linear-memory marginal variances.
package arima

import "context"

// forecastKernel prepares a caller-owned historical origin and forecasts future levels.
// Its two independent stages reuse only immutable coefficients and local workspaces:
// means use fitted historical innovations, while variances use future impulse responses.
// Time O((N+H)*(p+q+d+1)), Space O(N+H), without H-by-H covariance matrices.
func forecastKernel(ctx context.Context, parameters Parameters, history []float64, horizon int) (*ForecastResult, error) {
	// Recreate exactly the differencing and residual initialization used by Fit.
	differenced, err := difference(history, parameters.Order.Difference)
	if err != nil {
		return nil, err
	}
	residuals, err := innovations(ctx, differenced, parameters)
	if err != nil {
		return nil, err
	}
	origin, err := lastAnchors(history, parameters.Order.Difference)
	if err != nil {
		return nil, err
	}

	means, err := forecastMeans(ctx, parameters, differenced, residuals, origin, horizon)
	if err != nil {
		return nil, err
	}
	variances, err := forecastVariances(ctx, parameters, horizon)
	if err != nil {
		return nil, err
	}
	return &ForecastResult{Mean: means, Variance: variances, Horizon: horizon}, nil
}

// forecastMeans advances the differenced conditional ARMA recurrence at one origin.
// AR terms use already predicted differences after the origin; historical MA
// residuals contribute until they leave the q-lag window, while future eps=0.
// Inverse integration updates the supplied local anchors after every step.
// Time O(H*(p+q+d+1)), Space O(H); all output arrays are independently owned.
func forecastMeans(ctx context.Context, parameters Parameters, historyDifferences, residuals []float64, origin anchors, horizon int) ([]float64, error) {
	pastLength := len(historyDifferences)
	predictedDifferences := make([]float64, horizon)
	means := make([]float64, horizon)

	for step := 0; step < horizon; step++ {
		if step&contextCheckMask == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		prediction := parameters.Drift
		for lag, coefficient := range parameters.AR {
			index := pastLength + step - lag - 1
			if index >= pastLength {
				prediction += coefficient * predictedDifferences[index-pastLength]
			} else {
				prediction += coefficient * historyDifferences[index]
			}
		}
		for lag, coefficient := range parameters.MA {
			index := pastLength + step - lag - 1
			if index < pastLength { // A future innovation's conditional mean is exactly zero.
				prediction += coefficient * residuals[index]
			}
		}
		if !finite(prediction) {
			return nil, ErrOverflow
		}
		predictedDifferences[step] = prediction
		level, err := origin.integrate(prediction, parameters.Order.Difference)
		if err != nil {
			return nil, err
		}
		means[step] = level
	}
	return means, nil
}

// forecastVariances constructs stationary and integrated ARMA impulse responses.
// For psi_0=1 and psi_j=theta_j+sum_i phi_i psi_{j-i}, one or two cumulative
// sums account for integrated levels. The marginal error variance at horizon h
// is sigma^2*sum_{j=0}^{h-1}(integratedPsi_j)^2, not a parameter uncertainty band.
// Time O(H*(p+q+d+1)), Space O(H); no dense forecast covariance is ever allocated.
func forecastVariances(ctx context.Context, parameters Parameters, horizon int) ([]float64, error) {
	stationaryResponse := make([]float64, horizon)
	variances := make([]float64, horizon)
	firstIntegral, secondIntegral, sumSquares := 0.0, 0.0, 0.0

	for index := 0; index < horizon; index++ {
		if index&contextCheckMask == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		response := 0.0
		if index == 0 {
			response = 1 // Unit innovation at the forecast origin.
		} else if index <= len(parameters.MA) {
			response = parameters.MA[index-1] // theta_index, zero outside q.
		}
		for lag, phi := range parameters.AR {
			if index > lag {
				response += phi * stationaryResponse[index-lag-1]
			}
		}
		if !finite(response) {
			return nil, ErrOverflow
		}
		stationaryResponse[index] = response

		if parameters.Order.Difference >= 1 {
			firstIntegral += response // d=1: response to accumulated level shocks.
			response = firstIntegral
		}
		if parameters.Order.Difference == 2 {
			secondIntegral += response // d=2: integrate the first response again.
			response = secondIntegral
		}
		if !finite(response) {
			return nil, ErrOverflow
		}
		sumSquares += response * response
		variance := parameters.InnovationVariance * sumSquares
		if !finite(variance) {
			return nil, ErrOverflow
		}
		variances[index] = variance
	}
	return variances, nil
}
