// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima performs causal, anchored multistep ARIMA inference.
// This file separates conditional level forecasts from linear-memory marginal variances.
package arima

import "context"

// forecastKernel constructs a complete level and marginal-variance forecast at one origin.
// The historical innovation workspace and the future impulse-response workspace
// remain separate, preventing accidental use of future observed values.
//
// Implementation:
//   - Stage 1: difference caller history, reconstruct conditional residuals,
//     and extract original-coordinate level/slope anchors.
//   - Stage 2: advance expected future differences and integrate anchored means.
//   - Stage 3: compute integrated impulse variances and publish both full arrays.
//
// Behavior highlights:
//   - A failure in any stage discards all local arrays; no partial result escapes.
//   - Both inference tracks consume the same immutable coefficient snapshot.
//
// Inputs:
//   - ctx: active context; parameters: owned validated model parameter copy.
//   - history: caller-owned finite original levels, read-only during the call.
//   - horizon: positive checked length of the requested forecast window.
//
// Returns:
//   - *ForecastResult: detached Mean and Variance of exactly horizon points.
//
// Errors:
//   - ErrOverflow: unrepresentable difference, innovation, integration or variance.
//   - ctx.Err(): cancellation; no partial forecast is published.
//
// Determinism:
//   - Difference/innovation passes precede ascending future-mean and variance loops.
//
// Complexity:
//   - Time O((N+H)*(p+q+d+1)), Space O(N+H), for history N and horizon H.
//
// Notes:
//   - A Forecast call does not update or retain a cursor in the immutable model.
//
// AI-Hints:
//   - Do not substitute a training-time residual buffer: the caller can forecast
//     a different observed origin, and its conditional MA history must be recomputed.
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

// forecastMeans advances the future conditional ARMA means and reconstructs levels.
// Known historical y/epsilon values provide the conditioning state; previously
// forecast differences replace observations at future AR indices, while future
// innovations contribute zero to the conditional mean.
//
// Implementation:
//   - Stage 1: allocate horizon-length owned difference and level buffers.
//   - Stage 2: for each horizon, combine c, lagged observed/predicted AR y,
//     and only *known historical* MA residuals; reject nonfinite arithmetic.
//   - Stage 3: feed the predicted difference into historical anchors and publish level.
//
// Behavior highlights:
//   - MA contributions naturally stop when their lag references move beyond
//     the known origin; no synthetic future residuals are created.
//
// Inputs:
//   - ctx: active context with deterministic cooperative checks.
//   - parameters: validated fixed physical coefficients.
//   - historyDifferences: full differenced observed prefix, read-only.
//   - residuals: full conditional innovations on that observed prefix, read-only.
//   - origin: local integration anchors in original level units.
//   - horizon: positive buffer-admitted number of predicted levels.
//
// Returns:
//   - []float64: detached finite original-coordinate means, length horizon.
//
// Errors:
//   - ErrOverflow: required predicted y, slope, or physical level is nonfinite.
//   - ctx.Err(): canceled computation; the private forecast buffer is discarded.
//
// Determinism:
//   - Horizon steps ascend; within each step AR and MA lags ascend.
//
// Complexity:
//   - Time O(H*(p+q+d+1)), Space O(H), including both owned forecast buffers.
//
// Notes:
//   - The integration state is passed by value, never shared with another call.
//
// AI-Hints:
//   - For d=2, update the last first difference before updating the level;
//     starting from zero or shifting indices by one changes every forecast.
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

// forecastVariances computes marginal level-forecast innovation variances in O(H) space.
// It constructs the stationary ARMA impulse response and integrates it once or
// twice before accumulating squared responses for each forecast horizon.
//
// Implementation:
//   - Stage 1: allocate a horizon-sized response buffer and variance result.
//   - Stage 2: compute psi[0]=1, then theta[j]+sum(phi[i]*psi[j-i]) for j>0.
//   - Stage 3: integrate response d times, accumulate its squared energy, multiply
//     by innovation variance, and publish each finite marginal variance.
//
// Behavior highlights:
//   - Each variance includes only innovations that affect that particular horizon.
//   - There is no horizon-by-horizon covariance matrix allocation or inversion.
//
// Inputs:
//   - ctx: non-nil active context checked during the horizon loop.
//   - parameters: fixed AR/MA coefficients, d<=2, positive innovation variance.
//   - horizon: positive checked requested number of forecast horizons.
//
// Returns:
//   - []float64: detached finite marginal innovation variances, length horizon.
//
// Errors:
//   - ErrOverflow: response integration, accumulated square energy or final
//     innovation variance becomes unrepresentable.
//   - ctx.Err(): canceled calculation, with no published partial variances.
//
// Determinism:
//   - Response index increases from zero, and each AR lag is scanned forward.
//
// Complexity:
//   - Time O(H*(p+q+d+1)), Space O(H), no O(H^2) allocation.
//
// Notes:
//   - Parameter-estimation uncertainty and covariance across horizons are omitted.
//
// AI-Hints:
//   - Integrating forecast means is not enough to obtain level uncertainty;
//     the *impulse responses* must be integrated before their squares are summed.
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
