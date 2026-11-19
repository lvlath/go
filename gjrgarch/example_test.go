// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test demonstrates fixed, reproducible financial risk workflows.
// Return fixtures are illustrative stress-like data, not claims of historical prices.
package gjrgarch_test

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/gjrgarch"
)

// equityStressReturns is a fixed 40-session sequence of illustrative daily
// decimal equity returns, with drawdown, sharp downside gaps, and rebounds.
// An explicit static fixture makes the example reproducible without market APIs.
func equityStressReturns() []float64 {
	return []float64{
		0.006, -0.009, 0.012, 0.003, -0.017, 0.011, -0.013, 0.008,
		0.004, -0.022, -0.036, 0.017, -0.041, 0.026, -0.072, 0.045,
		-0.038, 0.031, -0.028, -0.019, 0.024, -0.015, 0.016, 0.009,
		-0.013, 0.018, -0.025, 0.014, 0.005, -0.011, -0.008, 0.017,
		-0.023, 0.012, 0.006, -0.014, 0.019, -0.008, 0.011, -0.017,
	}
}

// ExampleModel_equityGapStress compares the next-session volatility attributed
// to a 3% equity-index gap down versus an equally sized rebound.
//
// Scenario:
//   - A portfolio risk desk receives daily close-to-close equity returns.
//   - A sudden -3% shock is tested against +3%, using an identical prior variance.
//   - The volatility difference is an exposure-control input, not an order signal.
//
// Process:
//   - Filter a fixed 40-session stress-like path from stationary initialization.
//   - Construct two posterior states differing only in the last signed innovation.
//   - Compare the next conditional volatilities and report basis-point difference.
//
// Policy:
//   - Fixed illustrative returns, not an actual S&P 500 historical dataset.
//   - The model is an explicitly specified scenario, not an empirically fitted estimate.
func ExampleModel_equityGapStress() {
	ctx := context.Background()
	model, err := gjrgarch.New(gjrgarch.Parameters{
		Mean: 0.0003, Omega: 0.000018, Alpha: 0.07, Gamma: 0.14, Beta: 0.77,
	})
	if err != nil {
		fmt.Println("invalid stress model")
		return
	}
	observations := equityStressReturns()
	variances, err := model.FilterStationary(ctx, observations)
	if err != nil {
		fmt.Println("invalid equity series")
		return
	}
	latestVariance := variances[len(variances)-1]
	fall, err := model.InitialState(-0.03, latestVariance)
	if err != nil {
		fmt.Println("invalid downside scenario")
		return
	}
	rebound, err := model.InitialState(0.03, latestVariance)
	if err != nil {
		fmt.Println("invalid upside scenario")
		return
	}
	fallVariance, err := model.ForecastNextVariance(fall)
	if err != nil {
		fmt.Println("downside forecast unavailable")
		return
	}
	reboundVariance, err := model.ForecastNextVariance(rebound)
	if err != nil {
		fmt.Println("upside forecast unavailable")
		return
	}
	// Convert decimal daily sigma into basis points of index return.
	gapBasisPoints := math.Round((math.Sqrt(fallVariance) - math.Sqrt(reboundVariance)) * 10000)
	fmt.Printf("equity sessions=%d downside_vol_higher=%t gap_bp=%.0f\n",
		len(observations), fallVariance > reboundVariance, gapBasisPoints)
	// Output:
	// equity sessions=40 downside_vol_higher=true gap_bp=39
}

// ExampleModel_cryptoCollateralFeed models a margin-monitoring input from
// five-minute perpetual-futures return bars arriving sequentially.
//
// Scenario:
//   - A collateral service marks the next bar's conditional standard deviation.
//   - It counts threshold crossings without allowing the incoming bar to affect
//     its own conditional variance (which would leak future information).
//   - A production gateway would add calibrated risk limits, market-data QA,
//     liquidation constraints, and audit logging outside this package.
//
// Process:
//   - Start with an explicitly chosen pre-feed conditional variance.
//   - Consume fixed illustrative five-minute return bars using State and Step.
//   - Query each next-bar variance with the no-slice ForecastNextVariance facade.
func ExampleModel_cryptoCollateralFeed() {
	model, err := gjrgarch.New(gjrgarch.Parameters{
		Omega: 0.000004, Alpha: 0.08, Gamma: 0.18, Beta: 0.7,
	})
	if err != nil {
		fmt.Println("invalid feed model")
		return
	}
	// Fractional 5-minute returns, with a short-lived selloff and rebound.
	bars := []float64{
		0.004, -0.006, 0.003, -0.008, 0.012, -0.004, -0.017, 0.009,
		-0.005, 0.007, -0.011, 0.006, 0.003, -0.014, 0.018, -0.026,
		-0.013, 0.021, -0.035, 0.024, -0.019, 0.007, -0.012, 0.006,
		0.008, -0.009, 0.003, 0.004, -0.007, 0.011, -0.004, -0.015,
		0.009, -0.008, 0.006, -0.012, 0.004, -0.006, 0.013, -0.002,
		-0.004, 0.007, 0.004, -0.009, 0.005, -0.006, 0.003, -0.005,
	}
	state, err := model.InitialState(bars[0], 0.00003)
	if err != nil {
		fmt.Println("invalid feed start")
		return
	}
	const volatilityReviewLevel = 0.009 // 0.9% conditional sigma per five-minute bar.
	alerts := 0
	for i, observed := range bars {
		if i > 0 {
			state, err = model.Step(state, observed)
			if err != nil {
				fmt.Println("feed variance unavailable")
				return
			}
		}
		variance, forecastErr := model.ForecastNextVariance(state)
		if forecastErr != nil {
			fmt.Println("feed forecast unavailable")
			return
		}
		if math.Sqrt(variance) >= volatilityReviewLevel {
			alerts++
		}
	}
	fmt.Printf("futures bars=%d next_bar_review_events=%d causal_steps=%d\n", len(bars), alerts, state.Steps)
	// Output:
	// futures bars=48 next_bar_review_events=18 causal_steps=48
}

// ExampleModel_tenDayLiquidityBuffer constructs a forecast-based liquidity
// planning proxy for a desk holding an equity exposure through a volatile week.
//
// Scenario:
//   - Treasury needs a conservative planning view across ten future sessions.
//   - It aggregates model-implied daily conditional expected variances as a
//     Gaussian proxy, then compares the result with a fixed review budget.
//   - This is NOT an exact ten-day GJR predictive-tail distribution or VaR backtest.
//
// Process:
//   - Forecast from a known downside observation with a known variance.
//   - Add expected daily variances and compute a 99% Gaussian planning proxy.
//   - Compare that proxy against a budget chosen outside the algorithm.
func ExampleModel_tenDayLiquidityBuffer() {
	model, err := gjrgarch.New(gjrgarch.Parameters{
		Omega: 0.000018, Alpha: 0.09, Gamma: 0.14, Beta: 0.72,
	})
	if err != nil {
		fmt.Println("invalid liquidity model")
		return
	}
	state, err := model.InitialState(-0.028, 0.00032)
	if err != nil {
		fmt.Println("invalid liquidity state")
		return
	}
	curve, err := model.ForecastVariance(state, 10)
	if err != nil {
		fmt.Println("forecast unavailable")
		return
	}
	cumulativeVariance := 0.0
	for _, variance := range curve {
		cumulativeVariance += variance
	}
	const gaussian99Quantile = 2.3263478740408408
	const tenDayLossReviewLevel = 0.12 // 12% equity-exposure loss proxy.
	proxy := gaussian99Quantile * math.Sqrt(cumulativeVariance)
	fmt.Printf("treasury horizon=%d day1_vol=%.1f%% gaussian_proxy=%.1f%% review=%t\n",
		len(curve), 100*math.Sqrt(curve[0]), 100*proxy, proxy > tenDayLossReviewLevel)
	// Output:
	// treasury horizon=10 day1_vol=2.1% gaussian_proxy=13.1% review=true
}

// ExampleFit_modelValidationGate evaluates a frozen candidate against a strictly
// chronological held-out return period, preserving the last training state.
//
// Scenario:
//   - A model-risk team receives fixed training returns and an explicit candidate.
//   - Short-budget QMLE produces a last valid candidate without proof of convergence.
//   - Every held-out observation is scored using the variance known BEFORE that
//     observation arrives; no in-sample filtering or stationary reset leaks forward.
//
// Process:
//   - Obtain the candidate and verify its training objective did not deteriorate.
//   - Filter training returns once and construct their terminal posterior state.
//   - Predict each held-out variance before advancing with that held-out return.
//   - Keep the release gate closed until convergence and independent risk review.
//
// Notes:
//   - The deterministic sequence is illustrative, not retrieved market history.
//   - Gaussian predictive loss is a scoring diagnostic, not a VaR coverage proof.
func ExampleFit_modelValidationGate() {
	training := equityStressReturns()
	heldOut := []float64{
		-0.012, 0.008, 0.004, -0.027, 0.016, -0.006, 0.011, -0.021,
		0.005, 0.007, -0.009, 0.014, -0.017, 0.012, 0.002, -0.008,
	}
	ctx := context.Background()
	initial := gjrgarch.Parameters{Omega: 0.00003, Alpha: 0.08, Gamma: 0.1, Beta: 0.64}
	fit, fitErr := gjrgarch.Fit(ctx, training, gjrgarch.Options{
		Initial: initial, MaxIterations: 1, VarianceFloor: 1e-12,
	})
	if fitErr != nil && !errors.Is(fitErr, gjrgarch.ErrNotConverged) {
		fmt.Println("training failed without a valid model")
		return
	}
	if fit == nil || fit.Model == nil {
		fmt.Println("candidate missing")
		return
	}
	// The last in-sample h belongs to the last in-sample return. The first
	// held-out forecast must incorporate that return before scoring day one.
	trainingVariances, err := fit.Model.FilterStationary(ctx, training)
	if err != nil {
		fmt.Println("training history invalid")
		return
	}
	state, err := fit.Model.InitialState(training[len(training)-1], trainingVariances[len(trainingVariances)-1])
	if err != nil {
		fmt.Println("terminal posterior invalid")
		return
	}
	mean := fit.Model.Parameters().Mean
	averageHeldOutLoss := 0.0
	for _, observed := range heldOut {
		variance, forecastErr := fit.Model.ForecastNextVariance(state)
		if forecastErr != nil {
			fmt.Println("held-out variance unavailable")
			return
		}
		innovation := (observed - mean) / math.Sqrt(variance)
		averageHeldOutLoss += 0.5 * (math.Log(2*math.Pi) + math.Log(variance) + innovation*innovation)
		state, err = fit.Model.Step(state, observed)
		if err != nil {
			fmt.Println("held-out posterior invalid")
			return
		}
	}
	averageHeldOutLoss /= float64(len(heldOut))
	history := fit.Diagnostics.Objective
	objectiveNotWorse := history[len(history)-1] <= history[0]
	deployable := fitErr == nil && fit.Diagnostics.Converged
	fmt.Printf("training=%d holdout=%d objective_not_worse=%t heldout_nll=%.3f deployable=%t\n",
		len(training), len(heldOut), objectiveNotWorse, averageHeldOutLoss, deployable)
	// Output:
	// training=40 holdout=16 objective_not_worse=true heldout_nll=-2.724 deployable=false
}

// ExampleFit_failClosedRiskDeployment illustrates release governance with a
// strict optimizer budget and rejection of stale model-bound state.
//
// Scenario:
//   - A margin service has a validated production model and persisted posterior state.
//   - A proposed replacement is trained under a deliberately inadequate budget.
//   - The service must not deploy that candidate or reuse the old model's state.
//
// Process:
//   - Fit an explicitly initialized candidate and classify ErrNotConverged.
//   - Verify cross-model state rejection through the published ModelID contract.
//   - Continue risk calculations using the independently validated incumbent.
func ExampleFit_failClosedRiskDeployment() {
	incumbent, err := gjrgarch.New(gjrgarch.Parameters{
		Omega: 0.00002, Alpha: 0.08, Gamma: 0.12, Beta: 0.7,
	})
	if err != nil {
		fmt.Println("invalid incumbent")
		return
	}
	storedState, err := incumbent.InitialState(-0.018, 0.00034)
	if err != nil {
		fmt.Println("invalid persisted state")
		return
	}
	candidate, fitErr := gjrgarch.Fit(context.Background(), equityStressReturns(), gjrgarch.Options{
		Initial:       gjrgarch.Parameters{Omega: 0.00003, Alpha: 0.11, Gamma: 0.08, Beta: 0.64},
		MaxIterations: 1,
	})
	if candidate == nil || candidate.Model == nil {
		fmt.Println("candidate missing")
		return
	}
	_, stateErr := candidate.Model.ForecastNextVariance(storedState)
	fallbackVariance, forecastErr := incumbent.ForecastNextVariance(storedState)
	if forecastErr != nil {
		fmt.Println("incumbent forecast failed")
		return
	}
	fmt.Printf("release_blocked=%t stale_state_rejected=%t incumbent_next_variance_positive=%t\n",
		errors.Is(fitErr, gjrgarch.ErrNotConverged),
		errors.Is(stateErr, gjrgarch.ErrStateVersion), fallbackVariance > 0)
	// Output:
	// release_blocked=true stale_state_rejected=true incumbent_next_variance_positive=true
}
