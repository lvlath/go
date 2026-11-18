// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo_test demonstrates realistic simulations using the driver API.
// Each model is an explicitly illustrative scenario, never a built-in driver law.
package montecarlo_test

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/montecarlo"
	"github.com/lvlath/go/rng"
)

// ExampleEstimateParallel_basketOption values a five-asset European basket call.
//
// Scenario:
//   - A treasury desk holds a one-year call on a diversified equity basket.
//   - Each risk-neutral terminal price follows its own geometric Brownian motion.
//   - The mean discounted payoff estimates value; the standard error estimates
//     sampling noise for that model and does not validate volatility assumptions.
//
// Process:
//   - Fix observable spots, portfolio weights, risk-free rate and volatilities.
//   - Simulate independent terminal asset shocks with reproducible per-path RNGs.
//   - Report the option price, a 95% normal-MC half-width and exercise frequency.
//
// AI-Hints:
//   - Correlation, dividends and calibration are deliberately excluded; a real
//     pricing engine must supply those model details inside its own Sampler.
func ExampleEstimateParallel_basketOption() {
	spots := [...]float64{128, 96, 141, 83, 115}
	weights := [...]float64{.25, .20, .20, .20, .15}
	volatilities := [...]float64{.22, .18, .26, .16, .21}
	const (
		riskFreeRate = .035
		maturity     = 1.0
		strike       = 115.0
		paths        = uint64(60000)
	)

	model := &funcSampler{2, func(_ context.Context, _ uint64, random rng.RNG) ([]float64, error) {
		basket := 0.0
		for asset := range spots {
			volatility := volatilities[asset]
			terminal := spots[asset] * math.Exp(
				(riskFreeRate-.5*volatility*volatility)*maturity+
					volatility*math.Sqrt(maturity)*random.Normal(),
			)
			basket += weights[asset] * terminal
		}
		payoff := math.Max(basket-strike, 0)
		exercised := 0.0
		if payoff > 0 {
			exercised = 1
		}
		return []float64{math.Exp(-riskFreeRate*maturity) * payoff, exercised}, nil
	}}

	result, err := montecarlo.EstimateParallel(context.Background(), paths, 20261118, 4, model)
	if err != nil {
		fmt.Println("basket error:", err)
		return
	}
	halfWidth := 1.96 * result.MeanStandardError[0]
	fmt.Printf("basket assets=%d trials=%d value=%.2f usd ci95_half=%.2f usd exercise=%.1f%%\n",
		len(spots), result.Completed, result.Mean[0], halfWidth, 100*result.Mean[1])

	// Output:
	// basket assets=5 trials=60000 value=5.82 usd ci95_half=0.07 usd exercise=55.2%
}

// ExampleRunWithQuantiles_portfolioTailRisk estimates loss VaR and expected shortfall.
//
// Scenario:
//   - A risk manager evaluates a $2.5m portfolio under a two-regime return law.
//   - The stressed regime has higher volatility and a negative conditional drift.
//   - The 99th percentile of simulated signed losses is loss-VaR; expected shortfall is
//     the mean loss conditional on exceeding that estimated threshold.
//
// Process:
//   - First pass: exact empirical 95%/99% loss quantiles, no public raw samples.
//   - Second pass: replay identical streams and reduce [tail loss, tail flag].
//   - Report VaR, tail conditional mean, and actual tail scenario count.
//
// AI-Hints:
//   - Expected shortfall is not implied by a quantile alone; exact replay must
//     preserve every sampler draw and the threshold's >= tie convention.
func ExampleRunWithQuantiles_portfolioTailRisk() {
	const (
		notional = 2500000.0
		paths    = uint64(60000)
		seed     = uint64(95801)
	)
	portfolioLoss := func(random rng.RNG) float64 {
		volatility, drift := .012, .0004
		if random.UniformOpen() < .09 {
			volatility, drift = .052, -.008
		}
		return -notional * (drift + volatility*random.Normal())
	}
	lossSampler := &funcSampler{1, func(_ context.Context, _ uint64, random rng.RNG) ([]float64, error) {
		return []float64{portfolioLoss(random)}, nil
	}}

	plan := montecarlo.Plan{Paths: paths, Seed: seed, Workers: 4}
	first, err := montecarlo.RunWithQuantiles(context.Background(), plan,
		[]float64{.95, .99}, lossSampler)
	if err != nil {
		fmt.Println("risk error:", err)
		return
	}
	var99, err := first.Quantiles.At(1, 0)
	if err != nil {
		fmt.Println("quantile error:", err)
		return
	}

	// Reusing seed+pathID exactly reproduces the original outcomes without
	// retaining an n-by-d matrix across the two experiment passes.
	tailSampler := &funcSampler{2, func(_ context.Context, _ uint64, random rng.RNG) ([]float64, error) {
		loss := portfolioLoss(random)
		if loss >= var99 {
			return []float64{loss, 1}, nil
		}
		return []float64{0, 0}, nil
	}}
	second, err := montecarlo.EstimateParallel(context.Background(), paths, seed, 8, tailSampler)
	if err != nil {
		fmt.Println("tail error:", err)
		return
	}
	shortfall := second.Mean[0] / second.Mean[1]
	tailCount := math.Round(float64(paths) * second.Mean[1])
	fmt.Printf("portfolio notional=%.0f usd VaR99=%.0f usd ES99=%.0f usd tail_cases=%.0f raw_retained=%t\n",
		notional, var99, shortfall, tailCount, first.Samples != nil)

	// Output:
	// portfolio notional=2500000 usd VaR99=177458 usd ES99=240954 usd tail_cases=600 raw_retained=false
}

// ExampleRunWithQuantiles_fatigueReliability screens structural crack growth.
//
// Scenario:
//   - Maintenance engineering models one component under variable-amplitude loads.
//   - Paris-law da/dN=C*(DeltaK)^m evolves an initial 2mm crack over repeated
//     blocks of stress cycles; a prescribed crack length triggers inspection.
//   - Results guide screening only, not certified damage-tolerance decisions.
//
// Process:
//   - Run independent sequences of load blocks using path-specific Gaussian shocks.
//   - Return final crack length (mm) and an inspection-trigger indicator.
//   - Inspect mean growth, upper-tail crack length and exceedance probability.
//
// AI-Hints:
//   - C's units must match m/(cycle*(MPa*sqrt(m))^m); sample-model uncertainty
//     and unmodelled crack geometry are not included in standard error.
func ExampleRunWithQuantiles_fatigueReliability() {
	const (
		initialCrackM           = .002
		inspectionLimitM        = .00227
		parisCoefficient        = 1.5e-11
		parisExponent           = 3.0
		geometryFactor          = 1.12
		cyclesPerBlock          = 100.0
		loadBlocks              = 200
		paths            uint64 = 5000
	)
	fatigue := &funcSampler{2, func(_ context.Context, _ uint64, random rng.RNG) ([]float64, error) {
		crackM := initialCrackM
		for block := 0; block < loadBlocks; block++ {
			// The stress-range floor prevents a negative/nonphysical DeltaK.
			stressRangeMPa := math.Max(40, 100+20*random.Normal())
			deltaK := geometryFactor * stressRangeMPa * math.Sqrt(math.Pi*crackM)
			crackM += cyclesPerBlock * parisCoefficient * math.Pow(deltaK, parisExponent)
		}
		needsInspection := 0.0
		if crackM >= inspectionLimitM {
			needsInspection = 1
		}
		return []float64{1000 * crackM, needsInspection}, nil
	}}

	result, err := montecarlo.RunWithQuantiles(context.Background(), montecarlo.Plan{
		Paths: paths, Seed: 81001, Workers: 4,
	}, []float64{.95, .99}, fatigue)
	if err != nil {
		fmt.Println("fatigue error:", err)
		return
	}
	p99CrackMM, err := result.Quantiles.At(1, 0)
	if err != nil {
		fmt.Println("fatigue quantile error:", err)
		return
	}
	fmt.Printf("fatigue cycles=%d trials=%d mean_crack=%.3f mm p99_crack=%.3f mm inspect=%.2f%%\n",
		int(cyclesPerBlock)*loadBlocks, result.Completed,
		result.Mean[0], p99CrackMM, 100*result.Mean[1])

	// Output:
	// fatigue cycles=20000 trials=5000 mean_crack=2.257 mm p99_crack=2.284 mm inspect=13.04%
}

// ExampleRun_gridReserveReliability estimates microgrid energy-not-served risk.
//
// Scenario:
//   - A grid-connected microgrid balances uncertain wind/solar and hourly demand with a
//     150MWh battery, finite converter power, and a firm import contract.
//   - A reliability planner cares about expected unserved energy, 95% daily
//     shortfall and average outage hours rather than the mean demand alone.
//
// Process:
//   - Simulate 24 hourly balances with renewable resource uncertainty.
//   - Discharge before shedding load; charge from surplus subject to caps.
//   - Return unserved MWh and outage hours, then inspect mean and tail metrics.
//
// AI-Hints:
//   - Battery dispatch is a scenario assumption, not a built-in optimizer;
//     inter-hour correlation and network constraints need richer Samplers.
func ExampleRun_gridReserveReliability() {
	const (
		batteryCapacityMWh = 150.0
		batteryPowerMW     = 40.0
		firmImportMW       = 165.0
		paths              = uint64(12000)
	)
	grid := &funcSampler{2, func(_ context.Context, _ uint64, random rng.RNG) ([]float64, error) {
		chargeMWh := 90.0
		unservedMWh, outageHours := 0.0, 0.0
		for hour := 0; hour < 24; hour++ {
			// A fixed load-shape and renewable-nameplate profile makes units explicit.
			peakShape := 55 * math.Sin(2*math.Pi*(float64(hour)-7)/24)
			demandMW := 305 + peakShape + 18*random.Normal()
			windMW := math.Max(0, 98+28*random.Normal())
			solarProfile := math.Max(0, math.Sin(math.Pi*(float64(hour)-6)/12))
			solarMW := math.Max(0, 145*solarProfile+14*random.Normal())
			deficitMW := demandMW - (windMW + solarMW + firmImportMW)
			if deficitMW > 0 {
				discharge := math.Min(deficitMW, math.Min(chargeMWh, batteryPowerMW))
				chargeMWh -= discharge
				remaining := deficitMW - discharge
				if remaining > 0 {
					unservedMWh += remaining // one-hour interval
					outageHours++
				}
			} else {
				// Surplus recharges within converter and battery energy limits.
				chargeMWh += math.Min(-deficitMW,
					math.Min(batteryPowerMW, batteryCapacityMWh-chargeMWh))
			}
		}
		return []float64{unservedMWh, outageHours}, nil
	}}

	result, err := montecarlo.Run(context.Background(), montecarlo.Plan{
		Paths: paths, Seed: 19919, Workers: 4, Quantiles: []float64{.95},
	}, grid)
	if err != nil {
		fmt.Println("grid error:", err)
		return
	}
	p95Unserved, err := result.Quantiles.At(0, 0)
	if err != nil {
		fmt.Println("grid quantile error:", err)
		return
	}
	fmt.Printf("microgrid days=%d EENS=%.1f MWh/day p95=%.1f MWh outage=%.2f h/day\n",
		result.Completed, result.Mean[0], p95Unserved, result.Mean[1])

	// Output:
	// microgrid days=12000 EENS=102.8 MWh/day p95=230.0 MWh outage=4.23 h/day
}

// ExampleRunWithSamples_interruptedCalibration models a failed sensor-batch run.
//
// Scenario:
//   - A vibration laboratory processes independent test-rig calibrations.
//   - An interlock aborts the batch when sample ID3 reports an unsafe condition.
//   - Previously validated records must remain an auditable contiguous prefix;
//     unexecuted calibrations must not be represented by synthetic zeros.
//
// Process:
//   - Run a serial experiment with retained rows and requested quantiles.
//   - Trigger cancellation from the interlock and inspect Result + errors.Is.
//   - Print only verified prefix count, mean magnitude and disabled quantiles.
//
// AI-Hints:
//   - Partial results are not a successful experiment; quantile estimates are
//     intentionally nil even if sufficient earlier rows already exist.
func ExampleRunWithSamples_interruptedCalibration() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	calibration := &funcSampler{2, func(_ context.Context, id uint64, _ rng.RNG) ([]float64, error) {
		if id == 3 {
			cancel() // emergency interlock, before publishing the unsafe observation
			return nil, context.Canceled
		}
		// An archived instrument fixture records acceleration and temperature bias.
		return []float64{2 + .4*float64(id), 21 + float64(id)}, nil
	}}
	result, err := montecarlo.RunWithSamples(ctx, montecarlo.Plan{
		Paths: 100, Seed: 24, Workers: 1, Quantiles: []float64{.95},
	}, calibration)
	if result == nil {
		fmt.Println("calibration error:", err)
		return
	}
	fmt.Printf("calibration requested=%d verified=%d accel_mean=%.2f m/s2 cancelled=%t quantiles_published=%t\n",
		result.Requested, result.Completed, result.Mean[0],
		errors.Is(err, context.Canceled), result.Quantiles != nil)

	// Output:
	// calibration requested=100 verified=3 accel_mean=2.40 m/s2 cancelled=true quantiles_published=false
}
