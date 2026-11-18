// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm_test demonstrates five application-shaped GBM workflows.
// Price histories and shock scripts are explicitly synthetic, never live quotes.
package gbm_test

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/montecarlo"
	"github.com/lvlath/go/rng"
)

// fixtureNormals is a deterministic counterfactual stress stream, not a PRNG.
// Unlike stochastic experiments, fixed shocks provide fully reproducible price
// ledgers for incident replay, contract tests, and scenario demonstrations.
type fixtureNormals struct {
	shocks []float64
	index  int
}

// Uint64 is unused: GBM must take Gaussian shocks through Normal only.
func (f *fixtureNormals) Uint64() uint64 { return 0 }

// UniformOpen is unused: a GBM kernel must never build its own Gaussian draw.
func (f *fixtureNormals) UniformOpen() float64 { return 0.5 }

// Normal reads the next scenario shock in the same order as grid increments.
func (f *fixtureNormals) Normal() float64 {
	if f.index >= len(f.shocks) {
		// A depleted scenario is invalid numeric input, not a reason to panic.
		return math.NaN()
	}
	value := f.shocks[f.index]
	f.index++
	return value
}

// ExampleSimulate_intradayExecutionGuard applies irregular-tick GBM to an
// execution service's counterfactual short-horizon price-gap monitoring.
//
// Scenario:
//   - A 976 ms burst contains 33 irregularly spaced synthetic trade timestamps.
//   - A liquidity controller tracks the maximum one-tick move in basis points.
//   - Fixed Gaussian shock fixtures allow deterministic incident replay.
//
// Process:
//   - Convert explicitly measured milliseconds into TimeSeconds.
//   - Simulate a positive price path using a named per-sqrt-second volatility.
//   - Reduce prices to peak one-step dislocation and breach count.
//
// Interpretation:
//   - This diagnoses model-implied gap exposure, not actual market liquidity.
//   - Volatility is already per sqrt(second); no annualization occurs in gbm.
//
// Complexity: O(ticks) simulation, O(ticks) returned data, O(1) aggregation.
// AI-Hints: never pass Unix epoch seconds as grid.Values; rebase at first tick.
func ExampleSimulate_intradayExecutionGuard() {
	milliseconds := []int{
		0, 14, 32, 33, 58, 77, 91, 123, 149, 177, 195,
		220, 267, 282, 303, 339, 370, 395, 429, 468, 491, 529,
		552, 580, 607, 645, 690, 733, 780, 809, 850, 923, 976,
	}
	shocks := []float64{
		0.4, -0.8, 0.2, 1.3, -0.5, 0.6, 0.1, -0.9,
		1.7, -0.2, 0.8, -1.2, 0.3, 0.7, -0.4, 0.2,
		-1.8, 1.0, -0.7, 0.5, -0.1, -0.3, 1.4, -0.6,
		0.9, -1.3, 0.4, 0.1, -0.2, 0.8, -1.0, 0.3,
	}
	elapsedSeconds := make([]float64, len(milliseconds))
	for index, elapsed := range milliseconds {
		elapsedSeconds[index] = float64(elapsed) / 1000 // Explicit ms -> s policy.
	}

	const gapLimitBasisPoints = 4.0
	params := gbm.Parameters{Initial: 184.32, Drift: 0.00001, Volatility: 0.0018, Unit: gbm.TimeSeconds}
	stream := &fixtureNormals{shocks: shocks}
	result, err := gbm.Simulate(context.Background(), params,
		gbm.TimeGrid{Values: elapsedSeconds, Unit: gbm.TimeSeconds}, stream)
	if err != nil || !result.Complete {
		fmt.Println("simulation error:", err)
		return
	}

	maximumGap, breaches := 0.0, 0
	for index := 1; index < len(result.Values); index++ {
		gap := math.Abs(result.Values[index]/result.Values[index-1]-1) * 10000
		maximumGap = math.Max(maximumGap, gap)
		if gap > gapLimitBasisPoints {
			breaches++
		}
	}
	fmt.Printf("ticks=%d span=%.3fs shocks=%d peak-gap=%.2fbp breaches=%d\n",
		len(result.Values), result.Times[len(result.Times)-1], stream.index, maximumGap, breaches)
	fmt.Printf("starting-quote=%.2f ending-quote=%.2f\n", result.Values[0], result.Values[len(result.Values)-1])
	// Output:
	// ticks=33 span=0.976s shocks=32 peak-gap=5.12bp breaches=6
	// starting-quote=184.32 ending-quote=184.35
}

// europeanCallSampler bridges GBM terminal observations into the exact public
// montecarlo.Sampler interface. Each invocation receives its own path-owned RNG.
type europeanCallSampler struct {
	model    gbm.Parameters
	maturity float64
	strike   float64
	rate     float64
}

// Dimension reports one discounted payoff per logical Monte Carlo path.
// It is immutable and does not depend on scheduling or path ID.
func (s europeanCallSampler) Dimension() int { return 1 }

// Sample simulates the risk-neutral terminal share price, then discounts payoff.
// An incomplete trajectory must never be accepted as a full-maturity sample.
//
// Implementation:
//   - Stage 1: consume the driver's private RNG in exactly one GBM step.
//   - Stage 2: reject any propagated GBM error or incomplete path.
//   - Stage 3: discount max(S_T-K,0) and return a one-coordinate observation.
//
// Inputs: ctx is cancellable; pathID is a stable logical ID; random is path-owned.
// Returns: one discounted payoff and nil error on complete success.
// Errors: preserves GBM and context errors, or rejects broken completion invariant.
// Determinism: one Normal draw in the sampler's one terminal interval.
// Complexity: Time O(1), Space O(1).
// AI-Hints: set model.Drift=r-q; the Ito correction is already inside gbm.
func (s europeanCallSampler) Sample(ctx context.Context, _ uint64, random rng.RNG) ([]float64, error) {
	trajectory, err := gbm.SimulateTerminal(ctx, s.model, s.maturity, random)
	if err != nil {
		return nil, err
	}
	if trajectory == nil || !trajectory.Complete {
		return nil, errors.New("European call: incomplete GBM trajectory")
	}
	terminalPrice := trajectory.Values[len(trajectory.Values)-1]
	payoff := math.Max(terminalPrice-s.strike, 0)
	return []float64{math.Exp(-s.rate*s.maturity) * payoff}, nil
}

// ExampleSimulate_europeanCallHedgingCheck prices a dividend-paying European
// call with Monte Carlo and checks deterministic scheduling and analytic sanity.
//
// Scenario:
//   - Spot 100, strike 105, risk-free rate 4%, dividend yield 1%, sigma 23%.
//   - A 0.75-year call requires expectation under drift (rate-dividend).
//   - An operational service must preserve its estimate across 1 and 4 workers.
//
// Process:
//   - Adapt gbm.SimulateTerminal to montecarlo.Sampler using discounted payoff.
//   - Price 8192 reproducible paths and compare 1-worker/4-worker reductions.
//   - Independently compute Black-Scholes's closed-form call value.
//
// Interpretation:
//   - The analytic bound is a broad model-consistency smoke check, not a VaR
//     guarantee or a claim that market prices follow GBM.
//
// Complexity: O(paths) GBM work; O(1) dimension per path, driver-controlled storage.
// AI-Hints: terminal-only paths cannot price barriers or Asian averages.
func ExampleSimulate_europeanCallHedgingCheck() {
	const spot, strike = 100.0, 105.0
	const rate, dividend, sigma, years = 0.04, 0.01, 0.23, 0.75
	sampler := europeanCallSampler{
		model: gbm.Parameters{
			Initial: spot, Drift: rate - dividend, Volatility: sigma, Unit: gbm.TimeYears,
		},
		maturity: years, strike: strike, rate: rate,
	}
	plan := montecarlo.Plan{Paths: 8192, Seed: 20261118, Workers: 1}
	serial, err := montecarlo.Run(context.Background(), plan, sampler)
	if err != nil || !serial.Complete {
		fmt.Println("serial error:", err)
		return
	}
	plan.Workers = 4
	parallel, err := montecarlo.Run(context.Background(), plan, sampler)
	if err != nil || !parallel.Complete {
		fmt.Println("parallel error:", err)
		return
	}

	sigmaT := sigma * math.Sqrt(years)
	d1 := (math.Log(spot/strike) + (rate-dividend+0.5*sigma*sigma)*years) / sigmaT
	d2 := d1 - sigmaT
	normalCDF := func(value float64) float64 { return 0.5 * (1 + math.Erf(value/math.Sqrt2)) }
	analytic := spot*math.Exp(-dividend*years)*normalCDF(d1) -
		strike*math.Exp(-rate*years)*normalCDF(d2)
	fmt.Printf("paths=%d analytic=%.2f estimate-positive=%t\n",
		plan.Paths, analytic, serial.Mean[0] > 0)
	fmt.Printf("worker-replay=%t analytic-distance-under-1=%t\n",
		math.Float64bits(serial.Mean[0]) == math.Float64bits(parallel.Mean[0]),
		math.Abs(serial.Mean[0]-analytic) < 1.0)
	// Output:
	// paths=8192 analytic=6.73 estimate-positive=true
	// worker-replay=true analytic-distance-under-1=true
}

// ExampleSimulate_foreignExchangeReserveStress simulates a five-year foreign-
// currency liability under persistent inflation and occasional strong shocks.
//
// Scenario:
//   - A procurement portfolio owes foreign-currency invoices in coming years.
//   - Today's home-currency FX quote is 4.35 per one foreign-currency unit.
//   - Counterfactual drift 25%/year and volatility 40%/sqrt(year) are severe.
//
// Process:
//   - Simulate quarterly observation points on an explicit five-year grid.
//   - Compute the peak conversion rate and latest-year reserve exposure.
//   - Compare peak exposure against an explicit 50%-above-spot alert threshold.
//
// Interpretation:
//   - These fixed innovation values are a stress narrative, not calibrated FX
//     observations and not an empirical probability of devaluation.
//
// Complexity: O(quarters) time and returned memory, O(1) summary scratch.
// AI-Hints: a five-year stress model must not silently treat quarters as days.
func ExampleSimulate_foreignExchangeReserveStress() {
	const quarters = 20
	times := make([]float64, quarters+1)
	for index := range times {
		times[index] = float64(index) / 4 // Explicit year-fraction convention.
	}
	shocks := []float64{
		-0.6, 1.1, 0.3, -1.2, 0.9, 1.8, -0.4, 0.6, -1.5, 0.2,
		1.3, 0.5, -0.9, 1.4, -0.2, 0.7, -0.6, 1.6, -0.3, 0.8,
	}
	stream := &fixtureNormals{shocks: shocks}
	params := gbm.Parameters{Initial: 4.35, Drift: 0.25, Volatility: 0.40, Unit: gbm.TimeYears}
	result, err := gbm.Simulate(context.Background(), params,
		gbm.TimeGrid{Values: times, Unit: gbm.TimeYears}, stream)
	if err != nil || !result.Complete {
		fmt.Println("stress error:", err)
		return
	}

	peak := result.Values[0]
	for _, price := range result.Values[1:] {
		peak = math.Max(peak, price)
	}
	const foreignCurrencyInvoice = 2_000_000.0
	peakLiability := peak * foreignCurrencyInvoice
	fmt.Printf("horizon=%.0fy checkpoints=%d peak-fx=%.2f last-fx=%.2f\n",
		result.Times[len(result.Times)-1], len(result.Values), peak, result.Values[len(result.Values)-1])
	fmt.Printf("peak-liability=%.0f alert-above-50pct=%t\n",
		math.Round(peakLiability), peak > 1.5*params.Initial)
	// Output:
	// horizon=5y checkpoints=21 peak-fx=30.57 last-fx=30.57
	// peak-liability=61149582 alert-above-50pct=true
}

// ExampleSimulate_zeroVolatilityProcurementBudget models a fixed contract whose
// annual spend grows at a precisely agreed 6.2% geometric escalator rate.
//
// Scenario:
//   - A data-center vendor agreement begins at 2.4 million per contract year.
//   - Quarterly invoices reference the current deterministic annualized rate.
//   - Budget officers need a five-year cash-outlay total, with no RNG entropy.
//
// Process:
//   - Use sigma=0, drift=ln(1.062) and quarterly time points in TimeYears.
//   - Sum the twenty quarter-year payments and compute the year-five rate.
//   - Prove the RNG's Gaussian method is not even consulted.
//
// Interpretation:
//   - This is a contractual escalation calculation, not a stochastic forecast.
//   - The quarterly billing convention is an application decision, not GBM math.
//
// Complexity: O(quarters) simulation and total calculation, O(quarters) output.
// AI-Hints: sigma=0 still requires a nonnil RNG for canonical input validation.
func ExampleSimulate_zeroVolatilityProcurementBudget() {
	const quarters = 20
	times := make([]float64, quarters+1)
	for index := range times {
		times[index] = float64(index) / 4
	}
	params := gbm.Parameters{
		Initial: 2_400_000, Drift: math.Log(1.062), Volatility: 0, Unit: gbm.TimeYears,
	}
	stream := &fixtureNormals{}
	result, err := gbm.Simulate(context.Background(), params,
		gbm.TimeGrid{Values: times, Unit: gbm.TimeYears}, stream)
	if err != nil || !result.Complete {
		fmt.Println("budget error:", err)
		return
	}

	payments := 0.0
	for _, annualRate := range result.Values[1:] {
		payments += annualRate / 4 // Invoice at quarter end; explicit cashflow policy.
	}
	fmt.Printf("quarterly-invoices=%d year-five-rate=%.0f total-paid=%.0f\n",
		quarters, math.Round(result.Values[len(result.Values)-1]), math.Round(payments))
	fmt.Printf("Gaussian-draws=%d contract-factor=%.4f\n",
		stream.index, result.Values[len(result.Values)-1]/params.Initial)
	// Output:
	// quarterly-invoices=20 year-five-rate=3242155 total-paid=14105541
	// Gaussian-draws=0 contract-factor=1.3509
}

// ExampleSimulate_explicitTelemetryUnitConversion protects an IoT risk service
// from applying annualized volatility directly to elapsed sensor seconds.
//
// Scenario:
//   - A real-time risk service observes a two-minute irregular quote window.
//   - A separate model specification supplies drift and volatility per year.
//   - GBM refuses the incompatible TimeYears/TimeSeconds pairing.
//
// Process:
//   - Confirm a mismatched time grid raises ErrInvalidOption via errors.Is.
//   - Apply an explicitly documented 365.25-day calendar-year convention outside gbm.
//   - Re-run the same positive-elapsed-second grid with per-second parameters.
//
// Interpretation:
//   - No exchange-specific trading calendar or 252-session convention is implied.
//   - The chosen year fraction is a caller-controlled domain assumption.
//
// Complexity: O(observations) simulation and detached output memory.
// AI-Hints: sigma_seconds=sigma_years/sqrt(secondsPerYear), not /secondsPerYear.
func ExampleSimulate_explicitTelemetryUnitConversion() {
	seconds := []float64{0, 0.4, 1.7, 3.1, 5.2, 8.8, 12.5, 16, 23, 31, 41, 55, 70, 88, 102, 120}
	annual := gbm.Parameters{Initial: 102.75, Drift: 0.07, Volatility: 0.18, Unit: gbm.TimeYears}
	secondGrid := gbm.TimeGrid{Values: seconds, Unit: gbm.TimeSeconds}
	_, mismatchErr := gbm.Simulate(context.Background(), annual, secondGrid, &fixtureNormals{})

	const secondsPerCalendarYear = 365.25 * 24 * 60 * 60
	perSecond := gbm.Parameters{
		Initial:    annual.Initial,
		Drift:      annual.Drift / secondsPerCalendarYear,
		Volatility: annual.Volatility / math.Sqrt(secondsPerCalendarYear),
		Unit:       gbm.TimeSeconds,
	}
	stream := &fixtureNormals{shocks: []float64{
		0.2, -0.5, 0.1, 0.4, -0.8, 1.0, -0.3, 0.6,
		0.3, -0.1, -0.7, 0.9, 0.5, -0.2, 0.4,
	}}
	result, err := gbm.Simulate(context.Background(), perSecond, secondGrid, stream)
	if err != nil || !result.Complete {
		fmt.Println("conversion error:", err)
		return
	}
	fmt.Printf("unit-mismatch=%t corrected=%t elapsed-seconds=%.0f\n",
		errors.Is(mismatchErr, gbm.ErrInvalidOption), result.Complete,
		result.Times[len(result.Times)-1])
	fmt.Printf("ticks=%d Gaussian-draws=%d end-price=%.4f\n",
		len(result.Values), stream.index, result.Values[len(result.Values)-1])
	// Output:
	// unit-mismatch=true corrected=true elapsed-seconds=120
	// ticks=16 Gaussian-draws=15 end-price=102.7703
}
