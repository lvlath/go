// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap_test provides executable production-style index pipelines.
// The five outputs assert structural facts rather than RNG-version-specific numbers.
// The examples use illustrative, deterministic data rather than live market feeds.
package bootstrap_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/lvlath/go/bootstrap"
)

const (
	// marketRiskTailProbability defines the one-day loss quantile for a 99% VaR.
	marketRiskTailProbability = 0.99
	// lowerConfidencePercentile and upperConfidencePercentile delimit a 95% interval.
	lowerConfidencePercentile = 0.025
	upperConfidencePercentile = 0.975
)

// ExampleSampleIndices_portfolioTailRisk estimates sampling uncertainty around
// 99% historical VaR and Expected Shortfall from a clustered-return portfolio.
//
// Scenario:
//   - A risk engine monitors 252 daily returns with calm and stressed regimes.
//   - Stationary blocks retain local clusters while allowing geometric run lengths.
//   - The business layer gathers returns and computes VaR/ES for each replicate;
//     the bootstrap package itself returns only the observation indices.
//
// Implementation:
//   - Stage 1: Construct fixed illustrative daily returns and stress episodes.
//   - Stage 2: Draw 400 stationary resamples of the same 252-day sample length.
//   - Stage 3: Gather losses and calculate nearest-rank VaR plus a weighted
//     1% upper-tail ES outside bootstrap, then empirical percentile bands.
//   - Stage 4: Translate historical tail loss into an explicit capital-at-risk
//     decision, then report a replay-invariant ensemble quality check.
//
// Determinism:
//   - A fixed seed and logical replicate streams permit exact replay.
//
// Complexity:
//   - Sampling O(R*n) expected; sorting losses per replicate O(R*n*log n).
//
// Behavior highlights:
//   - The empirical ES estimator includes a fractional boundary observation,
//     so exactly 1% of the equally weighted historical distribution contributes.
//   - The reported historical risk figures come from unchanged source history;
//     the bootstrap percentile band's values depend on the pinned RNG revision.
//
// AI-Hints:
//   - With 252 trading days a 1% tail contains only 2.52 effective observations;
//     the example is a sensitivity audit, not an approved trading risk model.
//   - Calibrate block length from training history and record the RNG identity.
//   - The loss-reserve decision is deterministic illustrative business logic,
//     not a regulatory capital mandate or backtested VaR coverage claim.
func ExampleSampleIndices_portfolioTailRisk() {
	// A fixed trading-year pattern with contiguous volatility clusters is
	// deliberately synthetic: its units are fraction-of-capital daily returns.
	base := []float64{
		0.004, -0.003, 0.002, 0.001, -0.005, 0.006, -0.004,
		0.003, -0.002, 0.005, -0.006, 0.001, -0.003, 0.002,
	}
	returns := make([]float64, 252)
	for day := range returns {
		returns[day] = base[day%len(base)]
		// The episodes represent clustered shocks rather than independent spikes.
		switch {
		case day >= 44 && day < 57:
			returns[day] -= 0.021
		case day >= 126 && day < 135:
			returns[day] -= 0.034
		case day >= 208 && day < 217:
			returns[day] -= 0.016
		}
	}

	const replicates = 400
	result, err := bootstrap.SampleStationaryIndices(
		context.Background(), len(returns), len(returns), 8, replicates, 0,
	)
	if err != nil {
		fmt.Println("sampling error:", err)
		return
	}

	varEstimates := make([]float64, 0, result.Completed)
	esEstimates := make([]float64, 0, result.Completed)
	for _, sampledIndices := range result.Indices {
		losses := make([]float64, len(sampledIndices))
		for k, originalRow := range sampledIndices {
			losses[k] = -returns[originalRow]
		}
		sort.Float64s(losses)
		varEstimates = append(varEstimates, sampleQuantile(losses, marketRiskTailProbability))
		esEstimates = append(esEstimates, fractionalTailExpectedShortfall(losses, marketRiskTailProbability))
	}

	// Use unchanged history for the point estimates. Index sampling is not a
	// data-cleaning stage: the caller, not bootstrap, owns all financial values.
	originalLosses := make([]float64, len(returns))
	for i, dailyReturn := range returns {
		originalLosses[i] = -dailyReturn
	}
	sort.Float64s(originalLosses)
	historicalVaR := sampleQuantile(originalLosses, marketRiskTailProbability)
	historicalES := fractionalTailExpectedShortfall(originalLosses, marketRiskTailProbability)
	varLower, varUpper := empiricalPercentileBand(varEstimates)
	esLower, esUpper := empiricalPercentileBand(esEstimates)

	// Translate the historical tail estimate into a capacity decision using a
	// declared illustrative portfolio scale and an independent reserve policy.
	// An index sampler never chooses a risk limit or validates capital adequacy.
	const portfolioCapitalUSD = 50_000_000.0
	const maximumTailReserveUSD = 1_750_000.0
	tailReserveUSD := historicalES * portfolioCapitalUSD
	reserveReviewNeeded := tailReserveUSD > maximumTailReserveUSD
	fmt.Printf("historical VaR99=%.2f%% ES99=%.2f%%; tail days=%.2f\n",
		100*historicalVaR, 100*historicalES, float64(len(returns))*(1-marketRiskTailProbability))
	fmt.Printf("capital reserve=$%.2fM; policy review=%t\n",
		tailReserveUSD/1_000_000, reserveReviewNeeded)
	fmt.Printf("bootstrap bands ordered: %t; completed: %d\n",
		varLower <= varUpper && esLower <= esUpper, result.Completed)
	// Output:
	// historical VaR99=3.70% ES99=3.82%; tail days=2.52
	// capital reserve=$1.91M; policy review=true
	// bootstrap bands ordered: true; completed: 400
}

// ExampleSampleIndices_serialRegression resamples synchronized feature/response
// pairs to examine the variability of a regression slope under serial effects.
//
// Scenario:
//   - Daily grid demand follows a load curve and persistent forecast errors.
//   - Circular blocks preserve short-term dependence without breaking (X,y) pairs.
//   - A downstream OLS slope is re-estimated on each resampled pair sequence.
//
// Implementation:
//   - Stage 1: Construct 84 aligned demand and price observations.
//   - Stage 2: Draw 256 fixed-length circular-block index replicates.
//   - Stage 3: Fit the complete-history reference slope and each resampled slope.
//   - Stage 4: Calculate a stress uplift and compare it with operational reserve,
//     then check the fitted replicate band without claiming a HAC correction.
//
// Complexity:
//   - Sampling and scalar per-replicate regression O(R*n); band sorting O(R log R).
//
// AI-Hints:
//   - Use each index for both response and predictor values, never separate draws.
//   - Block bootstrap is not itself the Newey-West covariance estimator.
func ExampleSampleIndices_serialRegression() {
	const days = 84
	price, consumption := make([]float64, days), make([]float64, days)
	for day := 0; day < days; day++ {
		// Seasonal and trend components make each contiguous block identifiable.
		price[day] = 33 + float64(day)*0.17 + float64(day%7)*1.1
		serialError := 0.23*float64((day/7)%5) - 0.43*float64(day%3)
		consumption[day] = 110 + 1.8*price[day] + serialError
	}

	result, err := bootstrap.SampleCircularIndices(
		context.Background(), days, days, 7, 256, 2026,
	)
	if err != nil {
		fmt.Println("sampling error:", err)
		return
	}
	// Preserve each original (price, consumption) pair; independently sampling
	// columns would change the regression objective and destroy serial structure.
	allRows := make([]int, days)
	for i := range allRows {
		allRows[i] = i
	}
	originalSlope, ok := pairwiseSlope(price, consumption, allRows)
	if !ok {
		fmt.Println("invalid original regression design")
		return
	}
	betas := make([]float64, 0, result.Completed)
	for _, indices := range result.Indices {
		slope, valid := pairwiseSlope(price, consumption, indices)
		if !valid {
			fmt.Println("singular resampled regression design")
			return
		}
		betas = append(betas, slope)
	}
	lower, upper := empiricalPercentileBand(betas)
	// For a hypothetical five-unit price stress, the fitted slope projects
	// extra consumption; an eight-megawatt reserve is an independent policy.
	const priceShock = 5.0
	const demandReserveMW = 8.0
	stressUpliftMW := priceShock * originalSlope
	fmt.Printf("historical slope=%.3f MW/price-unit, training pairs=%d\n",
		originalSlope, days)
	fmt.Printf("stress uplift=%.2f MW; reserve review=%t\n",
		stressUpliftMW, stressUpliftMW > demandReserveMW)
	fmt.Printf("paired bootstrap band finite: %t; complete models=%d\n",
		!math.IsNaN(lower) && !math.IsInf(upper, 0) && lower <= upper, result.Completed)
	// Output:
	// historical slope=1.803 MW/price-unit, training pairs=84
	// stress uplift=9.02 MW; reserve review=true
	// paired bootstrap band finite: true; complete models=256
}

// ExampleSampleIndices_turbineThreshold calibrates a vibration threshold from
// moving blocks of historical telemetry, retaining localized stress events.
//
// Scenario:
//   - A turbine generates 72 hourly vibration measurements in g units.
//   - Operators estimate a high percentile threshold from block-resampled data.
//   - An extreme held-out event can trigger an alert without retraining on it.
//
// Implementation:
//   - Stage 1: Construct a training-only vibration trace with burst episodes.
//   - Stage 2: Draw 180 nonwrapping 6-hour blocks per resample.
//   - Stage 3: Estimate each 98th-percentile threshold and upper uncertainty bound.
//   - Stage 4: Apply the training-only threshold to heldout measurements.
//
// Complexity:
//   - Sampling O(R*n); quantiles O(R*n*log n); upper band O(R log R).
//
// AI-Hints:
//   - Never include future alert outcomes in the calibration trace.
//   - A bootstrap percentile bound does not replace mechanical sensor validation.
func ExampleSampleIndices_turbineThreshold() {
	vibration := make([]float64, 72)
	for hour := range vibration {
		vibration[hour] = 0.28 + 0.025*float64(hour%8)
		if hour >= 19 && hour < 25 {
			vibration[hour] += 0.44
		}
		if hour >= 49 && hour < 55 {
			vibration[hour] += 0.52
		}
	}

	result, err := bootstrap.SampleMovingIndices(
		context.Background(), len(vibration), len(vibration), 6, 180, 73,
	)
	if err != nil {
		fmt.Println("sampling error:", err)
		return
	}
	thresholds := make([]float64, 0, result.Completed)
	for _, indices := range result.Indices {
		trace := make([]float64, len(indices))
		for k, sourceIndex := range indices {
			trace[k] = vibration[sourceIndex]
		}
		sort.Float64s(trace)
		thresholds = append(thresholds, sampleQuantile(trace, 0.98))
	}
	_, upper := empiricalPercentileBand(thresholds)
	baseline := append([]float64(nil), vibration...)
	sort.Float64s(baseline)
	trainingThreshold := sampleQuantile(baseline, 0.98)
	const normalHeldout, criticalHeldout = 0.22, 2.40
	fmt.Printf("training p98=%.3f g; moving block=%d hours\n", trainingThreshold, 6)
	fmt.Printf("heldout nominal clear=%t; fault trip=%t; calibrated=%d\n",
		normalHeldout < upper, criticalHeldout > upper, result.Completed)
	// Output:
	// training p98=0.925 g; moving block=6 hours
	// heldout nominal clear=true; fault trip=true; calibrated=180
}

// ExampleSampleIndices_regulatoryReplay demonstrates exact prefix stability
// when an overnight audit expands a reproducible scenario set from 2 to 5.
//
// Implementation:
//   - Stage 1: Construct one persisted Plan with an explicit zero seed.
//   - Stage 2: Generate two replicates, then extend the plan to five.
//   - Stage 3: Verify the first two indices are unchanged and record RNG identity.
//
// Complexity:
//   - Expected Time O(R*L) for generation; equality check O(2*L).
//
// AI-Hints:
//   - A new Seed for the larger run destroys exact prefix replay.
//   - Keep RNG/AlgorithmID with stored audit metadata.
func ExampleSampleIndices_regulatoryReplay() {
	plan := bootstrap.Plan{Kind: bootstrap.StationaryBlocks, Length: 96, BlockLength: 9, Replicates: 2, Seed: 0}
	first, err := bootstrap.SampleIndices(context.Background(), 64, plan)
	if err != nil {
		fmt.Println("sampling error:", err)
		return
	}
	plan.Replicates = 5
	extended, err := bootstrap.SampleIndices(context.Background(), 64, plan)
	if err != nil {
		fmt.Println("sampling error:", err)
		return
	}
	preserved := first.RNG == extended.RNG
	for row := range first.Indices {
		for col, index := range first.Indices[row] {
			if index != extended.Indices[row][col] {
				preserved = false
			}
		}
	}
	fmt.Printf("seed=%d; retained=%d; extended=%d; identical=%t\n",
		plan.Seed, first.Completed, extended.Completed, preserved)
	// Output: seed=0; retained=2; extended=5; identical=true
}

// ExampleSampleIndices_resourceAdmission protects a risk batch from a huge
// requested result while retaining context-error precedence on canceled calls.
//
// Implementation:
//   - Stage 1: Submit a representable but over-budget output plan.
//   - Stage 2: Inspect ErrBudget without allocating the requested matrix.
//   - Stage 3: Show that an already canceled call reports cancellation first.
//
// Complexity:
//   - Time O(1), Space O(1) for both rejection paths.
//
// AI-Hints:
//   - Partial results are valid only after computation began, not on admission errors.
//   - Do not retry a rejected plan by silently changing BlockLength.
func ExampleSampleIndices_resourceAdmission() {
	plan := bootstrap.Plan{Kind: bootstrap.CircularBlocks, Length: 1 << 26, BlockLength: 5, Replicates: 3, Seed: 0}
	result, err := bootstrap.SampleIndices(context.Background(), 30, plan)
	budgetRejected := result == nil && errors.Is(err, bootstrap.ErrBudget)

	canceled, stop := context.WithCancel(context.Background())
	stop()
	result, err = bootstrap.SampleIndices(canceled, 0, bootstrap.Plan{Kind: 255})
	contextWins := result == nil && errors.Is(err, context.Canceled)
	fmt.Printf("max logical output=%d MiB; huge plan rejected=%t\n",
		bootstrap.MaxResultBytes/(1024*1024), budgetRejected)
	fmt.Printf("canceled request takes precedence=%t; result unpublished=%t\n",
		contextWins, result == nil)
	// Output:
	// max logical output=256 MiB; huge plan rejected=true
	// canceled request takes precedence=true; result unpublished=true
}

// sampleQuantile returns the nearest-rank quantile of a sorted empirical sample.
// It uses the ceiling of q*n as the one-based rank, then clamps the resulting
// position into the available slice for finite nonempty inputs.
//
// Implementation:
//   - Stage 1: Convert the probability to the nearest-rank integer position.
//   - Stage 2: Guard both ends of the sorted observation domain.
//   - Stage 3: Return the original order statistic without interpolation.
//
// Inputs:
//   - sorted: finite, nonempty ascending observations, owned by the caller.
//   - quantile: probability in [0,1], fixed by the calling example.
//
// Returns:
//   - float64: exact selected observation value, in original physical units.
//
// Determinism:
//   - Sorting is performed by the caller; this helper never mutates the slice.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - This nearest-rank convention is not linear interpolation or a claim
//     that an empirical 1% tail with 252 points has many observations.
func sampleQuantile(sorted []float64, quantile float64) float64 {
	index := int(math.Ceil(quantile*float64(len(sorted)))) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

// empiricalPercentileBand returns the empirical 95% percentile interval.
// Replicate estimates are copied first, so sorting never alters caller-owned
// ensembles or the deterministic index matrix.
//
// Implementation:
//   - Stage 1: Copy the nonempty, finite replicate statistics.
//   - Stage 2: Sort ascending and select nearest ranks 2.5% and 97.5%.
//   - Stage 3: Publish interval endpoints, without performing any bootstrap draws.
//
// Inputs:
//   - estimates: nonempty, finite per-replicate scalar estimates.
//
// Returns:
//   - float64, float64: ascending lower/upper endpoints in estimate units.
//
// Determinism:
//   - IEEE finite comparison under stable sorting of scalar value copies.
//
// Complexity:
//   - Time O(R log R), Space O(R), where R=len(estimates).
//
// AI-Hints:
//   - A percentile interval does not automatically achieve 95% coverage;
//     distributional assumptions and the block-length choice matter.
func empiricalPercentileBand(estimates []float64) (float64, float64) {
	sorted := append([]float64(nil), estimates...)
	sort.Float64s(sorted)
	return sampleQuantile(sorted, lowerConfidencePercentile),
		sampleQuantile(sorted, upperConfidencePercentile)
}

// fractionalTailExpectedShortfall estimates the upper-tail mean of losses.
// It counts exactly (1-q)*n observations' worth of probability mass by
// including full largest losses and a weighted boundary observation.
//
// Implementation:
//   - Stage 1: Compute tail mass and its integral and fractional parts.
//   - Stage 2: Accumulate all complete upper-tail observations, then the
//     exact proportional boundary contribution.
//   - Stage 3: Divide by the total upper-tail mass to express mean loss.
//
// Inputs:
//   - sorted: nonempty ascending finite losses, in fractions of capital.
//   - q: tail cutoff in (0,1), fixed and validated by the example design.
//
// Returns:
//   - float64: empirical upper-tail loss average in the same units.
//
// Determinism:
//   - Largest to smaller order; no random state or mutation is involved.
//
// Complexity:
//   - Time O((1-q)*n+1), Space O(1).
//
// AI-Hints:
//   - Do not simply average observations >= nearest-rank VaR: ties and a
//     fractional tail count change the intended Expected Shortfall definition.
func fractionalTailExpectedShortfall(sorted []float64, q float64) float64 {
	tailMass := (1 - q) * float64(len(sorted))
	fullCount := int(math.Floor(tailMass))
	boundaryWeight := tailMass - float64(fullCount)
	sum := 0.0
	for i := 0; i < fullCount; i++ {
		sum += sorted[len(sorted)-1-i]
	}
	if boundaryWeight > 0 {
		sum += boundaryWeight * sorted[len(sorted)-1-fullCount]
	}
	return sum / tailMass
}

// pairwiseSlope computes an intercept-enabled scalar OLS coefficient.
// Its only input selection is an index row applied equally to x and y;
// bootstrap itself never receives or fits these numeric observations.
//
// Implementation:
//   - Stage 1: Accumulate x/y means over the selected ordered pairs.
//   - Stage 2: Accumulate centered x/y covariance and x variance.
//   - Stage 3: Reject zero/nonfinite variance and publish a finite slope.
//
// Inputs:
//   - x: finite explanatory variables; y: aligned finite response values.
//   - indices: nonempty positions valid in both input slices.
//
// Returns:
//   - float64: coefficient in response units per predictor unit.
//   - bool: false for zero/nonfinite variance or unrepresentable slope.
//
// Determinism:
//   - Index order is preserved through both ascending summation passes.
//
// Complexity:
//   - Time O(len(indices)), Space O(1).
//
// AI-Hints:
//   - Pairwise index alignment is mandatory. This scalar educational helper
//     is not a numerical replacement for the production pivoted QR OLS model.
func pairwiseSlope(x, y []float64, indices []int) (float64, bool) {
	meanX, meanY := 0.0, 0.0
	for _, source := range indices {
		meanX += x[source]
		meanY += y[source]
	}
	meanX /= float64(len(indices))
	meanY /= float64(len(indices))

	covariance, variance := 0.0, 0.0
	for _, source := range indices {
		dx := x[source] - meanX
		covariance += dx * (y[source] - meanY)
		variance += dx * dx
	}
	if variance <= 0 || math.IsNaN(variance) || math.IsInf(variance, 0) {
		return 0, false
	}
	slope := covariance / variance
	return slope, !math.IsNaN(slope) && !math.IsInf(slope, 0)
}
