// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng_test demonstrates reproducible production-shaped stochastic workflows.
// The observations are explicit synthetic fixtures, not claims about live operations.
package rng_test

import (
	"fmt"
	"math"
	"sync"

	"github.com/lvlath/go/rng"
)

// ExampleFactory_Stream simulates a fulfillment network's hourly demand spikes.
//
// Scenario:
//   - Eight distribution zones have different picking capacities and demand multipliers.
//   - A 24-hour demand curve contains daytime traffic and evening dispatch peaks.
//   - Lognormal demand shocks allow overload risk to be replayed for incident analysis.
//
// Process:
//   - Build explicit baseline loads and zone capacities in orders per hour.
//   - Allocate a separate logical zone stream, then draw one shock per hour.
//   - Re-run the same batch and compare the entire [zone][hour] demand ledger.
//
// Interpretation:
//   - The overload count is a synthetic stress fixture, not a measured forecast.
//   - The exact ledger can be retained with the RNG version for postmortem replay.
//
// Complexity:
//   - Time O(zones*hours), Space O(zones*hours) for the returned ledger.
//
// AI-Hints:
//   - Do not derive zone path IDs from runtime worker slots or a shared generator.
func ExampleFactory_Stream() {
	const seed uint64 = 20261114
	const zones = 8
	baseline := [24]float64{
		75, 68, 62, 60, 65, 82, 120, 170,
		215, 248, 265, 280, 300, 305, 298, 320,
		355, 390, 420, 405, 360, 285, 195, 120,
	}
	capacities := [zones]int{245, 280, 310, 335, 365, 390, 420, 460}
	zoneFactor := [zones]float64{0.72, 0.86, 0.95, 1, 1.05, 1.12, 1.2, 1.28}
	factory := rng.NewFactory()

	// Every run owns fresh streams; each logical zone owns exactly one path ID.
	simulate := func() ([zones][24]int, int) {
		var demand [zones][24]int
		overloaded := 0
		for zone := uint64(0); zone < zones; zone++ {
			stream := factory.Stream(seed, zone)
			for hour := range baseline {
				shock := math.Exp(0.13 * stream.Normal())
				orders := int(math.Round(baseline[hour] * zoneFactor[zone] * shock))
				demand[zone][hour] = orders
				if orders > capacities[zone] {
					overloaded++
				}
			}
		}
		return demand, overloaded
	}

	first, overloads := simulate()
	second, repeatedOverloads := simulate()
	fmt.Printf("zones=%d hours=%d hourly-replay=%t overload-count-replay=%t\n",
		zones, len(baseline), first == second, overloads == repeatedOverloads)
	fmt.Println("capacity-stress-detected:", overloads > 0)
	// Output:
	// zones=8 hours=24 hourly-replay=true overload-count-replay=true
	// capacity-stress-detected: true
}

// ExampleFactory_parallelPaths prices a dividend-adjusted European call with GBM.
//
// Scenario:
//   - A quantitative risk service prices a European option by simulated terminal payoffs.
//   - Portfolio runs use different worker counts depending on machine availability.
//   - The same logical path must produce the same terminal value under either schedule.
//
// Process:
//   - Compute each logical payoff in an index-addressed output slot.
//   - Run one-worker and eight-worker schedules using identical seed/path IDs.
//   - Reduce in ascending path index and compare against the Black-Scholes sanity value.
//
// Interpretation:
//   - The closed-form comparison is a wide numerical smoke bound, not a CI guarantee.
//   - The input parameters are synthetic and do not represent a tradable quote.
//
// Complexity:
//   - Time O(paths) total, Space O(paths) for the stable payoff ledger.
//
// AI-Hints:
//   - Worker-local sums would give schedule-sensitive floating-point reductions.
func ExampleFactory_parallelPaths() {
	const paths = 16384
	const spot, strike = 100.0, 105.0
	const rate, dividend, volatility, expiry = 0.04, 0.01, 0.23, 0.75
	const seed uint64 = 20261114

	factory := rng.NewFactory()
	payoff := func(path uint64) float64 {
		stream := factory.Stream(seed, path)
		drift := (rate - dividend - 0.5*volatility*volatility) * expiry
		terminal := spot * math.Exp(drift+volatility*math.Sqrt(expiry)*stream.Normal())
		return math.Max(terminal-strike, 0)
	}

	// Each goroutine publishes independent indices; the reduction is single-order.
	price := func(workers uint64) float64 {
		payoffs := make([]float64, paths)
		var group sync.WaitGroup
		for worker := uint64(0); worker < workers; worker++ {
			group.Add(1)
			go func(workerID uint64) {
				defer group.Done()
				for path := workerID; path < paths; path += workers {
					payoffs[path] = payoff(path)
				}
			}(worker)
		}
		group.Wait()
		var sum float64
		for _, result := range payoffs {
			sum += result
		}
		return math.Exp(-rate*expiry) * sum / float64(paths)
	}

	// The GBM/Black-Scholes formula is an independent financial-model sanity check.
	sigmaT := volatility * math.Sqrt(expiry)
	d1 := (math.Log(spot/strike) + (rate-dividend+0.5*volatility*volatility)*expiry) / sigmaT
	d2 := d1 - sigmaT
	cdf := func(z float64) float64 { return 0.5 * (1 + math.Erf(z/math.Sqrt2)) }
	analytic := spot*math.Exp(-dividend*expiry)*cdf(d1) - strike*math.Exp(-rate*expiry)*cdf(d2)

	serial := price(1)
	parallel := price(8)
	fmt.Println("parallel-price-bit-replay:", math.Float64bits(serial) == math.Float64bits(parallel))
	fmt.Println("gbm-analytic-sanity:", math.Abs(serial-analytic) < 0.75)
	// Output:
	// parallel-price-bit-replay: true
	// gbm-analytic-sanity: true
}

// ExampleFactory_blockBootstrap generates reproducible circular latency bursts.
//
// Scenario:
//   - Thirty-two quarter-hour p95 service-latency measurements include an outage burst.
//   - Individual-point shuffling would erase within-block temporal dependence.
//   - A caller gathers consecutive cyclic four-observation blocks per replication.
//
// Process:
//   - Fix block starts with one bounded raw draw per block.
//   - Use a 32-point source so masking introduces no bias for uniform raw words.
//   - Calculate the share of resamples exceeding an operational latency threshold.
//
// Interpretation:
//   - This is an educational fixed-size resampler, not a confidence-interval
//     procedure and not a replacement for bootstrap.SampleIndices.
//   - For non-power-of-two source lengths, use a vetted rejection sampler.
//
// Complexity:
//   - Time O(replicates*sourceLength), Space O(1) extra per replication.
//
// AI-Hints:
//   - Never generalize raw % n or bit masking to arbitrary n: modulo bias results.
func ExampleFactory_blockBootstrap() {
	const replicates = 512
	const sourceLength = 32
	const blockLength = 4
	const latencySLA = 250.0 // milliseconds
	latencyMs := [sourceLength]float64{
		72, 77, 81, 89, 84, 80, 76, 73,
		79, 86, 92, 101, 107, 114, 108, 98,
		95, 102, 130, 315, 420, 398, 290, 175,
		110, 101, 96, 88, 83, 78, 75, 73,
	}

	factory := rng.NewFactory()
	drawScenario := func(replicate int) (int, float64) {
		stream := factory.Stream(2048, uint64(replicate))
		breaches := 0
		total := 0.0
		for block := 0; block < sourceLength/blockLength; block++ {
			// For uniform raw words, sourceLength=2^5 makes all starts equiprobable.
			start := stream.Uint64() & (sourceLength - 1)
			for offset := uint64(0); offset < blockLength; offset++ {
				observed := latencyMs[(start+offset)%sourceLength]
				total += observed
				if observed > latencySLA {
					breaches++
				}
			}
		}
		return breaches, total / sourceLength
	}

	// The ensemble is indexed by replicate ID, never by the worker evaluating it.
	stressCount := 0
	replay := true
	for replicate := 0; replicate < replicates; replicate++ {
		breaches, mean := drawScenario(replicate)
		breachesAgain, meanAgain := drawScenario(replicate)
		replay = replay && breaches == breachesAgain && mean == meanAgain
		if breaches >= 8 {
			stressCount++
		}
	}
	fmt.Printf("replicates=%d replay=%t meaningful-stress-paths=%t\n",
		replicates, replay, stressCount > 0 && stressCount < replicates)
	// Output:
	// replicates=512 replay=true meaningful-stress-paths=true
}

// ExampleFactory_sensorNoise injects replayable measurement and packet-loss noise.
//
// Scenario:
//   - A temperature controller tracks a warming industrial motor every 100 ms.
//   - Field readings include normal sensor error and sporadic missing packets.
//   - A scalar predict/update observer must replay the same faults post-incident.
//
// Process:
//   - Draw one Gaussian reading perturbation and one raw packet code per step.
//   - Update a causal one-dimensional Kalman-style variance/mean on received data.
//   - Re-run the exact logical device path and compare error statistics.
//
// Interpretation:
//   - This demonstrates stochastic fault injection, not a calibrated control system.
//   - The missing-packet classifier uses a byte threshold, with an exact 256-code domain.
//
// Complexity:
//   - Time O(samples), Space O(1) for this online observer.
//
// AI-Hints:
//   - Normal -> Uint64 -> Normal preserves the pair's pending Gaussian sine value.
func ExampleFactory_sensorNoise() {
	const steps = 1000
	const measurementVariance = 0.16 // degrees Celsius squared
	const processVariance = 0.005    // degrees Celsius squared
	const temperatureDrift = 0.003   // degrees Celsius per sample

	factory := rng.NewFactory()
	simulate := func(pathID uint64) (float64, int) {
		stream := factory.Stream(1907, pathID)
		estimate, variance := 38.0, 1.0
		squaredError := 0.0
		missingPackets := 0
		for sample := 0; sample < steps; sample++ {
			truth := 38.0 + temperatureDrift*float64(sample+1)
			measurement := truth + math.Sqrt(measurementVariance)*stream.Normal()
			packetCode := stream.Uint64() & 0xff

			// Prediction uses the known slow warming trend, not a future reading.
			estimate += temperatureDrift
			variance += processVariance
			if packetCode < 8 {
				missingPackets++
			} else {
				gain := variance / (variance + measurementVariance)
				estimate += gain * (measurement - estimate)
				variance *= 1 - gain
			}
			delta := estimate - truth
			squaredError += delta * delta
		}
		return math.Sqrt(squaredError / steps), missingPackets
	}

	firstError, firstMissing := simulate(19)
	replayError, replayMissing := simulate(19)
	fmt.Println("fault-sequence-replay:", firstError == replayError && firstMissing == replayMissing)
	fmt.Println("sensor-filter-rmse-reasonable:", firstError < 0.30)
	fmt.Println("sporadic-dropouts:", firstMissing > 0 && firstMissing < 100)
	// Output:
	// fault-sequence-replay: true
	// sensor-filter-rmse-reasonable: true
	// sporadic-dropouts: true
}

// ExampleFactory_normalCache audits an interleaved risk/shock event sequence.
//
// Scenario:
//   - A power-grid stress model samples wind deviations in correlated pairs.
//   - Between deviations, it examines a raw event code to choose a maintenance flag.
//   - A fragile implementation would clear the pending Gaussian on that raw draw.
//
// Process:
//   - Reproduce the same stream twice: interleaved and analytically reconstructed.
//   - Verify the maintenance code and both Box-Muller pair members.
//   - Check that the next raw word has the same position after cache use.
//
// Complexity:
//   - Time O(1), Space O(1); no extra stream synchronization is required.
//
// AI-Hints:
//   - Never reset Box-Muller spare on raw draws, uniform draws, or event decisions.
func ExampleFactory_normalCache() {
	factory := rng.NewFactory()
	stream := factory.Stream(0, 99)
	firstShock := stream.Normal()
	maintenanceCode := stream.Uint64()
	secondShock := stream.Normal() // Cached: this must consume ZERO extra raw words.
	afterPair := stream.Uint64()

	// The oracle builds both normals from two explicit raw projections,
	// without calling production Normal or observing private cache fields.
	oracle := factory.Stream(0, 99)
	u1 := oracle.UniformOpen()
	u2 := oracle.UniformOpen()
	radius := math.Sqrt(-2 * math.Log(u1))
	angle := 2 * math.Pi * u2
	oracleFirst := radius * math.Cos(angle)
	oracleSecond := radius * math.Sin(angle)
	oracleCode := oracle.Uint64()
	oracleAfterPair := oracle.Uint64()

	fmt.Println("gaussian-pair-replay:", firstShock == oracleFirst && secondShock == oracleSecond)
	fmt.Println("event-word-position-stable:", maintenanceCode == oracleCode && afterPair == oracleAfterPair)
	// Output:
	// gaussian-pair-replay: true
	// event-word-position-stable: true
}

// ExampleFactory_FillNormal illustrates owned multi-channel sensor-noise buffers.
//
// Scenario:
//   - A monitoring service precomputes 256 normalized disturbances for six channels.
//   - Channel IDs are stable logical paths; fresh simulations replay in the same order.
//   - Batch output is caller-owned and can be scored without mutating the RNG factory.
//
// Process:
//   - Preallocate each channel's history and call FillNormal once per channel.
//   - Compare one batch to the canonical consecutive stream draws.
//   - Report a zero-allocation hot path by keeping all storage with the caller.
//
// Complexity:
//   - Time O(channels*samples), Space O(channels*samples) for caller buffers.
//
// AI-Hints:
//   - A second FillNormal call starts over; use an owned Stream for continuation.
func ExampleFactory_FillNormal() {
	const channels = 6
	const samples = 256
	factory := rng.NewFactory()
	data := make([][]float64, channels)
	matched := true
	for channel := 0; channel < channels; channel++ {
		data[channel] = make([]float64, samples)
		factory.FillNormal(4002, uint64(channel), data[channel])
		stream := factory.Stream(4002, uint64(channel))
		for _, observed := range data[channel] {
			want := stream.Normal()
			matched = matched && math.Float64bits(observed) == math.Float64bits(want)
		}
	}
	fmt.Printf("channels=%d samples=%d matches-stream=%t\n", channels, samples, matched)
	// Output:
	// channels=6 samples=256 matches-stream=true
}
