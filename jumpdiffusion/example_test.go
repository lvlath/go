// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion_test demonstrates executable risk and systems workflows.
// Examples use model parameters, reproducible streams, and explicit failure policy.
package jumpdiffusion_test

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/jumpdiffusion"
	"github.com/lvlath/go/montecarlo"
	"github.com/lvlath/go/rng"
)

// ExampleSimulate_flashCrashCollateral demonstrates an auditable crypto collateral stress replay.
//
// Scenario:
//   - A treasury holds an asset marked at 60,000 USD with a 25% drawdown limit.
//   - A 30-calendar-day stress exercise injects one -2-sigma negative log jump
//     on day eight into an otherwise deterministically isolated jump process.
//   - The exercise measures the worst peak-to-trough loss, not crash probability.
//
// Process:
//   - Construct daily elapsed model times without assuming 252 trading sessions.
//   - Use an explicit scripted stress fixture so the very rare event is reproducible.
//   - Generate the canonical trajectory and compute a risk-control drawdown.
//
// Notes:
//   - The scripted stream represents a deliberately selected stress path;
//     it must never be used to estimate the probability of a flash crash.
//   - Setting diffusion volatility to zero isolates the jump shock for attribution.
func ExampleSimulate_flashCrashCollateral() {
	const (
		initialMark   = 60_000.0
		drawdownLimit = 0.25
		calendarDays  = 365.0
		stressDays    = 30
		shockDay      = 8
	)

	model := jumpdiffusion.Parameters{
		Diffusion: gbm.Parameters{
			Initial: initialMark, Drift: 0.02, Volatility: 0, Unit: gbm.TimeYears,
		},
		Intensity: 2, LogJumpMean: -0.20, LogJumpStdDev: 0.10,
		Convention: jumpdiffusion.Compensated,
	}

	times := make([]float64, stressDays+1)
	uniforms := make([]float64, 0, stressDays+1)
	for day := 1; day <= stressDays; day++ {
		times[day] = float64(day) / calendarDays
		if day == shockDay {
			// Near-one uniform forces one event; the next uniform terminates
			// inversion. The jump Normal is a preselected -2-sigma stress.
			uniforms = append(uniforms, 0.999, 0.1)
		} else {
			uniforms = append(uniforms, 0.1)
		}
	}
	stream := &scriptedRNG{uniforms: uniforms, normals: []float64{-2}}

	path, err := jumpdiffusion.SimulatePath(context.Background(), model, times, stream)
	if err != nil || path == nil || !path.Complete {
		fmt.Println("stress simulation failed:", err)
		return
	}

	peak, trough, worstDrawdown := path.Values[0], path.Values[0], 0.0
	for _, price := range path.Values {
		peak = math.Max(peak, price)
		trough = math.Min(trough, price)
		worstDrawdown = math.Max(worstDrawdown, 1-price/peak)
	}
	fmt.Printf("stress days=%d trough=$%.0f max_drawdown=%.1f%% collateral_alarm=%t\n",
		len(path.Values)-1, trough, 100*worstDrawdown, worstDrawdown >= drawdownLimit)
	// Output: stress days=30 trough=$40551 max_drawdown=32.9% collateral_alarm=true
}

// discountedOptionSampler is a complete-horizon montecarlo.Sampler adapter.
// The caller provides one independent stream per pathID; no stream is retained.
type discountedOptionSampler struct {
	model    jumpdiffusion.Parameters
	strike   float64
	horizon  float64
	discount float64
}

// Dimension exposes call, put, and discounted terminal asset in that fixed order.
// Three components allow option pricing and martingale diagnostics in one run.
//
// Inputs: none; Returns: 3; Errors: none; Determinism: constant coordinate order.
// Complexity: O(1) time/space; AI-Hints: never reorder coordinates after sampling.
func (discountedOptionSampler) Dimension() int {
	return 3
}

// Sample converts a complete Merton terminal path into discounted payoffs.
// An interrupted path is rejected rather than silently mixed into a full trial.
//
// Implementation:
//   - Stage 1: simulate one terminal interval with the supplied path stream.
//   - Stage 2: reject errors/incomplete results before accessing the terminal price.
//   - Stage 3: publish call, put, and asset values in documented coordinate order.
//
// Inputs: ctx and path-owned RNG come from montecarlo.Run; pathID selects the stream.
// Returns: fresh finite vector of length 3; no mutable state is retained.
// Errors: preserves Simulate failure, or returns an adapter-invariant error.
// Determinism: same per-path stream order as SimulateTerminal; complexity O(B).
// Notes: positive payoffs are not themselves claims of market calibration.
// AI-Hints: never use a partial Result.Values[last] as a terminal payoff.
func (s discountedOptionSampler) Sample(ctx context.Context, _ uint64, random rng.RNG) ([]float64, error) {
	result, err := jumpdiffusion.SimulateTerminal(ctx, s.model, s.horizon, random)
	if err != nil {
		return nil, err
	}
	if result == nil || !result.Complete {
		return nil, errors.New("option sampler: incomplete terminal path")
	}

	terminal := result.Values[len(result.Values)-1]
	return []float64{
		s.discount * math.Max(terminal-s.strike, 0),
		s.discount * math.Max(s.strike-terminal, 0),
		s.discount * terminal,
	}, nil
}

// ExampleSimulateTerminal_optionPricing compares Merton and jump-free option estimates.
//
// Scenario:
//   - A European call and put share spot=100, strike=105, and a nine-month horizon.
//   - The risk-neutral Merton compensator preserves the expected discounted stock.
//   - Independent seeded experiments expose baseline and jump-sensitive prices.
//
// Process:
//   - Construct terminal-only samplers for compensated jumps and lambda=0 GBM.
//   - Run 512 independent paths per model through montecarlo.Run.
//   - Consume means and mean standard errors, not individual partial trajectories.
//
// Notes:
//   - The comparison describes model risk, not an observed market option premium.
//   - A proper difference confidence interval requires experiment-level uncertainty.
func ExampleSimulateTerminal_optionPricing() {
	const (
		spot    = 100.0
		strike  = 105.0
		rate    = 0.035
		horizon = 0.75
		paths   = uint64(512)
	)

	model := jumpdiffusion.Parameters{
		Diffusion: gbm.Parameters{
			Initial: spot, Drift: rate, Volatility: 0.22, Unit: gbm.TimeYears,
		},
		Intensity: 0.8, LogJumpMean: -0.12, LogJumpStdDev: 0.17,
		Convention: jumpdiffusion.Compensated,
	}
	option := discountedOptionSampler{
		model: model, strike: strike, horizon: horizon, discount: math.Exp(-rate * horizon),
	}
	jumps, jumpErr := montecarlo.Run(context.Background(),
		montecarlo.Plan{Paths: paths, Seed: 2026, Workers: 2}, option)

	option.model.Intensity = 0
	baseline, gbmErr := montecarlo.Run(context.Background(),
		montecarlo.Plan{Paths: paths, Seed: 2027, Workers: 2}, option)
	if jumpErr != nil || gbmErr != nil || jumps == nil || baseline == nil ||
		!jumps.Complete || !baseline.Complete {
		fmt.Println("option comparison failed:", jumpErr, gbmErr)
		return
	}

	// On every path, call-put equals discounted (terminal-strike). Therefore
	// the aggregated portfolio must satisfy put-call parity independently of
	// whether the process has jumps. This is a useful sampler integration audit.
	jumpParity := math.Abs(jumps.Mean[0]-jumps.Mean[1]-
		(jumps.Mean[2]-strike*option.discount)) < 1e-10
	baselineParity := math.Abs(baseline.Mean[0]-baseline.Mean[1]-
		(baseline.Mean[2]-strike*option.discount)) < 1e-10

	// The model gap is a monetary contrast; its standard error is needed to
	// distinguish simulation noise from a potential jump-model pricing effect.
	priceDifference := jumps.Mean[0] - baseline.Mean[0]
	combinedSE := math.Hypot(jumps.MeanStandardError[0], baseline.MeanStandardError[0])
	uncertaintyValid := combinedSE > 0 && !math.IsInf(combinedSE, 0) &&
		!math.IsNaN(priceDifference) && !math.IsInf(priceDifference, 0)
	fmt.Printf("option_paths=%d put_call_parity=%t gap_uncertainty_valid=%t both_calls_positive=%t\n",
		jumps.Completed, jumpParity && baselineParity,
		uncertaintyValid, jumps.Mean[0] > 0 && baseline.Mean[0] > 0)
	// Output: option_paths=512 put_call_parity=true gap_uncertainty_valid=true both_calls_positive=true
}

// ExampleSimulate_highIntensityCapacity estimates overload exposure for an event-ingestion queue.
//
// Scenario:
//   - A telemetry gateway can drain 40 events per second, while a modeled feed
//     produces an average 50 independent events per second.
//   - A ten-minute synthetic replay estimates cumulative service deficit and
//     the share of individual seconds in which arrivals exceed capacity.
//   - The jump-diffusion price field is used only as a synthetic count index:
//     a fixed log jump permits event counts to be recovered from successive marks.
//
// Process:
//   - Build 600 one-second intervals, each forcing the PTRS Poisson branch.
//   - Run a genuine seeded stream rather than manufacturing 50 arrivals per step.
//   - Decode step counts and compare realized arrivals with the service budget.
//
// Notes:
//   - This is a queue-capacity stress illustration, not a financial asset model.
//   - Real arrivals may be clustered/self-exciting; a Poisson model assumes
//     independent increments and must be validated before production sizing.
func ExampleSimulate_highIntensityCapacity() {
	const (
		seconds          = 600
		arrivalRate      = 50.0
		servicePerSec    = 40
		logIndexPerEvent = 0.001
	)
	model := jumpdiffusion.Parameters{
		Diffusion: gbm.Parameters{Initial: 100, Unit: gbm.TimeSeconds},
		Intensity: arrivalRate, LogJumpMean: logIndexPerEvent,
		LogJumpStdDev: 0, Convention: jumpdiffusion.Uncompensated,
	}
	times := make([]float64, seconds+1)
	for instant := 1; instant <= seconds; instant++ {
		times[instant] = float64(instant)
	}
	path, err := jumpdiffusion.SimulatePath(
		context.Background(), model, times, rng.NewStream(2026, 113),
	)
	if err != nil || path == nil || !path.Complete {
		fmt.Println("capacity replay failed:", err)
		return
	}

	// The deterministic per-event log multiplier makes each interval count
	// recoverable; no Gaussian approximation or exported private kernel is used.
	totalArrivals, overloadedSeconds := 0, 0
	for instant := 1; instant < len(path.Values); instant++ {
		logIncrement := math.Log(path.Values[instant] / path.Values[instant-1])
		arrivals := int(math.Round(logIncrement / logIndexPerEvent))
		totalArrivals += arrivals
		if arrivals > servicePerSec {
			overloadedSeconds++
		}
	}
	budget := seconds * servicePerSec
	expected := int(arrivalRate) * seconds
	fmt.Printf("queue seconds=%d expected=%d service_budget=%d realized_over_budget=%t majority_seconds_overloaded=%t\n",
		seconds, expected, budget, totalArrivals > budget, overloadedSeconds > seconds/2)
	// Output: queue seconds=600 expected=30000 service_budget=24000 realized_over_budget=true majority_seconds_overloaded=true
}

// ExampleSimulateSeeded_scenarioReplay demonstrates pathID-based audit replay.
//
// Scenario:
//   - A risk audit must reproduce three independent irregular-grid simulations
//     without relying on worker scheduling, a system clock, or global RNG state.
//   - All scenario IDs are logical identifiers, not goroutine or node numbers.
//
// Process:
//   - Run two fresh streams for each of three path IDs using SimulateSeeded.
//   - Compare exact floating-point bits at every observation and price.
//   - Count only fully complete, bit-for-bit reproducible paths as audited.
func ExampleSimulateSeeded_scenarioReplay() {
	model := jumpdiffusion.Parameters{
		Diffusion: gbm.Parameters{
			Initial: 90_000, Drift: 0.04, Volatility: 0.30, Unit: gbm.TimeYears,
		},
		Intensity: 1.8, LogJumpMean: -0.13, LogJumpStdDev: 0.09,
		Convention: jumpdiffusion.Compensated,
	}
	grid := gbm.TimeGrid{
		Values: []float64{0, 0.03, 0.07, 0.10, 0.13, 0.20, 0.27,
			0.32, 0.40, 0.48, 0.55, 0.67, 0.81, 1},
		Unit: gbm.TimeYears,
	}

	replayed := 0
	for _, pathID := range []uint64{3, 11, 29} {
		first, err1 := jumpdiffusion.SimulateSeeded(context.Background(), model, grid, 2026, pathID)
		second, err2 := jumpdiffusion.SimulateSeeded(context.Background(), model, grid, 2026, pathID)
		if err1 != nil || err2 != nil || first == nil || second == nil || !first.Complete || !second.Complete {
			fmt.Println("replay failed:", err1, err2)
			return
		}
		identical := len(first.Values) == len(second.Values)
		for index := range first.Values {
			identical = identical &&
				math.Float64bits(first.Times[index]) == math.Float64bits(second.Times[index]) &&
				math.Float64bits(first.Values[index]) == math.Float64bits(second.Values[index])
		}
		if identical {
			replayed++
		}
	}
	fmt.Printf("audit trajectories=%d bit_replayed=%d observations=%d\n",
		3, replayed, len(grid.Values))
	// Output: audit trajectories=3 bit_replayed=3 observations=14
}

// ExampleSimulatePath_creditEventReserve estimates reserve requirements under credit gaps.
//
// Scenario:
//   - A credit desk carries 2.5M USD exposure backed by a mark-index of 100.
//   - A 30-day scenario allows only downward credit-event jumps and negative drift.
//   - A deterministic drift-only loss is a lower bound for the required reserve.
//
// Process:
//   - Simulate a realistically irregular daily observation schedule in TimeDays.
//   - Translate the lowest valid mark into an exposure-based loss reserve.
//   - Compare the reserve with the analytic drift-only lower bound.
//
// Notes:
//   - This is an explicitly uncompensated physical stress assumption.
//   - The guarantee relies on nonpositive drift and nonpositive jump magnitudes.
func ExampleSimulatePath_creditEventReserve() {
	const exposureUSD = 2_500_000.0
	model := jumpdiffusion.Parameters{
		Diffusion: gbm.Parameters{
			Initial: 100, Drift: -0.0005, Volatility: 0, Unit: gbm.TimeDays,
		},
		Intensity: 0.08, LogJumpMean: -0.035, LogJumpStdDev: 0,
		Convention: jumpdiffusion.Uncompensated,
	}
	times := []float64{0, 1, 2, 3, 4, 5, 7, 10, 11, 12, 15, 18, 20, 22, 25, 30}
	path, err := jumpdiffusion.SimulatePath(context.Background(), model, times, rng.NewStream(2026, 91))
	if err != nil || path == nil || !path.Complete {
		fmt.Println("credit reserve simulation failed:", err)
		return
	}

	lowest := path.Values[0]
	for _, mark := range path.Values[1:] {
		lowest = math.Min(lowest, mark)
	}
	reserveUSD := exposureUSD * (1 - lowest/model.Diffusion.Initial)
	floorUSD := exposureUSD * (1 - math.Exp(model.Diffusion.Drift*times[len(times)-1]))
	fmt.Printf("credit marks=%d drift_floor=$%.0f reserve_positive=%t reserve_above_floor=%t\n",
		len(path.Values), floorUSD, reserveUSD > 0, reserveUSD+1e-8 >= floorUSD)
	// Output: credit marks=16 drift_floor=$37220 reserve_positive=true reserve_above_floor=true
}
