// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm_test demonstrates auditable hidden-state modeling through the
// public API. All observations are deterministic synthetic engineering fixtures;
// the scenarios demonstrate contracts, not pretrained real-world classifiers.
package hmm_test

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/lvlath/go/hmm"
	"github.com/lvlath/go/matrix"
)

// exampleMatrix turns a rectangular, finite static scenario table into an
// owning Matrix. It checks each allocation and write rather than concealing
// malformed fixture data behind a nil pointer or a panic.
// Implementation:
//   - Stage 1: Reject missing columns, missing rows and ragged input.
//   - Stage 2: Allocate the matrix and copy every value in row-major order.
//   - Stage 3: Publish the matrix only after all Set calls have succeeded.
//
// Complexity: O(rows*cols) time and owned space.
// AI-Hints: Use a separate training-only scaling stage for heterogeneous units.
func exampleMatrix(rows [][]float64) (matrix.Matrix, error) {
	if len(rows) == 0 || len(rows[0]) == 0 {
		return nil, fmt.Errorf("example matrix requires nonempty rows and columns")
	}
	width := len(rows[0])
	for row := range rows {
		if len(rows[row]) != width {
			return nil, fmt.Errorf("example matrix has ragged row %d", row)
		}
	}
	m, err := matrix.NewZeros(len(rows), width)
	if err != nil {
		return nil, err
	}
	for row, values := range rows {
		for col, value := range values {
			if err = m.Set(row, col, value); err != nil {
				return nil, fmt.Errorf("matrix fixture (%d,%d): %w", row, col, err)
			}
		}
	}
	return m, nil
}

// exampleCategoricalModel assembles one deterministic categorical HMM from
// explicit stochastic rows, leaving constructor policy and validation to hmm.
// Complexity: O(K^2+K*A) time and memory for matrix preparation and model copies.
// AI-Hints: State indices encode model regimes; symbol IDs encode observed events.
func exampleCategoricalModel(prior []float64, transitions, emissions [][]float64) (*hmm.CategoricalModel, error) {
	a, err := exampleMatrix(transitions)
	if err != nil {
		return nil, err
	}
	b, err := exampleMatrix(emissions)
	if err != nil {
		return nil, err
	}
	return hmm.NewCategorical(hmm.CategoricalSpec{
		Common:   hmm.Common{Prior: prior, Transition: a},
		Emission: b,
	})
}

// exampleGaussianModel creates a diagonal-Gaussian HMM with two independent
// matrix construction steps and an immutable public constructor boundary.
// Complexity: O(K^2+K*D) preparation time and owned model space.
// AI-Hints: Each variance belongs to the square of the corresponding feature unit.
func exampleGaussianModel(prior []float64, transitions, means, variances [][]float64) (*hmm.GaussianModel, error) {
	a, err := exampleMatrix(transitions)
	if err != nil {
		return nil, err
	}
	mu, err := exampleMatrix(means)
	if err != nil {
		return nil, err
	}
	variance, err := exampleMatrix(variances)
	if err != nil {
		return nil, err
	}
	return hmm.NewGaussian(hmm.GaussianSpec{
		Common:    hmm.Common{Prior: prior, Transition: a},
		Means:     mu,
		Variances: variance,
	})
}

// ExampleCategoricalModel_Filter demonstrates a real-time e-commerce intent
// engine consuming an ordered, anonymized clickstream. The event taxonomy is
// stable: 0=content view, 1=specification/price comparison, 2=checkout action.
//
// Scenario:
//   - An application observes research, repeated comparisons and checkout intent.
//   - Three hidden states describe discovery, evaluation, and transaction intent.
//   - The service decides whether to offer a checkout-assistance flow using
//     the posterior available *at that event*, not a retrospective smoother.
//
// Process:
//   - Install an explicit fixed-alphabet probability model.
//   - Read both an early, lower-confidence prefix and the final causal posterior.
//   - Report probability movement, final evidence and the policy threshold.
//
// Determinism: chronological event order and K-state transition order are fixed.
// Complexity: O(T*K^2) time and O(T*K) retained history.
// AI-Hints: The threshold is illustrative; production calibration needs separate holdout data.
func ExampleCategoricalModel_Filter() {
	model, err := exampleCategoricalModel(
		[]float64{.75, .20, .05},
		[][]float64{{.85, .14, .01}, {.10, .76, .14}, {.03, .17, .80}},
		[][]float64{{.83, .15, .02}, {.20, .65, .15}, {.04, .16, .80}},
	)
	if err != nil {
		fmt.Println("model:", err)
		return
	}
	// The synthetic journey moves from catalog exploration to an interrupted
	// checkout; event numbers are stable identifiers, not monetary amounts.
	const eventComparison = uint32(1)
	const eventCheckout = uint32(2)
	const highIntentThreshold = .65
	events := []uint32{0, 0, 0, eventComparison, 0, eventComparison, eventComparison, 0, eventComparison, eventCheckout, eventComparison, eventCheckout, eventCheckout}
	result, err := model.FilterWithHistory(context.Background(), events)
	if err != nil {
		fmt.Println("filter:", err)
		return
	}
	beforeCheckout, err := result.Posterior.At(8, 2)
	if err != nil {
		fmt.Println("early posterior:", err)
		return
	}
	afterCheckout, err := result.Posterior.At(len(events)-1, 2)
	if err != nil {
		fmt.Println("last posterior:", err)
		return
	}
	fmt.Printf("events=%d purchase-before=%.3f purchase-after=%.3f assist=%t\n",
		result.Last.Steps, beforeCheckout, afterCheckout, afterCheckout >= highIntentThreshold)
	// Output:
	// events=13 purchase-before=0.036 purchase-after=0.901 assist=true
}

// ExampleGaussianModel_FilterStep demonstrates chronological market-risk
// surveillance using *daily* return and realized-volatility-proxy features.
// These synthetic inputs resemble a volatility shock but are not market data.
//
// Scenario:
//   - Regime 0 is a stable, low-volatility return process.
//   - Regime 1 is a drawdown/high-volatility process.
//   - The monitoring decision is made after each daily close, never from a
//     future-conditioned smoother or from hindsight labels.
//
// Process:
//   - Freeze physical feature units and an explicit two-regime model.
//   - Feed the day-by-day measurements through FilterStep(ctx,prev,x).
//   - Count stress observations and record the first event crossing a fixed threshold.
//
// Determinism: each call has exactly one new observation and one immutable checkpoint.
// Complexity: O(T*(K^2+K*D)) time, O(K) live filter memory.
// AI-Hints: Use separately calibrated thresholds and keep the same scaling/model ID in production.
func ExampleGaussianModel_FilterStep() {
	model, err := exampleGaussianModel(
		[]float64{.96, .04},
		[][]float64{{.975, .025}, {.14, .86}},
		[][]float64{{.0015, .012}, {-.030, .060}},
		[][]float64{{.00009, .000025}, {.0008, .0004}},
	)
	if err != nil {
		fmt.Println("model:", err)
		return
	}
	// Each row is an already closed day: [log return, realized volatility proxy].
	measurements := [][]float64{
		{.004, .013}, {.001, .012}, {-.003, .015}, {.002, .011},
		{-.009, .019}, {-.023, .045}, {-.058, .079}, {-.041, .071},
		{-.015, .052}, {.002, .037}, {-.004, .021}, {.003, .015},
	}
	const stressThreshold = .80
	firstStressDay := -1
	stressDays := 0
	var checkpoint *hmm.FilterState
	for day, sample := range measurements {
		next, stepErr := model.FilterStep(context.Background(), checkpoint, sample)
		if stepErr != nil {
			fmt.Println("filter:", stepErr)
			return
		}
		checkpoint = next
		stressProbability := math.Exp(checkpoint.LogPosterior[1])
		if stressProbability >= stressThreshold {
			stressDays++
			if firstStressDay == -1 {
				firstStressDay = day + 1 // Display 1-based day count, not a time index.
			}
		}
	}
	fmt.Printf("days=%d first-stress-day=%d stress-days=%d final-stress=%.3f\n",
		checkpoint.Steps, firstStressDay, stressDays, math.Exp(checkpoint.LogPosterior[1]))
	// Output:
	// days=12 first-stress-day=6 stress-days=5 final-stress=0.001
}

// ExampleCategoricalModel_Smooth explains retrospective incident attribution
// after evidence collected during and after a plant safety investigation.
//
// Scenario:
//   - Initially a normal sensor event supports an apparently healthy machine.
//   - Subsequent alarms change the posterior for the *same earlier time*.
//   - Operations must preserve the original filter output in an incident log;
//     revised smoothing is an explanatory audit artifact, not a live signal.
//
// Process:
//   - Run the causal history and the full-sequence smoother on identical data.
//   - Compare both at time zero and verify their evidence is consistent.
//   - Publish the changed probability rather than overwriting the historical belief.
//
// Determinism: the absorbing transition model isolates the look-ahead effect.
// Complexity: O(T*K^2) time; smoothing O(T*K) message memory.
// AI-Hints: Never rewrite past trading/maintenance decisions using smoothed posteriors.
func ExampleCategoricalModel_Smooth() {
	model, err := exampleCategoricalModel(
		[]float64{.5, .5},
		[][]float64{{1, 0}, {0, 1}},
		[][]float64{{.9, .1}, {.1, .9}},
	)
	if err != nil {
		fmt.Println("model:", err)
		return
	}
	// An early normal indicator is followed by escalating corroborating alarms.
	observations := []uint32{0, 1, 1, 1, 1, 1}
	live, err := model.FilterWithHistory(context.Background(), observations)
	if err != nil {
		fmt.Println("filter:", err)
		return
	}
	audit, err := model.Smooth(context.Background(), observations)
	if err != nil {
		fmt.Println("smooth:", err)
		return
	}
	initialNormal, err := live.Posterior.At(0, 0)
	if err != nil {
		fmt.Println("live posterior:", err)
		return
	}
	reviewedNormal, err := audit.Posterior.At(0, 0)
	if err != nil {
		fmt.Println("audit posterior:", err)
		return
	}
	fmt.Printf("live-normal=%.3f retrospective-normal=%.5f evidence-agrees=%t\n",
		initialNormal, reviewedNormal, math.Abs(live.LogLikelihood-audit.LogLikelihood) < 1e-12)
	// Output:
	// live-normal=0.900 retrospective-normal=0.00015 evidence-agrees=true
}

// ExampleGaussianModel_Decode reconstructs a feasible production-cycle phase
// sequence from two physical sensor channels: motor current (A) and vibration
// velocity (mm/s RMS), treated as independent conditionally on each phase.
//
// Scenario:
//   - States 0/1/2 represent idle, warm-up, and full production load.
//   - A deterministic trajectory passes through startup, stable processing,
//     and cooling with noisy intermediate sensor values.
//   - The Viterbi path helps annotate maintenance logs; it cannot replace a
//     safety interlock, and it is not an argmax of individual smoothed rows.
//
// Process:
//   - Build explicitly ordered state and feature dictionaries.
//   - Decode the complete finite matrix into one maximum-joint-probability path.
//   - Count phase transitions and expose whether the inferred run ever reached load.
//
// Determinism: exact ties choose the lowest predecessor then final-state index.
// Complexity: O(T*(K^2+K*D)) time, O(T*K+T*D) memory.
// AI-Hints: Validate feature units and calendar/sample gaps outside the HMM.
func ExampleGaussianModel_Decode() {
	model, err := exampleGaussianModel(
		[]float64{.98, .02, 0},
		[][]float64{{.82, .18, 0}, {.05, .80, .15}, {0, .12, .88}},
		[][]float64{{1.2, .30}, {8.5, 1.1}, {18, 3.8}},
		[][]float64{{.36, .09}, {2.25, .25}, {4, 1}},
	)
	if err != nil {
		fmt.Println("model:", err)
		return
	}
	observations, err := exampleMatrix([][]float64{
		{1.3, .30}, {1.1, .28}, {1.6, .33}, {2.2, .40},
		{6.9, .88}, {8.1, 1.03}, {9.6, 1.21}, {10.0, 1.27},
		{15.4, 2.90}, {17.8, 3.50}, {18.7, 4.20}, {19.2, 3.90},
		{17.1, 3.60}, {9.1, 1.12}, {2.1, .44},
	})
	if err != nil {
		fmt.Println("observations:", err)
		return
	}
	decoded, err := model.Decode(context.Background(), observations)
	if err != nil {
		fmt.Println("decode:", err)
		return
	}
	phaseSwitches := 0
	reachedProduction := false
	for i, state := range decoded.States {
		if state == 2 {
			reachedProduction = true
		}
		if i > 0 && state != decoded.States[i-1] {
			phaseSwitches++
		}
	}
	fmt.Printf("samples=%d switches=%d reached-production=%t path=%v\n",
		len(decoded.States), phaseSwitches, reachedProduction, decoded.States)
	// Output:
	// samples=15 switches=4 reached-production=true path=[0 0 0 0 1 1 1 1 2 2 2 2 2 1 0]
}

// ExampleFitCategorical learns event-intent parameters from independent
// anonymized web sessions instead of creating invented cross-visitor transitions.
//
// Scenario:
//   - Each inner event sequence begins at a new visitor's session start.
//   - Event codes are 0=browse, 1=compare, 2=checkout; regimes are discovery,
//     consideration and transaction-oriented behavior.
//   - A caller supplies an explicit initial model and a Dirichlet penalty to
//     control zero-count stochastic rows without hidden random initialization.
//
// Process:
//   - Fit the deterministic MAP-EM objective over separate sessions.
//   - Inspect objective history, expected state visits, and convergence status.
//   - Do not treat ErrNotConverged as successful deployment of the partial model.
//
// Determinism: sequence/time/state accumulation order is fixed.
// Complexity: O(iterations*sum(T_s)*K^2); peak private pair table O(max(T_s)*K^2).
// AI-Hints: Preserve independent session boundaries in training and evaluation.
func ExampleFitCategorical() {
	initial := hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{.65, .30, .05}}}
	var err error
	initial.Common.Transition, err = exampleMatrix([][]float64{{.77, .20, .03}, {.13, .72, .15}, {.06, .18, .76}})
	if err != nil {
		fmt.Println("transition:", err)
		return
	}
	initial.Emission, err = exampleMatrix([][]float64{{.82, .16, .02}, {.17, .70, .13}, {.04, .17, .79}})
	if err != nil {
		fmt.Println("emission:", err)
		return
	}
	sessions := [][]uint32{
		{0, 0, 1, 2}, {0, 1, 1, 2}, {0, 0, 0, 1, 1, 2},
		{1, 1, 2}, {0, 1, 2, 2}, {0, 0, 1, 0, 1},
		{0, 1, 1, 1, 2}, {0, 0, 1, 2}, {1, 1, 2, 2},
	}
	const priorPseudocount = .25
	fit, err := hmm.FitCategorical(context.Background(), sessions, initial, hmm.FitOptions{
		MaxIterations: 60, Tolerance: 1e-5, Pseudocount: priorPseudocount,
	})
	if err != nil && !errors.Is(err, hmm.ErrNotConverged) {
		fmt.Println("fit:", err)
		return
	}
	if fit == nil || fit.Model == nil {
		fmt.Println("no accepted model")
		return
	}
	objective := fit.Diagnostics.Objective
	var visits float64
	for _, occupancy := range fit.Diagnostics.Occupancy {
		visits += occupancy
	}
	fmt.Printf("sessions=%d observations=%d occupancy=%.0f steps=%d monotone=%t converged=%t\n",
		len(sessions), totalExampleSymbols(sessions), visits,
		fit.Diagnostics.Iterations, objective[len(objective)-1] >= objective[0], fit.Diagnostics.Converged)
	// Output:
	// sessions=9 observations=39 occupancy=39 steps=18 monotone=true converged=true
}

// totalExampleSymbols counts only supplied events across independent sessions.
// It does not add any synthetic event at a session boundary.
// Complexity: O(total observations) time, O(1) space.
func totalExampleSymbols(sessions [][]uint32) int {
	total := 0
	for _, session := range sessions {
		total += len(session)
	}
	return total
}

// ExampleFitGaussian trains a two-regime thermal monitor from independent
// cooling-system operation cycles, then checks an out-of-sample hot reading.
// Both features are temperatures in Celsius, so the single scalar VarianceFloor
// has coherent squared-degree units for both coordinates.
//
// Scenario:
//   - One regime has moderate housing/exhaust temperatures; another is hot.
//   - Each machine cycle is a separate T_s-by-2 observation sequence.
//   - Only a converged model is admitted to the prospective filter call.
//
// Process:
//   - Supply a prevalidated positive-variance initial specification.
//   - Fit independent cycles with a nonzero prior/transition Dirichlet penalty.
//   - Inspect convergence and feed one heldout high-temperature observation.
//
// Determinism: sequence order and time/state/feature loops are fixed.
// Complexity: O(iterations*sum(T_s)*(K^2+K*D)) plus feature input copies.
// AI-Hints: Never call FilterStep on a failed fit without explicitly accepting its partial-model policy.
func ExampleFitGaussian() {
	transition, err := exampleMatrix([][]float64{{.91, .09}, {.18, .82}})
	if err != nil {
		fmt.Println("transition:", err)
		return
	}
	means, err := exampleMatrix([][]float64{{42, 45}, {76, 81}})
	if err != nil {
		fmt.Println("means:", err)
		return
	}
	variances, err := exampleMatrix([][]float64{{4, 4}, {9, 12}})
	if err != nil {
		fmt.Println("variances:", err)
		return
	}
	initial := hmm.GaussianSpec{
		Common:    hmm.Common{Prior: []float64{.75, .25}, Transition: transition},
		Means:     means,
		Variances: variances,
	}
	data := [][][]float64{
		{{41, 44}, {42, 45}, {43, 46}, {43, 45}, {41, 44}, {42, 45}},
		{{42, 44}, {43, 45}, {59, 65}, {74, 78}, {76, 81}, {77, 82}, {74, 80}},
		{{41, 43}, {42, 46}, {44, 47}, {73, 79}, {78, 84}, {76, 82}, {44, 46}},
		{{43, 45}, {42, 44}, {72, 78}, {77, 83}, {75, 81}, {42, 45}},
	}
	series := make([]matrix.Matrix, 0, len(data))
	for _, rows := range data {
		m, matrixErr := exampleMatrix(rows)
		if matrixErr != nil {
			fmt.Println("training matrix:", matrixErr)
			return
		}
		series = append(series, m)
	}
	const minimumVarianceCelsiusSquared = .25
	fit, err := hmm.FitGaussian(context.Background(), series, initial, hmm.FitOptions{
		MaxIterations: 60, Tolerance: 1e-5, Pseudocount: .1, VarianceFloor: minimumVarianceCelsiusSquared,
	})
	if err != nil && !errors.Is(err, hmm.ErrNotConverged) {
		fmt.Println("fit:", err)
		return
	}
	if fit == nil || fit.Model == nil || !fit.Diagnostics.Converged {
		fmt.Println("training did not converge; no alert model published")
		return
	}
	hot, err := fit.Model.FilterStep(context.Background(), nil, []float64{79, 83})
	if err != nil {
		fmt.Println("heldout filter:", err)
		return
	}
	fmt.Printf("cycles=%d converged=%t hot-regime=%.3f model-id-length=%d\n",
		len(series), fit.Diagnostics.Converged, math.Exp(hot.LogPosterior[1]), len(fit.Model.ModelID()))
	// Output:
	// cycles=4 converged=true hot-regime=1.000 model-id-length=64
}
