// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm_test verifies EM error publication, dead-state preservation, and objective diagnostics.
// The tests exercise actual statistical and ownership laws using the exported fitting facades.
package hmm_test

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/hmm"
	"github.com/lvlath/go/matrix"
)

// TestFitCategorical_ImpossibleIndependentSequence rejects absent support during the E-step.
// Both sequences use in-range symbols, so validation succeeds before the model encounters a zero-probability observation.
//
// Implementation:
//   - Stage 1: construct a one-state model emitting only symbol zero.
//   - Stage 2: verify that the first training sequence is supported and the second is impossible.
//   - Stage 3: require a nil fit, the structural-impossibility sentinel, and unchanged initialization.
//
// Behavior highlights:
//   - A zero emission probability is not an out-of-range alphabet index.
//
// Inputs:
//   - t: active test; two separate two-observation sequences.
//
// Returns:
//   - None; the impossible initial E-step must not publish partial counts.
//
// Errors:
//   - ErrImpossibleSequence, not ErrShape or ErrOverflow.
//
// Determinism:
//   - One state, deterministic transition, fixed zero-support symbol.
//
// Complexity:
//   - Time O(T), Space O(T) for K=1 and the fixed training collection.
//
// Notes:
//   - Positive pseudocount does not repair an impossible initial E-step before responsibilities exist.
//
// AI-Hints:
//   - Exercise model support with valid symbol indices; an invalid index would never enter Forward-Backward.
func TestFitCategorical_ImpossibleIndependentSequence(t *testing.T) {
	ctx := context.Background()
	spec := singleStateCategoricalFixture(t, []float64{1, 0})
	model, err := hmm.NewCategorical(spec)
	if err != nil {
		t.Fatal(err)
	}
	sequences := [][]uint32{{0, 0}, {0, 1}}
	first, err := model.Smooth(ctx, sequences[0])
	if err != nil || first == nil || first.LogLikelihood != 0 {
		t.Fatalf("supported sequence: result=%v err=%v", first, err)
	}
	second, err := model.Smooth(ctx, sequences[1])
	requireHMMFailure(t, second, err, hmm.ErrImpossibleSequence)
	fit, err := hmm.FitCategorical(ctx, sequences, spec, hmm.FitOptions{Pseudocount: .5})
	requireHMMFailure(t, fit, err, hmm.ErrImpossibleSequence)
	if !slices.Equal(sequences[0], []uint32{0, 0}) || !slices.Equal(sequences[1], []uint32{0, 1}) {
		t.Fatal("training observations were mutated")
	}
	requireUnchangedMatrix(t, spec.Emission, [][]float64{{1, 0}})
}

// TestFitGaussian_DeadStateAndZeroFeaturePreservation verifies constrained moments and unchanged dead-state parameters.
// Two independent singleton runs observe a constant-zero feature and a second feature with population variance one.
//
// Implementation:
//   - Stage 1: initialize an absorbing live state and a structurally unvisited second state.
//   - Stage 2: fit one M-step with a positive variance floor and no pseudocount.
//   - Stage 3: compare the full canonical model identity with independently derived exact parameters.
//
// Behavior highlights:
//   - Zero feature scale must not divide by zero.
//   - The dead-state emission and transition row are preserved; singleton runs create no transitions.
//
// Inputs:
//   - t: active test; two features in fixed physical units, two independent singletons.
//
// Returns:
//   - None; the accepted model, occupancy, objective, and observed likelihood are verified.
//
// Errors:
//   - ErrNotConverged because one improving M-step exhausts the explicitly limited budget.
//
// Determinism:
//   - The prior and transitions make responsibilities exactly [1,0] for every observation.
//
// Complexity:
//   - O(T*(K*K+K*D)) time, O(T*K+K*D+K*K) space for singleton sequences.
//
// Notes:
//   - All expected means and variances are exactly representable in binary64, so identity comparison is meaningful.
//
// AI-Hints:
//   - Do not infer dead-state preservation only from a live-state prediction; compare the canonical full model identity.
//   - Use population variance in the M-step, not sample variance.
func TestFitGaussian_DeadStateAndZeroFeaturePreservation(t *testing.T) {
	ctx := context.Background()
	common := hmm.Common{Prior: []float64{1, 0}, Transition: dense(t, [][]float64{{1, 0}, {0, 1}})}
	initial := hmm.GaussianSpec{
		Common: common,
		Means:  dense(t, [][]float64{{0, 0}, {50, 20}}), Variances: dense(t, [][]float64{{1, 4}, {4, 4}}),
	}
	observations := []matrix.Matrix{dense(t, [][]float64{{0, 0}}), dense(t, [][]float64{{0, 2}})}
	const floor = .25
	fit, err := hmm.FitGaussian(ctx, observations, initial, hmm.FitOptions{
		MaxIterations: 1, Tolerance: 1e-10, VarianceFloor: floor,
	})
	if !errors.Is(err, hmm.ErrNotConverged) || fit == nil || fit.Model == nil {
		t.Fatalf("fit=%v err=%v, want one accepted nonconverged iteration", fit, err)
	}
	diagnostics := fit.Diagnostics
	if diagnostics.Converged || diagnostics.Iterations != 1 || len(diagnostics.Objective) != 2 ||
		!slices.Equal(diagnostics.Occupancy, []float64{2, 0}) {
		t.Fatalf("unexpected diagnostics: %+v", diagnostics)
	}
	if !finite(diagnostics.Objective[0]) || !finite(diagnostics.Objective[1]) || diagnostics.Objective[1] <= diagnostics.Objective[0] {
		t.Fatalf("expected a finite improvement: %v", diagnostics.Objective)
	}

	// Independent closed-form M-step: live mean=[0,1], population variance=[0,1].
	// Projection raises only the zero component; the dead state's entire row remains intact.
	expected, err := hmm.NewGaussian(hmm.GaussianSpec{
		Common: common,
		Means:  dense(t, [][]float64{{0, 1}, {50, 20}}), Variances: dense(t, [][]float64{{floor, 1}, {4, 4}}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if fit.Model.ModelID() != expected.ModelID() {
		t.Fatalf("fitted identity=%s want=%s", fit.Model.ModelID(), expected.ModelID())
	}
	state, err := fit.Model.FilterStep(ctx, nil, []float64{0, 1})
	if err != nil || state == nil {
		t.Fatalf("fitted prediction: state=%v err=%v", state, err)
	}
	// At the two-dimensional mean, log density is -log(2*pi)-log(sqrt(floor)).
	approx(t, state.LogLikelihood, -math.Log(2*math.Pi)-.5*math.Log(floor))
	// Each training row is one standard deviation from the second-feature mean.
	approx(t, diagnostics.Objective[1], 2*state.LogLikelihood-1)
	requireUnchangedMatrix(t, initial.Means, [][]float64{{0, 0}, {50, 20}})
	requireUnchangedMatrix(t, initial.Variances, [][]float64{{1, 4}, {4, 4}})
	requireUnchangedMatrix(t, observations[0], [][]float64{{0, 0}})
	requireUnchangedMatrix(t, observations[1], [][]float64{{0, 2}})
}

// TestEM_UnrepresentablePenalizedObjectiveRollback separates valid probability updates from an overflowing MAP objective.
// A finite, very large pseudocount may make the scalar diagnostic unrepresentable even for a valid stationary simplex.
//
// Implementation:
//   - Stage 1: construct stationary symmetric categorical and Gaussian parameter sets.
//   - Stage 2: request an explicitly finite pseudocount of MaxFloat64 and evaluate the fit.
//   - Stage 3: require the numerical rollback result with the initial model, no accepted iterations, and no false convergence.
//
// Behavior highlights:
//   - The simplex itself remains valid; the error belongs to objective representability.
//
// Inputs:
//   - t: active test; symmetric initial parameters and finite training data.
//
// Returns:
//   - None; both families must keep the last valid model on the numerical rejection.
//
// Errors:
//   - ErrNumerical under the current nonfinite-objective rollback policy.
//
// Determinism:
//   - Initial and candidate simplexes are symmetric, and the mathematical penalty exceeds the float64 range.
//
// Complexity:
//   - Constant work for the fixed small fixtures, with the normal fit workspace.
//
// Notes:
//   - The initial objective is not a convergence certificate if its value is -Inf.
//
// AI-Hints:
//   - Do not reject a valid finite Pseudocount as ErrInvalidOption merely because this objective overflows.
//   - Do not conflate objective overflow with an invalid probability row or impossible observations.
func TestEM_UnrepresentablePenalizedObjectiveRollback(t *testing.T) {
	ctx := context.Background()
	options := hmm.FitOptions{MaxIterations: 1, Pseudocount: math.MaxFloat64}
	t.Run("categorical", func(t *testing.T) {
		// Three equal masses yield a penalty of 3*lambda*log(1/3), below -MaxFloat64.
		spec := singleStateCategoricalFixture(t, []float64{1.0 / 3, 1.0 / 3, 1.0 / 3})
		initial, err := hmm.NewCategorical(spec)
		if err != nil {
			t.Fatal(err)
		}
		fit, err := hmm.FitCategorical(ctx, [][]uint32{{0, 1, 2}}, spec, options)
		if !errors.Is(err, hmm.ErrNumerical) || fit == nil || fit.Model == nil {
			t.Fatalf("fit=%v err=%v, want last model/ErrNumerical", fit, err)
		}
		if fit.Model.ModelID() != initial.ModelID() || fit.Diagnostics.Converged || fit.Diagnostics.Iterations != 0 ||
			len(fit.Diagnostics.Objective) != 1 || !math.IsInf(fit.Diagnostics.Objective[0], -1) ||
			!slices.Equal(fit.Diagnostics.Occupancy, []float64{3}) {
			t.Fatalf("invalid rollback diagnostics: %+v", fit.Diagnostics)
		}
	})
	t.Run("gaussian", func(t *testing.T) {
		// Two equal prior masses and two uniform transition rows yield six log(1/2) penalty terms.
		spec := hmm.GaussianSpec{
			Common: hmm.Common{Prior: []float64{.5, .5}, Transition: dense(t, [][]float64{{.5, .5}, {.5, .5}})},
			Means:  dense(t, [][]float64{{0}, {0}}), Variances: dense(t, [][]float64{{1}, {1}}),
		}
		initial, err := hmm.NewGaussian(spec)
		if err != nil {
			t.Fatal(err)
		}
		fit, err := hmm.FitGaussian(ctx, []matrix.Matrix{dense(t, [][]float64{{-1}, {1}})}, spec, options)
		if !errors.Is(err, hmm.ErrNumerical) || fit == nil || fit.Model == nil {
			t.Fatalf("fit=%v err=%v, want last model/ErrNumerical", fit, err)
		}
		if fit.Model.ModelID() != initial.ModelID() || fit.Diagnostics.Converged || fit.Diagnostics.Iterations != 0 ||
			len(fit.Diagnostics.Objective) != 1 || !math.IsInf(fit.Diagnostics.Objective[0], -1) {
			t.Fatalf("invalid rollback diagnostics: %+v", fit.Diagnostics)
		}
		if len(fit.Diagnostics.Occupancy) != 2 {
			t.Fatalf("unexpected occupancy shape: %v", fit.Diagnostics.Occupancy)
		}
		for _, occupancy := range fit.Diagnostics.Occupancy {
			approx(t, occupancy, 1)
		}
	})
}

// TestFitCategorical_PositivePseudocountRevivesZero verifies the Dirichlet MAP boundary.
// A supported training symbol starts with a finite likelihood but an infinite negative prior penalty.
//
// Implementation:
//   - Stage 1: initialize emission [1,0] and observe symbol zero in an independent singleton.
//   - Stage 2: perform one explicit MAP step with lambda=1/2, without pretending the initial score is finite.
//   - Stage 3: verify both new symbol probabilities and the exact mathematical penalized objective.
//
// Behavior highlights:
//   - A structurally zero unused emission can be revived by a positive pseudocount.
//   - Moving from -Inf to a finite objective is improvement, not convergence by itself.
//
// Inputs:
//   - t: active test; one state, two symbols, one observed symbol, lambda=1/2.
//
// Returns:
//   - None; the test checks one accepted iteration and a later stationary fit.
//
// Errors:
//   - ErrNotConverged with a valid result for the one-iteration budget.
//
// Determinism:
//   - All responsibilities are one, so expected counts are exact.
//
// Complexity:
//   - Time O(A), Space O(A) for each one-state singleton iteration.
//
// Notes:
//   - The initial -Inf here is the true log-Dirichlet boundary, not arithmetic overflow.
//
// AI-Hints:
//   - Do not replace zero probabilities with epsilon before the E-step.
//   - A finite new score cannot justify declaring convergence from an infinite starting objective.
func TestFitCategorical_PositivePseudocountRevivesZero(t *testing.T) {
	ctx := context.Background()
	spec := singleStateCategoricalFixture(t, []float64{1, 0})
	sequences := [][]uint32{{0}}
	const lambda = .5
	options := hmm.FitOptions{MaxIterations: 1, Pseudocount: lambda}
	fit, err := hmm.FitCategorical(ctx, sequences, spec, options)
	if !errors.Is(err, hmm.ErrNotConverged) || fit == nil || fit.Model == nil {
		t.Fatalf("fit=%v err=%v, want one accepted step and ErrNotConverged", fit, err)
	}
	diagnostics := fit.Diagnostics
	if diagnostics.Converged || diagnostics.Iterations != 1 || len(diagnostics.Objective) != 2 ||
		!math.IsInf(diagnostics.Objective[0], -1) || !finite(diagnostics.Objective[1]) ||
		!slices.Equal(diagnostics.Occupancy, []float64{1}) {
		t.Fatalf("unexpected boundary diagnostics: %+v", diagnostics)
	}
	// Counts [1,0] plus lambda give [3/4,1/4]; neither transition nor prior contributes a penalty.
	for i, want := range []float64{.75, .25} {
		state, err := fit.Model.FilterStep(ctx, nil, uint32(i))
		if err != nil || state == nil {
			t.Fatalf("symbol %d: state=%v err=%v", i, state, err)
		}
		approx(t, math.Exp(state.LogLikelihood), want)
	}
	approx(t, diagnostics.Objective[1], math.Log(.75)+lambda*(math.Log(.75)+math.Log(.25)))
	// A second evaluation of the stationary optimum may now declare convergence.
	options.MaxIterations = 2
	converged, err := hmm.FitCategorical(ctx, sequences, spec, options)
	if err != nil || converged == nil || converged.Model == nil || !converged.Diagnostics.Converged {
		t.Fatalf("stationary fit=%v err=%v", converged, err)
	}
	if converged.Model.ModelID() != fit.Model.ModelID() {
		t.Fatal("stationary MAP solution changed after the second evaluation")
	}
}

// TestFitCategorical_StationaryMLEDoesNotPublishDecline checks the roundoff stop law.
// The exact multinomial maximum is known before running EM because K=1 and the initial emission
// is the empirical frequency vector. Re-evaluating this maximum must not publish a decreasing score.
//
// Implementation:
//   - Stage 1: construct the stationary frequency vector [1,6,4]/11 and matching symbol counts.
//   - Stage 2: run one EM iteration with the existing strict objective policy and explicit tolerance.
//   - Stage 3: verify the stationary likelihood, monotonic diagnostics, and retained model on a rounded decline.
//
// Behavior highlights:
//   - Different libm/toolchain implementations may round the same optimum differently.
//   - A zero-accepted-iteration plateau is valid only when the old model and old objective are retained.
//
// Inputs:
//   - t: active test; fixed empirical counts, no pseudocount, one hidden state.
//
// Returns:
//   - None; the test requires convergence and independent likelihood/probability agreement.
//
// Errors:
//   - None on this stationary, well-conditioned problem.
//
// Determinism:
//   - Fixed counts and symbol order, no random seeds or search in the regression test.
//
// Complexity:
//   - Time O(T+A), Space O(T+A) for one-state EM on the fixed sample.
//
// Notes:
//   - Both an exactly nondecreasing accepted step and a rejected roundoff-only candidate are admissible.
//   - The test deliberately does not assert one particular last-bit rounding across architectures.
//
// AI-Hints:
//   - Do not widen the production monotonicity tolerance to make a stationary test pass.
//   - When no candidate is accepted, verify the original ModelID and one-element objective history.
func TestFitCategorical_StationaryMLEDoesNotPublishDecline(t *testing.T) {
	ctx := context.Background()
	const total = 11
	counts := []int{1, 6, 4}
	probabilities := []float64{1.0 / total, 6.0 / total, 4.0 / total}
	symbols := []uint32{0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2}
	spec := singleStateCategoricalFixture(t, probabilities)
	initial, err := hmm.NewCategorical(spec)
	if err != nil {
		t.Fatal(err)
	}
	fit, err := hmm.FitCategorical(ctx, [][]uint32{symbols}, spec, hmm.FitOptions{MaxIterations: 1, Tolerance: 1e-12})
	if err != nil || fit == nil || fit.Model == nil || !fit.Diagnostics.Converged {
		t.Fatalf("stationary fit=%v err=%v", fit, err)
	}
	diagnostics := fit.Diagnostics
	if diagnostics.Iterations < 0 || diagnostics.Iterations > 1 || len(diagnostics.Objective) != diagnostics.Iterations+1 ||
		!slices.Equal(diagnostics.Occupancy, []float64{total}) {
		t.Fatalf("invalid stationary diagnostics: %+v", diagnostics)
	}
	if diagnostics.Iterations == 0 && fit.Model.ModelID() != initial.ModelID() {
		t.Fatal("a rejected roundoff candidate changed the model identity")
	}
	var expected float64
	for i, probability := range probabilities {
		// The independent multinomial objective is sum(count[a]*log(count[a]/T)).
		expected += float64(counts[i]) * math.Log(probability)
		state, err := fit.Model.FilterStep(ctx, nil, uint32(i))
		if err != nil || state == nil || !finite(state.LogLikelihood) || math.Abs(math.Exp(state.LogLikelihood)-probability) > 2e-15 {
			t.Fatalf("stationary symbol %d: state=%v err=%v want=%g", i, state, err, probability)
		}
	}
	for i, value := range diagnostics.Objective {
		if !finite(value) || math.Abs(value-expected) > 2e-13 {
			t.Fatalf("objective[%d]=%.17g want=%.17g", i, value, expected)
		}
		if i > 0 && value < diagnostics.Objective[i-1] {
			t.Fatalf("published objective declined: %v", diagnostics.Objective)
		}
	}
}
