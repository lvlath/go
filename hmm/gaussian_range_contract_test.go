// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm_test distinguishes representable Gaussian inputs from unrepresentable results.
// The tests use public inference and fitting methods, plus exact rational population-moment oracles.
package hmm_test

import (
	"context"
	"errors"
	"math"
	"math/big"
	"slices"
	"testing"

	"github.com/lvlath/go/hmm"
	"github.com/lvlath/go/matrix"
)

// TestGaussianInference_OverflowClassificationByStage separates three emission failure modes.
// All inputs are finite and all initial variances are positive; only a computed quantity overflows.
//
// Implementation:
//   - Stage 1: build scalar or two-feature models with deliberately extreme finite observations.
//   - Stage 2: call step, batch filtering, smoothing, decoding, and Gaussian fitting.
//   - Stage 3: require ErrOverflow, no result, and unchanged caller-owned observations.
//
// Behavior highlights:
//   - A nonrepresentable standardized residual, quadratic term, and sum of finite log terms are distinct cases.
//
// Inputs:
//   - t: active test; each subtest uses finite observations in the model's feature coordinates.
//
// Returns:
//   - None; failures are matched through errors.Is.
//
// Errors:
//   - ErrOverflow, never ErrImpossibleSequence or ErrNonFinite for the finite inputs.
//
// Determinism:
//   - Fixed one-state models and explicit values eliminate state-selection ambiguity.
//
// Complexity:
//   - O(D) time and space per operation; D is one or two.
//
// Notes:
//   - All parameter values are admissible under the default variance floor.
//
// AI-Hints:
//   - Do not test arithmetic overflow by passing an infinite observation, which fails input validation instead.
func TestGaussianInference_OverflowClassificationByStage(t *testing.T) {
	ctx := context.Background()
	cases := []struct {
		name                          string
		means, variances, observation []float64
	}{
		{"standardized_residual", []float64{0}, []float64{.25}, []float64{math.MaxFloat64}},
		{"quadratic_term", []float64{0}, []float64{1}, []float64{2e154}},
		{"sum_of_finite_log_terms", []float64{0, 0}, []float64{1, 1}, []float64{1.5e154, 1.5e154}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			spec := singleStateGaussianFixture(t, tc.means, tc.variances)
			model, err := hmm.NewGaussian(spec)
			if err != nil {
				t.Fatalf("finite model rejected: %v", err)
			}
			observations := dense(t, [][]float64{tc.observation})
			// The first row has no representable complete log emission.
			state, err := model.FilterStep(ctx, nil, tc.observation)
			if state != nil || !errors.Is(err, hmm.ErrOverflow) || errors.Is(err, hmm.ErrImpossibleSequence) {
				t.Fatalf("step: state=%v err=%v, want nil/ErrOverflow", state, err)
			}
			filtered, err := model.FilterDefault(ctx, observations)
			requireHMMFailure(t, filtered, err, hmm.ErrOverflow)
			withHistory, err := model.FilterWithHistory(ctx, observations)
			requireHMMFailure(t, withHistory, err, hmm.ErrOverflow)
			smoothed, err := model.Smooth(ctx, observations)
			requireHMMFailure(t, smoothed, err, hmm.ErrOverflow)
			decoded, err := model.Decode(ctx, observations)
			requireHMMFailure(t, decoded, err, hmm.ErrOverflow)
			fit, err := hmm.FitGaussian(ctx, []matrix.Matrix{observations}, spec, hmm.FitOptions{})
			requireHMMFailure(t, fit, err, hmm.ErrOverflow)
			requireUnchangedMatrix(t, observations, [][]float64{tc.observation})
		})
	}
}

// TestGaussianInference_AccumulatedEvidenceOverflow checks the sum of finite emission scores.
// Each individual observation is admissible and has a finite log density, but their joint evidence is not representable.
//
// Implementation:
//   - Stage 1: construct a scalar unit-variance, one-state model and obtain a valid first checkpoint.
//   - Stage 2: append a second observation of the same magnitude to the causal and retrospective calls.
//   - Stage 3: require overflow and prove the original checkpoint remains unchanged.
//
// Behavior highlights:
//   - This reaches accumulation after successful emission evaluation, unlike single-emission overflow.
//
// Inputs:
//   - t: active test; two finite values of 1.5e154 in a scalar feature.
//
// Returns:
//   - None; the first score must be finite, whereas the total must fail.
//
// Errors:
//   - ErrOverflow for the two-observation result; no impossible-sequence classification.
//
// Determinism:
//   - The hidden state is unique, and the observations have identical densities.
//
// Complexity:
//   - Time O(1), Space O(1) for the fixed two-row fixture.
//
// Notes:
//   - The exact quadratic magnitude is below MaxFloat64 per row and above it for two rows.
//
// AI-Hints:
//   - Verify finite increments before testing accumulation overflow; otherwise the earlier kernel may mask the target.
func TestGaussianInference_AccumulatedEvidenceOverflow(t *testing.T) {
	ctx := context.Background()
	spec := singleStateGaussianFixture(t, []float64{0}, []float64{1})
	model, err := hmm.NewGaussian(spec)
	if err != nil {
		t.Fatal(err)
	}
	observation := []float64{1.5e154}
	first, err := model.FilterStep(ctx, nil, observation)
	if err != nil || first == nil || !finite(first.LogLikelihood) || first.LogLikelihood >= -math.MaxFloat64/2 {
		t.Fatalf("first score must be finite and negative beyond half the range: state=%v err=%v", first, err)
	}
	before := *first
	before.LogPosterior = slices.Clone(first.LogPosterior)

	second, err := model.FilterStep(ctx, first, observation)
	if second != nil || !errors.Is(err, hmm.ErrOverflow) || errors.Is(err, hmm.ErrImpossibleSequence) {
		t.Fatalf("second step: state=%v err=%v", second, err)
	}
	if first.ModelID != before.ModelID || first.Steps != before.Steps ||
		math.Float64bits(first.LogLikelihood) != math.Float64bits(before.LogLikelihood) ||
		!slices.Equal(first.LogPosterior, before.LogPosterior) {
		t.Fatal("failed extension mutated the accepted checkpoint")
	}
	batch := dense(t, [][]float64{observation, observation})
	filtered, err := model.FilterWithHistory(ctx, batch)
	requireHMMFailure(t, filtered, err, hmm.ErrOverflow)
	smoothed, err := model.Smooth(ctx, batch)
	requireHMMFailure(t, smoothed, err, hmm.ErrOverflow)
	decoded, err := model.Decode(ctx, batch)
	requireHMMFailure(t, decoded, err, hmm.ErrOverflow)
	fit, err := hmm.FitGaussian(ctx, []matrix.Matrix{batch}, spec, hmm.FitOptions{})
	requireHMMFailure(t, fit, err, hmm.ErrOverflow)
}

// TestFitGaussian_IndependentSequenceEvidenceOverflow tests aggregation across valid runs.
// Each independent singleton has finite evidence and a representable within-sequence workspace.
//
// Implementation:
//   - Stage 1: verify a finite evidence score for each singleton under the initial model.
//   - Stage 2: fit the collection without concatenating the independent sequences.
//   - Stage 3: require an arithmetic error before publication of fitting diagnostics.
//
// Behavior highlights:
//   - The total log evidence over the collection is the overflowing quantity, not an emission or transition.
//
// Inputs:
//   - t: active test; two separate one-row matrices and a unit-variance model.
//
// Returns:
//   - None; the initial E-step cannot publish a valid aggregate objective.
//
// Errors:
//   - ErrOverflow and nil fit result.
//
// Determinism:
//   - Identical independent sequence scores, each less than -MaxFloat64/2.
//
// Complexity:
//   - Time O(1), Space O(1) for the fixed collection.
//
// Notes:
//   - This is separate from overflow of cumulative evidence within one sequence.
//
// AI-Hints:
//   - Do not concatenate sequences to manufacture the same error; that would miss the aggregate E-step boundary.
func TestFitGaussian_IndependentSequenceEvidenceOverflow(t *testing.T) {
	ctx := context.Background()
	spec := singleStateGaussianFixture(t, []float64{0}, []float64{1})
	model, err := hmm.NewGaussian(spec)
	if err != nil {
		t.Fatal(err)
	}
	sequences := []matrix.Matrix{dense(t, [][]float64{{1.5e154}}), dense(t, [][]float64{{1.5e154}})}
	for i, sequence := range sequences {
		result, err := model.Smooth(ctx, sequence)
		if err != nil || result == nil || !finite(result.LogLikelihood) || result.LogLikelihood >= -math.MaxFloat64/2 {
			t.Fatalf("sequence %d: expected finite independent evidence, result=%v err=%v", i, result, err)
		}
	}
	fit, err := hmm.FitGaussian(ctx, sequences, spec, hmm.FitOptions{})
	requireHMMFailure(t, fit, err, hmm.ErrOverflow)
}

// TestFitGaussian_UnrepresentablePopulationVariance verifies a genuinely overflowing fitted variance.
// Exact rational central moments distinguish final variance overflow from an avoidable intermediate overflow.
//
// Implementation:
//   - Stage 1: prove the one-state population variance exceeds the exact MaxFloat64 rational.
//   - Stage 2: verify that inference with the initial model has finite full-sequence evidence.
//   - Stage 3: fit and require ErrOverflow rather than a clamped floor, nonfinite model, or impossible support.
//
// Behavior highlights:
//   - The opposite-sign case also requires scaled differences and a scaled mean reconstruction.
//
// Inputs:
//   - t: active test; finite scalar observations and an admissible broad initial variance.
//
// Returns:
//   - None; the exact oracle establishes the expected failure before fitting.
//
// Errors:
//   - ErrOverflow with nil fit result during initial sufficient-statistics construction.
//
// Determinism:
//   - K=1 makes every responsibility exactly one; the oracle uses exact binary64 inputs.
//
// Complexity:
//   - O(T) rational operations for the independent oracle and O(T) HMM work for K=D=1.
//
// Notes:
//   - The large-opposite-sign fixture has an unrepresentable anchor residual but finite initial evidence.
//
// AI-Hints:
//   - Test final variance representability separately from sum-of-squares representability.
//   - Do not replace a genuinely overflowing variance with VarianceFloor.
func TestFitGaussian_UnrepresentablePopulationVariance(t *testing.T) {
	ctx := context.Background()
	cases := []struct {
		name        string
		values      []float64
		initialMean float64
	}{
		{"central_square", []float64{-1.35e154, 1.35e154}, 0},
		{"opposite_sign_anchor", []float64{1.4e308, -1.4e308, -1.4e308}, -1.4e308 / 3},
	}
	limit := new(big.Rat).SetFloat64(math.MaxFloat64)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if exactPopulationVariance(t, tc.values).Cmp(limit) <= 0 {
				t.Fatal("oracle fixture does not have an unrepresentable population variance")
			}
			rows := make([][]float64, len(tc.values))
			for i, value := range tc.values {
				rows[i] = []float64{value}
			}
			spec := singleStateGaussianFixture(t, []float64{tc.initialMean}, []float64{math.MaxFloat64})
			model, err := hmm.NewGaussian(spec)
			if err != nil {
				t.Fatal(err)
			}
			observations := dense(t, rows)
			result, err := model.Smooth(ctx, observations)
			if err != nil || result == nil || !finite(result.LogLikelihood) {
				t.Fatalf("initial model must have finite evidence: result=%v err=%v", result, err)
			}
			fit, err := hmm.FitGaussian(ctx, []matrix.Matrix{observations}, spec, hmm.FitOptions{MaxIterations: 1})
			requireHMMFailure(t, fit, err, hmm.ErrOverflow)
			requireUnchangedMatrix(t, observations, rows)
		})
	}
}

// TestFitGaussian_CandidateEmissionOverflow verifies failure during candidate evaluation.
// An initially representable two-state model becomes too sharp at the explicit variance floor.
//
// Implementation:
//   - Stage 1: use opposite singleton observations with near-disjoint initial Gaussian emissions.
//   - Stage 2: verify both independent sequences are representable with the initial parameters.
//   - Stage 3: fit once and require the candidate evaluation error without any published fit result.
//
// Behavior highlights:
//   - The old-model E-step has zero-rounded cross-state responsibilities and finite central moments.
//   - The constrained M-step produces small finite variances, but an off-state log density is unrepresentable.
//
// Inputs:
//   - t: active test; finite state means, unit variances, and explicit VarianceFloor.
//
// Returns:
//   - None; the initial model remains usable following a rejected fit.
//
// Errors:
//   - ErrOverflow when evaluating the candidate, not an input-domain or convergence error.
//
// Determinism:
//   - Two singleton sequences eliminate within-sequence transition changes.
//
// Complexity:
//   - Time O(KD), Space O(KD+K*K) for this fixed two-state fixture.
//
// Notes:
//   - The current Gaussian emission kernel evaluates every state's density, including the low-responsibility state.
//
// AI-Hints:
//   - Do not silently treat a numerically unrepresentable Gaussian emission as structural zero support.
func TestFitGaussian_CandidateEmissionOverflow(t *testing.T) {
	ctx := context.Background()
	const center = 5e153
	spec := hmm.GaussianSpec{
		Common: hmm.Common{Prior: []float64{.5, .5}, Transition: dense(t, [][]float64{{1, 0}, {0, 1}})},
		Means:  dense(t, [][]float64{{-center}, {center}}), Variances: dense(t, [][]float64{{1}, {1}}),
	}
	model, err := hmm.NewGaussian(spec)
	if err != nil {
		t.Fatal(err)
	}
	sequences := []matrix.Matrix{dense(t, [][]float64{{-center}}), dense(t, [][]float64{{center}})}
	for i, sequence := range sequences {
		result, err := model.Smooth(ctx, sequence)
		if err != nil || result == nil || !finite(result.LogLikelihood) {
			t.Fatalf("initial sequence %d: result=%v err=%v", i, result, err)
		}
		// Matching state has unit responsibility in binary64; the other state rounds to zero.
		for state := 0; state < 2; state++ {
			got, err := result.Posterior.At(0, state)
			want := 0.0
			if state == i {
				want = 1
			}
			if err != nil || got != want {
				t.Fatalf("initial posterior(%d,%d)=%g want=%g err=%v", i, state, got, want, err)
			}
		}
	}
	fit, err := hmm.FitGaussian(ctx, sequences, spec, hmm.FitOptions{MaxIterations: 1, VarianceFloor: 1e-8})
	requireHMMFailure(t, fit, err, hmm.ErrOverflow)
	requireUnchangedMatrix(t, spec.Means, [][]float64{{-center}, {center}})
	requireUnchangedMatrix(t, spec.Variances, [][]float64{{1}, {1}})
	// The original immutable model is not replaced by the failed candidate.
	if state, err := model.FilterStep(ctx, nil, []float64{-center}); err != nil || state == nil || !finite(state.LogLikelihood) {
		t.Fatalf("original model no longer usable: state=%v err=%v", state, err)
	}
}

// TestGaussianDecode_WideLosingPathContinues checks extended-score ordering across several steps.
// The losing state accumulates an unrepresentable joint score while the winning state remains finite.
//
// Implementation:
//   - Stage 1: build two absorbing states, with the matching state at the lower index.
//   - Stage 2: decode three identical finite observations, extending the losing path beyond its first overflow.
//   - Stage 3: verify the exact winning state sequence and analytic finite joint score.
//
// Behavior highlights:
//   - The existence of an overflowing losing branch does not invalidate a representable optimum.
//   - State ordering forces comparison of a wide candidate against a finite incumbent.
//
// Inputs:
//   - t: active test; two states, one feature, three observations, equal prior masses.
//
// Returns:
//   - None; the complete winning witness and score are independently checked.
//
// Errors:
//   - None: every emission is individually finite and the maximizing path is representable.
//
// Determinism:
//   - Both states are absorbing; the matching state's path is the unique optimum.
//
// Complexity:
//   - Time O(T*K*K), Space O(T*K) plus exceptional extended-score arithmetic.
//
// Notes:
//   - The observation is finite even though the losing cumulative log score is not.
//
// AI-Hints:
//   - Do not abort Viterbi merely because a nonwinning branch needs extended arithmetic.
func TestGaussianDecode_WideLosingPathContinues(t *testing.T) {
	ctx := context.Background()
	const center = 1.5e154
	model, err := hmm.NewGaussian(hmm.GaussianSpec{
		Common: hmm.Common{Prior: []float64{.5, .5}, Transition: dense(t, [][]float64{{1, 0}, {0, 1}})},
		Means:  dense(t, [][]float64{{center}, {0}}), Variances: dense(t, [][]float64{{1}, {1}}),
	})
	if err != nil {
		t.Fatal(err)
	}
	observations := dense(t, [][]float64{{center}, {center}, {center}})
	decoded, err := model.Decode(ctx, observations)
	if err != nil || decoded == nil || !slices.Equal(decoded.States, []int{0, 0, 0}) {
		t.Fatalf("decoded=%v err=%v, want unique finite winning path", decoded, err)
	}
	approx(t, decoded.LogJointProbability, math.Log(.5)-1.5*math.Log(2*math.Pi))
}
