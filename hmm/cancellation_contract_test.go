// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm_test verifies cooperative cancellation at the public inference and fit boundaries.
// Every observed checkpoint is exercised with a real canceled context and immutable input.
package hmm_test

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/hmm"
	"github.com/lvlath/go/matrix"
)

// TestSmoothingAndEM_CancellationCheckpoints exercises all polled stages of two model families.
// It includes emission preparation, forward and backward recurrences, transition expectations,
// both Gaussian moment passes, and the evaluation of an EM candidate.
//
// Implementation:
//   - Stage 1: construct nontrivial two-state models and two independent observation sequences.
//   - Stage 2: discover each operation's cancellation polls using an ordinary active context.
//   - Stage 3: replay at every poll, require cancellation without partial publication, and check input ownership.
//
// Behavior highlights:
//   - No test relies on sleeps, goroutine scheduling, or private function call counts.
//   - The reference fit has a one-iteration budget and deliberately nonstationary initial parameters.
//
// Inputs:
//   - t: active test; fixtures remain unchanged during all replays.
//
// Returns:
//   - None; each public operation and poll is identified in the subtest name.
//
// Errors:
//   - Reference fits: ErrNotConverged with a diagnostic model after one accepted step.
//   - Interrupted calls: context.Canceled with no published result.
//
// Determinism:
//   - Each reference and replay uses identical observations and explicit options.
//
// Complexity:
//   - O(C*W) upper-bound time for C polls and reference work W;
//     per-operation peak memory is unchanged because replays are sequential.
//
// Notes:
//   - This test concerns cancellation semantics, not elapsed-time responsiveness.
//   - A cancellation at a later candidate E-step must not return the earlier fit as success.
//
// AI-Hints:
//   - Preserve errors.Is(context.Canceled) through internal sequence-index wrappers.
//   - Reuse immutable models; do not mutate inputs to enter defensive private branches.
func TestSmoothingAndEM_CancellationCheckpoints(t *testing.T) {
	common := hmm.Common{
		Prior:      []float64{.6, .4},
		Transition: dense(t, [][]float64{{.8, .2}, {.3, .7}}),
	}
	categoricalSpec := hmm.CategoricalSpec{
		Common: common, Emission: dense(t, [][]float64{{.75, .25}, {.25, .75}}),
	}
	gaussianSpec := hmm.GaussianSpec{
		Common: common,
		Means:  dense(t, [][]float64{{-1}, {1}}), Variances: dense(t, [][]float64{{2}, {2}}),
	}
	categoricalModel, err := hmm.NewCategorical(categoricalSpec)
	if err != nil {
		t.Fatal(err)
	}
	gaussianModel, err := hmm.NewGaussian(gaussianSpec)
	if err != nil {
		t.Fatal(err)
	}
	categoricalData := [][]uint32{{0, 1, 1}, {1, 0, 1}}
	gaussianRows := [][][]float64{{{-2}, {-1}, {2}}, {{1}, {2}, {-1}}}
	gaussianData := []matrix.Matrix{dense(t, gaussianRows[0]), dense(t, gaussianRows[1])}
	options := hmm.FitOptions{MaxIterations: 1, Tolerance: 1e-10, Pseudocount: .25, VarianceFloor: .25}

	cases := []struct {
		name         string
		run          func(context.Context) (core.Nilable, error)
		referenceErr error
	}{
		{"categorical_smooth", func(ctx context.Context) (core.Nilable, error) {
			return categoricalModel.Smooth(ctx, categoricalData[0])
		}, nil},
		{"gaussian_smooth", func(ctx context.Context) (core.Nilable, error) {
			return gaussianModel.Smooth(ctx, gaussianData[0])
		}, nil},
		{"categorical_fit", func(ctx context.Context) (core.Nilable, error) {
			return hmm.FitCategorical(ctx, categoricalData, categoricalSpec, options)
		}, hmm.ErrNotConverged},
		{"gaussian_fit", func(ctx context.Context) (core.Nilable, error) {
			return hmm.FitGaussian(ctx, gaussianData, gaussianSpec, options)
		}, hmm.ErrNotConverged},
		{"categorical_filter", func(ctx context.Context) (core.Nilable, error) {
			return categoricalModel.FilterWithHistory(ctx, categoricalData[0])
		}, nil},
		{"gaussian_filter", func(ctx context.Context) (core.Nilable, error) {
			return gaussianModel.FilterWithHistory(ctx, gaussianData[0])
		}, nil},
		{"categorical_decode", func(ctx context.Context) (core.Nilable, error) {
			return categoricalModel.Decode(ctx, categoricalData[0])
		}, nil},
		{"gaussian_decode", func(ctx context.Context) (core.Nilable, error) {
			return gaussianModel.Decode(ctx, gaussianData[0])
		}, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			exerciseCancellationCheckpoints(t, tc.run, tc.referenceErr)
		})
	}

	// Check the real caller-owned inputs after all interrupted and completed calls.
	if !slices.Equal(categoricalData[0], []uint32{0, 1, 1}) ||
		!slices.Equal(categoricalData[1], []uint32{1, 0, 1}) || !slices.Equal(common.Prior, []float64{.6, .4}) {
		t.Fatal("caller-owned categorical observations or prior were mutated")
	}
	for i, observations := range gaussianData {
		requireUnchangedMatrix(t, observations, gaussianRows[i])
	}
	requireUnchangedMatrix(t, common.Transition, [][]float64{{.8, .2}, {.3, .7}})
	requireUnchangedMatrix(t, categoricalSpec.Emission, [][]float64{{.75, .25}, {.25, .75}})
	requireUnchangedMatrix(t, gaussianSpec.Means, [][]float64{{-1}, {1}})
	requireUnchangedMatrix(t, gaussianSpec.Variances, [][]float64{{2}, {2}})

	// A fresh construction must still have the original identities after all replays.
	categoricalAfter, err := hmm.NewCategorical(categoricalSpec)
	if err != nil || categoricalAfter == nil || categoricalAfter.ModelID() != categoricalModel.ModelID() {
		t.Fatalf("categorical identity changed: err=%v", err)
	}
	gaussianAfter, err := hmm.NewGaussian(gaussianSpec)
	if err != nil || gaussianAfter == nil || gaussianAfter.ModelID() != gaussianModel.ModelID() {
		t.Fatalf("Gaussian identity changed: err=%v", err)
	}
}

// TestFilterStep_ContextPrecedenceAndRejectedSnapshot verifies error atomicity for both typed models.
// A canceled or nil context must win before a malformed external snapshot is inspected.
//
// Implementation:
//   - Stage 1: obtain legitimate first-step states, then create detached wrong-identity copies.
//   - Stage 2: test nil and canceled contexts, followed by a live context with the wrong identity.
//   - Stage 3: assert nil returned states, exact error identities, and unchanged caller snapshots.
//
// Behavior highlights:
//   - Wrong ModelID remains an error; the filter never silently renormalizes or replaces the snapshot.
//
// Inputs:
//   - t: active test; valid observations for both model families.
//
// Returns:
//   - None; all checks use the public FilterStep methods.
//
// Errors:
//   - ErrInvalidOption, context.Canceled, ErrStateVersion in the documented precedence order.
//
// Determinism:
//   - Cancellation is completed before invocation, and snapshot copies are stable throughout each call.
//
// Complexity:
//   - O(K) additional test memory for snapshot copies; constant number of calls.
//
// Notes:
//   - The input is intentionally malformed as a caller-supplied checkpoint, not through private model mutation.
//
// AI-Hints:
//   - Never reuse an incompatible ModelID just because the vector dimensions agree.
func TestFilterStep_ContextPrecedenceAndRejectedSnapshot(t *testing.T) {
	ctx := context.Background()
	categoricalModel := categorical(t)
	gaussianModel, err := hmm.NewGaussian(singleStateGaussianFixture(t, []float64{0}, []float64{1}))
	if err != nil {
		t.Fatal(err)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	cases := []struct {
		name string
		step func(context.Context, *hmm.FilterState) (*hmm.FilterState, error)
	}{
		{"categorical", func(ctx context.Context, previous *hmm.FilterState) (*hmm.FilterState, error) {
			return categoricalModel.FilterStep(ctx, previous, 0)
		}},
		{"gaussian", func(ctx context.Context, previous *hmm.FilterState) (*hmm.FilterState, error) {
			return gaussianModel.FilterStep(ctx, previous, []float64{0})
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			initial, err := tc.step(ctx, nil)
			if err != nil || initial == nil {
				t.Fatalf("initial state=%v err=%v", initial, err)
			}
			invalid := *initial
			invalid.ModelID = "different-model"
			invalid.LogPosterior = slices.Clone(initial.LogPosterior)
			before := invalid
			before.LogPosterior = slices.Clone(invalid.LogPosterior)
			for _, call := range []struct {
				ctx  context.Context
				want error
			}{{nil, hmm.ErrInvalidOption}, {canceled, context.Canceled}, {ctx, hmm.ErrStateVersion}} {
				result, err := tc.step(call.ctx, &invalid)
				if result != nil || !errors.Is(err, call.want) {
					t.Fatalf("step=%v err=%v want=%v", result, err, call.want)
				}
				if invalid.ModelID != before.ModelID || invalid.Steps != before.Steps ||
					math.Float64bits(invalid.LogLikelihood) != math.Float64bits(before.LogLikelihood) ||
					!slices.Equal(invalid.LogPosterior, before.LogPosterior) {
					t.Fatal("failed public step mutated its input snapshot")
				}
			}
		})
	}
}
