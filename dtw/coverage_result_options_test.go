// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test verifies public result ownership and legacy/functional option contracts.
// Validation uses real API calls, not test-only production exports or fabricated private state.
package dtw_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/dtw"
)

// TestCoverageResult_NilFiniteAndUnreachable distinguishes published result states.
// A nil receiver, finite zero distance, and a structural no-path outcome are not equivalent.
// Implementation:
//   - Stage 1: exercise nil-safe predicates and Clone.
//   - Stage 2: create successful and unreachable results through Align.
//   - Stage 3: verify numeric predicates and sentinel-based path access.
//
// Complexity: Time O(n*m), Space O(m) for the small alignment fixtures.
// AI-Hints: IsFinite checks numeric representation, not whether path tracking was enabled.
func TestCoverageResult_NilFiniteAndUnreachable(t *testing.T) {
	var missing *dtw.Result
	mustEqualBool(t, missing.IsNil(), true, "nil result")
	mustEqualBool(t, missing.IsFinite(), false, "nil is not finite")
	if missing.Clone() != nil {
		t.Fatal("nil Clone must return nil")
	}
	path, err := missing.PathOrError()
	mustErrorIs(t, err, dtw.ErrNilResult, "nil path access")
	if path != nil {
		t.Fatalf("nil result published a path: %v", path)
	}

	finite, err := dtw.Align([]float64{0}, []float64{0})
	mustNoError(t, err, "zero distance")
	mustEqualBool(t, finite.IsNil(), false, "finite result exists")
	mustEqualBool(t, finite.IsFinite(), true, "zero distance is finite")
	mustFloatEqual(t, finite.Distance, 0, 0, "zero distance")
	path, err = finite.PathOrError()
	mustErrorIs(t, err, dtw.ErrPathNotTracked, "finite result without tracked path")
	if path != nil {
		t.Fatalf("untracked result published a path: %v", path)
	}

	unreachable, err := dtw.Align([]float64{0, 1}, []float64{0}, dtw.WithWindow(0))
	mustNoError(t, err, "distance-only unreachable")
	mustEqualBool(t, unreachable.IsNil(), false, "unreachable result exists")
	mustEqualBool(t, unreachable.IsFinite(), false, "unreachable distance is infinite")
	mustEqualBool(t, unreachable.Reachable, false, "unreachable endpoint")
	mustInf(t, unreachable.Distance, "unreachable distance")
	path, err = unreachable.PathOrError()
	mustErrorIs(t, err, dtw.ErrNoPath, "unreachable takes priority over path-not-tracked")
	if path != nil {
		t.Fatalf("unreachable result published a path: %v", path)
	}

	// Result is caller-owned; its numeric predicate must reject explicitly corrupted values too.
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		owned := &dtw.Result{Distance: value}
		mustEqualBool(t, owned.IsFinite(), false, "nonfinite result distance")
	}
}

// TestCoverageResult_CloneDetachesAllArtifacts verifies the complete ownership contract.
// A clone must preserve metadata and all values while isolating later caller mutations.
// Implementation:
//   - Stage 1: request path, local and accumulated surfaces for a strict diagonal alignment.
//   - Stage 2: verify clone metadata and every artifact cell against the source.
//   - Stage 3: mutate each clone artifact, then verify the original remains intact.
//
// Determinism: exact integer costs and a unique diagonal witness avoid tolerance ambiguity.
// Complexity: Time and Space O(n*m) for artifacts; O(n+m) for path storage.
// AI-Hints: Checking different outer pointers alone does not prove detached backing storage.
func TestCoverageResult_CloneDetachesAllArtifacts(t *testing.T) {
	result, err := dtw.Align([]float64{1, 4}, []float64{2, 6}, dtw.WithWindow(0),
		dtw.WithReturnPath(true), dtw.WithReturnAccumulated(true), dtw.WithReturnLocalCost(true))
	mustNoError(t, err, "alignment with all artifacts")
	if result == nil || result.Accumulated == nil || result.LocalCost == nil {
		t.Fatalf("missing alignment artifacts: %+v", result)
	}
	clone := result.Clone()
	if clone == nil || clone == result || clone.Accumulated == nil || clone.LocalCost == nil ||
		clone.Accumulated == result.Accumulated || clone.LocalCost == result.LocalCost {
		t.Fatalf("Clone did not detach all artifacts: %+v", clone)
	}
	if clone.Window != result.Window || clone.SlopePenalty != result.SlopePenalty ||
		clone.MemoryMode != result.MemoryMode || clone.PathTracked != result.PathTracked || clone.Reachable != result.Reachable {
		t.Fatalf("Clone did not preserve alignment metadata: %+v", clone)
	}
	mustFloatEqual(t, clone.Distance, 3, 0, "cloned exact objective")
	mustPathEqual(t, clone.Path, dtw.Path{{I: 0, J: 0}, {I: 1, J: 1}}, "cloned witness")
	mustMatrixShape(t, clone.LocalCost, 2, 2, "cloned local matrix")
	mustMatrixShape(t, clone.Accumulated, 3, 3, "cloned padded DP matrix")
	// Verify all local values and all DP cells, including the +Inf off-band representation.
	for i := 0; i < result.LocalCost.Rows(); i++ {
		for j := 0; j < result.LocalCost.Cols(); j++ {
			want, readErr := result.LocalCost.At(i, j)
			mustNoError(t, readErr, "source local cell")
			mustCoverageMatrixCell(t, clone.LocalCost, i, j, want, "cloned local cell")
		}
	}
	for i := 0; i < result.Accumulated.Rows(); i++ {
		for j := 0; j < result.Accumulated.Cols(); j++ {
			want, readErr := result.Accumulated.At(i, j)
			mustNoError(t, readErr, "source DP cell")
			mustCoverageMatrixCell(t, clone.Accumulated, i, j, want, "cloned DP cell")
		}
	}
	// PathOrError must also return independent storage, not the clone's path slice.
	witness, err := clone.PathOrError()
	mustNoError(t, err, "access cloned path")
	mustPathEqual(t, witness, clone.Path, "path accessor witness")
	witness[0] = dtw.Coord{I: 1, J: 1}
	mustPathEqual(t, clone.Path, dtw.Path{{I: 0, J: 0}, {I: 1, J: 1}}, "detached path accessor")

	// Modify the clone and prove that none of its backing arrays alias the source.
	clone.Path[0] = dtw.Coord{I: 1, J: 1}
	mustNoError(t, clone.LocalCost.Set(0, 0, 99), "mutate cloned local matrix")
	mustNoError(t, clone.Accumulated.Set(1, 1, 99), "mutate cloned DP matrix")
	mustPathEqual(t, result.Path, dtw.Path{{I: 0, J: 0}, {I: 1, J: 1}}, "original path")
	mustCoverageMatrixCell(t, result.LocalCost, 0, 0, 1, "original local cost")
	mustCoverageMatrixCell(t, result.Accumulated, 1, 1, 1, "original accumulated cost")
}

// TestCoverageOptions_LegacyValidationAndExecution verifies finalization through public Options.
// Legacy fields reach finalizeOptions directly, unlike validated functional option constructors.
// Implementation:
//   - Stage 1: validate valid and invalid legacy configurations.
//   - Stage 2: call DTW with each configuration and preserve the same error identity.
//   - Stage 3: validate a non-trivial penalty-bearing alignment and its path witness.
//
// Complexity: Time O(n*m), Space O(n*m) only for the explicit FullMatrix path case.
// AI-Hints: Legacy ReturnPath must not silently promote TwoRows to FullMatrix.
func TestCoverageOptions_LegacyValidationAndExecution(t *testing.T) {
	for _, tc := range []struct {
		name    string
		options dtw.Options
		want    []error
	}{
		{"defaults", dtw.DefaultOptions(), nil},
		{"invalid-window", dtw.Options{Window: -2, MemoryMode: dtw.TwoRows}, []error{dtw.ErrInvalidWindow, dtw.ErrBadInput}},
		{"negative-penalty", dtw.Options{Window: -1, SlopePenalty: -1, MemoryMode: dtw.TwoRows}, []error{dtw.ErrInvalidPenalty, dtw.ErrBadInput}},
		{"nan-penalty", dtw.Options{Window: -1, SlopePenalty: math.NaN(), MemoryMode: dtw.TwoRows}, []error{dtw.ErrInvalidPenalty, dtw.ErrBadInput}},
		{"infinite-penalty", dtw.Options{Window: -1, SlopePenalty: math.Inf(1), MemoryMode: dtw.TwoRows}, []error{dtw.ErrInvalidPenalty, dtw.ErrBadInput}},
		{"invalid-memory", dtw.Options{Window: -1, MemoryMode: dtw.MemoryMode(-1)}, []error{dtw.ErrBadInput}},
		{"legacy-path-needs-full", dtw.Options{Window: -1, ReturnPath: true, MemoryMode: dtw.TwoRows}, []error{dtw.ErrPathNeedsMatrix}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			validationErr := tc.options.Validate()
			_, path, alignErr := dtw.DTW([]float64{1}, []float64{1}, &tc.options)
			if len(tc.want) == 0 {
				mustNoError(t, validationErr, "valid legacy options")
				mustNoError(t, alignErr, "valid legacy alignment")
				return
			}
			for _, target := range tc.want {
				mustErrorIs(t, validationErr, target, "legacy Validate")
				mustErrorIs(t, alignErr, target, "legacy DTW")
			}
			if path != nil {
				t.Fatalf("invalid options published a path: %v", path)
			}
		})
	}
	// One horizontal step is mandatory and costs exactly one quarter; all local costs can be zero.
	opts := dtw.Options{Window: 1, SlopePenalty: 0.25, ReturnPath: true, MemoryMode: dtw.FullMatrix}
	mustNoError(t, opts.Validate(), "valid explicit legacy policy")
	distance, path, err := dtw.DTW([]float64{0, 2}, []float64{0, 0, 2}, &opts)
	mustNoError(t, err, "legacy alignment with penalty")
	mustFloatEqual(t, distance, 0.25, 0, "legacy objective")
	mustPathEqual(t, dtw.Path(path), dtw.Path{{I: 0, J: 0}, {I: 0, J: 1}, {I: 1, J: 2}}, "legacy witness")
}

// TestCoverageOptions_MemoryModeAndMetricOverride verifies explicit functional policies.
// The selected storage mode must not change a finite distance; the last metric option wins.
// Implementation: run each legal memory mode, reject an invalid enum, then compare metric order.
// Complexity: Time O(1), Space O(1) per one-cell alignment.
func TestCoverageOptions_MemoryModeAndMetricOverride(t *testing.T) {
	for _, mode := range []dtw.MemoryMode{dtw.NoMemory, dtw.TwoRows, dtw.FullMatrix} {
		got, err := dtw.Align([]float64{1}, []float64{4}, dtw.WithMemoryMode(mode))
		mustNoError(t, err, "legal memory policy")
		if got == nil || got.MemoryMode != mode || got.Distance != 3 {
			t.Fatalf("mode=%v result=%+v; want absolute distance 3 with selected mode", mode, got)
		}
	}
	got, err := dtw.Align([]float64{1}, []float64{4}, dtw.WithMemoryMode(dtw.MemoryMode(-1)))
	mustErrorIs(t, err, dtw.ErrBadInput, "invalid memory mode")
	if got != nil {
		t.Fatalf("invalid memory mode published %+v", got)
	}
	got, err = dtw.Align([]float64{1}, []float64{4}, dtw.WithSquaredCost(), dtw.WithAbsoluteCost())
	mustNoError(t, err, "absolute override")
	mustFloatEqual(t, got.Distance, 3, 0, "absolute override distance")
	got, err = dtw.Align([]float64{1}, []float64{4}, dtw.WithAbsoluteCost(), dtw.WithSquaredCost())
	mustNoError(t, err, "squared override")
	mustFloatEqual(t, got.Distance, 9, 0, "squared override distance")
}

// TestCoverageOptions_ValidateFiniteIsOnlyAnInputPolicy distinguishes two validation layers.
// Disabling input checks permits a custom cost function to interpret missing samples,
// but never permits nonfinite local costs or silently disables the default finite-input policy.
// Implementation:
//   - Stage 1: reject nonfinite second-sequence data with checks enabled.
//   - Stage 2: disable input checks and return a deterministic finite replacement cost.
//   - Stage 3: return NaN from the callback and require a numeric failure even with checks disabled.
//
// Complexity: Time O(1), Space O(1) per fixture.
// AI-Hints: WithValidateFinite(false) is not a license to publish NaN or Inf distances.
func TestCoverageOptions_ValidateFiniteIsOnlyAnInputPolicy(t *testing.T) {
	a, b := []float64{1}, []float64{math.NaN()}
	finiteCost := dtw.WithCostFunc(func(_, _ float64) (float64, error) { return 2, nil })
	got, err := dtw.Align(a, b, finiteCost, dtw.WithValidateFinite(true))
	mustErrorIs(t, err, dtw.ErrNaNInf, "enabled input validation")
	if got != nil {
		t.Fatalf("invalid input published %+v", got)
	}
	got, err = dtw.Align(a, b, finiteCost, dtw.WithValidateFinite(false))
	mustNoError(t, err, "explicit custom missing-value cost")
	mustFloatEqual(t, got.Distance, 2, 0, "finite custom cost")
	got, err = dtw.Align(a, b, dtw.WithValidateFinite(false),
		dtw.WithCostFunc(func(_, _ float64) (float64, error) { return math.NaN(), nil }))
	mustErrorIs(t, err, dtw.ErrNaNInf, "local-cost validation remains enabled")
	if got != nil {
		t.Fatalf("nonfinite local cost published %+v", got)
	}
}
