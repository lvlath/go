// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test verifies matrix options through the published Options
// value and observable construction/validation behavior. It does not read
// private policy fields or require a test-only production bridge.
package matrix_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

// TestMatrixOptionDefaultsAsObservablePolicy fixes the default finite-storage
// domain and checks that an explicit set of documented default options leaves
// the same public policy as the parameterless resolver.
func TestMatrixOptionDefaultsAsObservablePolicy(t *testing.T) {
	base := matrixMustOptions(t)
	explicit := matrixMustOptions(t,
		matrix.WithEpsilon(matrix.DefaultEpsilon),
		matrix.WithValidateNaNInf(),
		matrix.WithDisallowInfDistances(),
		matrix.WithUndirected(),
		matrix.WithAllowMulti(),
		matrix.WithDisallowLoops(),
		matrix.WithUnweighted(),
		matrix.WithAutoZeroWeights(),
		matrix.WithEdgeThreshold(matrix.DefaultEdgeThreshold),
		matrix.WithKeepWeights(),
	)
	// WithEdgeThreshold deliberately records explicitness, not just its scalar
	// value. Those policies differ when exporting +Inf-as-absence adjacency.
	if base == explicit {
		t.Fatal("explicit edge threshold must retain its explicitness")
	}
	noExplicitThreshold := matrixMustOptions(t,
		matrix.WithEpsilon(matrix.DefaultEpsilon), matrix.WithValidateNaNInf(),
		matrix.WithDisallowInfDistances(), matrix.WithUndirected(),
		matrix.WithAllowMulti(), matrix.WithDisallowLoops(),
		matrix.WithUnweighted(), matrix.WithAutoZeroWeights(),
		matrix.WithKeepWeights(),
	)
	if base != noExplicitThreshold {
		t.Fatalf("documented default policy changed: base=%+v explicit=%+v", base, noExplicitThreshold)
	}

	strict, err := matrix.NewPreparedDense(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err = strict.Set(0, 0, math.Inf(1)); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("default Dense accepted infinite observation: %v", err)
	}
	if err = strict.Set(0, 0, math.NaN()); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("default Dense accepted NaN observation: %v", err)
	}
}

// TestMatrixOptionsLastWriterWins compares the published immutable policy
// snapshots rather than exposing private fields. Independent public-operation
// tests verify the meaning of the selected modes; here only assembly order
// and stable last-writer semantics are under test.
func TestMatrixOptionsLastWriterWins(t *testing.T) {
	switches := []struct {
		name string
		off  matrix.Option
		on   matrix.Option
	}{
		{"direction", matrix.WithUndirected(), matrix.WithDirected()},
		{"multiedges", matrix.WithDisallowMulti(), matrix.WithAllowMulti()},
		{"loops", matrix.WithDisallowLoops(), matrix.WithAllowLoops()},
		{"weights", matrix.WithUnweighted(), matrix.WithWeighted()},
		{"numeric_validation", matrix.WithNoValidateNaNInf(), matrix.WithValidateNaNInf()},
		{"infinite_distances", matrix.WithDisallowInfDistances(), matrix.WithAllowInfDistances()},
		{"weight_export", matrix.WithBinaryWeights(), matrix.WithKeepWeights()},
		{"epsilon", matrix.WithEpsilon(1e-7), matrix.WithEpsilon(1e-6)},
		{"threshold", matrix.WithEdgeThreshold(0.25), matrix.WithEdgeThreshold(0.75)},
	}
	for _, item := range switches {
		t.Run(item.name, func(t *testing.T) {
			// Reconstruct both outcomes through one production resolver; no
			// shadow copy of private Options is maintained for the tests.
			off := matrixMustOptions(t, item.on, item.off)
			wantOff := matrixMustOptions(t, item.off)
			on := matrixMustOptions(t, item.off, item.on)
			wantOn := matrixMustOptions(t, item.on)
			if off != wantOff || on != wantOn {
				t.Fatalf("last writer was not preserved: off=%+v want=%+v on=%+v want=%+v", off, wantOff, on, wantOn)
			}
			if wantOff == wantOn {
				t.Fatalf("options in %s do not select distinct public policies", item.name)
			}
			repeated := matrixMustOptions(t, item.on, item.on)
			if repeated != wantOn {
				t.Fatalf("reapplying option changed policy: got %+v, want %+v", repeated, wantOn)
			}
		})
	}
}

// TestMatrixOptionsDerivedInvariants anchors mandatory policy implications:
// metric closure admits +Inf absence, whereas unweighted representation
// cannot preserve explicit weighted zero-edge semantics.
func TestMatrixOptionsDerivedInvariants(t *testing.T) {
	closure := matrixMustOptions(t, matrix.WithMetricClosure())
	explicitlyAllowed := matrixMustOptions(t, matrix.WithMetricClosure(), matrix.WithAllowInfDistances())
	explicitlyDisallowed := matrixMustOptions(t, matrix.WithMetricClosure(), matrix.WithDisallowInfDistances())
	if closure != explicitlyAllowed || closure != explicitlyDisallowed {
		t.Fatal("metric closure did not force distance infinity policy")
	}
	notWeighted := matrixMustOptions(t, matrix.WithPreserveZeroWeights(), matrix.WithUnweighted())
	plain := matrixMustOptions(t, matrix.WithUnweighted())
	if notWeighted != plain {
		t.Fatal("unweighted mode incorrectly retained weighted zero-edge policy")
	}
	weightedPlain := matrixMustOptions(t, matrix.WithWeighted())
	weightedPreserve := matrixMustOptions(t, matrix.WithWeighted(), matrix.WithPreserveZeroWeights())
	if weightedPlain == weightedPreserve {
		t.Fatal("weighted zero-edge preservation had no policy effect")
	}
	if matrixMustOptions(t, matrix.WithWeighted(), matrix.WithPreserveZeroWeights(), matrix.WithAutoZeroWeights()) != weightedPlain {
		t.Fatal("WithAutoZeroWeights did not reset explicitly preserved zero edges")
	}
	if matrixMustOptions(t, matrix.WithBinaryWeights(), matrix.WithKeepWeights()) != matrixMustOptions(t, matrix.WithKeepWeights()) {
		t.Fatal("keep-weights must win when selected last")
	}
	if matrixMustOptions(t, matrix.WithKeepWeights(), matrix.WithBinaryWeights()) != matrixMustOptions(t, matrix.WithBinaryWeights()) {
		t.Fatal("binary-weights must win when selected last")
	}
}

// TestMatrixOptionsPublicNumericEffects executes the resolved policy rather
// than relying on implementation metadata. The default rejects all nonfinite
// inputs, an explicit distance policy permits only +Inf, and an explicit
// no-validation mode permits raw telemetry for later sanitation.
func TestMatrixOptionsPublicNumericEffects(t *testing.T) {
	distance, err := matrix.NewPreparedDense(1, 1, matrix.WithAllowInfDistances())
	if err != nil {
		t.Fatal(err)
	}
	if err = distance.Set(0, 0, math.Inf(1)); err != nil {
		t.Fatalf("distance +Inf rejected: %v", err)
	}
	if err = distance.Set(0, 0, math.Inf(-1)); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("distance -Inf accepted: %v", err)
	}
	dirty, err := matrix.NewPreparedDense(1, 1, matrix.WithNoValidateNaNInf())
	if err != nil {
		t.Fatal(err)
	}
	if err = dirty.Set(0, 0, math.NaN()); err != nil {
		t.Fatalf("explicit dirty storage rejected: %v", err)
	}
	restored, err := matrix.NewPreparedDense(1, 1,
		matrix.WithNoValidateNaNInf(), matrix.WithValidateNaNInf())
	if err != nil {
		t.Fatal(err)
	}
	if err = restored.Set(0, 0, math.NaN()); !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("last-writer validation did not take effect: %v", err)
	}
}

// TestMatrixOptionsEpsilonAffectsDistanceValidation tests the numerical policy
// on an actual diagonal distance, not a private epsilon field. A diagonal
// outside the final tolerance is rejected, while one inside it is admitted.
func TestMatrixOptionsEpsilonAffectsDistanceValidation(t *testing.T) {
	diagonal, err := matrix.NewDense(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err = diagonal.Set(0, 0, 5e-6); err != nil {
		t.Fatal(err)
	}
	if err = matrix.ValidateDistanceMatrix(diagonal, matrix.WithEpsilon(1e-7)); !errors.Is(err, matrix.ErrBadShape) {
		t.Fatalf("tight diagonal tolerance did not reject: %v", err)
	}
	if err = matrix.ValidateDistanceMatrix(diagonal, matrix.WithEpsilon(1e-5)); err != nil {
		t.Fatalf("relaxed diagonal tolerance rejected: %v", err)
	}
}

// TestMatrixOptionAliasesHaveIdenticalPolicy tests compatibility aliases via
// the same public finalization path used by constructors and adapters.
func TestMatrixOptionAliasesHaveIdenticalPolicy(t *testing.T) {
	equivalences := []struct {
		name     string
		original matrix.Option
		alias    matrix.Option
	}{
		{"finite_policy", matrix.WithNoValidateNaNInf(), matrix.DisableValidateNaNInf()},
		{"threshold", matrix.WithEdgeThreshold(3.5), matrix.WithThreshold(3.5)},
		{"binary_weights", matrix.WithBinaryWeights(), matrix.WithBinary()},
	}
	for _, item := range equivalences {
		t.Run(item.name, func(t *testing.T) {
			if matrixMustOptions(t, item.original) != matrixMustOptions(t, item.alias) {
				t.Fatalf("compatibility alias %s changed public policy", item.name)
			}
		})
	}
}

// TestMatrixOptionsErrors tests public numeric validation and nil-option
// classification, with options checked before any large matrix allocation.
func TestMatrixOptionsErrors(t *testing.T) {
	var nilOption matrix.Option
	for _, options := range [][]matrix.Option{{nilOption}, {matrix.WithDirected(), nilOption}} {
		if _, err := matrix.NewMatrixOptions(options...); !errors.Is(err, matrix.ErrNilOption) {
			t.Fatalf("nil option failed classification: %v", err)
		}
	}
	for _, bad := range []struct {
		name     string
		option   matrix.Option
		sentinel error
	}{
		{"epsilon_nan", matrix.WithEpsilon(math.NaN()), matrix.ErrNaNInf},
		{"epsilon_inf", matrix.WithEpsilon(math.Inf(1)), matrix.ErrNaNInf},
		{"epsilon_negative", matrix.WithEpsilon(-1), matrix.ErrOutOfRange},
		{"threshold_nan", matrix.WithEdgeThreshold(math.NaN()), matrix.ErrNaNInf},
		{"threshold_inf", matrix.WithEdgeThreshold(math.Inf(-1)), matrix.ErrNaNInf},
		{"threshold_negative", matrix.WithEdgeThreshold(-1), matrix.ErrOutOfRange},
		{"alias_nan", matrix.WithThreshold(math.NaN()), matrix.ErrNaNInf},
		{"alias_negative", matrix.WithThreshold(-1), matrix.ErrOutOfRange},
	} {
		t.Run(bad.name, func(t *testing.T) {
			if _, err := matrix.NewMatrixOptions(bad.option); !errors.Is(err, bad.sentinel) {
				t.Fatalf("NewMatrixOptions classified %s as %v rather than %v", bad.name, err, bad.sentinel)
			}
		})
	}
}
