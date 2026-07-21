// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package matrix_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

// 1) TestDefaultOptions_Documented verifies that NewMatrixOptions() equals documented defaults.
func TestDefaultOptions_Documented(t *testing.T) {
	o, _ := matrix.NewMatrixOptionsSnapshotTestOnly()

	// numeric
	if o.Eps != matrix.DefaultEpsilon {
		t.Fatalf("eps default mismatch: got %v, want %v", o.Eps, matrix.DefaultEpsilon)
	}
	if o.ValidateNaNInf != matrix.DefaultValidateNaNInf {
		t.Fatalf("validateNaNInf default mismatch: got %v, want %v", o.ValidateNaNInf, matrix.DefaultValidateNaNInf)
	}
	if o.AllowInfDistances != matrix.DefaultAllowInfDistances {
		t.Fatalf("allowInfDistances default mismatch: got %v, want %v", o.AllowInfDistances, matrix.DefaultAllowInfDistances)
	}

	// build policy
	if o.Directed != matrix.DefaultDirected {
		t.Fatalf("directed default mismatch: got %v, want %v", o.Directed, matrix.DefaultDirected)
	}

	if o.AllowMulti != matrix.DefaultAllowMulti {
		t.Fatalf("allowMulti default mismatch: got %v, want %v", o.AllowMulti, matrix.DefaultAllowMulti)
	}
	if o.AllowLoops != matrix.DefaultAllowLoops {
		t.Fatalf("allowLoops default mismatch: got %v, want %v", o.AllowLoops, matrix.DefaultAllowLoops)
	}
	if o.Weighted != matrix.DefaultWeighted {
		t.Fatalf("weighted default mismatch: got %v, want %v", o.Weighted, matrix.DefaultWeighted)
	}
	if o.MetricClose != matrix.DefaultMetricClosure {
		t.Fatalf("metricClose default mismatch: got %v, want %v", o.MetricClose, matrix.DefaultMetricClosure)
	}

	// export policy
	if o.EdgeThreshold != matrix.DefaultEdgeThreshold {
		t.Fatalf("edgeThreshold default mismatch: got %v, want %v", o.EdgeThreshold, matrix.DefaultEdgeThreshold)
	}
	if o.KeepWeights != matrix.DefaultKeepWeights {
		t.Fatalf("keepWeights default mismatch: got %v, want %v", o.KeepWeights, matrix.DefaultKeepWeights)
	}
	if o.BinaryWeights != matrix.DefaultBinaryWeights {
		t.Fatalf("binaryWeights default mismatch: got %v, want %v", o.BinaryWeights, matrix.DefaultBinaryWeights)
	}
}

// 2) TestNewMatrixOptions_OrderAndIdempotence ensures each Option toggles exactly its intended field.
func TestNewMatrixOptions_OrderAndIdempotence(t *testing.T) {
	o1, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithDirected(), matrix.WithUndirected()) // last wins
	if o1.Directed != false {
		t.Fatalf("last-writer-wins failed: directed=%v, want false", o1.Directed)
	}
	o2, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithUndirected(), matrix.WithDirected())
	if o2.Directed != true {
		t.Fatalf("last-writer-wins failed: directed=%v, want true", o2.Directed)
	}

	o3, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithAllowMulti(), matrix.WithDisallowMulti())
	if o3.AllowMulti != false {
		t.Fatalf("allowMulti last-writer-wins failed: %v", o3.AllowMulti)
	}
	o4, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithDisallowMulti(), matrix.WithAllowMulti())
	if o4.AllowMulti != true {
		t.Fatalf("allowMulti last-writer-wins failed: %v", o4.AllowMulti)
	}

	o5, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithAllowLoops(), matrix.WithDisallowLoops())
	if o5.AllowLoops != false {
		t.Fatalf("allowLoops last-writer-wins failed: %v", o5.AllowLoops)
	}
	o6, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithDisallowLoops(), matrix.WithAllowLoops())
	if o6.AllowLoops != true {
		t.Fatalf("allowLoops last-writer-wins failed: %v", o6.AllowLoops)
	}

	o7, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithUnweighted(), matrix.WithWeighted())
	if o7.Weighted != true {
		t.Fatalf("weighted last-writer-wins failed: %v", o7.Weighted)
	}
	o8, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithWeighted(), matrix.WithUnweighted())
	if o8.Weighted != false {
		t.Fatalf("weighted last-writer-wins failed: %v", o8.Weighted)
	}

	o9, _ := matrix.GatherOptionsSnapshotTestOnly(
		matrix.WithEpsilon(1e-6),
		matrix.WithNoValidateNaNInf(),
		matrix.WithDirected(),
		matrix.WithDisallowMulti(),
		matrix.WithAllowLoops(),
		matrix.WithWeighted(),
		matrix.WithMetricClosure(),
		matrix.WithEdgeThreshold(0.25),
		matrix.WithBinaryWeights(),
	)
	if got := o9.Eps; got != 1e-6 {
		t.Fatalf("eps: got %v, want 1e-6", got)
	}
	if got := o9.ValidateNaNInf; got {
		t.Fatalf("validateNaNInf: got %v, want false", got)
	}
	if got := o9.Directed; !got {
		t.Fatalf("directed: got %v, want true", got)
	}
	if got := o9.AllowMulti; got {
		t.Fatalf("allowMulti: got %v, want false", got)
	}
	if got := o9.AllowLoops; !got {
		t.Fatalf("allowLoops: got %v, want true", got)
	}
	if got := o9.Weighted; !got {
		t.Fatalf("weighted: got %v, want true", got)
	}
	if got := o9.MetricClose; !got {
		t.Fatalf("metricClose: got %v, want true", got)
	}
	if got := o9.EdgeThreshold; got != 0.25 {
		t.Fatalf("edgeThreshold: got %v, want 0.25", got)
	}
	if got := o9.KeepWeights; got {
		t.Fatalf("keepWeights: got %v, want false", got)
	}
}

// 3) allowMulti last-writer-wins.
func TestNewMatrixOptions_LastWriterWins_AllowMulti(t *testing.T) {
	o1, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithAllowMulti(), matrix.WithDisallowMulti())
	if o1.AllowMulti {
		t.Fatalf("allowMulti last-writer-wins failed: got %v, want false", o1.AllowMulti)
	}

	o2, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithDisallowMulti(), matrix.WithAllowMulti())
	if !o2.AllowMulti {
		t.Fatalf("allowMulti last-writer-wins failed: got %v, want true", o2.AllowMulti)
	}
}

// 4) allowLoops last-writer-wins.
func TestNewMatrixOptions_LastWriterWins_AllowLoops(t *testing.T) {
	o1, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithAllowLoops(), matrix.WithDisallowLoops())
	if o1.AllowLoops {
		t.Fatalf("allowLoops last-writer-wins failed: got %v, want false", o1.AllowLoops)
	}

	o2, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithDisallowLoops(), matrix.WithAllowLoops())
	if !o2.AllowLoops {
		t.Fatalf("allowLoops last-writer-wins failed: got %v, want true", o2.AllowLoops)
	}
}

// 5) weighted last-writer-wins.
func TestNewMatrixOptions_LastWriterWins_Weighted(t *testing.T) {
	o1, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithUnweighted(), matrix.WithWeighted())
	if !o1.Weighted {
		t.Fatalf("weighted last-writer-wins failed: got %v, want true", o1.Weighted)
	}

	o2, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithWeighted(), matrix.WithUnweighted())
	if o2.Weighted {
		t.Fatalf("weighted last-writer-wins failed: got %v, want false", o2.Weighted)
	}
}

// 6) metricClose must imply allowInfDistances (distance-policy invariant).
func TestMetricClosure_EnablesAllowInfDistances(t *testing.T) {
	o, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithMetricClosure())
	if !o.MetricClose {
		t.Fatalf("metricClose expected true, got %v", o.MetricClose)
	}
	if !o.AllowInfDistances {
		t.Fatalf("metricClose must imply allowInfDistances=true, got %v", o.AllowInfDistances)
	}
}

// 7) export weight mode must be internally consistent and last-writer-wins.
func TestExportWeightMode_LastWriterWins(t *testing.T) {
	o1, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithKeepWeights())
	if !o1.KeepWeights || o1.BinaryWeights {
		t.Fatalf("keepWeights mode mismatch: keep=%v binary=%v", o1.KeepWeights, o1.BinaryWeights)
	}

	o2, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithBinaryWeights())
	if o2.KeepWeights || !o2.BinaryWeights {
		t.Fatalf("binaryWeights mode mismatch: keep=%v binary=%v", o2.KeepWeights, o2.BinaryWeights)
	}

	o3, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithKeepWeights(), matrix.WithBinaryWeights())
	if o3.KeepWeights || !o3.BinaryWeights {
		t.Fatalf("last-writer-wins mismatch: keep=%v binary=%v", o3.KeepWeights, o3.BinaryWeights)
	}

	o4, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithBinaryWeights(), matrix.WithKeepWeights())
	if !o4.KeepWeights || o4.BinaryWeights {
		t.Fatalf("last-writer-wins mismatch: keep=%v binary=%v", o4.KeepWeights, o4.BinaryWeights)
	}
}

// 8) epsilon setter must store the value exactly and be idempotent.
func TestWithEpsilon_SetsValue(t *testing.T) {
	o, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithEpsilon(1e-6), matrix.WithEpsilon(1e-6))
	if o.Eps != 1e-6 {
		t.Fatalf("eps mismatch: got %v, want %v", o.Eps, 1e-6)
	}
}

// 9) edgeThreshold setter must store the value exactly and be idempotent.
func TestWithEdgeThreshold_SetsValue(t *testing.T) {
	o, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithEdgeThreshold(0.25), matrix.WithEdgeThreshold(0.25))
	if o.EdgeThreshold != 0.25 {
		t.Fatalf("edgeThreshold mismatch: got %v, want %v", o.EdgeThreshold, 0.25)
	}
}

// 10) validateNaNInf toggles + deprecated alias must match behavior.
func TestValidateNaNInfToggles_AndAlias(t *testing.T) {
	o1, _ := matrix.GatherOptionsSnapshotTestOnly()
	if o1.ValidateNaNInf != true {
		t.Fatalf("default validateNaNInf expected true, got %v", o1.ValidateNaNInf)
	}

	o2, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithNoValidateNaNInf())
	if o2.ValidateNaNInf != false {
		t.Fatalf("WithNoValidateNaNInf expected false, got %v", o2.ValidateNaNInf)
	}

	o3, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithValidateNaNInf())
	if o3.ValidateNaNInf != true {
		t.Fatalf("WithValidateNaNInf expected true, got %v", o3.ValidateNaNInf)
	}

	o4, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.DisableValidateNaNInf())
	if o4.ValidateNaNInf != false {
		t.Fatalf("DisableValidateNaNInf expected false, got %v", o4.ValidateNaNInf)
	}
}

// 11) allowInfDistances must be togglable and last-writer-wins.
func TestAllowInfDistances_ToggleAndOrder(t *testing.T) {
	o1, _ := matrix.GatherOptionsSnapshotTestOnly()
	if o1.AllowInfDistances {
		t.Fatalf("default allowInfDistances expected false, got %v", o1.AllowInfDistances)
	}

	o2, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithAllowInfDistances())
	if !o2.AllowInfDistances {
		t.Fatalf("WithAllowInfDistances expected true, got %v", o2.AllowInfDistances)
	}

	o3, _ := matrix.GatherOptionsSnapshotTestOnly(matrix.WithAllowInfDistances(), matrix.WithDisallowInfDistances())
	if o3.AllowInfDistances {
		t.Fatalf("last-writer-wins expected false, got %v", o3.AllowInfDistances)
	}
}

// 12) TestDeprecatedAlias verifies DisableValidateNaNInf equals WithNoValidateNaNInf.
func TestDeprecatedAlias(t *testing.T) {
	a, err := matrix.GatherOptionsSnapshotTestOnly(matrix.WithNoValidateNaNInf())
	if err != nil {
		t.Fatalf("WithNoValidateNaNInf: %v", err)
	}

	b, err := matrix.GatherOptionsSnapshotTestOnly(matrix.DisableValidateNaNInf())
	if err != nil {
		t.Fatalf("DisableValidateNaNInf: %v", err)
	}

	if a.ValidateNaNInf || b.ValidateNaNInf {
		t.Fatalf("validate alias mismatch: both should set validateNaNInf=false")
	}

	c, err := matrix.GatherOptionsSnapshotTestOnly(matrix.WithEdgeThreshold(3.5))
	if err != nil {
		t.Fatalf("WithEdgeThreshold: %v", err)
	}

	d, err := matrix.GatherOptionsSnapshotTestOnly(matrix.WithThreshold(3.5))
	if err != nil {
		t.Fatalf("WithThreshold: %v", err)
	}

	if c.EdgeThreshold != d.EdgeThreshold {
		t.Fatalf("threshold alias mismatch: canonical=%v alias=%v", c.EdgeThreshold, d.EdgeThreshold)
	}

	e, err := matrix.GatherOptionsSnapshotTestOnly(matrix.WithBinaryWeights())
	if err != nil {
		t.Fatalf("WithBinaryWeights: %v", err)
	}

	f, err := matrix.GatherOptionsSnapshotTestOnly(matrix.WithBinary())
	if err != nil {
		t.Fatalf("WithBinary: %v", err)
	}

	if e.KeepWeights != f.KeepWeights || e.BinaryWeights != f.BinaryWeights {
		t.Fatalf("binary alias mismatch: canonical keep=%v binary=%v; alias keep=%v binary=%v",
			e.KeepWeights, e.BinaryWeights, f.KeepWeights, f.BinaryWeights)
	}
}

// 13) TestNewMatrixOptions_RejectsNilOption
func TestNewMatrixOptions_RejectsNilOption(t *testing.T) {
	var nilOpt matrix.Option

	_, err := matrix.NewMatrixOptions(nilOpt)
	if !errors.Is(err, matrix.ErrNilOption) {
		t.Fatalf("NewMatrixOptions(nil): got %v, want ErrNilOption", err)
	}
}

// 14) TestGatherOptions_RejectsNilOption
func TestGatherOptions_RejectsNilOption(t *testing.T) {
	var nilOpt matrix.Option

	_, err := matrix.GatherOptionsSnapshotTestOnly(nilOpt)
	if !errors.Is(err, matrix.ErrNilOption) {
		t.Fatalf("GatherOptionsSnapshotTestOnly(nil): got %v, want ErrNilOption", err)
	}

	_, err = matrix.GatherOptionsSnapshotTestOnly(matrix.WithDirected(), nilOpt)
	if !errors.Is(err, matrix.ErrNilOption) {
		t.Fatalf("GatherOptionsSnapshotTestOnly(WithDirected,nil): got %v, want ErrNilOption", err)
	}
}

// 15) TestWithEpsilon_ReturnsErrorsInsteadOfPanics
func TestWithEpsilon_ReturnsErrorsInsteadOfPanics(t *testing.T) {
	tests := []struct {
		name string
		eps  float64
		want error
	}{
		{name: "NaN", eps: math.NaN(), want: matrix.ErrNaNInf},
		{name: "+Inf", eps: math.Inf(+1), want: matrix.ErrNaNInf},
		{name: "-Inf", eps: math.Inf(-1), want: matrix.ErrNaNInf},
		{name: "negative", eps: -1, want: matrix.ErrOutOfRange},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := matrix.GatherOptionsSnapshotTestOnly(matrix.WithEpsilon(tt.eps))
			if !errors.Is(err, tt.want) {
				t.Fatalf("WithEpsilon(%v): got %v, want %v", tt.eps, err, tt.want)
			}
		})
	}
}

// 16) TestWithEdgeThreshold_ReturnsErrorsInsteadOfPanics
func TestWithEdgeThreshold_ReturnsErrorsInsteadOfPanics(t *testing.T) {
	tests := []struct {
		name      string
		threshold float64
		want      error
	}{
		{name: "NaN", threshold: math.NaN(), want: matrix.ErrNaNInf},
		{name: "+Inf", threshold: math.Inf(+1), want: matrix.ErrNaNInf},
		{name: "-Inf", threshold: math.Inf(-1), want: matrix.ErrNaNInf},
		{name: "negative", threshold: -1, want: matrix.ErrOutOfRange},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := matrix.GatherOptionsSnapshotTestOnly(matrix.WithEdgeThreshold(tt.threshold))
			if !errors.Is(err, tt.want) {
				t.Fatalf("WithEdgeThreshold(%v): got %v, want %v", tt.threshold, err, tt.want)
			}
		})
	}
}

// 17) TestDeprecatedOptions_PreserveCanonicalErrors
func TestDeprecatedOptions_PreserveCanonicalErrors(t *testing.T) {
	_, err := matrix.GatherOptionsSnapshotTestOnly(matrix.WithThreshold(math.NaN()))
	if !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("WithThreshold(NaN): got %v, want ErrNaNInf", err)
	}

	_, err = matrix.GatherOptionsSnapshotTestOnly(matrix.WithThreshold(-1))
	if !errors.Is(err, matrix.ErrOutOfRange) {
		t.Fatalf("WithThreshold(-1): got %v, want ErrOutOfRange", err)
	}
}
