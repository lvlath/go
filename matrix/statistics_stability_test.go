// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test validates the numerical statistics contract exclusively
// through the public matrix API, including the non-Dense fallback interface.
package matrix_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/matrix"
)

// TestCovarianceRepresentableMeanAndVariance anchors two independent numerical
// failures: raw sums and raw Gram matrices can overflow before the final value.
func TestCovarianceRepresentableMeanAndVariance(t *testing.T) {
	t.Run("constant-huge-mean", func(t *testing.T) {
		input := statsTestDense(t, [][]float64{{1e308}, {1e308}})
		covariance, means, err := matrix.Covariance(input)
		if err != nil {
			t.Fatalf("Covariance: %v", err)
		}
		statsTestClose(t, means[0], 1e308, 0, 0)
		statsTestClose(t, statsTestValue(t, covariance, 0, 0), 0, 0, 0)
	})

	t.Run("representable-variance-after-division", func(t *testing.T) {
		input := statsTestDense(t, [][]float64{{-1e154}, {0}, {1e154}})
		covariance, means, err := matrix.Covariance(input)
		if err != nil {
			t.Fatalf("Covariance: %v", err)
		}
		statsTestClose(t, means[0], 0, 0, 0)
		statsTestClose(t, statsTestValue(t, covariance, 0, 0), 1e308, 0, 2e-15)
	})
}

// TestCenterColumnsLargeOffset verifies that normalization preserves an ulp
// of spread even when the published float64 mean cannot represent its midpoint.
func TestCenterColumnsLargeOffset(t *testing.T) {
	anchor := 1e308
	next := math.Nextafter(anchor, math.Inf(1))
	input := statsTestDense(t, [][]float64{{anchor}, {next}})
	before := input.Clone()
	centered, means, err := matrix.CenterColumns(input)
	if err != nil {
		t.Fatalf("CenterColumns: %v", err)
	}
	if len(means) != 1 || math.IsNaN(means[0]) || math.IsInf(means[0], 0) {
		t.Fatalf("unrepresentable mean: %v", means)
	}
	spread := (next - anchor) / 2
	statsTestClose(t, statsTestValue(t, centered, 0, 0), -spread, 0, 1e-14)
	statsTestClose(t, statsTestValue(t, centered, 1, 0), spread, 0, 1e-14)
	for i := 0; i < 2; i++ {
		if statsTestValue(t, input, i, 0) != statsTestValue(t, before, i, 0) {
			t.Fatal("input mutated")
		}
	}
}

// TestCovarianceSignedSymmetryAndPSD checks negative cross-covariance,
// bit-identical mirrored cells, and nonnegative quadratic forms up to roundoff.
func TestCovarianceSignedSymmetryAndPSD(t *testing.T) {
	input := statsTestDense(t, [][]float64{{1, 3, 2}, {2, 2, 4}, {3, 1, 6}, {4, 0, 8}})
	cov, means, err := matrix.Covariance(input)
	if err != nil {
		t.Fatalf("Covariance: %v", err)
	}
	for j, want := range []float64{2.5, 1.5, 5} {
		statsTestClose(t, means[j], want, 0, 0)
	}
	for j := 0; j < 3; j++ {
		for k := j + 1; k < 3; k++ {
			left := statsTestValue(t, cov, j, k)
			right := statsTestValue(t, cov, k, j)
			if math.Float64bits(left) != math.Float64bits(right) {
				t.Fatalf("asymmetric pair (%d,%d): %x vs %x", j, k, math.Float64bits(left), math.Float64bits(right))
			}
		}
	}
	statsTestClose(t, statsTestValue(t, cov, 0, 1), -5.0/3, 0, 3e-15)
	for _, direction := range [][3]float64{{1, 0, 0}, {1, 1, -1}, {1, -2, 3}, {-2, 3, 1}} {
		quadratic := 0.0
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				quadratic += direction[i] * statsTestValue(t, cov, i, j) * direction[j]
			}
		}
		if quadratic < -1e-12 {
			t.Fatalf("negative covariance quadratic form: %g", quadratic)
		}
	}
}

// TestCorrelationDirectMomentDomain checks that Pearson correlation does not
// depend on the representability of unscaled covariance intermediates.
func TestCorrelationDirectMomentDomain(t *testing.T) {
	input := statsTestDense(t, [][]float64{{-1e200, 1e200, 7}, {0, 0, 7}, {1e200, -1e200, 7}})
	cov, _, err := matrix.Covariance(input)
	if cov != nil || !errors.Is(err, matrix.ErrOverflow) {
		t.Fatalf("Covariance must fail with overflow and nil matrix: cov=%v err=%v", cov, err)
	}
	corr, means, stds, err := matrix.Correlation(input)
	if err != nil {
		t.Fatalf("Correlation must be independently representable: %v", err)
	}
	if len(means) != 3 || len(stds) != 3 {
		t.Fatalf("metadata lengths: %d %d", len(means), len(stds))
	}
	statsTestClose(t, stds[0], 1e200, 0, 2e-15)
	statsTestClose(t, stds[1], 1e200, 0, 2e-15)
	statsTestClose(t, stds[2], 0, 0, 0)
	statsTestClose(t, statsTestValue(t, corr, 0, 0), 1, 0, 0)
	statsTestClose(t, statsTestValue(t, corr, 1, 1), 1, 0, 0)
	statsTestClose(t, statsTestValue(t, corr, 0, 1), -1, 0, 2e-15)
	for j := 0; j < 3; j++ {
		statsTestClose(t, statsTestValue(t, corr, 2, j), 0, 0, 0)
		statsTestClose(t, statsTestValue(t, corr, j, 2), 0, 0, 0)
	}
}

// TestNormalizeRowsL2ExtremeValues covers representable huge and subnormal
// norms, all-zero rows, and truly unrepresentable finite final norms.
func TestNormalizeRowsL2ExtremeValues(t *testing.T) {
	inputs := []struct {
		name      string
		row       []float64
		wantNorm  float64
		wantFirst float64
	}{
		{"huge", []float64{1e308}, 1e308, 1},
		{"subnormal", []float64{math.SmallestNonzeroFloat64}, math.SmallestNonzeroFloat64, 1},
		{"zeros", []float64{0, 0}, 0, 0},
		{"three-four", []float64{3, 4}, 5, 0.6},
	}
	for _, test := range inputs {
		t.Run(test.name, func(t *testing.T) {
			input := statsTestDense(t, [][]float64{test.row})
			out, norms, err := matrix.NormalizeRowsL2(input)
			if err != nil {
				t.Fatalf("NormalizeRowsL2: %v", err)
			}
			statsTestClose(t, norms[0], test.wantNorm, 0, 3e-15)
			statsTestClose(t, statsTestValue(t, out, 0, 0), test.wantFirst, 0, 3e-15)
		})
	}
	t.Run("overflow", func(t *testing.T) {
		input := statsTestDense(t, [][]float64{{math.MaxFloat64, math.MaxFloat64}})
		out, norms, err := matrix.NormalizeRowsL2(input)
		if out != nil || norms != nil || !errors.Is(err, matrix.ErrOverflow) {
			t.Fatalf("expected nil outputs and ErrOverflow: out=%v norms=%v err=%v", out, norms, err)
		}
		if !errors.Is(err, matrix.ErrNaNInf) {
			t.Fatalf("legacy nonfinite-norm classification was lost: %v", err)
		}
	})
}

// TestStatisticsValidationPrecedence establishes nil, size, and domain
// ordering and treats non-finite observations as errors rather than results.
func TestStatisticsValidationPrecedence(t *testing.T) {
	var nilDense *matrix.Dense
	for _, input := range []matrix.Matrix{nil, nilDense} {
		_, _, err := matrix.CenterColumns(input)
		if !errors.Is(err, matrix.ErrNilMatrix) {
			t.Fatalf("CenterColumns typed nil: %v", err)
		}
		_, _, err = matrix.Covariance(input)
		if !errors.Is(err, matrix.ErrNilMatrix) {
			t.Fatalf("Covariance typed nil: %v", err)
		}
		_, _, _, err = matrix.Correlation(input)
		if !errors.Is(err, matrix.ErrNilMatrix) {
			t.Fatalf("Correlation typed nil: %v", err)
		}
		_, _, err = matrix.NormalizeRowsL2(input)
		if !errors.Is(err, matrix.ErrNilMatrix) {
			t.Fatalf("NormalizeRowsL2 typed nil: %v", err)
		}
	}

	one, err := matrix.NewDense(1, 0)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = matrix.Covariance(one)
	if !errors.Is(err, matrix.ErrDimensionMismatch) {
		t.Fatalf("Covariance n=1 c=0: %v", err)
	}
	_, _, _, err = matrix.Correlation(one)
	if !errors.Is(err, matrix.ErrDimensionMismatch) {
		t.Fatalf("Correlation n=1 c=0: %v", err)
	}

	zeroCols, err := matrix.NewDense(2, 0)
	if err != nil {
		t.Fatal(err)
	}
	cov, means, err := matrix.Covariance(zeroCols)
	if err != nil || cov.Rows() != 0 || cov.Cols() != 0 || len(means) != 0 {
		t.Fatalf("zero-feature covariance: cov=%v means=%v err=%v", cov, means, err)
	}
	corr, means, stds, err := matrix.Correlation(zeroCols)
	if err != nil || corr.Rows() != 0 || corr.Cols() != 0 || len(means) != 0 || len(stds) != 0 {
		t.Fatalf("zero-feature correlation: corr=%v means=%v stds=%v err=%v", corr, means, stds, err)
	}
	centered, columnMeans, err := matrix.CenterColumns(zeroCols)
	if err != nil || centered != zeroCols || len(columnMeans) != 0 {
		t.Fatalf("zero-element centered alias law: centered=%v means=%v err=%v", centered, columnMeans, err)
	}
	normalized, rowNorms, err := matrix.NormalizeRowsL2(zeroCols)
	if err != nil || normalized != zeroCols || len(rowNorms) != 2 {
		t.Fatalf("zero-element normalized alias law: normalized=%v norms=%v err=%v", normalized, rowNorms, err)
	}

	// Non-finite values are injected through an explicit opted-out Dense
	// constructor, not by ignoring the normal constructor's numeric policy.
	nonFinite, err := matrix.NewPreparedDense(2, 1, matrix.WithNoValidateNaNInf())
	if err != nil {
		t.Fatal(err)
	}
	if err = nonFinite.Set(0, 0, math.NaN()); err != nil {
		t.Fatal(err)
	}
	_, _, err = matrix.Covariance(nonFinite)
	if !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("Covariance NaN: %v", err)
	}
	_, _, _, err = matrix.Correlation(nonFinite)
	if !errors.Is(err, matrix.ErrNaNInf) {
		t.Fatalf("Correlation NaN: %v", err)
	}
}

// TestStatisticsFallbackParityErrorsAndOwnership proves identical numeric
// contracts for Dense and Matrix.At backends without exposing production internals.
func TestStatisticsFallbackParityErrorsAndOwnership(t *testing.T) {
	input := statsTestDense(t, [][]float64{{1, 2, 3}, {3, 4, 5}, {5, 8, 7}, {7, 10, 9}})
	fallback := &statsTestFallback{Matrix: input}
	for _, operation := range []struct {
		name string
		fn   func(matrix.Matrix) (matrix.Matrix, []float64, error)
	}{
		{"covariance", matrix.Covariance},
		{"center-columns", matrix.CenterColumns},
		{"normalize-rows-L2", matrix.NormalizeRowsL2},
	} {
		t.Run(operation.name, func(t *testing.T) {
			denseOut, denseMetadata, err := operation.fn(input)
			if err != nil {
				t.Fatal(err)
			}
			fallbackOut, fallbackMetadata, err := operation.fn(fallback)
			if err != nil {
				t.Fatal(err)
			}
			if denseOut.Rows() != fallbackOut.Rows() || denseOut.Cols() != fallbackOut.Cols() {
				t.Fatalf("shape mismatch: (%d,%d) versus (%d,%d)", denseOut.Rows(), denseOut.Cols(), fallbackOut.Rows(), fallbackOut.Cols())
			}
			for i := 0; i < denseOut.Rows(); i++ {
				for j := 0; j < denseOut.Cols(); j++ {
					statsTestClose(t, statsTestValue(t, denseOut, i, j), statsTestValue(t, fallbackOut, i, j), 0, 0)
				}
			}
			for i := range denseMetadata {
				statsTestClose(t, denseMetadata[i], fallbackMetadata[i], 0, 0)
			}
			if len(denseMetadata) > 0 {
				denseMetadata[0] = 123456
			}
			// Another call must not observe changes to caller-owned metadata.
			_, again, err := operation.fn(input)
			if err != nil {
				t.Fatal(err)
			}
			if len(again) > 0 && again[0] == 123456 {
				t.Fatal("metadata aliases persistent workspace")
			}
		})
	}

	denseCorr, denseMeans, denseStds, err := matrix.Correlation(input)
	if err != nil {
		t.Fatal(err)
	}
	fallbackCorr, fallbackMeans, fallbackStds, err := matrix.Correlation(fallback)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < denseCorr.Rows(); i++ {
		for j := 0; j < denseCorr.Cols(); j++ {
			statsTestClose(t, statsTestValue(t, denseCorr, i, j), statsTestValue(t, fallbackCorr, i, j), 0, 0)
		}
		statsTestClose(t, denseMeans[i], fallbackMeans[i], 0, 0)
		statsTestClose(t, denseStds[i], fallbackStds[i], 0, 0)
	}

	injected := errors.New("custom Matrix.At failure")
	bad := &statsTestFallback{Matrix: input, failure: injected, failRow: 1, failCol: 2}
	_, _, err = matrix.Covariance(bad)
	if !errors.Is(err, injected) {
		t.Fatalf("custom accessor error lost: %v", err)
	}
}

// TestCovarianceHiddenCancellation tests a representable covariance whose
// normalized dot product loses the only surviving original-coordinate term.
// The exact sample covariance is -1; broad relative tolerances are invalid
// because the correct result is tiny compared with the canceled 1e150 terms.
func TestCovarianceHiddenCancellation(t *testing.T) {
	input := statsTestDense(t, [][]float64{{1e150, 1}, {-1e150, 1}, {1, -2}})
	centered, centeredMeans, err := matrix.CenterColumns(input)
	if err != nil {
		t.Fatal(err)
	}
	statsTestClose(t, centeredMeans[0], 1.0/3, 0, 2e-15)
	statsTestClose(t, statsTestValue(t, centered, 2, 0), 2.0/3, 0, 2e-15)

	covariance, means, err := matrix.Covariance(input)
	if err != nil {
		t.Fatal(err)
	}
	statsTestClose(t, means[0], 1.0/3, 0, 2e-15)
	statsTestClose(t, statsTestValue(t, covariance, 0, 1), -1, 0, 1e-14)
	statsTestClose(t, statsTestValue(t, covariance, 1, 0), -1, 0, 1e-14)

	correlation, _, _, err := matrix.Correlation(input)
	if err != nil {
		t.Fatal(err)
	}
	statsTestClose(t, statsTestValue(t, correlation, 0, 1), -1/(math.Sqrt(3)*1e150), 0, 1e-13)
}

// TestStatisticsConcurrentReadOnlyReductions validates that separate calls
// retain no shared mutable numerical workspace. Worker goroutines send errors
// through a channel; they never call t.Fatal or FailNow on a non-owner thread.
func TestStatisticsConcurrentReadOnlyReductions(t *testing.T) {
	const (
		workers = 64
		rows    = 64
		cols    = 8
	)
	values := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		values[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			values[i][j] = float64((i*7+j*11)%97 - 48)
		}
	}
	input := statsTestDense(t, values)
	failures := make(chan error, workers)
	for worker := 0; worker < workers; worker++ {
		go func() {
			cov, _, err := matrix.Covariance(input)
			if err != nil {
				failures <- err
				return
			}
			if cov.Rows() != cols || cov.Cols() != cols {
				failures <- errors.New("unexpected covariance shape")
				return
			}
			corr, _, _, err := matrix.Correlation(input)
			if err != nil {
				failures <- err
				return
			}
			if corr.Rows() != cols || corr.Cols() != cols {
				failures <- errors.New("unexpected correlation shape")
				return
			}
			_, norms, err := matrix.NormalizeRowsL2(input)
			if err != nil {
				failures <- err
				return
			}
			if len(norms) != rows {
				failures <- errors.New("unexpected row-norm count")
				return
			}
			failures <- nil
		}()
	}
	for worker := 0; worker < workers; worker++ {
		if err := <-failures; err != nil {
			t.Fatalf("worker %d: %v", worker, err)
		}
	}
}

// TestStatisticsCheckedShapeArithmetic anchors integer overflow guards at
// public entry points, without attempting an impossible allocation.
func TestStatisticsCheckedShapeArithmetic(t *testing.T) {
	huge := statsTestShapeOnly{rows: 1 << 61, cols: 16}
	_, _, err := matrix.CenterColumns(huge)
	if !errors.Is(err, matrix.ErrInvalidDimensions) {
		t.Fatalf("center-columns unchecked dimension product: %v", err)
	}
	_, _, err = matrix.NormalizeRowsL2(huge)
	if !errors.Is(err, matrix.ErrInvalidDimensions) {
		t.Fatalf("normalize-rows unchecked dimension product: %v", err)
	}
	_, _, err = matrix.Covariance(huge)
	if !errors.Is(err, matrix.ErrInvalidDimensions) {
		t.Fatalf("covariance unchecked dimension product: %v", err)
	}
	_, _, _, err = matrix.Correlation(huge)
	if !errors.Is(err, matrix.ErrInvalidDimensions) {
		t.Fatalf("correlation unchecked dimension product: %v", err)
	}

	// The input cell count can fit in int even when the c*c output cannot.
	wide := statsTestShapeOnly{rows: 2, cols: 1 << 33}
	_, _, err = matrix.Covariance(wide)
	if !errors.Is(err, matrix.ErrInvalidDimensions) {
		t.Fatalf("covariance unchecked quadratic output shape: %v", err)
	}
}

// TestStatisticsNonfiniteInputs establishes the strict finite observation
// domain independently for NaN, +Inf and -Inf on every public reduction.
func TestStatisticsNonfiniteInputs(t *testing.T) {
	for _, value := range []struct {
		name  string
		value float64
	}{
		{"nan", math.NaN()},
		{"positive-inf", math.Inf(1)},
		{"negative-inf", math.Inf(-1)},
	} {
		t.Run(value.name, func(t *testing.T) {
			input, err := matrix.NewPreparedDense(2, 1, matrix.WithNoValidateNaNInf())
			if err != nil {
				t.Fatal(err)
			}
			if err = input.Set(1, 0, value.value); err != nil {
				t.Fatal(err)
			}
			_, _, err = matrix.CenterColumns(input)
			if !errors.Is(err, matrix.ErrNaNInf) {
				t.Fatalf("CenterColumns: %v", err)
			}
			_, _, err = matrix.NormalizeRowsL2(input)
			if !errors.Is(err, matrix.ErrNaNInf) {
				t.Fatalf("NormalizeRowsL2: %v", err)
			}
			_, _, err = matrix.Covariance(input)
			if !errors.Is(err, matrix.ErrNaNInf) {
				t.Fatalf("Covariance: %v", err)
			}
			_, _, _, err = matrix.Correlation(input)
			if !errors.Is(err, matrix.ErrNaNInf) {
				t.Fatalf("Correlation: %v", err)
			}
		})
	}
}

// TestStatisticsResultDetachAndDeterminism checks repeated bitwise output
// equality and proves that caller mutation of published outputs cannot mutate
// the input or alter results obtained from subsequent calls.
func TestStatisticsResultDetachAndDeterminism(t *testing.T) {
	input := statsTestDense(t, [][]float64{{1, 3}, {2, -4}, {4, 9}, {8, 12}})
	first, firstMeans, err := matrix.Covariance(input)
	if err != nil {
		t.Fatal(err)
	}
	for repetition := 0; repetition < 10; repetition++ {
		again, means, err := matrix.Covariance(input)
		if err != nil {
			t.Fatal(err)
		}
		for j := 0; j < 2; j++ {
			if math.Float64bits(firstMeans[j]) != math.Float64bits(means[j]) {
				t.Fatalf("mean nondeterminism at feature %d", j)
			}
			for k := 0; k < 2; k++ {
				if math.Float64bits(statsTestValue(t, first, j, k)) != math.Float64bits(statsTestValue(t, again, j, k)) {
					t.Fatalf("covariance nondeterminism at pair (%d,%d)", j, k)
				}
			}
		}
	}
	if err = first.Set(0, 0, 999); err != nil {
		t.Fatal(err)
	}
	if statsTestValue(t, input, 0, 0) != 1 {
		t.Fatal("covariance result mutated input through alias")
	}
	centered, means, err := matrix.CenterColumns(input)
	if err != nil {
		t.Fatal(err)
	}
	if err = centered.Set(0, 0, 777); err != nil {
		t.Fatal(err)
	}
	means[0] = 555
	if statsTestValue(t, input, 0, 0) != 1 {
		t.Fatal("centered result mutated input through alias")
	}
	_, meansAgain, err := matrix.CenterColumns(input)
	if err != nil || meansAgain[0] == 555 {
		t.Fatalf("centered metadata aliases another call: means=%v err=%v", meansAgain, err)
	}
}

// TestCenterColumnsUnrepresentableCenteredValue distinguishes a legitimate
// overflow of the final centered observation from premature summation
// overflow. No partial materialized matrix or mean slice may escape.
func TestCenterColumnsUnrepresentableCenteredValue(t *testing.T) {
	input := statsTestDense(t, [][]float64{
		{math.MaxFloat64},
		{-math.MaxFloat64},
		{-math.MaxFloat64},
	})
	centered, means, err := matrix.CenterColumns(input)
	if centered != nil || means != nil || !errors.Is(err, matrix.ErrOverflow) {
		t.Fatalf("expected nil output and true centered overflow: centered=%v means=%v err=%v", centered, means, err)
	}
}

// TestStatisticsConstantFeaturesAvoidExactRescue anchors the performance law
// for identically constant columns. Their centered products are provably zero,
// so a binary-rational rescue would be both unnecessary and expensive.
// This external test also verifies the complete zero-correlation policy.
func TestStatisticsConstantFeaturesAvoidExactRescue(t *testing.T) {
	const (
		rows = 32
		cols = 24
	)
	observations := make([][]float64, rows)
	for i := range observations {
		observations[i] = make([]float64, cols)
		for j := range observations[i] {
			observations[i][j] = float64(j+1) * 1e150
		}
	}
	input := statsTestDense(t, observations)
	cov, means, err := matrix.Covariance(input)
	if err != nil {
		t.Fatalf("Covariance(constant): %v", err)
	}
	corr, corrMeans, stds, err := matrix.Correlation(input)
	if err != nil {
		t.Fatalf("Correlation(constant): %v", err)
	}
	for j := 0; j < cols; j++ {
		statsTestClose(t, means[j], observations[0][j], 0, 0)
		statsTestClose(t, corrMeans[j], observations[0][j], 0, 0)
		if stds[j] != 0 {
			t.Fatalf("constant feature %d has positive std %g", j, stds[j])
		}
		for k := 0; k < cols; k++ {
			if value := statsTestValue(t, cov, j, k); value != 0 {
				t.Fatalf("covariance (%d,%d) of constants = %g, want 0", j, k, value)
			}
			if value := statsTestValue(t, corr, j, k); value != 0 {
				t.Fatalf("correlation (%d,%d) of constants = %g, want 0", j, k, value)
			}
		}
	}
}
