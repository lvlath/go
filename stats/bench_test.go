// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats_test benchmarks scalar kernels and real Dense matrix workloads.
// Large matrix regimes are explicit opt-in because 1e6×128 uses ~1 GiB per
// matrix before accounting for independent Transform outputs.
// AI-Hints:
//   - Compare ns/op, B/op, and allocs/op on the same CPU and toolchain.
//   - Never include data construction in the timed loop or benchmark error paths.
package stats_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/stats"
)

const (
	// benchmarkMillion fixes the explicitly requested large observation count.
	benchmarkMillion = 1_000_000

	// benchmarkQuickRows keeps routine benchmark execution modest and repeatable.
	benchmarkQuickRows = 4096

	// benchmarkFeatureModulus produces a bounded, nonconstant data distribution.
	benchmarkFeatureModulus = 101

	// benchmarkHeavyEnvironment makes GiB-scale benchmarks opt-in, never silent.
	benchmarkHeavyEnvironment = "LVLATH_STATS_HEAVY_BENCH"
)

// benchmarkValues builds a deterministic large-offset scalar fixture.
// Every value is finite; the repeating low-order variation exercises centered
// moment accuracy without timing random generation or allocating per sample.
// Inputs: count is the exact number of scalar observations.
// Returns: caller-owned []float64 with len=count.
// Determinism: index modulus fixes every value.
// Complexity: Time/space O(count), entirely outside the timed kernel.
func benchmarkValues(count int) []float64 {
	values := make([]float64, count)
	for i := range values {
		values[i] = 1e12 + float64(i%benchmarkFeatureModulus)
	}
	return values
}

// benchmarkDense creates a real matrix.Dense fixture for fitted scaling.
// Its column-wise observations are nonconstant and the output shape is
// verified before timing to avoid an accidentally degenerate workload.
// Inputs: b is the benchmark owner; rows and cols are positive dimensions.
// Returns: independent *matrix.Dense with exactly rows×cols finite cells.
// Determinism: row-major Set, with no random generator state.
// Complexity: Time/space O(rows*cols), excluded from timed sections.
func benchmarkDense(b *testing.B, rows, cols int) *matrix.Dense {
	b.Helper()
	input, err := matrix.NewDense(rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			value := float64((row+3*col)%benchmarkFeatureModulus) + float64(col)
			if err = input.Set(row, col, value); err != nil {
				b.Fatal(err)
			}
		}
	}
	if input.Rows() != rows || input.Cols() != cols {
		b.Fatalf("fixture shape changed to %dx%d", input.Rows(), input.Cols())
	}
	return input
}

// benchmarkMatrixRegimes selects real dimensions and the expensive opt-in gate.
// The 1e6×128 regime is not silently downscaled: it is either run at the
// claimed shape or explicitly skipped until the environment flag is set.
// Inputs: no parameters; reads only the documented benchmark opt-in flag.
// Returns: fixed ordered workload descriptors.
// Determinism: d ascending within each chosen observation count.
// Complexity: O(1) time/space.
func benchmarkMatrixRegimes() []struct{ rows, cols int } {
	regimes := []struct{ rows, cols int }{
		{benchmarkQuickRows, 1},
		{benchmarkQuickRows, 16},
		{benchmarkQuickRows, 128},
	}
	if os.Getenv(benchmarkHeavyEnvironment) == "1" {
		regimes = append(regimes,
			struct{ rows, cols int }{benchmarkMillion, 1},
			struct{ rows, cols int }{benchmarkMillion, 16},
			struct{ rows, cols int }{benchmarkMillion, 128},
		)
	}
	return regimes
}

// BenchmarkMoments_Update measures 1e6 sequential compensated updates.
// The entire source slice is built before timing; each timed iteration creates
// one fresh accumulator and processes all observations in a fixed order.
// Complexity: O(1e6) per b.N iteration; no expected heap allocations per update.
// AI-Hints: do not use a constant-only workload; it avoids meaningful M2 work.
func BenchmarkMoments_Update(b *testing.B) {
	values := benchmarkValues(benchmarkMillion)
	b.SetBytes(int64(len(values)) * 8)
	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		var moments stats.Moments
		for _, value := range values {
			if err := moments.Update(value); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkMoments_Merge measures Chan composition of two 500k-sample states.
// The expensive source reductions are prepared outside timing; the timed
// operation merges independent states, not the initial 1e6 Updates.
// Complexity: O(1) per merge regardless of the combined sample count.
// AI-Hints: do not accidentally time the sample-building phase inside b.N.
func BenchmarkMoments_Merge(b *testing.B) {
	values := benchmarkValues(benchmarkMillion)
	left, right := stats.NewMoments(), stats.NewMoments()
	for _, value := range values[:benchmarkMillion/2] {
		if err := left.Update(value); err != nil {
			b.Fatal(err)
		}
	}
	for _, value := range values[benchmarkMillion/2:] {
		if err := right.Update(value); err != nil {
			b.Fatal(err)
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for iteration := 0; iteration < b.N; iteration++ {
		// Copying a value struct is safe in this sequential benchmark because
		// the source is quiescent and has no pointer-backed numeric state.
		destination := *left
		if err := destination.Merge(right); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkStandardizer_Fit measures training over dense multi-column matrices.
// Every case uses the true matrix shape in its name, excluding construction
// costs and any inference work from the timed segment.
// Complexity: O(rows*cols) per b.N iteration and O(cols) fitted state.
// AI-Hints: set LVLATH_STATS_HEAVY_BENCH=1 for 1e6×{1,16,128} workloads.
func BenchmarkStandardizer_Fit(b *testing.B) {
	for _, regime := range benchmarkMatrixRegimes() {
		name := fmt.Sprintf("N%d_D%d", regime.rows, regime.cols)
		b.Run(name, func(b *testing.B) {
			input := benchmarkDense(b, regime.rows, regime.cols)
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				if _, err := stats.FitStandardizer(input, stats.ScaleOptions{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkStandardizer_Transform measures immutable inference and output allocation.
// The fitted model and input matrix are built before the timed loop, so
// reported allocations are attributable to the real detached output.
// Complexity: O(rows*cols) per operation and O(rows*cols) output bytes.
// AI-Hints: output allocation is a contract requirement, not a removable cost.
func BenchmarkStandardizer_Transform(b *testing.B) {
	for _, regime := range benchmarkMatrixRegimes() {
		name := fmt.Sprintf("N%d_D%d", regime.rows, regime.cols)
		b.Run(name, func(b *testing.B) {
			input := benchmarkDense(b, regime.rows, regime.cols)
			fit, err := stats.FitStandardizer(input, stats.ScaleOptions{})
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for iteration := 0; iteration < b.N; iteration++ {
				if _, err := fit.Model.Transform(input); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
