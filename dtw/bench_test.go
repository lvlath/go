// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test benchmarks distinct DTW algorithmic regimes.
// Benchmarks fail fast on setup errors and always report allocations.
package dtw_test

import (
	"testing"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/matrix"
)

var benchSinkResult *dtw.Result

func benchmarkAlign(b *testing.B, name string, a, seqB []float64, opts ...dtw.Option) {
	b.Helper()
	b.ReportAllocs()

	res, err := dtw.Align(a, seqB, opts...)
	if err != nil {
		b.Fatalf("%s setup Align: %v", name, err)
	}
	if res == nil {
		b.Fatalf("%s setup Align: nil result", name)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		res, err = dtw.Align(a, seqB, opts...)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkResult = res
	}
}

func benchmarkAlignMatrix(b *testing.B, name string, x, y matrix.Matrix, opts ...dtw.Option) {
	b.Helper()
	b.ReportAllocs()

	res, err := dtw.AlignMatrix(x, y, opts...)
	if err != nil {
		b.Fatalf("%s setup AlignMatrix: %v", name, err)
	}
	if res == nil {
		b.Fatalf("%s setup AlignMatrix: nil result", name)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		res, err = dtw.AlignMatrix(x, y, opts...)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkResult = res
	}
}

func BenchmarkDTW_DistanceOnly_RollingRows_500x500(b *testing.B) {
	a := makeRamp(500)
	seqB := makeRamp(500)

	benchmarkAlign(b, "distance-only rolling rows", a, seqB, dtw.WithMemoryMode(dtw.TwoRows))
}

func BenchmarkDTW_FullMatrix_Path_250x250(b *testing.B) {
	a := makeRamp(250)
	seqB := makeRamp(250)

	benchmarkAlign(b, "full matrix path", a, seqB, dtw.WithReturnPath(true))
}

func BenchmarkDTW_WindowBand_Reachable_500x500_w10(b *testing.B) {
	a := makeRamp(500)
	seqB := makeRamp(500)

	benchmarkAlign(b, "reachable window band", a, seqB, dtw.WithWindow(10))
}

func BenchmarkDTW_WindowBand_NoPath_500x501_w0(b *testing.B) {
	a := makeRamp(500)
	seqB := makeRamp(501)

	benchmarkAlign(b, "no-path strict diagonal", a, seqB, dtw.WithWindow(0))
}

func BenchmarkDTW_MatrixSquaredL2_200x200_dim8(b *testing.B) {
	x := makeFeatureRamp(b, 200, 8)
	y := makeFeatureRamp(b, 200, 8)

	benchmarkAlignMatrix(b, "matrix squared L2", x, y, dtw.WithMemoryMode(dtw.TwoRows))
}

func makeRamp(n int) []float64 {
	values := make([]float64, n)
	for idx := range values {
		values[idx] = float64(idx)
	}

	return values
}

func makeFeatureRamp(b *testing.B, rows, cols int) *matrix.Dense {
	b.Helper()

	dense, err := matrix.NewPreparedDense(rows, cols)
	if err != nil {
		b.Fatalf("NewPreparedDense(%d,%d): %v", rows, cols, err)
	}

	values := make([]float64, rows*cols)
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			values[row*cols+col] = float64(row + col)
		}
	}

	if err = dense.Fill(values); err != nil {
		b.Fatalf("Fill(%d,%d): %v", rows, cols, err)
	}

	return dense
}
