// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External benchmarks measure the numeric change-resolution and range regimes.
// Input construction precedes timing; iteration loops call only public facades.
package series_test

import (
	"math"
	"testing"

	"github.com/lvlath/go/series"
)

// BenchmarkLogReturns_NearEqualLarge measures the stable log1p route at scale.
// Inputs alternate between adjacent representable large positive float64 values
// to keep the workload in the small-relative-change numerical regime.
func BenchmarkLogReturns_NearEqualLarge(b *testing.B) {
	const sampleCount = 1024
	const referenceValue = 1e300

	values := make([]float64, sampleCount)
	for i := range values {
		if i%2 == 0 {
			values[i] = referenceValue
		} else {
			values[i] = math.Nextafter(referenceValue, math.Inf(1))
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := series.LogReturns(values); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDifference_MillionSamples measures linear subtraction and output cost.
// The sample count is fixed and all values are finite, so it exercises the
// arithmetic loop rather than error handling or any benchmark setup.
func BenchmarkDifference_MillionSamples(b *testing.B) {
	const sampleCount = 1_000_000
	values := make([]float64, sampleCount)
	for i := range values {
		values[i] = float64(i)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := series.Difference(values, 1); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkWindowRanges_Overlapping measures the count/append path at stride one.
// It includes allocation and publication of each fully overlapping Range but
// excludes fixture construction and any timestamps unrelated to index windows.
func BenchmarkWindowRanges_Overlapping(b *testing.B) {
	const length = 100_000
	const width = 128
	const stride = 1

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := series.WindowRanges(length, width, stride); err != nil {
			b.Fatal(err)
		}
	}
}
