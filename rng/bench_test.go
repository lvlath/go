// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng_test benchmarks the exact public draw, batch and stream contracts.
// Stream creation is deliberately isolated from hot-path consumption.
package rng_test

import (
	"testing"

	"github.com/lvlath/go/rng"
)

var sinkWord uint64
var sinkFloat float64
var sinkStream rng.RNG

// BenchmarkUint64 measures one raw hot-path step after one stream setup.
// It excludes seed hashing and records allocations independently.
func BenchmarkUint64(b *testing.B) {
	s := rng.NewFactory().Stream(10, 20)
	b.ReportAllocs()
	b.ResetTimer()
	var x uint64
	for i := 0; i < b.N; i++ {
		x = s.Uint64()
	}
	sinkWord = x
}

// BenchmarkUniformOpen measures one exact 52-bit conversion per draw.
// The stream is constructed outside the timed benchmark loop.
func BenchmarkUniformOpen(b *testing.B) {
	s := rng.NewFactory().Stream(10, 20)
	b.ReportAllocs()
	b.ResetTimer()
	var x float64
	for i := 0; i < b.N; i++ {
		x = s.UniformOpen()
	}
	sinkFloat = x
}

// BenchmarkNormal measures the amortized cached-pair Gaussian sequence.
// Half the calls consume two raw words and half consume none.
func BenchmarkNormal(b *testing.B) {
	s := rng.NewFactory().Stream(10, 20)
	b.ReportAllocs()
	b.ResetTimer()
	var x float64
	for i := 0; i < b.N; i++ {
		x = s.Normal()
	}
	sinkFloat = x
}

// BenchmarkStream measures the fixed SHA derivation and fresh state allocation.
// Unlike draw benchmarks, this includes one escaped RNG object per operation.
func BenchmarkStream(b *testing.B) {
	f := rng.NewFactory()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkStream = f.Stream(10, uint64(i))
	}
}

// BenchmarkFillUint64_Batch4096 isolates a 4096-word caller-owned batch.
// The destination allocation happens before the timed measurements.
func BenchmarkFillUint64_Batch4096(b *testing.B) {
	factory := rng.NewFactory()
	values := make([]uint64, 4096)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		factory.FillUint64(0, 9, values)
	}
	sinkWord = values[len(values)-1]
}

// BenchmarkFillUniformOpen_Batch4096 exercises one-word midpoint mapping per cell.
// It measures only the versioned new-stream batch facade, not allocation of dst.
func BenchmarkFillUniformOpen_Batch4096(b *testing.B) {
	factory := rng.NewFactory()
	values := make([]float64, 4096)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		factory.FillUniformOpen(0, 9, values)
	}
	sinkFloat = values[len(values)-1]
}

// BenchmarkFillNormal_Batch256 measures short sensor bursts with pair caching.
// It specifically observes the fresh-stream restart cost plus 128 Gaussian pairs.
func BenchmarkFillNormal_Batch256(b *testing.B) {
	factory := rng.NewFactory()
	values := make([]float64, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		factory.FillNormal(0, 9, values)
	}
	sinkFloat = values[len(values)-1]
}

// BenchmarkFillNormal_Batch4096 measures long consecutive Gaussian vectors.
// This regime amortizes fixed SHA derivation across 2048 Box-Muller pairs.
func BenchmarkFillNormal_Batch4096(b *testing.B) {
	factory := rng.NewFactory()
	values := make([]float64, 4096)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		factory.FillNormal(0, 9, values)
	}
	sinkFloat = values[len(values)-1]
}
