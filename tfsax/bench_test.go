// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax_test benchmarks distinct generated workloads and implemented search arms.
// Setup is excluded from query timings; uncertified TFSAX and LB modes are not benchmarked.
package tfsax_test

import (
	"context"
	"testing"

	"github.com/lvlath/go/tfsax"
)

// BenchmarkGenerateWarp64x128 measures independently seeded local-warp corpus construction.
// The fixture belongs to the timed operation because generation is the measured workload.
func BenchmarkGenerateWarp64x128(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tfsax.GenerateFamily(tfsax.FamilyWarp, 29, 64, 128); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkExactScan64x128 measures the public exact facade, including corpus validation.
// Its timings are not directly interchangeable with prevalidated internal benchmark kernels.
func BenchmarkExactScan64x128(b *testing.B) {
	corpus, err := tfsax.GenerateFamily(tfsax.FamilyWarp, 29, 64, 128)
	if err != nil {
		b.Fatal(err)
	}
	query := tfsax.Query{Values: corpus[0].Values, TopK: 10}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tfsax.ScanExact(ctx, corpus, query, 8); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSAXColdRanking64x128 measures an entire one-shot SAX build plus query.
// It intentionally includes model construction and full candidate encoding in the loop.
func BenchmarkSAXColdRanking64x128(b *testing.B) {
	corpus, err := tfsax.GenerateFamily(tfsax.FamilyWarp, 29, 64, 128)
	if err != nil {
		b.Fatal(err)
	}
	query := tfsax.Query{Values: corpus[0].Values, TopK: 10}
	ctx := context.Background()
	options := tfsax.SAXSearchOptions{Window: 8, Shortlist: 16, Segments: 8, Alphabet: 8}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tfsax.ScanSAXWithOptions(ctx, corpus, query, options); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSAXPreparedRanking64x128 isolates a warm immutable SAX index search.
// This regime measures query encoding, MINDIST ranking and exact-DTW Top16 refinement.
func BenchmarkSAXPreparedRanking64x128(b *testing.B) {
	corpus, err := tfsax.GenerateFamily(tfsax.FamilyWarp, 29, 64, 128)
	if err != nil {
		b.Fatal(err)
	}
	index, err := tfsax.PrepareSAXIndex(corpus, 8, 8)
	if err != nil {
		b.Fatal(err)
	}
	query := tfsax.Query{Values: corpus[0].Values, TopK: 10}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := index.Search(ctx, query, 8, 16); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSAXIndexBuild64x128 measures the cold symbolic index construction cost.
// Candidate generation is excluded; raw copies and encoding are included deliberately.
func BenchmarkSAXIndexBuild64x128(b *testing.B) {
	corpus, err := tfsax.GenerateFamily(tfsax.FamilyWarp, 29, 64, 128)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := tfsax.PrepareSAXIndex(corpus, 8, 8); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSAXPreparedHeavyTail64x128 measures an outlier-sensitive shortlist.
// Unlike the smooth-warp workload, rare shocks can alter MINDIST ranking sharply.
func BenchmarkSAXPreparedHeavyTail64x128(b *testing.B) {
	corpus, err := tfsax.GenerateFamily(tfsax.FamilyHeavyTail, 29, 64, 128)
	if err != nil {
		b.Fatal(err)
	}
	index, err := tfsax.PrepareSAXIndex(corpus, 8, 8)
	if err != nil {
		b.Fatal(err)
	}
	query := tfsax.Query{Values: corpus[10].Values, TopK: 10}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := index.Search(ctx, query, 8, 16); err != nil {
			b.Fatal(err)
		}
	}
}
