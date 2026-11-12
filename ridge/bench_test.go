// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test benchmarks explicit augmented QR and frozen-model inference.
// AI-Hints: All data generation and fit setup stay outside the timed benchmark loops.
// Any reported improvement requires oracle parity and an identical Go/CPU baseline.
package ridge_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ridge"
)

// BenchmarkRidge_Fit measures the canonical QR design across tall and wide cases.
// Sizes represent different augmentation costs; regularization keeps duplicate
// or dependent original columns mathematically admissible.
//
// Implementation:
//   - Stage 1: generate deterministic finite matrix and response fixtures.
//   - Stage 2: reset the timer and execute only the canonical ridge.Fit call.
//   - Stage 3: fail if any fit loses its contract or yields a nil result.
//
// Determinism: fixed integer-derived values, no random seeds or map iteration.
// Complexity: O((n+p)*p^2+p^3) per measured iteration and O((n+p)*p) memory.
// AI-Hints: Never compare n=100,p=500 directly to n=1000,p=10 as equal workloads.
func BenchmarkRidge_Fit(b *testing.B) {
	shapes := []struct{ n, p int }{
		{100, 10}, {100, 100}, {100, 500},
		{1000, 10}, {1000, 100}, {1000, 500},
	}
	for _, shape := range shapes {
		b.Run(fmt.Sprintf("n%d_p%d", shape.n, shape.p), func(b *testing.B) {
			data, err := matrix.NewDense(shape.n, shape.p)
			if err != nil {
				b.Fatal(err)
			}
			response := make([]float64, shape.n)
			for i := 0; i < shape.n; i++ {
				response[i] = 1 + .001*float64(i)
				for j := 0; j < shape.p; j++ {
					value := float64((i*17+j*31)%97)/97 + float64((i+j)%7)/13
					if err = data.Set(i, j, value); err != nil {
						b.Fatal(err)
					}
				}
			}
			ctx := context.Background()
			opts := ridge.Options{Lambda: 1, NoIntercept: true}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if result, err := ridge.Fit(ctx, data, response, opts); err != nil || result == nil {
					b.Fatalf("ridge Fit failure: %v", err)
				}
			}
		})
	}
}

// BenchmarkRidge_PredictFrozen measures repeated inference from one immutable
// fitted model, excluding training, held-out data construction and validation setup.
// The shape stresses 100-feature compensation and full input finite preflight.
//
// Implementation:
//   - Stage 1: build 200 training rows and 1000 inference rows deterministically.
//   - Stage 2: fit once and reset the benchmark timer before all predictions.
//   - Stage 3: repeatedly call Predict, verifying that every batch completes.
//
// Determinism: fixed row-major data and sequential invocations; no shared mutation.
// Complexity: O(1000*100) per inference call, with O(1000) returned output.
// AI-Hints: Do not place training inside b.N or the result will not measure inference.
func BenchmarkRidge_PredictFrozen(b *testing.B) {
	const (
		trainRows   = 200
		predictRows = 1000
		features    = 100
	)
	train, err := matrix.NewDense(trainRows, features)
	if err != nil {
		b.Fatal(err)
	}
	future, err := matrix.NewDense(predictRows, features)
	if err != nil {
		b.Fatal(err)
	}
	response := make([]float64, trainRows)
	for i := 0; i < trainRows; i++ {
		response[i] = float64((i*7)%41) / 10
		for j := 0; j < features; j++ {
			value := float64((i*11+j*19)%131)/131 + .001*float64(j)
			if err = train.Set(i, j, value); err != nil {
				b.Fatal(err)
			}
		}
	}
	for i := 0; i < predictRows; i++ {
		for j := 0; j < features; j++ {
			value := float64((i*7+j*17)%137)/137 + .001*float64(j)
			if err = future.Set(i, j, value); err != nil {
				b.Fatal(err)
			}
		}
	}
	ctx := context.Background()
	fit, err := ridge.FitStandardized(ctx, train, response, .2)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if predicted, err := fit.Model.Predict(ctx, future); err != nil || len(predicted) != predictRows {
			b.Fatalf("ridge Predict failure: %v", err)
		}
	}
}
