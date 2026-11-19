// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima_test checks concurrent immutability, cancellation and resource discipline.
// Every invocation uses only the exported public API and sentinel-first assertions.
package arima_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync/atomic"
	"testing"

	"github.com/lvlath/go/arima"
)

// cancelAfterQueries is a deterministic test context: no timers or scheduler dependence.
// Err remains canceled after the configured number of observations.
type cancelAfterQueries struct {
	context.Context
	calls atomic.Int64
	limit int64
}

func (c *cancelAfterQueries) Err() error {
	if c.calls.Add(1) > c.limit {
		return context.Canceled
	}
	return nil
}

func TestCancellationIsAtomicDuringFitAndForecast(t *testing.T) {
	values := syntheticARMA(10_000)
	fitContext := &cancelAfterQueries{Context: context.Background(), limit: 4}
	result, err := arima.Fit(fitContext, values, arima.Options{Order: arima.Order{AR: 1, MA: 1}, IncludeDrift: true})
	mustErrorIs(t, err, context.Canceled)
	if result != nil {
		t.Fatal("partial model unexpectedly published after cancellation")
	}

	model := mustModel(t, arima.Parameters{AR: []float64{0.3}, MA: []float64{-0.15}, InnovationVariance: 0.4, Order: arima.Order{AR: 1, MA: 1}})
	forecastContext := &cancelAfterQueries{Context: context.Background(), limit: 4}
	forecast, err := model.Forecast(forecastContext, values, 1024)
	mustErrorIs(t, err, context.Canceled)
	if forecast != nil {
		t.Fatal("partial forecast unexpectedly published after cancellation")
	}
}

func TestConcurrentImmutableForecastAndDetachedBuffers(t *testing.T) {
	model := mustModel(t, arima.Parameters{
		AR: []float64{0.4, -0.08}, MA: []float64{-0.25, 0.05}, Drift: 1.1,
		InnovationVariance: 2.5, Order: arima.Order{AR: 2, MA: 2, Difference: 1},
	})
	original := []float64{101, 103, 107, 109, 112, 116, 118, 119, 125, 128, 132, 135, 137, 142, 146, 149}
	want, err := model.Forecast(context.Background(), original, 32)
	if err != nil {
		t.Fatal(err)
	}

	const workers = 16
	failures := make(chan error, workers)
	for worker := 0; worker < workers; worker++ {
		go func() {
			history := append([]float64(nil), original...)
			forecast, forecastErr := model.Forecast(context.Background(), history, 32)
			if forecastErr != nil {
				failures <- forecastErr
				return
			}
			for index := range want.Mean {
				if forecast.Mean[index] != want.Mean[index] || forecast.Variance[index] != want.Variance[index] {
					failures <- fmt.Errorf("different forecast at horizon %d", index)
					return
				}
			}
			forecast.Mean[0] = math.NaN() // The result must never alias another forecast.
			failures <- nil
		}()
	}
	for worker := 0; worker < workers; worker++ {
		if err := <-failures; err != nil {
			t.Fatal(err)
		}
	}
	if math.IsNaN(want.Mean[0]) || model.Parameters().AR[0] != 0.4 {
		t.Fatal("concurrent consumers mutated model or independent results")
	}
}

func TestForecastHorizonAllocationsRemainLinearBufferCount(t *testing.T) {
	model := mustModel(t, arima.Parameters{AR: []float64{0.4}, MA: []float64{-0.2}, InnovationVariance: 1, Order: arima.Order{AR: 1, MA: 1}})
	history := syntheticARMA(200)
	small := testing.AllocsPerRun(5, func() {
		result, err := model.Forecast(context.Background(), history, 1)
		if err != nil || result.IsNil() {
			t.Fatal(err)
		}
	})
	large := testing.AllocsPerRun(5, func() {
		result, err := model.Forecast(context.Background(), history, 10_000)
		if err != nil || result.IsNil() {
			t.Fatal(err)
		}
	})
	if large > small+4 {
		t.Fatalf("forecast allocations grow with horizon: H1=%g, H10000=%g", small, large)
	}
}

func TestArithmeticOverflowRejectsUnusableResults(t *testing.T) {
	model := mustModel(t, arima.Parameters{InnovationVariance: 1, Order: arima.Order{Difference: 2}})
	forecast, err := model.Forecast(context.Background(), []float64{1e308, -1e308, 1e308}, 3)
	mustErrorIs(t, err, arima.ErrOverflow)
	if forecast != nil {
		t.Fatal("overflow returned partial forecast")
	}

	_, err = arima.Fit(context.Background(), []float64{1e308, -1e308, 1e308, 0, 1}, arima.Options{Order: arima.Order{Difference: 1}})
	if !errors.Is(err, arima.ErrOverflow) {
		t.Fatalf("expected differencing overflow, got %v", err)
	}
}

func TestStrictEffectiveSampleBoundary(t *testing.T) {
	ctx := context.Background()
	order := arima.Order{AR: 1, Difference: 2, MA: 1}
	// EffectiveN=(n-2)-1; the strict condition is EffectiveN>1+1+1+1.
	_, err := arima.Fit(ctx, []float64{2, 4, 7, 11, 16, 22, 29}, arima.Options{Order: order, IncludeDrift: true})
	mustErrorIs(t, err, arima.ErrInsufficientData)
	// The first admissible sample has n=8; it may still fail numerically, not on size.
	_, err = arima.Fit(ctx, []float64{2, 4, 7, 11, 16, 22, 29, 37}, arima.Options{Order: order, IncludeDrift: true})
	if errors.Is(err, arima.ErrInsufficientData) {
		t.Fatalf("rejected first admissible sample: %v", err)
	}
}
