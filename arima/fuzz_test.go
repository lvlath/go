// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima_test fuzzes numeric admission and forecast safety from the public surface.
// No reflection, recovery, timeouts, or private test-only access is required.
package arima_test

import (
	"context"
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/arima"
)

func FuzzNewAR1NumericDomain(f *testing.F) {
	for _, bits := range []uint64{0, math.Float64bits(0.2), math.Float64bits(-1), math.Float64bits(1), math.Float64bits(math.Inf(1)), math.Float64bits(math.NaN())} {
		var bytes [8]byte
		binary.LittleEndian.PutUint64(bytes[:], bits)
		f.Add(bytes[:])
	}
	f.Fuzz(func(t *testing.T, input []byte) {
		if len(input) < 8 {
			return
		}
		phi := math.Float64frombits(binary.LittleEndian.Uint64(input[:8]))
		model, err := arima.New(arima.Parameters{AR: []float64{phi}, InnovationVariance: 1, Order: arima.Order{AR: 1}})
		if err == nil {
			if model == nil || math.Abs(phi) >= 1 || math.IsNaN(phi) || math.IsInf(phi, 0) {
				t.Fatal("accepted an invalid AR(1) parameter")
			}
			return
		}
		if model != nil || !(errors.Is(err, arima.ErrNonFinite) || errors.Is(err, arima.ErrNonStationary)) {
			t.Fatalf("bad root/domain classification: %v", err)
		}
	})
}

func FuzzForecastHistoryNoPartial(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5})
	f.Add([]byte{0, 0, 0, 0, 0, 0})
	f.Fuzz(func(t *testing.T, input []byte) {
		if len(input) > 64 {
			return // Keep fuzz cases bounded and focused on numeric policy.
		}
		model, err := arima.New(arima.Parameters{AR: []float64{0.5}, MA: []float64{-0.2}, InnovationVariance: 1, Order: arima.Order{AR: 1, MA: 1, Difference: 2}})
		if err != nil {
			t.Fatal(err)
		}
		history := make([]float64, len(input))
		for i, value := range input {
			history[i] = float64(int(value)-128) * 1e300
		}
		result, forecastErr := model.Forecast(context.Background(), history, 3)
		if forecastErr != nil && result != nil {
			t.Fatal("partial result leaked on validation, overflow or short-history error")
		}
		if forecastErr == nil {
			if result == nil || result.Horizon != 3 {
				t.Fatal("invalid success shape")
			}
			for _, variance := range result.Variance {
				if variance <= 0 || math.IsInf(variance, 0) || math.IsNaN(variance) {
					t.Fatal("invalid forecast uncertainty")
				}
			}
		}
	})
}
