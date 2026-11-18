// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm_test fuzzes the exported validation and numeric result boundary.
package gbm_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/gbm"
)

// FuzzSimulateBoundary tests arbitrary coefficients against error and prefix laws.
// It is deterministic for a given case because every Normal is scripted as zero.
func FuzzSimulateBoundary(f *testing.F) {
	f.Add(100.0, 0.02, 0.2, 1.0)
	f.Add(1e-300, 1000.0, 0.0, 1.0)
	f.Add(1.0, 0.0, 1e160, 1e-320)
	f.Add(math.MaxFloat64, -0.4, 0.0, 1.0)
	f.Add(math.SmallestNonzeroFloat64, -2.0, 0.0, 1.0)
	f.Fuzz(func(t *testing.T, initial, drift, volatility, horizon float64) {
		params := gbm.Parameters{Initial: initial, Drift: drift, Volatility: volatility}
		grid := gbm.TimeGrid{Values: []float64{0, horizon}}
		result, err := gbm.Simulate(context.Background(), params, grid, &scripted{})
		if err == nil {
			if result == nil || !result.Complete || len(result.Times) != 2 || len(result.Values) != 2 {
				t.Fatalf("successful path invalid: %+v", result)
			}
			for _, value := range result.Values {
				if !isPositiveFinite(value) {
					t.Fatalf("successful price is not positive finite: %g", value)
				}
			}
			return
		}
		if result != nil {
			if !errors.Is(err, gbm.ErrOverflow) || result.Complete ||
				len(result.Times) != 1 || len(result.Values) != 1 {
				t.Fatalf("invalid runtime prefix: result=%+v error=%v", result, err)
			}
		} else if !errors.Is(err, gbm.ErrOrder) && !errors.Is(err, gbm.ErrNonFinite) &&
			!errors.Is(err, gbm.ErrDomain) {
			t.Fatalf("unclassified preflight error: %v", err)
		}
	})
}
