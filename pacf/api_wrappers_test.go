// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf_test tests optional public facades against the canonical computation.
// Wrapper tests demand exact output and error parity, never alternative estimands.
package pacf_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/pacf"
)

// TestAPIWrappers_ExactParity rejects mathematical or allocation-policy drift
// between explicit wrappers and their corresponding canonical Options values.
func TestAPIWrappers_ExactParity(t *testing.T) {
	values := []float64{6, -2, 9, 4, -1, 5, 0, 3, 8, -4, 2, 7, 1}
	for _, center := range []bool{false, true} {
		want, err := pacf.Compute(context.Background(), values, pacf.Options{MaxLag: 7, Center: center})
		if err != nil {
			t.Fatal(err)
		}

		var got *pacf.Result
		if center {
			got, err = pacf.ComputeCentered(context.Background(), values, 7)
		} else {
			got, err = pacf.ComputeUncentered(context.Background(), values, 7)
		}
		if err != nil {
			t.Fatal(err)
		}
		mustEqualInt(t, got.Count, want.Count, "wrapper sample count")
		for lag := range got.Values {
			if math.Float64bits(got.Values[lag]) != math.Float64bits(want.Values[lag]) ||
				math.Float64bits(got.PredictionVariance[lag]) != math.Float64bits(want.PredictionVariance[lag]) {
				t.Fatalf("center=%t lag=%d wrapper differs from canonical result", center, lag)
			}
			point, pointErr := pacf.ComputeAtLag(context.Background(), values, lag, center)
			if pointErr != nil {
				t.Fatal(pointErr)
			}
			if math.Float64bits(point) != math.Float64bits(got.Values[lag]) {
				t.Fatalf("center=%t lag=%d point query differs", center, lag)
			}
		}
	}
}

// TestAPIWrappers_ErrorParity verifies that wrappers neither swallow errors nor
// replace canceled-context errors with ordinary invalid-parameter sentinels.
func TestAPIWrappers_ErrorParity(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	tests := []struct {
		name string
		call func() error
		want error
	}{
		{"centered constant", func() error {
			_, err := pacf.ComputeCentered(context.Background(), []float64{7, 7, 7}, 2)
			return err
		}, pacf.ErrZeroVariance},
		{"uncentered zero", func() error {
			_, err := pacf.ComputeUncentered(context.Background(), []float64{0, 0}, 1)
			return err
		}, pacf.ErrZeroVariance},
		{"point invalid lag", func() error {
			_, err := pacf.ComputeAtLag(context.Background(), []float64{1, 2}, 2, true)
			return err
		}, pacf.ErrInvalidOption},
		{"point canceled", func() error {
			_, err := pacf.ComputeAtLag(canceled, []float64{1, 2}, 1, true)
			return err
		}, context.Canceled},
		{"point nil", func() error {
			_, err := pacf.ComputeAtLag(context.Background(), nil, 0, false)
			return err
		}, pacf.ErrNilInput},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mustErrorIs(t, tc.call(), tc.want)
		})
	}
}
