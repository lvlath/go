// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test validates failure propagation across independent fit stages.
// These external tests use only the public Fit facade and Matrix interface.
package ridge_test

import (
	"context"
	"errors"
	"testing"

	"github.com/lvlath/go/ridge"
)

// TestRidge_ReaderFailureAfterPreflight confirms that training is not published
// when a custom Matrix.At reader fails during design construction or diagnostics.
// The test reader's storage values remain constant, but its access can fail.
func TestRidge_ReaderFailureAfterPreflight(t *testing.T) {
	cases := []struct {
		name         string
		failRead     int
		intercept    bool
		standardized bool
	}{
		// For a 2x1 origin design, preflight performs two At calls and
		// materialization performs the next two calls.
		{name: "augmented design input failure", failRead: 3, intercept: false},
		// Affine training additionally reads data to learn the center.
		{name: "training center input failure", failRead: 3, intercept: true},
		// Once preflight and design finish in origin mode, the next At
		// belongs to SSE verification, after the completed QR solve.
		{name: "residual verification input failure", failRead: 5, intercept: false},
		// The stats dependency must retain the custom Matrix.At root error.
		{name: "standardizer input failure", failRead: 3, intercept: true, standardized: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			marker := errors.New("test reader: unavailable observation")
			reads := 0
			x := &syntheticMatrix{rows: 2, cols: 1}
			x.at = func(row, _ int) (float64, error) {
				reads++
				if reads >= tc.failRead {
					return 0, marker
				}
				return float64(row + 1), nil
			}
			result, err := ridge.Fit(context.Background(), x, []float64{1, 2}, ridge.Options{
				Lambda:      0.5,
				NoIntercept: !tc.intercept,
				Standardize: tc.standardized,
			})
			if result != nil || !errors.Is(err, marker) {
				t.Fatalf("failRead=%d reads=%d result=%v err=%v", tc.failRead, reads, result, err)
			}
			if reads != tc.failRead {
				t.Fatalf("reader failed at call %d, expected %d", reads, tc.failRead)
			}
		})
	}
}
