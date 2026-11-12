// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test verifies cooperative cancellation inside QR preparation.
// The caller owns the test context; no production test-only entry points exist.
package ridge_test

import (
	"context"
	"errors"
	"testing"

	"github.com/lvlath/go/ridge"
)

// cancelAfterPollContext cancels its underlying context on a deterministic
// Err observation, modeling a deadline that expires between preparation stages.
// The cancellation remains sticky and the original context identity is preserved.
type cancelAfterPollContext struct {
	context.Context
	cancel   context.CancelFunc
	checks   int
	cancelAt int
}

// Err counts an observable cooperative cancellation checkpoint and cancels at
// the caller-selected boundary. The context is used by one training goroutine.
func (c *cancelAfterPollContext) Err() error {
	c.checks++
	if c.checks == c.cancelAt {
		c.cancel()
	}
	return c.Context.Err()
}

// TestRidge_QRPreparationCancellation checks that cancellation during either
// column scaling or penalty-first matrix materialization returns no fitted model.
// The selected checkpoints are deterministic for this small external fixture;
// the contract under test is cancellation identity and atomic publication.
func TestRidge_QRPreparationCancellation(t *testing.T) {
	cases := []struct {
		name     string
		cancelAt int
	}{
		{name: "column-scale-start", cancelAt: 11},
		{name: "column-scale-scan", cancelAt: 12},
		{name: "balanced-matrix-write", cancelAt: 16},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			probe := &cancelAfterPollContext{Context: ctx, cancel: cancel, cancelAt: tc.cancelAt}
			fit, err := ridge.FitThroughOrigin(probe, dense(t, [][]float64{{1}, {2}}), []float64{1, 2}, 0.5)
			if fit != nil || !errors.Is(err, context.Canceled) {
				t.Fatalf("cancelAt=%d checks=%d: fit=%v err=%v", tc.cancelAt, probe.checks, fit, err)
			}
		})
	}
}
