// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm_test controls cancellation at exact logical step boundaries.
package gbm_test

import (
	"context"
	"testing"

	"github.com/lvlath/go/gbm"
)

// cancelAfterFirstCheck models a cancellation immediately after public preflight.
// Its Err implementation does not depend on wall-clock time or goroutine races.
type cancelAfterFirstCheck struct {
	context.Context
	checks int
}

// Err returns nil once and context.Canceled at every later query.
func (c *cancelAfterFirstCheck) Err() error {
	c.checks++
	if c.checks > 1 {
		return context.Canceled
	}
	return nil
}

// TestGBM08CancelBeforeSecondStep proves prefix [t0,t1] survives a cancel.
func TestGBM08CancelBeforeSecondStep(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	source := &scripted{onDraw: func(count int) {
		if count == 1 {
			cancel()
		}
	}}
	result, err := gbm.Simulate(ctx,
		gbm.Parameters{Initial: 2, Volatility: 0.3},
		gbm.TimeGrid{Values: []float64{0, 1, 2, 3}}, source)
	mustErrorIs(t, err, context.Canceled)
	mustPrefix(t, result, 2)
	mustEqualInt(t, source.count, 1, "no draw on canceled second step")
	mustNear(t, result.Times[1], 1, 0)
}

// TestCancelAfterPreflightReturnsInitialPrefix checks the earliest runtime boundary.
func TestCancelAfterPreflightReturnsInitialPrefix(t *testing.T) {
	ctx := &cancelAfterFirstCheck{Context: context.Background()}
	source := &scripted{}
	result, err := gbm.Simulate(ctx,
		gbm.Parameters{Initial: 5, Volatility: 1},
		gbm.TimeGrid{Values: []float64{0, 1, 2}}, source)
	mustErrorIs(t, err, context.Canceled)
	mustPrefix(t, result, 1)
	mustEqualInt(t, source.count, 0, "cancel before first stochastic draw")
}

// TestPreCanceledReturnsNilBeforeInputValidation preserves context error priority.
func TestPreCanceledReturnsNilBeforeInputValidation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := gbm.Simulate(ctx, gbm.Parameters{}, gbm.TimeGrid{}, nil)
	mustErrorIs(t, err, context.Canceled)
	mustNilState(t, result, true, "pre-canceled result")
}

// TestPartialPrefixRemainsDetached checks ownership even on numerical failure.
func TestPartialPrefixRemainsDetached(t *testing.T) {
	times := []float64{0, 1, 2}
	source := &scripted{draws: []float64{0, 1e308}}
	result, err := gbm.Simulate(context.Background(),
		gbm.Parameters{Initial: 9, Drift: 0.5, Volatility: 1},
		gbm.TimeGrid{Values: times}, source)
	mustErrorIs(t, err, gbm.ErrOverflow)
	mustPrefix(t, result, 2)
	times[1] = 20
	mustNear(t, result.Times[1], 1, 0)
	result.Times[0] = -10
	mustNear(t, times[0], 0, 0)
}
