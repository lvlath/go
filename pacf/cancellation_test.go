// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package pacf_test verifies cooperative cancellation without timers or sleeps.
// The test context cancels predictably while keeping poll placement private.
package pacf_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/pacf"
)

// checkedCancelContext simulates context cancellation after a configured
// number of Err polls while inheriting the remaining Context interface.
// It belongs to one test goroutine; it is intentionally not shared concurrently.
type checkedCancelContext struct {
	context.Context
	checks    int
	threshold int
}

// Err returns context.Canceled once the deterministic poll budget is exhausted.
// This lets tests interrupt active loops without scheduler timing or sleeps.
func (c *checkedCancelContext) Err() error {
	c.checks++
	if c.checks >= c.threshold {
		return context.Canceled
	}
	return nil
}

// observeContextPolls discovers the current full-call poll budget through the
// public facade; cancellation tests then use relative positions, not fixed
// implementation-private loop offsets or hardcoded poll counts.
func observeContextPolls(t *testing.T, values []float64, options pacf.Options) int {
	t.Helper()
	probe := &checkedCancelContext{Context: context.Background(), threshold: math.MaxInt}
	result, err := pacf.Compute(probe, values, options)
	if err != nil || result == nil {
		t.Fatalf("uncanceled baseline PACF must succeed: %v", err)
	}
	return probe.checks
}

// TestPACF_ContextCancellationDuringStages demands a nil result at every
// interruption, including after some lag coefficients were computed privately.
// Cutoffs are derived from a successful full call to avoid pinning poll topology.
func TestPACF_ContextCancellationDuringStages(t *testing.T) {
	x := make([]float64, 64)
	for i := range x {
		x[i] = math.Sin(float64(i)*0.31) + 0.4*math.Cos(float64(i)*0.17)
	}
	options := pacf.Options{MaxLag: 30, Center: true}
	fullChecks := observeContextPolls(t, x, options)

	// The contract is cancellation atomicity, not the number of Err calls per stage.
	thresholds := []int{1, 2, fullChecks / 4, fullChecks / 2, fullChecks - 1, fullChecks}
	for _, threshold := range thresholds {
		if threshold < 1 {
			t.Fatal("expected multiple cancellation opportunities")
		}
		ctx := &checkedCancelContext{Context: context.Background(), threshold: threshold}
		result, err := pacf.Compute(ctx, x, options)
		mustNilState(t, result, true, "interrupted operation")
		mustErrorIs(t, err, context.Canceled)
	}
}

// TestPACF_ContextCancellationDuringLargePairScan checks that long covariance
// scans expose cancellation opportunities before publishing any complete result.
func TestPACF_ContextCancellationDuringLargePairScan(t *testing.T) {
	x := make([]float64, 8192)
	for i := range x {
		x[i] = math.Sin(float64(i)*0.11) + 0.1
	}
	options := pacf.Options{MaxLag: 60}
	fullChecks := observeContextPolls(t, x, options)
	ctx := &checkedCancelContext{Context: context.Background(), threshold: fullChecks / 3}
	result, err := pacf.Compute(ctx, x, options)
	mustNilState(t, result, true, "interrupted covariance scan")
	mustErrorIs(t, err, context.Canceled)
}

// TestPACF_ContextCancellationInsideLargeCoefficientRows exercises the tail of
// expensive Durbin-Levinson work, including private coefficient publication.
// It derives the tail checkpoints from a completed computation instead of
// freezing an internal loop implementation into the public test contract.
func TestPACF_ContextCancellationInsideLargeCoefficientRows(t *testing.T) {
	const sampleCount = 1030
	const highestLag = 1025
	const tailCheckpoints = 3
	x := make([]float64, sampleCount)
	for i := range x {
		x[i] = math.Sin(float64(i)*0.17) + 0.3*math.Cos(float64(i)*0.23)
	}
	options := pacf.Options{MaxLag: highestLag}
	fullChecks := observeContextPolls(t, x, options)
	if fullChecks <= tailCheckpoints {
		t.Fatal("full computation must offer more context polls than tail checkpoint count")
	}

	for distanceFromEnd := 0; distanceFromEnd < tailCheckpoints; distanceFromEnd++ {
		ctx := &checkedCancelContext{Context: context.Background(), threshold: fullChecks - distanceFromEnd}
		result, err := pacf.Compute(ctx, x, options)
		mustNilState(t, result, true, "canceled high-order recurrence")
		mustErrorIs(t, err, context.Canceled)
	}
}
