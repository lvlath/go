// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap_test verifies cancellation by deterministic context polling.
// It uses no timers, random scheduling, sleeps or testing assertions in workers.
package bootstrap_test

import (
	"context"
	"testing"

	"github.com/lvlath/go/bootstrap"
)

// pollContext implements a deterministic, single-goroutine context whose Err
// transitions after a specified number of checks, making partial rows testable.
type pollContext struct {
	context.Context
	cancelAfter int
	calls       int
	done        chan struct{}
	canceled    bool
	reason      error
}

// newPollContext creates a probe that preserves the given context error identity.
func newPollContext(after int, reason error) *pollContext {
	return &pollContext{
		Context: context.Background(), cancelAfter: after, reason: reason,
		done: make(chan struct{}),
	}
}

// Done exposes the event signaled when the synthetic context changes state.
func (c *pollContext) Done() <-chan struct{} {
	return c.done
}

// Err changes state only after the observed poll count passes cancelAfter.
// A very large threshold produces an ordinary noncanceling probe.
func (c *pollContext) Err() error {
	c.calls++
	if c.calls <= c.cancelAfter {
		return nil
	}
	if !c.canceled {
		close(c.done)
		c.canceled = true
	}
	return c.reason
}

// TestSampleIndices_CancellationPartialContract checks that an interrupted
// second row is discarded while the completed first row stays byte-identical.
func TestSampleIndices_CancellationPartialContract(t *testing.T) {
	for _, kind := range []bootstrap.Kind{
		bootstrap.MovingBlocks, bootstrap.CircularBlocks, bootstrap.StationaryBlocks,
	} {
		t.Run(kindName(kind), func(t *testing.T) {
			plan := bootstrap.Plan{Kind: kind, Length: 2049, BlockLength: 7, Replicates: 3, Seed: 13}
			fullFirst := plan
			fullFirst.Replicates = 1
			probe := newPollContext(int(^uint(0)>>1), context.Canceled)
			first, err := bootstrap.SampleIndices(probe, 97, fullFirst)
			if err != nil {
				t.Fatal(err)
			}
			mustResultShape(t, first, 97, 2049, 1)

			// The first row consumes the same checks regardless of Replicates.
			// Ten additional polls guarantee that cancellation interrupts row two.
			interrupt := newPollContext(probe.calls+10, context.Canceled)
			partial, err := bootstrap.SampleIndices(interrupt, 97, plan)
			mustErrorIs(t, err, context.Canceled)
			mustResultShape(t, partial, 97, 2049, 1)
			mustSameRows(t, partial.Indices, first.Indices, "completed prefix after cancellation")
		})
	}
}

// TestSampleIndices_CancellationBeforeFirstPublication ensures the facade
// distinguishes pre-compute rejection from interrupted work with zero rows.
func TestSampleIndices_CancellationBeforeFirstPublication(t *testing.T) {
	plan := bootstrap.Plan{Kind: bootstrap.MovingBlocks, Length: 4097, BlockLength: 5, Replicates: 3, Seed: 0}
	ctx := newPollContext(2, context.DeadlineExceeded)
	partial, err := bootstrap.SampleIndices(ctx, 19, plan)
	mustErrorIs(t, err, context.DeadlineExceeded)
	mustResultShape(t, partial, 19, 4097, 0)
}

// TestSampleIndices_CancellationAtAdmission rejects a context canceled before
// the first public call without allocating or returning a partial artifact.
func TestSampleIndices_CancellationAtAdmission(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := bootstrap.SampleIndices(ctx, 19, bootstrap.Plan{
		Kind: bootstrap.MovingBlocks, Length: 1024, BlockLength: 4, Replicates: 2,
	})
	mustErrorIs(t, err, context.Canceled)
	mustNilState(t, result, true, "pre-admission cancellation")
}

// TestSampleIndices_CancellationAtFinalGate proves that a cancellation observed
// immediately after the final complete row is still returned as an error. The
// row is retained because it was fully published before the final context poll.
//
// Implementation:
//   - Stage 1: Count exactly how many context polls a successful run performs.
//   - Stage 2: Cause cancellation on the final poll with the same seed and plan.
//   - Stage 3: Require all completed rows and native context error identity.
//
// Determinism:
//   - No sleeps, timers or goroutine scheduling; both runs use the same stream IDs.
//
// AI-Hints:
//   - Never convert a fully generated but finally canceled request into nil error.
func TestSampleIndices_CancellationAtFinalGate(t *testing.T) {
	for _, kind := range []bootstrap.Kind{
		bootstrap.MovingBlocks, bootstrap.CircularBlocks, bootstrap.StationaryBlocks,
	} {
		t.Run(kindName(kind), func(t *testing.T) {
			plan := bootstrap.Plan{
				Kind: kind, Length: 57, BlockLength: 5, Replicates: 2, Seed: 20,
			}
			probe := newPollContext(int(^uint(0)>>1), context.Canceled)
			complete, err := bootstrap.SampleIndices(probe, 31, plan)
			if err != nil {
				t.Fatal(err)
			}
			mustResultShape(t, complete, 31, plan.Length, plan.Replicates)

			// The threshold defers cancellation until the final Err poll.
			interrupt := newPollContext(probe.calls-1, context.Canceled)
			partial, err := bootstrap.SampleIndices(interrupt, 31, plan)
			mustErrorIs(t, err, context.Canceled)
			mustResultShape(t, partial, 31, plan.Length, plan.Replicates)
			mustSameRows(t, partial.Indices, complete.Indices, "final gate retained rows")
		})
	}
}
