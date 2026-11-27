// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap_test protects default-wrapper parity without private production access.
// This file compares values, errors, ownership, and exact context polling behavior.
// AI-Hints: a convenience wrapper must not synthesize reset[0] or add validation passes.
package vwap_test

import (
	"context"
	"testing"

	"github.com/lvlath/go/vwap"
)

// TestComputeDefault_Parity verifies that the wrapper changes only Options{} boilerplate.
// Every successful result is owning; every rejection retains the canonical error identities.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: run canonical and default facades on the same fixed cases.
//   - Stage 3: require independent owning results and unchanged sentinel classification.
//
// Behavior highlights:
//   - This test protects default-wrapper numerical, ownership, and error parity without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - A default facade must not manufacture reset[0] or discard the mask.
func TestComputeDefault_Parity(t *testing.T) {
	for _, fixture := range wrapperFixtures() {
		t.Run(fixture.name, func(t *testing.T) {
			canonical, err := vwap.Compute(fixture.ctx, fixture.price, fixture.volume, fixture.reset, vwap.Options{})
			wrapped, wrapperErr := vwap.ComputeDefault(fixture.ctx, fixture.price, fixture.volume, fixture.reset)
			if fixture.want != nil {
				requireError(t, err, fixture.want)
				requireError(t, wrapperErr, fixture.want)
				if canonical != nil || wrapped != nil {
					t.Fatal("error returned a partial result")
				}
				return
			}
			if err != nil || wrapperErr != nil {
				t.Fatalf("canonical=%v wrapper=%v", err, wrapperErr)
			}
			want := rationalOracle(fixture.price, fixture.volume, fixture.reset)
			requireResult(t, canonical, want)
			requireResult(t, wrapped, want)
			if wrapped.Values == nil || wrapped.Valid == nil {
				t.Fatal("successful empty output has nil channels")
			}
			if len(wrapped.Values) != 0 {
				wrapped.Values[0], wrapped.Valid[0] = 12345, !wrapped.Valid[0]
				requireResult(t, canonical, want)
			}
		})
	}
}

// TestComputeDefault_CancellationPolling prevents extra polls or validation in the wrapper.
// Separate deterministic contexts give the same cancellation boundary to each invocation.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: give canonical and wrapper calls separate identical counted contexts.
//   - Stage 3: require equal poll counts and equal cancellation or successful results.
//
// Behavior highlights:
//   - This test protects exact context-poll delegation by the default facade without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Extra wrapper validation would move observable cancellation boundaries.
func TestComputeDefault_CancellationPolling(t *testing.T) {
	price, volume, reset := benchmarkHistory(2050, 64)
	for at := 1; at <= 10; at++ {
		canonicalContext, wrapperContext := newCountedContext(at), newCountedContext(at)
		canonical, err := vwap.Compute(canonicalContext, price, volume, reset, vwap.Options{})
		wrapped, wrapperErr := vwap.ComputeDefault(wrapperContext, price, volume, reset)
		if canonicalContext.calls != wrapperContext.calls {
			t.Fatalf("poll divergence: canonical=%d wrapper=%d", canonicalContext.calls, wrapperContext.calls)
		}
		if err != nil {
			requireError(t, wrapperErr, context.Canceled)
			if canonical != nil || wrapped != nil {
				t.Fatal("cancellation returned a partial result")
			}
		} else {
			if wrapperErr != nil {
				t.Fatal(wrapperErr)
			}
			want := rationalOracle(price, volume, reset)
			requireResult(t, canonical, want)
			requireResult(t, wrapped, want)
		}
	}
}

// TestNewDefault_Lifecycle verifies construction, anchor, continuation, and reset equivalence.
// Both constructors must produce detached streams with identical exact checkpoints.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: evolve independent streams made by both constructors.
//   - Stage 3: require identical exact snapshots until one stream is intentionally reset.
//
// Behavior highlights:
//   - This test protects default construction, first-anchor, continuation, and Reset parity without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Never prime a new stream with an implicit successful observation.
func TestNewDefault_Lifecycle(t *testing.T) {
	canonical := newStream(t)
	wrapped, err := vwap.NewDefault()
	if err != nil || wrapped == nil {
		t.Fatalf("NewDefault: stream=%v error=%v", wrapped, err)
	}
	if wrapped.Snapshot() != canonical.Snapshot() {
		t.Fatal("empty snapshots differ")
	}
	value, err := wrapped.Push(context.Background(), 10, 1, false)
	requireError(t, err, vwap.ErrAnchor)
	requireValue(t, value, vwap.Value{})
	fixture := restartFixture()
	for i := range fixture.price {
		want := mustPush(t, canonical, fixture.price[i], fixture.volume[i], fixture.reset[i])
		requireValue(t, mustPush(t, wrapped, fixture.price[i], fixture.volume[i], fixture.reset[i]), want)
		if canonical.Snapshot() != wrapped.Snapshot() {
			t.Fatalf("exact checkpoint differs after row %d", i)
		}
	}
	wrapped.Reset()
	if wrapped.Snapshot() == canonical.Snapshot() {
		t.Fatal("reset one stream unexpectedly changed its independent peer")
	}
	canonical.Reset()
	if wrapped.Snapshot() != canonical.Snapshot() {
		t.Fatal("constructor parity lost after Reset")
	}
}
