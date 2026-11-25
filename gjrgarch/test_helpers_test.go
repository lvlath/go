// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test defines transparent, reflect-free UES contract assertions.
// Failure messages describe the failing semantic assertion rather than field dumps.
package gjrgarch_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/gjrgarch"
)

// mustErrorIs enforces sentinel identity through the canonical errors.Is protocol.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("error %v does not wrap sentinel %v", err, target)
	}
}

// nilState is the tiny, reflection-free subset used by core.Nilable results.
type nilState interface{ IsNil() bool }

// isNilLike handles nil interfaces and typed-nil values implementing IsNil.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	nilable, ok := value.(nilState)
	return ok && nilable.IsNil()
}

// mustNilState asserts the exact typed-nil interpretation expected by UES.
func mustNilState(t *testing.T, value any, wantNil bool, operation string) {
	t.Helper()
	if got := isNilLike(value); got != wantNil {
		t.Fatalf("%s: nil-state got=%t want=%t", operation, got, wantNil)
	}
}

// mustEqualBool validates one explicit Boolean contract claim.
func mustEqualBool(t *testing.T, got, want bool, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %t want %t", operation, got, want)
	}
}

// mustEqualInt validates an exact count such as Steps or iterations.
func mustEqualInt(t *testing.T, got, want int, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d want %d", operation, got, want)
	}
}

// mustNearRelative asserts a numerical value using a scale-aware error budget.
func mustNearRelative(t *testing.T, got, want, tolerance float64, operation string) {
	t.Helper()
	if !finiteTest(got) || math.Abs(got-want) > tolerance*math.Max(1, math.Abs(want)) {
		t.Fatalf("%s: got %.17g want %.17g with tolerance %.3g", operation, got, want, tolerance)
	}
}

// finiteTest avoids importing internal numeric validators in the external package.
func finiteTest(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

// pollCancellationContext counts synchronous cancellation polls while retaining
// the standard context contract: Err and Done agree after cancellation.
// The counter belongs to one synchronous Fit or evaluation call; no goroutine
// or wall-clock deadline is used. A zero stopAt only records the poll count.
type pollCancellationContext struct {
	context.Context
	cancel context.CancelFunc
	stopAt int
	polls  int
}

// Err cancels the embedded context at the chosen poll and delegates its result.
// Complexity: O(1) per poll; cancellation is monotonic.
func (c *pollCancellationContext) Err() error {
	c.polls++
	if c.stopAt > 0 && c.polls == c.stopAt {
		c.cancel()
	}
	return c.Context.Err()
}

// newPollCancellationContext creates a real, caller-owned cancellable context.
// Register cleanup even on test failure so no cancellation resources are retained.
func newPollCancellationContext(t *testing.T, stopAt int) *pollCancellationContext {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	return &pollCancellationContext{Context: ctx, cancel: cancel, stopAt: stopAt}
}

// referenceSingletonLoss evaluates the one-observation density independently.
// With no following observation there is no lagged recursion or gradient state.
// Complexity: O(1) time and space, for moderate finite test fixtures only.
func referenceSingletonLoss(p gjrgarch.Parameters, observation float64) float64 {
	rho := p.Alpha + p.Beta + p.Gamma/2
	h := p.Omega / (1 - rho)
	epsilon := observation - p.Mean
	return 0.5 * (math.Log(2*math.Pi) + math.Log(h) + epsilon*epsilon/h)
}
