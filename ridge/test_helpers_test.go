// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge_test provides typed, reflection-free contract assertions.
// Helpers deliberately assert mathematical values and error identities, not text.
package ridge_test

import (
	"context"
	"errors"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ridge"
)

// dense builds an owning rectangular Matrix fixture with fully checked writes.
// A malformed fixture fails immediately in the calling test goroutine.
func dense(t *testing.T, values [][]float64) *matrix.Dense {
	t.Helper()
	if len(values) == 0 || len(values[0]) == 0 {
		t.Fatal("empty matrix fixture")
	}
	result, err := matrix.NewDense(len(values), len(values[0]))
	if err != nil {
		t.Fatal(err)
	}
	for i, row := range values {
		if len(row) != len(values[0]) {
			t.Fatal("ragged matrix fixture")
		}
		for j, value := range row {
			if err = result.Set(i, j, value); err != nil {
				t.Fatal(err)
			}
		}
	}
	return result
}

// closeTo checks a scale-aware floating point equality for ordinary model fixtures.
// Exact-binary rational oracle tests use the stricter assertRationalClose instead.
func closeTo(t *testing.T, got, want float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-want) > 1e-9*(1+math.Abs(want)) {
		t.Fatalf("got %.17g want %.17g", got, want)
	}
}

// mustErrorIs checks sentinel ancestry without depending on diagnostic wording.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("error=%v, want errors.Is(_, %v)", err, target)
	}
}

// errorIs retains the existing test helper spelling while sharing one assertion.
func errorIs(t *testing.T, err, target error) { mustErrorIs(t, err, target) }

// isNilLike reports recognized typed-nil states without runtime reflection.
// Additional custom fixture types must explicitly implement core.Nilable.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if candidate, ok := value.(core.Nilable); ok {
		return candidate.IsNil()
	}
	switch typed := value.(type) {
	case *ridge.Model:
		return typed == nil
	default:
		return false
	}
}

// mustNilState checks typed-nil semantics through the supported Nilable protocol.
func mustNilState(t *testing.T, value any, wantNil bool, operation string) {
	t.Helper()
	if actual := isNilLike(value); actual != wantNil {
		t.Fatalf("%s: nil state=%t want=%t", operation, actual, wantNil)
	}
}

// exerciseRidgeCancellation checks atomic publication at every observed context poll.
// The baseline execution discovers the finite checkpoint count without hard-coding
// implementation-specific call numbers or relying on goroutine scheduling.
// Implementation:
//   - Stage 1: run a successful deterministic operation with a counting context.
//   - Stage 2: rerun once for each poll, canceling the underlying context at that poll.
//   - Stage 3: require nil publication and preservation of context.Canceled.
//
// Behavior highlights: the operation and its input must be replayable and read-only.
// Inputs: t is the owning test; operation reports whether any result was published.
// Returns: no value; a failure identifies the exact checkpoint.
// Errors: only context.Canceled is accepted on the interrupted executions.
// Determinism: poll numbers increase from one to the successful execution's count.
// Complexity: O(k*T) time for k polls and one operation of cost T; O(1) extra state.
// Notes: the existing cancelAfterPollContext is used only from the calling goroutine.
// AI-Hints: do not use time.Sleep or assume a fixed number of dependency context polls.
func exerciseRidgeCancellation(t *testing.T, operation func(context.Context) (bool, error)) {
	t.Helper()
	run := func(cancelAt int) (bool, error, int) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		probe := &cancelAfterPollContext{Context: ctx, cancel: cancel, cancelAt: cancelAt}
		published, err := operation(probe)
		return published, err, probe.checks
	}

	// Zero is outside the one-based cancellation range and measures a successful run.
	published, err, checkpoints := run(0)
	if err != nil || !published || checkpoints == 0 {
		t.Fatalf("cancellation baseline: published=%t err=%v polls=%d", published, err, checkpoints)
	}
	for poll := 1; poll <= checkpoints; poll++ {
		// Every execution has a fresh context; canceled contexts are never reused.
		published, err, observed := run(poll)
		if published || !errors.Is(err, context.Canceled) {
			t.Fatalf("cancel at poll %d/%d (observed %d): published=%t err=%v", poll, checkpoints, observed, published, err)
		}
	}
}

// exactAffinePrediction evaluates the returned physical model with exact binary rationals.
// It is independent of the production compensated sum and rounds only the final result.
// Implementation:
//   - Stage 1: lift the published intercept and each finite coefficient to big.Rat.
//   - Stage 2: sum exact products in the original feature order.
//   - Stage 3: round once to binary64, including infinity for an out-of-range result.
//
// Behavior highlights: no intermediate binary64 product or sum can overflow.
// Inputs: beta and features are matching finite vectors; intercept is finite.
// Returns: the independently rounded affine value.
// Errors: invalid test fixture dimensions fail the owning test.
// Determinism: ascending feature order and nearest-even conversion.
// Complexity: O(p) rational operations; storage depends on operand bit widths.
// Notes: use fitted, rounded coefficients rather than ideal real-number coefficients.
// AI-Hints: never compare a cancellation-sensitive prediction with an ordinary dot product.
func exactAffinePrediction(t *testing.T, beta []float64, intercept float64, features []float64) float64 {
	t.Helper()
	if len(beta) != len(features) {
		t.Fatal("affine oracle: mismatched feature count")
	}
	var sum, slope, feature, product big.Rat
	sum.SetFloat64(intercept)
	for j, coefficient := range beta {
		slope.SetFloat64(coefficient)
		feature.SetFloat64(features[j])
		product.Mul(&slope, &feature)
		sum.Add(&sum, &product)
	}
	value, _ := sum.Float64()
	return value
}

// scalarOriginDiagnostics derives exact diagnostics for one unscaled observation.
// The independent closed form is beta=x*y/(x*x+lambda), not a QR implementation.
// Implementation:
//   - Stage 1: lift the finite binary64 fixture to exact rational numbers.
//   - Stage 2: evaluate the closed-form optimum and its original-coordinate residual.
//   - Stage 3: compute SSE, penalty and objective without intermediate float64 rounding.
//
// Behavior highlights: can distinguish finite components from an overflowing sum.
// Inputs: finite x and y, strictly positive finite lambda; one training observation.
// Returns: three separately owned exact rational diagnostics.
// Errors: none for the documented fixture domain.
// Determinism: exact arithmetic in a fixed expression order.
// Complexity: O(1) rational operations for one scalar observation.
// Notes: normal equations here are an independent test oracle, never the fit algorithm.
// AI-Hints: do not materialize y*y as float64 when checking diagnostic boundaries.
func scalarOriginDiagnostics(x, y, lambda float64) (sse, penalty, objective *big.Rat) {
	feature := new(big.Rat).SetFloat64(x)
	response := new(big.Rat).SetFloat64(y)
	weight := new(big.Rat).SetFloat64(lambda)
	denominator := new(big.Rat).Mul(feature, feature)
	denominator.Add(denominator, weight)
	beta := new(big.Rat).Mul(feature, response)
	beta.Quo(beta, denominator)
	residual := new(big.Rat).Mul(feature, beta)
	residual.Sub(response, residual)
	sse = new(big.Rat).Mul(residual, residual)
	penalty = new(big.Rat).Mul(beta, beta)
	penalty.Mul(penalty, weight)
	penalty.Quo(penalty, big.NewRat(2, 1))
	objective = new(big.Rat).Quo(sse, big.NewRat(2, 1))
	objective.Add(objective, penalty)
	return sse, penalty, objective
}
