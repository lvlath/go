// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols_test defines external-only fixture builders and contract assertions.
// Helpers in this file are used by regression, oracle, and error-precedence tests.
package ols_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/big"
	"sync/atomic"
	"testing"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/ols"
)

// mustDense constructs an owning finite n-by-p fixture through the public API.
// It reports ragged rows or rejected data without silently altering dimensions.
func mustDense(t *testing.T, data [][]float64) *matrix.Dense {
	t.Helper()
	rows := len(data)
	cols := 0
	if rows > 0 {
		cols = len(data[0])
	}
	m, err := matrix.NewDense(rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	for row, values := range data {
		if len(values) != cols {
			t.Fatalf("ragged fixture at row %d: got %d columns, want %d", row, len(values), cols)
		}
		for col, value := range values {
			if err = m.Set(row, col, value); err != nil {
				t.Fatalf("fixture Set(%d,%d): %v", row, col, err)
			}
		}
	}
	return m
}

// mustNear compares numerical results using a mixed absolute/relative budget.
// A missing or nonfinite computed result cannot accidentally pass NaN comparisons.
func mustNear(t *testing.T, got, want, tolerance float64) {
	t.Helper()
	if !finiteForTest(got) || !finiteForTest(want) || math.Abs(got-want) > tolerance*math.Max(1, math.Abs(want)) {
		t.Fatalf("got %.17g, want %.17g (tolerance %.3g)", got, want, tolerance)
	}
}

// finiteForTest classifies float64 test observations without reflection.
// It prevents a NaN difference from silently satisfying a numeric assertion.
func finiteForTest(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// mustErrorIs verifies semantic error identity rather than fragile error strings.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("got error %v; want errors.Is(..., %v)", err, target)
	}
}

// mustEqualInt anchors exact dimensional or rank invariants.
func mustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %d; want %d", op, got, want)
	}
}

// mustEqualBool anchors discrete nilability/policy invariants without reflection.
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s = %v; want %v", op, got, want)
	}
}

// fixtureMatrix is a purpose-built public Matrix implementation for testing
// boundary conditions that an ordinary finite Dense cannot represent.
// In particular, its At may report an injected error or expose nonfinite data.
type fixtureMatrix struct {
	rows, cols int
	values     []float64
	readErr    error
	errRow     int
	errCol     int
	onRead     func(row, col int)
}

// IsNil implements the nil-safe core.Nilable boundary for typed-nil fixtures.
func (m *fixtureMatrix) IsNil() bool { return m == nil }

// Rows reports a fixed test-supplied row count without dereferencing nil.
func (m *fixtureMatrix) Rows() int {
	if m == nil {
		return 0
	}
	return m.rows
}

// Cols reports a fixed feature width, including intentional invalid sizes.
func (m *fixtureMatrix) Cols() int {
	if m == nil {
		return 0
	}
	return m.cols
}

// At provides controlled read faults or raw nonfinite cells through Matrix.
// The onRead observer deterministically triggers cancellation tests without sleeps.
func (m *fixtureMatrix) At(row, col int) (float64, error) {
	if m == nil {
		return 0, errors.New("fixture: nil receiver should have been rejected")
	}
	if row < 0 || col < 0 || row >= m.rows || col >= m.cols {
		return 0, matrix.ErrOutOfRange
	}
	if m.onRead != nil {
		m.onRead(row, col)
	}
	if m.readErr != nil && row == m.errRow && col == m.errCol {
		return 0, m.readErr
	}
	return m.values[row*m.cols+col], nil
}

// Set is a minimal shape-checked test adapter write, not an OLS entry point.
func (m *fixtureMatrix) Set(row, col int, value float64) error {
	if m == nil || row < 0 || col < 0 || row >= m.rows || col >= m.cols {
		return matrix.ErrOutOfRange
	}
	m.values[row*m.cols+col] = value
	return nil
}

// Clone supplies the interface-mandated independent fixture value snapshot.
func (m *fixtureMatrix) Clone() matrix.Matrix {
	if m == nil {
		return nil
	}
	clone := *m
	clone.values = append([]float64(nil), m.values...)
	return &clone
}

// mustMatrixValues compares original caller-owned values after Fit for mutation.
// It also checks shapes so a silent matrix resizing cannot go undetected.
func mustMatrixValues(t *testing.T, got matrix.Matrix, want [][]float64) {
	t.Helper()
	mustEqualInt(t, got.Rows(), len(want), "matrix rows")
	if len(want) == 0 {
		return
	}
	mustEqualInt(t, got.Cols(), len(want[0]), "matrix cols")
	for row := range want {
		for col, value := range want[row] {
			cell, err := got.At(row, col)
			if err != nil || cell != value {
				t.Fatalf("matrix changed at (%d,%d): %v, %v; want %v", row, col, cell, err, value)
			}
		}
	}
}

// exactOLSResponse is an independent oracle for the binary64 values actually
// published by the fitted Model. It avoids assuming that QR must return
// bit-for-bit textbook coefficients for a nontrivial design.
func exactOLSResponse(t *testing.T, model *ols.Model, features []float64) float64 {
	t.Helper()
	beta, intercept := model.Coefficients()
	if len(features) != len(beta) {
		t.Fatalf("oracle feature width %d, want %d", len(features), len(beta))
	}
	var total, product, featureRat, slopeRat big.Rat
	total.SetFloat64(intercept)
	for index, feature := range features {
		featureRat.SetFloat64(feature)
		slopeRat.SetFloat64(beta[index])
		product.Mul(&featureRat, &slopeRat)
		total.Add(&total, &product)
	}
	value, _ := total.Float64()
	return value
}

// readProbeMatrix wraps a stable public Matrix and observes successful read order.
// One selected read can report an upstream error or deliberately invalid value
// without changing the underlying Dense, the OLS Model, or the feature shape.
//
// Implementation:
//   - Stage 1: delegate dimensions and ownership operations to Matrix.
//   - Stage 2: count At calls and inject a single configured fault.
//   - Stage 3: forward every other read to the underlying Matrix unchanged.
//
// Behavior highlights:
//   - A zero faultAt records reads without injecting a fault.
//   - A nil faultErr with a positive faultAt returns faultValue instead.
//
// Inputs: a stable non-nil Matrix, a one-based read ordinal, and a fault.
// Returns: a test-only Matrix implementation suitable for public Fit/Predict.
// Errors: the configured upstream error or the delegated Matrix.At error.
// Determinism: reads are counted in the exact order issued by one operation.
// Complexity: O(1) per read plus underlying access; O(1) additional storage.
// Notes: use one probe per synchronous operation; do not share it among workers.
// AI-Hints: inject read failures, not changes to production solver internals.
// A nonfinite substitution tests defensive handling of an invalid backend,
// not a promise that callers may mutate their matrices during inference.
type readProbeMatrix struct {
	matrix.Matrix         // Stable source and delegated shape/ownership methods.
	reads         int     // Number of observed At calls in this operation.
	faultAt       int     // One-based fault ordinal; zero disables injection.
	faultErr      error   // Original upstream error, if one is injected.
	faultValue    float64 // Replacement cell only when faultErr is nil.
}

// IsNil reports whether this test adapter or its source is absent.
// It implements the optional core.Nilable interface without reflection.
//
// Implementation: check adapter presence, then the embedded interface.
// Behavior highlights: the receiver is safe to inspect when nil.
// Inputs: a probe created with a non-nil concrete matrix or a nil receiver.
// Returns: true for an absent adapter or absent source interface.
// Errors: none.
// Determinism: pure presence check.
// Complexity: O(1) time and space.
// Notes: test construction supplies a concrete non-nil Dense; arbitrary typed-nil sources are not supported.
// AI-Hints: do not use this fixture as a general-purpose matrix adapter.
func (m *readProbeMatrix) IsNil() bool {
	return m == nil || m.Matrix == nil
}

// At observes one public matrix read and injects a fault at a selected ordinal.
// The original source remains unchanged, including after a failed operation.
//
// Implementation:
//   - Stage 1: increment the operation-local read count.
//   - Stage 2: return the configured fault at its selected ordinal.
//   - Stage 3: otherwise delegate to the original Matrix.At.
//
// Behavior highlights: an error retains its identity for errors.Is assertions.
// Inputs: row and col are requested matrix coordinates.
// Returns: a delegated or injected scalar/error pair.
// Errors: configured faultErr or an original Matrix.At failure.
// Determinism: one increment per read, with no timing or random dependence.
// Complexity: O(1) plus delegated At; O(1) additional space.
// Notes: the fixture is deliberately single-operation and not concurrent-safe.
// AI-Hints: construct a fresh probe for every fault position.
func (m *readProbeMatrix) At(row, col int) (float64, error) {
	m.reads++ // Count even a failed read so the fault location is unambiguous.
	if m.reads == m.faultAt {
		// Preserve upstream error identity or expose an intentionally invalid value.
		return m.faultValue, m.faultErr
	}
	// All other accesses retain the same finite source data and shape.
	return m.Matrix.At(row, col)
}

// pollCancellationContext triggers real context cancellation at one observed poll.
// Its embedded context owns Done, Deadline, Value, and the sticky Err state.
//
// Implementation:
//   - Stage 1: wrap a real context.WithCancel context for one operation.
//   - Stage 2: count Err observations with an atomic counter.
//   - Stage 3: close the real cancellation context at the selected observation.
//
// Behavior highlights: the zero cancelAt value only records observations.
// Inputs: a cancellable context, cancel function, and one-based poll ordinal.
// Returns: a context.Context suitable for deterministic cancellation tests.
// Errors: context.Canceled once triggered, delegated from the real context.
// Determinism: these synchronous fixtures have a fixed observable polling order.
// Complexity: O(1) per poll and O(1) auxiliary state.
// Notes: the atomic counter prevents a data race even if a dependency polls concurrently;
// the tests themselves do not depend on nondeterministic scheduling.
// AI-Hints: never use sleep-based cancellation to target an internal phase.
type pollCancellationContext struct {
	context.Context                    // Real cancellable context and standard semantics.
	cancel          context.CancelFunc // Closes Done and makes the cancellation error sticky.
	cancelAt        int64              // One-based cancellation ordinal; zero records only.
	polls           atomic.Int64       // Observed Err calls, including dependency polls.
}

// Err samples the context and triggers real cancellation at the selected poll.
// After cancellation the embedded context returns the same error on every call.
//
// Implementation: increment the counter, cancel if selected, delegate Err.
// Behavior highlights: Done is closed by the real context before Err returns Canceled.
// Inputs: the operation-local poll context.
// Returns: nil or the original context error.
// Errors: context.Canceled after the selected poll.
// Determinism: the ordinal is observed without timers or sleeps.
// Complexity: O(1) per poll; O(1) extra space.
// Notes: a zero cancelAt does not trigger cancellation.
// AI-Hints: do not return a fabricated non-nil Err while Done remains open.
func (c *pollCancellationContext) Err() error {
	// Record the exact poll before deciding whether cancellation is due.
	if c.polls.Add(1) == c.cancelAt {
		c.cancel()
	}
	// The embedded context, not the counter, determines the error identity.
	return c.Context.Err()
}

// exerciseCancellation verifies atomic failure at every observed cancellation poll.
// A successful baseline supplies the actual number of polls, avoiding private
// constants, brittle hard-coded call counts, timers, or scheduler assumptions.
//
// Implementation:
//   - Stage 1: execute one successful baseline with cancellation disabled.
//   - Stage 2: replay the operation with a fresh context canceled at each poll.
//   - Stage 3: require context.Canceled, closed Done, and no published result.
//
// Behavior highlights: cancellation is injected inside the public API and its dependencies.
// Inputs: t and a synchronous operation returning result presence plus error.
// Returns: none; assertion failures identify the selected cancellation ordinal.
// Errors: test failure if the baseline or any cancellation contract is violated.
// Determinism: increasing ordinals, independent real contexts, no sleeping.
// Complexity: O(C*T) time for C observed polls and one operation cost T;
// O(1) harness state per replay beyond the operation's own allocations.
// Notes: intended for small deterministic fixtures; never reuse a canceled context.
// AI-Hints: preserve nil-result assertions, not just cancellation error identity.
func exerciseCancellation(t *testing.T, operation func(context.Context) (bool, error)) {
	t.Helper()
	// Establish a successful run and record every actual context observation.
	parent, cancel := context.WithCancel(context.Background())
	baseline := &pollCancellationContext{Context: parent, cancel: cancel}
	published, err := operation(baseline)
	polls := baseline.polls.Load()
	cancel() // Release the baseline context before creating independent replays.
	if err != nil || !published || polls == 0 {
		t.Fatalf("cancellation baseline: published=%t polls=%d err=%v", published, polls, err)
	}

	// Replay every checkpoint, including late publication checks after valid work.
	for ordinal := int64(1); ordinal <= polls; ordinal++ {
		t.Run(fmt.Sprintf("poll_%d", ordinal), func(t *testing.T) {
			parent, cancel := context.WithCancel(context.Background())
			defer cancel()
			probe := &pollCancellationContext{Context: parent, cancel: cancel, cancelAt: ordinal}
			published, err := operation(probe)
			if published {
				t.Fatalf("poll %d published a partial or complete result after cancellation", ordinal)
			}
			mustErrorIs(t, err, context.Canceled)
			// Inspect the real parent, without introducing another observed poll.
			mustErrorIs(t, parent.Err(), context.Canceled)
			select {
			case <-parent.Done():
				// Cancellation is real: the Done channel is closed.
			default:
				t.Fatal("cancellation error returned while Done remained open")
			}
		})
	}
}

// exerciseReadFaults verifies that every public matrix read preserves its failure.
// A successful baseline discovers the read schedule, including validation,
// preprocessing, ordinary inference, and exact-fallback rereads when present.
//
// Implementation:
//   - Stage 1: record one successful operation with a stable finite matrix.
//   - Stage 2: replay with a distinct read probe failing at each observed read.
//   - Stage 3: require the original upstream sentinel and a nil public result.
//
// Behavior highlights: source data never change, even during injected failures.
// Inputs: a stable input Matrix and a synchronous operation over that Matrix.
// Returns: none; test failures identify the one-based read ordinal.
// Errors: test failure if error identity or result atomicity is lost.
// Determinism: increasing fault ordinals, a fresh adapter for each run.
// Complexity: O(R*T) time for R observed reads and one operation cost T;
// O(1) probe storage per replay beyond the operation's own allocations.
// Notes: use small full-rank matrices to keep exhaustive replays inexpensive.
// AI-Hints: no fake QR output, internal solver replacement, or global test hooks.
func exerciseReadFaults(t *testing.T, source matrix.Matrix, operation func(matrix.Matrix) (bool, error)) {
	t.Helper()
	// Record the read schedule from the same successful public operation.
	baseline := &readProbeMatrix{Matrix: source}
	published, err := operation(baseline)
	if err != nil || !published || baseline.reads == 0 {
		t.Fatalf("read baseline: published=%t reads=%d err=%v", published, baseline.reads, err)
	}

	// A wrapping error also verifies errors.Is rather than fragile text equality.
	root := errors.New("upstream observation unavailable")
	for ordinal := 1; ordinal <= baseline.reads; ordinal++ {
		t.Run(fmt.Sprintf("read_%d", ordinal), func(t *testing.T) {
			probe := &readProbeMatrix{
				Matrix:   source,
				faultAt:  ordinal,
				faultErr: fmt.Errorf("observation read %d: %w", ordinal, root),
			}
			published, err := operation(probe)
			if published {
				t.Fatalf("read %d failure published a result", ordinal)
			}
			mustErrorIs(t, err, root)
			mustErrorIs(t, err, probe.faultErr)
		})
	}
}

// mustUnitSlopeModel fits an exact three-column identity system through OLS.
// This fixture isolates prediction arithmetic from generic QR rounding effects.
//
// Implementation: fit I3 with response [1,1,1] and no intercept, then check parameters.
// Behavior highlights: coefficients are exact, not rounded by the test.
// Inputs: the test handle; the fixture is fixed and has full column rank.
// Returns: a successfully fitted immutable Model with slopes [1,1,1] and b=+0.
// Errors: test failure if fitting or the exact identity invariant fails.
// Determinism: fixed identity basis and response order.
// Complexity: O(1) for this fixed-size fixture, including its QR solve.
// Notes: no hidden Model construction or access to private fields.
// AI-Hints: exact equality is justified here by the binary identity system only.
func mustUnitSlopeModel(t *testing.T) *ols.Model {
	t.Helper()
	// The identity basis makes both rank and the exact solution unambiguous.
	x := mustDense(t, [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})
	fit, err := ols.FitThroughOrigin(context.Background(), x, []float64{1, 1, 1})
	if err != nil || fit == nil || fit.Model == nil {
		t.Fatalf("identity fit failed: %v, %v", fit, err)
	}
	beta, intercept := fit.Model.Coefficients()
	if len(beta) != 3 || beta[0] != 1 || beta[1] != 1 || beta[2] != 1 || intercept != 0 || math.Signbit(intercept) {
		t.Fatalf("identity fit: beta=%v intercept=%g; want [1 1 1], +0", beta, intercept)
	}
	return fit.Model
}
