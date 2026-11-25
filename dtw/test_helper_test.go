// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test provides contract-focused tests for the public DTW API.
// Shared assertions and controlled public-interface fixtures keep tests independent.
package dtw_test

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"testing"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/matrix"
)

func mustNoError(t *testing.T, err error, op string) {
	t.Helper()

	if err != nil {
		t.Fatalf("%s: unexpected error: %v", op, err)
	}
}

func mustErrorIs(t *testing.T, err, target error, op string) {
	t.Helper()

	if !errors.Is(err, target) {
		t.Fatalf("%s: errors.Is(err, %v)=false; got %v", op, target, err)
	}
}

func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()

	if got != want {
		t.Fatalf("%s: got %t, want %t", op, got, want)
	}
}

func mustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()

	if got != want {
		t.Fatalf("%s: got %d, want %d", op, got, want)
	}
}

func mustFloatEqual(t *testing.T, got, want, epsilon float64, op string) {
	t.Helper()

	if math.IsInf(got, 0) || math.IsInf(want, 0) {
		if got != want {
			t.Fatalf("%s: got %v, want %v", op, got, want)
		}

		return
	}

	if math.IsNaN(got) || math.IsNaN(want) {
		t.Fatalf("%s: NaN comparison is invalid: got=%v want=%v", op, got, want)
	}

	if math.Abs(got-want) > epsilon {
		t.Fatalf("%s: got %.12g, want %.12g, epsilon %.12g", op, got, want, epsilon)
	}
}

func mustFinite(t *testing.T, value float64, op string) {
	t.Helper()

	if math.IsNaN(value) || math.IsInf(value, 0) {
		t.Fatalf("%s: got non-finite value %v", op, value)
	}
}

func mustInf(t *testing.T, value float64, op string) {
	t.Helper()

	if !math.IsInf(value, 1) {
		t.Fatalf("%s: got %v, want +Inf", op, value)
	}
}

func mustPathValid(t *testing.T, path dtw.Path, n, m int, op string) {
	t.Helper()

	if len(path) == 0 {
		t.Fatalf("%s: empty path", op)
	}

	if path[0] != (dtw.Coord{I: 0, J: 0}) {
		t.Fatalf("%s: first coord got %+v, want {I:0 J:0}", op, path[0])
	}

	last := path[len(path)-1]
	if last.I != n-1 || last.J != m-1 {
		t.Fatalf("%s: last coord got %+v, want {I:%d J:%d}", op, last, n-1, m-1)
	}

	for idx := 1; idx < len(path); idx++ {
		prev := path[idx-1]
		curr := path[idx]
		di := curr.I - prev.I
		dj := curr.J - prev.J

		if di < 0 || dj < 0 {
			t.Fatalf("%s: non-monotone step %d: %+v -> %+v", op, idx, prev, curr)
		}

		if !((di == 1 && dj == 1) || (di == 1 && dj == 0) || (di == 0 && dj == 1)) {
			t.Fatalf("%s: invalid step %d: di=%d dj=%d path=%+v", op, idx, di, dj, path)
		}
	}
}

func mustDistanceMatchesPath(
	t *testing.T,
	a, b []float64,
	path dtw.Path,
	penalty float64,
	want float64,
	op string,
) {
	t.Helper()

	total := 0.0

	for idx, coord := range path {
		total += math.Abs(a[coord.I] - b[coord.J])

		if idx == 0 {
			continue
		}

		prev := path[idx-1]
		di := coord.I - prev.I
		dj := coord.J - prev.J

		if (di == 1 && dj == 0) || (di == 0 && dj == 1) {
			total += penalty
		}
	}

	mustFloatEqual(t, total, want, 1e-9, op)
}

func mustPathEqual(t *testing.T, got, want dtw.Path, op string) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s: path length got %d, want %d; got=%v want=%v", op, len(got), len(want), got, want)
	}

	for idx := range want {
		if got[idx] != want[idx] {
			t.Fatalf("%s: path[%d] got %+v, want %+v; got=%v want=%v", op, idx, got[idx], want[idx], got, want)
		}
	}
}

func mustMatrixShape(t *testing.T, got interface {
	Rows() int
	Cols() int
}, rows, cols int, op string) {
	t.Helper()

	if got == nil {
		t.Fatalf("%s: got nil matrix", op)
	}

	if got.Rows() != rows || got.Cols() != cols {
		t.Fatalf("%s: shape got %dx%d, want %dx%d", op, got.Rows(), got.Cols(), rows, cols)
	}
}

// coverageReadFaultMatrix represents a read-only Matrix backend with an injected I/O failure.
// Successful reads delegate to an unchanged source; the fixture never mutates source values.
// A fresh instance is required for each call; concurrent use is not supported.
// AI-Hints:
//   - Inject a failure, not a change of shape or numeric value between validation and compute.
//   - Use a single-cell source when targeting the second read in a two-stage adapter.
type coverageReadFaultMatrix struct {
	matrix.Matrix
	remainingReads int
	readErr        error
}

// At reads a source cell until the configured successful-read allowance is exhausted.
// The injected sentinel models a backend access failure independent of index validity.
// Implementation:
//   - Stage 1: check the remaining allowance before delegating.
//   - Stage 2: consume one allowance and read the unchanged underlying matrix.
//
// Inputs:
//   - i, j: zero-based matrix indices.
//
// Returns:
//   - The underlying value/error, or the injected readErr once the allowance is exhausted.
//
// Determinism:
//   - A fixed allowance yields a reproducible failure point for sequential readers.
//
// Complexity:
//   - Time O(1) plus the delegated At cost; Space O(1).
//
// Notes:
//   - Test-only state is local to one invocation; no global fault hooks are needed.
//
// AI-Hints:
//   - Do not use this fixture for parallel Matrix reads.
func (m *coverageReadFaultMatrix) At(i, j int) (float64, error) {
	// Return the backend failure without changing the matrix payload.
	if m.remainingReads == 0 {
		return 0, m.readErr
	}
	// Count the delegated access and preserve its error identity.
	m.remainingReads--
	return m.Matrix.At(i, j)
}

// coverageShapeMatrix supplies deliberately malformed shape metadata to adapter validators.
// Its embedded Matrix supplies the remaining interface methods; no large buffer is allocated.
// AI-Hints:
//   - Only use invalid dimensions that the adapter must reject before allocating or reading cells.
type coverageShapeMatrix struct {
	matrix.Matrix
	rows, cols int
}

// Rows returns the reported row count, including intentionally invalid values.
// This isolates shape validation from storage allocation; Time and Space O(1).
func (m coverageShapeMatrix) Rows() int { return m.rows }

// Cols returns the reported column count, including intentionally invalid values.
// This isolates shape validation from storage allocation; Time and Space O(1).
func (m coverageShapeMatrix) Cols() int { return m.cols }

// coverageCheckpointContext injects cancellation at a selected context observation.
// It wraps a real cancellable context, keeping Err and Done consistent and race-safe.
// The trigger is a test mechanism, not a promise about the number of production checks.
type coverageCheckpointContext struct {
	context.Context
	cancel       context.CancelFunc
	observations atomic.Int32
	cancelAt     int32
}

// Err observes cancellation and, at the selected checkpoint, cancels the parent.
// This makes preparation-stage cancellation testable without sleeps or timeouts.
// Implementation:
//   - Stage 1: count observations atomically.
//   - Stage 2: cancel the real context at the selected observation.
//   - Stage 3: return the underlying context error.
//
// Returns:
//   - nil before cancellation, context.Canceled afterward (or the parent's error).
//
// Complexity:
//   - Time O(1), Space O(1); cancellation is idempotent.
//
// Notes:
//   - Do not assert an exact observation count as part of the public DTW contract.
//
// AI-Hints:
//   - This fixture is for pre-dispatch tests, not deterministic parallel scheduling.
func (c *coverageCheckpointContext) Err() error {
	if c.observations.Add(1) == c.cancelAt {
		c.cancel()
	}
	return c.Context.Err()
}

// mustCoverageMatrixCell checks a matrix value through its public accessor.
// Exact comparison is appropriate for the small integer-valued ownership fixtures.
// Complexity: Time O(1) plus At, Space O(1).
func mustCoverageMatrixCell(t *testing.T, m matrix.Matrix, i, j int, want float64, op string) {
	t.Helper()
	// Reject read failures before inspecting the numeric value.
	got, err := m.At(i, j)
	mustNoError(t, err, op)
	mustFloatEqual(t, got, want, 0, op)
}

// mustCoverageSearchFailure asserts a pre-compute failure with no published ranking.
// All supplied sentinels must survive errors.Is, including joined numeric errors.
// Complexity: Time O(k) sentinel checks, Space O(1), where k=len(targets).
func mustCoverageSearchFailure(t *testing.T, result *dtw.SearchResult, err error, targets ...error) {
	t.Helper()
	if result != nil {
		t.Fatalf("pre-compute failure published a result: %+v", result)
	}
	for _, target := range targets {
		mustErrorIs(t, err, target, "Search pre-compute failure")
	}
}

// mustMatrixKernelFailure checks that an invalid adapter input publishes no result.
// It preserves every required sentinel independently, including joined Matrix causes.
//
// Implementation:
//   - Stage 1: reject any nonnil result pointer.
//   - Stage 2: verify each required error identity using the existing assertion.
//
// Inputs:
//   - t: owning test; result: typed result pointer; err: actual failure.
//   - targets: required sentinel identities, in diagnostic order.
//
// Returns:
//   - None; a contract violation fails the owning test.
//
// Determinism:
//   - Checks targets in the supplied order.
//
// Complexity:
//   - Time O(k) error identity checks, Space O(1), where k=len(targets).
//
// Notes:
//   - Use for failures whose contract requires nil, not documented partial results.
//
// AI-Hints:
//   - Do not use reflection or error-string matching for typed nil and joined errors.
func mustMatrixKernelFailure[T any](t *testing.T, result *T, err error, targets ...error) {
	t.Helper() // Attribute any failed assertion to the calling scenario.
	if result != nil {
		t.Fatalf("failed matrix kernel published a result: %+v; error=%v", result, err)
	}
	for _, target := range targets {
		mustErrorIs(t, err, target, "matrix kernel failure") // Check each error family separately.
	}
}

// coverageCorruptReadMatrix models a backend that delivers a nonfinite value after
// a successful validation read. The underlying logical source is never mutated.
//
// Behavior highlights:
//   - This deliberately faulty backend tests defense against corrupted reads,
//     not the ordinary behavior of a stable, conforming Matrix implementation.
//
// Notes:
//   - Use a fresh one-cell fixture per invocation; do not share it across workers.
//
// AI-Hints:
//   - Never manufacture a concurrent mutation of Dense to exercise this guard.
type coverageCorruptReadMatrix struct {
	matrix.Matrix
	remainingReads int
	corruptValue   float64
}

// At returns a valid source value until the read allowance has been consumed.
// It then delivers a corrupt numeric payload without a transport error.
//
// Implementation:
//   - Stage 1: delegate to the unchanged source while a read allowance remains.
//   - Stage 2: return the configured nonfinite payload after that allowance.
//
// Inputs:
//   - i, j: zero-based source indices; used only for valid coordinates.
//
// Returns:
//   - Source value and error, or the injected corrupt value with nil error.
//
// Errors:
//   - Any delegated Matrix.At error is preserved.
//
// Determinism:
//   - The fault point follows the explicitly configured sequential read allowance.
//
// Complexity:
//   - Time O(1) plus delegated At; Space O(1).
//
// Notes:
//   - Does not claim to be a stable numeric backend after fault injection.
//
// AI-Hints:
//   - Restrict the fixture to an intentional post-validation corruption test.
func (m *coverageCorruptReadMatrix) At(i, j int) (float64, error) {
	if m.remainingReads == 0 {
		return m.corruptValue, nil // Model a successful transport of an invalid numeric value.
	}
	m.remainingReads-- // Consume a read without modifying the underlying matrix.
	return m.Matrix.At(i, j)
}
