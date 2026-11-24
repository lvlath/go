// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// External test helpers centralize error identity, nilability, and finite oracles.
// They deliberately do not import reflect or an assertion framework.
package pca_test

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"testing"

	"github.com/lvlath/go/matrix"
)

// countReadsMatrix counts public Matrix.At calls without changing cell values.
// It is used only in a single-threaded contract test of reconstruction work.
type countReadsMatrix struct {
	matrix.Matrix
	reads int
}

func (m *countReadsMatrix) At(row, col int) (float64, error) {
	m.reads++
	return m.Matrix.At(row, col)
}

// fixture builds a fully owned finite matrix from a rectangular test ledger.
// An empty/ragged fixture is a test setup error, not an algorithmic observation.
func fixture(t *testing.T, rows [][]float64) *matrix.Dense {
	t.Helper()
	if len(rows) == 0 || len(rows[0]) == 0 {
		t.Fatal("empty test matrix")
	}
	m, err := matrix.NewDense(len(rows), len(rows[0]))
	if err != nil {
		t.Fatal(err)
	}
	for i, row := range rows {
		if len(row) != len(rows[0]) {
			t.Fatal("ragged fixture")
		}
		for j, v := range row {
			if err = m.Set(i, j, v); err != nil {
				t.Fatal(err)
			}
		}
	}
	return m
}

// mustErrorIs checks the stable sentinel hierarchy, including joined causes.
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("expected errors.Is(%v,%v)", err, target)
	}
}

// mustNear refuses NaN/Inf, even when the subtraction would otherwise hide them.
// The absolute threshold is intended for small well-scaled exact fixtures.
func mustNear(t *testing.T, got, want, tol float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) ||
		math.IsNaN(tol) || math.IsInf(tol, 0) || tol < 0 || math.Abs(got-want) > tol {
		t.Fatalf("got %.17g want %.17g absolute tolerance %.2g", got, want, tol)
	}
}

// mustRelativeNear uses both absolute and scale-relative budgets for a finite oracle.
// It does not turn an infinite expected result into a successful numeric comparison.
func mustRelativeNear(t *testing.T, got, want, relative, absolute float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) ||
		math.IsNaN(relative) || math.IsInf(relative, 0) || relative < 0 ||
		math.IsNaN(absolute) || math.IsInf(absolute, 0) || absolute < 0 {
		t.Fatalf("invalid finite comparison: got=%g want=%g rtol=%g atol=%g", got, want, relative, absolute)
	}
	scale := math.Max(math.Abs(got), math.Abs(want))
	bound := absolute + relative*scale
	if math.Abs(got-want) > bound {
		t.Fatalf("got %.17g want %.17g bound %.4g", got, want, bound)
	}
}

// mustCell reads one coordinate and fails only from the owner test goroutine.
func mustCell(t *testing.T, m matrix.Matrix, i, j int) float64 {
	t.Helper()
	value, err := m.At(i, j)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

// mustEqualInt requires an exact shape/rank/length for a discrete invariant.
func mustEqualInt(t *testing.T, got, want int, operation string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d want %d", operation, got, want)
	}
}

// readFaultMatrix models a read-only matrix backed by a fallible data source.
// It preserves shape, values and ownership while allowing one access to fail.
// Implementation:
//   - Stage 1: count the ordered public At calls.
//   - Stage 2: return the configured cause at the selected one-based read.
//   - Stage 3: delegate all other reads to the unchanged backing matrix.
//
// Behavior highlights: No data-dependent mutation or replacement of the solver.
// Inputs: A stable Matrix, a positive failing read index and a non-nil cause.
// Returns: Original cell values or the exact injected cause.
// Errors: The configured backend failure or a delegated At error.
// Determinism: The wrapper is owned by a single synchronous test invocation.
// Complexity: O(1) additional time and storage per access.
// Notes: This is an access-failure probe, not a model of concurrent writes.
// AI-Hints: Never use this wrapper for parallel inference on a shared counter.
type readFaultMatrix struct {
	matrix.Matrix       // Stable backing observations; not mutated by the probe.
	failAt        int   // One-based failing access; zero means no injected failure.
	reads         int   // Number of public reads observed in this invocation.
	cause         error // Upstream error whose identity must survive delegation.
}

// At returns the backing cell unless this is the selected failing access.
// The error is delivered once, so silently swallowing it cannot pass the test.
// Implementation: Count, compare with the selected index, then delegate.
// Behavior highlights: All successful reads have identical original values.
// Inputs: row and col are matrix coordinates in the backing domain.
// Returns: The unchanged cell or the original failure sentinel.
// Errors: The injected cause, or the backing Matrix.At error.
// Determinism: One fixed read sequence per invocation.
// Complexity: O(1) wrapper work plus the underlying read.
// Notes: The wrapper is intentionally not shared across goroutines.
// AI-Hints: Inject access errors rather than modifying observations after validation.
func (m *readFaultMatrix) At(row, col int) (float64, error) {
	m.reads++
	if m.reads == m.failAt {
		return 0, m.cause
	}
	return m.Matrix.At(row, col)
}

// checkpointContext schedules cancellation at a public Context.Err observation.
// Its embedded real cancelable context preserves the Done/Err consistency law.
// Implementation: Count checks atomically, cancel once at the configured check,
// and delegate context semantics to context.WithCancel.
// Behavior highlights: No wall-clock timeouts, sleeps, or scheduler assumptions.
// Inputs: A real cancelable context and a one-based cancellation checkpoint.
// Returns: Normal context behavior until the selected checkpoint, then cancellation.
// Errors: context.Canceled after cancellation; nil beforehand.
// Determinism: Each sequential PCA invocation sees a fixed check sequence.
// Complexity: O(1) additional time per Err call and O(1) state.
// Notes: A zero stopAt observes checks without requesting cancellation.
// AI-Hints: Calibrate the successful path before selecting checkpoints.
type checkpointContext struct {
	context.Context                    // Real context supplies Deadline, Done and Value.
	cancel          context.CancelFunc // Idempotent cancellation closes the real Done channel.
	checks          atomic.Int64       // Concurrent-safe count of Err observations.
	stopAt          int64              // Immutable one-based checkpoint, or zero for observation.
}

// Err observes a cancellation checkpoint and returns the real context error.
// The selected observation triggers cancellation before its error is returned.
// Implementation: Increment, conditionally cancel, then delegate to Context.Err.
// Behavior highlights: Once canceled, subsequent calls keep returning cancellation.
// Inputs: An initialized checkpointContext with a real cancellation function.
// Returns: The underlying context error, without inventing an inconsistent state.
// Errors: context.Canceled once the configured observation is reached.
// Determinism: Atomic observation ordering; tests invoke PCA synchronously.
// Complexity: O(1) extra work and storage.
// Notes: Context methods remain safe for concurrent callers.
// AI-Hints: Do not return context.Canceled before Done is actually closed.
func (c *checkpointContext) Err() error {
	if c.checks.Add(1) == c.stopAt {
		c.cancel()
	}
	return c.Context.Err()
}

// newCheckpointContext constructs a genuine cancelable checkpoint probe.
// The caller must release the returned cancellation function after each run.
// Implementation: Derive a context from Background and attach the check counter.
// Behavior highlights: The probe uses standard cancellation, not fake deadlines.
// Inputs: stopAt is a one-based observation number, or zero to count only.
// Returns: An owned checkpoint context and its cancellation function.
// Errors: None; context.Background is always non-nil.
// Determinism: Fresh counters start at zero for each call.
// Complexity: O(1) time and storage.
// Notes: Do not reuse the context between independent sweep cases.
// AI-Hints: Cancel each probe immediately after the public call returns.
func newCheckpointContext(stopAt int64) (*checkpointContext, context.CancelFunc) {
	ctx, cancel := context.WithCancel(context.Background())
	return &checkpointContext{Context: ctx, cancel: cancel, stopAt: stopAt}, cancel
}

// verifyCancellationSweep checks atomic failure at every observed checkpoint.
// It discovers the count on a successful call instead of depending on private
// loop counts or fixed source-line numbers.
// Implementation:
//   - Stage 1: run the public operation with a counting context.
//   - Stage 2: repeat with cancellation at each observed Err call.
//   - Stage 3: require context.Canceled and absence of a published artifact.
//
// Behavior highlights: Exercises late cancellation as well as entry validation.
// Inputs: A test and a deterministic public operation returning publication status.
// Returns: None; all checks fail in the owning test goroutine.
// Errors: A successful probe that publishes nothing, or any non-atomic cancellation.
// Determinism: No external clock, goroutine scheduling, or private state access.
// Complexity: O(q*T) time for q checkpoints and operation cost T; O(1) extra state.
// Notes: Use small stable fixtures because this deliberately repeats the operation.
// AI-Hints: The test does not claim cancellation inside a non-context-aware solver.
func verifyCancellationSweep(t *testing.T, run func(context.Context) (bool, error)) {
	t.Helper()
	baseline, release := newCheckpointContext(0)
	published, err := run(baseline)
	checks := baseline.checks.Load()
	release()
	if err != nil || !published || checks == 0 {
		t.Fatalf("checkpoint baseline: published=%t checks=%d err=%v", published, checks, err)
	}
	for index := int64(1); index <= checks; index++ {
		ctx, cancel := newCheckpointContext(index)
		published, err = run(ctx)
		cancel()
		if published || !errors.Is(err, context.Canceled) {
			t.Fatalf("checkpoint %d/%d: published=%t err=%v; want no result and cancellation", index, checks, published, err)
		}
	}
	t.Logf("verified %d cancellation checkpoints", checks)
}

// verifyReadFailureSweep checks that every observed backend error is propagated.
// All successful reads remain unchanged, including during repeated training passes.
// Implementation:
//   - Stage 1: discover the At count on a successful generic Matrix call.
//   - Stage 2: repeat the operation with one failing access at each index.
//   - Stage 3: verify the original cause and absence of partial output.
//
// Behavior highlights: Tests errors after the initial complete finite-value scan.
// Inputs: Stable source Matrix and a public operation returning publication status.
// Returns: None; failure diagnostics name the observed one-based read index.
// Errors: Missing original cause or a result published after a failed read.
// Determinism: Each run uses a fresh wrapper; no backing values are changed.
// Complexity: O(q*T) time for q observed reads; O(1) additional wrapper state.
// Notes: Only read failures are injected; no substitute matrix/statistics solver.
// AI-Hints: Do not assume which dependency pass owns an arbitrary read index.
func verifyReadFailureSweep(t *testing.T, source matrix.Matrix, run func(matrix.Matrix) (bool, error)) {
	t.Helper()
	baseline := &readFaultMatrix{Matrix: source}
	published, err := run(baseline)
	if err != nil || !published || baseline.reads == 0 {
		t.Fatalf("read baseline: published=%t reads=%d err=%v", published, baseline.reads, err)
	}
	cause := errors.New("observation backend: interrupted read")
	for index := 1; index <= baseline.reads; index++ {
		probe := &readFaultMatrix{Matrix: source, failAt: index, cause: cause}
		published, err = run(probe)
		if published || !errors.Is(err, cause) {
			t.Fatalf("read %d/%d: published=%t err=%v; want original cause and no result", index, baseline.reads, published, err)
		}
	}
	t.Logf("verified %d backend read failure positions", baseline.reads)
}
