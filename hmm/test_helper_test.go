// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm_test supplies shared public-contract test fixtures and assertions.
// Helpers reject invalid numeric comparisons and exercise cancellation without timing races.
package hmm_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/big"
	"sync/atomic"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/hmm"
	"github.com/lvlath/go/matrix"
)

// dense materializes a rectangular fixture using the actual Matrix constructor.
// Every value is checked by Set and setup failures stop the current test.
//
// Implementation:
//   - Stage 1: verify the fixture has a defined rectangular shape.
//   - Stage 2: allocate a Dense of the exact fixture extent.
//   - Stage 3: fill row-major cells and return the owning matrix.
//
// Behavior highlights:
//   - Rejects ragged input before allocation; never truncates or pads fixture data.
//
// Inputs:
//   - t: active test or benchmark; values: nonempty rectangular row slice.
//
// Returns:
//   - matrix.Matrix: independent finite fixture storage.
//
// Errors:
//   - Shape, allocation, and Set failures stop the test through testing.TB.
//
// Determinism:
//   - Rows and columns are filled in increasing index order.
//
// Complexity:
//   - Time O(R*C+R), Space O(R*C).
//
// Notes:
//   - A deliberately invalid logical matrix belongs in a probe, not a failed fixture setup.
//
// AI-Hints:
//   - Keep production fixture construction outside the measured benchmark loop.
func dense(t testing.TB, values [][]float64) matrix.Matrix {
	t.Helper()
	if len(values) == 0 {
		t.Fatal("dense fixture requires at least one row")
	}
	cols := len(values[0])
	for i, row := range values {
		if len(row) != cols {
			t.Fatalf("ragged fixture row %d: got %d columns, want %d", i, len(row), cols)
		}
	}
	m, e := matrix.NewZeros(len(values), cols)
	if e != nil {
		t.Fatal(e)
	}
	for i, r := range values {
		for j, v := range r {
			if e = m.Set(i, j, v); e != nil {
				t.Fatal(e)
			}
		}
	}
	return m
}

// categorical constructs the existing two-state absorbing reference model.
// Its unequal emissions distinguish observation evidence without transition uncertainty.
//
// Implementation:
//   - Stage 1: create equal prior weights, identity transitions, and two emission rows.
//   - Stage 2: delegate validation and ownership to NewCategorical.
//   - Stage 3: stop on setup error or return the immutable model.
//
// Behavior highlights:
//   - Both symbols are supported by both states; state persistence is exact.
//
// Inputs:
//   - t: active test or benchmark.
//
// Returns:
//   - *hmm.CategoricalModel: non-nil reference model.
//
// Errors:
//   - Unexpected construction failures stop the current test.
//
// Determinism:
//   - All fixture probabilities and their index order are fixed.
//
// Complexity:
//   - O(1) time and space for the fixed two-state, two-symbol model.
//
// Notes:
//   - This fixture is not suitable for testing an unsupported symbol within the alphabet.
//
// AI-Hints:
//   - Use an explicit zero-emission model when checking ErrImpossibleSequence.
func categorical(t testing.TB) *hmm.CategoricalModel {
	t.Helper()
	m, e := hmm.NewCategorical(hmm.CategoricalSpec{Common: hmm.Common{Prior: []float64{.5, .5}, Transition: dense(t, [][]float64{{1, 0}, {0, 1}})}, Emission: dense(t, [][]float64{{.9, .1}, {.1, .9}})})
	if e != nil {
		t.Fatal(e)
	}
	return m
}

// testComparisonRelativeTolerance is the existing finite-value comparison tolerance.
// Individual high-accuracy oracles select their tighter limits explicitly.
const testComparisonRelativeTolerance = 1e-10

// approx compares two finite results using the existing absolute/relative scale.
// A NaN must fail explicitly because ordinary ordered comparisons with NaN are false.
//
// Implementation:
//   - Stage 1: reject nonfinite actual or expected values.
//   - Stage 2: calculate a reference-scaled absolute error bound.
//   - Stage 3: stop the test when the error exceeds the bound.
//
// Behavior highlights:
//   - Preserves the existing tolerance; no numerical admission policy is relaxed.
//
// Inputs:
//   - t: active test or benchmark; a: actual value; b: finite expected value.
//
// Returns:
//   - None; a failed comparison stops the current test.
//
// Errors:
//   - Nonfinite values and excessive finite differences fail through testing.TB.
//
// Determinism:
//   - One scalar comparison with a fixed reference scale.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Test intentional infinity sentinels separately with math.IsInf.
//
// AI-Hints:
//   - Never let NaN pass an absolute-error assertion merely because NaN > tolerance is false.
func approx(t testing.TB, a, b float64) {
	t.Helper()
	if !finite(a) || !finite(b) || math.Abs(a-b) > testComparisonRelativeTolerance*math.Max(1, math.Abs(b)) {
		t.Fatalf("got %.17g expected %.17g", a, b)
	}
}

// finite reports whether a scalar lies in the finite binary64 domain.
// Both infinities and NaN are excluded from ordinary numeric assertions.
// Complexity: Time O(1), Space O(1).
func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

// errAdmissionRead identifies a deliberate read failure in a caller-owned Matrix.
// Tests match it through errors.Is, rather than inspecting dependency error text.
var errAdmissionRead = errors.New("hmm_test: observation source unavailable")

// errAdmissionWrite identifies an unexpected write to a read-only test fixture.
// HMM must not invoke Set on caller-owned observation or parameter matrices.
var errAdmissionWrite = errors.New("hmm_test: read-only matrix")

// admissionProbeMatrix supplies small logical shapes without allocating matrix storage.
// It exposes access counters and optional read failures for validation-order tests.
//
// Implementation:
//   - Stage 1: expose the caller-selected stable shape and scalar value.
//   - Stage 2: record At and Set calls; return the configured error for failed reads.
//   - Stage 3: allow Clone to return an independently counted fixture.
//
// Behavior highlights:
//   - No large buffer, global state, reflection, or intentional panic is needed.
//   - Set always rejects mutation, including attempts to write the original value.
//
// Inputs:
//   - rows, cols: logical extents; zero or negative extents are used for admission tests.
//   - value: the value returned for a successful read.
//   - readErr: optional source-owned sentinel returned by At.
//
// Returns:
//   - A matrix.Matrix-compatible fixture with explicit Nilable behavior.
//
// Errors:
//   - At preserves readErr and rejects out-of-range coordinates.
//   - Set returns errAdmissionWrite.
//
// Determinism:
//   - Every valid read has the same scalar result; counters follow call order.
//
// Complexity:
//   - Time O(1), Space O(1) per operation.
//
// Notes:
//   - The fixture is for single-goroutine admission tests, not concurrent writers.
//
// AI-Hints:
//   - Use real Dense matrices for ordinary numerical fixtures.
//   - Do not allocate huge backing arrays merely to test shape rejection.
type admissionProbeMatrix struct {
	rows, cols int
	value      float64
	readErr    error
	reads      int
	writes     int
}

var (
	_ matrix.Matrix = (*admissionProbeMatrix)(nil)
	_ core.Nilable  = (*admissionProbeMatrix)(nil)
)

// Rows returns the stable logical row count.
// The count is independent of physical storage, which this fixture never allocates.
// Complexity: Time O(1), Space O(1).
func (m *admissionProbeMatrix) Rows() int { return m.rows }

// Cols returns the stable logical column count.
// Shape queries do not count as reads of numerical cells.
// Complexity: Time O(1), Space O(1).
func (m *admissionProbeMatrix) Cols() int { return m.cols }

// IsNil implements the typed-nil convention without dereferencing the receiver.
// A non-nil fixture with zero rows or columns remains a non-nil matrix.
// Complexity: Time O(1), Space O(1).
func (m *admissionProbeMatrix) IsNil() bool { return m == nil }

// At records a cell read and returns the configured scalar or source error.
// The counter detects reads that should have been prevented by shape validation.
//
// Implementation:
//   - Stage 1: record the attempted read.
//   - Stage 2: validate the requested coordinates against the logical extent.
//   - Stage 3: preserve the configured source error or return the scalar value.
//
// Behavior highlights:
//   - Never accesses a backing slice.
//
// Inputs:
//   - row, col: zero-based matrix coordinates.
//
// Returns:
//   - float64, error: the fixed scalar and optional source error.
//
// Errors:
//   - matrix.ErrOutOfRange: coordinate outside the logical shape.
//   - readErr: configured dependency failure at a valid coordinate.
//
// Determinism:
//   - One counter increment per call, regardless of success.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Only the fixture's counter changes; the observation value stays immutable.
//
// AI-Hints:
//   - Assert both errors.Is and the expected read count at admission boundaries.
func (m *admissionProbeMatrix) At(row, col int) (float64, error) {
	m.reads++
	if row < 0 || row >= m.rows || col < 0 || col >= m.cols {
		return 0, matrix.ErrOutOfRange
	}
	return m.value, m.readErr
}

// Set records and rejects a write to the read-only fixture.
// The original value remains intact, including when a caller ignores the error.
// Errors: errAdmissionWrite on every call.
// Complexity: Time O(1), Space O(1).
func (m *admissionProbeMatrix) Set(int, int, float64) error {
	m.writes++
	return errAdmissionWrite
}

// Clone returns an independent fixture with identical shape and read policy.
// Access counters belong to each individual fixture and restart at zero.
// Complexity: Time O(1), Space O(1).
func (m *admissionProbeMatrix) Clone() matrix.Matrix {
	if m == nil {
		return (*admissionProbeMatrix)(nil)
	}
	return &admissionProbeMatrix{rows: m.rows, cols: m.cols, value: m.value, readErr: m.readErr}
}

// requireAdmissionAccess checks read counts and verifies the absence of writes.
// It reports the dependency-boundary behavior rather than inspecting HMM internals.
//
// Implementation:
//   - Stage 1: mark this function as a test helper.
//   - Stage 2: compare the observed At count with the expected admission count.
//   - Stage 3: reject any mutation attempt.
//
// Inputs:
//   - t: active test or subtest; m: the exact fixture passed to HMM.
//   - reads: expected number of numerical cell access attempts.
//
// Returns:
//   - None; the current test fails if a counter differs.
//
// Errors:
//   - None returned; unexpected accesses fail through testing.TB.
//
// Determinism:
//   - Compares counters from a completed synchronous public call.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Do not count shape metadata access as a cell read.
//
// AI-Hints:
//   - Assert reads=0 for rejected shapes and reads=1 for a first-cell source failure.
func requireAdmissionAccess(t testing.TB, m *admissionProbeMatrix, reads int) {
	t.Helper()
	if m.reads != reads || m.writes != 0 {
		t.Fatalf("matrix access: reads=%d want=%d, writes=%d want=0", m.reads, reads, m.writes)
	}
}

// requireHMMFailure checks a classified failure with no published result.
// It accepts typed-nil result pointers through the actual core.Nilable interface.
//
// Implementation:
//   - Stage 1: match the expected sentinel using errors.Is.
//   - Stage 2: reject a non-nil result using the Nilable contract.
//   - Stage 3: keep arithmetic overflow separate from absent model support.
//
// Behavior highlights:
//   - Never uses reflection or error-string comparisons.
//
// Inputs:
//   - t: active test; result: public HMM result or typed nil.
//   - err, want: actual error and required non-nil sentinel.
//
// Returns:
//   - None; the test stops on a contract violation.
//
// Errors:
//   - Failure is reported through testing.TB, not by returning another error.
//
// Determinism:
//   - Depends only on the completed public call.
//
// Complexity:
//   - O(error-chain length) time, O(1) additional space for ordinary wrapped errors.
//
// Notes:
//   - Do not use for the documented partial-fit results on nonconvergence or rollback.
//
// AI-Hints:
//   - A typed-nil interface is not necessarily equal to nil; use IsNil.
func requireHMMFailure(t testing.TB, result core.Nilable, err, want error) {
	t.Helper()
	if !errors.Is(err, want) {
		t.Fatalf("error=%v, want errors.Is(%v)", err, want)
	}
	if result != nil && !result.IsNil() {
		t.Fatalf("unexpected published result on failure: %T", result)
	}
	if errors.Is(want, hmm.ErrOverflow) && (errors.Is(err, hmm.ErrImpossibleSequence) || errors.Is(err, hmm.ErrNonFinite)) {
		t.Fatalf("arithmetic overflow was also classified as absent support or invalid input: %v", err)
	}
}

// checkpointContext wraps a real cancelable context and counts cancellation polls.
// A selected poll triggers cancellation synchronously, without wall-clock sleeps.
//
// Implementation:
//   - Stage 1: embed a context.WithCancel child and retain its CancelFunc.
//   - Stage 2: count active Err calls atomically; cancel at stopAt when positive.
//   - Stage 3: delegate terminal errors and Done to the real child context.
//
// Behavior highlights:
//   - Once canceled, Err is sticky and Done is closed by context.WithCancel.
//   - stopAt=0 records checkpoints without injecting cancellation.
//
// Inputs:
//   - Context and cancel: a matching real context/cancel pair.
//   - stopAt: one-based poll to cancel, or zero to observe a complete call.
//
// Returns:
//   - A context.Context suitable for deterministic interruption testing.
//
// Errors:
//   - context.Canceled after the selected checkpoint or external cancellation.
//
// Determinism:
//   - The test sweeps every poll observed in a sequential reference execution.
//
// Complexity:
//   - Time O(1), Space O(1) per poll.
//
// Notes:
//   - This is a cancellation test double, not a production context policy.
//   - The atomic counter avoids a data race if a caller polls concurrently.
//
// AI-Hints:
//   - Do not hardcode private Forward/Backward/EM checkpoint counts in tests.
//   - Use a real context.WithCancel so Err and Done retain coherent semantics.
type checkpointContext struct {
	context.Context
	cancel context.CancelFunc
	stopAt int64
	polls  atomic.Int64
}

var _ context.Context = (*checkpointContext)(nil)

// Err injects cancellation exactly at the selected active poll.
// The underlying context owns the actual cancellation state and error identity.
//
// Implementation:
//   - Stage 1: return an existing terminal error without advancing the counter.
//   - Stage 2: count an active poll and trigger cancellation if requested.
//   - Stage 3: read and return the underlying context's current error.
//
// Inputs:
//   - Receiver: a non-nil checkpointContext with a matching CancelFunc.
//
// Returns:
//   - error: nil while active, otherwise the real cancellation error.
//
// Errors:
//   - context.Canceled for a canceled child.
//
// Determinism:
//   - Sequential polling determines the exact interruption point.
//
// Complexity:
//   - Time O(1), Space O(1) per call.
//
// Notes:
//   - No timeout, goroutine scheduling assumption, or artificial error is introduced.
//
// AI-Hints:
//   - Do not return context.Canceled without actually closing the context's Done channel.
func (c *checkpointContext) Err() error {
	if err := c.Context.Err(); err != nil {
		return err
	}
	if c.polls.Add(1) == c.stopAt {
		c.cancel()
	}
	return c.Context.Err()
}

// exerciseCancellationCheckpoints checks failure atomicity at every observed poll.
// A reference execution discovers the poll count; each replay cancels at one point.
//
// Implementation:
//   - Stage 1: run the public operation to its expected uncanceled outcome.
//   - Stage 2: replay the same immutable input at each observed cancellation poll.
//   - Stage 3: require a closed Done channel, context.Canceled, and no result.
//
// Behavior highlights:
//   - Covers initialization, forward, backward, xi, and repeated EM passes without private calls.
//
// Inputs:
//   - t: active test; run: a repeatable public HMM operation with stable caller input.
//   - referenceErr: nil or the expected documented error of the uncanceled operation.
//
// Returns:
//   - None; any unexpected publication or classification fails its checkpoint subtest.
//
// Errors:
//   - None returned; errors.Is verifies context.Canceled for interrupted calls.
//
// Determinism:
//   - Uses one synchronous replay per observed poll, with no sleeps or deadlines.
//
// Complexity:
//   - With C checkpoints and reference time W, O(C*W) upper-bound time;
//     the replays are sequential and keep only one operation's workspace live.
//
// Notes:
//   - The fixture must be small, deterministic, and complete more than one poll.
//   - Reference ErrNotConverged is acceptable only with a non-nil diagnostic result.
//
// AI-Hints:
//   - Prefer this sweep to guessing a fixed private loop counter after a refactor.
//   - It tests actual public cancellation behavior, not the existence of a particular private branch.
func exerciseCancellationCheckpoints(t *testing.T, run func(context.Context) (core.Nilable, error), referenceErr error) {
	t.Helper()
	base, cancel := context.WithCancel(context.Background())
	reference := &checkpointContext{Context: base, cancel: cancel}
	result, err := run(reference)
	polls := reference.polls.Load()
	cancel()
	if !errors.Is(err, referenceErr) || result == nil || result.IsNil() {
		t.Fatalf("invalid reference outcome: result=%T err=%v want=%v", result, err, referenceErr)
	}
	if polls < 2 {
		t.Fatalf("reference call did not reach an internal cancellation checkpoint: polls=%d", polls)
	}
	for at := int64(1); at <= polls; at++ {
		t.Run(fmt.Sprintf("poll_%d", at), func(t *testing.T) {
			base, cancel := context.WithCancel(context.Background())
			defer cancel()
			ctx := &checkpointContext{Context: base, cancel: cancel, stopAt: at}
			result, err := run(ctx)
			select {
			case <-ctx.Done():
			default:
				t.Fatal("operation returned before the selected cancellation checkpoint")
			}
			requireHMMFailure(t, result, err, context.Canceled)
		})
	}
}

// singleStateGaussianFixture creates a one-state diagonal model specification.
// A unit prior and transition remove hidden-state ambiguity from analytic tests.
// Complexity: Time O(D), Space O(D) for the detached fixture matrices.
func singleStateGaussianFixture(t testing.TB, means, variances []float64) hmm.GaussianSpec {
	t.Helper()
	return hmm.GaussianSpec{
		Common: hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})},
		Means:  dense(t, [][]float64{means}), Variances: dense(t, [][]float64{variances}),
	}
}

// singleStateCategoricalFixture creates a one-state categorical specification.
// It isolates observation support and emission MAP updates from transition learning.
// Complexity: Time O(A), Space O(A) for the detached fixture matrices.
func singleStateCategoricalFixture(t testing.TB, probabilities []float64) hmm.CategoricalSpec {
	t.Helper()
	return hmm.CategoricalSpec{
		Common:   hmm.Common{Prior: []float64{1}, Transition: dense(t, [][]float64{{1}})},
		Emission: dense(t, [][]float64{probabilities}),
	}
}

// exactPopulationVariance computes a rational one-feature, equal-weight oracle.
// It works on the exact binary64 inputs, without reproducing the HMM moment kernel.
//
// Implementation:
//   - Stage 1: sum the original finite inputs as exact rationals and divide by n.
//   - Stage 2: accumulate squared residuals from that exact rational mean.
//   - Stage 3: divide by n, preserving population rather than sample semantics.
//
// Behavior highlights:
//   - No float64 intermediate mean, square, or variance is required.
//
// Inputs:
//   - t: active test or benchmark; values: nonempty finite observation slice, read-only.
//
// Returns:
//   - *big.Rat: exact population variance for the supplied binary64 values.
//
// Errors:
//   - Invalid fixture inputs stop the test before rational arithmetic.
//
// Determinism:
//   - Exact rational accumulation in input order.
//
// Complexity:
//   - O(n) rational operations; bit complexity depends on operand widths.
//
// Notes:
//   - A test oracle only; it does not replace the production weighted E/M-step.
//
// AI-Hints:
//   - Compare the rational result with exact MaxFloat64 before rounding to float64.
func exactPopulationVariance(t testing.TB, values []float64) *big.Rat {
	t.Helper()
	if len(values) == 0 {
		t.Fatal("exact population variance requires observations")
	}
	for i, value := range values {
		if !finite(value) {
			t.Fatalf("nonfinite rational-oracle input at %d", i)
		}
	}
	count := big.NewRat(int64(len(values)), 1)
	sum := new(big.Rat)
	for _, value := range values {
		sum.Add(sum, new(big.Rat).SetFloat64(value))
	}
	mean := new(big.Rat).Quo(sum, count)
	moment := new(big.Rat)
	for _, value := range values {
		delta := new(big.Rat).Sub(new(big.Rat).SetFloat64(value), mean)
		moment.Add(moment, delta.Mul(delta, delta))
	}
	return moment.Quo(moment, count)
}

// requireUnchangedMatrix checks shape and exact binary64 contents after a call.
// It establishes read-only ownership without comparing implementation-specific fields.
//
// Inputs:
//   - t: active test; got: original caller-owned matrix; want: its expected cell values.
//
// Returns:
//   - None; shape, access, or cell differences fail the test.
//
// Complexity:
//   - Time O(rows*cols), Space O(1) excluding diagnostic strings on failure.
//
// AI-Hints:
//   - Compare input contents after both successful training and failed publication.
func requireUnchangedMatrix(t testing.TB, got matrix.Matrix, want [][]float64) {
	t.Helper()
	if len(want) == 0 || got == nil {
		t.Fatal("matrix comparison requires nonempty expected rows and a present input")
	}
	if nilable, ok := got.(core.Nilable); ok && nilable.IsNil() {
		t.Fatal("matrix comparison received typed nil")
	}
	if got.Rows() != len(want) || got.Cols() != len(want[0]) {
		t.Fatalf("matrix shape changed to %dx%d", got.Rows(), got.Cols())
	}
	for i, row := range want {
		for j, expected := range row {
			value, err := got.At(i, j)
			if err != nil || math.Float64bits(value) != math.Float64bits(expected) {
				t.Fatalf("matrix cell (%d,%d): value=%g want=%g err=%v", i, j, value, expected, err)
			}
		}
	}
}

// plainMatrixFixture exposes only the public Matrix method set, without core.Nilable.
// Embedding the interface rather than a concrete Dense exercises generic Matrix admission.
//
// Implementation:
//   - Stage 1: wrap an existing non-nil Matrix through its public interface.
//   - Stage 2: forward the Matrix methods without adding IsNil or accessing private storage.
//
// Behavior highlights:
//   - No additional mutable state or numerical transformation is introduced.
//
// Inputs:
//   - Matrix: a non-nil caller-owned matrix, stable during a call.
//
// Returns:
//   - A non-Nilable implementation of matrix.Matrix.
//
// Errors:
//   - The wrapped Matrix's original errors.
//
// Determinism:
//   - Identical coordinates and values are forwarded to the same wrapped matrix.
//
// Complexity:
//   - O(1) wrapper overhead per operation; storage and computation belong to the wrapped Matrix.
//
// Notes:
//   - Do not create this fixture around a nil or typed-nil embedded interface.
//
// AI-Hints:
//   - A valid generic Matrix need not expose concrete Dense storage or an IsNil method.
type plainMatrixFixture struct{ matrix.Matrix }

var _ matrix.Matrix = plainMatrixFixture{}
