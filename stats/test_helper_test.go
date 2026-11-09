// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats_test provides independent reference calculations and fixtures.
// This file does not use reflection, private stats state, or test-only exports.
// AI-Hints:
//   - Oracle arithmetic must never delegate to stats production kernels.
//   - Compare numerical outputs with magnitude-aware tolerances, not one epsilon.
package stats_test

import (
	"errors"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/stats"
)

const (
	// relativeBudget permits a small bounded error from compensated floating
	// operations, without hiding large-offset catastrophic cancellation.
	relativeBudget = 2e-12

	// subnormalBudget allows at most two representable subnormal float64 steps.
	subnormalBudget = 2
)

// mustErrorIs checks error identity without coupling to formatted messages.
// It fails the active test when errors.Is does not classify the observed error
// as the required package-level sentinel.
// Implementation: use testing.Helper and errors.Is to preserve caller location.
// Inputs: t is the current test, got the observed error, want the sentinel.
// Returns: none; aborts this test on incorrect classification.
// Determinism: exact sentinel identity comparison via errors.Is.
// Complexity: O(chain depth) error matching, O(1) local storage.
func mustErrorIs(t *testing.T, got, want error) {
	t.Helper()
	if !errors.Is(got, want) {
		t.Fatalf("error: got %v; want errors.Is(_, %v)", got, want)
	}
}

// requireNear checks a numerical result against a scale-aware oracle budget.
// The relative comparison is evaluated through division to avoid overflowing
// an error budget when the expected result approaches MaxFloat64.
// Implementation:
//   - Stage 1: require finite operands and a quick exact-equality fast path.
//   - Stage 2: permit two ULPs at the subnormal floor.
//   - Stage 3: compare relative differences using the larger operand scale.
//
// Inputs: t is the test; got and want are finite float64 results.
// Returns: none; fails the test when the discrepancy exceeds the budget.
// Determinism: fixed IEEE comparisons, no randomness.
// Complexity: O(1) time and space.
// AI-Hints: avoid max(1, abs(want)); it makes near-zero variance tests vacuous.
func requireNear(t *testing.T, got, want float64) {
	t.Helper()
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) {
		t.Fatalf("non-finite comparison: got %v, want %v", got, want)
	}
	if got == want {
		return
	}

	difference := math.Abs(got - want)
	if difference <= subnormalBudget*math.SmallestNonzeroFloat64 {
		return
	}
	magnitude := math.Max(math.Abs(got), math.Abs(want))
	if magnitude == 0 || difference/magnitude > relativeBudget {
		t.Fatalf("numeric mismatch: got %.17g, oracle %.17g (relative %.3g)", got, want, difference/magnitude)
	}
}

// requireSummaryEqual checks every observable snapshot field, not just Count.
// This is used to prove rollback and source ownership without reflecting on
// private implementation details or exposing test-only production APIs.
// Implementation: compare the complete public Summary value by field values.
// Inputs: t is the test; got and want are non-nil, fully materialized summaries.
// Returns: none; fails on any observable state deviation.
// Determinism: exact comparison of public snapshot fields.
// Complexity: O(1) time and space.
func requireSummaryEqual(t *testing.T, got, want *stats.Summary) {
	t.Helper()
	if got == nil || want == nil || *got != *want {
		t.Fatalf("snapshot mismatch: got %+v, want %+v", got, want)
	}
}

// oracleRatSummary computes independent exact mean and central variance.
// The inputs are interpreted as their exact binary float64 rational values;
// no production stats method participates in the reference computation.
// Implementation:
//   - Stage 1: convert the original finite float64 values to exact big.Rat.
//   - Stage 2: compute the mean, then squared deviations in a separate pass.
//   - Stage 3: round only the two results to nearest-even float64.
//
// Inputs: values is a finite nonempty slice and ddof is 0 or 1 with n>ddof.
// Returns: nearest float64 mean/variance, or Inf if mathematically too large.
// Determinism: original input iteration order, exact rational operations.
// Complexity: O(n) arbitrary-precision operations; deliberately test-only.
// AI-Hints: the two-pass oracle is allowed here because its arithmetic is exact.
func oracleRatSummary(values []float64, ddof int) (float64, float64) {
	count := big.NewRat(int64(len(values)), 1)
	sum := new(big.Rat)
	for _, value := range values {
		sum.Add(sum, new(big.Rat).SetFloat64(value))
	}
	mean := new(big.Rat).Quo(sum, count)

	centralSquareSum := new(big.Rat)
	for _, value := range values {
		deviation := new(big.Rat).Sub(new(big.Rat).SetFloat64(value), mean)
		centralSquareSum.Add(centralSquareSum, deviation.Mul(deviation, deviation))
	}
	denominator := big.NewRat(int64(len(values)-ddof), 1)
	variance := new(big.Rat).Quo(centralSquareSum, denominator)

	// big.Rat.Float64 performs nearest-even conversion of the exact oracle
	// without introducing a fixed intermediate precision or early rounding.
	floatMean, _ := mean.Float64()
	floatVariance, _ := variance.Float64()
	return floatMean, floatVariance
}

// mustDense constructs an independent dense matrix from rectangular test rows.
// The constructor validates row lengths to avoid accidental ragged test input;
// empty-row shapes are instead constructed explicitly with matrix.NewDense.
// Implementation: validate the fixture and Set every cell in row-major order.
// Inputs: t is the active test; rows is a nonempty rectangular grid.
// Returns: independently allocated *matrix.Dense owned by the caller.
// Determinism: identical fixture shape and values on every invocation.
// Complexity: Time/space O(rows*cols) including dense allocation.
func mustDense(t *testing.T, rows [][]float64) *matrix.Dense {
	t.Helper()
	if len(rows) == 0 {
		t.Fatal("mustDense needs at least one row; construct zero-row matrices explicitly")
	}
	cols := len(rows[0])
	m, err := matrix.NewDense(len(rows), cols)
	if err != nil {
		t.Fatal(err)
	}
	for row, values := range rows {
		if len(values) != cols {
			t.Fatal("ragged fixture")
		}
		for col, value := range values {
			if err = m.Set(row, col, value); err != nil {
				t.Fatal(err)
			}
		}
	}
	return m
}

// mustCell reads a matrix element using the public Matrix interface.
// It never reaches into Dense storage and therefore also tests API compatibility.
// Implementation: call At and fail if the matrix rejects the validated indices.
// Inputs: t is the current test; input, row, col identify a valid test cell.
// Returns: the requested float64 value.
// Determinism: delegates to the specified matrix's At implementation.
// Complexity: O(1) for matrix.Dense.
func mustCell(t *testing.T, input matrix.Matrix, row, col int) float64 {
	t.Helper()
	value, err := input.At(row, col)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

// cloneMoments copies a full accumulator through the public Merge empty identity.
// No implementation detail is accessed, and no test-only state setter is needed.
// Implementation: merge the source into a new zero-value destination.
// Inputs: t is the test; source must be a valid, quiescent accumulator.
// Returns: a separate mutable accumulator with the same numeric state.
// Determinism: follows Merge(empty,source) contract.
// Complexity: O(1) time and space.
func cloneMoments(t *testing.T, source *stats.Moments) *stats.Moments {
	t.Helper()
	copyMoments := stats.NewMoments()
	if err := copyMoments.Merge(source); err != nil {
		t.Fatal(err)
	}
	return copyMoments
}

// scriptedMatrix is an independently defined Matrix for validation/error tests.
// It intentionally can expose bad dimensions, nonfinite cells, and At failures
// that ordinary Dense validation prevents at construction or Set time.
//
// Fields are fixture-only; the production stats implementation never owns them.
// Its IsNil implementation follows matrix's established typed-nil convention.
// Complexity: O(1) per access; no backing store is needed.
type scriptedMatrix struct {
	// rows and cols are the public shape reported by the interface. Negative
	// values intentionally model malformed custom Matrix implementations.
	rows int
	cols int

	// value is the fallback observation returned by At for any valid cell
	// when an explicit per-coordinate cells grid has not been supplied.
	value float64

	// cells optionally overrides the uniform value, including with NaN or Inf
	// values that ordinary Dense.Set correctly refuses to store.
	cells [][]float64

	// readErr is the injected Matrix.At error. With readErrAt==0 it is
	// returned for every valid read; otherwise it fires at one read ordinal.
	readErr error

	// readCount optionally observes attempted valid At calls. It belongs to
	// the test and must never be retained by a production stats result.
	readCount *int

	// readErrAt chooses one 1-based read ordinal for a delayed injected failure.
	// Zero preserves the existing always-fail behavior when readErr is non-nil.
	readErrAt int
}

// IsNil implements the pointer nilability convention without reflect.
// A typed-nil *scriptedMatrix must be rejected before Rows or Cols is called.
// Inputs: receiver may be nil.
// Returns: true exactly when the receiver is nil.
// Complexity: O(1) time and space.
func (m *scriptedMatrix) IsNil() bool { return m == nil }

// Rows reports the fixture's configured row count, including negative values.
// Invalid dimensions are intentional and exercised by the public stats validator.
// Returns: the configured row count; no implicit normalization.
// Complexity: O(1) time and space.
func (m *scriptedMatrix) Rows() int { return m.rows }

// Cols reports the fixture's configured column count, including negative values.
// Invalid dimensions must be rejected by stats before any element reads.
// Returns: the configured column count.
// Complexity: O(1) time and space.
func (m *scriptedMatrix) Cols() int { return m.cols }

// At exposes a fixed fixture value or a deliberate error at a valid coordinate.
// It checks public Matrix bounds first, then counts reads so validation tests
// can prove that malformed shapes never trigger an element-level operation.
// Implementation:
//   - Stage 1: validate row/column bounds before indexing optional backing cells.
//   - Stage 2: count valid read attempts, then return a configured read failure.
//   - Stage 3: return the cell-specific or uniform fixture value.
//
// Inputs: row and col must lie in the configured nonnegative element domain.
// Returns: configured value or readErr; ErrOutOfRange for invalid coordinates.
// Determinism: fixed bounds check, counter increment, and cell-read order.
// Complexity: O(1) time and space.
// AI-Hints: a scripted test matrix is still subject to Matrix.At index laws.
func (m *scriptedMatrix) At(row, col int) (float64, error) {
	if row < 0 || col < 0 || row >= m.rows || col >= m.cols {
		return 0, matrix.ErrOutOfRange
	}
	if m.readCount != nil {
		*m.readCount++
		// An ordinal read failure models a Matrix.At error during a second
		// validation/computation pass without changing any value or shape.
		if m.readErr != nil && m.readErrAt > 0 && *m.readCount == m.readErrAt {
			return 0, m.readErr
		}
	}
	if m.readErr != nil && m.readErrAt == 0 {
		return 0, m.readErr
	}
	if m.cells != nil {
		return m.cells[row][col], nil
	}
	return m.value, nil
}

// Set rejects writes because the scripted fixture is read-only by design.
// Production stats code must never call Set on an input Matrix; this explicit
// unsupported-operation error makes an accidental write observable in tests.
// Implementation: reject all calls without changing stored values or counters.
// Inputs: row, col, value are ignored because the fixture permits no writes.
// Returns: ErrMatrixNotImplemented, rather than misclassifying a nonnil Matrix.
// Errors: matrix.ErrMatrixNotImplemented for any attempted input mutation.
// Determinism: fixed failure with no state change or allocation.
// Complexity: O(1) time and space.
// AI-Hints: unsupported writes are not nil-receiver failures.
func (m *scriptedMatrix) Set(row, col int, value float64) error {
	return matrix.ErrMatrixNotImplemented
}

// Clone returns a deep copy of the scripted matrix's stored values and counters.
// The fixture follows the Matrix.Clone ownership contract even when its optional
// cells and instrumentation are populated, so tests cannot hide aliasing defects.
// Implementation:
//   - Stage 1: return nil when called on a nil receiver.
//   - Stage 2: copy scalar configuration, cell rows, and the optional read count.
//   - Stage 3: publish the separately allocated test fixture.
//
// Inputs: receiver may be nil, empty, or contain mutable cell rows.
// Returns: a detached Matrix without shared cell or read-count backing.
// Errors: none; a nil receiver produces a nil interface result.
// Determinism: copies cells by ascending row then column.
// Complexity: O(rows*cols) time and space when optional cell data is present.
// AI-Hints: a shallow [][]float64 copy violates Matrix.Clone's deep-copy law.
func (m *scriptedMatrix) Clone() matrix.Matrix {
	if m == nil {
		return nil
	}
	copySM := *m
	if m.cells != nil {
		copySM.cells = make([][]float64, len(m.cells))
		for row := range m.cells {
			copySM.cells[row] = append([]float64(nil), m.cells[row]...)
		}
	}
	if m.readCount != nil {
		count := *m.readCount
		copySM.readCount = &count
	}
	return &copySM
}

// Compile-time interface checks prevent the external matrix fixture from
// silently drifting away from the same API contracts used by production stats.
var (
	_ matrix.Matrix = (*scriptedMatrix)(nil)
	_ core.Nilable  = (*scriptedMatrix)(nil)
)

// newRat lifts a float64 into the exact rational domain of the test oracle.
// Its name is local to stats_test to avoid confusion with production scaling.
// Inputs: value must be finite.
// Returns: an independently allocated exact binary big.Rat.
// Complexity: O(1) arbitrary-precision operations for a float64 input.
func newRat(value float64) *big.Rat {
	return new(big.Rat).SetFloat64(value)
}

// isNilLike classifies only nil states understood by the public stats contract.
// Nilable Matrix and result implementations use core.Nilable; Moments is an
// accumulator without that optional interface, so its nil pointer is explicit.
// Implementation:
//   - Stage 1: accept an untyped nil interface.
//   - Stage 2: consult the established core.Nilable method when implemented.
//   - Stage 3: explicitly recognize the remaining pointer-backed accumulator.
//
// Inputs: value is any contract-bearing public result or Matrix reference.
// Returns: true for an actual nil or a conventionally typed-nil object.
// Errors: none.
// Determinism: explicit type comparisons, no dynamic discovery of struct fields.
// Complexity: O(1) time and space.
// AI-Hints: arbitrary pointer implementations without IsNil are not guaranteed.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}
	if nilable, ok := value.(core.Nilable); ok {
		return nilable.IsNil()
	}
	switch typed := value.(type) {
	case *stats.Moments:
		return typed == nil
	default:
		return false
	}
}

// mustNilState validates result ownership/absence without reflect.DeepEqual.
// Unlike comparing any(value)==nil, it recognizes supported typed-nil results
// and points at the calling test when a nil/error result law is violated.
// Implementation: call isNilLike and compare with the requested nil state.
// Inputs: t is the test; value is a public result; wantNil is expected absence;
// op identifies the operation being checked without parsing error strings.
// Returns: none; fails the test on an unexpected nil-state classification.
// Determinism: exact boolean state comparison.
// Complexity: O(1) time and space.
func mustNilState(t *testing.T, value any, wantNil bool, op string) {
	t.Helper()
	if got := isNilLike(value); got != wantNil {
		t.Fatalf("%s: nil-state got %t, want %t", op, got, wantNil)
	}
}

// weightedObservation describes one constant-valued partition without
// materializing all its samples. The count is an exact uint64 multiplicity.
// A list of weighted observations is only test-oracle input; production code
// receives its equivalent accumulator partitions through the public API.
type weightedObservation struct {
	// value is the original finite binary float64 observation.
	value float64

	// count is the number of copies of value; all test cases require count > 0.
	count uint64
}

// oracleWeightedRatSummary derives exact mean and M2 from weighted observations.
// It is independent of the production Welford/Chan code and calculates central
// deviations as rationals rather than subtracting two sums of large squares.
//
// Implementation:
//   - Stage 1: reconstruct exact integer counts and their signed weighted sum.
//   - Stage 2: calculate the exact rational center and squared deviations.
//   - Stage 3: divide by Count-ddof and round only the observable output values.
//
// Behavior highlights:
//   - Supports partition counts up to MaxUint64 without converting them to int64.
//   - Does not allocate Count elements to test enormous merged populations.
//
// Inputs:
//   - observations: nonempty finite-value, positive-count weighted partitions.
//   - ddof: 0 or 1, with total Count greater than ddof.
//
// Returns:
//   - mean: nearest float64 weighted arithmetic mean.
//   - variance: nearest float64 central variance, possibly +Inf on overflow.
//
// Errors:
//   - None; test fixtures establish the valid preconditions before this oracle.
//
// Determinism:
//   - Exact rational operations are independent of partition combination order.
//
// Complexity:
//   - Time O(partitions) arbitrary-precision arithmetic, space O(partitions)
//     in bounded-size rational operands, independent of total observation count.
//
// Notes:
//   - An infinite oracle variance means the public Summary must return ErrOverflow.
//
// AI-Hints:
//   - Do not replace central deviations with E[x*x]-E[x]*E[x].
//   - Preserve counts in big.Int; float64(MaxUint64) rounds to 2^64.
func oracleWeightedRatSummary(observations []weightedObservation, ddof int) (float64, float64) {
	count := new(big.Int)
	sum := new(big.Rat)
	for _, observation := range observations {
		multiplicity := new(big.Int).SetUint64(observation.count)
		count.Add(count, multiplicity)

		term := new(big.Rat).SetFloat64(observation.value)
		term.Mul(term, new(big.Rat).SetInt(multiplicity))
		sum.Add(sum, term)
	}

	mean := new(big.Rat).Quo(sum, new(big.Rat).SetInt(count))
	centralSum := new(big.Rat)
	for _, observation := range observations {
		deviation := new(big.Rat).Sub(newRat(observation.value), mean)
		squaredDeviation := new(big.Rat).Mul(deviation, deviation)
		weightedDeviation := squaredDeviation.Mul(squaredDeviation, new(big.Rat).SetInt(new(big.Int).SetUint64(observation.count)))
		centralSum.Add(centralSum, weightedDeviation)
	}

	denominator := new(big.Int).Sub(count, big.NewInt(int64(ddof)))
	variance := new(big.Rat).Quo(centralSum, new(big.Rat).SetInt(denominator))
	expectedMean, _ := mean.Float64()
	expectedVariance, _ := variance.Float64()
	return expectedMean, expectedVariance
}

// repeatedMoments constructs a constant-valued population with a huge count.
// It builds the requested multiplicity via binary doubling and public Merge,
// making counter-overflow and rounded-count paths observable without fixtures.
//
// Implementation:
//   - Stage 1: seed one accumulator with a single finite value.
//   - Stage 2: fold selected binary powers into a separate destination.
//   - Stage 3: double an independent source only while count bits remain.
//
// Behavior highlights:
//   - A population of 2^63 observations requires only 64 Merge operations.
//   - Every source accumulator is separate; self-merge is never used.
//
// Inputs:
//   - t: owning test, which handles any impossible construction error.
//   - value: finite observation to repeat.
//   - count: requested uint64 population size, including the empty case.
//
// Returns:
//   - *stats.Moments: caller-owned accumulator with exactly count observations.
//
// Errors:
//   - Fails the owning test if the public Update/Merge rejects valid input.
//
// Determinism:
//   - Ascending bit order and fixed destination/source Merge operands.
//
// Complexity:
//   - Time O(log(count+1)), Space O(1), no sample-sized allocations.
//
// Notes:
//   - The helper tests public behavior and never exposes private state fields.
//
// AI-Hints:
//   - Do not construct near-MaxUint64 counts with a literal update loop.
//   - Doubling must operate on an independent source, not m.Merge(m).
func repeatedMoments(t *testing.T, value float64, count uint64) *stats.Moments {
	t.Helper()

	destination := stats.NewMoments()
	if count == 0 {
		return destination
	}

	power := stats.NewMoments()
	if err := power.Update(value); err != nil {
		t.Fatal(err)
	}

	for count > 0 {
		if count&1 != 0 {
			if err := destination.Merge(power); err != nil {
				t.Fatal(err)
			}
		}
		count >>= 1
		if count != 0 {
			if err := power.Merge(cloneMoments(t, power)); err != nil {
				t.Fatal(err)
			}
		}
	}

	return destination
}

// mustEqualBool verifies an exact boolean protocol invariant without reflection.
// It reports the operation name and both values so failures identify the
// contract breach rather than merely printing two unrelated booleans.
//
// Implementation:
//   - Stage 1: mark the caller as the source of any assertion failure.
//   - Stage 2: compare the returned boolean with the expected value.
//   - Stage 3: fail only when they differ, with the operation name included.
//
// Behavior highlights:
//   - Does not perform any implicit truthiness conversion or tolerance check.
//
// Inputs:
//   - t: owning test; got/want: boolean states; op: diagnostic label.
//
// Returns:
//   - None; either preserves the test or fails it immediately.
//
// Errors:
//   - No returned errors; failures are reported through testing.T.
//
// Determinism:
//   - Exact boolean comparison with no external state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper is used for semantic states such as core.Nilable.IsNil.
//
// AI-Hints:
//   - Do not replace this with reflect.DeepEqual or compare error text.
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %t, want %t", op, got, want)
	}
}
