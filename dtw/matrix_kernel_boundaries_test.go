// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package dtw_test verifies private matrix-kernel boundaries through test-build
// adapters, alongside the public DTW facades. No production API is extended.
package dtw_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/dtw"
	"github.com/lvlath/go/matrix"
)

// TestMatrixKernel_DenseCopyRejectsInvalidSource verifies independent copy failures.
// Invalid shape, nil state, backend access, and dirty numeric payloads must not
// publish a partially initialized Dense matrix.
//
// Implementation:
//   - Stage 1: build bounded fixtures for each independent failure family.
//   - Stage 2: invoke the original copy boundary via its test-build adapter.
//   - Stage 3: assert nil output, joined sentinel identity, and unchanged source.
//
// Behavior highlights:
//   - Product overflow is tested without allocating an oversized matrix.
//   - Dirty numeric input is distinct from a backend access error.
//
// Inputs:
//   - t: owning test; sources include nil, typed nil, invalid shapes, and raw Dense.
//
// Returns:
//   - None; a broken error or publication contract fails the test.
//
// Errors:
//   - ErrBadInput with Matrix shape/read causes; ErrNaNInf with Matrix numeric cause.
//
// Determinism:
//   - Sequential cases; the read failure is injected after one valid cell.
//
// Complexity:
//   - O(1) time and space per fixture; all actual matrices have at most two cells.
//
// Notes:
//   - No Matrix allocation policy or production helper is replaced.
//
// AI-Hints:
//   - Negative finite values are legal for a generic copy; nonfinite values are not.
func TestMatrixKernel_DenseCopyRejectsInvalidSource(t *testing.T) {
	base := regressionDense(t, 1, 2, 2, 3) // Keep real backing storage small.
	var typedNil *matrix.Dense
	readErr := errors.New("injected dense copy read failure")
	for _, tc := range []struct {
		name   string
		source matrix.Matrix
		want   []error
	}{
		{"nil", nil, []error{dtw.ErrBadInput, matrix.ErrNilMatrix}},
		{"typed-nil", typedNil, []error{dtw.ErrBadInput, matrix.ErrNilMatrix}},
		{"negative-rows", coverageShapeMatrix{Matrix: base, rows: -1, cols: 2}, []error{dtw.ErrBadInput, matrix.ErrInvalidDimensions}},
		{"negative-columns", coverageShapeMatrix{Matrix: base, rows: 1, cols: -1}, []error{dtw.ErrBadInput, matrix.ErrInvalidDimensions}},
		{"product-overflow", coverageShapeMatrix{Matrix: base, rows: math.MaxInt, cols: 2}, []error{dtw.ErrBadInput, matrix.ErrInvalidDimensions}},
		{"late-read", &coverageReadFaultMatrix{Matrix: base, remainingReads: 1, readErr: readErr}, []error{dtw.ErrBadInput, readErr}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := dtw.DenseCopyOfMatrixForTest(tc.source) // Exercise the independent copy contract.
			mustMatrixKernelFailure(t, got, err, tc.want...)
		})
	}
	for _, tc := range []struct {
		name  string
		value float64
	}{
		{"nan", math.NaN()}, {"positive-infinity", math.Inf(1)}, {"negative-infinity", math.Inf(-1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source, err := matrix.NewPreparedDense(1, 2, matrix.WithNoValidateNaNInf()) // Raw source, finite-policy destination.
			mustNoError(t, err, "allocate dirty source")
			mustNoError(t, source.Fill([]float64{7, tc.value}), "store dirty final cell")
			got, err := dtw.DenseCopyOfMatrixForTest(source)
			mustMatrixKernelFailure(t, got, err, dtw.ErrNaNInf, matrix.ErrNaNInf)
			mustCoverageMatrixCell(t, source, 0, 0, 7, "unchanged finite cell")
			value, err := source.At(0, 1)
			mustNoError(t, err, "read original dirty cell")
			if math.Float64bits(value) != math.Float64bits(tc.value) {
				t.Fatalf("copy failure changed source bits: got %x want %x", math.Float64bits(value), math.Float64bits(tc.value))
			}
		})
	}
	mustCoverageMatrixCell(t, base, 0, 0, 2, "unchanged shared source")
	mustCoverageMatrixCell(t, base, 0, 1, 3, "unchanged shared source")
}

// TestMatrixKernel_DenseCopyOwnershipAndShape verifies finite generic copies.
// The copy preserves rectangular and zero-area shapes and never aliases its input.
//
// Implementation:
//   - Stage 1: copy a signed rectangular source through a generic backend.
//   - Stage 2: compare all values and mutate source and destination independently.
//   - Stage 3: verify legal zero-area shapes remain nonnil copies.
//
// Behavior highlights:
//   - The helper copies signed finite values; it is not a cost-domain validator.
//
// Inputs:
//   - t: owning test with small Dense and generic Matrix fixtures.
//
// Returns:
//   - None; any shape, numeric, or aliasing discrepancy fails the test.
//
// Errors:
//   - No errors are expected for these finite, well-shaped sources.
//
// Determinism:
//   - Values are compared in row-major order.
//
// Complexity:
//   - Time O(rows*cols), Space O(rows*cols) for each copied fixture.
//
// Notes:
//   - A zero-area matrix is not the same as a nil matrix.
//
// AI-Hints:
//   - Verify isolation in both mutation directions, not only pointer inequality.
func TestMatrixKernel_DenseCopyOwnershipAndShape(t *testing.T) {
	values := []float64{-3, math.Copysign(0, -1), 5, 7, -11, 13}
	source := regressionDense(t, 2, 3, values...)
	backend := &coverageReadFaultMatrix{Matrix: source, remainingReads: len(values)} // Generic successful reader.
	got, err := dtw.DenseCopyOfMatrixForTest(backend)
	mustNoError(t, err, "copy signed rectangular matrix")
	if got == nil || got == source {
		t.Fatalf("copy must own independent Dense storage: %v", got)
	}
	mustMatrixShape(t, got, 2, 3, "copy shape")
	for index, want := range values {
		value, readErr := got.At(index/3, index%3)
		mustNoError(t, readErr, "copied cell")
		if math.Float64bits(value) != math.Float64bits(want) {
			t.Fatalf("cell %d bits: got %x want %x", index, math.Float64bits(value), math.Float64bits(want))
		}
	}
	mustNoError(t, got.Set(0, 0, 101), "mutate destination")
	mustCoverageMatrixCell(t, source, 0, 0, -3, "source detached from destination")
	mustNoError(t, source.Set(1, 2, 103), "mutate source")
	mustCoverageMatrixCell(t, got, 1, 2, 13, "destination detached from source")

	for _, shape := range [][2]int{{0, 0}, {0, 3}, {2, 0}} {
		empty := regressionDense(t, shape[0], shape[1])
		copyMatrix, copyErr := dtw.DenseCopyOfMatrixForTest(empty)
		mustNoError(t, copyErr, "copy zero-area matrix")
		if copyMatrix == nil || copyMatrix == empty {
			t.Fatalf("zero-area copy must be nonnil and detached: %v", copyMatrix)
		}
		mustMatrixShape(t, copyMatrix, shape[0], shape[1], "zero-area shape")
	}
}

// TestMatrixKernel_FlatLocalRejectsMalformedDomain protects the private flat adapter.
// These failures cannot be exercised through a correctly validated matrix facade.
//
// Implementation:
//   - Stage 1: build small buffers with invalid dimensions or local values.
//   - Stage 2: request a strict diagonal to keep off-band validation observable.
//   - Stage 3: verify the first applicable sentinel and absence of a result.
//
// Behavior highlights:
//   - Shape checks precede numeric checks; every local cell must be finite and nonnegative.
//
// Inputs:
//   - t: owning test; dimensions include platform MaxInt without a large allocation.
//
// Returns:
//   - None; incorrect precedence or result publication fails the test.
//
// Errors:
//   - ErrEmptyInput, ErrBadInput, ErrNaNInf, ErrNegativeCost, and ErrNilOption.
//
// Determinism:
//   - Invalid numeric values are placed at the same off-diagonal position.
//
// Complexity:
//   - O(1) time and space per small fixture.
//
// Notes:
//   - The bridge invokes the actual kernel without exposing private options.
//
// AI-Hints:
//   - A zero window does not authorize ignoring malformed off-band precomputed costs.
func TestMatrixKernel_FlatLocalRejectsMalformedDomain(t *testing.T) {
	for _, tc := range []struct {
		name       string
		rows, cols int
		costs      []float64
		want       error
	}{
		{"zero-rows-first", 0, 2, []float64{math.NaN()}, dtw.ErrEmptyInput},
		{"zero-columns", 2, 0, nil, dtw.ErrEmptyInput},
		{"negative-rows", -1, 2, nil, dtw.ErrBadInput},
		{"negative-columns", 2, -1, nil, dtw.ErrBadInput},
		{"product-overflow", math.MaxInt, 2, []float64{0}, dtw.ErrBadInput},
		{"nil-buffer", 1, 1, nil, dtw.ErrBadInput},
		{"short-buffer", 2, 2, []float64{math.NaN()}, dtw.ErrBadInput},
		{"long-buffer", 1, 1, []float64{1, 2}, dtw.ErrBadInput},
		{"off-band-nan", 2, 2, []float64{0, math.NaN(), 0, 0}, dtw.ErrNaNInf},
		{"off-band-positive-infinity", 2, 2, []float64{0, math.Inf(1), 0, 0}, dtw.ErrNaNInf},
		{"off-band-negative-infinity", 2, 2, []float64{0, math.Inf(-1), 0, 0}, dtw.ErrNaNInf},
		{"off-band-negative-cost", 2, 2, []float64{0, -1, 0, 0}, dtw.ErrNegativeCost},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := dtw.ComputeFromFlatLocalForTest(tc.costs, tc.rows, tc.cols, dtw.WithWindow(0))
			mustMatrixKernelFailure(t, got, err, tc.want)
			if errors.Is(tc.want, dtw.ErrBadInput) && errors.Is(err, dtw.ErrNaNInf) {
				t.Fatalf("shape failure incorrectly classified as a numeric failure: %v", err)
			}
		})
	}
	got, err := dtw.ComputeFromFlatLocalForTest(nil, 0, 0, nil) // Options precede shape validation.
	mustMatrixKernelFailure(t, got, err, dtw.ErrNilOption)
}

// TestMatrixKernel_FlatLocalRectangularWitness verifies index translation and scoring.
// A rectangular local surface has a unique, independently computed optimal path.
//
// Implementation:
//   - Stage 1: use a 2-by-3 surface and a half-unit stretching penalty.
//   - Stage 2: align using the private flat adapter and the public matrix facade.
//   - Stage 3: compare the known objective, witness, and artifact ownership.
//
// Behavior highlights:
//   - Precomputed local costs never call the scalar metric callback.
//   - The flat adapter does not publish the facade-owned local-cost artifact.
//
// Inputs:
//   - t: owning test; finite row-major costs with unique optimal witness.
//
// Returns:
//   - None; a wrong objective, witness, or publication policy fails the test.
//
// Errors:
//   - No errors are expected; any invocation of the supplied callback is an error.
//
// Determinism:
//   - The unique optimal path is diagonal followed by a horizontal step.
//
// Complexity:
//   - O(rows*cols) time and space with explicitly requested accumulated storage.
//
// Notes:
//   - Only the matrix facade attaches the complete detached LocalCost matrix.
//
// AI-Hints:
//   - Compare with an independently calculated distance, not just backend parity.
func TestMatrixKernel_FlatLocalRectangularWitness(t *testing.T) {
	costs := []float64{1, 7, 9, 8, 2, 3}
	const rows, cols, penalty, objective = 2, 3, 0.5, 6.5
	unexpectedMetric := errors.New("precomputed adapter invoked scalar metric")
	opts := []dtw.Option{
		dtw.WithWindow(1), dtw.WithSlopePenalty(penalty), dtw.WithReturnPath(true),
		dtw.WithReturnAccumulated(true), dtw.WithReturnLocalCost(true),
		dtw.WithCostFunc(func(_, _ float64) (float64, error) { return 0, unexpectedMetric }),
	}
	flat, err := dtw.ComputeFromFlatLocalForTest(costs, rows, cols, opts...)
	mustNoError(t, err, "flat local alignment")
	public, err := dtw.AlignCostMatrix(regressionDense(t, rows, cols, costs...), opts...)
	mustNoError(t, err, "public matrix alignment")
	wantPath := dtw.Path{{I: 0, J: 0}, {I: 1, J: 1}, {I: 1, J: 2}}
	for _, got := range []*dtw.Result{flat, public} {
		if got == nil || !got.Reachable || !got.PathTracked || got.Accumulated == nil {
			t.Fatalf("incomplete reachable result: %+v", got)
		}
		mustFloatEqual(t, got.Distance, objective, 0, "independent path sum") // 1 + 2 + 3 + penalty.
		mustPathEqual(t, got.Path, wantPath, "canonical witness")
		mustMatrixShape(t, got.Accumulated, rows+1, cols+1, "accumulated boundary shape")
		mustCoverageMatrixCell(t, got.Accumulated, rows, cols, objective, "terminal DP value")
		mustCoverageMatrixCell(t, got.Accumulated, 0, 0, 0, "empty-prefix origin")
		mustCoverageMatrixCell(t, got.Accumulated, 0, 1, math.Inf(1), "unreachable prefix")
	}
	if flat.LocalCost != nil || public.LocalCost == nil {
		t.Fatalf("unexpected local artifact policy: flat=%v public=%v", flat.LocalCost, public.LocalCost)
	}
	// The private result's path and accumulated matrix are not aliases of public output or input.
	flat.Path[0] = dtw.Coord{I: 9, J: 9}
	mustPathEqual(t, public.Path, wantPath, "detached public witness")
	mustNoError(t, flat.Accumulated.Set(rows, cols, 99), "mutate private artifact")
	mustCoverageMatrixCell(t, public.Accumulated, rows, cols, objective, "detached public DP")
	for index, want := range []float64{1, 7, 9, 8, 2, 3} {
		mustFloatEqual(t, costs[index], want, 0, "unchanged flat input")
		mustCoverageMatrixCell(t, public.LocalCost, index/cols, index%cols, want, "complete public local surface")
	}
}

// TestMatrixKernel_FlatLocalTerminalStates separates absent and overflowing optima.
// The same private adapter must preserve the classified recurrence's terminal law.
//
// Implementation:
//   - Stage 1: construct a rectangular strict-diagonal no-path case.
//   - Stage 2: exercise it with path tracking disabled and enabled.
//   - Stage 3: contrast structural absence with mandatory numerical overflow.
//
// Behavior highlights:
//   - Only structural absence yields an unreachable +Inf result.
//
// Inputs:
//   - t: owning test; small precomputed local surfaces.
//
// Returns:
//   - None; a confused terminal state fails the test.
//
// Errors:
//   - ErrNoPath only for requested unreachable witnesses; ErrOverflow and ErrNaNInf for overflow.
//
// Determinism:
//   - Radius zero removes alternative paths.
//
// Complexity:
//   - O(1) time and space for the fixed fixtures.
//
// Notes:
//   - Overflow must not publish an unusable Result.
//
// AI-Hints:
//   - Do not use a single finite cost as a terminal-overflow fixture.
func TestMatrixKernel_FlatLocalTerminalStates(t *testing.T) {
	for _, trackPath := range []bool{false, true} {
		got, err := dtw.ComputeFromFlatLocalForTest([]float64{0, 0}, 2, 1, dtw.WithWindow(0), dtw.WithReturnPath(trackPath))
		if trackPath {
			mustErrorIs(t, err, dtw.ErrNoPath, "requested unreachable witness")
		} else {
			mustNoError(t, err, "distance-only unreachable result")
		}
		if got == nil || got.Reachable || got.PathTracked != trackPath || len(got.Path) != 0 {
			t.Fatalf("incorrect structural no-path state: %+v", got)
		}
		mustInf(t, got.Distance, "structural absence distance")
	}
	got, err := dtw.ComputeFromFlatLocalForTest([]float64{1e308, 0, 0, 1e308}, 2, 2, dtw.WithWindow(0))
	mustMatrixKernelFailure(t, got, err, dtw.ErrOverflow, dtw.ErrNaNInf)
	if errors.Is(err, dtw.ErrNoPath) {
		t.Fatalf("terminal numeric overflow became structural absence: %v", err)
	}
}

// TestMatrixKernel_AccumulatedInitializerRejectsInvalidShape verifies constructor failures.
// Matrix permits zero-area storage, but a DP artifact must contain its empty-prefix origin.
//
// Implementation:
//   - Stage 1: exercise invalid dimensions and checked shape-product overflow.
//   - Stage 2: exercise zero-area matrices where the origin cannot be written.
//   - Stage 3: require nil output and preserve the originating Matrix sentinel.
//
// Behavior highlights:
//   - A zero-sized artifact fails at origin initialization, not numeric validation.
//
// Inputs:
//   - t: owning test; invalid dimensions never require huge storage.
//
// Returns:
//   - None; incorrect constructor error or publication fails the test.
//
// Errors:
//   - ErrBadInput with ErrInvalidDimensions or ErrOutOfRange.
//
// Determinism:
//   - Constructor and boundary writes execute sequentially.
//
// Complexity:
//   - O(1) time and space per rejected fixture.
//
// Notes:
//   - Valid recurrence calls always include the additional boundary row and column.
//
// AI-Hints:
//   - Use product overflow, not a runtime out-of-memory event, to test constructor rejection.
func TestMatrixKernel_AccumulatedInitializerRejectsInvalidShape(t *testing.T) {
	for _, tc := range []struct {
		name       string
		rows, cols int
		want       error
	}{
		{"negative-rows", -1, 1, matrix.ErrInvalidDimensions},
		{"negative-columns", 1, -1, matrix.ErrInvalidDimensions},
		{"product-overflow", math.MaxInt, 2, matrix.ErrInvalidDimensions},
		{"zero-rows", 0, 2, matrix.ErrOutOfRange},
		{"zero-columns", 2, 0, matrix.ErrOutOfRange},
		{"zero-area", 0, 0, matrix.ErrOutOfRange},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := dtw.NewAccumulatedCostMatrixForTest(tc.rows, tc.cols)
			mustMatrixKernelFailure(t, got, err, dtw.ErrBadInput, tc.want)
			if errors.Is(err, dtw.ErrNaNInf) {
				t.Fatalf("invalid dimensions incorrectly classified as numeric data: %v", err)
			}
		})
	}
}

// TestMatrixKernel_AccumulatedInitializerBoundaryPolicy verifies the exact initial surface.
// All cells except the empty-prefix origin must represent unreachable DP states.
//
// Implementation:
//   - Stage 1: initialize positive square, rectangular, and single-axis shapes.
//   - Stage 2: inspect every cell and validate the preserved distance-domain policy.
//   - Stage 3: mutate one result and verify a second allocation is independent.
//
// Behavior highlights:
//   - Positive infinity is allowed; NaN and negative infinity are rejected.
//
// Inputs:
//   - t: owning test; positive dimensions include a single-cell origin.
//
// Returns:
//   - None; incorrect initialization or numeric policy fails the test.
//
// Errors:
//   - Matrix ErrNaNInf is expected for prohibited writes only.
//
// Determinism:
//   - Every row and column is inspected explicitly.
//
// Complexity:
//   - Time O(rows*cols), Space O(rows*cols) per fixture.
//
// Notes:
//   - The initializer alone has no information about future finite or overflow DP cells.
//
// AI-Hints:
//   - Inspect the interior too; it is initially +Inf, not zero-filled local cost.
func TestMatrixKernel_AccumulatedInitializerBoundaryPolicy(t *testing.T) {
	for _, shape := range [][2]int{{1, 1}, {1, 4}, {4, 1}, {3, 4}} {
		got, err := dtw.NewAccumulatedCostMatrixForTest(shape[0], shape[1])
		mustNoError(t, err, "initialize accumulated matrix")
		if got == nil {
			t.Fatal("successful initialization returned nil")
		}
		mustMatrixShape(t, got, shape[0], shape[1], "initial DP shape")
		for i := 0; i < shape[0]; i++ {
			for j := 0; j < shape[1]; j++ {
				want := math.Inf(1) // Every uncomputed prefix is structurally absent initially.
				if i == 0 && j == 0 {
					want = 0 // The empty-prefix origin is the unique finite seed.
				}
				mustCoverageMatrixCell(t, got, i, j, want, "initial DP cell")
			}
		}
		for _, value := range []float64{math.NaN(), math.Inf(-1)} {
			mustErrorIs(t, got.Set(0, 0, value), matrix.ErrNaNInf, "reject invalid accumulated value")
			mustCoverageMatrixCell(t, got, 0, 0, 0, "origin unchanged after rejected write")
		}
		mustNoError(t, got.Set(0, 0, math.Inf(1)), "permit positive infinity in DP storage")
		mustCoverageMatrixCell(t, got, 0, 0, math.Inf(1), "stored positive infinity")
		fresh, err := dtw.NewAccumulatedCostMatrixForTest(shape[0], shape[1])
		mustNoError(t, err, "independent allocation")
		if fresh == nil || fresh == got {
			t.Fatalf("initializer reused a mutable result: %v", fresh)
		}
		mustCoverageMatrixCell(t, fresh, 0, 0, 0, "independent origin")
	}
}
