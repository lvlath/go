// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix provides exponent-aware storage for economy SVD.
// This file retains an independent binary scale for every working column.
package matrix

import (
	"math"
)

const (
	// svdMachineEpsilon is the spacing immediately above 1 in binary64.
	// The rank threshold and default correlation tolerance are multiples of it.
	svdMachineEpsilon = 0x1p-52

	// svdDefaultSweeps caps complete lexicographic Jacobi sweeps by default.
	svdDefaultSweeps = 100

	// svdDefaultToleranceMultiplier sets the default correlation tolerance.
	svdDefaultToleranceMultiplier = 8

	// svdBasisSafetyMultiplier bounds near-dependent completion candidates.
	svdBasisSafetyMultiplier = 8
)

// svdWorkspace stores working columns as physicalColumn = 2^exponents[j]*data[:,j].
// Each nonzero stored column has maximum absolute coordinate in [0.5, 1),
// avoiding global scaling that would erase an otherwise representable tiny mode.
// The right basis is square only in k=min(m,n), never in max(m,n).
//
// Implementation:
//   - Stage 1: orient the rectangular input so rows >= cols.
//   - Stage 2: normalize each column by its own exact power of two.
//   - Stage 3: initialize a k-by-k identity as the right-side rotation accumulator.
//
// Behavior highlights:
//   - exponents may exceed the finite binary64 exponent range during rotations.
//   - The caller-owned input snapshot is consumed only after successful validation.
//
// Determinism: working columns and identity coordinates are scanned in ascending order.
// Complexity: Time O(m*n), Space O(m*n+k*k), k=min(m,n).
// AI-Hints: Do not replace exponents with one global maximum; diag(1e308, 1e-300)
// would otherwise lose a real nonzero singular value before the first sweep.
type svdWorkspace struct {
	rows      int
	cols      int
	data      []float64 // Row-major normalized columns, owned by the current SVD call.
	exponents []int     // Physical binary exponent per stored column.
	norms     []float64 // L2 norms of normalized columns, updated after each rotation.
	right     []float64 // Row-major k-by-k rotation accumulator, independent of input.
}

// svdNewWorkspace transposes wide matrices before constructing the Jacobi basis.
// Ownership of the validated input snapshot passes into the workspace for tall inputs.
//
// Implementation:
//   - Stage 1: choose the tall orientation without a full square workspace.
//   - Stage 2: transpose only when m<n, preserving original row/column values.
//   - Stage 3: normalize each column and initialize an orthogonal right basis.
//
// Inputs: snapshot is a private finite m-by-n row-major input; m,n>=1.
// Returns: Detached workspace with finite normalized columns.
// Errors: None after checked public dimensions and finite input validation.
// Determinism: Original input row/column order and identity diagonal order.
// Complexity: Time O(m*n), Space O(m*n+k*k) including the input snapshot.
// AI-Hints: The wide case must map U/VT back after decomposition, not before.
func svdNewWorkspace(snapshot []float64, m, n int) svdWorkspace {
	rows, cols := m, n
	working := snapshot
	if m < n {
		rows, cols = n, m
		working = make([]float64, rows*cols)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				working[j*cols+i] = snapshot[i*n+j]
			}
		}
	}

	state := svdWorkspace{
		rows:      rows,
		cols:      cols,
		data:      working,
		exponents: make([]int, cols),
		norms:     make([]float64, cols),
		right:     make([]float64, cols*cols),
	}
	for j := 0; j < cols; j++ {
		state.normalizeColumn(j, 0)
		state.right[j*cols+j] = 1
	}
	return state
}

// normalizeColumn renormalizes one working column after a physical-space update.
// The incoming column equals data[:,column]*2^baseExponent; the stored result
// replaces it with an equivalent binary-scale representation whenever nonzero.
//
// Implementation:
//   - Stage 1: find the largest absolute entry without a squared reduction.
//   - Stage 2: apply its exact binary exponent to all entries in one pass.
//   - Stage 3: cache the safe L2 norm in local normalized coordinates.
//
// Inputs: column is valid; baseExponent describes the supplied column values.
// Returns: No value; updates private data, exponents, and normalized norms.
// Errors: None for finite rotated columns with bounded normalized coordinates.
// Determinism: Row scan and normalization always progress in ascending row order.
// Complexity: Time O(rows), Space O(1).
// Notes: Values smaller than the column's representable relative range may underflow.
// AI-Hints: Physical exponents can exceed 1024; never materialize 2^exponent.
func (w *svdWorkspace) normalizeColumn(column, baseExponent int) {
	maximum := 0.0
	for i := 0; i < w.rows; i++ {
		maximum = math.Max(maximum, math.Abs(w.data[i*w.cols+column]))
	}
	if maximum == 0 {
		w.exponents[column] = 0
		w.norms[column] = 0
		return
	}
	_, shift := math.Frexp(maximum)
	for i := 0; i < w.rows; i++ {
		index := i*w.cols + column
		w.data[index] = math.Ldexp(w.data[index], -shift)
	}
	w.exponents[column] = baseExponent + shift
	w.norms[column] = svdColumnNorm(w.data, w.rows, w.cols, column)
}

// svdColumnNorm computes a column L2 norm with scale-free hypot accumulation.
// It is used only on normalized columns or orthonormal basis candidates.
//
// Inputs: data holds finite rows-by-cols coordinates; column is in range.
// Returns: Nonnegative finite L2 norm for the normalized column.
// Errors: None for the bounded internal coordinates.
// Determinism: Increasing row indices, no parallel reductions.
// Complexity: Time O(rows), Space O(1).
// AI-Hints: Avoid sum(v*v) on input data; input can contain MaxFloat64.
func svdColumnNorm(data []float64, rows, cols, column int) float64 {
	length := 0.0
	for i := 0; i < rows; i++ {
		length = math.Hypot(length, data[i*cols+column])
	}
	return length
}

// svdScaledMagnitudeRatio compares two positive column norms without forming either
// physical norm, returning the ratio smaller/larger in [0,1] and which is larger.
//
// Implementation:
//   - Stage 1: split normalized norms into bounded mantissa and binary exponent.
//   - Stage 2: compare exponent then mantissa to determine the larger norm.
//   - Stage 3: restore only the small/large ratio, which cannot overflow.
//
// Inputs: positive cached norms and their independently maintained exponents.
// Returns: ratio in [0,1], whether first is larger, and whether magnitudes tie.
// Errors: None for nonzero normalized columns.
// Determinism: Exact binary exponent/mantissa comparisons, no heuristic tie bound.
// Complexity: Time O(1), Space O(1).
// AI-Hints: Do not compute beta/alpha from physical norms; their squares may overflow.
func svdScaledMagnitudeRatio(a float64, expA int, b float64, expB int) (ratio float64, firstLarger, equal bool) {
	mantissaA, normExpA := math.Frexp(a)
	mantissaB, normExpB := math.Frexp(b)
	powerA := expA + normExpA
	powerB := expB + normExpB
	if powerA == powerB && mantissaA == mantissaB {
		return 1, false, true
	}
	firstLarger = powerA > powerB || (powerA == powerB && mantissaA > mantissaB)
	if firstLarger {
		return math.Ldexp(mantissaB/mantissaA, powerB-powerA), true, false
	}
	return math.Ldexp(mantissaA/mantissaB, powerA-powerB), false, false
}

// svdPhysicalNorm restores the one physical singular value at the output boundary.
// Only an unrepresentable final singular value is classified as ErrOverflow.
//
// Inputs: normalized column norm and its binary scale exponent.
// Returns: Nonnegative finite binary64 singular value or zero for a zero column.
// Errors: ErrOverflow for a final infinity; no phantom Inf enters a result.
// Determinism: One binary exponent restoration per output singular value.
// Complexity: Time O(1), Space O(1).
// AI-Hints: Keep the exponent until the final result; scale*norm may overflow early.
func svdPhysicalNorm(norm float64, exponent int) (float64, error) {
	if norm == 0 {
		return 0, nil
	}
	value := math.Ldexp(norm, exponent)
	if math.IsInf(value, 0) || math.IsNaN(value) {
		return 0, ErrOverflow
	}
	return value, nil
}
