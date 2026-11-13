// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm checks presence, shape, finiteness, and stochastic probability domains.
// This boundary rejects invalid matrices before passing their copied data to numerical kernels.
package hmm

import (
	"context"
	"fmt"
	"math"

	"github.com/lvlath/go/matrix"
)

// probabilityTolerance bounds the accepted absolute drift from unit simplex mass.
// It applies to constructor normalization and externally restored log-posterior evidence.
//
// Implementation:
//   - Stage 1: Declare a fixed absolute floating-point allowance.
//   - Stage 2: Reject rows exceeding the allowance.
//   - Stage 3: Normalize only accepted copies before hashing or inference.
//
// Behavior highlights:
//   - This is not a sparsity threshold: exact zero probabilities remain zero.
//
// Inputs:
//   - No runtime inputs; measured in probability/log-normalization units.
//
// Returns:
//   - One compile-time tolerance shared by validators.
//
// Errors:
//   - ErrProbability and ErrStateVersion are reported by callers when exceeded.
//
// Determinism:
//   - The same constant is used for every state and every model.
//
// Complexity:
//   - O(1) constant storage and comparisons.
//
// Notes:
//   - The threshold does not permit nonfinite or negative values.
//
// AI-Hints:
//   - Never replace exact zero transitions by this tolerance; it would create forbidden paths.
const (
	probabilityTolerance = 1e-12

	// defaultFitMaxIterations caps the accepted Baum-Welch M-steps when no limit is supplied.
	defaultFitMaxIterations = 100

	// defaultFitTolerance controls relative objective improvement stopping for zero FitOptions.
	defaultFitTolerance = 1e-6

	// defaultVarianceFloor bounds fitted diagonal variance in squared feature units.
	defaultVarianceFloor = 1e-8
)

// finite recognizes ordinary finite binary64 values without modifying the operand.
// It is the shared arithmetic boundary between accepted input and log-domain sentinels.
//
// Implementation:
//   - Stage 1: Check NaN.
//   - Stage 2: Reject either sign of infinity.
//   - Stage 3: Return the boolean classification.
//
// Behavior highlights:
//   - Negative finite values are valid numbers; separate domain validators restrict them.
//
// Inputs:
//   - x: binary64 input in the domain where the caller defines validity.
//
// Returns:
//   - bool: true for any finite x, including signed zero.
//
// Errors:
//   - None directly; callers map failures to ErrNonFinite, ErrOverflow, or ErrNumerical.
//
// Determinism:
//   - Pure IEEE-754 predicates, independent of model state.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Do not apply this predicate to valid log-probability -Inf sentinels.
//
// AI-Hints:
//   - Use math.IsInf(x, -1) separately when checking impossible log mass.
func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

// checkContext enforces the common public cancellation and nil-context law.
// No input data is inspected until this validation succeeds.
//
// Implementation:
//   - Stage 1: Reject a nil context.
//   - Stage 2: Return the context cancellation or deadline error verbatim.
//   - Stage 3: Allow an active context through.
//
// Behavior highlights:
//   - It never replaces a cancellation error with an HMM-domain error.
//
// Inputs:
//   - ctx: non-nil context; its cancellation may originate from another goroutine.
//
// Returns:
//   - error: nil only if the context is present and currently active.
//
// Errors:
//   - ErrInvalidOption for nil context; ctx.Err() for canceled/deadline-exceeded.
//
// Determinism:
//   - No context mutation; precedence is nil before cancellation.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Kernels also poll ctx during long chronological loops.
//
// AI-Hints:
//   - This check does not guarantee completion if cancellation occurs later.
func checkContext(ctx context.Context) error {
	if ctx == nil {
		return ErrInvalidOption
	}
	return ctx.Err()
}

// isNilMatrix detects absent matrix interfaces and supported typed-nil matrices.
// It follows the core.Nilable IsNil convention without reflect or unsafe.
//
// Implementation:
//   - Stage 1: Check the interface itself for nil.
//   - Stage 2: If implemented, call the pointer-safe IsNil method.
//   - Stage 3: Otherwise classify the non-nil interface as present.
//
// Behavior highlights:
//   - Custom pointer-backed matrix implementations must expose IsNil to detect typed nil.
//
// Inputs:
//   - m: matrix.Matrix interface, possibly containing a typed-nil concrete pointer.
//
// Returns:
//   - bool: whether the supported matrix representation is absent.
//
// Errors:
//   - None directly; its callers report ErrNilInput before shape checks.
//
// Determinism:
//   - Fixed type assertion and optional method call; no reflection.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The interface contract alone cannot detect every custom typed nil without IsNil.
//
// AI-Hints:
//   - Implement IsNil on custom Matrix pointer types; avoid guessing nil by Rows()==0.
func isNilMatrix(m matrix.Matrix) bool {
	if m == nil {
		return true
	}
	// All supported lvlath matrices implement the Nilable convention.
	if n, ok := m.(interface{ IsNil() bool }); ok {
		return n.IsNil()
	}
	return false
}

// readMatrix validates a numerical matrix and flattens it into independent row-major storage.
// Later HMM kernels operate only on this detached slice rather than interface At calls.
//
// Implementation:
//   - Stage 1: Reject nil/typed-nil and incompatible shape, then check rows*cols arithmetic.
//   - Stage 2: Read every cell in deterministic row-major order and reject nonfinite inputs.
//   - Stage 3: Publish only the fully validated copy; discard it on any error.
//
// Behavior highlights:
//   - No live matrix alias is retained; allocation precedes the first At read.
//
// Inputs:
//   - m: caller-owned read-only matrix; rows, cols: required exact shape with nonnegative dimensions.
//
// Returns:
//   - []float64: independent length rows*cols; nil on failure.
//
// Errors:
//   - ErrNilInput, ErrShape, ErrOverflow, ErrNonFinite, or a wrapped Matrix.At error.
//
// Determinism:
//   - Row index increases first, then column index; the first failed At ends the scan.
//
// Complexity:
//   - Time O(rows*cols), Space O(rows*cols).
//
// Notes:
//   - Callers must prevent concurrent matrix mutation during the scan.
//
// AI-Hints:
//   - Do not Clone and assume the underlying Matrix is finite; validate each copied cell.
func readMatrix(m matrix.Matrix, rows, cols int) ([]float64, error) {
	if isNilMatrix(m) {
		return nil, ErrNilInput
	}
	if m.Rows() != rows || m.Cols() != cols {
		return nil, ErrShape
	}
	if rows < 0 || cols < 0 || (cols > 0 && rows > int(^uint(0)>>1)/cols) {
		return nil, ErrOverflow
	}
	out := make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			v, err := m.At(i, j)
			if err != nil {
				return nil, fmt.Errorf("hmm: read (%d,%d): %w", i, j, err)
			}
			if !finite(v) {
				return nil, ErrNonFinite
			}
			out[i*cols+j] = v
		}
	}
	return out, nil
}

// normalizeRows checks finite copied probability rows and divides by their accepted sum.
// It preserves exact structural zeros while repairing only small row-sum residuals.
//
// Implementation:
//   - Stage 1: For each row, accumulate the mass and reject cells outside [0,1].
//   - Stage 2: Reject nonfinite totals or absolute unit-sum residual above probabilityTolerance.
//   - Stage 3: Divide each accepted row in place by its sum.
//
// Behavior highlights:
//   - Partially normalized internal scratch is discarded by constructors if a later row fails.
//
// Inputs:
//   - a: already detached and finite row-major buffer; n, cols: valid positive row dimensions.
//
// Returns:
//   - error: nil on accepted stochastic rows; a contains normalized copies.
//
// Errors:
//   - ErrProbability for negative, above-one, nonfinite-sum, or non-unit mass.
//
// Determinism:
//   - Rows and cells are visited in ascending index order.
//
// Complexity:
//   - Time O(n*cols), Space O(1) beyond the supplied detached buffer.
//
// Notes:
//   - The helper mutates its private argument; never pass caller-owned specification slices.
//
// AI-Hints:
//   - A probability of zero is meaningful and must remain zero after normalization.
func normalizeRows(a []float64, n, cols int) error {
	for i := 0; i < n; i++ {
		row := a[i*cols : (i+1)*cols]
		var sum float64
		for _, v := range row {
			if v < 0 || v > 1 {
				return ErrProbability
			}
			sum += v
		}
		if !finite(sum) || math.Abs(sum-1) > probabilityTolerance {
			return ErrProbability
		}
		for j := range row {
			row[j] /= sum
		}
	}
	return nil
}

// validateCommon copies and validates shared Markov priors and transition probabilities.
// It establishes the K-state stochastic chain consumed by both emission families.
//
// Implementation:
//   - Stage 1: Check that Prior and Transition are present, and derive K>=1.
//   - Stage 2: Validate finite prior entries and copy the exact K-by-K transition matrix.
//   - Stage 3: Normalize valid copies and return a private chain before model-specific hashing.
//
// Behavior highlights:
//   - A[i,j] always denotes the transition from source state i to next state j.
//
// Inputs:
//   - c: caller-owned specification with length-K Prior and K-by-K Transition.
//
// Returns:
//   - chain: independent stochastic parameters, or zero chain with error.
//
// Errors:
//   - ErrNilInput, ErrShape, ErrNonFinite, ErrProbability, ErrOverflow, and Matrix.At failures.
//
// Determinism:
//   - Priors are copied in ascending index order; transitions in row-major order.
//
// Complexity:
//   - Time O(K^2), Space O(K^2+K).
//
// Notes:
//   - The returned chain is internal and immutable after model construction.
//
// AI-Hints:
//   - Do not transpose transition orientation or round positive probabilities to zero.
func validateCommon(c Common) (chain, error) {
	if c.Prior == nil || isNilMatrix(c.Transition) {
		return chain{}, ErrNilInput
	}
	k := len(c.Prior)
	if k == 0 {
		return chain{}, ErrShape
	}
	p := append([]float64(nil), c.Prior...)
	for _, v := range p {
		if !finite(v) {
			return chain{}, ErrNonFinite
		}
	}
	a, err := readMatrix(c.Transition, k, k)
	if err != nil {
		return chain{}, err
	}
	if err = normalizeRows(p, 1, k); err != nil {
		return chain{}, err
	}
	if err = normalizeRows(a, k, k); err != nil {
		return chain{}, err
	}
	return chain{k: k, prior: p, transition: a}, nil
}

// validateSymbols validates an entire categorical observation sequence before inference.
// Every symbol is a zero-based emission-column index, not a measured scalar value.
//
// Implementation:
//   - Stage 1: Distinguish nil from non-nil empty slices.
//   - Stage 2: Scan symbols and reject those outside the alphabet.
//   - Stage 3: Return success only after the complete sequence is validated.
//
// Behavior highlights:
//   - A single invalid later symbol rejects the whole call; no partial history is returned.
//
// Inputs:
//   - x: read-only sequence; alphabet: positive number of encoded symbol categories.
//
// Returns:
//   - error: nil only when all observations belong to [0, alphabet).
//
// Errors:
//   - ErrNilInput, ErrEmptyInput, or ErrShape.
//
// Determinism:
//   - First invalid symbol in chronological order determines failure.
//
// Complexity:
//   - Time O(T), Space O(1).
//
// Notes:
//   - Symbol encoders and dictionary versioning belong to the application.
//
// AI-Hints:
//   - Do not pass numeric telemetry values as categorical bin indexes without encoding.
func validateSymbols(x []uint32, alphabet int) error {
	if x == nil {
		return ErrNilInput
	}
	if len(x) == 0 {
		return ErrEmptyInput
	}
	for _, v := range x {
		// The common signed width preserves both operands on 32- and 64-bit Go.
		// No conversion of the observation to int occurs until it is known in-range.
		if int64(v) >= int64(alphabet) {
			return ErrShape
		}
	}
	return nil
}

// validateObservations checks Gaussian observation presence, length, and feature dimensions.
// Its owning flattened copy stabilizes the inputs for a single inference call.
//
// Implementation:
//   - Stage 1: Reject absent or zero-row inputs.
//   - Stage 2: Require exactly D columns and finite cells via readMatrix.
//   - Stage 3: Return the observation count and detached row-major values.
//
// Behavior highlights:
//   - No missing-value imputation, interpolation, or arbitrary feature reorder occurs.
//
// Inputs:
//   - x: caller-owned T-by-D Matrix; dimension: fitted positive feature count.
//
// Returns:
//   - T, []float64, error: T observations and T*D values on success.
//
// Errors:
//   - ErrNilInput, ErrEmptyInput, ErrShape, ErrNonFinite, ErrOverflow, or Matrix.At errors.
//
// Determinism:
//   - The same deterministic row-major order as readMatrix.
//
// Complexity:
//   - Time O(T*D), Space O(T*D).
//
// Notes:
//   - Feature column order and physical units must match the model.
//
// AI-Hints:
//   - Validate timestamp gaps using series outside hmm before converting to the matrix.
func validateObservations(x matrix.Matrix, dimension int) (int, []float64, error) {
	if isNilMatrix(x) {
		return 0, nil, ErrNilInput
	}
	if x.Rows() == 0 {
		return 0, nil, ErrEmptyInput
	}
	v, err := readMatrix(x, x.Rows(), dimension)
	return x.Rows(), v, err
}

// normalizeFit validates and finalizes the explicit Baum-Welch training policy.
// A zero-valued option means the documented default, except Pseudocount=0 stays zero.
//
// Implementation:
//   - Stage 1: Reject negative iterations, negative or nonfinite floating options.
//   - Stage 2: Fill zero iteration, tolerance, and variance-floor fields with named defaults.
//   - Stage 3: Return the normalized value without changing the caller's copy.
//
// Behavior highlights:
//   - The floor is measured in squared feature units and is not an additive jitter.
//
// Inputs:
//   - o: value-copy FitOptions with explicit nonnegative configuration.
//
// Returns:
//   - FitOptions, error: validated settings with effective defaults.
//
// Errors:
//   - ErrInvalidOption for a negative or nonfinite field.
//
// Determinism:
//   - Option fields are evaluated in a fixed order before any training data.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - A zero Tolerance selects the default; zero cannot mean an exact-zero threshold.
//
// AI-Hints:
//   - Use Pseudocount=0 for unsmoothed EM; do not invent a hidden Dirichlet prior.
func normalizeFit(o FitOptions) (FitOptions, error) {
	if o.MaxIterations < 0 || !finite(o.Tolerance) || o.Tolerance < 0 || !finite(o.Pseudocount) || o.Pseudocount < 0 || !finite(o.VarianceFloor) || o.VarianceFloor < 0 {
		return o, ErrInvalidOption
	}
	if o.MaxIterations == 0 {
		o.MaxIterations = defaultFitMaxIterations
	}
	if o.Tolerance == 0 {
		o.Tolerance = defaultFitTolerance
	}
	if o.VarianceFloor == 0 {
		o.VarianceFloor = defaultVarianceFloor
	}
	return o, nil
}

// validState authenticates the shape and normalized log-mass of a streaming checkpoint.
// The checkpoint must belong to the exact immutable model and an already processed prefix.
//
// Implementation:
//   - Stage 1: Accept nil as the explicit start-of-stream marker.
//   - Stage 2: Reject identity/length/step/evidence violations and invalid log entries.
//   - Stage 3: Validate LogSumExp approximately zero and guard the next uint64 increment.
//
// Behavior highlights:
//   - Legitimate impossible state entries use -Inf; all-impossible checkpoints fail.
//
// Inputs:
//   - c: validated private chain; prev: caller-owned read-only saved FilterState or nil.
//
// Returns:
//   - error: nil only if the next step can safely consume the checkpoint.
//
// Errors:
//   - ErrStateVersion for incompatible or unnormalized data; ErrOverflow if Steps is MaxUint64.
//
// Determinism:
//   - LogPosterior is inspected in ascending state index order.
//
// Complexity:
//   - Time O(K), Space O(1).
//
// Notes:
//   - State integrity is value-based; the helper never mutates the exported slice.
//
// AI-Hints:
//   - Do not copy FilterState by value and then mutate shared LogPosterior storage.
func validState(c *chain, prev *FilterState) error {
	if prev == nil {
		return nil
	}
	if prev.ModelID != c.id || prev.Steps == 0 || len(prev.LogPosterior) != c.k {
		return ErrStateVersion
	}
	if !finite(prev.LogLikelihood) {
		return ErrStateVersion
	}
	for _, v := range prev.LogPosterior {
		if math.IsNaN(v) || math.IsInf(v, 1) || v > probabilityTolerance {
			return ErrStateVersion
		}
	}
	z := LogSumExpSlice(prev.LogPosterior)
	if !finite(z) || math.Abs(z) > probabilityTolerance {
		return ErrStateVersion
	}
	if prev.Steps == ^uint64(0) {
		return ErrOverflow
	}
	return nil
}
