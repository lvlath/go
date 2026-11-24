// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test verifies factor admission, original-coordinate solves and cancellation.
// These tests use public factor constructors and Solve, never fabricated packed factor storage.
package matrix_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

// TestCoverageSolveResultNilableProtocol checks the pointer-level result contract.
// A nil result is distinguishable from an allocated result even through core.Nilable.
// Implementation: Compare an absent pointer, an allocated zero result and a successful solve.
// Behavior highlights: IsNil checks pointer identity, not field contents or numerical validity.
// Inputs: A nil pointer, an empty allocated result and a 1x1 full-rank system.
// Returns: Test assertions only.
// Errors: The successful system must return no error; absence itself is not an error.
// Determinism: Constant fixtures and pure pointer comparisons.
// Complexity: O(1) for each nil check and scalar solve.
// Notes: Do not treat an allocated empty result as a valid numerical solution.
// AI-Hints: Check err before accessing Solution; an interface containing a typed nil is not == nil.
func TestCoverageSolveResultNilableProtocol(t *testing.T) {
	var missing *matrix.SolveResult
	var value core.Nilable = missing
	if value == nil || !value.IsNil() || !missing.IsNil() {
		t.Fatal("typed-nil SolveResult lost nilable semantics")
	}
	allocated := &matrix.SolveResult{}
	value = allocated
	if value.IsNil() {
		t.Fatal("allocated zero result must not be classified as nil")
	}
	a := NewFilledDense(t, 1, 1, []float64{2})
	b := NewFilledDense(t, 1, 1, []float64{6})
	result, err := matrix.SolveLeastSquares(context.Background(), a, b, matrix.FactorOptions{})
	if err != nil || result.IsNil() {
		t.Fatalf("scalar solve: result=%v err=%v", result, err)
	}
	value = result
	if value.IsNil() || MustAt(t, result.Solution, 0, 0) != 3 || result.Rank != 1 {
		t.Fatal("successful result lost its numerical or nilable contract")
	}
}

// TestCoverageFactorConstructorAdmission verifies shape, rank and context admission.
// Invalid inputs cannot construct a partially initialized factor.
// Implementation: Exercise square-only shape restrictions, zero matrices, nil receivers,
// and cancellation or missing contexts before numerical work begins.
// Behavior highlights: Nil, invalid shape, rank deficiency and indefiniteness remain distinct.
// Inputs: Small finite matrices, empty descriptors and explicitly canceled contexts.
// Returns: Test assertions only.
// Errors: Public shape, rank, context and positive-definiteness sentinels.
// Determinism: Fixed admission order without timers or fabricated private factor storage.
// Complexity: Constant fixture storage; each construction is bounded by small matrix order.
// Notes: Required condition-diagnostic overflow is covered separately with a finite witness.
// AI-Hints: Use a public constructor for every factor; never modify its packed storage.
func TestCoverageFactorConstructorAdmission(t *testing.T) {
	ctx := context.Background()
	opts := matrix.FactorOptions{}
	var absent *matrix.Dense
	if f, err := matrix.FactorLU(ctx, absent, opts); f != nil {
		t.Fatal("nil LU published factor")
	} else {
		AssertErrorIs(t, err, matrix.ErrNilMatrix)
	}
	if f, err := matrix.FactorLU(ctx, MustDense(t, 2, 1), opts); f != nil {
		t.Fatal("rectangular LU published factor")
	} else {
		AssertErrorIs(t, err, matrix.ErrNonSquare)
	}
	if f, err := matrix.FactorLU(ctx, MustDense(t, 2, 2), opts); f != nil {
		t.Fatal("zero LU published factor")
	} else {
		AssertErrorIs(t, err, matrix.ErrRankDeficient)
		AssertErrorIs(t, err, matrix.ErrSingular)
	}
	if f, err := matrix.FactorCholesky(ctx, absent, 0, 0); f != nil {
		t.Fatal("nil Cholesky published factor")
	} else {
		AssertErrorIs(t, err, matrix.ErrNilMatrix)
	}
	if f, err := matrix.FactorCholesky(ctx, MustDense(t, 2, 1), 0, 0); f != nil {
		t.Fatal("rectangular Cholesky published factor")
	} else {
		AssertErrorIs(t, err, matrix.ErrNonSquare)
	}
	if f, err := matrix.FactorCholesky(ctx, MustDense(t, 2, 2), 0, 0); f != nil {
		t.Fatal("zero matrix published Cholesky factor")
	} else {
		AssertErrorIs(t, err, matrix.ErrNotPositiveDefinite)
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if f, err := matrix.FactorQR(canceled, nil, opts); f != nil {
		t.Fatal("canceled QR published factor")
	} else {
		AssertErrorIs(t, err, context.Canceled)
	}
	if f, err := matrix.FactorCholesky(nil, nil, 0, 0); f != nil {
		t.Fatal("nil-context Cholesky published factor")
	} else {
		AssertErrorIs(t, err, matrix.ErrInvalidOption)
	}
}

// TestCoverageFactorConditionOverflow verifies nonrepresentable required condition diagnostics.
// The small pivot is nonzero but its inverse is outside the finite float64 range.
// Implementation: Permit the pivot in LU/QR with explicit rank tolerance, then require
// the inverse basis solve and condition diagnostic to refuse publication in all three factors.
// Behavior highlights: The input is finite and nonsingular in exact arithmetic.
// Inputs: diag(1,1e-310), whose 1-norm condition is approximately 1e310.
// Returns: Test assertions only.
// Errors: ErrOverflow from the required finite diagnostic.
// Determinism: Diagonal factorization, no pivot ambiguity.
// Complexity: O(1) for this two-dimensional witness.
// Notes: RankTolerance does not authorize infinite condition metadata.
// AI-Hints: This is not a regression for RHS range recovery; factor admission itself must fail.
func TestCoverageFactorConditionOverflow(t *testing.T) {
	ctx := context.Background()
	a := NewFilledDense(t, 2, 2, []float64{1, 0, 0, 1e-310})
	opts := matrix.FactorOptions{RankTolerance: math.SmallestNonzeroFloat64}
	if f, err := matrix.FactorLU(ctx, a, opts); f != nil {
		t.Fatal("LU published unrepresentable condition")
	} else {
		AssertErrorIs(t, err, matrix.ErrOverflow)
	}
	if f, err := matrix.FactorQR(ctx, a, opts); f != nil {
		t.Fatal("QR published unrepresentable condition")
	} else {
		AssertErrorIs(t, err, matrix.ErrOverflow)
	}
	if f, err := matrix.FactorCholesky(ctx, a, 0, 0); f != nil {
		t.Fatal("Cholesky published unrepresentable condition")
	} else {
		AssertErrorIs(t, err, matrix.ErrOverflow)
	}
}

// TestCoverageLeastSquaresAdmissionStages verifies both input domains before QR work.
// Separate malformed A and B fixtures protect the exact public error precedence.
// Implementation: Exercise nil context, nil coefficient matrix, empty operands, wide design,
// nonfinite data and failures from each backend's finite scan.
// Behavior highlights: No input-validation failure returns a partial SolveResult.
// Inputs: Full-rank tall design and compatible RHS, plus explicit invalid variants.
// Returns: Test assertions only.
// Errors: ErrInvalidOption, ErrNilMatrix, ErrBadShape, ErrNaNInf, ErrDimensionMismatch and backend errors.
// Determinism: Ordered table; options and shapes are checked before numeric scans.
// Complexity: O(m*n+m*r) validation, bounded small allocations for fixtures.
// Notes: This tests the combined facade, not only FactorQR in isolation.
// AI-Hints: Use a faulting RHS with a valid design to reach its independent finite scan.
func TestCoverageLeastSquaresAdmissionStages(t *testing.T) {
	ctx := context.Background()
	a := NewFilledDense(t, 3, 2, []float64{1, 0, 0, 1, 1, 1})
	b := NewFilledDense(t, 3, 1, []float64{2, 3, 5})
	for _, tc := range []struct {
		name string
		ctx  context.Context
		a, b matrix.Matrix
		want error
	}{
		{"nil-context", nil, a, b, matrix.ErrInvalidOption},
		{"nil-design", ctx, nil, b, matrix.ErrNilMatrix},
		{"empty-design", ctx, MustDense(t, 0, 2), b, matrix.ErrBadShape},
		{"empty-rhs", ctx, a, MustDense(t, 3, 0), matrix.ErrBadShape},
		{"wide-design", ctx, MustDense(t, 1, 2), MustDense(t, 1, 1), matrix.ErrBadShape},
		{"rhs-shape", ctx, a, MustDense(t, 2, 1), matrix.ErrDimensionMismatch},
		{"nonfinite-design", ctx, coverageRawDense(t, 3, 2, []float64{1, 0, 0, 1, math.NaN(), 1}), b, matrix.ErrNaNInf},
		{"nonfinite-rhs", ctx, a, coverageRawDense(t, 3, 1, []float64{2, 3, math.Inf(1)}), matrix.ErrNaNInf},
		{"design-read", ctx, &coverageMatrix{Matrix: a, read: coverageFaultAt(2, 1, 0)}, b, errCoverageBackend},
		{"rhs-read", ctx, a, &coverageMatrix{Matrix: b, read: coverageFaultAt(2, 0, 0)}, errCoverageBackend},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := matrix.SolveLeastSquares(tc.ctx, tc.a, tc.b, matrix.FactorOptions{})
			AssertErrorIs(t, err, tc.want)
			if result != nil {
				t.Fatal("failed least squares published result")
			}
		})
	}
}

// TestCoverageCholeskyRepresentableMidpoint verifies the accepted symmetric system near MaxFloat64.
// The naive sum of off-diagonal entries overflows, but their midpoint and the SPD solution are finite.
// Implementation: Factor an explicitly tolerated asymmetric matrix, solve its exact accepted
// midpoint system, and compare residual metadata with an independent rational oracle.
// Behavior highlights: Residuals refer to the accepted midpoint, not the asymmetric raw coefficients.
// Inputs: Powers of two times dyadic coefficients; expected solution is exactly (1/4,-1/4).
// Returns: Test assertions only.
// Errors: No failure for this representable SPD system.
// Determinism: Exactly representable input coefficients and midpoint.
// Complexity: O(n^3) factorization, O(n^2) oracle work and storage.
// Notes: A stable midpoint is not equivalent to adding two extreme magnitudes before division.
// AI-Hints: Preserve accepted matrix semantics when interpreting Cholesky ResidualNorms.
func TestCoverageCholeskyRepresentableMidpoint(t *testing.T) {
	scale := math.Ldexp(1, 1023)
	a := NewFilledDense(t, 2, 2, []float64{1.75 * scale, 1.25 * scale, 1.5 * scale, 1.75 * scale})
	accepted := NewFilledDense(t, 2, 2, []float64{1.75 * scale, 1.375 * scale, 1.375 * scale, 1.75 * scale})
	b := NewFilledDense(t, 2, 1, []float64{0.09375 * scale, -0.09375 * scale})
	factor, err := matrix.FactorCholesky(context.Background(), a, 0.25*scale, 0)
	if err != nil {
		t.Fatal(err)
	}
	result, err := factor.Solve(context.Background(), b)
	if err != nil || result == nil || len(result.ResidualNorms) != 1 {
		t.Fatalf("solve result=%v err=%v", result, err)
	}
	factorNear(t, MustAt(t, result.Solution, 0, 0), 0.25, 0, coverageRelativeTolerance)
	factorNear(t, MustAt(t, result.Solution, 1, 0), -0.25, 0, coverageRelativeTolerance)
	want := coverageResidualOracle(t, accepted, result.Solution, b, 0)
	factorNear(t, result.ResidualNorms[0], want, 0, coverageRelativeTolerance)
	CompareExact(t, [][]float64{{1.75 * scale, 1.25 * scale}, {1.5 * scale, 1.75 * scale}}, a)
}

// TestCoverageCholeskyAdditiveToleranceOverflow isolates threshold addition from multiplication.
// Both tolerance terms are finite, but their required sum cannot be represented.
// Implementation: Supply unequal finite off-diagonals and a huge absolute plus relative tolerance.
// Behavior highlights: ErrOverflow must not silently admit an infinite symmetry threshold.
// Inputs: SPD-scale 2x2 coefficients and finite tolerance values.
// Returns: Test assertions only.
// Errors: ErrOverflow before any factor is published.
// Determinism: One unequal off-diagonal pair.
// Complexity: O(1) for this fixture.
// Notes: Equal pairs bypass unnecessary tolerance arithmetic in the accepted contract.
// AI-Hints: Test the addition overflow independently from relTol*magnitude overflow.
func TestCoverageCholeskyAdditiveToleranceOverflow(t *testing.T) {
	a := NewFilledDense(t, 2, 2, []float64{3, 1, 0.5, 3})
	f, err := matrix.FactorCholesky(context.Background(), a, math.MaxFloat64, math.MaxFloat64/2)
	AssertErrorIs(t, err, matrix.ErrOverflow)
	if f != nil {
		t.Fatal("unrepresentable threshold published Cholesky factor")
	}
}

// TestCoverageWideRHSNontrivialReflectors protects small coordinates during a tall QR solve.
// Orthogonal but non-coordinate design columns require real Householder reflectors without
// mixing the physically independent large and tiny response channels.
// Implementation: Fit two disjoint sensor groups with dyadic RHS amplitudes, verify every
// physical coefficient, and compare each residual norm with an independent rational reduction.
// Behavior highlights: A shared RHS scale cannot erase the second group's nonzero solution.
// Inputs: Four observations, two design columns, and a three-column RHS spanning 1000 binary orders.
// Returns: Test assertions only.
// Errors: The full-rank, representable solve must succeed.
// Determinism: Exact input dyadics and unique column independence; no RNG.
// Complexity: O(m*n^2+n^3+m*n*r) plus multiprecision solve and residual work.
// Notes: Small components are tested with zero absolute tolerance to forbid a false zero success.
// AI-Hints: Compare in original coordinates, not against a normalized RHS or a normal-equation solve.
func TestCoverageWideRHSNontrivialReflectors(t *testing.T) {
	large := math.Ldexp(1, 500)
	tiny := math.Ldexp(1, -500)
	a := NewFilledDense(t, 4, 2, []float64{3, 0, 4, 0, 0, 1, 0, 1})
	b := NewFilledDense(t, 4, 3, []float64{3 * large, 0, 6, 4 * large, 0, 8, tiny, 0, -1, tiny, 0, -1})
	result, err := matrix.SolveLeastSquares(context.Background(), a, b, matrix.FactorOptions{})
	if err != nil || result == nil || result.Rank != 2 || len(result.ResidualNorms) != 3 {
		t.Fatalf("wide-range RHS solve: result=%v err=%v", result, err)
	}
	want := [][]float64{{large, 0, 2}, {tiny, 0, -1}}
	if result.Solution.Rows() != 2 || result.Solution.Cols() != 3 {
		t.Fatal("solution shape mismatch")
	}
	for row := range want {
		for col, expected := range want[row] {
			factorNear(t, MustAt(t, result.Solution, row, col), expected, 0, coverageRelativeTolerance)
		}
	}
	for col, residual := range result.ResidualNorms {
		factorNear(t, residual, coverageResidualOracle(t, a, result.Solution, b, col), 0, coverageRelativeTolerance)
	}
}

// TestCoverageFactorConstructionCancellationCheckpoints exercises every observed constructor poll.
// The deterministic cancellation is delivered through a real context rather than timeout guesses.
// Implementation: Run each public constructor successfully, then replay cancellation at every poll.
// Behavior highlights: No canceled factor construction may publish a reusable factor.
// Inputs: A well-conditioned finite 3x3 SPD matrix and default options.
// Returns: Test assertions only.
// Errors: Original context.Canceled at every observed interruption.
// Determinism: No sleeps; fixed input and sequential replay.
// Complexity: O(P*n^3) for P observed polls, O(n^2) live factor storage.
// Notes: The dynamic poll count avoids test dependency on hard-coded private offsets.
// AI-Hints: Keep cancellation testing separate from rank and arithmetic failures.
func TestCoverageFactorConstructionCancellationCheckpoints(t *testing.T) {
	a := NewFilledDense(t, 3, 3, []float64{4, 1, 0, 1, 3, 0, 0, 0, 2})
	for _, tc := range []struct {
		name string
		call func(context.Context) (bool, error)
	}{
		{"lu", func(ctx context.Context) (bool, error) {
			f, err := matrix.FactorLU(ctx, a, matrix.FactorOptions{})
			return f != nil, err
		}},
		{"qr", func(ctx context.Context) (bool, error) {
			f, err := matrix.FactorQR(ctx, a, matrix.FactorOptions{})
			return f != nil, err
		}},
		{"cholesky", func(ctx context.Context) (bool, error) {
			f, err := matrix.FactorCholesky(ctx, a, 0, 0)
			return f != nil, err
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			coverageReplayCancellation(t, tc.call)
		})
	}
}

// TestCoverageFactorSolveCancellationCheckpoints protects fast and extended-range workspaces.
// Each successful public solve is replayed with cancellation at every observed cooperative poll.
// Implementation: Reuse immutable factors on both ordinary RHS and widely separated coordinates;
// count and replay all public context checks independently of their private implementation offsets.
// Behavior highlights: Interrupted wide arithmetic and residual verification publish no partial result.
// Inputs: A finite block SPD system, an ordinary RHS, and one RHS with a 1000-order spread.
// Returns: Test assertions only.
// Errors: context.Canceled for every replay; subsequent uncanceled solves still succeed.
// Determinism: Fixed matrix, fixed solver order, sequential context polling.
// Complexity: O(P*Tsolve) for each small fixture, with bounded per-call workspace.
// Notes: The wide RHS genuinely needs exponent-range handling because its relative small entry is below machine spacing.
// AI-Hints: Test through the public factor.Solve methods; never export solveWide for coverage.
func TestCoverageFactorSolveCancellationCheckpoints(t *testing.T) {
	ctx := context.Background()
	a := NewFilledDense(t, 3, 3, []float64{4, 1, 0, 1, 3, 0, 0, 0, 2})
	lu, err := matrix.FactorLU(ctx, a, matrix.FactorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	qr, err := matrix.FactorQR(ctx, a, matrix.FactorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	chol, err := matrix.FactorCholesky(ctx, a, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	large := math.Ldexp(1, 500)
	tiny := math.Ldexp(1, -500)
	for _, solver := range []struct {
		name string
		call func(context.Context, matrix.Matrix) (*matrix.SolveResult, error)
	}{{"lu", lu.Solve}, {"qr", qr.Solve}, {"cholesky", chol.Solve}} {
		t.Run(solver.name, func(t *testing.T) {
			for _, tc := range []struct {
				name string
				b    matrix.Matrix
			}{{"fast", NewFilledDense(t, 3, 1, []float64{5, 4, 2})},
				{"wide", NewFilledDense(t, 3, 2, []float64{3 * large, 0, -2 * large, 0, 2 * tiny, 0})}} {
				t.Run(tc.name, func(t *testing.T) {
					coverageReplayCancellation(t, func(runCtx context.Context) (bool, error) {
						result, runErr := solver.call(runCtx, tc.b)
						return result != nil, runErr
					})
					// A canceled call cannot poison a reusable factor's private state.
					result, runErr := solver.call(ctx, tc.b)
					if runErr != nil || result == nil || len(result.ResidualNorms) != tc.b.Cols() {
						t.Fatalf("factor reuse after cancellation: result=%v err=%v", result, runErr)
					}
					for col, residual := range result.ResidualNorms {
						factorNear(t, residual, coverageResidualOracle(t, a, result.Solution, tc.b, col), 0, coverageRelativeTolerance)
					}
					if tc.name == "wide" {
						for row, expected := range []float64{large, -large, tiny} {
							factorNear(t, MustAt(t, result.Solution, row, 0), expected, 0, coverageRelativeTolerance)
							factorNear(t, MustAt(t, result.Solution, row, 1), 0, 0, 0)
						}
					}
				})
			}
		})
	}
}

// TestCoverageLeastSquaresCancellationCheckpoints verifies cancellation of the complete convenience API.
// It covers both factor construction and subsequent application to an independent RHS.
// Implementation: Replay observed cooperative checkpoints on a full-rank tall system.
// Behavior highlights: Interrupted least squares publishes no solution or diagnostics.
// Inputs: Three observations, two independent columns and one exactly consistent RHS.
// Returns: Test assertions only.
// Errors: context.Canceled at every observed interruption boundary.
// Determinism: Fresh single-owner cancellation context for each replay.
// Complexity: O(P*T) work, where P is the observed poll count and T the small solve cost.
// Notes: The original design matrix and RHS remain unchanged.
// AI-Hints: Exercise the convenience facade as well as reusable QR factor methods.
func TestCoverageLeastSquaresCancellationCheckpoints(t *testing.T) {
	a := NewFilledDense(t, 3, 2, []float64{1, 0, 0, 1, 1, 1})
	b := NewFilledDense(t, 3, 1, []float64{2, 3, 5})
	coverageReplayCancellation(t, func(ctx context.Context) (bool, error) {
		result, err := matrix.SolveLeastSquares(ctx, a, b, matrix.FactorOptions{})
		return result != nil, err
	})
	CompareExact(t, [][]float64{{1, 0}, {0, 1}, {1, 1}}, a)
	CompareExact(t, [][]float64{{2}, {3}, {5}}, b)
}
