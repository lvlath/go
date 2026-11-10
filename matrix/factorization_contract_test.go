// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test exercises reusable factorizations exclusively through public API.
// The fixtures test exact permutations, numerical rank, ownership, and residuals.
package matrix_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/matrix"
)

// TestFactorLU_PermutationMatrix fixes P*A=L*U permutation mapping on a matrix
// whose first pivot is zero, and anchors the exact solution and residual.
func TestFactorLU_PermutationMatrix(t *testing.T) {
	input := factorFixture(t, [][]float64{{0, 1}, {1, 0}})
	factor, err := matrix.FactorLU(context.Background(), input, matrix.FactorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	permutation := factor.Permutation()
	if len(permutation) != 2 || permutation[0] != 1 || permutation[1] != 0 {
		t.Fatalf("P mapping = %v, want [1 0]", permutation)
	}
	result, err := factor.Solve(context.Background(), factorFixture(t, [][]float64{{2}, {3}}))
	if err != nil {
		t.Fatal(err)
	}
	factorNear(t, factorCell(t, result.Solution, 0, 0), 3, 0, 0)
	factorNear(t, factorCell(t, result.Solution, 1, 0), 2, 0, 0)
	factorNear(t, result.ResidualNorms[0], 0, 0, 0)
	factorNear(t, result.ConditionEstimate, 1, 0, 0)
	if result.Rank != 2 {
		t.Fatalf("rank = %d, want 2", result.Rank)
	}
}

// TestFactorLU_OriginalIndexTie verifies that the pivot candidate with the
// smallest original row ID wins even when previous swaps reorder rows.
func TestFactorLU_OriginalIndexTie(t *testing.T) {
	input := factorFixture(t, [][]float64{
		{1, 1, 0},
		{3, 0, 1},
		{-3, 1, 1},
	})
	factor, err := matrix.FactorLU(context.Background(), input, matrix.FactorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	got := factor.Permutation()
	if got[0] != 1 {
		t.Fatalf("first tie should select original row 1, got %v", got)
	}
}

// TestFactorLU_MultipleRightHandSidesAndCondition establishes the exact
// identity property, detached solutions, and measured condition number.
func TestFactorLU_MultipleRightHandSidesAndCondition(t *testing.T) {
	identity := factorFixture(t, [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})
	factor, err := matrix.FactorLU(context.Background(), identity, matrix.FactorOptions{ConditionLimit: 1})
	if err != nil {
		t.Fatal(err)
	}
	b := factorFixture(t, [][]float64{{1, 4}, {-2, 5}, {3, -6}})
	result, err := factor.Solve(context.Background(), b)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			factorNear(t, factorCell(t, result.Solution, i, j), factorCell(t, b, i, j), 0, 0)
		}
	}
	factorNear(t, result.ConditionEstimate, 1, 0, 0)
	for _, residual := range result.ResidualNorms {
		factorNear(t, residual, 0, 0, 0)
	}
	result.ResidualNorms[0] = 500
	if err = result.Solution.Set(0, 0, 500); err != nil {
		t.Fatal(err)
	}
	again, err := factor.Solve(context.Background(), b)
	if err != nil || factorCell(t, again.Solution, 0, 0) != 1 || again.ResidualNorms[0] != 0 {
		t.Fatalf("result aliased retained state: again=%v err=%v", again, err)
	}
}

// TestFactorLU_DetachedPermutationAndInput proves both constructor ownership
// and returned permutation copy semantics after mutation by the caller.
func TestFactorLU_DetachedPermutationAndInput(t *testing.T) {
	input := factorFixture(t, [][]float64{{0, 1}, {1, 0}})
	factor, err := matrix.FactorLU(context.Background(), input, matrix.FactorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	factor.Permutation()[0] = 999
	if factor.Permutation()[0] != 1 {
		t.Fatal("returned permutation aliases factor storage")
	}
	if err = input.Set(0, 0, 100); err != nil {
		t.Fatal(err)
	}
	result, err := factor.Solve(context.Background(), factorFixture(t, [][]float64{{2}, {3}}))
	if err != nil {
		t.Fatal(err)
	}
	factorNear(t, factorCell(t, result.Solution, 0, 0), 3, 0, 0)
}

// TestFactorQR_TallReconstruction checks that solving multiple exact columns
// of A*x=B recovers x; additionally it verifies the least-squares residual
// is orthogonal to every column of the original overdetermined A.
func TestFactorQR_TallReconstruction(t *testing.T) {
	a := factorFixture(t, [][]float64{{1, 0}, {0, 1}, {1, 1}, {2, -1}})
	factor, err := matrix.FactorQR(context.Background(), a, matrix.FactorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	b := factorFixture(t, [][]float64{{1, 2}, {2, -1}, {3, 1}, {0, 5}})
	result, err := factor.Solve(context.Background(), b)
	if err != nil {
		t.Fatal(err)
	}
	for k, expected := range [][2]float64{{1, 2}, {2, -1}} {
		factorNear(t, factorCell(t, result.Solution, 0, k), expected[0], 1e-12, 1e-12)
		factorNear(t, factorCell(t, result.Solution, 1, k), expected[1], 1e-12, 1e-12)
		factorNear(t, result.ResidualNorms[k], 0, 1e-11, 0)
	}

	// A noisy data vector need not lie in Col(A). The normal-equation
	// orthogonality is tested as a mathematical invariant, not as an
	// algorithm: the production solver never forms normal equations.
	noisy := factorFixture(t, [][]float64{{1}, {2}, {4}, {1}})
	fit, err := factor.Solve(context.Background(), noisy)
	if err != nil {
		t.Fatal(err)
	}
	for j := 0; j < 2; j++ {
		orthogonality := 0.0
		for i := 0; i < 4; i++ {
			predicted := 0.0
			for k := 0; k < 2; k++ {
				predicted += factorCell(t, a, i, k) * factorCell(t, fit.Solution, k, 0)
			}
			orthogonality += factorCell(t, a, i, j) * (predicted - factorCell(t, noisy, i, 0))
		}
		factorNear(t, orthogonality, 0, 1e-12, 0)
	}
}

// TestFactorQR_PermutationMapping anchors the zero-based LAPACK-style column
// mapping while demonstrating that x is returned in original coordinates.
func TestFactorQR_PermutationMapping(t *testing.T) {
	a := factorFixture(t, [][]float64{{1, 10}, {0, 0}, {0, 2}})
	factor, err := matrix.FactorQR(context.Background(), a, matrix.FactorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	perm := factor.Permutation()
	if len(perm) != 2 || perm[0] != 1 || perm[1] != 0 {
		t.Fatalf("column mapping = %v, want [1 0]", perm)
	}
	perm[0] = 999
	if factor.Permutation()[0] != 1 {
		t.Fatal("QR permutation leaked private storage")
	}
	result, err := factor.Solve(context.Background(), factorFixture(t, [][]float64{{43}, {0}, {8}}))
	if err != nil {
		t.Fatal(err)
	}
	factorNear(t, factorCell(t, result.Solution, 0, 0), 3, 1e-12, 1e-12)
	factorNear(t, factorCell(t, result.Solution, 1, 0), 4, 1e-12, 1e-12)
}

// TestSolveLeastSquares_RankLoss rejects duplicated columns and wide matrices
// rather than silently returning a pseudo-inverse or arbitrary minimum-norm x.
func TestSolveLeastSquares_RankLoss(t *testing.T) {
	duplicates := factorFixture(t, [][]float64{{1, 1}, {2, 2}, {3, 3}})
	factor, err := matrix.FactorQR(context.Background(), duplicates, matrix.FactorOptions{})
	if factor != nil {
		t.Fatal("rank-deficient constructor returned factor")
	}
	factorErrorIs(t, err, matrix.ErrRankDeficient)
	result, err := matrix.SolveLeastSquares(context.Background(), duplicates, factorFixture(t, [][]float64{{1}, {2}, {3}}), matrix.FactorOptions{})
	if result != nil {
		t.Fatal("rank-deficient least squares returned result")
	}
	factorErrorIs(t, err, matrix.ErrRankDeficient)

	wide := factorFixture(t, [][]float64{{1, 2}})
	factor, err = matrix.FactorQR(context.Background(), wide, matrix.FactorOptions{})
	if factor != nil {
		t.Fatal("wide constructor returned factor")
	}
	factorErrorIs(t, err, matrix.ErrBadShape)
}

// TestFactorCholesky_SymmetricButIndefinite verifies that positive diagonal
// entries are not mistaken for the stronger positive-definite property.
func TestFactorCholesky_SymmetricButIndefinite(t *testing.T) {
	a := factorFixture(t, [][]float64{{1, 2}, {2, 1}})
	factor, err := matrix.FactorCholesky(context.Background(), a, 0, 0)
	if factor != nil {
		t.Fatal("indefinite matrix returned SPD factor")
	}
	factorErrorIs(t, err, matrix.ErrNotPositiveDefinite)
}

// TestFactorCholesky_SymmetricMidpointAndSolve checks the accepted matrix and
// verifies the exact solution of a well-conditioned SPD fixture.
func TestFactorCholesky_SymmetricMidpointAndSolve(t *testing.T) {
	a := factorFixture(t, [][]float64{{4, 2}, {2, 3}})
	factor, err := matrix.FactorCholesky(context.Background(), a, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	result, err := factor.Solve(context.Background(), factorFixture(t, [][]float64{{6}, {5}}))
	if err != nil {
		t.Fatal(err)
	}
	factorNear(t, factorCell(t, result.Solution, 0, 0), 1, 1e-14, 1e-14)
	factorNear(t, factorCell(t, result.Solution, 1, 0), 1, 1e-14, 1e-14)
	factorNear(t, result.ResidualNorms[0], 0, 1e-13, 0)

	asymmetric := factorFixture(t, [][]float64{{4, 2 + 1e-12}, {2 - 1e-12, 3}})
	if rejected, err := matrix.FactorCholesky(context.Background(), asymmetric, 0, 0); rejected != nil || !errors.Is(err, matrix.ErrAsymmetry) {
		t.Fatalf("exact symmetry was not enforced: factor=%v err=%v", rejected, err)
	}
	midpoint, err := matrix.FactorCholesky(context.Background(), asymmetric, 1e-10, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := midpoint.Solve(context.Background(), factorFixture(t, [][]float64{{6}, {5}}))
	if err != nil {
		t.Fatal(err)
	}
	factorNear(t, factorCell(t, res.Solution, 0, 0), 1, 1e-12, 1e-12)
	factorNear(t, factorCell(t, res.Solution, 1, 0), 1, 1e-12, 1e-12)
}

// TestFactorConditionEstimates distinguishes actual induced norms from a
// diagonal-ratio surrogate; identity must report exactly one for all factors.
func TestFactorConditionEstimates(t *testing.T) {
	ctx := context.Background()
	a := factorFixture(t, [][]float64{{2, 0}, {0, 1}})
	b := factorFixture(t, [][]float64{{2}, {1}})
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
	for name, solve := range map[string]func(context.Context, matrix.Matrix) (*matrix.SolveResult, error){
		"LU": lu.Solve, "QR": qr.Solve, "Cholesky": chol.Solve,
	} {
		result, err := solve(ctx, b)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		factorNear(t, result.ConditionEstimate, 2, 1e-14, 1e-14)
	}
	_, err = matrix.FactorLU(ctx, a, matrix.FactorOptions{ConditionLimit: 1.5})
	factorErrorIs(t, err, matrix.ErrIllConditioned)
	_, err = matrix.FactorQR(ctx, a, matrix.FactorOptions{ConditionLimit: 1.5})
	factorErrorIs(t, err, matrix.ErrIllConditioned)
}

// TestSolve_ResidualScaled uses a large but finite system where ordinary
// intermediate products and a naively squared residual can exceed float64.
func TestSolve_ResidualScaled(t *testing.T) {
	ctx := context.Background()
	a := factorFixture(t, [][]float64{{1e150, -1e150}, {1e150, 1e150}})
	b := factorFixture(t, [][]float64{{0}, {2e150}})
	lu, err := matrix.FactorLU(ctx, a, matrix.FactorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	result, err := lu.Solve(ctx, b)
	if err != nil {
		t.Fatal(err)
	}
	factorNear(t, factorCell(t, result.Solution, 0, 0), 1, 1e-13, 1e-13)
	factorNear(t, factorCell(t, result.Solution, 1, 0), 1, 1e-13, 1e-13)
	factorNear(t, result.ResidualNorms[0], 0, 1e138, 0)
}

// TestFactorOptionsAndCancellation anchors precedence and cancellation without
// timers, so errors remain deterministic in both ordinary and race test modes.
func TestFactorOptionsAndCancellation(t *testing.T) {
	ctx := context.Background()
	identity := factorFixture(t, [][]float64{{1}})
	if _, err := matrix.FactorLU(ctx, identity, matrix.FactorOptions{RankTolerance: math.NaN()}); !errors.Is(err, matrix.ErrInvalidOption) {
		t.Fatalf("invalid tolerance: %v", err)
	}
	if _, err := matrix.FactorQR(ctx, identity, matrix.FactorOptions{ConditionLimit: 0.5}); !errors.Is(err, matrix.ErrInvalidOption) {
		t.Fatalf("invalid condition: %v", err)
	}
	if _, err := matrix.FactorCholesky(ctx, identity, -1, 0); !errors.Is(err, matrix.ErrInvalidOption) {
		t.Fatalf("invalid SPD tolerance: %v", err)
	}
	if _, err := matrix.FactorLU(nil, identity, matrix.FactorOptions{}); !errors.Is(err, matrix.ErrInvalidOption) {
		t.Fatalf("nil context: %v", err)
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	factor, err := matrix.FactorLU(cancelled, identity, matrix.FactorOptions{})
	if factor != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled LU: factor=%v err=%v", factor, err)
	}
	qr, err := matrix.FactorQR(ctx, identity, matrix.FactorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	result, err := qr.Solve(cancelled, identity)
	if result != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled QR solve: result=%v err=%v", result, err)
	}
}

// TestFactorRankThresholdAndNilReceivers anchors accepted-relative-rank
// equality and the no-panic contract of absent factor instances.
func TestFactorRankThresholdAndNilReceivers(t *testing.T) {
	ctx := context.Background()
	a := factorFixture(t, [][]float64{{1, 0}, {0, 1e-8}})
	if f, err := matrix.FactorLU(ctx, a, matrix.FactorOptions{RankTolerance: 1e-7}); f != nil || !errors.Is(err, matrix.ErrRankDeficient) {
		t.Fatalf("LU threshold: f=%v err=%v", f, err)
	}
	if f, err := matrix.FactorQR(ctx, a, matrix.FactorOptions{RankTolerance: 1e-7}); f != nil || !errors.Is(err, matrix.ErrRankDeficient) {
		t.Fatalf("QR threshold: f=%v err=%v", f, err)
	}
	var absentLU *matrix.LUFactor
	var absentQR *matrix.QRFactor
	var absentCholesky *matrix.CholeskyFactor
	if absentLU.Permutation() != nil || absentQR.Permutation() != nil {
		t.Fatal("nil permutation receiver must return nil")
	}
	for _, solve := range []func(context.Context, matrix.Matrix) (*matrix.SolveResult, error){
		absentLU.Solve, absentQR.Solve, absentCholesky.Solve,
	} {
		if result, err := solve(ctx, a); result != nil || !errors.Is(err, matrix.ErrNilMatrix) {
			t.Fatalf("nil-factor Solve: result=%v err=%v", result, err)
		}
	}
}

// TestFactors_DetachedConcurrentSolve verifies that 64 concurrent calls
// share only immutable factor state and no mutable solution or residual slices.
func TestFactors_DetachedConcurrentSolve(t *testing.T) {
	ctx := context.Background()
	a := factorFixture(t, [][]float64{{4, 1}, {1, 3}})
	b := factorFixture(t, [][]float64{{5}, {4}})
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
	const workers = 64
	errorsChannel := make(chan error, workers*3)
	var work sync.WaitGroup
	for _, solve := range []func(context.Context, matrix.Matrix) (*matrix.SolveResult, error){lu.Solve, qr.Solve, chol.Solve} {
		for worker := 0; worker < workers; worker++ {
			work.Add(1)
			go func(run func(context.Context, matrix.Matrix) (*matrix.SolveResult, error)) {
				defer work.Done()
				result, err := run(ctx, b)
				if err == nil && result != nil {
					first, readErr := result.Solution.At(0, 0)
					if readErr != nil || math.Abs(first-1) > 1e-12 {
						err = fmt.Errorf("unexpected solution first cell=%g: %v", first, readErr)
					}
				}
				errorsChannel <- err
			}(solve)
		}
	}
	work.Wait()
	close(errorsChannel)
	for err := range errorsChannel {
		if err != nil {
			t.Fatal(err)
		}
	}
}

// TestSolveLeastSquaresValidationPrecedence ensures a missing right-hand side
// is rejected before expensive QR factorization and options always win over
// input-nil or shape failures, as required by the public validation contract.
func TestSolveLeastSquaresValidationPrecedence(t *testing.T) {
	ctx := context.Background()
	badA := factorFixture(t, [][]float64{{1, 1}, {2, 2}, {3, 3}})
	if result, err := matrix.SolveLeastSquares(ctx, badA, nil, matrix.FactorOptions{RankTolerance: math.NaN()}); result != nil || !errors.Is(err, matrix.ErrInvalidOption) {
		t.Fatalf("options precedence: result=%v err=%v", result, err)
	}
	if result, err := matrix.SolveLeastSquares(ctx, badA, nil, matrix.FactorOptions{}); result != nil || !errors.Is(err, matrix.ErrNilMatrix) {
		t.Fatalf("nil RHS must precede rank computation: result=%v err=%v", result, err)
	}

	// The receiver of a normal Matrix interface can be a typed nil *Dense.
	// Canonical ValidateNotNil must detect this without reflect.
	var typedNil *matrix.Dense
	if factor, err := matrix.FactorQR(ctx, typedNil, matrix.FactorOptions{}); factor != nil || !errors.Is(err, matrix.ErrNilMatrix) {
		t.Fatalf("typed-nil QR: factor=%v err=%v", factor, err)
	}
	validA := factorFixture(t, [][]float64{{1, 0}, {0, 1}, {1, 1}})
	oneRow := factorFixture(t, [][]float64{{1}})
	if result, err := matrix.SolveLeastSquares(ctx, validA, oneRow, matrix.FactorOptions{}); result != nil || !errors.Is(err, matrix.ErrDimensionMismatch) {
		t.Fatalf("RHS shape rejection: result=%v err=%v", result, err)
	}
}

// TestFactorCholeskySymmetryThresholdOverflow rejects an unrepresentable
// allowance for a genuinely asymmetric pair. Exact equality is covered by
// TestFactorCholeskyExactSymmetryAvoidsToleranceOverflow, because an equal
// pair satisfies the inequality without a tolerance calculation.
func TestFactorCholeskySymmetryThresholdOverflow(t *testing.T) {
	coefficients := factorFixture(t, [][]float64{{1e308, 1e308}, {5e307, 1e308}})
	factor, err := matrix.FactorCholesky(context.Background(), coefficients, 0, math.MaxFloat64)
	if factor != nil {
		t.Fatal("unrepresentable symmetry tolerance produced factor")
	}
	factorErrorIs(t, err, matrix.ErrOverflow)
}

// TestFactorQRAndCholeskyDetachedSource verifies that both models retain
// accepted snapshots and do not observe subsequent caller mutations of A.
func TestFactorQRAndCholeskyDetachedSource(t *testing.T) {
	ctx := context.Background()
	a := factorFixture(t, [][]float64{{4, 1}, {1, 3}})
	b := factorFixture(t, [][]float64{{5}, {4}})
	qr, err := matrix.FactorQR(ctx, a, matrix.FactorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	chol, err := matrix.FactorCholesky(ctx, a, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err = a.Set(0, 0, 100); err != nil {
		t.Fatal(err)
	}
	for _, solve := range []func(context.Context, matrix.Matrix) (*matrix.SolveResult, error){qr.Solve, chol.Solve} {
		result, err := solve(ctx, b)
		if err != nil {
			t.Fatal(err)
		}
		factorNear(t, factorCell(t, result.Solution, 0, 0), 1, 1e-12, 1e-12)
		factorNear(t, factorCell(t, result.Solution, 1, 0), 1, 1e-12, 1e-12)
	}
}

// TestFactorConditionEqualityAccepted confirms that the diagnostic cutoff is
// strictly greater, not greater-or-equal, for both option-bearing constructors.
func TestFactorConditionEqualityAccepted(t *testing.T) {
	a := factorFixture(t, [][]float64{{2, 0}, {0, 1}})
	options := matrix.FactorOptions{ConditionLimit: 2}
	if factor, err := matrix.FactorLU(context.Background(), a, options); factor == nil || err != nil {
		t.Fatalf("LU must accept condition == limit: %v", err)
	}
	if factor, err := matrix.FactorQR(context.Background(), a, options); factor == nil || err != nil {
		t.Fatalf("QR must accept condition == limit: %v", err)
	}
}

// TestFactorLUSingularErrorChain preserves the established ErrSingular class
// when a newly classified LU rank deficiency is caused by an exact zero pivot.
func TestFactorLUSingularErrorChain(t *testing.T) {
	a := factorFixture(t, [][]float64{{1, 2}, {2, 4}})
	factor, err := matrix.FactorLU(context.Background(), a, matrix.FactorOptions{})
	if factor != nil {
		t.Fatal("singular LU unexpectedly published a factor")
	}
	factorErrorIs(t, err, matrix.ErrRankDeficient)
	factorErrorIs(t, err, matrix.ErrSingular)
}

// TestFactorCholeskyExactSymmetryAvoidsToleranceOverflow proves that exact
// symmetric entries need no arithmetic threshold evaluation. The symmetry
// inequality is already satisfied for equal finite inputs, even when a caller
// supplies very large but finite (and therefore valid) tolerance parameters.
func TestFactorCholeskyExactSymmetryAvoidsToleranceOverflow(t *testing.T) {
	input := factorFixture(t, [][]float64{{4, 1}, {1, 3}})
	factor, err := matrix.FactorCholesky(context.Background(), input, math.MaxFloat64, math.MaxFloat64)
	if err != nil {
		t.Fatalf("exact-symmetric SPD matrix rejected: %v", err)
	}
	if factor == nil {
		t.Fatal("expected complete Cholesky factor")
	}
}
