// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch projects parameter candidates onto the stationary linear polytope.
// Active-set KKT equalities delegate to matrix's deterministic pivoted-QR solver.
package gjrgarch

import (
	"context"
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

const (
	// constraintCount is the number of linear inequalities for Alpha, Gamma and Beta.
	constraintCount = 4
	// weightCount is the dimension of the variance-weight projection subproblem.
	weightCount = 3
	// activeMaskCount enumerates every subset of constraintCount inequalities.
	activeMaskCount = 1 << constraintCount
	// kktTolerance accommodates binary64 QR roundoff, not an economic slack policy.
	kktTolerance = 32 * (0x1p-52)
	// maxActiveULPRepairs bounds rounding reconstruction along an active face.
	maxActiveULPRepairs = 128
	// bfgsCurvatureTolerance prevents unstable inverse updates near zero curvature.
	bfgsCurvatureTolerance = 1e-12
)

// constraintRows returns a fresh fixed-order copy of the Ax>=b inequality normals.
// Row indices mean Alpha>=0, Beta>=0, Alpha+Gamma>=0, and rho<=1-margin.
//
// Coordinates are [Alpha,Gamma,Beta], distinct from the full five-parameter vector.
// Time and space are O(1); no mutable global policy is retained.
//
// Implementation:
//   - Stage 1: Build fixed 4x3 coefficient normals for the Ax>=b feasible polytope.
//
// Behavior highlights:
//   - Constraints couple Alpha and Gamma and bound stationary rho.
//
// Inputs:
//   - None; numeric policy is fixed by package constants.
//
// Returns:
//   - [4][3]float64 value, independent per invocation.
//
// Errors:
//   - None.
//
// Determinism:
//   - Rows Alpha, Beta, Alpha+Gamma, negative rho in that exact order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Omega and Mean are not part of this 3D subproblem.
//
// AI-Hints:
//   - Do not swap inequality signs when adding a stationarity row.
func constraintRows() [constraintCount][weightCount]float64 {
	return [constraintCount][weightCount]float64{
		{1, 0, 0},      // Alpha must be nonnegative.
		{0, 0, 1},      // Beta must be nonnegative.
		{1, 1, 0},      // Alpha+Gamma must be nonnegative.
		{-1, -0.5, -1}, // Negative rho enforces the strict stationarity margin.
	}
}

// inequalityBounds returns fixed Ax>=b right-hand sides in the above row order.
// The negative stationarity bound is the only nonzero right-hand side.
//
// Implementation:
//   - Stage 1: Build fixed four halfspace lower bounds including the stationarity margin.
//
// Behavior highlights:
//   - Only final bound is nonzero and negative because of Ax>=b orientation.
//
// Inputs:
//   - None; uses stationarityMargin constant.
//
// Returns:
//   - [4]float64 independent right-hand-side vector.
//
// Errors:
//   - None.
//
// Determinism:
//   - Matches constraintRows exactly.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - The bound is an optimizer policy, not New constructor validation.
//
// AI-Hints:
//   - Changing margin changes feasible set and must not occur silently.
func inequalityBounds() [constraintCount]float64 {
	return [constraintCount]float64{0, 0, 0, -(1 - stationarityMargin)}
}

// dotWeight computes a fixed three-coordinate dot product.
// It avoids a generic matrix allocation for repeated inequality checks.
//
// Implementation:
//   - Stage 1: Compute a 3-coordinate product in fixed ascending order.
//
// Behavior highlights:
//   - Avoids heap matrices for cheap primal and Gram calculations.
//
// Inputs:
//   - x,y: weight-space vectors [Alpha,Gamma,Beta].
//
// Returns:
//   - float64 scalar product.
//
// Errors:
//   - None directly; caller validates final finite results.
//
// Determinism:
//   - Terms in Alpha, Gamma, Beta order.
//
// Complexity:
//   - Time O(3), Space O(1).
//
// Notes:
//   - This is not the five-dimensional optimizer dot product.
//
// AI-Hints:
//   - Do not accidentally use Mean/Omega coordinates here.
func dotWeight(x, y [weightCount]float64) float64 {
	return x[0]*y[0] + x[1]*y[1] + x[2]*y[2]
}

// activeIndices lists active constraint IDs in stable ascending order.
// Constraints with more than weightCount active normals are necessarily dependent.
//
// Implementation:
//   - Stage 1: Scan active-bit mask from lowest row to highest.
//   - Stage 2: Collect IDs into fixed array.
//
// Behavior highlights:
//   - Any set with >3 active normals cannot be linearly independent in 3D.
//
// Inputs:
//   - mask: deterministic subset of four constraint rows.
//
// Returns:
//   - Sorted active IDs and count; unused array slots are ignored.
//
// Errors:
//   - None; mask enumerated within solver range.
//
// Determinism:
//   - Ascending row IDs make active-set tie-break reproducible.
//
// Complexity:
//   - Time O(4), Space O(4) fixed.
//
// Notes:
//   - Rank checks still occur when count<=3.
//
// AI-Hints:
//   - Do not iterate active constraints from a map.
func activeIndices(mask int) ([constraintCount]int, int) {
	var ids [constraintCount]int
	count := 0
	for i := 0; i < constraintCount; i++ {
		if mask&(1<<i) != 0 {
			ids[count] = i
			count++
		}
	}
	return ids, count
}

// satisfiesInequalities checks the numeric halfspaces with strictly feasible output.
// Tolerance is used only to screen floating candidate errors, not to publish
// negative weights or a model outside the stationary polytope.
//
// Implementation:
//   - Stage 1: Check nonnegative Alpha/Beta/downside weight.
//   - Stage 2: Compute positive-term persistence.
//   - Stage 3: Enforce margin.
//
// Behavior highlights:
//   - Final primal feasibility is strict and not softened by QR tolerance.
//
// Inputs:
//   - point: weight coordinates [Alpha,Gamma,Beta].
//
// Returns:
//   - bool: true only for an admissible finite stationary weight vector.
//
// Errors:
//   - None; returns false on bad arithmetic or violated constraints.
//
// Determinism:
//   - Fixed sign checks then stable positive-term rho.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is a public-policy verification gate for private projected candidates.
//
// AI-Hints:
//   - Do not add an arbitrary tolerance to rho<=1-margin after KKT projection.
func satisfiesInequalities(point [weightCount]float64) bool {
	if point[0] < 0 || point[2] < 0 || point[0]+point[1] < 0 {
		return false
	}
	// Reuse the cancellation-resistant rho law from parameter validation.
	rho := point[2] + point[0]*0.5 + (point[0]+point[1])*0.5
	return finite(rho) && rho <= 1-stationarityMargin
}

// solveActiveSet projects onto the affine face determined by active normals.
// It solves (A_active*A_active^T)*lambda = b_active-A_active*x and then
// reconstructs y=x+A_active^T*lambda without forming a matrix inverse.
//
// Implementation:
//   - Stage 1: assemble a small symmetric Gram system in fixed row order.
//   - Stage 2: call matrix.SolveLeastSquares, retaining its pivot/condition policy.
//   - Stage 3: add normal displacements and return the KKT multipliers.
//
// Behavior highlights:
//   - Solves Euclidean projection onto the selected equality face, not normal equations for QMLE itself.
//
// Inputs:
//   - ctx: active context; point: initial 3D weights; ids/count: selected active inequality rows.
//
// Returns:
//   - Projected 3D candidate and active multipliers or zero arrays/error.
//
// Errors:
//   - Rank deficiency/ill-conditioning from matrix means this active set is unusable.
//   - Other matrix failures remain detectable with errors.Is in the caller.
//
// Determinism:
//   - Rows and dual variables follow ascending active IDs.
//
// Complexity:
//   - Time O(k^3) plus matrix residual verification, Space O(k^2), k<=3.
//
// Notes:
//   - k<=3; QR may reject a dependent active set.
//
// AI-Hints:
//   - Do not replace the Gram solve with clipping each parameter independently.
func solveActiveSet(
	ctx context.Context,
	point [weightCount]float64,
	ids [constraintCount]int,
	count int,
) ([weightCount]float64, [constraintCount]float64, error) {
	rows := constraintRows()
	bounds := inequalityBounds()
	gram, err := matrix.NewDense(count, count)
	if err != nil {
		return [weightCount]float64{}, [constraintCount]float64{}, err
	}
	rhs, err := matrix.NewDense(count, 1)
	if err != nil {
		return [weightCount]float64{}, [constraintCount]float64{}, err
	}

	for i := 0; i < count; i++ {
		row := rows[ids[i]]
		if err = rhs.Set(i, 0, bounds[ids[i]]-dotWeight(row, point)); err != nil {
			return [weightCount]float64{}, [constraintCount]float64{}, err
		}
		for j := 0; j < count; j++ {
			if err = gram.Set(i, j, dotWeight(row, rows[ids[j]])); err != nil {
				return [weightCount]float64{}, [constraintCount]float64{}, err
			}
		}
	}

	solved, err := matrix.SolveLeastSquares(ctx, gram, rhs, matrix.FactorOptions{})
	if err != nil {
		return [weightCount]float64{}, [constraintCount]float64{}, err
	}
	candidate := point
	var multipliers [constraintCount]float64
	for i := 0; i < count; i++ {
		multiplier, solveErr := solved.Solution.At(i, 0)
		if solveErr != nil {
			return [weightCount]float64{}, [constraintCount]float64{}, solveErr
		}
		if !finite(multiplier) {
			return [weightCount]float64{}, [constraintCount]float64{}, ErrOverflow
		}
		multipliers[i] = multiplier
		row := rows[ids[i]]
		for j := 0; j < weightCount; j++ {
			candidate[j] += multiplier * row[j]
		}
	}
	return candidate, multipliers, nil
}

// kktAdmissible verifies nonnegative dual multipliers and active equality residuals.
// Small Gram/QR residuals are accepted for face reconstruction only; the final
// primal inequalities are certified separately without tolerance.
//
// Implementation:
//   - Stage 1: Verify dual multipliers >= -QR tolerance and active-equality residual sizes.
//
// Behavior highlights:
//   - Tolerances only guard floating QR reconstruction; final primal inequalities remain strict.
//
// Inputs:
//   - candidate: tentative weights; multipliers: dual values; ids/count: active face.
//
// Returns:
//   - bool: whether dual/equality KKT tests are satisfied.
//
// Errors:
//   - None; failed numerical certificate returns false.
//
// Determinism:
//   - Ascending active IDs, no adaptive tie breaking.
//
// Complexity:
//   - Time O(k*3), Space O(1).
//
// Notes:
//   - Primal feasibility is evaluated later and without tolerance.
//
// AI-Hints:
//   - Do not interpret this check as full QMLE convergence.
func kktAdmissible(
	candidate [weightCount]float64,
	multipliers [constraintCount]float64,
	ids [constraintCount]int,
	count int,
) bool {
	rows := constraintRows()
	bounds := inequalityBounds()
	for i := 0; i < count; i++ {
		if multipliers[i] < -kktTolerance {
			return false
		}
		if math.Abs(dotWeight(rows[ids[i]], candidate)-bounds[ids[i]]) > kktTolerance {
			return false
		}
	}
	return true
}

// certifyActiveFace restores exact-bound coordinates lost to final IEEE rounding.
// It repairs only active face equations within the documented QR residual
// tolerance; it never changes any variance produced by the likelihood kernel.
//
// The final stationarity gap is made strictly feasible by moving a FREE weight
// at most a few ULPs inward, preserving the selected face's other equations.
// This deterministic final check is part of KKT projection, not a fitted-model
// or likelihood post-processing clamp.
//
// Implementation:
//   - Stage 1: Restore exact active lower-bound identities.
//   - Stage 2: Nudge only free parameters inward at most 128 ULPs.
//
// Behavior highlights:
//   - This adjusts floating KKT geometry, never a conditional variance value.
//
// Inputs:
//   - point: tentative projected weights; mask: binding inequality subset.
//
// Returns:
//   - Corrected feasible weights and bool indicating strict certification.
//
// Errors:
//   - None; unrepairable rounding returns false.
//
// Determinism:
//   - Fixed correction priority Beta, Gamma, then paired Alpha/Gamma.
//
// Complexity:
//   - Time O(128), Space O(1) bounded.
//
// Notes:
//   - Cannot repair arbitrary infeasible points; input must already pass KKT tolerance.
//
// AI-Hints:
//   - Do not use this routine to rescue a likelihood trial below its variance floor.
func certifyActiveFace(point [weightCount]float64, mask int) ([weightCount]float64, bool) {
	if mask&1 != 0 { // Binding Alpha=0.
		point[0] = 0
	}
	if mask&2 != 0 { // Binding Beta=0.
		point[2] = 0
	}
	if mask&4 != 0 { // Binding Alpha+Gamma=0.
		point[1] = -point[0]
	}
	if mask&8 != 0 { // Binding stationary-margin hyperplane.
		for attempt := 0; attempt < maxActiveULPRepairs; attempt++ {
			if satisfiesInequalities(point) {
				return point, true
			}
			switch {
			case mask&2 == 0 && point[2] > 0:
				// Decrease a free Beta without disturbing other active rows.
				point[2] = math.Nextafter(point[2], math.Inf(-1))
			case mask&4 == 0:
				// Gamma is free; decreasing Gamma decreases the persistence.
				point[1] = math.Nextafter(point[1], math.Inf(-1))
			case mask&1 == 0 && point[0] > 0:
				// With Alpha+Gamma binding, move Alpha and Gamma together.
				point[0] = math.Nextafter(point[0], math.Inf(-1))
				point[1] = -point[0]
			default:
				return point, false
			}
		}
	}
	return point, satisfiesInequalities(point)
}

// nearestActiveSet enumerates small KKT faces and selects the nearest valid point.
// Failed rank/condition cases are skipped; unrelated matrix errors are preserved.
//
// Implementation:
//   - Stage 1: Enumerate each nonempty active mask.
//   - Stage 2: Solve accepted faces.
//   - Stage 3: Select minimal squared displacement.
//
// Behavior highlights:
//   - Rank-deficient faces are skipped rather than silently pseudo-inverted.
//
// Inputs:
//   - ctx: active context; point: infeasible 3D weight proposal.
//
// Returns:
//   - Closest certified 3D projection, or zero/error.
//
// Errors:
//   - ErrDomain for no certified face or wrapped matrix cause; ErrOverflow for invalid distance; context errors.
//
// Determinism:
//   - Masks 1..15; strict distance improvement retains first equal candidate.
//
// Complexity:
//   - At most 15 face masks, each with a 3x3 pivoted-QR solve.
//
// Notes:
//   - Only called when simple inequality check has already rejected input point.
//
// AI-Hints:
//   - Do not replace active-set selection with per-coordinate clipping.
func nearestActiveSet(ctx context.Context, point [weightCount]float64) ([weightCount]float64, error) {
	var best [weightCount]float64
	bestDistance := math.Inf(1)
	found := false
	for mask := 1; mask < activeMaskCount; mask++ {
		if err := ctx.Err(); err != nil {
			return [weightCount]float64{}, err
		}
		ids, count := activeIndices(mask)
		if count > weightCount {
			continue
		}
		candidate, multipliers, err := solveActiveSet(ctx, point, ids, count)
		if err != nil {
			if errors.Is(err, matrix.ErrRankDeficient) || errors.Is(err, matrix.ErrIllConditioned) {
				continue
			}
			// Preserve BOTH the package-level failure class and the matrix cause.
			return [weightCount]float64{}, errors.Join(ErrDomain, err)
		}
		if !kktAdmissible(candidate, multipliers, ids, count) {
			continue
		}
		// Restore any exact active face lost by the QR floating-point output.
		candidate, feasible := certifyActiveFace(candidate, mask)
		if !feasible {
			continue
		}
		var squaredDistance float64
		for i := 0; i < weightCount; i++ {
			delta := candidate[i] - point[i]
			squaredDistance += delta * delta
		}
		if !finite(squaredDistance) {
			return [weightCount]float64{}, ErrOverflow
		}
		// Strict comparison makes the lowest mask the deterministic tie-breaker.
		if squaredDistance < bestDistance {
			bestDistance = squaredDistance
			best = candidate
			found = true
		}
	}
	if !found {
		return [weightCount]float64{}, ErrDomain
	}
	return best, nil
}

// project returns the exact-halfspace active-set Euclidean projection of x.
// Mean is unbounded and Omega is independently projected onto [floor,+Inf).
//
// Implementation:
//   - Stage 1: validate finite arithmetic and project the separable Omega bound.
//   - Stage 2: return the input unchanged if all variance weights are feasible.
//   - Stage 3: enumerate active KKT faces and publish the nearest admissible point.
//
// Behavior highlights:
//   - The fully coupled Alpha/Gamma/Beta projection obeys Euclidean geometry.
//
// Inputs:
//   - ctx: active; x: normalized five-parameter proposal; floor: feasible positive Omega bound.
//
// Returns:
//   - vector: projected admissible coordinates or zero/error.
//
// Errors:
//   - ErrOverflow on nonfinite proposal; ErrDomain/matrix/context errors from KKT.
//
// Determinism:
//   - Ascending active-mask order; distance ties preserve the first admissible mask.
//
// Complexity:
//   - Time O(1) in parameter dimension, with at most 15 small QR solves.
//   - Space O(1) bounded by at most 3x3 KKT matrices.
//
// Notes:
//   - This function NEVER post-hoc clamps a likelihood variance h[t].
//
// AI-Hints:
//   - Independent clipping of Alpha, Gamma and Beta violates coupled constraints.
func project(ctx context.Context, x vector, floor float64) (vector, error) {
	if err := ctx.Err(); err != nil {
		return vector{}, err
	}
	if !allFinite(x) {
		return vector{}, ErrOverflow
	}
	if x[coordinateOmega] < floor {
		x[coordinateOmega] = floor
	}
	point := [weightCount]float64{x[coordinateAlpha], x[coordinateGamma], x[coordinateBeta]}
	if satisfiesInequalities(point) {
		return x, nil
	}
	best, err := nearestActiveSet(ctx, point)
	if err != nil {
		return vector{}, err
	}
	x[coordinateAlpha], x[coordinateGamma], x[coordinateBeta] = best[0], best[1], best[2]
	return x, nil
}

// projectedStationary checks the scaled projected-gradient stopping criterion.
// The test compares ||x-Project(x-gradient)|| against tol*(1+||x||).
//
// Implementation:
//   - Stage 1: Subtract analytic gradient.
//   - Stage 2: Project trial.
//   - Stage 3: Measure Euclidean displacement from accepted point.
//
// Behavior highlights:
//   - Tests the constrained first-order condition instead of raw gradient norm.
//
// Inputs:
//   - ctx: active; x/gradient: normalized five coordinates; floor/tolerance: optimizer policy.
//
// Returns:
//   - bool convergence test and optional projection error.
//
// Errors:
//   - Projection/context errors propagated without a false Converged flag.
//
// Determinism:
//   - Fixed coordinate operations and one KKT projection.
//
// Complexity:
// O(1) parameter work plus one active-set projection.
//
// Notes:
//   - Tolerance is multiplied by 1+norm(x), preserving a scale-aware criterion.
//
// AI-Hints:
//   - A small objective change is not a valid replacement for this test.
func projectedStationary(ctx context.Context, x, gradient vector, floor, tolerance float64) (bool, error) {
	var trial vector
	for i := range x {
		trial[i] = x[i] - gradient[i]
	}
	projection, err := project(ctx, trial, floor)
	if err != nil {
		return false, err
	}
	for i := range x {
		projection[i] -= x[i]
	}
	return norm(projection) <= tolerance*(1+norm(x)), nil
}

// identityHessian constructs the deterministic initial inverse-BFGS approximation.
// The identity is also the safe curvature-failure restart matrix.
//
// Implementation:
//   - Stage 1: Set only the five diagonal entries of a zero 5x5 array to one.
//
// Behavior highlights:
//   - Initial and curvature-failure inverse Hessian has no hidden randomness.
//
// Inputs:
//   - None.
//
// Returns:
//   - [5][5]float64 identity matrix value.
//
// Errors:
//   - None.
//
// Determinism:
//   - Ascending diagonal index order.
//
// Complexity:
//   - Time O(5), Space O(25) fixed stack value.
//
// Notes:
//   - No existing optimizer history is read.
//
// AI-Hints:
//   - Restart only Hessian metric, never parameter point or likelihood.
func identityHessian() [coordinateCount][coordinateCount]float64 {
	var result [coordinateCount][coordinateCount]float64
	for i := range result {
		result[i][i] = 1
	}
	return result
}

// bfgsInverseUpdate applies a checked rank-two inverse-Hessian secant update.
// Tiny curvature or nonfinite updates restart the approximation at identity.
//
// Implementation:
//   - Stage 1: validate y^T*s and reject nonpositive/insufficient curvature.
//   - Stage 2: compute H*y and y^T*H*y using the prior matrix.
//   - Stage 3: update every matrix cell and restart if the result is nonfinite.
//
// Behavior highlights:
//   - Curvature failure restores identity rather than accepting nonfinite Hessian entries.
//
// Inputs:
//   - h: mutable private inverse Hessian; s: accepted step; y: accepted gradient difference.
//
// Returns:
//   - No return value; h updated in place or reset to identity.
//
// Errors:
//   - No returned errors; nonrepresentable curvature triggers deterministic identity restart.
//
// Determinism:
//   - Fixed row/column order; secant quantities belong to accepted points only.
//
// Complexity:
// Time O(5^2), Space O(5^2) bounded, no dynamic allocation.
//
// Notes:
//   - Does not mutate objective or model parameters.
//
// AI-Hints:
// Restarting H changes only search geometry, never objective mathematics.
func bfgsInverseUpdate(h *[coordinateCount][coordinateCount]float64, s, y vector) {
	curvature := dot(s, y)
	if !finite(curvature) || curvature <= bfgsCurvatureTolerance*norm(s)*norm(y) {
		*h = identityHessian()
		return
	}
	var hy vector
	for i := range hy {
		for j := range hy {
			hy[i] += h[i][j] * y[j]
		}
	}
	yhy := dot(y, hy)
	if !finite(yhy) {
		*h = identityHessian()
		return
	}
	inverseCurvature := 1 / curvature
	factor := (1 + yhy*inverseCurvature) * inverseCurvature
	if !finite(factor) {
		*h = identityHessian()
		return
	}
	var updated [coordinateCount][coordinateCount]float64
	for i := range updated {
		for j := range updated[i] {
			updated[i][j] = h[i][j] + factor*s[i]*s[j] -
				(hy[i]*s[j]+s[i]*hy[j])*inverseCurvature
			if !finite(updated[i][j]) {
				*h = identityHessian()
				return
			}
		}
	}
	*h = updated
}
