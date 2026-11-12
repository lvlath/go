// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ridge owns the single augmented-QR training kernel and fit diagnostics.
// Public model publication occurs only after every mandatory field is verified.
package ridge

import (
	"context"
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

// scaledSquares accumulates squared magnitudes without forming large products.
// It retains the largest magnitude and the dimensionless sum of squared ratios.
type scaledSquares struct {
	scale float64 // Maximum nonzero absolute term, zero only for an empty/zero norm.
	sum   float64 // Sum of ratio squares, in units of the current scale squared.
}

// add includes one finite real contribution in a running scaled squared norm.
// It rescales preceding contributions when a larger magnitude is discovered.
//
// Implementation:
//   - Stage 1: reject nonfinite values and ignore exact zeros.
//   - Stage 2: rescale the existing sum or add a dimensionless squared ratio.
//   - Stage 3: keep the largest magnitude as the reference exponent.
//
// Behavior highlights:
//   - Avoids overflow in value*value for representable final norms.
//
// Inputs:
//   - value is a finite residual or weighted slope, in its native units.
//
// Returns:
//   - error nil on success; the receiver is updated in place.
//
// Errors:
//   - ErrOverflow for nonfinite supplied values or unrepresentable accumulation.
//
// Determinism:
//   - one contribution at a time in caller-defined ascending index order.
//
// Complexity:
//   - Time O(1), Space O(1) per contribution.
//
// Notes:
//   - A tiny representable squared contribution may round away from a large sum.
//
// AI-Hints:
//   - Never replace this accumulator with a naive sum += value*value.
func (s *scaledSquares) add(value float64) error {
	if !finite(value) {
		return ErrOverflow
	}
	magnitude := math.Abs(value)
	if magnitude == 0 {
		return nil
	}
	if magnitude > s.scale {
		factor := s.scale / magnitude
		s.sum = 1 + s.sum*factor*factor
		s.scale = magnitude
	} else {
		factor := magnitude / s.scale
		s.sum += factor * factor
	}
	if !finite(s.sum) {
		return ErrOverflow
	}
	return nil
}

// value reconstructs the squared norm from its scaled mantissa and exponent.
// The reconstruction never squares an unscaled large number prematurely.
//
// Implementation:
//   - Stage 1: return exact zero for an all-zero accumulation.
//   - Stage 2: decompose the reference magnitude into a binary mantissa and exponent.
//   - Stage 3: restore the final squared units and verify float64 finiteness.
//
// Behavior highlights:
//   - Recoverable intermediate overflow is not mistaken for final overflow.
//
// Inputs:
//   - s is a completed accumulation of finite values.
//
// Returns:
//   - nonnegative finite squared norm, or zero/error.
//
// Errors:
//   - ErrOverflow when the final squared norm is unrepresentable.
//
// Determinism:
//   - one fixed Frexp/Ldexp reconstruction.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Extremely tiny squared norms may round to positive zero in float64.
//
// AI-Hints:
//   - Do not square s.scale directly before restoring the exponent.
func (s scaledSquares) value() (float64, error) {
	if s.scale == 0 {
		return 0, nil
	}
	mantissa, exponent := math.Frexp(s.scale)
	norm := math.Ldexp(mantissa*mantissa*s.sum, 2*exponent)
	if !finite(norm) {
		return 0, ErrOverflow
	}
	return norm, nil
}

// scaledPenalty evaluates Lambda*||theta||^2/2 in optimization coordinates.
// It never squares an unweighted theta before the Lambda factor is incorporated.
//
// Implementation:
//   - Stage 1: factor sqrt(Lambda/2) as sqrt(Lambda)*sqrt(1/2).
//   - Stage 2: multiply each slope by the weight and accumulate scaled squares.
//   - Stage 3: return the finite penalty without penalizing the intercept.
//
// Behavior highlights:
//   - Large/small Lambda products do not disappear before square root.
//
// Inputs:
//   - theta is the read-only p-vector of QR optimization slopes; Lambda>0 finite.
//
// Returns:
//   - finite nonnegative L2 penalty or zero/error.
//
// Errors:
//   - ErrOverflow for nonfinite weighted values or an unrepresentable penalty.
//
// Determinism:
//   - theta indices ascend in original feature order.
//
// Complexity:
//   - Time O(p), Space O(1), no p-vector temporary allocation.
//
// Notes:
//   - Lambda>0 with theta=0 yields exactly zero penalty.
//
// AI-Hints:
//   - When Standardize=true, replacing theta by physical beta changes the objective.
func scaledPenalty(theta []float64, lambda float64) (float64, error) {
	weight := math.Sqrt(lambda) * math.Sqrt(0.5)
	var accumulator scaledSquares
	for _, slope := range theta {
		if err := accumulator.add(slope * weight); err != nil {
			return 0, err
		}
	}
	return accumulator.value()
}

// convertCoefficients restores physical slopes and intercept after augmented QR.
// The unpenalized intercept in QR coordinates is corrected for fitted centers.
//
// Implementation:
//   - Stage 1: read the trailing QR intercept and undo its column scale if fitted.
//   - Stage 2: undo QR column equilibration and divide slopes by training scale.
//   - Stage 3: remove the feature-center contribution from the affine intercept.
//
// Behavior highlights:
//   - QR columns are internally ordered slopes-first and intercept-last.
//   - NoIntercept produces an exact positive-zero physical intercept.
//
// Inputs:
//   - solution is a finite q-by-1 slope-first QR solution; p is feature count;
//
// means/scales are detached p-vectors; divisors are positive QR column scales;
// noIntercept selects the hypothesis space.
//
// Returns:
//   - detached beta and theta p-vectors plus physical intercept, or nil/error.
//
// Errors:
//   - ErrOverflow on nonfinite conversion, and retained Matrix.At errors.
//
// Determinism:
//   - coefficient feature indices ascend; intercept correction is in that order.
//
// Complexity:
//   - Time O(p), Space O(p) for the independent coefficient vectors.
//
// Notes:
//   - Both QR column equilibration and train-only predictor scaling must be
//     undone; theta, not gamma or physical beta, enters the regularization.
//   - Fit does not publish the Model until objective diagnostics also succeed.
//
// AI-Hints:
//   - Never apply a mean correction when NoIntercept=true.
func convertCoefficients(solution matrix.Matrix, p int, means, scales, divisors []float64, noIntercept bool) ([]float64, []float64, float64, error) {
	beta := make([]float64, p)
	theta := make([]float64, p)
	intercept := 0.0
	if !noIntercept {
		// QR places the unpenalized intercept last. Undo its column scaling
		// before removing the fitted feature means from the physical offset.
		value, err := solution.At(p, 0)
		if err != nil {
			return nil, nil, 0, err
		}
		intercept = value / divisors[p]
		if !finite(intercept) {
			return nil, nil, 0, ErrOverflow
		}
	}
	for j := 0; j < p; j++ {
		// A right column scaling is a change of coefficient variables:
		// (A_j / d_j)*gamma_j == A_j*(gamma_j / d_j).
		// The ridge penalty belongs to theta, not the temporary gamma.
		value, err := solution.At(j, 0)
		if err != nil {
			return nil, nil, 0, err
		}
		theta[j] = value / divisors[j]
		beta[j] = theta[j] / scales[j]
		if !finite(theta[j]) || !finite(beta[j]) {
			return nil, nil, 0, ErrOverflow
		}
	}
	if !noIntercept {
		for j := 0; j < p; j++ {
			intercept = math.FMA(-means[j], beta[j], intercept)
			if !finite(intercept) {
				return nil, nil, 0, ErrOverflow
			}
		}
	}
	return beta, theta, intercept, nil
}

// residualSumSquares evaluates training error in original response units.
// It uses exactly the same physical-unit row prediction as the public inference API.
//
// Implementation:
//   - Stage 1: predict each training row with an active context.
//   - Stage 2: subtract the original y and add the residual to scaled squares.
//   - Stage 3: finalize a finite unnormalized SSE without storing an n-vector.
//
// Behavior highlights:
//   - SSE is not divided by n, degrees of freedom, or a transform scale.
//
// Inputs:
//   - ctx active; x/y validated read-only training data; model is private and fitted.
//
// Returns:
//   - finite nonnegative sum of squared residuals or zero/error.
//
// Errors:
//   - ErrOverflow for unrepresentable residual/SSE, ctx.Err(), Matrix.At errors.
//
// Determinism:
//   - row i ascends, prediction feature j ascends.
//
// Complexity:
//   - Time O(n*p), Space O(1) beyond the fitted Model.
//
// Notes:
//   - Input matrices are not snapshotted; concurrent mutation remains unsupported.
//
// AI-Hints:
//   - Do not use the augmented solver's residual norm as original-coordinate SSE.
func residualSumSquares(ctx context.Context, x matrix.Matrix, y []float64, model *Model) (float64, error) {
	var accumulator scaledSquares
	for i := range y {
		if i%contextCheckInterval == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		prediction, err := model.rowPrediction(ctx, x, i)
		if err != nil {
			return 0, err
		}
		residual := y[i] - prediction
		if err = accumulator.add(residual); err != nil {
			return 0, err
		}
	}
	return accumulator.value()
}

// fitKernel performs one augmented QR solve and publishes checked diagnostics.
// It is the unique training kernel; no normal-equation or OLS branch exists.
//
// Implementation:
//   - Stage 1: construct canonical augmented design and loss-preserving QR coordinates.
//   - Stage 2: solve the equilibrated penalty-first system with pivoted QR.
//   - Stage 3: restore physical coefficients, compute objective components, publish.
//
// Behavior highlights:
//   - All errors return nil; the Model is private until full verification.
//
// Inputs:
//   - ctx: previously validated active context; never retained.
//   - x/y: validated finite read-only observations, n rows and p columns.
//   - n, p: positive training dimensions with bounded augmented allocation.
//   - opts: validated Lambda, intercept, scale, and relative rank policy.
//
// Returns:
//   - *FitResult: complete original-unit model with checked objective terms.
//   - error: nil on success; every failure returns a nil result.
//
// Errors:
//   - ErrRankDeficient joined with matrix.ErrRankDeficient on QR rank rejection.
//   - ErrOverflow joined with matrix.ErrOverflow on QR numeric failure.
//   - ErrOverflow for unrepresentable restored coordinates or objective.
//   - ctx.Err() or wrapped Matrix.At/Set errors, without partial publication.
//
// Determinism:
//   - canonical design rows ascend, then penalty rows precede observations
//     in the QR view; solver pivot ties follow matrix's deterministic policy.
//
// Complexity:
//   - Time O((n+p)*q*q+q^3+n*p), Space O((n+p)*q+p),
//     including canonical augmentation, balanced QR input, and solver snapshots.
//
// Notes:
//   - No result is returned if residual or penalty cannot fit finite float64.
//
// AI-Hints:
//   - Do not use QR's transformed-coordinate residuals as FitResult.SSE.
//   - Divide gamma by its column divisor before restoring physical coefficients;
//     otherwise the fitted slopes and regularization penalty are both incorrect.
func fitKernel(ctx context.Context, x matrix.Matrix, y []float64, n, p int, opts Options) (*FitResult, error) {
	design, response, means, scales, err := buildDesign(ctx, x, y, n, p, opts)
	if err != nil {
		return nil, err
	}
	// Prepare the same augmented objective in slope-first, penalty-first QR
	// coordinates. The canonical design remains unchanged and detached.
	balanced, balancedRHS, divisors, err := prepareQRSystem(ctx, design, response, n, p, opts.NoIntercept)
	if err != nil {
		return nil, err
	}
	solved, err := matrix.SolveLeastSquares(ctx, balanced, balancedRHS, matrix.FactorOptions{
		RankTolerance: opts.RankTolerance,
	})
	if err != nil {
		switch {
		case errors.Is(err, matrix.ErrRankDeficient):
			return nil, errors.Join(ErrRankDeficient, err)
		case errors.Is(err, matrix.ErrOverflow):
			return nil, errors.Join(ErrOverflow, err)
		default:
			return nil, err
		}
	}

	q := p
	if !opts.NoIntercept {
		q += interceptColumns
	}
	if solved == nil || solved.Solution == nil || solved.Rank != q {
		return nil, ErrRankDeficient
	}
	beta, theta, intercept, err := convertCoefficients(solved.Solution, p, means, scales, divisors, opts.NoIntercept)
	if err != nil {
		return nil, err
	}
	model := &Model{
		p:         p,
		beta:      beta,
		intercept: intercept,
		means:     append([]float64(nil), means...),
		scales:    append([]float64(nil), scales...),
		fitted:    true,
	}

	// Diagnostics are evaluated on the model that will actually be returned,
	// not on the transient standardized coefficients or the augmented RHS.
	sse, err := residualSumSquares(ctx, x, y, model)
	if err != nil {
		return nil, err
	}
	penalty, err := scaledPenalty(theta, opts.Lambda)
	if err != nil {
		return nil, err
	}
	objective := sse/(2*float64(n)) + penalty
	if !finite(objective) {
		return nil, ErrOverflow
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return &FitResult{Model: model, ResidualSumSquares: sse, Penalty: penalty, Objective: objective}, nil
}
