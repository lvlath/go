// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package ols orchestrates the unique QR-based training kernel and diagnostics.
// This file owns the atomic model-publication boundary and original-unit SSE.
package ols

import (
	"context"
	"errors"
	"math"

	"github.com/lvlath/go/matrix"
)

// fitKernel performs the already-validated QR training and post-fit verification.
// Every intermediate object is private until all coefficients and diagnostics
// are finite; cancellation and errors therefore cannot expose a partial fit.
//
// Implementation:
//   - Stage 1: materialize a detached design and one-column response matrix.
//   - Stage 2: solve by pivoted QR; preserve dependency sentinels and full rank.
//   - Stage 3: convert parameters, compute the true original-unit SSE, and publish.
//
// Behavior highlights:
//   - No alternate numerical kernel, Gram matrix, pseudoinverse, or fallback is used.
//   - ConditionEstimate is inherited from the QR triangular factor.
//
// Inputs:
//   - ctx: active context, checked between materialization and post-fit rows.
//   - x/y: validated finite, read-only training data; never retained by Model.
//   - n, p, q: validated sample, feature, and unknown counts, n>=q.
//   - opts: the selected training basis and rank cutoff.
//
// Returns: *FitResult containing a frozen Model and finite diagnostics, or nil/error.
// Errors: ErrRankDeficient (also matrix.ErrRankDeficient), ErrOverflow (also
//
//	matrix.ErrOverflow when applicable), ctx.Err(), and wrapped dependency errors.
//
// Determinism: training rows ascend; solver pivots according to matrix tie rules.
// Complexity: Time O(n*q*q+q*q*q), Space O(n*q+p) including QR workspace.
// Notes: The internal Model object is not returned unless residual verification succeeds.
// AI-Hints: QR's reported residual norm is not the public squared SSE diagnostic.
func fitKernel(ctx context.Context, x matrix.Matrix, y []float64, n, p, q int, opts Options) (*FitResult, error) {
	// Stage 1: independent design coordinates keep input ownership intact.
	designMatrix, means, scales, err := design(ctx, x, n, p, q, opts)
	if err != nil {
		return nil, err
	}
	response, err := matrix.NewDense(n, 1)
	if err != nil {
		return nil, err
	}
	for row, value := range y {
		if row%contextCheckRowInterval == 0 {
			if err = ctx.Err(); err != nil {
				return nil, err
			}
		}
		if err = response.Set(row, 0, value); err != nil {
			return nil, err
		}
	}

	// Stage 2: the existing rectangular solver is the sole source of rank/QR.
	// A dependency error must retain both the OLS semantic classification and
	// the matrix root sentinel, so callers can branch with errors.Is at either
	// abstraction level. Cancellation is passed through without reclassification.
	solved, err := matrix.SolveLeastSquares(ctx, designMatrix, response, matrix.FactorOptions{
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
	if solved.Rank != q {
		return nil, ErrRankDeficient
	}

	// Stage 3: restore physical predictor units BEFORE calculating SSE. The
	// QR solution theta belongs to the centered/scaled design basis, whereas
	// caller-visible predictions and residual energy must agree with the
	// original x and y. No model pointer is published until all checks pass.
	beta, intercept, err := convertCoefficients(solved.Solution, p, means, scales, opts.NoIntercept)
	if err != nil {
		return nil, err
	}
	model := &Model{
		p:         p,
		beta:      beta,
		means:     append([]float64(nil), means...),
		scales:    append([]float64(nil), scales...),
		intercept: intercept,
		fitted:    true,
	}
	sse, err := residualSumSquares(ctx, x, y, model)
	if err != nil {
		return nil, err
	}
	if !finite(solved.ConditionEstimate) {
		return nil, ErrOverflow
	}
	return &FitResult{
		Model:              model,
		Rank:               solved.Rank,
		ResidualSumSquares: sse,
		ConditionEstimate:  solved.ConditionEstimate,
	}, nil
}

// residualSumSquares sums squared original-coordinate residuals without forming
// squares of individually large values that may overflow intermediate float64.
// It uses the same rowPrediction law exposed by Model.Predict on training data.
//
// Implementation:
//   - Stage 1: evaluate each training prediction and reject unrepresentable residuals.
//   - Stage 2: accumulate a scaled sum of squares, renormalizing on larger magnitudes.
//   - Stage 3: reconstruct squared SSE by binary exponent scaling and check finiteness.
//
// Behavior highlights:
//   - For finite residuals, intermediate residual*residual is never evaluated.
//   - SSE is not divided by n, residual degrees of freedom, or response variance.
//
// Inputs: ctx non-nil; x/y validated read-only n-row observations; m fitted and private.
// Returns: nonnegative finite SSE in squared y units, or zero/error.
// Errors: ErrOverflow for nonfinite residual/norm/SSE; ctx.Err(), Matrix.At errors.
// Determinism: input rows ascend, then prediction feature columns ascend.
// Complexity: Time O(n*p), Space O(1) auxiliary beyond Model.
// Notes: Float64 rounding can underflow extremely tiny squared errors to zero.
// AI-Hints: Do not replace this with naive residual*residual accumulation for huge y.
func residualSumSquares(ctx context.Context, x matrix.Matrix, y []float64, m *Model) (float64, error) {
	residualScale, scaledSquares := 0.0, 1.0
	for row := range y {
		if row%contextCheckRowInterval == 0 {
			if err := ctx.Err(); err != nil {
				return 0, err
			}
		}
		prediction, err := m.rowPrediction(ctx, x, row)
		if err != nil {
			return 0, err
		}
		residual := y[row] - prediction
		if !finite(residual) {
			return 0, ErrOverflow
		}
		magnitude := math.Abs(residual)
		if magnitude == 0 {
			continue
		}
		if magnitude > residualScale {
			// Rescale accumulated squares when the next residual sets a new maximum.
			ratio := residualScale / magnitude
			scaledSquares = 1 + scaledSquares*ratio*ratio
			residualScale = magnitude
		} else {
			// Accumulate small residual contributions without overflowing their squares.
			ratio := magnitude / residualScale
			scaledSquares += ratio * ratio
		}
	}
	if residualScale == 0 {
		return 0, nil
	}
	// Store the largest residual as mantissa*2^exponent. Multiplying its
	// mantissa square by scaledSquares avoids premature overflow/underflow
	// from residualScale*residualScale before the final exponent restoration.
	mantissa, exponent := math.Frexp(residualScale)
	sse := math.Ldexp(mantissa*mantissa*scaledSquares, 2*exponent)
	if !finite(sse) {
		return 0, ErrOverflow
	}
	return sse, nil
}
