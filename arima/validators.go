// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package arima enforces deterministic admission and numeric-domain constraints.
// This file centralizes order, context, sample and parameter validation policies.
package arima

import (
	"context"
	"math"
)

const (
	maxAROrder        = 8    // Maximum number of lagged differenced observations.
	maxMAOrder        = 8    // Maximum number of lagged conditional residuals.
	maxDifference     = 2    // Highest supported inverse-integration order.
	defaultIterations = 200  // Accepted-step budget for zero-valued MaxIterations.
	defaultTolerance  = 1e-7 // Scaled gradient threshold for zero-valued Tolerance.
	contextCheckMask  = 255  // Cancellation check at each 256-sample boundary.
)

// finite checks the accepted float64 numeric domain without extra allocations.
// Time O(1), Space O(1); zero and negative finite scalars are allowed here.
func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// validOrder recognizes the complete nonseasonal p/d/q order domain.
// Time O(1), Space O(1), including the valid all-zero white-noise order.
func validOrder(order Order) bool {
	return order.AR >= 0 && order.AR <= maxAROrder &&
		order.MA >= 0 && order.MA <= maxMAOrder &&
		order.Difference >= 0 && order.Difference <= maxDifference
}

// maxN selects the larger integer lag bound without changing either argument.
// Time O(1), Space O(1), used to fix the same burn-in in Fit and Forecast.
func maxN(left, right int) int {
	if left > right {
		return left
	}
	return right
}

// checkContext applies the first public precedence gate before reading samples.
// It returns ErrInvalidOption for a nil context, otherwise exactly ctx.Err().
// Time O(1), Space O(1), assuming a constant-time Context.Err implementation.
func checkContext(ctx context.Context) error {
	if ctx == nil {
		return ErrInvalidOption
	}
	return ctx.Err()
}

// checkSeries validates presence, minimum length and binary64 observations.
// Presence precedes shape and nonfinite detection, even when both fail.
// Time O(N), Space O(1); it never mutates or retains its input slice.
func checkSeries(values []float64, minimum int) error {
	if values == nil {
		return ErrNilInput
	}
	if len(values) == 0 {
		return ErrEmptyInput
	}
	if len(values) < minimum {
		return ErrInsufficientData
	}
	for _, value := range values {
		if !finite(value) {
			return ErrNonFinite
		}
	}
	return nil
}

// finalizeOptions turns zero solver limits into explicit canonical defaults.
// Invalid user controls are rejected before the facade inspects observations.
// Time O(1), Space O(1); it changes only its local by-value Options copy.
func finalizeOptions(options Options) (Options, error) {
	if !validOrder(options.Order) || options.MaxIterations < 0 ||
		!finite(options.Tolerance) || options.Tolerance < 0 {
		return options, ErrInvalidOption
	}
	if options.MaxIterations == 0 {
		options.MaxIterations = defaultIterations
	}
	if options.Tolerance == 0 {
		options.Tolerance = defaultTolerance
	}
	return options, nil
}

// checkParameters ensures shape, finiteness, variance and strict AR/MA root domains.
// Schur recursion avoids arbitrary complex-root tolerances, rejects equality
// at |k|=1 and does not clip or modify the supplied coefficients.
// Time O(p^2+q^2), Space O(p+q), with accepted polynomial degrees at most eight.
func checkParameters(parameters Parameters) error {
	if !validOrder(parameters.Order) || len(parameters.AR) != parameters.Order.AR ||
		len(parameters.MA) != parameters.Order.MA {
		return ErrInvalidOption
	}
	if !finite(parameters.Drift) || !finite(parameters.InnovationVariance) {
		return ErrNonFinite
	}
	if parameters.InnovationVariance <= 0 {
		return ErrSingular
	}
	for _, coefficient := range parameters.AR {
		if !finite(coefficient) {
			return ErrNonFinite
		}
	}
	for _, coefficient := range parameters.MA {
		if !finite(coefficient) {
			return ErrNonFinite
		}
	}
	if !schurStable(parameters.AR) {
		return ErrNonStationary
	}
	negativeMA := make([]float64, len(parameters.MA))
	for i, coefficient := range parameters.MA {
		negativeMA[i] = -coefficient // Schur works with 1-sum(a_i*z^i), whereas MA uses +theta.
	}
	if !schurStable(negativeMA) {
		return ErrNonInvertible
	}
	return nil
}
