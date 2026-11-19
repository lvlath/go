// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch fixes option defaults and rejects infeasible optimizer starts.
// This boundary does not select an initial model from the observations.
package gjrgarch

import "errors"

const (
	// defaultMaxIterations bounds accepted projected-BFGS updates when unspecified.
	defaultMaxIterations = 200
	// defaultTolerance controls the Euclidean projected-gradient termination test.
	defaultTolerance = 1e-7
	// defaultVarianceFloor is measured in squared input-return units.
	defaultVarianceFloor = 1e-12
	// armijoCoefficient is the sufficient-decrease fraction in the line search.
	armijoCoefficient = 1e-4
	// maxLineSearchHalvings bounds 21 trials including the full step.
	maxLineSearchHalvings = 20
)

// prepareOptions validates user policy, then applies defaults before reading data.
// It preserves the initiating mathematical failure class alongside ErrInvalidOption.
//
// Implementation:
//   - Stage 1: reject negative or nonfinite iteration/tolerance/floor fields.
//   - Stage 2: replace explicit zero defaults without inventing an Initial model.
//   - Stage 3: validate Initial and the stricter optimization feasible polytope.
//
// Complexity: Time O(1), Space O(1).
func prepareOptions(o Options) (Options, error) {
	if o.MaxIterations < 0 || !finite(o.Tolerance) || o.Tolerance < 0 ||
		!finite(o.VarianceFloor) || o.VarianceFloor < 0 {
		return Options{}, ErrInvalidOption
	}
	if o.MaxIterations == 0 {
		o.MaxIterations = defaultMaxIterations
	}
	if o.Tolerance == 0 {
		o.Tolerance = defaultTolerance
	}
	if o.VarianceFloor == 0 {
		o.VarianceFloor = defaultVarianceFloor
	}

	rho, err := stationary(o.Initial)
	if err != nil {
		return Options{}, errors.Join(ErrInvalidOption, err)
	}
	if o.Initial.Omega < o.VarianceFloor || rho > 1-stationarityMargin {
		return Options{}, ErrInvalidOption
	}
	return o, nil
}
