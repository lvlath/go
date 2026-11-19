// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test supplies independent high-precision likelihood fixtures.
// math/big reference arithmetic checks conditional Gaussian score values.
package gjrgarch_test

import (
	"context"
	"errors"
	"math"
	"math/big"
	"testing"

	"github.com/lvlath/go/gjrgarch"
)

// Independently accumulate variance with 256-bit arithmetic and convert to
// float64 only when evaluating the transcendental log/normalized residual.
func TestGJR05_BigFloatLikelihoodOracle(t *testing.T) {
	returns := []float64{-.014, .008, -.017, .021, -.009, .012, -.004, .007, -.013, .003, .015, -.02, .011, -.006, .009}
	parameters := gjrgarch.Parameters{Mean: .002, Omega: .00003, Alpha: .11, Gamma: .12, Beta: .68}
	options := gjrgarch.Options{Initial: parameters, MaxIterations: 1}
	result, err := gjrgarch.Fit(context.Background(), returns, options)
	if err != nil && !errors.Is(err, gjrgarch.ErrNotConverged) {
		t.Fatal(err)
	}
	if result == nil {
		t.Fatal("missing result")
	}
	const precision = 256
	number := func(x float64) *big.Float { return new(big.Float).SetPrec(precision).SetFloat64(x) }
	rho := parameters.Alpha + parameters.Beta + parameters.Gamma/2
	h := new(big.Float).Quo(number(parameters.Omega), number(1-rho))
	objective := 0.0
	for _, r := range returns {
		epsilon := r - parameters.Mean
		hv, _ := h.Float64()
		objective += .5 * (math.Log(2*math.Pi) + math.Log(hv) + epsilon*epsilon/hv)
		coefficient := parameters.Alpha
		if epsilon < 0 {
			coefficient += parameters.Gamma
		}
		shock := new(big.Float).Mul(number(epsilon), number(epsilon))
		shock.Mul(shock, number(coefficient))
		lag := new(big.Float).Mul(number(parameters.Beta), h)
		h = new(big.Float).Add(number(parameters.Omega), shock)
		h.Add(h, lag)
	}
	near(t, result.Diagnostics.Objective[0], objective/float64(len(returns)))
}
