// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test anchors the GJR-GARCH mathematical acceptance matrix.
// Each oracle uses the exported facade and independently computed expectations.
package gjrgarch_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/gjrgarch"
)

func p() gjrgarch.Parameters { return gjrgarch.Parameters{Omega: .1, Alpha: .1, Gamma: .2, Beta: .5} }
func near(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-10*(1+math.Abs(want)) {
		t.Fatalf("got %.16g want %.16g", got, want)
	}
}
func TestGJR01_HandOracle(t *testing.T) {
	m, e := gjrgarch.New(p())
	if e != nil {
		t.Fatal(e)
	}
	down, e := m.InitialState(-2, 1)
	if e != nil {
		t.Fatal(e)
	}
	up, e := m.InitialState(2, 1)
	if e != nil {
		t.Fatal(e)
	}
	a, e := m.ForecastVariance(down, 3)
	if e != nil {
		t.Fatal(e)
	}
	b, e := m.ForecastVariance(up, 1)
	if e != nil {
		t.Fatal(e)
	}
	for i, v := range []float64{1.8, 1.36, 1.052} {
		near(t, a[i], v)
	}
	near(t, b[0], 1)
	zero, e := m.InitialState(0, 1)
	if e != nil {
		t.Fatal(e)
	}
	z, e := m.ForecastVariance(zero, 1)
	if e != nil {
		t.Fatal(e)
	}
	near(t, z[0], .6)
}
func TestGJR02_PrefixCausality(t *testing.T) {
	m, e := gjrgarch.New(p())
	if e != nil {
		t.Fatal(e)
	}
	returns := []float64{.1, -.3, .15, -.7, .4, .1, -.2, .8, -.1}
	a, e := m.Filter(context.Background(), returns, 1)
	if e != nil {
		t.Fatal(e)
	}
	b, e := m.Filter(context.Background(), returns[:5], 1)
	if e != nil {
		t.Fatal(e)
	}
	for i := range b {
		near(t, a[i], b[i])
	}
	state, e := m.InitialState(returns[0], 1)
	if e != nil {
		t.Fatal(e)
	}
	near(t, a[0], state.LastVariance)
	for i := 1; i < len(returns); i++ {
		state, e = m.Step(state, returns[i])
		if e != nil {
			t.Fatal(e)
		}
		near(t, state.LastVariance, a[i])
	}
}
func TestGJR03_SymmetricReduction(t *testing.T) {
	params := p()
	params.Gamma = 0
	m, e := gjrgarch.New(params)
	if e != nil {
		t.Fatal(e)
	}
	x := []float64{-.5, .2, -.1, .8, -.3, .6}
	v, e := m.Filter(context.Background(), x, 1)
	if e != nil {
		t.Fatal(e)
	}
	h := 1.0
	for i := range x {
		near(t, v[i], h)
		h = params.Omega + params.Alpha*x[i]*x[i] + params.Beta*h
	}
}
func TestGJR04_ScaleEquivariance(t *testing.T) {
	base := p()
	a := 7.0
	scaled := base
	scaled.Mean *= a
	scaled.Omega *= a * a
	m, e := gjrgarch.New(base)
	if e != nil {
		t.Fatal(e)
	}
	n, e := gjrgarch.New(scaled)
	if e != nil {
		t.Fatal(e)
	}
	x := []float64{-.4, .1, -.2, .7, -.9, .1}
	y := make([]float64, len(x))
	for i, v := range x {
		y[i] = v * a
	}
	u, e := m.Filter(context.Background(), x, 1)
	if e != nil {
		t.Fatal(e)
	}
	v, e := n.Filter(context.Background(), y, a*a)
	if e != nil {
		t.Fatal(e)
	}
	for i := range u {
		near(t, v[i], u[i]*a*a)
	}
}
func TestGJR09_ValidationAndState(t *testing.T) {
	bad := p()
	bad.Beta = .9
	if _, e := gjrgarch.New(bad); !errors.Is(e, gjrgarch.ErrDomain) {
		t.Fatalf("stationarity: %v", e)
	}
	m, e := gjrgarch.New(p())
	if e != nil {
		t.Fatal(e)
	}
	if _, e = m.Filter(context.Background(), nil, 1); !errors.Is(e, gjrgarch.ErrNilInput) {
		t.Fatal(e)
	}
	if _, e = m.Filter(context.Background(), []float64{}, 1); !errors.Is(e, gjrgarch.ErrEmptyInput) {
		t.Fatal(e)
	}
	if _, e = m.Step(gjrgarch.State{}, .1); !errors.Is(e, gjrgarch.ErrStateVersion) {
		t.Fatal(e)
	}
	s, e := m.InitialState(-1, 1)
	if e != nil {
		t.Fatal(e)
	}
	s.ModelID = "another-model"
	if _, e = m.Step(s, .2); !errors.Is(e, gjrgarch.ErrStateVersion) {
		t.Fatal(e)
	}
	if _, e = m.Step(s, math.NaN()); !errors.Is(e, gjrgarch.ErrStateVersion) {
		t.Fatal(e)
	}
}
func TestResult_IsNilInterface(t *testing.T) {
	var result *gjrgarch.FitResult
	var nilable interface{ IsNil() bool } = result
	if !nilable.IsNil() {
		t.Fatal("nil result must be nilable")
	}
	if (new(gjrgarch.FitResult)).IsNil() {
		t.Fatal("an allocated result is not a typed nil pointer")
	}
}
func TestGJR08_BudgetAndObjective(t *testing.T) {
	data := []float64{-.3, .1, .5, -.2, .3, -.1, -.4, .2, .6, -.3, .1, .4, -.5, .2, .1, -.6, .3, .2, -.1, .4}
	r, e := gjrgarch.Fit(context.Background(), data, gjrgarch.Options{Initial: p(), MaxIterations: 1, VarianceFloor: 1e-12})
	if e != nil && !errors.Is(e, gjrgarch.ErrNotConverged) {
		t.Fatal(e)
	}
	if r == nil || r.Model == nil || len(r.Diagnostics.Objective) == 0 {
		t.Fatal("missing valid initial candidate")
	}
	if r.Diagnostics.Converged && errors.Is(e, gjrgarch.ErrNotConverged) {
		t.Fatal("false convergence")
	}
	initial := p()
	rho := initial.Alpha + initial.Beta + initial.Gamma/2
	h := initial.Omega / (1 - rho)
	objective := 0.0
	for _, v := range data {
		objective += .5 * (math.Log(2*math.Pi) + math.Log(h) + v*v/h)
		c := initial.Alpha
		if v < 0 {
			c += initial.Gamma
		}
		h = initial.Omega + c*v*v + initial.Beta*h
	}
	near(t, r.Diagnostics.Objective[0], objective/float64(len(data)))
}
