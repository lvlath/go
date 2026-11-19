// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gjrgarch_test tests public validation and deterministic failure categories.
// Sentinel comparisons use errors.Is and never depend on error-message text.
package gjrgarch_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/gjrgarch"
)

func TestNew_InvalidParameterMatrix(t *testing.T) {
	cases := []struct {
		name   string
		change func(*gjrgarch.Parameters)
		target error
	}{
		{"nonfinite mean", func(p *gjrgarch.Parameters) { p.Mean = math.NaN() }, gjrgarch.ErrNonFinite},
		{"omega zero", func(p *gjrgarch.Parameters) { p.Omega = 0 }, gjrgarch.ErrDomain},
		{"negative alpha", func(p *gjrgarch.Parameters) { p.Alpha = -1 }, gjrgarch.ErrDomain},
		{"negative beta", func(p *gjrgarch.Parameters) { p.Beta = -1 }, gjrgarch.ErrDomain},
		{"negative shock coefficient", func(p *gjrgarch.Parameters) { p.Gamma = -.3 }, gjrgarch.ErrDomain},
		{"overflowed shock coefficient", func(p *gjrgarch.Parameters) { p.Alpha = math.MaxFloat64; p.Gamma = math.MaxFloat64 }, gjrgarch.ErrOverflow},
		{"nonstationary", func(p *gjrgarch.Parameters) { p.Beta = .9 }, gjrgarch.ErrDomain},
		{"long-run overflow", func(p *gjrgarch.Parameters) { p.Omega = math.MaxFloat64; p.Beta = .5 }, gjrgarch.ErrOverflow},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x := p()
			tc.change(&x)
			m, err := gjrgarch.New(x)
			if m != nil {
				t.Fatal("invalid parameters published a model")
			}
			mustErrorIs(t, err, tc.target)
		})
	}
}
func TestAPI_NilReceiverAndFingerprint(t *testing.T) {
	var nilModel *gjrgarch.Model
	if nilModel.ID() != "" {
		t.Fatal("nil ID")
	}
	if nilModel.Parameters() != (gjrgarch.Parameters{}) {
		t.Fatal("nil params")
	}
	a := p()
	a.Mean = math.Copysign(0, -1)
	b := a
	b.Mean = 0
	m, e := gjrgarch.New(a)
	if e != nil {
		t.Fatal(e)
	}
	n, e := gjrgarch.New(b)
	if e != nil {
		t.Fatal(e)
	}
	if m.ID() != n.ID() {
		t.Fatal("signed zero identity drift")
	}
	if m.Parameters() != a {
		t.Fatal("parameter value not returned")
	}
	b.Beta = .2
	q, e := gjrgarch.New(b)
	if e != nil {
		t.Fatal(e)
	}
	if q.ID() == m.ID() {
		t.Fatal("nonunique fingerprints")
	}
}
func TestFilter_ErrorsAndCancel(t *testing.T) {
	m, e := gjrgarch.New(p())
	if e != nil {
		t.Fatal(e)
	}
	c, stop := context.WithCancel(context.Background())
	stop()
	if _, e = m.Filter(c, []float64{1}, 1); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
	if _, e = m.Filter(nil, []float64{1}, 1); !errors.Is(e, gjrgarch.ErrInvalidOption) {
		t.Fatal(e)
	}
	if _, e = m.Filter(context.Background(), []float64{1, math.Inf(1)}, 1); !errors.Is(e, gjrgarch.ErrNonFinite) {
		t.Fatal(e)
	}
	if _, e = m.Filter(context.Background(), []float64{1}, 0); !errors.Is(e, gjrgarch.ErrDomain) {
		t.Fatal(e)
	}
	if _, e = m.Filter(context.Background(), []float64{1}, math.NaN()); !errors.Is(e, gjrgarch.ErrNonFinite) {
		t.Fatal(e)
	}
	if _, e = m.InitialState(math.Inf(1), 1); !errors.Is(e, gjrgarch.ErrNonFinite) {
		t.Fatal(e)
	}
	if _, e = m.InitialState(1, 0); !errors.Is(e, gjrgarch.ErrStateVersion) {
		t.Fatal(e)
	}
}
func TestStep_CountAndRecursionOverflow(t *testing.T) {
	m, e := gjrgarch.New(p())
	if e != nil {
		t.Fatal(e)
	}
	state, e := m.InitialState(1, 1)
	if e != nil {
		t.Fatal(e)
	}
	state.Steps = ^uint64(0)
	if _, e = m.Step(state, 1); !errors.Is(e, gjrgarch.ErrOverflow) {
		t.Fatal(e)
	}
	state.Steps = 1
	state.LastResidual = math.MaxFloat64
	if _, e = m.Step(state, 0); !errors.Is(e, gjrgarch.ErrOverflow) {
		t.Fatal(e)
	}
	if _, e = m.Step(state, math.NaN()); !errors.Is(e, gjrgarch.ErrNonFinite) {
		t.Fatal(e)
	}
}
func TestForecast_InvalidHorizonState(t *testing.T) {
	m, e := gjrgarch.New(p())
	if e != nil {
		t.Fatal(e)
	}
	s, e := m.InitialState(0, 1)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = m.ForecastVariance(s, 0); !errors.Is(e, gjrgarch.ErrDomain) {
		t.Fatal(e)
	}
	if _, e = m.ForecastVariance(s, -1); !errors.Is(e, gjrgarch.ErrDomain) {
		t.Fatal(e)
	}
	s.Steps = 0
	if _, e = m.ForecastVariance(s, 1); !errors.Is(e, gjrgarch.ErrStateVersion) {
		t.Fatal(e)
	}
}
func TestFit_InvalidOptionsBeforeData(t *testing.T) {
	good := gjrgarch.Options{Initial: p()}
	if r, e := gjrgarch.Fit(nil, nil, good); r != nil || !errors.Is(e, gjrgarch.ErrInvalidOption) {
		t.Fatalf("%v %v", r, e)
	}
	canceled, stop := context.WithCancel(context.Background())
	stop()
	if r, e := gjrgarch.Fit(canceled, nil, good); r != nil || !errors.Is(e, context.Canceled) {
		t.Fatalf("%v %v", r, e)
	}
	choices := []gjrgarch.Options{
		{Initial: gjrgarch.Parameters{}},
		{Initial: p(), MaxIterations: -1},
		{Initial: p(), Tolerance: math.NaN()},
		{Initial: p(), VarianceFloor: -1},
		{Initial: p(), VarianceFloor: 1},
	}
	for i, o := range choices {
		if r, e := gjrgarch.Fit(context.Background(), nil, o); r != nil || !errors.Is(e, gjrgarch.ErrInvalidOption) {
			t.Fatalf("case %d: %v %v", i, r, e)
		}
	}
	if r, e := gjrgarch.Fit(context.Background(), nil, good); r != nil || !errors.Is(e, gjrgarch.ErrNilInput) {
		t.Fatalf("%v %v", r, e)
	}
	if r, e := gjrgarch.Fit(context.Background(), []float64{}, good); r != nil || !errors.Is(e, gjrgarch.ErrEmptyInput) {
		t.Fatalf("%v %v", r, e)
	}
	if r, e := gjrgarch.Fit(context.Background(), []float64{math.Inf(1)}, good); r != nil || !errors.Is(e, gjrgarch.ErrNonFinite) {
		t.Fatalf("%v %v", r, e)
	}
}

func TestGJR07_ExactBoundaryConvergence(t *testing.T) {
	observations := make([]float64, 40)
	initial := gjrgarch.Parameters{Omega: 1, Alpha: 0, Gamma: 0, Beta: 0}
	result, err := gjrgarch.Fit(context.Background(), observations, gjrgarch.Options{
		Initial: initial, VarianceFloor: 1, MaxIterations: 5,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Diagnostics.Converged {
		t.Fatal("feasible KKT boundary stationary point must converge")
	}
	fitted := result.Model.Parameters()
	if fitted.Alpha != 0 || fitted.Alpha+fitted.Gamma != 0 {
		t.Fatalf("boundary drift: %+v", fitted)
	}
}

func TestStep_ZeroAllocations(t *testing.T) {
	m, err := gjrgarch.New(p())
	if err != nil {
		t.Fatal(err)
	}
	state, err := m.InitialState(-.02, .0004)
	if err != nil {
		t.Fatal(err)
	}
	allocated := testing.AllocsPerRun(100, func() {
		_, stepErr := m.Step(state, .01)
		if stepErr != nil {
			t.Error(stepErr)
		}
	})
	if allocated != 0 {
		t.Fatalf("Step allocated %.1f times", allocated)
	}
}
