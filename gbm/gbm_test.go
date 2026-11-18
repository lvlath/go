// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm_test checks the public mathematical and validation contracts.
// Every scenario runs through the exported gbm facade, never package internals.
package gbm_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/gbm"
)

// TestGBM01Through04 anchors analytic deterministic and zero-normal oracles.
func TestGBM01Through04(t *testing.T) {
	testCases := []struct {
		name   string
		params gbm.Parameters
		times  []float64
		values []float64
		draws  int
	}{
		{"GBM01 zero drift/volatility", gbm.Parameters{Initial: 2},
			[]float64{0, 1, 3}, []float64{2, 2, 2}, 0},
		{"GBM02 deterministic compounding", gbm.Parameters{Initial: 2, Drift: math.Ln2},
			[]float64{0, 1, 2}, []float64{2, 4, 8}, 0},
		{"GBM03 Ito correction", gbm.Parameters{Initial: 1, Volatility: 1},
			[]float64{0, 1, 2}, []float64{1, math.Exp(-0.5), math.Exp(-1)}, 2},
		{"GBM04 drift equals Ito correction", gbm.Parameters{Initial: 1, Drift: 0.5, Volatility: 1},
			[]float64{0, 1, 3}, []float64{1, 1, 1}, 2},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			source := &scripted{}
			result, err := gbm.Simulate(context.Background(), tc.params,
				gbm.TimeGrid{Values: tc.times}, source)
			if err != nil {
				t.Fatal(err)
			}
			mustEqualBool(t, result.Complete, true, "full result")
			mustEqualInt(t, len(result.Values), len(tc.values), "path size")
			mustEqualInt(t, source.count, tc.draws, "Normal count")
			mustEqualInt(t, source.rawCalls+source.uniformCalls, 0, "non-Normal RNG calls")
			for index, expected := range tc.values {
				mustNear(t, result.Values[index], expected, 1e-13)
			}
		})
	}
}

// TestGBM07InitialOnlyAndDetached validates the one-point degenerate case.
func TestGBM07InitialOnlyAndDetached(t *testing.T) {
	times := []float64{0}
	source := &scripted{}
	result, err := gbm.Simulate(context.Background(),
		gbm.Parameters{Initial: 12, Volatility: 0.4}, gbm.TimeGrid{Values: times}, source)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualBool(t, result.Complete, true, "initial-only complete")
	mustEqualInt(t, len(result.Values), 1, "initial-only values")
	mustNear(t, result.Values[0], 12, 0)
	mustEqualInt(t, source.count+source.rawCalls+source.uniformCalls, 0, "no entropy")

	// Editing the source after publication cannot alter the Result's time array.
	times[0] = 2
	mustNear(t, result.Times[0], 0, 0)
	result.Times[0] = 11
	mustNear(t, times[0], 2, 0)
}

// TestResultIsNilInterface validates the actual core.Nilable assignment.
func TestResultIsNilInterface(t *testing.T) {
	var nilResult *gbm.Result
	var nilable core.Nilable = nilResult
	mustEqualBool(t, nilable.IsNil(), true, "typed-nil core.Nilable")
	mustNilState(t, nilResult, true, "nil Result")

	nilable = &gbm.Result{}
	mustEqualBool(t, nilable.IsNil(), false, "allocated Result")
	mustNilState(t, nilable, false, "allocated core.Nilable")
}

// TestGBM09Validation verifies sentinel families and preflight nil-result policy.
func TestGBM09Validation(t *testing.T) {
	base := gbm.Parameters{Initial: 1, Volatility: 1, Unit: gbm.TimeYears}
	validGrid := gbm.TimeGrid{Values: []float64{0, 1}, Unit: gbm.TimeYears}
	badCases := []struct {
		name   string
		params gbm.Parameters
		grid   gbm.TimeGrid
		rng    *scripted
		ctx    context.Context
		want   error
	}{
		{"nil context", base, validGrid, &scripted{}, nil, gbm.ErrInvalidOption},
		{"unit mismatch", base, gbm.TimeGrid{Values: []float64{0, 1}, Unit: gbm.TimeSeconds}, &scripted{}, context.Background(), gbm.ErrInvalidOption},
		{"invalid parameter unit", gbm.Parameters{Initial: 1, Unit: 99}, validGrid, &scripted{}, context.Background(), gbm.ErrInvalidOption},
		{"invalid grid unit", base, gbm.TimeGrid{Values: []float64{0}, Unit: 255}, &scripted{}, context.Background(), gbm.ErrInvalidOption},
		{"nil RNG", base, validGrid, nil, context.Background(), gbm.ErrNilInput},
		{"nil grid", base, gbm.TimeGrid{Unit: base.Unit}, &scripted{}, context.Background(), gbm.ErrNilInput},
		{"empty grid", base, gbm.TimeGrid{Values: []float64{}, Unit: base.Unit}, &scripted{}, context.Background(), gbm.ErrEmptyInput},
		{"duplicate time", base, gbm.TimeGrid{Values: []float64{0, 1, 1}, Unit: base.Unit}, &scripted{}, context.Background(), gbm.ErrOrder},
		{"decreasing time", base, gbm.TimeGrid{Values: []float64{0, 3, 1}, Unit: base.Unit}, &scripted{}, context.Background(), gbm.ErrOrder},
		{"nonzero origin", base, gbm.TimeGrid{Values: []float64{1, 2}, Unit: base.Unit}, &scripted{}, context.Background(), gbm.ErrOrder},
		{"negative origin", base, gbm.TimeGrid{Values: []float64{-1, 2}, Unit: base.Unit}, &scripted{}, context.Background(), gbm.ErrOrder},
		{"NaN time", base, gbm.TimeGrid{Values: []float64{0, math.NaN()}, Unit: base.Unit}, &scripted{}, context.Background(), gbm.ErrNonFinite},
		{"positive infinity time", base, gbm.TimeGrid{Values: []float64{0, math.Inf(1)}, Unit: base.Unit}, &scripted{}, context.Background(), gbm.ErrNonFinite},
		{"negative infinity origin", base, gbm.TimeGrid{Values: []float64{math.Inf(-1)}, Unit: base.Unit}, &scripted{}, context.Background(), gbm.ErrNonFinite},
		{"NaN initial", gbm.Parameters{Initial: math.NaN()}, validGrid, &scripted{}, context.Background(), gbm.ErrNonFinite},
		{"infinite initial", gbm.Parameters{Initial: math.Inf(1)}, validGrid, &scripted{}, context.Background(), gbm.ErrNonFinite},
		{"NaN drift", gbm.Parameters{Initial: 1, Drift: math.NaN()}, validGrid, &scripted{}, context.Background(), gbm.ErrNonFinite},
		{"infinite drift", gbm.Parameters{Initial: 1, Drift: math.Inf(-1)}, validGrid, &scripted{}, context.Background(), gbm.ErrNonFinite},
		{"NaN volatility", gbm.Parameters{Initial: 1, Volatility: math.NaN()}, validGrid, &scripted{}, context.Background(), gbm.ErrNonFinite},
		{"infinite volatility", gbm.Parameters{Initial: 1, Volatility: math.Inf(1)}, validGrid, &scripted{}, context.Background(), gbm.ErrNonFinite},
		{"zero initial", gbm.Parameters{Initial: 0}, validGrid, &scripted{}, context.Background(), gbm.ErrDomain},
		{"negative initial", gbm.Parameters{Initial: -1}, validGrid, &scripted{}, context.Background(), gbm.ErrDomain},
		{"negative volatility", gbm.Parameters{Initial: 1, Volatility: -1}, validGrid, &scripted{}, context.Background(), gbm.ErrDomain},
	}
	for _, tc := range badCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := gbm.Simulate(tc.ctx, tc.params, tc.grid, tc.rng)
			mustErrorIs(t, err, tc.want)
			mustNilState(t, result, true, "invalid preflight returns nil")
			if tc.rng != nil {
				mustEqualInt(t, tc.rng.count, 0, "invalid input consumes no Normal")
			}
		})
	}
}

// TestValidationPrecedence confirms that the first recognized failure wins.
func TestValidationPrecedence(t *testing.T) {
	badP := gbm.Parameters{Initial: -1, Unit: gbm.TimeYears}
	badGrid := gbm.TimeGrid{Unit: gbm.TimeSeconds}
	result, err := gbm.Simulate(nil, badP, badGrid, nil)
	mustErrorIs(t, err, gbm.ErrInvalidOption)
	mustNilState(t, result, true, "nil ctx precedence")

	result, err = gbm.Simulate(context.Background(), badP, badGrid, nil)
	mustErrorIs(t, err, gbm.ErrInvalidOption)
	mustNilState(t, result, true, "units precedence")

	badGrid.Unit = gbm.TimeYears
	result, err = gbm.Simulate(context.Background(), badP, badGrid, nil)
	mustErrorIs(t, err, gbm.ErrNilInput)
	mustNilState(t, result, true, "RNG precedence")
}

// TestZeroVolatilityNeverCallsAnyRNGMethod checks all three RNG methods.
func TestZeroVolatilityNeverCallsAnyRNGMethod(t *testing.T) {
	source := &scripted{}
	times := []float64{0, 0.03, 0.25, 0.55, 1.3, 2.0, 3.1}
	result, err := gbm.Simulate(context.Background(), gbm.Parameters{
		Initial: 145.25, Drift: -0.03, Volatility: 0, Unit: gbm.TimeYears,
	}, gbm.TimeGrid{Values: times, Unit: gbm.TimeYears}, source)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualBool(t, result.Complete, true, "deterministic completion")
	mustEqualInt(t, source.count+source.rawCalls+source.uniformCalls, 0, "no RNG calls")
	for index, elapsed := range times {
		mustNear(t, result.Values[index], 145.25*math.Exp(-0.03*elapsed), 1e-13)
	}
}
