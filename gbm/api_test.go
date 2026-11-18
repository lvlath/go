// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm_test tests thin public facades against their canonical equivalent.
package gbm_test

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/rng"
)

// TestSimulatePathFacadeParity ensures scalar convenience does not change math.
func TestSimulatePathFacadeParity(t *testing.T) {
	times := []float64{0, 0.04, 0.13, 0.26, 0.27, 0.5}
	p := gbm.Parameters{Initial: 153.75, Drift: 0.035, Volatility: 0.18, Unit: gbm.TimeYears}
	left, err := gbm.SimulatePath(context.Background(), p.Initial, p.Drift, p.Volatility,
		p.Unit, times, &scripted{draws: []float64{0.4, -0.9, 1.1, 0.25, -0.2}})
	if err != nil {
		t.Fatal(err)
	}
	right, err := gbm.Simulate(context.Background(), p,
		gbm.TimeGrid{Values: times, Unit: p.Unit}, &scripted{draws: []float64{0.4, -0.9, 1.1, 0.25, -0.2}})
	if err != nil {
		t.Fatal(err)
	}
	mustEqualInt(t, len(left.Values), len(right.Values), "facade size")
	for index := range left.Values {
		if math.Float64bits(left.Values[index]) != math.Float64bits(right.Values[index]) ||
			math.Float64bits(left.Times[index]) != math.Float64bits(right.Times[index]) {
			t.Fatalf("facade mismatch at index %d", index)
		}
	}
	mustEqualBool(t, left.Complete && right.Complete, true, "facade completion")
}

// TestSimulateTerminalCoversZeroAndPositiveHorizon verifies both wrapper shapes.
func TestSimulateTerminalCoversZeroAndPositiveHorizon(t *testing.T) {
	params := gbm.Parameters{Initial: 75, Drift: 0.05, Volatility: 0.3, Unit: gbm.TimeYears}
	zeroRNG := &scripted{}
	initial, err := gbm.SimulateTerminal(context.Background(), params, 0, zeroRNG)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualInt(t, len(initial.Values), 1, "zero-horizon shape")
	mustEqualInt(t, zeroRNG.count, 0, "zero-horizon Normal draws")

	oneRNG := &scripted{draws: []float64{0.4}}
	terminal, err := gbm.SimulateTerminal(context.Background(), params, 0.75, oneRNG)
	if err != nil {
		t.Fatal(err)
	}
	mustEqualInt(t, len(terminal.Values), 2, "terminal shape")
	mustEqualInt(t, oneRNG.count, 1, "terminal Normal draws")
	mustNear(t, terminal.Values[1], params.Initial*math.Exp(
		(params.Drift-0.5*params.Volatility*params.Volatility)*0.75+
			params.Volatility*math.Sqrt(0.75)*0.4), 1e-13)

	failed, err := gbm.SimulateTerminal(context.Background(), params, -1, &scripted{})
	mustErrorIs(t, err, gbm.ErrOrder)
	mustNilState(t, failed, true, "negative horizon")
}

// TestSimulateSeededMatchesExplicitStream anchors stream identity and replay.
func TestSimulateSeededMatchesExplicitStream(t *testing.T) {
	p := gbm.Parameters{Initial: 110, Drift: 0.03, Volatility: 0.19, Unit: gbm.TimeDays}
	grid := gbm.TimeGrid{Values: []float64{0, .5, 1, 1.5, 2, 3, 4}, Unit: p.Unit}
	first, err := gbm.SimulateSeeded(context.Background(), p, grid, 7192, 15)
	if err != nil {
		t.Fatal(err)
	}
	second, err := gbm.Simulate(context.Background(), p, grid, rng.NewStream(7192, 15))
	if err != nil {
		t.Fatal(err)
	}
	for index := range first.Values {
		if math.Float64bits(first.Values[index]) != math.Float64bits(second.Values[index]) {
			t.Fatalf("seeded facade differs at index %d", index)
		}
	}
}

// TestSimulateWrappersRetainErrorClasses verifies no wrapper hides sentinel errors.
func TestSimulateWrappersRetainErrorClasses(t *testing.T) {
	p := gbm.Parameters{Initial: 1, Volatility: 1, Unit: gbm.TimeYears}
	bad, err := gbm.SimulatePath(context.Background(), p.Initial, p.Drift, p.Volatility,
		p.Unit, nil, &scripted{})
	mustErrorIs(t, err, gbm.ErrNilInput)
	mustNilState(t, bad, true, "nil scalar time grid")

	bad, err = gbm.SimulateTerminal(context.Background(), p, math.NaN(), &scripted{})
	mustErrorIs(t, err, gbm.ErrNonFinite)
	mustNilState(t, bad, true, "nonfinite horizon")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	bad, err = gbm.SimulateSeeded(ctx, p, gbm.TimeGrid{}, 12, 34)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("context must win: %v", err)
	}
	mustNilState(t, bad, true, "pre-canceled seeded facade")
}

// TestSimulateConcurrentOwnedStreams checks actual parallel calls with
// independent logical path identifiers, without mutating any shared RNG.
func TestSimulateConcurrentOwnedStreams(t *testing.T) {
	const paths = 16
	params := gbm.Parameters{Initial: 75, Drift: 0.02, Volatility: 0.3, Unit: gbm.TimeYears}
	grid := gbm.TimeGrid{Values: []float64{0, 0.1, 0.25, 0.4, 0.75, 1}, Unit: params.Unit}

	// Index-addressed workers avoid schedule-dependent publication and make
	// errors inspectable in the parent test goroutine only.
	results := make([]*gbm.Result, paths)
	errorsByPath := make([]error, paths)
	var group sync.WaitGroup
	for index := range results {
		group.Add(1)
		go func(index int) {
			defer group.Done()
			results[index], errorsByPath[index] = gbm.SimulateSeeded(context.Background(),
				params, grid, 20261118, uint64(index))
		}(index)
	}
	group.Wait()

	for index, got := range results {
		if errorsByPath[index] != nil || got == nil || !got.Complete {
			t.Fatalf("path %d: result=%+v error=%v", index, got, errorsByPath[index])
		}
		sequential, err := gbm.SimulateSeeded(context.Background(), params, grid, 20261118, uint64(index))
		if err != nil {
			t.Fatal(err)
		}
		for point := range got.Values {
			if math.Float64bits(got.Values[point]) != math.Float64bits(sequential.Values[point]) {
				t.Fatalf("path=%d point=%d violates worker-independent replay", index, point)
			}
		}
	}
}
