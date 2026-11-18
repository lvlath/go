// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion_test verifies public wrapper equivalence and input precedence.
// These tests never reach into the unexported numerical kernels.
package jumpdiffusion_test

import (
	"context"
	"math"
	"testing"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/jumpdiffusion"
	"github.com/lvlath/go/rng"
)

// mustSamePath verifies both result payload and IEEE-754 output identity.
// It intentionally checks each element rather than comparing pointer identities.
func mustSamePath(t *testing.T, left, right *jumpdiffusion.Result) {
	t.Helper()
	if left == nil || right == nil || left.Complete != right.Complete ||
		len(left.Times) != len(right.Times) || len(left.Values) != len(right.Values) {
		t.Fatalf("path shape or completion mismatch: %+v, %+v", left, right)
	}
	for index := range left.Times {
		if math.Float64bits(left.Times[index]) != math.Float64bits(right.Times[index]) ||
			math.Float64bits(left.Values[index]) != math.Float64bits(right.Values[index]) {
			t.Fatalf("path differs at observation %d", index)
		}
	}
}

func TestPublicFacadeWrappers(t *testing.T) {
	p, grid := jdParams(), jdGrid()
	base, err := jumpdiffusion.Simulate(context.Background(), p, grid, rng.NewStream(412, 19))
	if err != nil {
		t.Fatal(err)
	}

	fromTimes, err := jumpdiffusion.SimulatePath(context.Background(), p, grid.Values, rng.NewStream(412, 19))
	if err != nil {
		t.Fatal(err)
	}
	mustSamePath(t, base, fromTimes)

	seeded, err := jumpdiffusion.SimulateSeeded(context.Background(), p, grid, 412, 19)
	if err != nil {
		t.Fatal(err)
	}
	mustSamePath(t, base, seeded)

	terminalGrid := gbm.TimeGrid{Values: []float64{0, 1}, Unit: p.Diffusion.Unit}
	wantTerminal, err := jumpdiffusion.Simulate(context.Background(), p, terminalGrid, rng.NewStream(412, 19))
	if err != nil {
		t.Fatal(err)
	}
	gotTerminal, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 1, rng.NewStream(412, 19))
	if err != nil {
		t.Fatal(err)
	}
	mustSamePath(t, wantTerminal, gotTerminal)

	// A wrapper may not turn the borrowed grid into live result storage.
	grid.Values[1] = 1234
	if fromTimes.Times[1] != 0.25 || seeded.Times[1] != 0.25 {
		t.Fatal("wrapper result aliases the input grid")
	}
}

func TestTerminalZeroAndInvalidHorizon(t *testing.T) {
	p := jdParams()
	random := &scriptedRNG{}
	initialOnly, err := jumpdiffusion.SimulateTerminal(context.Background(), p, 0, random)
	if err != nil {
		t.Fatal(err)
	}
	if !initialOnly.Complete || len(initialOnly.Values) != 1 || initialOnly.Values[0] != p.Diffusion.Initial ||
		random.ni != 0 || random.ui != 0 {
		t.Fatalf("zero-horizon contract violation: result=%+v stream=%+v", initialOnly, random)
	}
	for _, tc := range []struct {
		horizon float64
		want    error
	}{
		{-1, jumpdiffusion.ErrOrder},
		{math.NaN(), jumpdiffusion.ErrNonFinite},
		{math.Inf(1), jumpdiffusion.ErrNonFinite},
	} {
		res, callErr := jumpdiffusion.SimulateTerminal(context.Background(), p, tc.horizon, random)
		mustErrorIs(t, callErr, tc.want)
		if res != nil {
			t.Fatalf("preflight returned partial path: %+v", res)
		}
	}
}

// nilableRNG deliberately supports the optional reflect-free typed-nil probe.
// It must be rejected without calling any entropy-producing method.
type nilableRNG struct{}

func (r *nilableRNG) IsNil() bool   { return r == nil }
func (*nilableRNG) Uint64() uint64  { return 0 }
func (*nilableRNG) Normal() float64 { return 0 }
func (*nilableRNG) UniformOpen() float64 {
	return 0.5
}

func TestPreflightPrecedenceAndTypedNil(t *testing.T) {
	p, grid := jdParams(), jdGrid()
	p.Diffusion.Unit = gbm.TimeSeconds
	res, err := jumpdiffusion.Simulate(context.Background(), p, grid, nil)
	mustErrorIs(t, err, jumpdiffusion.ErrInvalidOption) // Units before nil RNG.
	if res != nil {
		t.Fatal("unit mismatch returned Result")
	}

	p = jdParams()
	var typedNil *nilableRNG
	res, err = jumpdiffusion.Simulate(context.Background(), p, grid, typedNil)
	mustErrorIs(t, err, jumpdiffusion.ErrNilInput)
	if res != nil {
		t.Fatal("typed-nil input returned Result")
	}

	p.Intensity = -1
	grid.Values = nil
	res, err = jumpdiffusion.Simulate(context.Background(), p, grid, rng.NewStream(1, 1))
	mustErrorIs(t, err, jumpdiffusion.ErrNilInput) // Presence before numeric domain.
	if res != nil {
		t.Fatal("nil grid returned Result")
	}

	p = jdParams()
	p.Intensity = -1
	grid.Values = []float64{0, 1, 1}
	res, err = jumpdiffusion.Simulate(context.Background(), p, grid, rng.NewStream(1, 1))
	mustErrorIs(t, err, jumpdiffusion.ErrOrder) // Timestamp order before coefficients.
	if res != nil {
		t.Fatal("unordered grid returned Result")
	}

	p = jdParams()
	p.Intensity = 0
	p.LogJumpMean = math.NaN()
	grid = jdGrid()
	res, err = jumpdiffusion.Simulate(context.Background(), p, grid, rng.NewStream(1, 1))
	mustErrorIs(t, err, jumpdiffusion.ErrNonFinite) // No lambda-zero validation bypass.
	if res != nil {
		t.Fatal("nonfinite jump mean returned Result")
	}
}

func TestWrapperCancellationIdentity(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	res, err := jumpdiffusion.SimulateSeeded(ctx, jdParams(), jdGrid(), 7, 8)
	mustErrorIs(t, err, context.Canceled)
	if res != nil {
		t.Fatal("pre-canceled wrapper returned Result")
	}
	res, err = jumpdiffusion.SimulatePath(nil, jdParams(), []float64{0, 1}, rng.NewStream(7, 8))
	mustErrorIs(t, err, jumpdiffusion.ErrInvalidOption)
	if res != nil {
		t.Fatal("nil context returned Result")
	}
}

// TestTerminalSeededDelegates checks that the combined convenience facade
// changes neither horizon semantics nor the original RNG sequence.
func TestTerminalSeededDelegates(t *testing.T) {
	model := jdParams()
	for _, horizon := range []float64{0, 0.3, 1} {
		want, err := jumpdiffusion.SimulateTerminal(context.Background(), model, horizon, rng.NewStream(705, 11))
		if err != nil {
			t.Fatal(err)
		}
		got, err := jumpdiffusion.SimulateTerminalSeeded(context.Background(), model, horizon, 705, 11)
		if err != nil {
			t.Fatal(err)
		}
		mustSamePath(t, want, got)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	out, err := jumpdiffusion.SimulateTerminalSeeded(ctx, model, 1, 705, 11)
	mustErrorIs(t, err, context.Canceled)
	if out != nil {
		t.Fatal("pre-canceled seeded terminal wrapper returned a result")
	}
}
