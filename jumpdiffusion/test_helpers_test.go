// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion_test provides focused external contract assertions.
// Scripted streams observe public RNG calls without accessing unexported code.
package jumpdiffusion_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/gbm"
	"github.com/lvlath/go/jumpdiffusion"
)

func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("got error %v, want errors.Is(%v)", err, target)
	}
}
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %t, want %t", op, got, want)
	}
}
func mustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d, want %d", op, got, want)
	}
}
func mustEqualString(t *testing.T, got, want, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %q, want %q", op, got, want)
	}
}
func isNilLike(value interface{ IsNil() bool }) bool { return value == nil || value.IsNil() }
func mustNilState(t *testing.T, value interface{ IsNil() bool }, want bool, op string) {
	t.Helper()
	mustEqualBool(t, isNilLike(value), want, op)
}
func mustApprox(t *testing.T, got, want, tol float64) {
	t.Helper()
	if math.IsNaN(got) || math.Abs(got-want) > tol {
		t.Fatalf("got %.17g, want %.17g +/- %.3g", got, want, tol)
	}
}
func jdParams() jumpdiffusion.Parameters {
	return jumpdiffusion.Parameters{Diffusion: gbm.Parameters{Initial: 100, Drift: 0.05, Volatility: 0.2, Unit: gbm.TimeYears}, Intensity: 1, LogJumpMean: -0.1, LogJumpStdDev: 0.1, Convention: jumpdiffusion.Compensated}
}
func jdGrid() gbm.TimeGrid {
	return gbm.TimeGrid{Values: []float64{0, 0.25, 0.5, 0.75, 1}, Unit: gbm.TimeYears}
}

type scriptedRNG struct {
	uniforms, normals []float64
	ui, ni            int
	onUniform         func()
}

func (*scriptedRNG) Uint64() uint64 { return 1 }
func (r *scriptedRNG) UniformOpen() float64 {
	if r.onUniform != nil {
		r.onUniform()
	}
	r.ui++
	if len(r.uniforms) == 0 {
		return 0.5
	}
	return r.uniforms[(r.ui-1)%len(r.uniforms)]
}
func (r *scriptedRNG) Normal() float64 {
	r.ni++
	if len(r.normals) == 0 {
		return 0
	}
	return r.normals[(r.ni-1)%len(r.normals)]
}
