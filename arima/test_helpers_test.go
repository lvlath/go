// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
// Package arima_test provides explicit sentinel, numeric, equality and nilability assertions.
// This file avoids reflection and third-party assertion frameworks.
package arima_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/arima"
)

func mustClose(t *testing.T, got, want float64) {
	t.Helper()
	if math.IsNaN(got) || math.Abs(got-want) > 1e-8*(1+math.Abs(want)) {
		t.Fatalf("got %.16g want %.16g", got, want)
	}
}
func mustVector(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length got %d want %d", len(got), len(want))
	}
	for i := range got {
		mustClose(t, got[i], want[i])
	}
}
func mustErrorIs(t *testing.T, err, target error) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("got error %v, want %v", err, target)
	}
}
func mustModel(t *testing.T, p arima.Parameters) *arima.Model {
	t.Helper()
	m, err := arima.New(p)
	if err != nil {
		t.Fatal(err)
	}
	return m
}
func mustNilState(t *testing.T, value interface{ IsNil() bool }, want bool, op string) {
	t.Helper()
	if value.IsNil() != want {
		t.Fatalf("%s: nil state got %v want %v", op, value.IsNil(), want)
	}
}
func isNilLike(value interface{ IsNil() bool }) bool { return value.IsNil() }
func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %v want %v", op, got, want)
	}
}
func mustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %d want %d", op, got, want)
	}
}
func mustEqualString(t *testing.T, got, want, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got %q want %q", op, got, want)
	}
}
