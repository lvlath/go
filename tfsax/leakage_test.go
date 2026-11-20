// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
package tfsax_test

import (
	"errors"
	"testing"

	"github.com/lvlath/go/tfsax"
)

func TestLeakageGapAndMembership(t *testing.T) {
	p := tfsax.SplitPlan{Train: tfsax.Interval{0, 100}, Calibration: tfsax.Interval{110, 200}, Test: tfsax.Interval{240, 300}}
	if !errors.Is(tfsax.ValidateSplit(p, 32, 4, 32, 0), tfsax.ErrLeakage) {
		t.Fatal("inadequate purge accepted")
	}
	p.Calibration.Start = 140
	if err := tfsax.ValidateSplit(p, 32, 4, 32, 0); err != nil {
		t.Fatal(err)
	}
	p.CalibrationIDs = []tfsax.CandidateID{2}
	p.TestIDs = []tfsax.CandidateID{2}
	if !errors.Is(tfsax.ValidateSplit(p, 32, 4, 32, 0), tfsax.ErrLeakage) {
		t.Fatal("overlapping membership")
	}
}
