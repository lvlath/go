// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax_test validates source-time provenance, purge and embargo separately.
// Missing, mismatched and crossed supports are never admitted as train-only data.
package tfsax_test

import (
	"errors"
	"testing"

	"github.com/lvlath/go/tfsax"
)

func temporalFixture() (tfsax.SplitPlan, []tfsax.WindowRecord) {
	plan := tfsax.SplitPlan{
		Train:       tfsax.Interval{Start: 0, End: 200},
		Calibration: tfsax.Interval{Start: 250, End: 450},
		Test:        tfsax.Interval{Start: 500, End: 700},
		TrainIDs:    []tfsax.CandidateID{1, 2}, CalibrationIDs: []tfsax.CandidateID{3},
		TestIDs: []tfsax.CandidateID{4}, FitIDs: []tfsax.CandidateID{1, 2},
	}
	records := []tfsax.WindowRecord{
		{ID: 1, Split: tfsax.SplitTrain, Support: tfsax.Interval{Start: 20, End: 52}, LabelEnd: 56},
		{ID: 2, Split: tfsax.SplitTrain, Support: tfsax.Interval{Start: 75, End: 107}, LabelEnd: 111},
		{ID: 3, Split: tfsax.SplitCalibration, Support: tfsax.Interval{Start: 270, End: 302}, LabelEnd: 306},
		{ID: 4, Split: tfsax.SplitTest, Support: tfsax.Interval{Start: 520, End: 552}, LabelEnd: 556},
	}
	return plan, records
}

func TestWindowRecords_SourceAndLabelContainment(t *testing.T) {
	plan, records := temporalFixture()
	if err := tfsax.ValidateWindowRecords(plan, records, 32, 4, 32, 0); err != nil {
		t.Fatal(err)
	}
	injected := append([]tfsax.WindowRecord(nil), records...)
	injected[3].Split = tfsax.SplitTrain
	mustErrorIs(t, tfsax.ValidateWindowRecords(plan, injected, 32, 4, 32, 0), tfsax.ErrLeakage)
	injected = append([]tfsax.WindowRecord(nil), records...)
	injected[2].Support = tfsax.Interval{Start: 420, End: 452}
	injected[2].LabelEnd = 456
	mustErrorIs(t, tfsax.ValidateWindowRecords(plan, injected, 32, 4, 32, 0), tfsax.ErrLeakage)
	injected = append([]tfsax.WindowRecord(nil), records...)
	injected[0].LabelEnd = 57
	mustErrorIs(t, tfsax.ValidateWindowRecords(plan, injected, 32, 4, 32, 0), tfsax.ErrInvalidOption)
	injected = append([]tfsax.WindowRecord(nil), records...)
	injected[0].Support.End = 50
	mustErrorIs(t, tfsax.ValidateWindowRecords(plan, injected, 32, 4, 32, 0), tfsax.ErrInvalidOption)
	mustErrorIs(t, tfsax.ValidateWindowRecords(plan, records[:3], 32, 4, 32, 0), tfsax.ErrInvalidInput)
	injected = append([]tfsax.WindowRecord(nil), records...)
	injected[3] = injected[2]
	mustErrorIs(t, tfsax.ValidateWindowRecords(plan, injected, 32, 4, 32, 0), tfsax.ErrDuplicateID)
	injected = append([]tfsax.WindowRecord(nil), records...)
	injected[3].ID = 500
	mustErrorIs(t, tfsax.ValidateWindowRecords(plan, injected, 32, 4, 32, 0), tfsax.ErrInvalidInput)
}

func TestWindowRecords_PurgeEmbargoAndFit(t *testing.T) {
	plan, records := temporalFixture()
	if err := tfsax.ValidateWindowRecords(plan, records, 32, 4, 32, 14); err != nil {
		t.Fatal(err)
	}
	mustErrorIs(t, tfsax.ValidateWindowRecords(plan, records, 32, 4, 32, 15), tfsax.ErrLeakage)
	plan.FitIDs = append(plan.FitIDs, 4)
	mustErrorIs(t, tfsax.ValidateWindowRecords(plan, records, 32, 4, 32, 0), tfsax.ErrLeakage)
	plan, records = temporalFixture()
	plan.TrainIDs = append(plan.TrainIDs, 1)
	mustErrorIs(t, tfsax.ValidateWindowRecords(plan, records, 32, 4, 32, 0), tfsax.ErrDuplicateID)
	plan, records = temporalFixture()
	plan.TestIDs = []tfsax.CandidateID{1}
	if !errors.Is(tfsax.ValidateWindowRecords(plan, records, 32, 4, 32, 0), tfsax.ErrLeakage) {
		t.Fatal("cross split membership was accepted")
	}
}
