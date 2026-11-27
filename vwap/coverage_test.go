// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package vwap_test checks overall and per-file coverage using Go's own profile.
// This file contains an opt-in CI gate, not a runtime dependency or market-data loader.
// AI-Hints: generate the profile from this checkout first; never claim a stale profile is current.
package vwap_test

import (
	"os"
	"strings"
	"testing"
)

// TestVWAP_CoverageThreshold enforces the inclusive 92% statement-coverage floor.
// The profile is supplied explicitly so ordinary go test never depends on a generated file.
//
// Implementation:
//   - Stage 1: read LVLATH_VWAP_COVERPROFILE, skipping only when it is not configured.
//   - Stage 2: parse its Go coverage records and discover executable production files.
//   - Stage 3: enforce the exact integer percentage for every file and the package total.
//
// Behavior highlights:
//   - A configured missing, empty, malformed, or incomplete profile fails rather than skipping.
//
// Inputs:
//   - LVLATH_VWAP_COVERPROFILE: absolute path to a fresh profile from the same checkout.
//
// Returns:
//   - A deterministic per-file report through testing.T; no production artifact changes.
//
// Errors:
//   - Test failures for unreadable/malformed profiles, missing files, or coverage below 92%.
//
// Determinism:
//   - File names are sorted; eligibility uses integer multiplication, not rounded percentages.
//
// Complexity:
//   - Time/space O(profile records + source bytes); runs only in an explicitly configured gate.
//
// Notes:
//   - The test cannot authenticate a profile's provenance; CI must generate it immediately first.
//
// AI-Hints:
//   - Use an absolute path: Go executes package tests with the package directory as cwd.
//   - Never set the variable during the command that is still generating that same profile.
func TestVWAP_CoverageThreshold(t *testing.T) {
	path := os.Getenv(coverageProfileEnv)
	if path == "" {
		t.Skip("set " + coverageProfileEnv + " to a freshly generated absolute profile path")
	}
	// An explicitly requested gate must not hide a missing or unreadable input.
	profile, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	report, parseErr := readCoverageProfile(profile)
	closeErr := profile.Close()
	if parseErr != nil {
		t.Fatal(parseErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	// Discover source files instead of silently exempting a future executable file.
	files, err := coverageSourceFiles(os.DirFS("."))
	if err != nil {
		t.Fatal(err)
	}
	total, err := validateCoverageFiles(report, files)
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range files {
		counts := report[file.name]
		if !file.executable && counts.statements == 0 {
			t.Logf("%s: N/A (no instrumented statements)", file.name)
			continue
		}
		t.Logf("%s: %d/%d = %.2f%%", file.name, counts.covered, counts.statements, counts.percent())
		if !counts.meetsThreshold() {
			t.Errorf("%s: statement coverage below %d%%", file.name, coverageMinimum)
		}
	}
	t.Logf("TOTAL: %d/%d = %.2f%%", total.covered, total.statements, total.percent())
	if !total.meetsThreshold() {
		t.Errorf("package statement coverage below %d%%", coverageMinimum)
	}
}

// TestCoverage_ProfileValidation protects the gate against incomplete or malformed evidence.
// Cases are in-memory Go strings; no external profile fixture is required by normal tests.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: parse fixed in-memory valid and adversarial Go profiles.
//   - Stage 3: require correct statement totals or explicit rejection.
//
// Behavior highlights:
//   - This test protects evidence parsing against malformed or duplicate records without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Execution frequency must not multiply covered statement counts.
func TestCoverage_ProfileValidation(t *testing.T) {
	for _, fixture := range coverageProfileFixtures() {
		t.Run(fixture.name, func(t *testing.T) {
			report, err := readCoverageProfile(strings.NewReader(fixture.text))
			if fixture.bad {
				if err == nil {
					t.Fatal("malformed profile was accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			got := report["api.go"]
			if got != fixture.want {
				t.Fatalf("counts=%+v, want=%+v", got, fixture.want)
			}
		})
	}
}

// TestCoverage_ThresholdAndFileInventory tests exact threshold arithmetic and source discovery.
// Missing executable files cannot be hidden by a high percentage in a different file.
//
// Implementation:
//   - Stage 1: prepare explicit, reproducible inputs for this contract boundary.
//   - Stage 2: check boundary ratios and an in-memory Go source tree.
//   - Stage 3: reject missing/removed sources and admit only inclusive exact threshold cases.
//
// Behavior highlights:
//   - This test protects exact 92-percent admission and executable-file inventory without altering production visibility.
//
// Inputs:
//   - The Go testing harness and fixed/procedural cases declared in code.
//
// Returns:
//   - A test or fuzz verdict; no public package state or external data is modified.
//
// Errors:
//   - Unexpected public errors, sentinel classes, masks, or output bits fail the owning test.
//
// Determinism:
//   - Case order, fixed seeds, and logical cancellation are explicit; fuzz inputs select reproducible cases.
//
// Complexity:
//   - The sum of the exercised public calls and independent oracle work; no performance claim is inferred from test duration.
//
// Notes:
//   - Runtime allocation counters and race instrumentation are checked separately from numerical equality.
//
// AI-Hints:
//   - Do not use display-rounded percentages for the gate decision.
func TestCoverage_ThresholdAndFileInventory(t *testing.T) {
	for _, counts := range []coverageCounts{{92, 100}, {23, 25}, {100, 100}} {
		if !counts.meetsThreshold() {
			t.Fatalf("inclusive threshold rejected %+v", counts)
		}
	}
	for _, counts := range []coverageCounts{{9199, 10000}, {0, 0}, {0, 10}} {
		if counts.meetsThreshold() {
			t.Fatalf("insufficient coverage accepted %+v", counts)
		}
	}
	files, err := coverageSourceFiles(coverageSourceFixture())
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 || files[0].name != "api.go" || !files[0].executable || files[1].executable {
		t.Fatalf("wrong executable inventory: %+v", files)
	}
	good := map[string]coverageCounts{"api.go": {23, 25}}
	total, err := validateCoverageFiles(good, files)
	if err != nil || total != good["api.go"] {
		t.Fatalf("valid report: total=%+v error=%v", total, err)
	}
	if _, err := validateCoverageFiles(map[string]coverageCounts{}, files); err == nil {
		t.Fatal("missing executable file accepted")
	}
	good["removed.go"] = coverageCounts{1, 1}
	if _, err := validateCoverageFiles(good, files); err == nil {
		t.Fatal("profile for a removed source file accepted")
	}
}
