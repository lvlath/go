// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mst_test runs the independently executable Steiner network scenarios.
// Each standalone example compiles against the actual public API and prints a deterministic contract witness.
package mst_test

import (
	"os/exec"
	"strings"
	"testing"
)

// TestMSTSteinerScenarios_Run verifies three standalone graph/network workflows.
// It prevents examples from drifting into unbuildable snippets or unverified output claims.
// Implementation:
//   - Stage 1: Select each standalone Go source and its invariant output fragments.
//   - Stage 2: Execute go run from the module root to resolve imports as consumers do.
//   - Stage 3: Match distinct, deterministic facts about network and traversal costs.
//
// Behavior highlights:
//   - Each scenario is a separately compiled main package, not a combined examples binary.
//
// Inputs:
//   - The repository's examples/mst_steiner_*.go sources and go.mod.
//
// Returns:
//   - Test success only if every scenario compiles, runs, and prints the expected contract facts.
//
// Errors:
//   - Subprocess/build errors and output mismatches fail the corresponding subtest.
//
// Determinism:
//   - Fixed input records and sorted result-witness publication, without clock or randomness.
//
// Complexity:
//   - Per case one Go build and one fixed-size graph computation; no unbounded subprocess loops.
//
// Notes:
//   - This is a live documentation execution test and requires a compatible Go toolchain.
//
// AI-Hints:
//   - Do not replace go run with a textual compile-only inspection; the runtime witness matters.
//   - Do not combine standalone main files in one go run invocation.
func TestMSTSteinerScenarios_Run(t *testing.T) {
	scenarios := []struct {
		name, filename string
		fragments      []string
	}{
		{
			name:     "chiplet-network-and-walk",
			filename: "mst_steiner_chiplet_routing.go",
			fragments: []string{
				"full_mst=15 steiner=8 terminals=4 selected_edges=4",
				"closed=16 fixed=12 closed_passages=8 fixed_passages=6",
				"selected=CPU-NOC CPU--NoC weight=2",
			},
		},
		{
			name:     "emergency-nonterminal-island",
			filename: "mst_steiner_emergency_mesh.go",
			fragments: []string{
				"strict_disconnected=true terminal_network=6 visited_vertices=4",
				"service_start=Depot service_end=Shelter passages=4 distance=7",
				"installed=hospital-hub weight=1",
			},
		},
		{
			name:     "fiber-multigraph-original-ids",
			filename: "mst_steiner_fiber_audit.go",
			fragments: []string{
				"same_network=true installed_cost=5 walk_distance=7",
				"step=1 HQ -> SW via=a-supply",
				"step=4 SW -> DR via=backup",
			},
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			cmd := exec.Command("go", "run", "./examples/"+scenario.filename)
			cmd.Dir = ".."
			output, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("go run %s: %v\n%s", scenario.filename, err, output)
			}
			text := string(output)
			for _, fragment := range scenario.fragments {
				if !strings.Contains(text, fragment) {
					t.Errorf("%s: missing %q\ncomplete output:\n%s", scenario.filename, fragment, text)
				}
			}
		})
	}
}
