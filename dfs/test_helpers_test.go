// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dfs_test

import (
	"errors"
	"fmt"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dfs"
)

// AI-HINTS (file):
//   - Tests protect the contract, not accidental implementation details.
//   - Use exact comparisons when the package guarantees deterministic order.
//   - Use invariant helpers only when the contract itself is invariant-based.
//   - Use errors.Is for error protocol checks; never compare error strings.
//   - Keep helpers small, explicit, and mathematically honest.
//   - Do not weaken deterministic contracts with unordered comparisons.
//   - Add regression anchors for every real bug fixed in the package.

// mustNoError fails the test if err is non-nil.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Abort immediately when an unexpected error is present.
//
// Behavior highlights:
//   - The helper is strict and does not reinterpret the error.
//
// Inputs:
//   - err: error expected to be nil.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure if err is non-nil.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Use only for success paths.
//
// AI-Hints:
//   - Prefer mustErrorIs for sentinel-based failure paths.
func mustNoError(t *testing.T, err error) {
	t.Helper()

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// mustErrorIs fails the test unless err matches target via errors.Is.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Require a non-nil error.
//   - Stage 3: Check sentinel compatibility via errors.Is.
//
// Behavior highlights:
//   - The helper validates error protocol, not message text.
//
// Inputs:
//   - err: actual error value.
//   - target: expected sentinel or wrapped target.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure if the protocol check fails.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(chain length), Space O(1).
//
// Notes:
//   - This helper replaces string-based error checks.
//
// AI-Hints:
//   - Never replace errors.Is with substring matching in dfs tests.
func mustErrorIs(t *testing.T, err error, target error) {
	t.Helper()

	if err == nil {
		t.Fatalf("expected error matching %v, got nil", target)
	}

	if !errors.Is(err, target) {
		t.Fatalf("expected errors.Is(err, %v)=true, got err=%v", target, err)
	}
}

// mustNilState validates whether value is nil-like or non-nil-like according to wantNil.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Evaluate nil-ness through isNilLike.
//   - Stage 3: Abort if the observed nil-state differs from the expected one.
//
// Behavior highlights:
//   - Reflect-free.
//   - Correctly handles typed nil when the value implements core.Nilable.
//   - Also supports a small explicit set of nilable containers used in dfs tests.
//
// Inputs:
//   - value: tested value.
//   - wantNil: expected nil-state.
//   - op: short failure label.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure if the nil-state does not match expectation.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper intentionally replaces a duplicated mustNil/mustNotNil pair.
//
// AI-Hints:
//   - Prefer Nilable-aware checks over reflection for typed nil detection.
func mustNilState(t *testing.T, value any, wantNil bool, op string) {
	t.Helper()

	gotNil := isNilLike(value)
	if gotNil == wantNil {
		return
	}

	if wantNil {
		t.Fatalf("FAILED [%s]: expected nil-like value; dynamic_type=%T value=%#v", op, value, value)
	}

	if value == nil {
		t.Fatalf("FAILED [%s]: expected non-nil value, got <nil>", op)
	}

	t.Fatalf("FAILED [%s]: expected non-nil value; dynamic_type=%T value=%#v", op, value, value)
}

// isNilLike reports whether value should be treated as nil in tests.
//
// Implementation:
//   - Stage 1: Detect plain nil.
//   - Stage 2: If value implements core.Nilable, delegate to IsNil.
//   - Stage 3: Handle a small explicit set of nilable container types used in dfs tests.
//   - Stage 4: Treat all other values as non-nil-like.
//
// Behavior highlights:
//   - Reflect-free and deterministic.
//   - Uses core.Nilable as the primary typed-nil mechanism.
//
// Inputs:
//   - value: value to inspect.
//
// Returns:
//   - bool: true if value should be treated as nil.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper is intentionally conservative and explicit.
//   - If a future test introduces a new nilable container, extend the switch deliberately.
//
// AI-Hints:
//   - Reuse core.Nilable instead of duplicating ad-hoc typed-nil logic across tests.
func isNilLike(value any) bool {
	if value == nil {
		return true
	}

	if nilable, ok := value.(core.Nilable); ok && nilable.IsNil() {
		return true
	}

	switch typed := value.(type) {
	case error:
		return typed == nil
	case []string:
		return typed == nil
	case [][]string:
		return typed == nil
	case map[string]bool:
		return typed == nil
	case map[string]int:
		return typed == nil
	case map[string]string:
		return typed == nil
	case *core.Graph:
		return typed == nil
	}

	return false
}

// mustEqualBool fails the test if got != want.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Compare booleans directly.
//   - Stage 3: Emit either a custom message or a default mismatch message.
//
// Behavior highlights:
//   - Supports optional custom failure context.
//
// Inputs:
//   - got: actual boolean value.
//   - want: expected boolean value.
//   - format/args: optional custom failure message.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Keep boolean assertions explicit.
//
// AI-Hints:
//   - Use boolean assertions for summary flags such as HasCycle.
func mustEqualBool(t *testing.T, got, want bool, format string, args ...any) {
	t.Helper()

	if got == want {
		return
	}

	if format != "" {
		t.Fatalf(format, args...)
	}

	t.Fatalf("bool mismatch: got=%t want=%t", got, want)
}

// mustEqualInt fails the test if got != want.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Compare integers directly.
//   - Stage 3: Emit either a custom message or a default mismatch message.
//
// Behavior highlights:
//   - Supports optional custom failure context.
//
// Inputs:
//   - got: actual integer value.
//   - want: expected integer value.
//   - format/args: optional custom failure message.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Use for lengths, depths, and positions.
//
// AI-Hints:
//   - Keep integer checks local and explicit in traversal tests.
func mustEqualInt(t *testing.T, got, want int, format string, args ...any) {
	t.Helper()

	if got == want {
		return
	}

	if format != "" {
		t.Fatalf(format, args...)
	}

	t.Fatalf("int mismatch: got=%d want=%d", got, want)
}

// mustEqualString fails the test if got != want.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Compare strings directly.
//   - Stage 3: Emit either a custom message or a default mismatch message.
//
// Behavior highlights:
//   - Supports optional custom failure context.
//
// Inputs:
//   - got: actual string.
//   - want: expected string.
//   - format/args: optional custom failure message.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(len(string)), Space O(1).
//
// Notes:
//   - Use for exact IDs, parents, or stable signatures.
//
// AI-Hints:
//   - Prefer exact string checks when the contract fixes the value.
func mustEqualString(t *testing.T, got, want string, format string, args ...any) {
	t.Helper()

	if got == want {
		return
	}

	if format != "" {
		t.Fatalf(format, args...)
	}

	t.Fatalf("string mismatch: got=%q want=%q", got, want)
}

// mustEqualSlice fails the test unless got and want match exactly element-by-element.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Compare lengths.
//   - Stage 3: Compare each element in deterministic order.
//
// Behavior highlights:
//   - This helper checks exact order.
//
// Inputs:
//   - got: actual string slice.
//   - want: expected string slice.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - Use only when the public contract guarantees deterministic order.
//
// AI-Hints:
//   - Never weaken deterministic contracts with unordered comparisons.
func mustEqualSlice(t *testing.T, got, want []string) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf(
			"slice length mismatch: got=%d want=%d got=%v want=%v",
			len(got), len(want), got, want,
		)
	}

	for index := range want {
		if got[index] != want[index] {
			t.Fatalf(
				"slice mismatch at index %d: got=%q want=%q got=%v want=%v",
				index, got[index], want[index], got, want,
			)
		}
	}
}

// mustEqualStringSet fails the test unless got and want contain the same string multiset.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Count occurrences in both slices.
//   - Stage 3: Compare the frequency maps exactly.
//
// Behavior highlights:
//   - Order is ignored deliberately.
//
// Inputs:
//   - got: actual string slice.
//   - want: expected string slice.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(n+m), Space O(k).
//
// Notes:
//   - Use only for invariant-based checks.
//
// AI-Hints:
//   - Do not use this helper to mask deterministic-order contracts.
func mustEqualStringSet(t *testing.T, got, want []string) {
	t.Helper()

	gotCounts := make(map[string]int, len(got))
	wantCounts := make(map[string]int, len(want))

	for _, value := range got {
		gotCounts[value]++
	}
	for _, value := range want {
		wantCounts[value]++
	}

	mustEqualIntMap(t, gotCounts, wantCounts)
}

// mustEqualNestedSlice fails the test unless got and want match exactly row-by-row.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Compare outer lengths.
//   - Stage 3: Compare each inner slice exactly.
//
// Behavior highlights:
//   - Exact outer order and exact inner order are both enforced.
//
// Inputs:
//   - got: actual nested string slices.
//   - want: expected nested string slices.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(total elements), Space O(1).
//
// Notes:
//   - Use only when the contract fixes both outer and inner order.
//
// AI-Hints:
//   - For witness cycles with fixed deterministic ordering, use exact comparison instead of set comparison.
func mustEqualNestedSlice(t *testing.T, got, want [][]string) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf(
			"nested slice length mismatch: got=%d want=%d got=%v want=%v",
			len(got), len(want), got, want,
		)
	}

	for index := range want {
		mustEqualSlice(t, got[index], want[index])
	}
}

// mustEqualNestedSliceSet fails the test unless got and want contain the same set of cycles.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Convert each cycle to a stable signature.
//   - Stage 3: Compare signatures as an unordered multiset.
//
// Behavior highlights:
//   - Inner sequence order remains meaningful.
//   - Outer sequence order is ignored.
//
// Inputs:
//   - got: actual nested string slices.
//   - want: expected nested string slices.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(total elements), Space O(number of cycles).
//
// Notes:
//   - Use only when canonical cycle content is fixed but outer cycle order is not part of the contract.
//
// AI-Hints:
//   - Do not use set-based comparison to weaken a deterministic outer-order contract.
func mustEqualNestedSliceSet(t *testing.T, got, want [][]string) {
	t.Helper()

	gotSignatures := make([]string, len(got))
	wantSignatures := make([]string, len(want))

	for index, cycle := range got {
		gotSignatures[index] = joinTestSignature(cycle)
	}
	for index, cycle := range want {
		wantSignatures[index] = joinTestSignature(cycle)
	}

	mustEqualStringSet(t, gotSignatures, wantSignatures)
}

// mustEqualIntMap fails the test unless got and want have the same keys and integer values.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Compare lengths.
//   - Stage 3: Compare expected keys and values.
//   - Stage 4: Reject unexpected extra keys.
//
// Behavior highlights:
//   - Full equality, not partial containment.
//
// Inputs:
//   - got: actual integer map.
//   - want: expected integer map.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic modulo Go map formatting in failure output.
//
// Complexity:
//   - Time O(len(got)+len(want)), Space O(1).
//
// Notes:
//   - Use complete expected maps.
//
// AI-Hints:
//   - Exact depth-map checks are stronger than ad-hoc spot checks.
func mustEqualIntMap(t *testing.T, got, want map[string]int) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("map length mismatch: got=%d want=%d got=%v want=%v", len(got), len(want), got, want)
	}

	for key, wantValue := range want {
		gotValue, ok := got[key]
		if !ok {
			t.Fatalf("map missing key %q: got=%v want=%v", key, got, want)
		}
		if gotValue != wantValue {
			t.Fatalf(
				"map value mismatch for key %q: got=%d want=%d got=%v want=%v",
				key, gotValue, wantValue, got, want,
			)
		}
	}

	for key := range got {
		if _, ok := want[key]; !ok {
			t.Fatalf("map has unexpected key %q: got=%v want=%v", key, got, want)
		}
	}
}

// mustEqualBoolMap fails the test unless got and want have the same keys and boolean values.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Compare lengths.
//   - Stage 3: Compare expected keys and values.
//   - Stage 4: Reject unexpected extra keys.
//
// Behavior highlights:
//   - Full equality, not partial containment.
//
// Inputs:
//   - got: actual boolean map.
//   - want: expected boolean map.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic modulo Go map formatting in failure output.
//
// Complexity:
//   - Time O(len(got)+len(want)), Space O(1).
//
// Notes:
//   - Use for exact visited-set expectations.
//
// AI-Hints:
//   - Exact visited-map checks prevent silent weakening of traversal coverage.
func mustEqualBoolMap(t *testing.T, got, want map[string]bool) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("map length mismatch: got=%d want=%d got=%v want=%v", len(got), len(want), got, want)
	}

	for key, wantValue := range want {
		gotValue, ok := got[key]
		if !ok {
			t.Fatalf("map missing key %q: got=%v want=%v", key, got, want)
		}
		if gotValue != wantValue {
			t.Fatalf(
				"map value mismatch for key %q: got=%t want=%t got=%v want=%v",
				key, gotValue, wantValue, got, want,
			)
		}
	}

	for key := range got {
		if _, ok := want[key]; !ok {
			t.Fatalf("map has unexpected key %q: got=%v want=%v", key, got, want)
		}
	}
}

// mustEqualStringMap fails the test unless got and want have the same keys and string values.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Compare lengths.
//   - Stage 3: Compare expected keys and values.
//   - Stage 4: Reject unexpected extra keys.
//
// Behavior highlights:
//   - Full equality, not partial containment.
//
// Inputs:
//   - got: actual string map.
//   - want: expected string map.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic modulo Go map formatting in failure output.
//
// Complexity:
//   - Time O(len(got)+len(want)), Space O(1).
//
// Notes:
//   - Use for exact DFS parent-map expectations.
//
// AI-Hints:
//   - Exact parent-map checks are appropriate when the DFS-tree shape is deterministic.
func mustEqualStringMap(t *testing.T, got, want map[string]string) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("map length mismatch: got=%d want=%d got=%v want=%v", len(got), len(want), got, want)
	}

	for key, wantValue := range want {
		gotValue, ok := got[key]
		if !ok {
			t.Fatalf("map missing key %q: got=%v want=%v", key, got, want)
		}
		if gotValue != wantValue {
			t.Fatalf(
				"map value mismatch for key %q: got=%q want=%q got=%v want=%v",
				key, gotValue, wantValue, got, want,
			)
		}
	}

	for key := range got {
		if _, ok := want[key]; !ok {
			t.Fatalf("map has unexpected key %q: got=%v want=%v", key, got, want)
		}
	}
}

// mustContainString fails the test unless values contains target.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Scan the slice in order.
//   - Stage 3: Abort if the target is absent.
//
// Behavior highlights:
//   - Membership is checked exactly.
//
// Inputs:
//   - values: string slice to scan.
//   - target: required value.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure if target is absent.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - Use sparingly; exact assertions are usually stronger.
//
// AI-Hints:
//   - Prefer exact expectations unless the contract itself is partial.
func mustContainString(t *testing.T, values []string, target string) {
	t.Helper()

	for _, value := range values {
		if value == target {
			return
		}
	}

	t.Fatalf("expected slice to contain %q, got=%v", target, values)
}

// mustHaveNoKey fails the test if key is present in m.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Abort if key exists.
//
// Behavior highlights:
//   - Narrow helper for root-parent absence checks.
//
// Inputs:
//   - m: map to inspect.
//   - key: key expected to be absent.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure if key is present.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Use for contracts such as “DFS-tree roots have no parent entry”.
//
// AI-Hints:
//   - Keep negative-key checks explicit for root semantics.
func mustHaveNoKey[T any](t *testing.T, m map[string]T, key string) {
	t.Helper()

	if _, ok := m[key]; ok {
		t.Fatalf("expected key %q to be absent, got map=%v", key, m)
	}
}

// mustPanicFree fails the test if fn panics.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Execute fn under a deferred panic guard.
//   - Stage 3: Fail if a panic is recovered.
//
// Behavior highlights:
//   - Useful for safety regressions.
//
// Inputs:
//   - fn: function expected not to panic.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure if fn panics.
//
// Determinism:
//   - Deterministic except for the panic payload value.
//
// Complexity:
//   - Time depends on fn, Space O(1) in the helper itself.
//
// Notes:
//   - This checks panic safety only.
//
// AI-Hints:
//   - Use this helper for regression anchors on previously unsafe paths.
func mustPanicFree(t *testing.T, fn func()) {
	t.Helper()

	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("unexpected panic: %v", recovered)
		}
	}()

	fn()
}

// mustTopoOrderRespectsEdges fails the test unless every dependency u->v is respected.
//
// Implementation:
//   - Stage 1: Build a vertex-to-index position map.
//   - Stage 2: Verify that every source appears before its destination.
//
// Behavior highlights:
//   - Validates the mathematical invariant of topological sorting.
//
// Inputs:
//   - order: topological order under test.
//   - edges: dependency edges represented as {from, to} pairs.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure if any dependency is violated.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(|order| + |edges|), Space O(|order|).
//
// Notes:
//   - Use when multiple valid total orders exist.
//
// AI-Hints:
//   - Prefer exact order checks only when the contract truly fixes one exact order.
func mustTopoOrderRespectsEdges(t *testing.T, order []string, edges [][2]string) {
	t.Helper()

	positions := make(map[string]int, len(order))
	for index, vertexID := range order {
		positions[vertexID] = index
	}

	for _, edge := range edges {
		fromIndex, fromOK := positions[edge[0]]
		toIndex, toOK := positions[edge[1]]

		if !fromOK {
			t.Fatalf("topological order is missing source vertex %q: order=%v", edge[0], order)
		}
		if !toOK {
			t.Fatalf("topological order is missing destination vertex %q: order=%v", edge[1], order)
		}
		if fromIndex >= toIndex {
			t.Fatalf(
				"topological invariant violated for edge %q->%q: order=%v positions=%v",
				edge[0], edge[1], order, positions,
			)
		}
	}
}

// mustCycleResult fails the test unless result matches the exact expected summary and exact cycle order.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Require a non-nil result.
//   - Stage 3: Check the summary flag exactly.
//   - Stage 4: Check witness cycles exactly.
//
// Behavior highlights:
//   - Exact outer order and exact inner order are both enforced.
//
// Inputs:
//   - result: actual cycle-detection result.
//   - wantHasCycle: expected summary flag.
//   - wantCycles: expected exact witness cycles.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(total cycle elements), Space O(1).
//
// Notes:
//   - Use this helper only when deterministic witness ordering is part of the contract.
//
// AI-Hints:
//   - Exact cycle-result comparison is stronger than unordered comparison when canonical output is sorted.
func mustCycleResult(t *testing.T, result *dfs.CycleDetectionResult, wantHasCycle bool, wantCycles [][]string) {
	t.Helper()

	mustNilState(t, result, false, "cycle result")
	mustEqualBool(t, result.HasCycle, wantHasCycle, "")
	mustEqualNestedSlice(t, result.Cycles, wantCycles)
}

// mustCycleResultSet fails the test unless result matches the expected summary and same cycle set.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Require a non-nil result.
//   - Stage 3: Check the summary flag exactly.
//   - Stage 4: Compare witness cycles as an unordered set of exact canonical cycles.
//
// Behavior highlights:
//   - Outer order is ignored.
//   - Inner cycle order remains exact.
//
// Inputs:
//   - result: actual cycle-detection result.
//   - wantHasCycle: expected summary flag.
//   - wantCycles: expected canonical cycle set.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(total cycle elements), Space O(number of cycles).
//
// Notes:
//   - Use only when outer order is intentionally not part of the tested contract.
//
// AI-Hints:
//   - Do not weaken deterministic output contracts by using set comparison unnecessarily.
func mustCycleResultSet(t *testing.T, result *dfs.CycleDetectionResult, wantHasCycle bool, wantCycles [][]string) {
	t.Helper()

	mustNilState(t, result, false, "cycle result")
	mustEqualBool(t, result.HasCycle, wantHasCycle, "")
	mustEqualNestedSliceSet(t, result.Cycles, wantCycles)
}

// joinTestSignature converts a string slice into a deterministic test-only signature.
//
// Implementation:
//   - Stage 1: Build a comma-separated signature in deterministic order.
//
// Behavior highlights:
//   - Internal test helper only.
//
// Inputs:
//   - values: string slice to encode.
//
// Returns:
//   - string: deterministic comma-joined signature.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(total bytes), Space O(total bytes).
//
// Notes:
//   - This helper intentionally stays local to tests.
//
// AI-Hints:
//   - Keep test signatures simple and explicit.
func joinTestSignature(values []string) string {
	if len(values) == 0 {
		return ""
	}

	signature := values[0]
	for index := 1; index < len(values); index++ {
		signature += "," + values[index]
	}

	return signature
}

// mustFmt formats a string for test-data construction.
//
// Implementation:
//   - Stage 1: Delegate to fmt.Sprintf.
//
// Behavior highlights:
//   - Thin readability helper only.
//
// Inputs:
//   - format: format string.
//   - args: format arguments.
//
// Returns:
//   - string: formatted result.
//
// Errors:
//   - N/A.
//
// Determinism:
//   - Deterministic for deterministic inputs.
//
// Complexity:
//   - Time depends on fmt formatting cost, Space depends on output size.
//
// Notes:
//   - Keep use limited to test-data assembly.
//
// AI-Hints:
//   - Prefer direct literals when short enough to remain readable.
func mustFmt(t *testing.T, format string, args ...any) string {
	t.Helper()
	return fmt.Sprintf(format, args...)
}
