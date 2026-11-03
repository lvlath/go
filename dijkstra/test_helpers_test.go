// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

package dijkstra_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
)

// AI-HINTS (file):
//   - Tests protect the package contract, not accidental implementation details.
//   - Use errors.Is for error protocol checks; never compare error strings.
//   - Keep helpers small, explicit, and deterministic.
//   - Prefer exact comparisons when the package contract guarantees exact deterministic output.
//   - Keep nil-state handling reflect-free and Nilable-aware.
//   - Never ignore graph fixture construction errors.
//   - Add regression anchors for every real bug fixed in the package.

// assertInfDistance fails the test unless got is positive infinity.
// The helper is intended for the canonical “known but unreachable” distance state.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Require math.IsInf(got, +1).
//
// Behavior highlights:
//   - The helper checks semantic unreachable-state, not just numeric equality.
//
// Inputs:
//   - got: observed distance value.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure if got is not +Inf.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Use this helper for canonical unreachable-distance assertions.
//
// AI-Hints:
//   - Prefer this helper over ad-hoc equality checks when asserting unreachable results.
func assertInfDistance(t *testing.T, got float64) {
	t.Helper()

	if !math.IsInf(got, 1) {
		t.Fatalf("expected +Inf distance, got=%v", got)
	}
}

// assertPathEqual fails the test unless got and want match exactly element-by-element.
// The helper enforces the deterministic path contract of the package.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Compare lengths.
//   - Stage 3: Compare each element in deterministic order.
//
// Behavior highlights:
//   - Exact order is required.
//   - The helper is appropriate only because PathTo is deterministic by contract.
//
// Inputs:
//   - got: actual reconstructed path.
//   - want: expected reconstructed path.
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
//   - Time O(n), Space O(1), where n is len(want).
//
// Notes:
//   - Do not replace this with unordered set comparison.
//
// AI-Hints:
//   - Path order matters here because dijkstra fixes deterministic predecessor selection.
func assertPathEqual(t *testing.T, got, want []string) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("path length mismatch: got=%d want=%d got=%v want=%v", len(got), len(want), got, want)
	}

	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("path mismatch at index %d: got=%q want=%q got=%v want=%v", index, got[index], want[index], got, want)
		}
	}
}

// mustErrorIs fails the test unless err matches target through errors.Is.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Require a non-nil error.
//   - Stage 3: Check sentinel compatibility through errors.Is.
//
// Behavior highlights:
//   - The helper validates error protocol rather than message text.
//
// Inputs:
//   - err: actual error.
//   - target: expected sentinel or wrapped target.
//
// Returns:
//   - N/A.
//
// Errors:
//   - Fatal test failure if protocol matching fails.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(chain length), Space O(1).
//
// Notes:
//   - This helper replaces any string-based error matching.
//
// AI-Hints:
//   - Never replace errors.Is with substring matching in dijkstra tests.
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
//   - Stage 3: Abort if observed nil-state differs from expectation.
//
// Behavior highlights:
//   - Reflect-free.
//   - Nilable-aware.
//   - Supports the explicit container/value kinds used in dijkstra tests.
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
//   - Fatal test failure if nil-state does not match expectation.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper centralizes typed-nil handling without reflection.
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
//   - Stage 2: Delegate to core.Nilable when available.
//   - Stage 3: Handle a small explicit set of nilable test value kinds.
//   - Stage 4: Treat all remaining values as non-nil-like.
//
// Behavior highlights:
//   - Reflect-free and deterministic.
//   - Nilable is the primary typed-nil mechanism.
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
//   - Extend deliberately if future tests introduce a new nilable value kind.
//
// AI-Hints:
//   - Reuse core.Nilable rather than duplicating typed-nil logic per test.
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
	case map[string]float64:
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
//   - Stage 3: Emit either a custom failure message or a default mismatch message.
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
//   - Use for summary flags and reachability predicates.
//
// AI-Hints:
//   - Keep boolean assertions explicit in contract tests.
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

// mustEqualString fails the test if got != want.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Compare strings directly.
//   - Stage 3: Emit either a custom failure message or a default mismatch message.
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
//   - Use for exact IDs and stable parent values.
//
// AI-Hints:
//   - Prefer exact string checks when the contract fixes the exact value.
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

// mustEqualInt fails the test when got and want differ.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Compare integer values directly.
//   - Stage 3: Emit caller-provided context or a canonical mismatch message.
//
// Behavior highlights:
//   - Supports precise assertions for vertex counts, edge counts, and map sizes.
//   - Does not use reflection or an external assertion framework.
//
// Inputs:
//   - got: the observed integer value.
//   - want: the required integer value.
//   - format: optional failure-message format.
//   - args: optional failure-message arguments.
//
// Returns:
//   - None.
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
//   - Use this helper only for exact integer contracts.
//
// AI-Hints:
//   - Prefer explicit count assertions over indirect topology assumptions.
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

// mustEqualFloat64 fails the test if got != want.
// The helper is intended for exact-value checks on deterministic test fixtures
// that avoid tolerance-based numeric contracts.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Compare float64 values directly.
//   - Stage 3: Emit either a custom failure message or a default mismatch message.
//
// Behavior highlights:
//   - Supports optional custom failure context.
//   - Intended for exact fixture values, not tolerance-based comparisons.
//
// Inputs:
//   - got: actual float64 value.
//   - want: expected float64 value.
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
//   - Use assertInfDistance for canonical +Inf checks.
//
// AI-Hints:
//   - Keep fixture values mathematically exact when relying on direct float equality.
func mustEqualFloat64(t *testing.T, got, want float64, format string, args ...any) {
	t.Helper()

	if got == want {
		return
	}

	if format != "" {
		t.Fatalf(format, args...)
	}

	t.Fatalf("float64 mismatch: got=%v want=%v", got, want)
}

// mustPanicFree fails the test if fn panics.
//
// Implementation:
//   - Stage 1: Mark the helper frame.
//   - Stage 2: Execute fn under a deferred panic guard.
//   - Stage 3: Fail if a panic is recovered.
//
// Behavior highlights:
//   - Useful for regression anchors on safety-sensitive paths.
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
//   - Deterministic except for the recovered panic payload.
//
// Complexity:
//   - Time depends on fn, Space O(1) in the helper itself.
//
// Notes:
//   - This helper validates panic safety only.
//
// AI-Hints:
//   - Use this helper for regressions around nil-safe result methods and wrapper safety.
func mustPanicFree(t *testing.T, fn func()) {
	t.Helper()

	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("unexpected panic: %v", recovered)
		}
	}()

	fn()
}

// mustBalancedError checks canonical errors.Is identity without matching messages.
// No partially constructed PathResult may accompany a rejected search call.
func mustBalancedError(t *testing.T, result *dijkstra.PathResult, err, want error) {
	t.Helper()
	if result != nil || !errors.Is(err, want) {
		t.Fatalf("result=%+v, err=%v; want nil result and errors.Is(%v)", result, err, want)
	}
}

// mustBalancedWitness checks original EdgeIDs, directedness and exact original cost.
// This independent postcondition prevents a reduced key from escaping as Distance.
func mustBalancedWitness(t *testing.T, graph *core.Graph, result *dijkstra.PathResult) {
	t.Helper()
	if result == nil {
		t.Fatal("nil result")
	}
	if !result.Reachable {
		if !math.IsInf(result.Distance, 1) || result.Vertices != nil || result.EdgeIDs != nil {
			t.Fatalf("invalid unreachable publication: %+v", result)
		}
		return
	}
	if len(result.Vertices) == 0 || len(result.Vertices) != len(result.EdgeIDs)+1 ||
		result.Vertices[0] != result.SourceID || result.Vertices[len(result.Vertices)-1] != result.TargetID {
		t.Fatalf("invalid witness shape: %+v", result)
	}
	seen := make(map[string]bool, len(result.Vertices))
	for _, id := range result.Vertices {
		if seen[id] {
			t.Fatalf("repeated original vertex in witness: %+v", result)
		}
		seen[id] = true
	}
	cost := float64(0)
	for i, id := range result.EdgeIDs {
		edge, err := graph.GetEdge(id)
		if err != nil {
			t.Fatalf("EdgeID=%q missing from original graph: %v", id, err)
		}
		from, to := result.Vertices[i], result.Vertices[i+1]
		if !(edge.From == from && edge.To == to) && (edge.Directed || edge.From != to || edge.To != from) {
			t.Fatalf("invalid original orientation edge_id=%q between %q and %q", id, from, to)
		}
		cost += edge.Weight
	}
	if cost != result.Distance {
		t.Fatalf("original edges cost %g differs from published Distance %g", cost, result.Distance)
	}
	vertices, edgeIDs, err := result.Path()
	if err != nil || len(vertices) != len(result.Vertices) || len(edgeIDs) != len(result.EdgeIDs) {
		t.Fatalf("Path did not reproduce validated witness: vertices=%v edges=%v err=%v", vertices, edgeIDs, err)
	}
}
