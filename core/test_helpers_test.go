// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran
// Package core_test contains test helpers for lvlath/core.
//
// Purpose:
//   - Provide small, deterministic fixtures and assertion utilities for core.Graph.
//   - Keep tests stdlib-only (no third-party assertion frameworks).
//   - Enforce concurrency-safe testing patterns (no *testing.T usage inside goroutines).

package core_test

import (
	"errors"
	"sort"
	"testing"

	"github.com/lvlath/go/core"
)

// Common vertex IDs used across core tests.
const (
	VertexEmpty   = ""
	VertexMissing = "missing"

	VertexA = "A"
	VertexB = "B"
	VertexC = "C"
	VertexD = "D"

	VertexP = "P"
	VertexQ = "Q"

	VertexU = "U"
	VertexV = "V"

	VertexV1 = "V1"
	VertexV2 = "V2"

	VertexX = "X"
	VertexY = "Y"

	VertexBase = "Base"
)

// Common edge IDs used across core tests (avoid magic strings in test bodies).
const (
	EdgeIDMissing = "edge-id-missing"
	EdgeIDFirst   = "e1"
)

// Common weights used across core tests (avoid magic numbers in test bodies).
const (
	Weight0 float64 = 0
	Weight1 float64 = 1
	Weight2 float64 = 2
	Weight3 float64 = 3
	Weight5 float64 = 5
	Weight7 float64 = 7
)

// Common cardinalities used across core tests (avoid magic numbers in test bodies).
const (
	Count0 = 0
	Count1 = 1
	Count2 = 2
	Count3 = 3
	Count4 = 4
)

// Common concurrency sizes used across core tests (avoid magic numbers in test bodies).
const (
	NAtomicEdgeIDs    = 100
	NConcurrentAdds   = 200
	NConcurrentRounds = 100

	NLoops   = 50
	NReaders = 50
	NCloners = 20
)

// ---------- special test data ----------
const (
	vertexJohn    = "John"
	vertexMary    = "Mary"
	vertexAlice   = "Alice"
	vertexBob     = "Bob"
	vertexCharlie = "Charlie"
	vertexDavid   = "David"
	vertexEmma    = "Emma"
	vertexFrank   = "Frank"
	vertexGrace   = "Grace"
	vertexHenry   = "Henry"
	vertexIvy     = "Ivy"
)

const (
	socialVertexCount       = 11
	socialTriangleEdgeCount = 3
	socialGroupEdgeCount    = 11
	socialTotalEdgeCount    = socialTriangleEdgeCount + socialGroupEdgeCount
	socialAfterHenryEdges   = 6
)

func newSocialComponentsGraph(t *testing.T) *core.Graph {
	t.Helper()

	g := MustNewGraph(t)
	for _, id := range []string{
		vertexJohn, vertexMary, // single vertices
		vertexAlice, vertexBob, vertexCharlie, // triangle vertices
		vertexDavid, vertexEmma, vertexFrank, vertexGrace, vertexHenry, vertexIvy, // group vertices
	} {
		MustErrorNil(t, g.AddVertex(id), "AddVertex("+id+")")
	}

	_, err := g.AddEdge(vertexAlice, vertexBob, Weight0, core.WithID("triangle_alice_bob"))
	MustErrorNil(t, err, "AddEdge("+vertexAlice+","+vertexBob+",triangle_alice_bob)")
	_, err = g.AddEdge(vertexBob, vertexCharlie, Weight0, core.WithID("triangle_bob_charlie"))
	MustErrorNil(t, err, "AddEdge("+vertexBob+","+vertexCharlie+",triangle_bob_charlie)")
	_, err = g.AddEdge(vertexCharlie, vertexAlice, Weight0, core.WithID("triangle_charlie_alice"))
	MustErrorNil(t, err, "AddEdge("+vertexCharlie+","+vertexAlice+",triangle_charlie_alice)")
	_, err = g.AddEdge(vertexDavid, vertexEmma, Weight0, core.WithID("group_david_emma"))
	MustErrorNil(t, err, "AddEdge("+vertexDavid+","+vertexEmma+",group_david_emma)")
	_, err = g.AddEdge(vertexDavid, vertexFrank, Weight0, core.WithID("group_david_frank"))
	MustErrorNil(t, err, "AddEdge("+vertexDavid+","+vertexFrank+",group_david_frank)")
	_, err = g.AddEdge(vertexDavid, vertexGrace, Weight0, core.WithID("group_david_grace"))
	MustErrorNil(t, err, "AddEdge("+vertexDavid+","+vertexGrace+",group_david_grace)")
	_, err = g.AddEdge(vertexDavid, vertexHenry, Weight0, core.WithID("group_david_henry"))
	MustErrorNil(t, err, "AddEdge("+vertexDavid+","+vertexHenry+",group_david_henry)")
	_, err = g.AddEdge(vertexEmma, vertexHenry, Weight0, core.WithID("group_emma_henry"))
	MustErrorNil(t, err, "AddEdge("+vertexEmma+","+vertexHenry+",group_emma_henry)")
	_, err = g.AddEdge(vertexEmma, vertexGrace, Weight0, core.WithID("group_emma_grace"))
	MustErrorNil(t, err, "AddEdge("+vertexEmma+","+vertexGrace+",group_emma_grace)")
	_, err = g.AddEdge(vertexEmma, vertexIvy, Weight0, core.WithID("group_emma_ivy"))
	MustErrorNil(t, err, "AddEdge("+vertexEmma+","+vertexIvy+",group_emma_ivy)")
	_, err = g.AddEdge(vertexFrank, vertexHenry, Weight0, core.WithID("group_frank_henry"))
	MustErrorNil(t, err, "AddEdge("+vertexFrank+","+vertexHenry+",group_frank_henry)")
	_, err = g.AddEdge(vertexGrace, vertexHenry, Weight0, core.WithID("group_grace_henry"))
	MustErrorNil(t, err, "AddEdge("+vertexGrace+","+vertexHenry+",group_grace_henry)")
	_, err = g.AddEdge(vertexGrace, vertexIvy, Weight0, core.WithID("group_grace_ivy"))
	MustErrorNil(t, err, "AddEdge("+vertexGrace+","+vertexIvy+",group_grace_ivy)")
	_, err = g.AddEdge(vertexHenry, vertexIvy, Weight0, core.WithID("group_henry_ivy"))
	MustErrorNil(t, err, "AddEdge("+vertexHenry+","+vertexIvy+",group_henry_ivy)")

	return g
}

// ---------- special test data ----------

// MustNewGraph constructs a graph for tests and fails the owning test on constructor error.
//
// Implementation:
// - Stage 1: Call core.NewGraph with the provided options.
// - Stage 2: Fail the test immediately if constructor validation returns an error.
// - Stage 3: Fail the test if the returned graph is nil.
// - Stage 4: Return the validated graph.
//
// Behavior highlights:
// - Centralizes constructor error handling for tests.
// - Keeps test bodies compact after NewGraph becomes error-returning.
//
// Inputs:
// - t: test context.
// - opts: zero or more constructor options.
//
// Returns:
// - *core.Graph: non-nil graph fixture.
//
// Errors:
// - Fatal test failure if core.NewGraph returns an error or nil graph.
//
// Determinism:
// - Deterministic.
//
// Complexity:
// - Time O(len(opts)), Space O(1) excluding graph allocation.
//
// Notes:
// - This is test-only glue; production code should handle constructor errors directly.
//
// AI-Hints:
// - Prefer MustNewGraph(...) over repeating g, err := core.NewGraph(...) in every test.
func MustNewGraph(t *testing.T, opts ...core.GraphOption) *core.Graph {
	t.Helper()

	g, err := core.NewGraph(opts...)
	MustErrorNil(t, err, "core.NewGraph(...)")
	MustNotNil(t, g, "core.NewGraph(...)")

	return g
}

// MustNewMixedGraph constructs a mixed graph for tests and fails the owning test on constructor error.
func MustNewMixedGraph(t *testing.T, opts ...core.GraphOption) *core.Graph {
	t.Helper()

	g, err := core.NewMixedGraph(opts...)
	MustErrorNil(t, err, "core.NewMixedGraph(...)")
	MustNotNil(t, g, "core.NewMixedGraph(...)")

	return g
}

// NewGraphFull RETURNS a Graph configured for broad contract coverage.
//
// Implementation:
//   - Stage 1: Call MustNewGraph with WithWeighted/WithMultiEdges/WithLoops.
//   - Stage 2: Return the constructed *core.Graph.
//
// Behavior highlights:
//   - Enables weights to exercise numeric storage.
//   - Enables multi-edges to exercise parallel-edge semantics.
//   - Enables loops to exercise self-loop semantics.
//
// Inputs:
//   - t: test context.
//
// Returns:
//   - *core.Graph: graph with {Weighted=true, MultiEdges=true, Loops=true}.
//
// Errors:
//   - Fatal test failure if construction fails.
//
// Determinism:
//   - Deterministic configuration (no randomness).
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is a test fixture constructor; it intentionally does not belong to production API.
//   - Keep it here to centralize test policy and avoid boilerplate.
//
// AI-Hints:
//   - Use NewGraphFull when you need maximum feature surface with minimal setup.
//   - For strict-policy tests, prefer building graphs explicitly to isolate constraints.
func NewGraphFull(t *testing.T) *core.Graph {
	t.Helper()
	return MustNewGraph(t, core.WithWeighted(), core.WithMultiEdges(), core.WithLoops())
}

// MustNotNil fails the test if val is nil, including "typed nil" values stored in interfaces.
// This helper is reflect-free and uses core.Nilable when available.
//
// Implementation:
//   - Stage 1: Reject untyped nil (val == nil).
//   - Stage 2: If val implements core.Nilable, call IsNil() to detect typed-nil receivers.
//   - Stage 3: For a small set of common nilable containers used in core tests, check nil directly.
//   - Stage 4: If nil is detected, fail with a type-rich message.
//
// Behavior highlights:
//   - Reflect-free and O(1) on the hot path.
//   - Produces actionable failures by printing the concrete dynamic type.
//
// Inputs:
//   - t: test context.
//   - val: value to validate (often *core.Graph, *core.Edge, slices, maps).
//   - op: short operation label for failure context.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if val is nil (untyped or typed nil).
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper is intentionally conservative: it does not attempt to detect arbitrary typed nils
//     for every possible type without reflection.
//   - Prefer implementing core.Nilable on pointer-backed types that commonly appear behind interfaces.
//
// AI-Hints:
//   - If you hit a typed-nil that is not detected, make that type implement core.Nilable in production code.
func MustNotNil(t *testing.T, val any, op string) {
	t.Helper()

	// Stage 1: Unyped nil interface check.
	if val == nil {
		failNil(t, op, "untyped nil", val)
		return
	}

	// Stage 2: Typed nil check via core.Nilable (the preferred, reflect-free mechanism).
	if n, ok := val.(core.Nilable); ok && n.IsNil() {
		failNil(t, op, "typed nil via core.Nilable", val)
		return
	}

	// Stage 3: Small, explicit set of common nilable containers used in tests.
	switch v := val.(type) {
	case error:
		// A nil concrete error inside an interface is usually caught by val==nil,
		// but we keep this branch to make intent explicit for tests.
		if v == nil {
			failNil(t, op, "typed nil error", val)
			return
		}
	case *int, *int64, *float64:
		if v == nil {
			failNil(t, op, "nil *int|*int64|*float64", val)
			return
		}
	case []string:
		if v == nil {
			failNil(t, op, "nil []string slice", val)
			return
		}
	case []int64:
		if v == nil {
			failNil(t, op, "nil []int64 slice", val)
			return
		}
	case []float64:
		if v == nil {
			failNil(t, op, "nil []float64 slice", val)
			return
		}
	case []*core.Edge:
		if v == nil {
			failNil(t, op, "nil []*core.Edge slice", val)
			return
		}
	case []*core.Vertex:
		if v == nil {
			failNil(t, op, "nil []*core.Vertex slice", val)
			return
		}
	case map[string]*core.Vertex:
		if v == nil {
			failNil(t, op, "nil map[string]*core.Vertex", val)
			return
		}
	case map[string]*core.Edge:
		if v == nil {
			failNil(t, op, "nil map[string]*core.Edge", val)
			return
		}
	}
}

// failNil fails with a type-rich nil diagnosis.
// This is internal to test helpers to keep messages consistent and deterministic.
func failNil(t *testing.T, op string, reason string, val any) {
	t.Helper()

	// %T prints the concrete dynamic type, which is critical for typed-nil debugging.
	if val == nil {
		t.Fatalf("FAILED [%s]: received <nil> (%s)", op, reason)
		return
	}
	t.Fatalf("FAILED [%s]: received nil-like value (%s); dynamic_type=%T", op, reason, val)
}

// MustErrorNil fails the test if err != nil.
//
// Implementation:
//   - Stage 1: If err == nil, return.
//   - Stage 2: Abort via t.Fatalf with operation context.
//
// Behavior highlights:
//   - Keeps error checks explicit and consistent without third-party frameworks.
//
// Inputs:
//   - t: *testing.T.
//   - err: error to validate.
//   - op: short operation label (e.g., "AddEdge(A,B,1)").
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if err != nil.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Prefer call-signature labels for op to speed up failure triage.
func MustErrorNil(t *testing.T, err error, op string) {
	t.Helper()

	if err == nil {
		return
	}

	t.Fatalf("%s: unexpected error: %v", op, err)
}

// MustErrorIs fails the test if !errors.Is(err, target).
//
// Implementation:
//   - Stage 1: Evaluate errors.Is(err, target).
//   - Stage 2: Abort via t.Fatalf with target and actual error.
//
// Behavior highlights:
//   - Enforces sentinel-error contracts precisely (core.Err*).
//
// Inputs:
//   - t: *testing.T.
//   - err: error to inspect.
//   - target: expected sentinel error.
//   - op: operation label for context.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if the sentinel does not match.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(depth of wrapped chain), Space O(1).
//
// AI-Hints:
//   - Assert errors.Is for sentinels; do not string-compare error messages.
func MustErrorIs(t *testing.T, err error, target error, op string) {
	t.Helper()

	if errors.Is(err, target) {
		return
	}

	t.Fatalf("%s: want errors.Is(err,%v)=true; got err=%v", op, target, err)
}

// MustEqualBool fails the test if got != want.
//
// Implementation:
//   - Stage 1: Compare booleans.
//   - Stage 2: Abort via t.Fatalf with got/want.
//
// Inputs:
//   - t: *testing.T.
//   - got, want: boolean values.
//   - op: operation label.
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
// AI-Hints:
//   - Use this instead of ad-hoc MustTrue/MustFalse helpers for centralized style.
func MustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()

	if got == want {
		return
	}

	t.Fatalf("%s: got=%t want=%t", op, got, want)
}

// MustEqualInt FAILS if got != want.
//
// Implementation:
//   - Stage 1: Compare ints.
//   - Stage 2: t.Fatalf with got/want.
//
// Behavior highlights:
//   - Avoids generic helpers to keep test style close to stdlib and explicit.
//
// Inputs:
//   - t: *testing.T.
//   - got, want: int values.
//   - op: operation label.
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
//   - Prefer for counts (Edges/Vertices/Neighbors).
//
// AI-Hints:
//   - Use MustEqualInt(len(x), N, "...") to keep failures actionable.
func MustEqualInt(t *testing.T, got, want int, op string) {
	t.Helper()

	if got == want {
		return
	}

	t.Fatalf("%s: got=%d want=%d", op, got, want)
}

// MustEqualInt64 fails the test if got != want for int64 values.
//
// Implementation:
//   - Stage 1: Mark as helper to attribute failures to the caller.
//   - Stage 2: Compare got and want.
//   - Stage 3: Fail via t.Fatalf with a stable, parseable message on mismatch.
//
// Behavior highlights:
//   - Exact integer equality check with zero allocations.
//   - Failure message includes both values for fast triage.
//
// Inputs:
//   - t: test context.
//   - got: observed int64 value.
//   - want: expected int64 value.
//   - op: short operation label describing the contract under test.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if got != want.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Prefer this helper for counts and sizes represented as int64.
//   - Keep op stable (e.g., "EdgeID numeric suffix", "Atomic counter value").
//
// AI-Hints:
//   - Use MustEqualInt64 for deterministic counters to avoid lossy casting to int on 32-bit targets.
func MustEqualInt64(t *testing.T, got, want int64, op string) {
	t.Helper()

	if got == want {
		return
	}

	t.Fatalf("%s: got=%d want=%d", op, got, want)
}

// MustEqualFloat64 fails the test if got != want for float64 values (exact equality).
//
// Implementation:
//   - Stage 1: Mark as helper to attribute failures to the caller.
//   - Stage 2: Compare got and want exactly.
//   - Stage 3: Fail via t.Fatalf with both values on mismatch.
//
// Behavior highlights:
//   - Exact comparison: no epsilon, no tolerance policy ambiguity.
//   - Useful when exact equality is a strict contract (e.g., sentinel constants, intentionally exact values).
//
// Inputs:
//   - t: test context.
//   - got: observed float64 value.
//   - want: expected float64 value.
//   - op: short operation label describing the contract under test.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if got != want.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This helper intentionally does NOT implement tolerance logic.
//   - If a value is the result of floating-point arithmetic where rounding is expected,
//     define a dedicated helper like MustFloat64WithinEpsilon(t, got, want, eps, op)
//     with an explicit epsilon argument.
//
// AI-Hints:
//   - Use exact float checks only when the value is known to be representable exactly
//     (e.g., 0, 1, 2, powers of two) or when the contract requires exact bitwise stability.
func MustEqualFloat64(t *testing.T, got, want float64, op string) {
	t.Helper()

	if got == want {
		return
	}

	t.Fatalf("%s: got=%g want=%g", op, got, want)
}

// MustEqualString FAILS if got != want.
//
// Implementation:
//   - Stage 1: Compare strings.
//   - Stage 2: t.Fatalf with got/want.
//
// Behavior highlights:
//   - Explicit, readable comparisons.
//
// Inputs:
//   - t: *testing.T.
//   - got, want: strings.
//   - op: operation label.
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
//   - Time O(n) compare, Space O(1).
//
// Notes:
//   - Use for vertex IDs, edge IDs, endpoints.
//
// AI-Hints:
//   - Prefer comparing endpoints via GetEdge(id) rather than scanning Edges().
func MustEqualString(t *testing.T, got, want string, op string) {
	t.Helper()

	if got == want {
		return
	}

	t.Fatalf("%s: got=%q want=%q", op, got, want)
}

// MustNotEqualString FAILS if got == want.
//
// Implementation:
//   - Stage 1: Compare strings.
//   - Stage 2: t.Fatalf with got/want.
//
// Behavior highlights:
//   - Explicit, readable comparisons.
//
// Inputs:
//   - t: *testing.T.
//   - got, want: strings.
//   - op: operation label.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(n) compare, Space O(1).
func MustNotEqualString(t *testing.T, got, want string, op string) {
	t.Helper()

	if got != want {
		return
	}

	t.Fatalf("%s: got=%q want=%q", op, got, want)
}

// MustSortedStrings FAILS if ids are not sorted ascending.
//
// Implementation:
//   - Stage 1: Use sort.StringsAreSorted.
//   - Stage 2: t.Fatalf with the slice.
//
// Behavior highlights:
//   - Enforces deterministic ordering contracts (Vertices/Edges/Neighbors).
//
// Inputs:
//   - t: *testing.T.
//   - ids: slice to validate.
//   - op: operation label.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if not sorted.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(n), Space O(1).
//
// Notes:
//   - Only checks ordering, not uniqueness.
//
// AI-Hints:
//   - Use for determinism guarantees: stable outputs simplify downstream algorithms.
func MustSortedStrings(t *testing.T, ids []string, op string) {
	t.Helper()

	if sort.StringsAreSorted(ids) {
		return
	}

	t.Fatalf("%s: not sorted asc: %v", op, ids)
}

// MustSameStringSet FAILS if a and b are not equal as sets (order-independent).
//
// Implementation:
//   - Stage 1: Copy and sort both slices.
//   - Stage 2: Compare element-wise.
//
// Behavior highlights:
//   - Replaces third-party ElementsMatch with deterministic stdlib logic.
//
// Inputs:
//   - t: *testing.T.
//   - a,b: slices to compare as sets.
//   - op: operation label.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure on mismatch.
//
// Determinism:
//   - Deterministic (sort-based).
//
// Complexity:
//   - Time O(n log n), Space O(n).
//
// Notes:
//   - Requires equal lengths; duplicates are treated as multiplicities.
//
// AI-Hints:
//   - Use when vertex ordering is allowed to vary but membership must be identical.
func MustSameStringSet(t *testing.T, a, b []string, op string) {
	t.Helper()

	if len(a) != len(b) {
		t.Fatalf("%s: len(a)=%d len(b)=%d; a=%v b=%v", op, len(a), len(b), a, b)
	}

	aa := append([]string(nil), a...)
	bb := append([]string(nil), b...)
	sort.Strings(aa)
	sort.Strings(bb)

	var i int
	for i = 0; i < len(aa); i++ {
		if aa[i] != bb[i] {
			t.Fatalf("%s: set mismatch at i=%d; a=%v b=%v", op, i, aa, bb)
		}
	}
}

// ExtractEdgeIDs RETURNS edge IDs preserving the incoming slice order.
//
// Implementation:
//   - Stage 1: Allocate output slice sized to edges.
//   - Stage 2: Copy Edge.ID into output.
//
// Behavior highlights:
//   - Small utility for edge-inventory comparisons.
//
// Inputs:
//   - edges: []*core.Edge.
//
// Returns:
//   - []string: IDs in the same order.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic for a fixed input slice.
//
// Complexity:
//   - Time O(n), Space O(n).
//
// Notes:
//   - Prefer comparing sets via MustSameStringSet if order is not part of the contract.
//
// AI-Hints:
//   - Combine with MustSortedStrings if Edges() ordering is contractual.
func ExtractEdgeIDs(edges []*core.Edge) []string {
	out := make([]string, len(edges))

	var i int
	for i = 0; i < len(edges); i++ {
		out[i] = edges[i].ID
	}

	return out
}

// MustAllErrorsNil fails the test if any non-nil error is received from errCh.
//
// Implementation:
//   - Stage 1: Range over errCh until it is closed.
//   - Stage 2: On the first non-nil error, fail via t.Fatalf.
//
// Behavior highlights:
//   - Enforces the rule "no *testing.T usage inside goroutines":
//     goroutines send errors to a channel; the parent goroutine validates.
//
// Inputs:
//   - t: *testing.T.
//   - errCh: receive-only channel of errors (must be closed by the caller).
//   - op: operation label describing the concurrent scenario.
//
// Returns:
//   - None.
//
// Errors:
//   - Fatal test failure if any error is non-nil.
//
// Determinism:
//   - Deterministic for a fixed produced error sequence.
//
// Complexity:
//   - Time O(k), Space O(1).
//
// AI-Hints:
//   - In concurrent tests, send only unexpected errors to errCh to keep failures signal-rich.
func MustAllErrorsNil(t *testing.T, errCh <-chan error, op string) {
	t.Helper()

	for err := range errCh {
		if err == nil {
			continue
		}
		t.Fatalf("%s: unexpected concurrent error: %v", op, err)
	}
}

func MustGraphCounts(t *testing.T, g *core.Graph, wantVertices, wantEdges int, op string) {
	t.Helper()
	MustEqualInt(t, g.VertexCount(), wantVertices, op+" VertexCount")
	MustEqualInt(t, g.EdgeCount(), wantEdges, op+" EdgeCount")
}

func MustUndirectedDegree(t *testing.T, g *core.Graph, id string, want int) {
	t.Helper()
	in, out, undirected, err := g.Degree(id)
	MustErrorNil(t, err, "Degree("+id+")")
	MustEqualInt(t, in, Count0, "Degree("+id+").in")
	MustEqualInt(t, out, Count0, "Degree("+id+").out")
	MustEqualInt(t, undirected, want, "Degree("+id+").undirected")
}
