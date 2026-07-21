package flow_test

import (
	"errors"
	"math"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/flow"
)

func mustNoError(t *testing.T, err error, op string) {
	t.Helper()
	if err != nil {
		t.Fatalf("%s: unexpected error: %v", op, err)
	}
}

func mustErrorIs(t *testing.T, err error, target error, op string) {
	t.Helper()
	if !errors.Is(err, target) {
		t.Fatalf("%s: want errors.Is(err, %v); got %v", op, target, err)
	}
}

func mustEqualFloat(t *testing.T, got, want float64, op string) {
	t.Helper()
	if math.IsNaN(got) || math.Abs(got-want) > 1e-9 {
		t.Fatalf("%s: got=%g want=%g", op, got, want)
	}
}

func mustEqualBool(t *testing.T, got, want bool, op string) {
	t.Helper()
	if got != want {
		t.Fatalf("%s: got=%t want=%t", op, got, want)
	}
}

func mustGraph(t *testing.T, opts ...core.GraphOption) *core.Graph {
	t.Helper()

	g, err := core.NewGraph(opts...)
	mustNoError(t, err, "core.NewGraph")

	return g
}

func mustAddVertex(t *testing.T, g *core.Graph, id string) {
	t.Helper()

	err := g.AddVertex(id)
	mustNoError(t, err, "AddVertex("+id+")")
}

func mustAddEdge(t *testing.T, g *core.Graph, from, to string, capacity float64) {
	t.Helper()

	_, err := g.AddEdge(from, to, capacity)
	mustNoError(t, err, "AddEdge("+from+","+to+")")
}

func mustFlowValue(t *testing.T, result *flow.MaxFlowResult, want float64, op string) {
	t.Helper()

	if result == nil {
		t.Fatalf("%s: nil result", op)
	}
	mustEqualFloat(t, result.Value, want, op)
}

// mustNoAugmentingPath verifies that a residual graph has no positive residual path.
// It is a proof helper for maximum-flow optimality.
//
// Implementation:
//   - Stage 1: BFS from source in the published residual graph.
//   - Stage 2: Traverse only directed residual edges with Weight > epsilon.
//   - Stage 3: Fail if sink becomes reachable.
//
// Behavior highlights:
//   - Uses core.Neighbors only on the already-published directed residual graph.
//   - Does not infer undirected input semantics.
//
// Inputs:
//   - t: test handle.
//   - residual: directed weighted residual graph.
//   - source: source vertex ID.
//   - sink: sink vertex ID.
//   - epsilon: residual threshold.
//
// Errors:
//   - Test failure on core.Neighbors error or reachable sink.
//
// Determinism:
//   - core.Neighbors provides deterministic edge ordering.
//
// Complexity:
//   - Time O(V + E_res), Space O(V).
//
// AI-Hints:
//   - Use this on result.Residual, not on the input graph.
func mustNoAugmentingPath(
	t *testing.T,
	residual *core.Graph,
	source string,
	sink string,
	epsilon float64,
) {
	t.Helper()

	if residual == nil {
		t.Fatalf("residual graph is nil")
	}

	seen := map[string]bool{source: true}
	queue := []string{source}

	for head := 0; head < len(queue); head++ {
		from := queue[head]

		edges, err := residual.Neighbors(from)
		mustNoError(t, err, "residual.Neighbors("+from+")")

		for _, edge := range edges {
			if edge.Weight <= epsilon || seen[edge.To] {
				continue
			}
			if edge.To == sink {
				t.Fatalf("residual graph still has positive path to sink via %s -> %s", from, sink)
			}

			seen[edge.To] = true
			queue = append(queue, edge.To)
		}
	}
}

// mustResidualDirectedWeighted verifies residual publication policy.
//
// Implementation:
//   - Stage 1: Reject nil result/residual.
//   - Stage 2: Check directed graph flag.
//   - Stage 3: Check weighted graph flag.
//
// Behavior highlights:
//   - The residual graph must not inherit undirected/unweighted input flags.
//
// Inputs:
//   - t: test handle.
//   - result: MaxFlowResult to verify.
//
// Errors:
//   - Test failure when residual policy is violated.
//
// Determinism:
//   - Pure graph-flag checks.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - This helper protects the residual graph law after refactors.
func mustResidualDirectedWeighted(t *testing.T, result *flow.MaxFlowResult) {
	t.Helper()

	if result == nil {
		t.Fatalf("nil MaxFlowResult")
	}
	if result.Residual == nil {
		t.Fatalf("nil residual graph")
	}
	if !result.Residual.Directed() {
		t.Fatalf("residual graph must be directed")
	}
	if !result.Residual.Weighted() {
		t.Fatalf("residual graph must be weighted")
	}
}

// mustCutCapacityEqualsFlow verifies the max-flow/min-cut certificate.
// It compares MaxFlowResult.Value with the effective original cut capacity.
//
// Implementation:
//   - Stage 1: Compute effective cut capacity through cutCapacityFromResult.
//   - Stage 2: Compare it against result.Value using mustEqualFloat.
//
// Behavior highlights:
//   - Honors epsilon filtering.
//   - Works for directed and undirected input graph policies.
//
// Inputs:
//   - t: test handle.
//   - original: original weighted capacity graph.
//   - result: successful MaxFlowResult.
//   - epsilon: same threshold used by MaxFlow.
//   - op: diagnostic operation name.
//
// Errors:
//   - Test failure on nil inputs or mismatched theorem value.
//
// Determinism:
//   - Deterministic for fixed graph, result, and epsilon.
//
// Complexity:
//   - Time O(V_cut + E), Space O(V_cut).
//
// AI-Hints:
//   - Always pass the same epsilon used by the MaxFlow call.
func mustCutCapacityEqualsFlow(
	t *testing.T,
	original *core.Graph,
	result *flow.MaxFlowResult,
	epsilon float64,
	op string,
) {
	t.Helper()

	if original == nil {
		t.Fatalf("nil original graph")
	}
	if result == nil {
		t.Fatalf("nil MaxFlowResult")
	}

	sourceSide := make(map[string]bool, len(result.CutSourceSide))
	for _, vertexID := range result.CutSourceSide {
		sourceSide[vertexID] = true
	}

	capacity := 0.0
	for _, edge := range original.Edges() {
		if edge.From == edge.To {
			continue
		}
		if edge.Weight <= epsilon {
			continue
		}

		fromSourceSide := sourceSide[edge.From]
		toSourceSide := sourceSide[edge.To]

		if edge.Directed {
			if fromSourceSide && !toSourceSide {
				capacity += edge.Weight
			}
			continue
		}

		if fromSourceSide != toSourceSide {
			capacity += edge.Weight
		}
	}

	mustEqualFloat(t, capacity, result.Value, op)
}

// mustSuccessfulCertificate verifies the full success certificate for a max-flow run.
// It checks value, completion state, residual policy, residual no-path, and min-cut theorem.
//
// Implementation:
//   - Stage 1: Verify scalar max-flow value.
//   - Stage 2: Verify the run is complete rather than partial.
//   - Stage 3: Verify residual graph publication policy.
//   - Stage 4: Verify no positive residual source-sink path remains.
//   - Stage 5: Verify effective min-cut capacity equals flow value.
//
// Behavior highlights:
//   - This is the main proof helper for successful runs.
//   - It intentionally validates more than the scalar value.
//
// Inputs:
//   - t: test handle.
//   - original: original weighted capacity graph.
//   - result: MaxFlowResult returned by MaxFlow.
//   - source: source vertex ID.
//   - sink: sink vertex ID.
//   - want: expected flow value.
//   - epsilon: same threshold used by MaxFlow.
//   - op: diagnostic operation name.
//
// Errors:
//   - Test failure when any certificate component is invalid.
//
// Determinism:
//   - Stable for fixed graph, algorithm, epsilon, and residual traversal order.
//
// Complexity:
//   - Time O(V + E + E_res), Space O(V).
//
// AI-Hints:
//   - Prefer this helper over tests that only compare result.Value.
func mustSuccessfulCertificate(
	t *testing.T,
	original *core.Graph,
	result *flow.MaxFlowResult,
	source string,
	sink string,
	want float64,
	epsilon float64,
	op string,
) {
	t.Helper()

	mustFlowValue(t, result, want, op+" value")
	mustEqualBool(t, result.Partial, false, op+" partial")
	mustResidualDirectedWeighted(t, result)
	mustNoAugmentingPath(t, result.Residual, source, sink, epsilon)
	mustCutCapacityEqualsFlow(t, original, result, epsilon, op+" min-cut theorem")
}

// test helper
func residualCapacity(t *testing.T, residual *core.Graph, from string, to string) float64 {
	t.Helper()

	if residual == nil {
		t.Fatalf("residualCapacity(%s,%s): nil residual graph", from, to)
	}

	edges, err := residual.Neighbors(from)
	mustNoError(t, err, "residual.Neighbors("+from+")")

	total := 0.0
	for _, edge := range edges {
		if edge.To == to {
			total += edge.Weight
		}
	}

	return total
}
