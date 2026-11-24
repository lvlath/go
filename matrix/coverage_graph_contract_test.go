// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix_test verifies adjacency publication, export and matrix-builder policies.
// All fixtures use public graph and matrix APIs, including caller-owned metadata and storage.
package matrix_test

import (
	"math"
	"slices"
	"testing"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

// TestCoverageAdjacencyMetadataAndGenericDegree verifies stable vertex identity and degree semantics.
// It compares weighted and binary row reductions over a generic Matrix backend, including loops.
// Implementation: Build valid graph snapshots, replace storage with equivalent generic matrices,
// then independently check degrees, detached vertex IDs and error publication.
// Behavior highlights: A loop counts once; classic zero and nonfinite absence do not contribute.
// Inputs: Fixed three-vertex graphs, an explicit directed policy and finite signed weights.
// Returns: Test assertions only; every construction/access error is checked.
// Errors: ErrNilMatrix, ErrNonSquare, ErrDimensionMismatch and backend errors are distinguished.
// Determinism: Vertex order is obtained from the published snapshot; no map iteration drives expectations.
// Complexity: O(V^2+E) work for each small graph, O(V^2) storage.
// Notes: Generic classic adjacency is intentionally used here, not a lossy conversion of +Inf encoding.
// AI-Hints: Preserve the adjacency encoding when replacing its public Mat field.
func TestCoverageAdjacencyMetadataAndGenericDegree(t *testing.T) {
	for _, tc := range []struct {
		name string
		mode matrix.Option
		want []float64
	}{
		{"weighted", matrix.WithWeighted(), []float64{4, -2, 0}},
		{"binary", matrix.WithUnweighted(), []float64{2, 1, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The graph carries the vertex identity; the generic backend carries the numeric row data.
			g, err := core.NewGraph(core.WithDirected(true), core.WithWeighted(), core.WithLoops())
			if err != nil {
				t.Fatal(err)
			}
			for _, id := range []string{"a", "b", "c"} {
				if err = g.AddVertex(id); err != nil {
					t.Fatal(err)
				}
			}
			am, err := matrix.BuildAdjacency(g, matrixMustOptions(t, matrix.WithDirected(), tc.mode, matrix.WithAllowLoops()))
			if err != nil {
				t.Fatal(err)
			}
			ids, err := am.VertexIDs()
			if err != nil || !slices.Equal(ids, []string{"a", "b", "c"}) {
				t.Fatalf("vertex IDs=%v err=%v", ids, err)
			}
			// Mutating the returned IDs must not alter the reverse index used by Neighbors.
			ids[0] = "changed"
			ids, err = am.VertexIDs()
			if err != nil || ids[0] != "a" {
				t.Fatalf("vertex ID alias: %v err=%v", ids, err)
			}
			count, err := am.VertexCount()
			if err != nil || count != 3 {
				t.Fatalf("vertex count=%d err=%v", count, err)
			}
			// A weighted self-loop contributes one rather than its stored weight.
			raw := coverageRawDense(t, 3, 3, []float64{2, 3, math.Inf(1), 0, 0, -2, math.NaN(), 0, 0})
			am.Mat = raw
			denseDegrees, err := matrix.DegreeVector(am)
			if err != nil || !slices.Equal(denseDegrees, tc.want) {
				t.Fatalf("dense degree=%v want=%v err=%v", denseDegrees, tc.want, err)
			}
			am.Mat = &coverageMatrix{Matrix: raw}
			degrees, err := matrix.DegreeVector(am)
			if err != nil || !slices.Equal(degrees, tc.want) {
				t.Fatalf("degree=%v want=%v err=%v", degrees, tc.want, err)
			}
			neighbors, err := am.Neighbors("a")
			if err != nil || !slices.Equal(neighbors, []string{"b"}) {
				t.Fatalf("neighbors=%v err=%v", neighbors, err)
			}
			// Refusing a valid backend cell must preserve the backend sentinel and publish no slice.
			am.Mat = &coverageMatrix{Matrix: raw, read: coverageFaultAt(0, 1, 0)}
			neighbors, err = am.Neighbors("a")
			AssertErrorIs(t, err, errCoverageBackend)
			if neighbors != nil {
				t.Fatal("failed Neighbors published partial data")
			}
			degrees, err = matrix.DegreeVector(am)
			AssertErrorIs(t, err, errCoverageBackend)
			if degrees != nil {
				t.Fatal("failed DegreeVector published partial data")
			}
			// A replaced rectangular backend is structurally invalid for adjacency queries.
			am.Mat = MustDense(t, 2, 2)
			_, err = am.VertexCount()
			AssertErrorIs(t, err, matrix.ErrDimensionMismatch)
			_, err = am.Neighbors("a")
			AssertErrorIs(t, err, matrix.ErrDimensionMismatch)
			am.Mat = MustDense(t, 3, 2)
			_, err = matrix.DegreeVector(am)
			AssertErrorIs(t, err, matrix.ErrNonSquare)
		})
	}
}

// TestCoverageAdjacencyPublicMetadataErrors rejects malformed caller-owned metadata.
// Tests exercise only public fields and constructors; no private reverse-index mutation is required.
// Implementation: Verify nil receivers, malformed public index maps and incompatible Mat replacement.
// Behavior highlights: Metadata validation fails before any vertex slice is returned.
// Inputs: Valid and malformed AdjacencyMatrix values with small owning Dense matrices.
// Returns: Test assertions only.
// Errors: ErrNilMatrix, ErrNonSquare and ErrDimensionMismatch.
// Determinism: Table traversal in declared order.
// Complexity: O(V) index validation and O(V^2) fixture storage.
// Notes: Invalid metadata is supplied only to the validation boundaries under test.
// AI-Hints: Do not manufacture private struct state to exercise defensive branches.
func TestCoverageAdjacencyPublicMetadataErrors(t *testing.T) {
	var absent *matrix.AdjacencyMatrix
	_, err := absent.VertexCount()
	AssertErrorIs(t, err, matrix.ErrNilMatrix)
	ids, err := absent.VertexIDs()
	AssertErrorIs(t, err, matrix.ErrNilMatrix)
	if ids != nil {
		t.Fatal("nil adjacency published IDs")
	}
	_, err = matrix.AdjacencyToGraph(absent)
	AssertErrorIs(t, err, matrix.ErrNilMatrix)
	_, err = matrix.DegreeVector(absent)
	AssertErrorIs(t, err, matrix.ErrNilMatrix)

	// Public metadata can be corrupted by the caller; validators must reject invalid sizes and indices.
	for _, indices := range []map[string]int{
		{"a": 0}, {"a": 0, "b": 2}, {"a": -1, "b": 1},
	} {
		am := &matrix.AdjacencyMatrix{Mat: MustDense(t, 2, 2), VertexIndex: indices}
		AssertErrorIs(t, matrix.ValidateGraphAdjacency(am), matrix.ErrDimensionMismatch)
		ids, err = am.VertexIDs()
		AssertErrorIs(t, err, matrix.ErrDimensionMismatch)
		if ids != nil {
			t.Fatal("invalid index map published IDs")
		}
	}
	am := &matrix.AdjacencyMatrix{Mat: MustDense(t, 1, 2)}
	AssertErrorIs(t, matrix.ValidateGraphAdjacency(am), matrix.ErrNonSquare)
	_, err = matrix.GraphFromAdjacency(am)
	AssertErrorIs(t, err, matrix.ErrDimensionMismatch)
}

// TestCoverageAdjacencyExportPolicies tests the directed and undirected export traversal.
// It checks preserved zero weights, strict thresholds, explicit loop policy and backend failure identity.
// Implementation: Create graph snapshots under each orientation, then export and inspect original IDs.
// Behavior highlights: Explicit zero threshold excludes zero; implicit threshold preserves finite zero
// in +Inf-absence mode. A forbidden self-loop introduced into public storage is rejected by core.
// Inputs: Directed/undirected snapshots of fixed weighted graphs.
// Returns: Test assertions only.
// Errors: ErrNilOption, ErrNaNInf, core.ErrLoopNotAllowed and the backend sentinel.
// Determinism: Graph insertion and export order are stable; count and endpoints are checked independently.
// Complexity: O(V^2+E) time and space for each fixture.
// Notes: Corruption tests replace public data only; no unexported fields are accessed.
// AI-Hints: +Inf is an absence sentinel, not a graph edge weight.
func TestCoverageAdjacencyExportPolicies(t *testing.T) {
	for _, directed := range []bool{true, false} {
		name := "undirected"
		orientation := matrix.WithUndirected()
		if directed {
			name = "directed"
			orientation = matrix.WithDirected()
		}
		t.Run(name, func(t *testing.T) {
			g, err := core.NewGraph(core.WithDirected(directed), core.WithWeighted())
			if err != nil {
				t.Fatal(err)
			}
			if _, err = g.AddEdge("a", "b", 0); err != nil {
				t.Fatal(err)
			}
			if _, err = g.AddEdge("b", "c", 2); err != nil {
				t.Fatal(err)
			}
			am, err := matrix.BuildAdjacency(g, matrixMustOptions(t, orientation, matrix.WithWeighted(), matrix.WithAllowMulti()))
			if err != nil {
				t.Fatal(err)
			}
			// Default export retains finite zero in the +Inf absence representation.
			exported, err := matrix.GraphFromAdjacency(am)
			if err != nil || exported == nil || len(exported.Edges()) != 2 {
				t.Fatalf("default export err=%v graph=%v", err, exported)
			}
			foundZero := false
			for _, edge := range exported.Edges() {
				if edge.Weight == 0 {
					foundZero = true
				}
			}
			if !foundZero {
				t.Fatal("finite zero edge lost in round trip")
			}
			// Explicit threshold is strict, and binary export assigns one to retained edges.
			exported, err = matrix.AdjacencyToGraph(am, matrix.WithEdgeThreshold(0), matrix.WithBinaryWeights())
			if err != nil || exported == nil || len(exported.Edges()) != 1 || exported.Edges()[0].Weight != 1 {
				t.Fatalf("threshold/binary export err=%v graph=%v", err, exported)
			}
			_, err = am.ToGraph(nil)
			AssertErrorIs(t, err, matrix.ErrNilOption)
			original := am.Mat
			am.Mat = &coverageMatrix{Matrix: original, read: coverageFaultAt(0, 1, 0)}
			exported, err = am.ToGraph()
			AssertErrorIs(t, err, errCoverageBackend)
			if exported != nil {
				t.Fatal("partial graph escaped backend failure")
			}
			// Both export orientations read the diagonal; dirty data is rejected before AddEdge.
			am.Mat = coverageRawDense(t, 3, 3, []float64{math.NaN(), 0, 0, 0, 0, 0, 0, 0, 0})
			_, err = am.ToGraph()
			AssertErrorIs(t, err, matrix.ErrNaNInf)
			// A user-written loop violates the construction policy even though Mat.Set itself is legal.
			am.Mat = original
			MustSet(t, am.Mat, 0, 0, 2)
			exported, err = am.ToGraph()
			AssertErrorIs(t, err, core.ErrLoopNotAllowed)
			if exported != nil {
				t.Fatal("invalid loop published graph")
			}
		})
	}
}

// TestCoverageAdjacencyAllowedLoopExport checks loop preservation rather than implicit loop insertion.
// The exported graph inherits the explicit source adjacency loop capability.
// Implementation: Build one valid weighted loop, export it and inspect the returned edge.
// Behavior highlights: A loop is emitted exactly once.
// Inputs: One vertex and one finite weighted loop.
// Returns: Test assertions only.
// Errors: All graph and matrix errors are checked.
// Determinism: One stable vertex and edge.
// Complexity: O(1) for this fixture.
// Notes: A positive weight exceeds the default classic export threshold.
// AI-Hints: Enable loops in both graph construction and adjacency policy.
func TestCoverageAdjacencyAllowedLoopExport(t *testing.T) {
	g, err := core.NewGraph(core.WithWeighted(), core.WithLoops())
	if err != nil {
		t.Fatal(err)
	}
	if _, err = g.AddEdge("a", "a", 4); err != nil {
		t.Fatal(err)
	}
	am, err := matrix.BuildAdjacency(g, matrixMustOptions(t, matrix.WithWeighted(), matrix.WithAllowLoops()))
	if err != nil {
		t.Fatal(err)
	}
	exported, err := am.ToGraph()
	if err != nil || exported == nil || len(exported.Edges()) != 1 {
		t.Fatalf("loop export graph=%v err=%v", exported, err)
	}
	edge := exported.Edges()[0]
	if edge.From != "a" || edge.To != "a" || edge.Weight != 4 {
		t.Fatalf("wrong loop: %+v", edge)
	}
}

// TestCoverageBuilderMalformedExternalInputs verifies errors for externally assembled vertex/edge lists.
// It covers low-level builders independently of core.Graph's stronger topology admission rules.
// Implementation: Pass malformed lists to both public builders and inspect the full result tuple.
// Behavior highlights: Invalid topology never publishes partial index maps or matrices.
// Inputs: Empty vertex domain with edges, duplicate IDs, nil edge, or unknown source.
// Returns: Test assertions only.
// Errors: ErrInvalidDimensions, ErrBadShape and ErrUnknownVertex, with the documented builder distinction.
// Determinism: Fixed vertex and edge order.
// Complexity: O(V+E) validation, bounded small allocations.
// Notes: Duplicate adjacency IDs and duplicate incidence IDs use different existing sentinels.
// AI-Hints: Do not bypass graph APIs to corrupt a core.Graph merely to reach builder guards.
func TestCoverageBuilderMalformedExternalInputs(t *testing.T) {
	options := matrixMustOptions(t, matrix.WithDirected(), matrix.WithWeighted())
	edge := &core.Edge{From: "a", To: "b", Weight: 1}
	for _, tc := range []struct {
		name     string
		vertices []string
		edges    []*core.Edge
		adjErr   error
		incErr   error
	}{
		{"empty-domain", nil, []*core.Edge{edge}, matrix.ErrInvalidDimensions, matrix.ErrInvalidDimensions},
		{"duplicate-id", []string{"a", "a"}, nil, matrix.ErrUnknownVertex, matrix.ErrBadShape},
		{"nil-edge", []string{"a", "b"}, []*core.Edge{nil}, matrix.ErrBadShape, matrix.ErrBadShape},
		{"unknown-source", []string{"b"}, []*core.Edge{edge}, matrix.ErrUnknownVertex, matrix.ErrUnknownVertex},
	} {
		t.Run(tc.name, func(t *testing.T) {
			idx, adj, err := matrix.BuildDenseAdjacency(tc.vertices, tc.edges, options)
			AssertErrorIs(t, err, tc.adjErr)
			if idx != nil || adj != nil {
				t.Fatal("failed adjacency published data")
			}
			idx, effective, inc, err := matrix.BuildDenseIncidence(tc.vertices, tc.edges, options)
			AssertErrorIs(t, err, tc.incErr)
			if idx != nil || effective != nil || inc != nil {
				t.Fatal("failed incidence published data")
			}
		})
	}
	idx, effective, inc, err := matrix.BuildDenseIncidence([]string{""}, nil, options)
	AssertErrorIs(t, err, core.ErrEmptyVertexID)
	if idx != nil || effective != nil || inc != nil {
		t.Fatal("empty ID published incidence")
	}
}
