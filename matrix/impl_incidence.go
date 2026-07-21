// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix - incidence builders (dense) with strict invariants.
//
// Deliverables (per TA-MATRIX):
//  1. Error-first lightweight getters (no panics): VertexCount/EdgeCount validate receiver and shape
//     and return sentinel errors (ErrNilMatrix, ErrDimensionMismatch) instead of panicking.
//  2. Clarified signs: directed uses −1 at source and +1 at target; undirected uses +1/+1;
//     self-loop in directed sums (−1 + +1) in the *same row* ⇒ algebraic zero; the builder MUST
//     skip such zero columns; in undirected a loop contributes +2 in the single incident row
//     (both half-edges touch the same vertex).
//  3. AllowMulti=false ⇒ first-edge-wins policy (directed: ordered (u,v); undirected: unordered {min,max}).
//  4. Deterministic order: vertices follow provided order; edge columns follow stable core edge order.
//  5. Sentinel errors unified (ErrGraphNil, ErrUnknownVertex, ErrDimensionMismatch, ErrNilMatrix).
//
// AI-Hints:
//   - Use AllowMulti=false when you need a canonical incidence (no duplicate columns).
//   - Incidence ignores numeric weights by design; it captures topology only (sign/endpoint).
//   - For undirected graphs, a self-loop appears as +2 in the single row - this is conventional in
//     incidence algebra; downstream tools that expect strictly {−1,0,+1} should normalize if needed.
//   - Determinism is guaranteed if you pass a deterministic vertex order and core returns edges by ID.
//
// Notes:
//   - Incidence matrices are purely structural: numeric edge weights are ignored by design.
//     Options.Weighted may be present for API symmetry but does not affect entries (only -1/0/+1, and +2 for undirected loops).
//   - Directed self-loops algebraically sum (-1 + +1) in the same row to a zero column. We do not materialize zero columns:
//     the builder MUST skip such columns to keep the incidence basis minimal and deterministic.
package matrix

import (
	"fmt"

	"github.com/lvlath/go/core"
)

// --- Incidence marks (no magic numbers) -------------------------------------------------------------

// srcMark is placed at the source vertex row in a directed incidence column (outgoing end).
const srcMark = -1.0 // −1 at "from" for directed graphs

// dstMark is placed at the target vertex row in a directed incidence column (incoming end).
const dstMark = +1.0 // +1 at "to" for directed graphs

// undirectedMark is placed at each incident vertex row for undirected non-loop edges.
const undirectedMark = +1.0 // +1 / +1 for undirected (two distinct endpoints)

// loopUndirectedMark is placed at the incident vertex row for undirected self-loops.
// Rationale: both half-edges touch the same vertex ⇒ +1 + +1 = +2 in that row.
const loopUndirectedMark = 2.0

// --- Public wrapper type ---------------------------------------------------------------------------

// IncidenceMatrix wraps a Matrix as a graph incidence representation.
// VertexIndex maps VertexID → row index in Mat.
// Edges holds the ordered list of *core.Edge corresponding to columns.
// Mat holds −1/0/+1 (and +2 for undirected loops) entries indicating incidence.
// opts preserves original construction options for round-trip fidelity.
type IncidenceMatrix struct {
	Mat         Matrix         // underlying incidence matrix (rows=|V|, cols=|E_eff|)
	VertexIndex map[string]int // mapping of VertexID to row index
	Edges       []*core.Edge   // ordered edges aligned to columns [0..cols)
	opts        Options        // original build options snapshot
}

// --- Constructor (public) --------------------------------------------------------------------------

// NewIncidenceMatrix CONSTRUCT a dense incidence matrix wrapper from core.Graph.
// Implementation:
//   - Stage 1: validate graph is non-nil (ErrGraphNil).
//   - Stage 2: extract stable vertex/edge lists from core contracts.
//   - Stage 3: delegate to BuildDenseIncidence (policy-aware, deterministic).
//   - Stage 4: wrap resulting Matrix with VertexIndex and Edges aligned to columns.
//
// Behavior highlights:
//   - No panics for user errors; only sentinel errors with context.
//   - Edge weights are ignored by design; the matrix encodes topology only.
//   - Empty graphs are supported:
//     If g has 0 vertices (and therefore 0 edges), the result is a valid 0×0 matrix
//     with empty VertexIndex and empty Edges. This is a legal degenerate case.
//
// Inputs:
//   - g: source graph (must be non-nil).
//   - opts: incidence build options (directed/allowMulti/allowLoops).
//
// Returns:
//   - *IncidenceMatrix: wrapper with Mat, VertexIndex, Edges, opts snapshot.
//
// Errors:
//   - ErrGraphNil, plus any BuildDenseIncidence sentinel wrapped with context.
//
// Determinism:
//   - Stable vertex order (per core) and column order (per core edge order).
//
// Complexity:
//   - Time O(|V|+|E|), Space O(|V|+|E|) for index and dense storage.
//
// Notes:
//   - For canonical layouts in tests, prefer lexicographic vertex IDs upstream.
//
// AI-Hints:
//   - Use AllowMulti=false to collapse parallel edges into a single column (first-edge-wins).
//   - For golden tests, prefer lexicographically sorted vertex orders for reproducibility.
func NewIncidenceMatrix(g *core.Graph, opts Options) (*IncidenceMatrix, error) {
	// Validate input graph (public sentinel for nil graph).
	if g == nil {
		return nil, fmt.Errorf("NewIncidenceMatrix: %w", ErrGraphNil)
	}

	// Pull vertices in the order defined by core; callers may already sort lexicographically.
	vertices := g.Vertices() // O(|V|); assumed stable per core contract

	// Pull edges in stable order (by Edge.ID asc per core); determinism depends on this.
	edges := g.Edges() // O(|E|)

	// Delegate to deterministic dense builder (validates inputs and options).
	idx, cols, mat, err := BuildDenseIncidence(vertices, edges, opts)
	if err != nil {
		return nil, fmt.Errorf("NewIncidenceMatrix: %w", err)
	}

	// Wrap high-level struct and return (Mat is already dense and bounds-checked).
	return &IncidenceMatrix{
		Mat:         mat,  // Matrix implementation (dense) returned by builder
		VertexIndex: idx,  // stable vertex→row mapping
		Edges:       cols, // column-aligned edges (post de-duplication if any)
		opts:        opts, // snapshot options for export fidelity
	}, nil
}

// --- Lightweight accessors with error-first invariants ---------------------------------------------

// --- Internal invariant validation ---------------------------------------------------------------

// operation names (no magic strings) used in invariant error wrapping.
const (
	opIncidenceVertexCount   = "IncidenceMatrix.VertexCount"
	opIncidenceEdgeCount     = "IncidenceMatrix.EdgeCount"
	opIncidenceVertexInc     = "IncidenceMatrix.VertexIncidence"
	opIncidenceEdgeEndpoints = "IncidenceMatrix.EdgeEndpoints"
)

// validateMeta CHECKS that the wrapper and its metadata are internally consistent.
// Implementation:
//   - Stage 1: validate receiver and underlying matrix are non-nil.
//   - Stage 2: validate Mat rows match len(VertexIndex).
//   - Stage 3: validate Mat cols match len(Edges).
//
// Behavior highlights:
//   - Prevents panics in getters that index metadata slices/maps.
//
// Inputs:
//   - op: operation name constant for error context (stable, no magic strings).
//
// Returns:
//   - rows, cols: matrix dimensions when valid.
//   - err: ErrNilMatrix or ErrDimensionMismatch wrapped with context.
//
// Errors:
//   - ErrNilMatrix when receiver or Mat is nil.
//   - ErrDimensionMismatch when metadata diverges from Mat shape.
//
// Determinism:
//   - Pure checks; no state mutation.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This is intentionally strict: metadata and Mat must always agree.
//
// AI-Hints:
//   - Call once per public method to guarantee "no panics" contract under misuse.
func (im *IncidenceMatrix) validateMeta(op string) (rows, cols int, err error) {
	// Guard nil receiver / nil matrix first (public contract: no panics).
	if im == nil || im.Mat == nil {
		return 0, 0, fmt.Errorf("%s: nil receiver or underlying Mat: %w", op, ErrNilMatrix)
	}

	// Read dimensions once to keep messages stable and avoid repeated calls.
	rows = im.Mat.Rows() // number of vertex rows
	cols = im.Mat.Cols() // number of edge columns

	// Vertex metadata must match the number of rows.
	if rows != len(im.VertexIndex) {
		return 0, 0, fmt.Errorf("%s: rows=%d vertexIndex=%d: %w",
			op, rows, len(im.VertexIndex), ErrDimensionMismatch)
	}

	// Edge metadata must match the number of columns.
	if cols != len(im.Edges) {
		return 0, 0, fmt.Errorf("%s: cols=%d edges=%d: %w",
			op, cols, len(im.Edges), ErrDimensionMismatch)
	}

	// Success: metadata and matrix are aligned.
	return rows, cols, nil
}

// VertexCount RETURN the number of vertices (matrix dimension) with invariant checks, no panics.
// Implementation:
//   - Stage 1: validate receiver and underlying Mat presence.
//   - Stage 2: ensure matrix dimension equals index table length.
//
// Behavior highlights:
//   - No panics: developer-misuse is reported as sentinel errors.
//
// Inputs:
//   - (receiver) *IncidenceMatrix: container with Mat and index tables.
//
// Returns:
//   - (int, error): vertex count or error.
//
// Errors:
//   - ErrNilMatrix (nil receiver or underlying Mat),
//   - ErrDimensionMismatch (Mat.Rows() != len(VertexIndex)).
//
// Determinism:
//   - Stable, pure read-only check.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Prefer using this method in user-facing surfaces; do not assume invariants silently.
//
// AI-Hints:
//   - If you need a panic-on-bug assertion in internal code, assert the error upstream once.
func (im *IncidenceMatrix) VertexCount() (int, error) {
	rows, _, err := im.validateMeta(opIncidenceVertexCount) // strict invariant check
	if err != nil {
		return 0, err
	}

	return rows, nil
}

// EdgeCount RETURN the number of edges (column count) with invariant checks, no panics.
// Implementation:
//   - Stage 1: validate receiver and underlying Mat presence.
//   - Stage 2: ensure matrix dimension equals edge columns length.
//
// Behavior highlights:
//   - No panics: developer-misuse is reported as sentinel errors.
//
// Inputs:
//   - (receiver) *IncidenceMatrix: container with Mat and column-aligned Edges.
//
// Returns:
//   - (int, error): edge/column count or error.
//
// Errors:
//   - ErrNilMatrix (nil receiver or underlying Mat),
//   - ErrDimensionMismatch (Mat.Cols() != len(Edges)).
//
// Determinism:
//   - Stable, pure read-only check.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Values in incidence are limited to {−1, 0, +1, +2} per semantics.
//
// AI-Hints:
//   - Counts are useful for quick capacity planning before row scans.
func (im *IncidenceMatrix) EdgeCount() (int, error) {
	_, cols, err := im.validateMeta(opIncidenceEdgeCount) // strict invariant check
	if err != nil {
		return 0, err
	}

	return cols, nil
}

// VertexIncidence COPY the signed incidence row for a given vertex into a new slice.
// Implementation:
//   - Stage 1: validate receiver/Mat (ErrNilMatrix) and resolve vertex row (ErrUnknownVertex).
//   - Stage 2: iterate columns j in [0..|E|) and copy Mat.At(row,j) into an output slice.
//
// Behavior highlights:
//   - Entries are {-1,0,+1} for non-loop edges; undirected self-loop yields +2 in that row.
//
// Inputs:
//   - vertexID: existing vertex identifier present in VertexIndex.
//
// Returns:
//   - []float64: a fresh slice of length |E| with signed incidence values for the vertex.
//
// Errors:
//   - ErrNilMatrix (nil receiver/Mat),
//   - ErrUnknownVertex (vertexID not found),
//   - wrapped Mat.At errors (e.g., out-of-range) with coordinates.
//
// Determinism:
//   - Fixed column order as in Edges; stable output for a fixed graph/options.
//
// Complexity:
//   - Time O(|E|), Space O(|E|) for the returned row.
//
// Notes:
//   - Weights of original graph are ignored; this is a structural view only.
//
// AI-Hints:
//   - Use on-demand when you need the per-vertex signed incidence pattern.
func (im *IncidenceMatrix) VertexIncidence(vertexID string) ([]float64, error) {
	// Validate wrapper invariants first to guarantee no panic / no out-of-sync reads.
	_, cols, err := im.validateMeta(opIncidenceVertexInc)
	if err != nil {
		return nil, err
	}
	// Resolve row index.
	row, ok := im.VertexIndex[vertexID]
	if !ok {
		return nil, fmt.Errorf("VertexIncidence: unknown vertex %q: %w", vertexID, ErrUnknownVertex)
	}
	// Allocate output row.
	out := make([]float64, cols)

	// Copy via safe At; bubble index errors.
	var val float64
	for j := 0; j < cols; j++ {
		val, err = im.Mat.At(row, j)
		if err != nil {
			return nil, fmt.Errorf("VertexIncidence: At(%d,%d): %w", row, j, err)
		}
		out[j] = val
	}

	return out, nil
}

// EdgeEndpoints RETURN (fromID,toID) for the edge aligned with column j.
// Implementation:
//   - Stage 1: validate receiver/Mat (ErrNilMatrix).
//   - Stage 2: bounds-check column j in [0..Cols) (ErrDimensionMismatch).
//   - Stage 3: return endpoints from Edges[j] as recorded by core.
//
// Behavior highlights:
//   - For undirected graphs we expose core’s stored ordering of endpoints.
//
// Inputs:
//   - j: zero-based column index.
//
// Returns:
//   - fromID, toID: endpoints as kept in core.Edge for this column.
//
// Errors:
//   - ErrNilMatrix, ErrDimensionMismatch on invalid j.
//
// Determinism:
//   - O(1) lookup; endpoints order matches the deterministic core edge order.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Useful for joining matrix columns back to edge metadata.
//
// AI-Hints:
//   - Validate j exactly once and reuse endpoints to avoid redundant map lookups.
func (im *IncidenceMatrix) EdgeEndpoints(j int) (fromID, toID string, err error) {
	// Validate wrapper invariants first to avoid panics on metadata indexing.
	_, cols, err := im.validateMeta(opIncidenceEdgeEndpoints)
	if err != nil {
		return "", "", err
	}

	// Bounds-check against the canonical column count (== len(im.Edges) after validateMeta).
	if j < 0 || j >= cols {
		return "", "", fmt.Errorf("EdgeEndpoints: column %d out of range [0,%d): %w",
			j, cols, ErrDimensionMismatch)
	}
	// Safe: j is within [0..len(im.Edges)).
	e := im.Edges[j] // column-aligned edge metadata

	return e.From, e.To, nil // endpoints as stored by core
}
