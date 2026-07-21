// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix - adjacency builders (dense) and metric-closure transform.
//
// Deliverables:
//  1. Directed + AllowMulti=false → first-edge-wins (ordered key (u,v)).
//  2. Undirected mirroring without loops (u==v is not mirrored).
//  3. Weighted adjacency preserves zero-weight edges by switching to +Inf no-edge
//     encoding when mixed zero/non-zero weights make 0-as-no-edge ambiguous.
//     Effectively-unweighted all-zero input may still degrade to binary in auto mode.
//  4. MetricClosure (Floyd–Warshall): diag=0, unreachable=+Inf (off-diagonal).
//  5. Deterministic iteration & stable vertex/edge order (no map order reliance).
//
// AI-Hints:
//   - For directed graphs, duplicate (u,v) edges are ignored when AllowMulti=false;
//     the *first* occurrence wins. For undirected graphs, the first of unordered
//     pair {min(u), max(v)} wins. This guarantees deterministic results.
//   - When input graph is effectively unweighted (all weights are 0 or graph flags
//     indicate unweighted) and options request Weighted=true, we intentionally build
//     a binary adjacency (1) to avoid an all-zero matrix.
//   - MetricClosure turns adjacency into pairwise shortest-path distances. It is
//     *not* an adjacency anymore, and ToGraph must return ErrMatrixNotImplemented
//   - Weighted mixed zero/non-zero input cannot use 0 as “no edge”. The builder
//     uses +Inf as the absence sentinel so finite 0 remains a real edge weight.
//
// Warning:
//   - IMPORTANT (0-weight edges):
//     Standard adjacency uses 0 as “no edge”. That encoding cannot represent
//     a real zero-weight edge. When weighted input contains both zero and non-zero
//     weights, the builder switches to +Inf-as-no-edge encoding so every finite
//     value, including 0, remains a valid edge weight.
//     All-zero input remains treated as effectively unweighted in auto mode to
//     preserve existing graph-adapter behavior; use explicit weighted-preservation
//     options in a follow-up if your domain needs all-zero weighted graphs.
package matrix

import (
	"fmt"
	"math"

	"github.com/lvlath/go/core"
)

// defaultReserve is the initial capacity for neighbor slices
const defaultReserve = 8

// AdjacencyMatrix wraps a Matrix as a graph adjacency representation.
// VertexIndex maps VertexID → row/col in Mat.
// vertexByIndex provides reverse lookup from column index to VertexID.
// Mat holds edge weights (float64), with unreachableWeight for no edge.
// opts preserves original build options for round‐trip fidelity.
type AdjacencyMatrix struct {
	Mat           Matrix         // underlying adjacency matrix
	VertexIndex   map[string]int // mapping of VertexID to index
	vertexByIndex []string       // reverse lookup by index
	opts          Options        // original construction options
}

// adjacencyExportPolicy freezes ToGraph export decisions for a single matrix scan.
//
// What:
//   - Holds threshold, weight-output mode, and adjacency absence encoding.
//
// Why:
//   - returnEdge is called inside an O(n²) loop; passing one resolved policy object
//     avoids repeated option/encoding decisions and prevents boolean-argument soup.
//
// Behavior highlights:
//   - thresholdSet distinguishes default threshold from an explicit user filter.
//   - infNoEdge preserves finite zero-weight edges during default export.
//
// AI-Hints:
//   - Do not apply DefaultEdgeThreshold to +Inf-no-edge weighted adjacency unless
//     the caller explicitly set a threshold.
type adjacencyExportPolicy struct {
	threshold     float64
	thresholdSet  bool
	keepWeights   bool
	binaryWeights bool
	infNoEdge     bool
}

// NewAdjacencyMatrix BUILD adjacency container from core.Graph.
// Implementation:
//   - Stage 1: validate input graph (ErrGraphNil).
//   - Stage 2: materialize vertex/edge lists (stable order from core).
//   - Stage 3: delegate to BuildDenseAdjacency (deterministic).
//   - Stage 4: construct reverse index and return.
//
// Behavior highlights:
//   - No panics for user errors; strict sentinels only.
//   - Stored opts snapshot preserves round-trip/export policy.
//
// Inputs:
//   - g: source graph (non-nil).
//   - opts: effective options (build/export policy snapshot).
//
// Returns:
//   - *AdjacencyMatrix with Dense backend.
//
// Errors:
//   - ErrGraphNil; plus any BuildDenseAdjacency errors.
//
// Determinism:
//   - Stable vertex order (core contract) and stable edge iteration.
//
// Complexity:
//   - Time O(n + m) for extraction + builder; Space O(n + m).
//
// Notes:
//   - The actual dense builder lives elsewhere; this wrapper just orchestrates.
//
// AI-Hints:
//   - Prefer passing Options via NewMatrixOptions(...) to keep defaults in sync.
func NewAdjacencyMatrix(g *core.Graph, opts Options) (*AdjacencyMatrix, error) {
	// Validate input graph
	if g == nil {
		return nil, ErrGraphNil
	}

	// Prepare vertex and edge slices
	vertices := g.Vertices() // get ordered vertices
	edges := g.Edges()       // get edges

	// Delegate to low‐level builder
	var (
		idx map[string]int // index map
		mat *Dense         // dense matrix result
		err error          // error placeholder
	)
	idx, mat, err = BuildDenseAdjacency(vertices, edges, opts)
	if err != nil {
		return nil, err
	}

	// Finalize reverse index
	rev := make([]string, len(vertices))
	//copy(rev, vertices)
	for id, i := range idx {
		rev[i] = id
	}

	// Wrap and return
	return &AdjacencyMatrix{
		Mat:           mat,
		VertexIndex:   idx,
		vertexByIndex: rev,
		opts:          opts,
	}, nil
}

// VertexIDs RETURNS an ordered list of vertex identifiers with initial validation.
// Implementation:
//   - Stage 1: run full structural validation of the adjacency matrix.
//   - Stage 2: allocate a detached slice matched to the index table size.
//   - Stage 3: deep-copy the internal vertex index array into the result.
//
// Behavior highlights:
//   - No panics: all validation failures are returned as explicit errors.
//   - Memory isolation: returns a new slice slice memory to prevent external mutations.
//
// Inputs:
//   - (receiver) *AdjacencyMatrix: matrix data structure containing the index registry.
//
// Returns:
//   - ([]string, error): copy of vertex ID registry or validation error.
//
// Errors:
//   - Propagates any structural or mismatch errors surfaced by ValidateGraphAdjacency.
//
// Determinism:
//   - Stable, guarantees consistent slice ordering across identical graph states.
//
// Complexity:
//   - Time: O(N) where N is the number of vertices (due to slice copy).
//   - Space: O(N) auxiliary space allocated for the returned slice.
//
// Notes:
//   - Safe for external consumption.
//   - Expensive to call repeatedly in hot loops due to explicit O(N) allocation.
//
// AI-Hints:
//   - If the returned order must strictly represent the layout, preserve 'am.vertexByIndex' mapping order.
func (am *AdjacencyMatrix) VertexIDs() ([]string, error) {
	if err := ValidateGraphAdjacency(am); err != nil {
		return nil, err
	}

	out := make([]string, len(am.vertexByIndex))
	copy(out, am.vertexByIndex)
	return out, nil
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
//   - (receiver) *AdjacencyMatrix: container with Mat and index tables.
//
// Returns:
//   - (int, error): vertex count or error.
//
// Errors:
//   - ErrNilMatrix (nil receiver or underlying Mat),
//   - ErrDimensionMismatch (Mat.Rows() != len(vertexByIndex)).
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
func (am *AdjacencyMatrix) VertexCount() (int, error) {
	if am == nil || am.Mat == nil {
		return 0, fmt.Errorf("AdjacencyMatrix.VertexCount: nil receiver or underlying Mat: %w", ErrNilMatrix)
	}
	if am.Mat.Rows() != len(am.vertexByIndex) {
		return 0, fmt.Errorf(
			"AdjacencyMatrix.VertexCount: inconsistent dimensions %d vs %d: %w",
			am.Mat.Rows(), len(am.vertexByIndex), ErrDimensionMismatch,
		)
	}

	return am.Mat.Rows(), nil
}

// Neighbors LIST adjacent vertex IDs reachable from u (row scan of adjacency row).
// Implementation:
//   - Stage 1: validate receiver and matrix presence.
//   - Stage 2: resolve source index via VertexIndex[u].
//   - Stage 3: scan row i over columns j and collect non-zero, finite entries.
//
// Behavior highlights:
//   - Deterministic: fixed vertex order (vertexByIndex), no map iteration order.
//   - In 0-as-no-edge adjacency, skips 0 and non-finite entries.
//   - In +Inf-as-no-edge weighted adjacency, finite 0 is returned as a real neighbor.
//
// Inputs:
//   - u: vertex ID (string) present in VertexIndex.
//
// Returns:
//   - []string: list of neighbor vertex IDs in stable column order.
//
// Errors:
//   - ErrNilMatrix (nil receiver or Mat),
//   - ErrUnknownVertex (u not in VertexIndex),
//   - ErrDimensionMismatch (Mat.Cols() != len(vertexByIndex)),
//   - bubbled matrix read errors (e.g., ErrOutOfRange) wrapped with coordinates.
//
// Determinism:
//   - Fixed col loop [0..n).
//
// Complexity:
//   - Time O(n), Space O(k) for k neighbors.
//
// Notes:
//   - Standard adjacency uses unreachableWeight (0) as no-edge.
//   - Finite weighted adjacency may use +Inf as no-edge so zero-weight edges remain visible.
//   - NaN/-Inf are skipped defensively; builders should not produce them.
//
// AI-Hints:
//   - Use WithWeighted/WithBinary builders to control adjacency semantics before calling.
//   - For dense traversals prefer *Dense Mat to avoid interface overhead in hot paths.
func (am *AdjacencyMatrix) Neighbors(u string) ([]string, error) {
	// Validate receiver
	if am == nil || am.Mat == nil {
		return nil, fmt.Errorf("Neighbors: nil AdjacencyMatrix or Mat: %w", ErrNilMatrix)
	}

	// Validate index exists
	srcIdx, ok := am.VertexIndex[u]
	if !ok {
		return nil, fmt.Errorf("Neighbors: unknown vertex %q: %w", u, ErrUnknownVertex)
	}

	// Validate shape
	cols := am.Mat.Cols()
	if cols != len(am.vertexByIndex) {
		return nil, fmt.Errorf(
			"Neighbors: dimension mismatch, cols=%d vs index=%d: %w",
			cols, len(am.vertexByIndex), ErrDimensionMismatch,
		)
	}

	// Prepare neighbor list and additional vars
	var (
		colIdx    int     // column index
		w         float64 // weight placeholder
		neighbors = make([]string, 0, defaultReserve)
		err       error                         // error placeholder
		vid       string                        // vertex ID resolved from the stable column index
		infNoEdge = am.adjacencyUsesInfNoEdge() // true when finite 0 is a real edge weight
	)

	// Execute scan
	for colIdx = 0; colIdx < cols; colIdx++ {
		// Never return the source vertex as its own neighbor, even if a loop exists.
		if colIdx == srcIdx {
			continue
		}
		w, err = am.Mat.At(srcIdx, colIdx)
		if err != nil {
			return nil, fmt.Errorf("Neighbors: At(%d,%d): %w", srcIdx, colIdx, err)
		}

		// Classify absence under the matrix's resolved adjacency encoding.
		// In +Inf-no-edge mode, finite 0 is a real edge and must be returned.
		if adjacencyEntryAbsent(w, infNoEdge) {
			continue
		}
		if adjacencyEntryAbsent(w, infNoEdge) {
			continue
		}

		// map index → vertex
		vid = am.vertexByIndex[colIdx]
		neighbors = append(neighbors, vid)
	}

	// Finalize
	return neighbors, nil
}

// adjacencyUsesInfNoEdge reports whether this adjacency interprets +Inf as absence.
//
// What:
//   - Returns true for metric/distance matrices and finite weighted adjacency
//     matrices built with +Inf-as-no-edge encoding.
//
// Why:
//   - In +Inf-no-edge encoding, finite 0 is a real edge weight.
//   - Consumers such as Neighbors, ToGraph, and DegreeVector must not treat 0 as absence.
//
// Implementation:
//   - MetricClosure always uses +Inf as no-path.
//   - For Dense-backed weighted adjacency, allowInfDistances indicates that the
//     builder selected a +Inf-capable absence encoding.
//   - Non-Dense externally mutated wrappers fall back to conservative 0-as-no-edge
//     behavior because their storage policy cannot be inspected.
//
// Determinism:
//   - O(1), pure metadata check.
//
// AI-Hints:
//   - Do not infer zero-weight absence without checking this helper.
//   - Do not expose Dense.allowInfDistances publicly just for this check.
func (am *AdjacencyMatrix) adjacencyUsesInfNoEdge() bool {
	if am == nil || am.Mat == nil {
		return false
	}
	if am.opts.metricClose {
		return true
	}
	if d, ok := am.Mat.(*Dense); ok {
		return am.opts.weighted && d.allowInfDistances
	}

	return false
}

// adjacencyEntryAbsent classifies a matrix entry as absent under the selected encoding.
//
// What:
//   - In +Inf-no-edge mode, +Inf/NaN/-Inf are treated as non-neighbor entries.
//   - In 0-as-no-edge mode, 0 and non-finite values are treated as absent.
//
// Why:
//   - This centralizes the zero-weight preservation law for adjacency consumers.
//
// Notes:
//   - Neighbors/DegreeVector are read-oriented convenience methods and skip non-finite
//     values defensively.
//   - ToGraph uses stricter export logic and reports NaN/-Inf as errors.
//
// AI-Hints:
//   - Do not use v == 0 directly in adjacency consumers.
func adjacencyEntryAbsent(v float64, infNoEdge bool) bool {
	if infNoEdge {
		return isNonFinite(v)
	}

	return v == unreachableWeight || isNonFinite(v)
}

// ToGraph CONVERT the stored adjacency to core.Graph with threshold/weight policy.
// Implementation:
//   - Stage 1: validate receiver and square shape against index table.
//   - Stage 2: guard metric-closure case (unsupported export).
//   - Stage 3: gather export options (threshold, (keep|binary) weights).
//   - Stage 4: add vertices in stable order, then emit edges deterministically.
//
// Behavior highlights:
//   - In classic 0-as-no-edge adjacency, threshold is strict: a[i,j] > threshold.
//   - In +Inf-as-no-edge weighted adjacency, default export preserves every finite
//     edge including 0; explicit WithEdgeThreshold enables strict filtering.
//   - keepWeights casts a[i,j] to float64 (truncate toward zero); binary emits weight=1.
//   - Orientation is inherited from the original build options (am.opts.directed).
//   - IMPORTANT (0-weight entries):
//     If the source adjacency uses +Inf as no-edge, finite 0 is a real edge and
//     default export preserves it. If the source adjacency uses 0 as no-edge,
//     finite 0 is absence and cannot be exported as an edge.
//
// Inputs:
//   - optFns ...Option: optional export overrides (edge threshold, weight policy).
//
// Returns:
//   - *core.Graph: newly constructed graph; error on contract violations.
//
// Errors:
//   - ErrNilMatrix, ErrDimensionMismatch, ErrMatrixNotImplemented (metric-closure),
//   - bubbled matrix/core errors wrapped with precise context.
//
// Determinism:
//   - Fixed i→j traversal with stable vertex order; no map iteration order.
//
// Complexity:
//   - Time O(n^2 + m), Space O(n) for transient slices.
//
// Notes:
//   - Export does not generate parallel edges by itself; it respects orientation and loop policy.
//   - No hidden allocations beyond necessary slices for vertex IDs.
//
// AI-Hints:
//   - Set a low EdgeThreshold (e.g., 0 or 0.5) to export all non-zero edges reliably.
//   - Use Binary weights to get a clean unweighted graph for structural analytics.
//   - KeepWeights only makes sense if the adjacency was built as weighted.
//   - Export direction always mirrors the source adjacency’s Directed policy, ensuring
//     round-trip fidelity. Override via adapters before building if you need to flip.
func (am *AdjacencyMatrix) ToGraph(optFns ...Option) (*core.Graph, error) {
	// Validate receiver: both the wrapper and the underlying matrix must be non-nil.
	if am == nil || am.Mat == nil {
		return nil, fmt.Errorf("ToGraph: %w", ErrNilMatrix) // unified sentinel for nil receiver
	}

	// Validate shape consistency: square matrix and index table aligned.
	n := am.Mat.Rows()                                    // number of rows
	if n != am.Mat.Cols() || n != len(am.vertexByIndex) { // square + index length
		return nil, fmt.Errorf("ToGraph: rows=%d cols=%d idx=%d: %w",
			am.Mat.Rows(), am.Mat.Cols(), len(am.vertexByIndex), ErrDimensionMismatch)
	}

	// Guard Metric-Closure: distance matrices are not exportable as simple edges.
	// NOTE: opts field is part of AdjacencyMatrix; metricClose is set by builders.
	if am.opts.metricClose { // single, explicit flag - no reflective tricks
		return nil, fmt.Errorf("ToGraph: metric-closure adjacency cannot be converted: %w", ErrMatrixNotImplemented)
	}

	// Gather export options (threshold/weights). Direction comes from source options.
	exp, err := gatherOptions(optFns...) // apply user overrides on documented defaults
	if err != nil {
		return nil, err
	}
	exportPolicy := adjacencyExportPolicy{
		threshold:     exp.edgeThreshold, // a[i,j] must be strictly greater to emit an edge
		thresholdSet:  exp.edgeThresholdSet,
		keepWeights:   exp.keepWeights, // true ⇒ weight=a[i,j]; false ⇒ weight=1
		binaryWeights: exp.binaryWeights,
		infNoEdge:     am.adjacencyUsesInfNoEdge(),
	}

	directed := am.opts.directed     // inherit orientation of the built adjacency
	allowLoops := am.opts.allowLoops // snapshot loop policy for core construction
	allowMulti := am.opts.allowMulti // snapshot multi-edge policy for core construction

	// Prepare the target graph with deterministic, policy-accurate flags.
	gOpts := make([]core.GraphOption, 0, 4) // preallocate small, fixed set
	// Direction: undirected export ⇒ core.WithDirected(false); else true.
	gOpts = append(gOpts, core.WithDirected(directed)) // pass through directedness as is
	// Core forbids non-zero weights in unweighted graphs.
	// Therefore, any export mode that emits a non-zero weight must build a weighted graph.
	if exportPolicy.keepWeights /* && weightedSrc */ || exportPolicy.binaryWeights { // only mark weighted if it matters
		gOpts = append(gOpts, core.WithWeighted())
	}
	// Loops / multi-edges: preserve build-time policy snapshot where sensible.
	// (While export won’t generate duplicates itself, we keep flags for fidelity.)
	if allowLoops {
		gOpts = append(gOpts, core.WithLoops())
	}
	if allowMulti {
		gOpts = append(gOpts, core.WithMultiEdges())
	}
	g, err := core.NewGraph(gOpts...) // construction is O(1); core owns its internals
	if err != nil {
		return nil, fmt.Errorf("ToGraph: NewGraph: %w", err)
	}
	// Vertex IDs are already in deterministic order within am.vertexByIndex.
	for _, vid := range am.vertexByIndex {
		// AddVertex is idempotent in core (by contract); ignore returned id if any.
		if err = g.AddVertex(vid); err != nil {
			// Surface core error verbatim; callers will handle via errors.Is for core sentinels.
			return nil, fmt.Errorf("ToGraph: AddVertex %q: %w", vid, err)
		}
	}

	// Deterministic nested loops over matrix entries with a single write site.
	// Directed: all ordered pairs (i,j). Undirected: upper triangle i..n-1 (incl. diag).
	var i, j int
	var fromID, toID string
	var val float64
	if directed {
		for i = 0; i < n; i++ { // iterate rows
			fromID = am.vertexByIndex[i] // resolve source id once per row
			for j = 0; j < n; j++ {      // iterate columns
				toID = am.vertexByIndex[j] // resolve target id
				val, err = am.Mat.At(i, j) // O(1) bounds-checked read
				if err != nil {
					return nil, fmt.Errorf("ToGraph: At(%d,%d): %w", i, j, err) // surface matrix read error
				}
				if err = returnEdge(g, fromID, toID, val, exportPolicy); err != nil {
					return nil, fmt.Errorf("ToGraph: entry[%d,%d] %q->%q: %w", i, j, fromID, toID, err)
				}
			}
		}
	} else {
		for i = 0; i < n; i++ { // upper triangle only to avoid duplicates
			fromID = am.vertexByIndex[i] // source id for this row
			for j = i; j < n; j++ {      // j starts at i ⇒ (i,i) loop once, (i,j) once
				toID = am.vertexByIndex[j] // target id
				val, err = am.Mat.At(i, j) // read once; no mirror read
				if err != nil {
					return nil, fmt.Errorf("ToGraph: At(%d,%d): %w", i, j, err)
				}
				if err = returnEdge(g, fromID, toID, val, exportPolicy); err != nil {
					return nil, fmt.Errorf("ToGraph: entry[%d,%d] %q->%q: %w", i, j, fromID, toID, err)
				}
			}
		}
	}

	// Successful, deterministic export complete.
	return g, nil
}

// returnEdge emits one graph edge when the matrix entry represents presence.
//
// Implementation:
//   - Stage 1: classify invalid non-finite values.
//   - Stage 2: classify absence according to adjacency encoding.
//   - Stage 3: apply threshold filtering:
//     classic 0-as-no-edge always uses threshold;
//     +Inf-as-no-edge uses threshold only when caller explicitly set it.
//   - Stage 4: derive exported weight and call core.AddEdge.
//
// Behavior highlights:
//   - Preserves finite zero-weight edges in +Inf-no-edge weighted adjacency.
//   - Rejects NaN and -Inf instead of silently dropping corrupted entries.
//   - Keeps default binary export behavior for classic adjacency.
//
// Inputs:
//   - g: target graph.
//   - fromID, toID: endpoints.
//   - aij: matrix entry.
//   - policy: frozen export policy.
//
// Returns:
//   - nil on skip or successful AddEdge.
//   - sentinel-preserving error on invalid data or core insertion failure.
//
// Errors:
//   - ErrNaNInf for NaN/-Inf matrix entries.
//   - core AddEdge errors wrapped with endpoint context.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Do not use aij > threshold as the only presence test in +Inf-no-edge mode.
//   - Do not silently ignore NaN/-Inf during graph export.
func returnEdge(g *core.Graph, fromID, toID string, aij float64, policy adjacencyExportPolicy) error {
	if math.IsNaN(aij) || math.IsInf(aij, -1) {
		return fmt.Errorf("ToGraph: invalid entry %q->%q=%v: %w", fromID, toID, aij, ErrNaNInf)
	}

	if policy.infNoEdge {
		if math.IsInf(aij, 1) {
			return nil
		}
		if policy.thresholdSet && !(aij > policy.threshold) {
			return nil
		}
	} else {
		if math.IsInf(aij, 1) || !(aij > policy.threshold) {
			return nil
		}
	}

	var w float64
	if policy.keepWeights {
		w = aij
	} else if policy.binaryWeights {
		w = defaultWeight
	} else {
		w = 0
	}

	if _, err := g.AddEdge(fromID, toID, w); err != nil {
		return fmt.Errorf("ToGraph: AddEdge %q->%q: %w", fromID, toID, err)
	}

	return nil
}

// DegreeVector COMPUTE per-vertex degree/strength from adjacency semantics.
//
//	– Unweighted/binary: degree counts present off-diagonal edges as 1.
//	– Weighted: strength sums present off-diagonal finite weights.
//	– Loops: counted as exactly 1 if present, regardless of stored weight.
//
// Implementation:
//   - Stage 1: validate container and square shape.
//   - Stage 2: fast-path on *Dense with direct flat access; else fallback via At.
//
// Behavior highlights:
//   - Absence is encoding-aware: 0-as-no-edge or +Inf-as-no-edge.
//   - In +Inf-as-no-edge weighted adjacency, finite 0 is present.
//   - NaN/-Inf are ignored defensively for this read-only summary.
//   - Deterministic i→j traversal.
//
// Returns:
//   - []float64 of length n.
//
// Errors:
//   - ErrNilMatrix, ErrNotSquare (via ValidateSquare), bubbled At errors.
//
// Determinism:
//   - Fixed loops; no map iteration.
//   - +Inf denotes “no edge” and must NOT contribute to sums.
//   - NaN is ignored (treated as no edge) for robustness.
//   - Loop order is fixed (i → j) for stable accumulation.
//
// Complexity:
//   - Time O(n^2), Space O(n).
//
// AI-Hints:
//   - For unweighted graphs, build a binary adjacency (1 for edges) to get pure degrees.
//   - For weighted directed graphs, this function returns row-sums (strength/out-degree).
//   - Prefer *Dense to avoid interface dispatch inside the double loop.
func (am *AdjacencyMatrix) DegreeVector() ([]float64, error) {
	// Validate container and matrix presence.
	if am == nil || am.Mat == nil {
		return nil, fmt.Errorf("DegreeVector: %w", ErrNilMatrix) // unified sentinel
	}
	// Validate square matrix (rows == cols).
	if err := ValidateSquare(am.Mat); err != nil {
		return nil, fmt.Errorf("DegreeVector: %w", err)
	}

	n := am.Mat.Rows()        // dimension of the matrix
	out := make([]float64, n) // allocate exactly one result vector
	infNoEdge := am.adjacencyUsesInfNoEdge()
	weighted := am.opts.weighted
	// Fast-path: direct flat access on *Dense (row-major).
	if d, ok := am.Mat.(*Dense); ok {
		var i, j, base int      // loop indices and row base offset
		var s, v float64        // accumulator and current value
		for i = 0; i < n; i++ { // fixed outer loop (row)
			s = 0                   // reset accumulator for row i
			base = i * n            // compute base offset once per row
			for j = 0; j < n; j++ { // fixed inner loop (col)
				v = d.data[base+j] // read A[i,j]

				// Ignore invalid/unreachable (policy).
				if adjacencyEntryAbsent(v, infNoEdge) {
					continue
				}
				if i == j {
					s += 1.0
					continue
				}
				if weighted {
					// Off-diagonal contributes raw positive weight.
					s += v
				} else {
					// Loop contributes exactly 1 if present.
					s += defaultWeight
				}
			}
			out[i] = s // store degree/strength of vertex i
		}

		return out, nil // return fast-path result
	}

	// Fallback: interface path via At (bounds-safe; deterministic).
	var i, j int            // loop indices
	var s, v float64        // accumulator and current value
	var err error           // bubbled error
	for i = 0; i < n; i++ { // iterate rows deterministically
		s = 0                   // reset accumulator
		for j = 0; j < n; j++ { // iterate cols
			v, err = am.Mat.At(i, j) // read A[i,j]
			if err != nil {
				return nil, fmt.Errorf("DegreeVector: At(%d,%d): %w", i, j, err)
			}

			// Ignore invalid/unreachable (policy).
			if adjacencyEntryAbsent(v, infNoEdge) {
				continue
			}
			if i == j {
				s += 1.0
				continue
			}
			if weighted {
				// Off-diagonal contributes raw positive weight.
				s += v
			} else {
				// Loop contributes exactly 1 if present.
				s += defaultWeight
			}
		}
		out[i] = s // assign row sum
	}

	return out, nil
}
