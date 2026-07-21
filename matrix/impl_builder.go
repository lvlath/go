// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix - canonical builders for Dense adjacency and incidence matrices.
// Deterministic, sentinel-accurate, and aligned with contracts.
//
// Purpose:
//   - Deterministic builders for dense adjacency and incidence matrices from core.Graph,
//     honoring Options (Directed/Weighted/AllowLoops/AllowMulti/MetricClosure).
//
// Policy & Contracts:
//   - Adjacency: 0/weight; metric-closure toggles to distances (+Inf as “no edge”, diag=0) then APSP.
//   - Incidence: directed (−1 on source, +1 on target; directed self-loop ⇒ skipped column),
//     undirected (+1/+1; self-loop ⇒ +2 in the single incident row).
//
// Determinism:
//   - First-edge-wins when AllowMulti=false (ordered or unordered key by directedness).
//   - Stable vertex order as provided by caller; no implicit sorting.
//
// AI-Hints:
//   - If you need lex order, pre-sort vertices in the caller.
//   - For sparse graphs, consider future sparse adapters; these are dense by design.
package matrix

import (
	"fmt"
	"math"

	"github.com/lvlath/go/core"
)

// adjacencyNoEdgeEncoding identifies the dense numeric sentinel used for absent edges.
//
// What:
//   - adjacencyNoEdgeZero means classic adjacency encoding: 0 is absence.
//   - adjacencyNoEdgeInf means finite-weight encoding: +Inf is absence and every
//     finite value, including 0 and negative values, is a real edge weight.
//
// Why:
//   - Weighted adjacency cannot use 0 for both “no edge” and “zero-weight edge”.
//   - +Inf is already the package-wide no-path/no-edge sentinel for distance-like
//     representations and is compatible with Dense.allowInfDistances.
//
// Determinism:
//   - The encoding is resolved once before allocation and never changes during the build.
//
// AI-Hints:
//   - Do not use NaN as an absence sentinel.
//   - Do not use -Inf as an absence sentinel.
//   - Do not infer edge presence from numeric truthiness in weighted +Inf encoding.
type adjacencyNoEdgeEncoding uint8

const (
	// adjacencyNoEdgeZero is the standard 0/weight adjacency encoding.
	//
	// Contract:
	//   - 0 means no edge.
	//   - Non-zero finite values may represent edge presence or edge weight.
	//   - A true zero-weight edge cannot be represented in this encoding.
	//
	// Use:
	//   - unweighted/binary adjacency;
	//   - weighted adjacency with no zero-weight edges;
	//   - compatibility all-zero auto-degrade mode.
	adjacencyNoEdgeZero adjacencyNoEdgeEncoding = iota

	// adjacencyNoEdgeInf is the finite-weight adjacency encoding.
	//
	// Contract:
	//   - +Inf means no edge.
	//   - Every finite value is an edge weight, including 0 and negative values.
	//   - Dense must be allocated with allowInfDistances=true.
	//
	// Use:
	//   - mixed zero/non-zero weighted input;
	//   - all-zero weighted input when WithPreserveZeroWeights is enabled;
	//   - weighted metric-closure preparation where zero edges must survive.
	adjacencyNoEdgeInf
)

// adjacencyBuildPolicy is the resolved local encoding contract for BuildDenseAdjacency.
//
// What:
//   - useWeight decides whether edge.Weight is written or binary defaultWeight is written.
//   - noEdge decides how absent edges are represented in the Dense matrix.
//
// Why:
//   - The builder must choose a single encoding before allocation so Dense policy,
//     initialization, edge writes, metric closure, Neighbors, ToGraph, and DegreeVector
//     all interpret the same matrix consistently.
//
// Behavior highlights:
//   - Resolved once before Dense allocation.
//   - Never recomputed inside edge loops.
//   - Does not mutate Options; it is a derived local policy.
//
// AI-Hints:
//   - Do not use opts.weighted directly inside write loops after this policy exists.
//   - Do not let downstream methods guess zero-edge support from raw values only.
type adjacencyBuildPolicy struct {
	// useWeight is true when raw edge.Weight values must be written.
	// false means topology is encoded with defaultWeight.
	useWeight bool

	// noEdge fixes the absent-edge sentinel for the entire matrix.
	// adjacencyNoEdgeZero uses 0 as absence.
	// adjacencyNoEdgeInf uses +Inf as absence.
	noEdge adjacencyNoEdgeEncoding
}

// defaultWeight - unit weight for unweighted adjacency/incidence writes.
const defaultWeight = 1.0

// unreachableWeight is the placeholder for "no edge" before metric-closure.
// We use 0 in adjacency (standard 0/weight adjacency). During metric-closure
// this turns into +Inf on off-diagonals, while diag is forced to 0.
const unreachableWeight = 0.0

// edgeWeightProfile summarizes edge-weight shape before adjacency allocation.
//
// What:
//   - hasEdges reports that at least one non-nil edge exists.
//   - hasZero reports that at least one edge has Weight == 0.
//   - hasNonZero reports that at least one edge has Weight != 0.
//
// Why:
//   - The builder needs to distinguish three cases:
//     1) no zero weights        => 0-as-no-edge is safe;
//     2) mixed zero/non-zero    => +Inf-as-no-edge is required;
//     3) all zero weights       => ambiguous, controlled by preserveZeroWeights.
//
// Notes:
//   - This profile does not classify NaN/Inf as valid or invalid.
//     BuildDenseAdjacency still validates actual writes and returns ErrInvalidWeight.
//
// AI-Hints:
//   - Do not treat hasZero alone as “effectively unweighted”;
//     mixed zero/non-zero must preserve zero edges.
type edgeWeightProfile struct {
	hasEdges   bool // at least one non-nil edge was inspected
	hasZero    bool // at least one inspected edge has Weight == 0
	hasNonZero bool // at least one inspected edge has Weight != 0
}

// inspectEdgeWeights scans edge weights once for adjacency encoding selection.
//
// Implementation:
//   - Stage 1: iterate the stable edge slice exactly once.
//   - Stage 2: reject nil edge slots as bad input shape.
//   - Stage 3: record zero/non-zero presence without validating numeric finiteness.
//
// Behavior highlights:
//   - Deterministic scan order.
//   - No allocation.
//   - Does not replace per-edge validation during writing.
//
// Inputs:
//   - edges: stable edge slice provided by core or explicit caller.
//
// Returns:
//   - edgeWeightProfile on success.
//   - ErrBadShape when an edge slot is nil.
//
// Complexity:
//   - Time O(E), Space O(1).
//
// AI-Hints:
//   - Keep NaN/Inf validation at the write site so ErrInvalidWeight can include endpoints.
//   - Do not silently skip nil edges; explicit edge slices are part of public input shape.
func inspectEdgeWeights(edges []*core.Edge) (edgeWeightProfile, error) {
	var p edgeWeightProfile

	for i := 0; i < len(edges); i++ {
		if edges[i] == nil {
			return p, fmt.Errorf("inspectEdgeWeights: nil edge at index %d: %w", i, ErrBadShape)
		}

		p.hasEdges = true
		if edges[i].Weight == 0 {
			p.hasZero = true
			continue
		}

		p.hasNonZero = true
	}

	return p, nil
}

// orderedPair builds (u,v) key for directed de-duplication.
// Complexity: O(1).
func orderedPair(u, v int) pairKey { return pairKey{u: u, v: v} }

// unorderedPair builds {min,max} key for undirected de-duplication.
// Complexity: O(1).
func unorderedPair(u, v int) pairKey {
	if u <= v {
		return pairKey{u: u, v: v}
	}

	return pairKey{u: v, v: u}
}

// lookupIndex resolves a vertex ID to row/col index or returns ErrUnknownVertex.
// Complexity: O(1) expected (hash map).
func lookupIndex(idx map[string]int, id string) (int, error) {
	if i, ok := idx[id]; ok {
		return i, nil
	}

	return 0, fmt.Errorf("matrix: unknown vertex %q: %w", id, ErrUnknownVertex)
}

// resolveAdjacencyBuildPolicy derives the exact dense adjacency encoding.
//
// Implementation:
//   - Stage 1: inspect edge-weight profile once.
//   - Stage 2: start from Options.weighted and classic 0-as-no-edge encoding.
//   - Stage 3: if unweighted, keep binary 0/1 adjacency.
//   - Stage 4: if weighted all-zero:
//     preserveZeroWeights=false => degrade to binary compatibility mode;
//     preserveZeroWeights=true  => use +Inf-as-no-edge and preserve finite 0.
//   - Stage 5: if weighted mixed zero/non-zero, use +Inf-as-no-edge.
//   - Stage 6: otherwise keep 0-as-no-edge weighted adjacency.
//
// Behavior highlights:
//   - Mixed zero/non-zero weighted input is always preserved.
//   - All-zero weighted input is controlled explicitly by Options.preserveZeroWeights.
//   - No hidden mutation of Options.
//
// Errors:
//   - ErrBadShape from nil edge slots.
//
// Determinism:
//   - Stable O(E) scan; same inputs/options produce the same policy.
//
// Complexity:
//   - Time O(E), Space O(1).
//
// AI-Hints:
//   - Do not silently binary-degrade mixed zero/non-zero input.
//   - Do not remove all-zero auto-degrade; it preserves core unweighted adapter compatibility.
//   - Use WithPreserveZeroWeights for exact all-zero weighted graphs.
func resolveAdjacencyBuildPolicy(opts Options, edges []*core.Edge) (adjacencyBuildPolicy, error) {
	p, err := inspectEdgeWeights(edges)
	if err != nil {
		return adjacencyBuildPolicy{}, err
	}

	policy := adjacencyBuildPolicy{
		useWeight: opts.weighted,
		noEdge:    adjacencyNoEdgeZero,
	}

	if !policy.useWeight {
		return policy, nil
	}

	// Preserve existing useful behavior: all-zero weighted-looking input is treated
	// as effectively unweighted unless a future explicit option says otherwise.
	if p.hasZero && !p.hasNonZero {
		if !opts.preserveZeroWeights {
			policy.useWeight = false
			return policy, nil
		}

		policy.noEdge = adjacencyNoEdgeInf
		return policy, nil
	}

	// Mixed zero/non-zero weighted input requires +Inf no-edge encoding so that
	// finite zero remains a real edge weight.
	if p.hasZero && p.hasNonZero {
		policy.noEdge = adjacencyNoEdgeInf
	}

	return policy, nil
}

// initAdjacencyNoEdgeInf initializes every cell as absent using +Inf.
//
// What:
//   - Pre-fills a Dense adjacency matrix with +Inf before edge writes.
//
// Why:
//   - In finite weighted adjacency, 0 can be a valid edge weight.
//   - Therefore absence must be represented by a value outside the finite weight domain.
//   - +Inf is the selected sentinel because Dense.validateValue already supports it
//     under allowInfDistances=true.
//
// Implementation:
//   - Stage 1: validate receiver.
//   - Stage 2: row-major Set(i,j,+Inf) for every cell.
//
// Behavior highlights:
//   - Uses Dense.Set, not direct data writes.
//   - Requires the matrix to have allowInfDistances=true.
//   - Does not force diagonal to 0; ordinary adjacency diagonal absence is still absence.
//   - Metric closure normalizes the diagonal later before FloydWarshall.
//
// Errors:
//   - ErrNilMatrix for nil receiver.
//   - ErrNaNInf if Dense was not allocated with allowInfDistances=true.
//   - ErrOutOfRange should not occur unless Dense invariants are corrupted.
//
// Determinism:
//   - Fixed row-major initialization.
//
// Complexity:
//   - Time O(V²), Space O(1).
//
// AI-Hints:
//   - Do not replace this with direct d.data writes.
//   - Do not use this for binary adjacency.
//   - Do not normalize diagonal here; that belongs to distance conversion.
func initAdjacencyNoEdgeInf(d *Dense) error {
	if d == nil {
		return ErrNilMatrix
	}

	var (
		i, j int
		err  error
	)
	for i = 0; i < d.Rows(); i++ {
		for j = 0; j < d.Cols(); j++ {
			if err = d.Set(i, j, math.Inf(1)); err != nil {
				return fmt.Errorf("initAdjacencyNoEdgeInf: Set(%d,%d,+Inf): %w", i, j, err)
			}
		}
	}

	return nil
}

// normalizeInfNoEdgeDistanceDiagonal prepares a +Inf-no-edge adjacency for APSP.
//
// What:
//   - Converts diagonal absence/positive self-loop entries into distance diagonal 0.
//   - Preserves finite negative self-loops as negative-cycle witnesses.
//
// Why:
//   - Floyd–Warshall consumes distance matrices, whose diagonal baseline is 0.
//   - In +Inf-no-edge adjacency, missing loops are stored as +Inf.
//   - Positive self-loops do not improve dist(i,i), so min(0,w_loop>=0) is 0.
//   - Negative self-loops prove a negative cycle and must be preserved for
//     FloydWarshall/ValidateDistanceMatrix to classify as ErrNegativeCycle.
//
// Implementation:
//   - Stage 1: validate receiver.
//   - Stage 2: scan diagonal in increasing index order.
//   - Stage 3: reject NaN and -Inf.
//   - Stage 4: set +Inf, zero, and positive diagonal values to 0.
//   - Stage 5: keep finite negative diagonal values unchanged.
//
// Behavior highlights:
//   - Does not touch off-diagonal values.
//   - Assumes off-diagonal absence is already +Inf.
//   - Uses Dense.Set so numeric policy remains centralized.
//
// Errors:
//   - ErrNilMatrix for nil receiver.
//   - ErrNaNInf for NaN or -Inf diagonal values.
//   - Dense.Set errors wrapped with coordinates.
//
// Determinism:
//   - Fixed diagonal order.
//
// Complexity:
//   - Time O(V), Space O(1).
//
// AI-Hints:
//   - Do not call InitDistancesInPlace on +Inf-no-edge adjacency; it would erase
//     real zero-weight off-diagonal edges by converting 0 to +Inf.
//   - Do not preserve positive self-loop weights on the distance diagonal.
func normalizeInfNoEdgeDistanceDiagonal(d *Dense) error {
	if d == nil {
		return ErrNilMatrix
	}

	var v float64
	var err error

	for i := 0; i < d.Rows(); i++ {
		v, err = d.At(i, i)
		if err != nil {
			return fmt.Errorf("normalizeInfNoEdgeDistanceDiagonal: At(%d,%d): %w", i, i, err)
		}
		if isNaNOrNegInf(v) {
			return fmt.Errorf("normalizeInfNoEdgeDistanceDiagonal: invalid diagonal[%d,%d]=%v: %w", i, i, v, ErrNaNInf)
		}
		if math.IsInf(v, 1) || v >= 0 {
			if err = d.Set(i, i, 0); err != nil {
				return fmt.Errorf("normalizeInfNoEdgeDistanceDiagonal: Set(%d,%d,0): %w", i, i, err)
			}
		}
	}

	return nil
}

// BuildDenseAdjacency CONSTRUCTS a dense adjacency matrix from explicit vertices/edges
// with Options policy (directed/weighted/loops/multi, optional metric-closure).
// Implementation:
//   - Stage 1: validate vertex list and build VertexID→index map.
//   - Stage 2: allocate V×V dense and decide weight policy (degrade to binary if all-zero).
//   - Stage 3: populate entries deterministically; mirror when undirected (except loops).
//   - Stage 4: optional metric-closure (distances via Floyd–Warshall).
//
// Behavior highlights:
//   - First-edge-wins when AllowMulti=false (directed uses ordered (u,v), undirected uses unordered {min,max} keys).
//   - When AllowMulti=true, all parallel edges are materialized into the same adjacency cell
//     and the LAST edge in the stable input sequence wins. This provides deterministic behavior without
//     collapsing the input edge list itself.
//   - Loops are ignored when AllowLoops=false and written on the diagonal
//     when AllowLoops=true (1 in unweighted mode, raw weight in weighted mode).
//
// Inputs:
//   - vertices: canonical vertex order (stable; caller decides lex order if needed).
//   - edges: stable edge list (core contract: by Edge.ID asc).
//   - opts: Options defining Directed/Weighted/AllowLoops/AllowMulti/MetricClosure.
//
// Returns:
//   - vidx: VertexID→index map (row==col index).
//   - mat: V×V dense adjacency.
//   - err: ErrInvalidDimensions (edges present with 0 vertices), ErrUnknownVertex, ErrInvalidWeight, shape/set errors.
//
// Determinism:
//   - Fixed loops and write order; deterministic output for same inputs/options.
//
// Complexity:
//   - Time O(V^2 + E), Space O(V^2).
//
// Notes:
//   - Unweighted mode writes 1 for present edges; weighted mode uses edge weights.
//   - Empty graphs:
//     When vertices is empty and edges is also empty, this function returns a valid
//     0×0 Dense adjacency and an empty index map. This matches the Matrix contract
//     that allows zero-sized shapes and enables “empty graph” pipelines without
//     special-casing at call sites.
//   - IMPORTANT (0-weight edges):
//     Dense adjacency has two valid absence encodings:
//     1) standard 0-as-no-edge encoding;
//     2) +Inf-as-no-edge encoding for finite weighted adjacency.
//     When weighted input contains both zero and non-zero edge weights, the builder
//     automatically selects +Inf-as-no-edge so finite 0 remains a real edge weight.
//     When weighted input is all-zero, auto mode preserves historical behavior and
//     degrades to binary adjacency. Use WithPreserveZeroWeights to force exact
//     all-zero weighted adjacency.
//   - MetricClosure:
//     If +Inf-as-no-edge was already selected, metric closure preserves zero-weight
//     off-diagonal edges and only normalizes the diagonal before FloydWarshall.
//     If standard 0-as-no-edge was selected, InitDistancesInPlace converts
//     off-diagonal zeros to +Inf.
//
// AI-Hints:
//   - Use MetricClosure to turn adjacency into distances (+Inf as unreachable), diag forced to 0.
//   - Complex structural checks, edge case handling, and multi-dimensional matrix
//     memory mappings are bundled together to ensure all graph validation logic
//     happens in one place, avoiding state leakage across multiple sub-functions.
//
// nolint:gocyclo
func BuildDenseAdjacency(
	vertices []string,
	edges []*core.Edge,
	opts Options,
) (map[string]int, *Dense, error) {
	// --- Stage 1: Validate vertices and build index map ---

	// Empty graph handling:
	//   - vertices==0 and edges==0 ⇒ valid 0×0 adjacency (degenerate case).
	//   - vertices==0 and edges>0  ⇒ invalid input (edges cannot reference vertices).
	if len(vertices) == 0 {
		if len(edges) != 0 {
			return nil, nil, fmt.Errorf(
				"BuildDenseAdjacency: empty vertex set with %d edges: %w",
				len(edges), ErrInvalidDimensions,
			)
		}

		// Zero-graph law:
		// An empty graph maps to a valid 0×0 adjacency matrix. There are no entries
		// to initialize, but preserving numeric policy keeps downstream Clone/Induced
		// behavior coherent.
		mat, err := newDenseWithPolicy(
			0,
			0,
			opts.validateNaNInf,
			opts.allowInfDistances || opts.metricClose,
		)
		if err != nil {
			return nil, nil, fmt.Errorf("BuildDenseAdjacency: NewDense(0,0): %w", err)
		}
		// Return a non-nil empty map to make “empty index” explicit to callers.
		return make(map[string]int), mat, nil
	}

	V := len(vertices)

	// Build stable vertex→index mapping with linear scan in provided order.
	idx := make(map[string]int, V)
	var i int
	var id string
	for i, id = range vertices {
		// Defensive duplicate check.
		if _, dup := idx[id]; dup {
			return nil, nil, fmt.Errorf("BuildDenseAdjacency: duplicate vertex id %q: %w", id, ErrUnknownVertex)
		}
		idx[id] = i
	}

	// --- Stage 2: Resolve adjacency encoding and allocate Dense ---
	//
	// Encoding decision is made before allocation because Dense must know whether
	// +Inf writes are legal. This is the only place where weighted zero-edge
	// ambiguity is resolved.
	//
	// Cases:
	//   - unweighted/binary adjacency:
	//       0 means no edge, present edges write defaultWeight.
	//   - weighted input without zero weights:
	//       0 can still mean no edge safely, raw weights are written.
	//   - weighted mixed zero/non-zero input:
	//       +Inf means no edge, every finite value including 0 is an edge weight.
	//   - weighted all-zero input:
	//       auto mode degrades to binary for compatibility;
	//       WithPreserveZeroWeights uses +Inf no-edge encoding.
	var (
		mat         *Dense
		err         error
		buildPolicy adjacencyBuildPolicy
	)

	buildPolicy, err = resolveAdjacencyBuildPolicy(opts, edges)
	if err != nil {
		return nil, nil, fmt.Errorf("BuildDenseAdjacency: %w", err)
	}

	useWeight := buildPolicy.useWeight

	// Dense must allow +Inf when either:
	//   - metric closure needs +Inf as no-path;
	//   - weighted adjacency uses +Inf as no-edge to preserve finite zero weights.
	allowInfDistances := opts.metricClose || buildPolicy.noEdge == adjacencyNoEdgeInf

	if allowInfDistances {
		mat, err = newDenseWithPolicy(V, V, true, true)
	} else {
		// Regular adjacency: keep strict NaN/Inf validation.
		mat, err = NewDense(V, V)
	}

	if err != nil {
		return nil, nil, fmt.Errorf("BuildDenseAdjacency: NewDense(%d,%d): %w", V, V, err)
	}

	// For +Inf-no-edge adjacency, absence must be installed before edge writes.
	// This is required even when MetricClosure=true; otherwise zero-weight edges
	// and absent edges would both start as 0 and become indistinguishable.
	if buildPolicy.noEdge == adjacencyNoEdgeInf {
		if err = initAdjacencyNoEdgeInf(mat); err != nil {
			return nil, nil, fmt.Errorf("BuildDenseAdjacency: %w", err)
		}
	}

	// --- Stage 3: Populate adjacency entries (deterministic) ---
	directed := opts.directed
	allowMulti := opts.allowMulti
	allowLoops := opts.allowLoops

	// First-edge-wins set when AllowMulti=false.
	// For directed graphs key=(src,dst). For undirected, we normalize to {min,max}.
	seen := make(map[pairKey]struct{}, 64)

	var (
		e        *core.Edge
		ej       int
		src, dst int
		w        float64
		key      pairKey
	)

	for ej = 0; ej < len(edges); ej++ {
		e = edges[ej]
		// Resolve endpoints
		if src, err = lookupIndex(idx, e.From); err != nil {
			return nil, nil, fmt.Errorf("BuildDenseAdjacency: %w", err)
		}
		if dst, err = lookupIndex(idx, e.To); err != nil {
			return nil, nil, fmt.Errorf("BuildDenseAdjacency: %w", err)
		}
		// Loops policy
		if src == dst && !allowLoops {
			continue
		}
		// Multi-edge policy
		if !allowMulti {
			if directed {
				key = orderedPair(src, dst)
			} else {
				key = unorderedPair(src, dst)
			}
			if _, dup := seen[key]; dup {
				// Skip duplicate unordered pair, keep first.
				continue
			}
			// Mark unordered pair.
			seen[key] = struct{}{}
		}

		// Decide adjacency cell value for this edge:
		//   - in weighted mode we preserve the raw float64 edge weight,
		//     rejecting NaN and ±Inf via ErrInvalidWeight;
		//   - otherwise we write defaultWeight (binary adjacency).
		// NOTE:
		//   - zero-weight edges are preserved when buildPolicy selected +Inf no-edge encoding.
		//   - in auto all-zero mode, useWeight=false intentionally writes binary defaultWeight.
		//   - NaN/±Inf edge weights are rejected before Dense.Set so callers receive ErrInvalidWeight.
		if useWeight {
			w = float64(e.Weight)
			// Reject NaN/±Inf *before* writing into the matrix; this keeps all
			// Dense instances free from non-finite values, except for APSP
			// matrices that explicitly allow +Inf as "no path".
			if isNonFinite(w) {
				return nil, nil, fmt.Errorf("BuildDenseAdjacency: invalid weight for %q->%q: %w",
					e.From, e.To, ErrInvalidWeight)
			}
		} else {
			w = defaultWeight
		}

		// Write adjacency cell [src,dst]
		if err = mat.Set(src, dst, w); err != nil {
			return nil, nil, fmt.Errorf("BuildDenseAdjacency: Set(%d,%d): %w", src, dst, err)
		}
		// Mirror for undirected (not for loops)
		if !directed && src != dst {
			if err = mat.Set(dst, src, w); err != nil {
				return nil, nil, fmt.Errorf("BuildDenseAdjacency: Set(%d,%d): %w", dst, src, err)
			}
		}
	}

	// --- Stage 4: Optional metric closure (APSP) ---
	if opts.metricClose {
		// Convert adjacency encoding into distance-matrix encoding:
		//
		//   - If adjacency already uses +Inf as no-edge, only the diagonal needs
		//     distance normalization. Off-diagonal zero-weight edges must remain 0.
		//
		//   - If adjacency uses 0 as no-edge, InitDistancesInPlace performs the
		//     classic conversion: off-diagonal 0 -> +Inf, diagonal -> 0 except
		//     finite negative self-loop witnesses.
		//
		// FloydWarshall then validates distance semantics, performs APSP relaxation,
		// and detects negative cycles through the canonical public contract.
		if buildPolicy.noEdge == adjacencyNoEdgeInf {
			if err = normalizeInfNoEdgeDistanceDiagonal(mat); err != nil {
				return nil, nil, fmt.Errorf("BuildDenseAdjacency: %w", err)
			}
		} else {
			if err = InitDistancesInPlace(mat); err != nil {
				return nil, nil, fmt.Errorf("BuildDenseAdjacency: %w", err)
			}
		}

		if err = FloydWarshall(mat); err != nil {
			return nil, nil, fmt.Errorf("BuildDenseAdjacency: metric closure: %w", err)
		}
	} else {
		// Pure adjacency:
		//   - In 0-as-no-edge encoding, a clean no-loop diagonal is 0.
		//   - In +Inf-as-no-edge encoding, a missing loop must remain +Inf because
		//     finite 0 is a real loop edge.
		if !opts.allowLoops && buildPolicy.noEdge == adjacencyNoEdgeZero {
			for i = 0; i < V; i++ {
				if err = mat.Set(i, i, 0.0); err != nil {
					return nil, nil, fmt.Errorf("BuildDenseAdjacency: Set(%d,%d,0): %w", i, i, err)
				}
			}
		}
	}

	return idx, mat, nil
}

// BuildDenseIncidence CONSTRUCTS a dense incidence matrix from a vertex-id list and an edge list,
// applying Options policy deterministically.
// Implementation:
//   - Stage 1: validate vertex list and build VertexID→row index.
//   - Stage 2: compute effective column list (filter loops/multi deterministically).
//   - Stage 3: allocate V×E' dense (allow zero columns).
//   - Stage 4: populate columns with signed/undirected marks.
//
// Behavior highlights:
//   - Directed: −1 at source row, +1 at target row; directed self-loop ⇒ skipped column (not materialized).
//   - Undirected: +1 at both endpoints; undirected self-loop ⇒ +2 in the single row.
//   - DisallowMulti: first-edge-wins (ordered for directed; unordered for undirected).
//   - Columns preserve stable input order post filtering/dedup.
//
// Inputs:
//   - vertices: canonical vertex order (stable; caller decides lex order).
//   - edges: stable edge sequence (core contract: by Edge.ID asc).
//   - opts: Directed/AllowMulti/AllowLoops (Weighted is irrelevant for incidence).
//
// Returns:
//   - vidx: VertexID→row index.
//   - cols: effective column-aligned edges after filtering/dedup.
//   - mat: V×E' dense with entries in {−1,0,+1} (and +2 for undirected loops).
//   - err: ErrInvalidDimensions (edges present with 0 vertices), ErrUnknownVertex, dense Set/shape errors.
//
// Determinism:
//   - Stable rows/columns given stable inputs/options.
//
// Complexity:
//   - Time O(V + E), Space O(V + E) plus V×E' for dense data.
//
// Notes:
//   - Edge weights are intentionally ignored in incidence matrices; only the
//     signed incidence pattern matters (-1, 0, +1, and +2 for undirected loops).
//   - When AllowMulti=false, the first edge between a given pair of vertices
//     wins and subsequent parallel edges are ignored at the column level.
//   - When AllowMulti=true, each edge is assigned its own column.
//   - Empty graphs:
//     When vertices is empty and edges is also empty, this function returns a valid
//     0×0 Dense incidence and empty metadata slices/maps. This matches the Matrix
//     contract allowing zero-sized shapes and avoids forcing callers to special-case.
//
// AI-Hints:
//   - Use AllowMulti=false to get a canonical set of columns without duplicates.
func BuildDenseIncidence(
	vertices []string,
	edges []*core.Edge,
	opts Options,
) (map[string]int, []*core.Edge, *Dense, error) {
	// --- Stage 1: Validate and index ---

	// Empty graph handling:
	//   - vertices==0 and edges==0 ⇒ valid 0×0 incidence (degenerate case).
	//   - vertices==0 and edges>0  ⇒ invalid input (edges cannot be represented without vertices).
	if len(vertices) == 0 {
		if len(edges) != 0 {
			return nil, nil, nil, fmt.Errorf(
				"BuildDenseIncidence: empty vertex set with %d edges: %w",
				len(edges), ErrInvalidDimensions,
			)
		}
		// Allocate a degenerate 0×0 Dense; NewDense represents zero-area with nil backing slice.
		mat, err := NewDense(0, 0)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("BuildDenseIncidence: NewDense(0,0): %w", err)
		}
		// Return explicit empty metadata.
		return make(map[string]int), make([]*core.Edge, 0), mat, nil
	}

	V := len(vertices)
	// Build a stable vertex→row index map in the provided order; check duplicates defensively.
	idx := make(map[string]int, V)
	var i int
	var id string
	for i, id = range vertices {
		if id == "" {
			return nil, nil, nil, fmt.Errorf("BuildDenseIncidence: empty vertex id at %d: %w", i, core.ErrEmptyVertexID)
		}
		if _, dup := idx[id]; dup {
			return nil, nil, nil, fmt.Errorf("BuildDenseIncidence: duplicate vertex id %q: %w", id, ErrBadShape)
		}
		idx[id] = i
	}

	// --- Stage 2: Compute effective column list deterministically ---

	// Deduplicate when AllowMulti=false using a pairKey set; directed uses ordered (u,v),
	// undirected uses unordered {min,max}. Keep the *first* occurrence (stable scan order).
	directed := opts.directed
	allowMulti := opts.allowMulti
	allowLoops := opts.allowLoops

	eff := make([]*core.Edge, 0, len(edges))
	seen := make(map[pairKey]struct{}, 64)

	// Stable single pass over edges to construct the effective column list.
	var (
		e    *core.Edge
		ej   int
		u, v int
		key  pairKey
		ok   bool
	)
	for ej = 0; ej < len(edges); ej++ {
		e = edges[ej] // address is safe (backed by the slice)
		if e == nil {
			return nil, nil, nil, fmt.Errorf("BuildDenseIncidence: nil edge at index %d: %w", ej, ErrBadShape)
		}
		// Resolve endpoints to row indices; unknown vertex is a hard error.
		if u, ok = idx[e.From]; !ok {
			return nil, nil, nil, fmt.Errorf("BuildDenseIncidence: unknown source %q: %w", e.From, ErrUnknownVertex)
		}
		if v, ok = idx[e.To]; !ok {
			return nil, nil, nil, fmt.Errorf("BuildDenseIncidence: unknown target %q: %w", e.To, ErrUnknownVertex)
		}

		// Loops policy:
		// - If loops are disallowed: skip.
		// - If directed and u==v: skip zero column entirely (not materialized).
		if u == v {
			if !allowLoops {
				continue // policy: ignore self-loops when AllowLoops=false
			}
			if directed {
				continue // skip directed self-loop column
			}
		}

		// Multi-edge policy (first-edge-wins when disallowed)
		if !allowMulti {
			if directed {
				// Directed: ordered pair (u,v).
				key = orderedPair(u, v)
			} else {
				key = unorderedPair(u, v)
			}
			if _, dup := seen[key]; dup {
				continue // ignore duplicate; keep the first occurrence (stable)
			}
			seen[key] = struct{}{} // record this pair as seen
		}
		// Append the edge pointer to the column list; order preserved (stable).
		eff = append(eff, e)
	}

	// --- Stage 3: Allocate V×E' dense ---
	Ep := len(eff)
	var mat *Dense
	var err error
	// allow zero-column incidence
	if mat, err = NewDense(V, Ep); err != nil {
		return nil, nil, nil, fmt.Errorf("BuildDenseIncidence: NewDense(%d,%d): %w", V, Ep, err)
	}

	// --- Stage 4: Populate columns ---

	// Fill one column per effective edge with the correct signs per policy.
	var j, su, sv int
	for j = 0; j < Ep; j++ {
		e = eff[j]
		su = idx[e.From]
		sv = idx[e.To]

		if directed {
			// Directed incidence: su!=sv (directed loops already skipped).
			if err = mat.Set(su, j, srcMark); err != nil { // mark source with −1.
				return nil, nil, nil, fmt.Errorf("BuildDenseIncidence: Set(%d,%d,-1): %w", su, j, err)
			}
			if err = mat.Set(sv, j, dstMark); err != nil { // mark target with +1.
				return nil, nil, nil, fmt.Errorf("BuildDenseIncidence: Set(%d,%d,+1): %w", sv, j, err)
			}
			continue
		}

		// Undirected incidence:
		if su == sv {
			// Self-loop contributes +2 in the single incident row.
			if err = mat.Set(su, j, loopUndirectedMark); err != nil {
				return nil, nil, nil, fmt.Errorf("BuildDenseIncidence: Set(%d,%d,+2): %w", su, j, err)
			}
			continue
		}
		// Non-loop undirected: +1 at each endpoint.
		if err = mat.Set(su, j, undirectedMark); err != nil {
			return nil, nil, nil, fmt.Errorf("BuildDenseIncidence: Set(%d,%d,+1): %w", su, j, err)
		}
		if err = mat.Set(sv, j, undirectedMark); err != nil {
			return nil, nil, nil, fmt.Errorf("BuildDenseIncidence: Set(%d,%d,+1): %w", sv, j, err)
		}
	}

	return idx, eff, mat, nil
}
