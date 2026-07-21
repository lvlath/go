// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tsp - Held–Karp 1-tree (Lagrangian) lower bound for symmetric TSP.
//
// This module computes an admissible lower bound on OPT via the classical
// Held–Karp relaxation:
//
//   - Choose vertex r as the "root". For a multiplier vector π ∈ ℝ^n define
//     reduced costs   c'_{ij} = c_{ij} + π_i + π_j  (symmetric case).
//   - Build a minimum 1-tree T(π): MST on V\{r} using c', plus two cheapest
//     r-incident edges (w.r.t. c').
//   - Bound value (Lagrangian dual):
//     L(π) = cost_c'(T(π)) − 2 * Σ_i π_i
//     where cost_c' sums reduced costs of 1-tree edges.
//     Since Σ_(i,j)∈T (π_i+π_j) = Σ_i deg_T(i)*π_i, this matches the dual form.
//   - Update π by subgradient with components s_i = deg_T(i) − 2
//     (tour feasibility requires deg(i)=2 for every i).
//
// L(π) is a valid *lower bound* on the optimal tour cost for every π, and
// is typically much tighter than degree-1 or MST bounds. We provide:
//   - A deterministic, allocation-conscious implementation.
//   - O(n²) Prim for the MST part (dense distance matrices).
//   - A small subgradient loop with a safe step policy and optional UB feedback.
//
// Scope & safety:
//   - Symmetric instances only (opts.Symmetric == true). For ATSP, a 1-tree
//     bound is not directly admissible; callers must gate usage upstream.
//   - +Inf is allowed (missing edges); if a 1-tree cannot be formed
//     (disconnected V\{r} or <2 finite r-incident edges), ErrIncompleteGraph.
//   - NaN/negative weights are rejected (strict sentinels).
//
// Complexity (per call):
//   - O(iters * n²) time for a fixed iteration budget.
//   - O(n²) memory for dense weights; O(n) working arrays.
//
// Determinism:
//   - No RNG. Prim and r-edge selection break ties by vertex index.
//   - The subgradient schedule is purely arithmetic and reproducible.
//
// Integration:
//   - Options.BoundAlgo==OneTreeBound lets Branch-and-Bound use this bound at the root.
//   - The bound is symmetric-only and must be gated before use on ATSP instances.
package tsp

import (
	"math"
	"time"

	"github.com/lvlath/go/matrix"
)

// OneTreeConfig controls the Held-Karp 1-tree subgradient loop and optional
// wall-clock budget. The defaults are deterministic and conservative enough for
// branch-and-bound pruning without making lower-bound computation dominate solving.
//
// Implementation:
//   - MaxIter bounds the number of subgradient iterations.
//   - StepScale controls the initial subgradient step size.
//   - MinStep stops the loop once movement becomes numerically irrelevant.
//   - TimeLimit optionally caps wall-clock work for defensive production use.
//
// Behavior highlights:
//   - Zero-value fields are normalized by DefaultOneTreeConfig.
//   - Deterministic when TimeLimit is zero.
//   - Increasing MaxIter usually tightens the bound at O(n^2) work per iteration.
//
// Notes:
//   - This config does not validate metricity; callers still own matrix validation.
//
// AI-Hints:
//   - Prefer deterministic MaxIter tuning before introducing time budgets.
type OneTreeConfig struct {
	// MaxIter is the maximum number of subgradient iterations (≥ 1).
	MaxIter int
	// Alpha ∈ (0, 2): step scale. 0.8–1.2 is common; we default to 0.9.
	Alpha float64
	// UB: optional incumbent (feasible tour) cost for adaptive steps.
	// If UB ≤ 0 or +Inf, the schedule ignores UB and uses a decreasing sequence.
	UB float64
	// TimeLimit: optional per-call wall-clock budget (0 disables checks).
	TimeLimit time.Duration
}

// DefaultOneTreeConfig returns conservative production-grade defaults for the
// Held-Karp 1-tree lower-bound loop. The values favor deterministic useful pruning
// over aggressive runtime spending.
//
// Implementation:
//   - Stage 1: Set a bounded iteration count.
//   - Stage 2: Set stable subgradient step parameters.
//   - Stage 3: Leave wall-clock budget disabled by default.
//
// Behavior highlights:
//   - Allocation-free.
//   - Deterministic across runs.
//   - Safe as a drop-in bound configuration.
//
// Inputs:
//   - None.
//
// Returns:
//   - OneTreeConfig: normalized default configuration.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure constant construction.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// AI-Hints:
//   - Do not make TimeLimit non-zero by default; it hurts reproducibility.
func DefaultOneTreeConfig() OneTreeConfig {
	return OneTreeConfig{
		MaxIter:   32,
		Alpha:     0.9,
		UB:        math.Inf(1), // no UB feedback by default
		TimeLimit: 0,
	}
}

// OneTreeLowerBound computes the Held-Karp 1-tree lower bound for a symmetric
// TSP instance using root as the distinguished vertex. Each iteration builds a
// minimum 1-tree on reduced costs c'(u,v)=c(u,v)+pi[u]+pi[v], then updates pi
// by the degree violation deg[v]-2.
//
// Implementation:
//   - Stage 1: Validate matrix shape, root, symmetry policy, and numeric weights.
//   - Stage 2: Initialize reusable oneTreeEngine buffers.
//   - Stage 3: Repeatedly build a reduced-cost 1-tree.
//   - Stage 4: Convert reduced total back to an original-cost lower bound.
//   - Stage 5: Update multipliers with a deterministic subgradient step.
//   - Stage 6: Return the best stabilized lower bound and final degree vector.
//
// Behavior highlights:
//   - Supports only symmetric instances.
//   - Uses deterministic Prim O(n^2) over V\{root}.
//   - Returns the final 1-tree degrees for diagnostics.
//   - Stabilizes the returned bound for cross-platform floating-point consistency.
//
// Inputs:
//   - dist: weighted matrix.
//   - root: distinguished vertex, usually opts.StartVertex.
//   - symmetric: caller-declared symmetry flag.
//   - cfg: subgradient configuration.
//
// Returns:
//   - lb: best lower bound found.
//   - degrees: degree vector of the final 1-tree.
//   - err: nil when a 1-tree can be formed.
//
// Errors:
//   - ErrATSPNotSupportedByAlgo for asymmetric use.
//   - ErrIncompleteGraph when V\{root} is disconnected or root has fewer than two finite edges.
//   - ErrDimensionMismatch, ErrInvalidVertex, ErrNaNWeight, ErrNegativeWeight,
//     or ErrAsymmetry from validation.
//
// Determinism:
//   - Fixed vertex scans, deterministic Prim tie-breaking, deterministic step schedule.
//
// Complexity:
//   - Time O(cfg.MaxIter*n^2), Space O(n^2).
//
// Notes:
//   - This is a lower bound, not a tour constructor.
//   - Metric triangle inequality is not required for the 1-tree calculation itself.
//
// AI-Hints:
//   - Do not use this for ATSP.
//   - Reuse buffers through oneTreeEngine; do not allocate per iteration.
func OneTreeLowerBound(
	dist matrix.Matrix,
	root int,
	symmetric bool,
	cfg OneTreeConfig,
) (lb float64, degrees []int, err error) {
	if !symmetric {
		return 0, nil, ErrATSPNotSupportedByAlgo
	}
	if cfg.MaxIter <= 0 {
		return 0, nil, ErrInvalidOptions
	}
	if math.IsNaN(cfg.Alpha) || math.IsInf(cfg.Alpha, 0) || cfg.Alpha <= 0 || cfg.Alpha >= 2 {
		return 0, nil, ErrInvalidOptions
	}
	if math.IsNaN(cfg.UB) || math.IsInf(cfg.UB, -1) {
		return 0, nil, ErrInvalidOptions
	}
	if cfg.TimeLimit < 0 {
		return 0, nil, ErrInvalidOptions
	}

	weights, err := copyClosureReadyWeights(dist, true)
	if err != nil {
		return 0, nil, err
	}
	n := weights.n
	if err = validateStartVertex(n, root); err != nil {
		return 0, nil, err
	}

	eng := oneTreeEngine{
		n:      n,
		root:   root,
		w:      weights.w,
		pi:     make([]float64, n),
		deg:    make([]int, n),
		inTree: make([]bool, n),
		parent: make([]int, n),
		key:    make([]float64, n),
	}

	// Sparse deadline checks (every 2048 iterations of the inner MST/scan loops).
	var useDeadline bool   // whether to enforce a per-call wall-clock budget
	var deadline time.Time // absolute time when the budget expires
	var tick uint64
	if cfg.TimeLimit > 0 {
		useDeadline = true
		deadline = time.Now().Add(cfg.TimeLimit)
	}
	checkDeadline := func() bool {
		// keep overhead tiny; caller uses a small, fixed number of iterations
		tick++
		if !useDeadline || (tick&2047) != 0 { // only probe wall clock every 2048th tick to amortize
			return false
		}
		return time.Now().After(deadline)
	}

	// Subgradient loop.
	var (
		bestLB    = math.Inf(-1) // best L(π) observed so far
		sumPi     float64        // running Σ π_i
		i, iter   int            // iteration counter
		norm2     float64        // ||s||² where s_i = deg(i)−2
		redCost   float64        // reduced-cost sum of the current 1-tree, cost_c'(T(π))
		degDiff   int            // s_i = deg(i) − 2 for the current i
		haveUB    bool           // whether a finite positive UB was supplied
		usedUB    float64        // the UB used by the step-size formula
		step      float64        // current step size t
		lastBound float64        // current L(π) before taking the step
	)
	if !math.IsInf(cfg.UB, 0) && cfg.UB > 0 {
		haveUB = true
		usedUB = cfg.UB // capture the incumbent UB to drive the adaptive step t = α*(UB−L)/||s||²
	}

	for iter = 0; iter < cfg.MaxIter; iter++ {
		if checkDeadline() {
			return 0, nil, ErrTimeLimit
		}

		// Build a minimum 1-tree on reduced costs; fills eng.deg and returns cost_c'(T(π)).
		redCost, err = eng.buildOneTreeReduced()
		if err != nil {
			// Includes ErrIncompleteGraph if MST(V\{root}) is disconnected or <2 finite root edges exist.
			return 0, nil, err
		}

		// L(π) = cost_c'(T) − 2*Σπ (classical Held–Karp dual objective).
		sumPi = 0
		for i = 0; i < n; i++ {
			sumPi += eng.pi[i] // accumulate Σ π_i over all vertices
		}
		lastBound = redCost - 2*sumPi // compute the current dual value L(π)
		if lastBound > bestLB {
			bestLB = lastBound
		}

		// Check if T(π) is already a tour: deg(i)==2 for all i ⇒ subgradient s=0 ⇒ ||s||²=0.
		norm2 = 0
		for i = 0; i < n; i++ { // compute the squared 2-norm of the subgradient s
			degDiff = eng.deg[i] - 2 // s_i = deg_T(i) − 2 encodes degree violation at i
			norm2 += float64(degDiff * degDiff)
		}
		if norm2 == 0 {
			// The 1-tree is a Hamiltonian cycle; L(π) equals that tour's reduced-cost sum minus 2Σπ.
			break
		}

		// Step size policy:
		//  - With UB:  t = α * (UB − L) / ||s||²  (standard Held–Karp recipe),
		//    clamped to a non-negative value to avoid moving in the wrong direction.
		//  - Without UB: monotone decreasing schedule t = α / (1 + iter).
		if haveUB {
			step = usedUB - lastBound
			if step < 0 {
				step = 0 // avoid negative step if the current bound exceeds or matches UB
			}
			step = cfg.Alpha * step / norm2 // scale by α and normalize by the subgradient norm²
		} else {
			step = cfg.Alpha / (1.0 + float64(iter)) // diminishing steps ensure stability without UB feedback
		}
		if step == 0 {
			// No useful progress possible under the chosen schedule.
			break
		}

		// π_i ← π_i + t * (deg(i) − 2) for all i (plain subgradient ascent on the dual).
		for i = 0; i < n; i++ {
			eng.pi[i] += step * float64(eng.deg[i]-2)
		}
	}

	// Return the best lower bound seen (stabilized) and the final degrees (diagnostics).
	outDeg := make([]int, n)
	copy(outDeg, eng.deg)

	return round1e9(bestLB), outDeg, nil
}

// oneTreeEngine owns reusable mutable buffers for Held-Karp 1-tree construction.
// It avoids per-iteration allocations while keeping reduced-cost, degree, and Prim
// state explicit and testable.
//
// Implementation:
//   - Stores the validated dense/snapshot cost representation.
//   - Stores pi multipliers and degree vector.
//   - Reuses Prim arrays across buildOneTreeReduced calls.
//
// Behavior highlights:
//   - Internal-only.
//   - Not concurrency-safe.
//   - One engine instance is scoped to one lower-bound computation.
//
// Notes:
//   - Keeping this as a struct is preferable to closures because branch-and-bound
//     may later reuse or inspect bound state.
type oneTreeEngine struct {
	n    int       // number of vertices
	root int       // distinguished root vertex r
	w    []float64 // dense weights, length n*n (original costs c_{ij})

	// Lagrange multipliers (π_i).
	pi []float64

	// Working state for building a 1-tree.
	deg    []int  // degree in the current 1-tree
	inTree []bool // whether a vertex is already included in the MST over V\{root}
	parent []int  // Prim parent for MST over V\{root}
	key    []float64
}

// reduced returns the symmetric reduced cost c'(u,v)=c(u,v)+pi[u]+pi[v].
// The caller must pass valid distinct local vertices.
//
// Implementation:
//   - Stage 1: Fetch original cost.
//   - Stage 2: Add both endpoint multipliers.
//   - Stage 3: Return the reduced edge cost.
//
// Behavior highlights:
//   - Does not mutate engine state.
//   - Symmetric by construction when the original matrix is symmetric.
//
// Inputs:
//   - u: first local vertex.
//   - v: second local vertex.
//
// Returns:
//   - float64: reduced symmetric edge cost.
//
// Errors:
//   - None. Caller owns bounds validation.
//
// Determinism:
//   - Pure indexed arithmetic.
//
// Complexity:
//   - Time O(1), Space O(1).
func (e *oneTreeEngine) reduced(u, v int) float64 {
	return e.w[u*e.n+v] + e.pi[u] + e.pi[v]
}

// zeroDegrees clears the reusable degree vector before building the next 1-tree.
//
// Implementation:
//   - Stage 1: Scan the full degree slice.
//   - Stage 2: Set every degree entry to zero.
//
// Behavior highlights:
//   - Mutates only e.deg.
//   - Allocation-free.
//   - Called once per subgradient iteration.
//
// Inputs:
//   - None.
//
// Returns:
//   - None.
//
// Errors:
//   - None.
//
// Determinism:
//   - Fixed index-order clear.
//
// Complexity:
//   - Time O(n), Space O(1).
func (e *oneTreeEngine) zeroDegrees() {
	var i int
	for i = 0; i < e.n; i++ {
		e.deg[i] = 0 // reset degree counters before constructing the next 1-tree
	}
}

// buildOneTreeReduced builds a minimum 1-tree under current reduced costs.
// It first computes an MST over V\{root} with deterministic O(n^2) Prim, then
// adds the two cheapest reduced-cost edges incident to root.
//
// Implementation:
//   - Stage 1: Clear degrees and Prim state.
//   - Stage 2: Run Prim on all vertices except root.
//   - Stage 3: Reject disconnected V\{root}.
//   - Stage 4: Add the two cheapest finite root edges.
//   - Stage 5: Fill degree counts and return the reduced total.
//
// Behavior highlights:
//   - Mutates reusable degree and Prim buffers.
//   - Does not mutate multipliers pi.
//   - Deterministic under equal reduced costs.
//   - Returns reduced-cost total; caller converts to original lower bound.
//
// Inputs:
//   - None; reads engine root, costs, and multipliers.
//
// Returns:
//   - float64: total reduced cost of the constructed 1-tree.
//   - error: nil when a complete 1-tree exists.
//
// Errors:
//   - ErrIncompleteGraph when V\{root} is disconnected or fewer than two root edges are finite.
//
// Determinism:
//   - Fixed vertex scans and stable tie-breaking.
//
// Complexity:
//   - Time O(n^2), Space O(n).
//
// AI-Hints:
//   - Do not include root in the MST phase.
//   - Do not forget to add exactly two root edges.
func (e *oneTreeEngine) buildOneTreeReduced() (float64, error) {
	var inf = math.Inf(1)
	e.zeroDegrees()

	// ---- Prim over V \ {root} using reduced costs.
	var (
		v, u, best, iter int     // vertex indices / loop counters
		c                float64 // working reduced cost
		costReduced      float64 // total reduced cost of the 1-tree edges
	)

	// Initialize keys/parents/inTree for the (n−1)-vertex MST.
	for v = 0; v < e.n; v++ {
		e.inTree[v] = false
		e.parent[v] = -1
		e.key[v] = inf
	}
	// Pick a deterministic start in V\{root}.
	start := 0
	if start == e.root { // ensure the start vertex is not the root (root is excluded from the MST)
		start = 1 // pick the next available vertex
	}
	e.key[start] = 0 // seed Prim: first extraction will choose 'start'

	// Extract the minimum-key vertex (n−1 times), add it, and relax neighbors.
	for iter = 0; iter < e.n-1; iter++ {
		// Find best = argmin_{v ∉ tree, v≠root} key[v] with index tiebreak.
		best = -1
		for v = 0; v < e.n; v++ { // scan all vertices to pick the next MST vertex
			if v == e.root || e.inTree[v] {
				continue // skip the root and vertices already included into the MST
			}
			if best == -1 || e.key[v] < e.key[best] || (e.key[v] == e.key[best] && v < best) {
				best = v // maintain the current minimum by key, breaking ties by vertex index
			}
		}
		if best == -1 || math.IsInf(e.key[best], 0) {
			// Disconnected V\{root}: no MST ⇒ no 1-tree can be formed.
			return 0, ErrIncompleteGraph
		}

		// Include 'best'. If it has a parent, account for that edge.
		e.inTree[best] = true     // mark 'best' as part of the MST
		if e.parent[best] != -1 { // if not the starting vertex (which has no parent yet)
			u = e.parent[best]     // retrieve the MST parent of 'best'
			c = e.reduced(best, u) // reduced cost of the selected MST edge (best,u)
			costReduced += c       // accumulate the reduced-cost total
			e.deg[best]++          // update endpoint degrees in the emerging 1-tree
			e.deg[u]++
		}

		// Relax edges from 'best' to remaining vertices in V\{root}.
		for v = 0; v < e.n; v++ { // try improving keys for all not-yet-in-tree vertices
			if v == e.root || e.inTree[v] || v == best {
				continue // skip root, already-in-tree vertices, and self
			}
			c = e.reduced(best, v)
			if c < e.key[v] { // if joining via 'best' yields a smaller reduced cost
				e.key[v] = c       // update the tentative key (edge weight)
				e.parent[v] = best // and remember 'best' as the new parent
			}
		}
	}

	// ---- Add the two cheapest root edges by reduced cost.
	var (
		m1To, m2To int     // endpoints of the best and second-best root edges
		m1, m2     float64 // their reduced costs
	)
	m1, m2 = inf, inf
	m1To, m2To = -1, -1

	for v = 0; v < e.n; v++ { // scan all non-root vertices to find the two cheapest root edges
		if v == e.root {
			continue // root-to-root is invalid; we need r→v with v≠r
		}
		c = e.reduced(e.root, v)
		if c < m1 || (c == m1 && v < m1To) {
			// Promote current minimum to second minimum, then update the minimum.
			m2, m2To = m1, m1To
			m1, m1To = c, v
		} else if c < m2 || (c == m2 && v < m2To) {
			m2, m2To = c, v // update the second-best root edge
		}
	}
	if math.IsInf(m1, 0) || math.IsInf(m2, 0) {
		// Fewer than two finite root edges ⇒ no 1-tree exists.
		return 0, ErrIncompleteGraph
	}

	costReduced += m1 + m2 // include the two root edges into the reduced-cost total
	e.deg[e.root] += 2     // root has degree 2 in any 1-tree
	e.deg[m1To]++          // increment degrees at the endpoints of the two root edges
	e.deg[m2To]++

	return costReduced, nil
}
