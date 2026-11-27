// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package main documents the lvlath repository, a deterministic, contract-driven
// Go toolkit for graph algorithms, dense graph algebra, routing, network optimization,
// time-series alignment, exact affine levels, exact candle ranges, observed rolling
// Donchian boundaries, exact anchored volume-weighted means, tours, and reproducible fixtures.
// This file is the repository overview, not a re-exporting algorithm facade.
// Import the owning subpackage, for example github.com/lvlath/go/ema.
//
// lvlath is pure Go: no cgo and no external runtime dependencies. The repository
// is organized as small, composable subpackages. Each package owns a narrow
// mathematical contract and publishes result artifacts that callers can inspect,
// test, serialize, and reason about.
//
// ─────────────────────────────────────────────────────────────────────────────
// What lvlath is
// ─────────────────────────────────────────────────────────────────────────────
//
// lvlath is a package ecosystem, not a monolithic solver. The selected overview
// below introduces core roles; README.md contains the complete package map:
//
//   - core      - deterministic, capability-driven graph substrate.
//   - bfs       - unweighted hop-distance traversal and weak components.
//   - dfs       - finish-order traversal, cycle witnesses, and topological sort.
//   - dijkstra  - non-negative weighted single-source shortest paths.
//   - mst       - strict MST and explicit minimum spanning forest via Kruskal/Prim.
//   - flow      - max-flow / min-cut algorithms with residual graph artifacts.
//   - matrix    - dense row-major graph algebra, APSP, statistics, sanitation.
//   - ema       - exact affine exponential levels with explicit SMA/First seeds,
//     aligned readiness masks, single-owner streams, and bounded warmup snapshots.
//   - atr       - exact True Range and Wilder ranges over signed finite candles,
//     independent ATR readiness, atomic streams, and bounded exact seed replay.
//   - donchian  - observed high/low ranges with include/exclude-current policy,
//     nearest-even boundary midpoints, shared readiness and bounded monotonic deques.
//   - vwap      - exact price-volume means with explicit reset origins, positive-weight
//     readiness, atomic streams, and bounded exact rational continuation.
//   - dtw       - deterministic Dynamic Time Warping for scalar, cost-matrix,
//     and multivariate sequence alignment.
//   - tsp       - matrix-backed TSP/ATSP solvers: exact, approximate, matching,
//     local-search, and time-governed regimes.
//   - gridgraph - generated 2D lattice topology for pathfinding-style workflows.
//   - builder   - deterministic fixtures for examples, tests, and benchmarks.
//
// The main architectural rule is:
//
//	core owns topology;
//	matrix owns dense numeric artifacts;
//	algorithm packages own explicit result contracts;
//	ema owns signed finite level smoothing, not timestamps or execution policy;
//	atr owns candle range measurement, not channel widths, NATR, or risk decisions;
//	donchian owns observed range boundaries, not source timing or execution rules;
//	vwap owns exact anchored weighted means, not calendars, source admission, or fills.
//
// Each package documents implemented API behavior in its own doc.go. Repository
// theory, diagrams, pitfalls, and operational recipes live in docs/{PACKAGE}.md.
// Start with docs/TUTORIAL.md for the learning path and docs/lvlath_UES.md for
// the project engineering standard.
//
// ─────────────────────────────────────────────────────────────────────────────
// Error model and invariants
// ─────────────────────────────────────────────────────────────────────────────
//
// Each package defines its own reachable sentinel error categories. Classify them
// with errors.Is; a familiar name does not imply the same domain or failure law.
// For example, ema defines ErrNilContext, ErrNilInput, ErrInvalidOption,
// ErrNonFinite, ErrOverflow, and ErrStateVersion. It has no ErrShape category:
// one scalar input slice supplies both owning output channels. A short SMA history
// is successful but not ready. Cancellation returns ctx.Err() and no batch result.
//
// A result's IsNil method reports pointer presence, not completeness, readiness,
// convergence, or numeric validity. Read the package-specific fields after checking
// error. EMA's Value{} can mean either successful warmup or an error; only the
// returned error distinguishes those two outcomes.
// Donchian nil/nonempty mismatches match both ErrShape and ErrNilInput. Its complete
// finiteness scan precedes any finite ErrBar, including for a wholly masked history.
// VWAP instead checks rows in increasing index order: a finite negative volume
// precedes a later NaN. Its nil/nonempty shape also matches both ErrShape and
// ErrNilInput; its nonempty first row requires an explicit reset before row-domain
// checks. Empty options have no invalid value. Do not infer another package's
// validation order from a familiar sentinel name.
//
// Policy:
//   - Validate inputs early and explicitly (shape, bounds, NaN, ±Inf).
//   - Make ambiguous conditions explicit through options (e.g., symmetric vs ATSP).
//   - Disallow unsupported combinations in dispatchers (e.g., Christofides with ATSP).
//
// ─────────────────────────────────────────────────────────────────────────────
// Determinism and numerics
// ─────────────────────────────────────────────────────────────────────────────
//
//   - No global state, no implicit RNG. Algorithms are deterministic for the
//     same inputs and options.
//   - Where randomized neighborhood order is useful (local search), it is fully
//     controlled by Options.Seed and can be disabled; results remain stable under
//     the same seed.
//   - Numerical comparison rules belong to each package. Approximate solvers may
//     require justified error bounds; exact masks, indices, and tie decisions do not.
//   - EMA specifies one nearest-even rounding of each exact rational affine step.
//     Its oracle uses represented binary64 inputs and rounds each step separately;
//     compare output bits and masks, not rounded display strings or a blanket epsilon.
//     Zero outputs are canonical +0, including a period-one input of -0.
//   - Donchian selects the latest equal extrema, preserving their input bits.
//     Its exact-zero midpoint is +0; a negative nonzero mean may round to -0.
//     Invalid rows have +0 payloads and false masks. A finite midpoint does not
//     imply a representable Upper-Lower width or a safe ratio denominator.
//   - VWAP accumulates exact represented-binary64 products and weights, then rounds
//     each prefix ratio once. Exact zero is +0; negative nonzero underflow may be -0.
//     The first anchor is mandatory; positive cumulative weight, not Count alone,
//     controls readiness. No floating price*volume product is formed.
//
// ─────────────────────────────────────────────────────────────────────────────
// Package overview (balanced highlights)
// ─────────────────────────────────────────────────────────────────────────────
//
// core
//
//	Provides the deterministic graph substrate used by downstream packages.
//	Graph capabilities such as directedness, weights, loops, multi-edges, and
//	mixed edges are explicit model choices. Stable topology surfaces make tests,
//	witnesses, examples, and algorithm results reproducible.
//
// bfs
//
//	Computes unweighted hop-distance traversals and weak components. BFS result
//	fields separate discovered, queued, processed, depth, and parent semantics.
//	Partial results are meaningful under cancellation or hook failure according
//	to the package contract.
//
// dfs
//
//	Provides deterministic depth-first traversal, finish-order output, cycle
//	witnesses, and topological sorting. dfs.Result.Order is finish order, not
//	discovery order. Cycle detection publishes deterministic witnesses rather
//	than pretending to enumerate every simple cycle.
//
// dijkstra
//
//	Computes single-source shortest paths on non-negative weighted graphs.
//	The package separates unknown targets, known-but-unreachable targets,
//	disabled path tracking, and no-path states. +Inf is the canonical distance
//	for known unreachable vertices under the active policy.
//
// mst
//
//	Computes minimum spanning trees and explicit minimum spanning forests over
//	weighted, undirected graphs. MinimumSpanningTree is the canonical facade;
//	Kruskal and Prim are focused wrappers. mst.Result reports algorithm, mode,
//	root, detached selected edges, total weight, vertex count, component count,
//	and deterministic component roots.
//
// flow
//
//	Provides max-flow / min-cut workflows through Ford-Fulkerson,
//	Edmonds-Karp, and Dinic-style algorithms. Weights represent capacities in
//	this package. Results expose residual-network state, which is part of the
//	algorithm artifact rather than incidental debug data.
//
// matrix
//
//	Implements deterministic row-major dense matrices, graph-to-matrix adapters,
//	metric closure, Floyd-Warshall, algebra facades, LU/QR/inverse/eigen helpers,
//	statistical transforms, sanitation, and tolerant comparison. Matrix policy
//	preserves important graph meanings such as zero-weight edges, +Inf absence,
//	incidence signs, metric-closure distance surfaces, and zero-shape matrices.
//
// ema
//
//	Compute(ctx, values, Options) produces detached Values and Valid arrays with
//	the original length. ComputeSMA and ComputeFirst select the seed explicitly;
//	NewSMA and NewFirst construct streams under the same policies. No period
//	default exists: require 1 <= Period < MaxInt. Negative values and subnormals
//	are accepted; NaN and infinities are rejected even during warmup.
//
//	SMA first becomes valid at P-1 with the exact represented-input sum/P,
//	rounded once. First seeding is ready at index zero. Later steps compute
//	RN64(((P-1)*previous+2*x)/(P+1)), not a floating-point difference update.
//	Period one selects the input, except that -0 becomes +0. Empty input is a
//	successful non-nil Result with non-nil empty arrays; short SMA input is masked.
//
//	Push and Restore are error-atomic, not concurrently synchronized. A stream
//	has one owner; independent streams and stable-input batch calls may run in
//	parallel. Reset preserves options and discards history. Snapshot preserves
//	an unfinished exact seed as bounded decimal SeedSum in units of 2^-1074.
//	Version=1 and Algorithm="ema/affine-rn/v1" identify the continuation law.
//	Persist source identity and offsets separately but atomically with state;
//	structural validation neither authenticates history nor deduplicates events.
//
//	The package imports no other lvlath package. Callers may validate timestamps
//	with series or consume the level as a separately implemented Keltner
//	centerline, but EMA provides neither ATR nor channel widths or decisions.
//	See docs/EMA.md and the six executable ema/example_test.go scenarios.
//
// atr
//
//	Compute(ctx, high, low, close, Options) owns aligned TR, ATR, and Valid arrays.
//	ComputePeriod/NewPeriod select an explicit observation count. ComputeDefault
//	and NewDefault explicitly choose DefaultPeriod=14; Options{} remains invalid.
//	All positive int periods are admitted. Prices may be negative, but must be
//	finite with low<=close<=high. Invalid geometry is ErrBar, not a clamp policy.
//
//	The first TR is RN64(high-low), without an invented previous close. Later TR
//	includes gaps from the last accepted close. Only ATR has warmup: the exact mean
//	of P rounded TR values first becomes ready at P-1. Later updates round the exact
//	Wilder rational ((P-1)*previousATR+TR)/P once. P1 gives bit-identical ATR=TR.
//	A ready zero is valid, not permission to divide by zero. IsNil checks presence.
//
//	One Stream has one owner. Failed Push/Restore changes no state. Snapshot v1,
//	atr/wilder-rn/v1, retains PreviousClose and an exact unfinished SeedSum in
//	2^-1074 units. Restore bounds canonical decimal length to 660 and validates
//	phase/configuration before commit, but cannot authenticate source history.
//	Save the source revision and next offset atomically with the state. Corrections
//	require replay from an earlier checkpoint; event identity stays outside ATR.
//
//	The kernel imports only standard-library packages. Push uses fixed 34-limb
//	work and no internal heap allocation; Compute uses two linear passes and O(n)
//	output. Snapshot serialization and custom context costs are separate. See
//	docs/ATR.md, six atr/example_test.go scenarios, and examples/atr CSV replay.
//
// donchian
//
//	Compute(ctx, high, low, Options) returns owning Upper, Middle, Lower, and
//	Valid channels of input length. ComputeIncludeCurrent and ComputeExcludeCurrent
//	select the mode and delegate unchanged to Compute. No default period exists;
//	all positive int periods are valid. No Stream or Snapshot is implemented.
//
//	Include-current uses [i-P+1,i], first ready at P-1; exclude-current uses
//	[i-P,i-1], first ready at P. Read Valid before all three payloads. Empty
//	input produces non-nil empty channels; short history is successful warmup.
//	A ready zero is a measurement, while IsNil only checks pointer presence.
//
//	Observed extrema retain the latest equal value's bits. Middle is the
//	nearest-even rounding of their exact mean, without false overflow or loss
//	of representable subnormal results. Finite signed H/L must satisfy high>=low;
//	all supplied bars are validated, including the last excluded observation.
//	Every error returns nil. Numeric prefix causality is not partial publication.
//
//	Two monotonic rings give O(n) total time and O(min(P,n)) index workspace.
//	Wholly masked histories allocate no rings; four aligned outputs still use
//	O(n) memory. The three numerical slices are disjoint, capacity-limited views
//	of one packed backing array. Keeping one retains the array; copying Result
//	by value remains shallow. Inputs are read-only and are not retained.
//
//	A current-inclusive channel is causal after bar availability, but does not
//	provide a prior-bar breakout threshold. Exclude removes the current bar
//	from that threshold, not historical revision risk or execution assumptions.
//	The caller owns timestamps, closes, units, missing bars, source identity,
//	derived widths, ATR alignment, and artifact metadata. Independent calls
//	on stable input are safe; shared result mutation needs synchronization.
//	See docs/DONCHIAN.md and the five offline donchian/example_test.go scenarios.
//
// vwap
//
//	Compute(ctx,price,volume,reset,Options) owns aligned Values/Valid. ComputeDefault
//	and NewDefault delegate with Options{}, without choosing an anchor or calendar.
//	New constructs a single-owner Stream. A nonempty batch requires reset[0]=true;
//	a fresh/reset stream requires reset=true on its first accepted Push.
//
//	Finite signed prices and nonnegative weights are valid. No positive segment
//	weight means +0/false; later zero-weight rows retain the preceding ready value
//	and still increment Count. Reset occurs before the accepted row. Failed
//	Push/Restore cannot erase an active segment; a separate Reset is unconditional.
//
//	Two fixed 67-limb magnitudes hold VolumeSum in 2^-1074 units and signed
//	WeightedSum in 2^-2148 units. One nearest-even rounding publishes the exact
//	ratio. Extreme products and cancellation preserve a representable mean;
//	bar close/HLC3 weighting remains a proxy for missing underlying trade prices.
//
//	Snapshot v1, vwap/exact-weighted/v1, carries exact ASCII decimal sums, each
//	bounded to 1300 bytes. Restore checks phase, bit lengths, Count-based volume
//	and convexity bounds before one assignment. It does not authenticate history.
//	Count is not a feed cursor; persist source/revision, cursor, and anchor policy
//	atomically outside the package. Corrections require verified prefix replay.
//
//	No other lvlath package is imported. Caller-side composition with ATR,
//	Donchian, or series must preserve units, aligned masks, source availability,
//	and finite consumer arithmetic. No rolling window, merge, trade simulation,
//	liquidity inference, or execution-quality guarantee is supplied. See
//	docs/VWAP.md and the five offline vwap/example_test.go scenarios.
//
// dtw
//
//	Implements Dynamic Time Warping through canonical Result-returning facades:
//	Align for scalar sequences, AlignCostMatrix for precomputed local-cost
//	surfaces, and AlignMatrix for multivariate time-step matrices. Window,
//	slope-penalty, memory mode, path tracking, accumulated matrix output, and
//	local-cost artifacts are explicit policy choices.
//
// tsp
//
//	Provides deterministic matrix-backed Traveling Salesman Problem solvers.
//	SolveMatrix is the canonical matrix facade; SolveGraph adapts core.Graph at
//	the boundary. ExactHeldKarp and completed BranchAndBound are exact regimes;
//	Christofides is a symmetric metric approximation pipeline; BlossomMatch is
//	the exact matching mode required for the 1.5 ratio; GreedyMatch is explicit
//	and weaker; TwoOptOnly and ThreeOptOnly are local-search regimes. Result
//	publishes tour, cost, algorithm, exactness, optimality, timeout state,
//	approximation ratio, metric-closure flag, and detached IDs.
//
// gridgraph
//
//	Generates grid/lattice topology for pathfinding, teaching, and benchmark
//	fixtures. Use it when a map-like graph should be generated from coordinate
//	rules instead of manually wiring hundreds of edges.
//
// builder
//
//	Builds deterministic graph and data fixtures for examples, tests, and
//	benchmarks. Use it to avoid random, unstable, or hand-wired setup when the
//	goal is reproducible algorithm behavior.
//
// ─────────────────────────────────────────────────────────────────────────────
// Performance and trade-offs (concise, per area)
// ─────────────────────────────────────────────────────────────────────────────
//
//   - core: deterministic graph surfaces and capability checks are favored over
//     raw map-only shortcuts. Algorithms rely on this stability.
//   - bfs / dfs: traversal is O(V+E). Hooks and filters add per-event overhead;
//     cancellation and partial-result behavior are contract-visible.
//   - dijkstra: O((V+E) log V) style priority-queue routing over non-negative
//     weights. Runtime walls and max-distance cutoffs are policy gates, not
//     graph mutations.
//   - mst: Kruskal is O(E log E + E·α(V)); Prim is O(E log E) for the current
//     edge-frontier heap implementation. Do not document Prim as O(E log V)
//     unless the implementation changes to a vertex-key decrease-key heap.
//   - flow: Edmonds-Karp is simpler and easier to audit; Dinic is typically
//     faster on larger layered networks. Residual graph construction is part of
//     the real cost.
//   - matrix: dense algorithms trade memory O(R*C) for predictable row-major
//     kernels and graph-algebra convenience. Zero-shape matrices are valid
//     structural results.
//   - ema: Compute performs two linear passes, O(n*W) time and O(n+W) storage,
//     W=34 fixed limbs. Push uses O(W) work and local candidate state with no
//     internal heap allocation. Snapshot/Restore use bounded O(D*W) work,
//     D<=660 decimal bytes; Snapshot may allocate the returned seed string.
//     Context implementations, JSON, logging, and caller I/O are separate costs.
//   - donchian: O(n) total validation/kernel work and output initialization, O(n)
//     owning output and O(min(P,n)) index workspace. One insertion can remove
//     O(P) candidates; per-bar O(1) is amortized. Cancellation polling has
//     phase-local iteration bounds, not a hard real-time latency guarantee.
//   - vwap: two batch scans use O(n*L) time, O(n) owning output, and O(L)
//     workspace, L=67. Push uses O(L) work independent of history length and no
//     internal heap allocation. Snapshot/Restore use bounded O(D*L) work,
//     D<=1300; returned decimal strings, JSON, logging, and custom contexts have
//     separate allocation costs. This is not a hard real-time latency guarantee.
//   - dtw: scalar Align is O(n*m); AlignMatrix adds O(n*m*d) local-cost
//     construction. Rolling-row distance mode uses O(m) DP memory; path or
//     accumulated-matrix output requires O(n*m) memory.
//   - tsp: Held-Karp is exponential O(n²·2^n) and size-guarded. Branch-and-Bound
//     is exact only when completed. Christofides requires symmetric complete
//     metric input and exact Blossom matching to publish the 1.5 ratio. Local
//     search improves tours but does not certify global optimality.
//   - builder / gridgraph: generated fixtures reduce setup mistakes and improve
//     benchmark reproducibility; they should not hide the topology being tested.
//
// ─────────────────────────────────────────────────────────────────────────────
// Scope boundaries and known limitations
// ─────────────────────────────────────────────────────────────────────────────
//
//   - core: no persistent graph database, no distributed graph storage, and no
//     hidden algorithm-specific topology mutation.
//   - bfs: unweighted hop-distance traversal only. It is not a weighted routing
//     algorithm.
//   - dfs: cycle detection returns deterministic witnesses, not exhaustive
//     simple-cycle enumeration.
//   - dijkstra: finite negative weights are rejected by design. Use a different
//     shortest-path algorithm for negative-weight domains.
//   - mst: directed optimum branching/arborescence is out of scope. Exact
//     MST/MSF, graph Steiner approximation, and tree-covering walks are distinct
//     public questions; their costs and guarantees must not be interchanged.
//     Strict MST does not silently downgrade to forest mode.
//   - flow: capacities are finite non-negative values. Generalized flows,
//     min-cost flow, and multi-commodity flow are out of scope.
//   - matrix: dense storage is not a sparse-matrix engine. Metric closure is a
//     distance artifact and must not be exported as original topology.
//   - ema: no timestamp inference, gap filling, event deduplication, forecasting,
//     calibrated anomaly threshold, or checkpoint authentication. The recurrence
//     retains exponential state rather than a rolling P-observation window.
//     Do not substitute MACD's private rounding law or use warmup zeros as data.
//   - donchian: no Stream, Snapshot, implicit period, timestamps, close validation,
//     execution, or width/ratio calculation. Batch errors never publish a prefix.
//     Caller-side width or division can overflow despite finite channel values.
//   - vwap: no timestamp or event-availability check, implicit HLC3, order book,
//     rolling period, automatic correction, or checkpoint authentication. Explicit
//     anchors do not prove causal selection; ready means do not approve trades,
//     percentage denominators, or mixing incompatible instrument/volume units.
//   - dtw: exact DTW is not generally a metric and does not imply triangle
//     inequality. Path recovery requires full accumulated storage. AlignMatrix
//     uses squared L2 row costs, so feature scale is caller responsibility.
//   - tsp: solver kernels consume matrix-backed complete cost models. Sparse
//     graph-native TSP kernels and Concorde-style branch-and-cut are out of
//     scope. Branch-and-Bound with a time limit may return a feasible incumbent,
//     but that result is not certified optimal. Greedy matching does not publish
//     the Christofides 1.5 ratio.
//   - builder / gridgraph: fixture generation is deterministic infrastructure,
//     not proof that an algorithm is correct for all possible graph families.
//
// ─────────────────────────────────────────────────────────────────────────────
// Where to go next
// ─────────────────────────────────────────────────────────────────────────────
//
//   - docs/TUTORIAL.md - start here for an end-to-end tour, selection matrix,
//     and practical guidance on determinism and numeric stability.
//   - {package}/doc.go - the formal API contracts, options, and edge cases.
//   - docs/EMA.md - exact affine proofs, error/phase tables, and replay programs.
//   - docs/ATR.md - exact range/seed proofs, signed candle domain, masks, and replay.
//   - docs/DONCHIAN.md - observed windows, exact midpoint, error/mask tables,
//     bounded rings, correction horizons and caller-owned range comparisons.
//   - docs/VWAP.md - exact weights, anchor/readiness laws, transactional replay,
//     snapshot trust, typed checkpoints, and Go-native verification.
//   - README.md - repository package map and current contract documentation routes.
//   - docs/{ALGORITHM}.md - compact backgrounders with formulas, diagrams,
//     and runnable examples for each algorithm.
//
// lvlath targets modern Go, is entirely in Go, and has no external dependencies.
package main
