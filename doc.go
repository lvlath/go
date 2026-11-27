// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package main documents lvlath, a deterministic, contract-driven Go toolkit for
// graph topology, routing, optimization, dense algebra, time-series analysis,
// statistical inference, stochastic simulation, econometrics and causal indicators.
// This is a repository overview, not a re-exporting algorithm facade. Import the
// owning subpackage (for example github.com/lvlath/go/dtw).
//
// lvlath is pure Go. The module is github.com/lvlath/go; read go.mod for the selected
// toolchain (Go 1.23.4 at this revision). Core owns topology, matrix owns dense
// numeric artifacts, and independent kernels own their mathematical domains.
// The five families below are navigation categories, not mandatory import layers.
//
// # Error model and invariants
//
// Use errors.Is for package-level sentinel classes. Numeric domains differ: graph
// weights are finite; a real zero, absent edge, known unreachable +Inf, arithmetic
// overflow and invalid data are distinct. A false readiness mask is not a zero
// measurement. IsNil reports pointer presence where documented, not correctness.
// Check error before consuming result data; use partial/diagnostic/incumbent
// artifacts only under their explicit package contracts. Preserve validation order,
// stable tie rules, caller ownership and cancellation behavior.
//
// # Determinism and numerics
//
// Fixed iteration and reduction order, explicit policy and versioned random
// streams support reproducibility within the documented numeric environment.
// Exact represented-input arithmetic and approximate numerics are different laws;
// use independent oracles, not a global epsilon. Graph locks do not establish
// whole-run snapshot isolation. Immutable models can be shared under stable-input
// contracts; mutable streams, accumulators and RNG instances have one owner.
// Training parameters remain frozen during inference. Timestamps, source revisions,
// durable cursors, authentication and policy remain with callers; structural state
// validation and a public fingerprint do not authenticate history.
//
// # Package overview (balanced highlights)
//
// A. Topology, paths, capacity and discrete optimization (11 packages)
//
//   - core: Deterministic capability-driven graph topology, stable IDs and traversal surfaces.
//   - builder: Deterministic graph constructors and numeric datasets for experiments.
//   - gridgraph: Rectangular lattice connectivity, island analysis, 0-1 expansion and conversion to
//     core.Graph.
//   - bfs: Unweighted shortest-hop traversal, deterministic paths and weak components.
//   - dfs: Depth-first forest, finish order, cycle witnesses and topological sort.
//   - dijkstra: Full SSSP and four target methods: Dijkstra, bidirectional, consistent A*, balanced
//     bidirectional A*.
//   - mst: Prim/Kruskal MST/MSF, KMB graph Steiner approximation, closed or fixed-end tree-covering
//     walks.
//   - flow: Canonical MaxFlow (Dinic, Edmonds-Karp, Ford-Fulkerson) with MaxFlowResult residual, cut and interruption evidence.
//   - matrix: Row-major dense algebra, graph adapters, APSP, stable statistics, pivoted solves and
//     direct economy SVD.
//   - match: Minimum-weight perfect matching on an ordered even matrix subset; exact-or-error
//     Blossom and explicit Greedy.
//   - tsp: Matrix-backed TSP/ATSP, exact Held-Karp and completed Branch-and-Bound, Christofides and
//     local search.
//
// B. Time, features and alignment (7 packages)
//
//   - series: Exact integer-nanosecond grids, causal lag/difference/returns, complete index windows.
//   - acf: Direct fixed-window sample autocorrelation with common or lag-adjusted denominator and
//     pair counts.
//   - pacf: Common-n Durbin-Levinson partial autocorrelation and order-wise prediction-error
//     variance.
//   - paa: Fractional equal-duration compression with exact rational segment geometry and correctly
//     rounded means.
//   - sax: Immutable symbolic encoders over fractional PAA, explicit normalization and compatible
//     Euclidean MINDIST.
//   - dtw: Scalar, cost-matrix and multivariate alignment; bounded classifications and exact
//     fixed-length squared-scalar top-K search.
//   - tfsax: Research-only exact absolute-cost DTW oracle versus classical SAX TopM retrieval.
//
// C. Statistics, regression, hidden states and inference (6 packages)
//
//   - stats: Stable scalar moments, Chan merges, immutable fitted column standardization.
//   - ols: Full-rank least squares by column-pivoted QR, physical-unit coefficients and frozen
//     prediction.
//   - ridge: Positive-Lambda L2 regression via protected augmented pivoted QR, including
//     wide/collinear designs.
//   - pca: Immutable training covariance eigenbasis, projection and reconstruction in original
//     units.
//   - hmm: Categorical and diagonal-Gaussian hidden Markov models, filtering, smoothing, Viterbi and
//     multi-sequence MAP-EM.
//   - lstm: Frozen single-layer unidirectional PyTorch IFGO recurrent inference over masked
//     batch-first tensors.
//
// D. Stochastic computation and econometrics (7 packages)
//
//   - rng: Version-pinned SHA-256/PCG-DXSM logical-path streams, open uniforms and cached normals.
//   - bootstrap: Moving, Circular and Stationary dependent-data block-index ensembles.
//   - montecarlo: Bounded-parallel independent vector sampling with ordered moments and optional
//     exact empirical quantiles.
//   - gbm: Exact-in-distribution lognormal endpoint paths on explicit elapsed-time grids.
//   - jumpdiffusion: Experimental Merton compound-Poisson lognormal jumps on GBM grids.
//   - gjrgarch: Stationary asymmetric GJR-GARCH(1,1), causal variance, expected-variance forecasts
//     and Gaussian QMLE.
//   - arima: Fixed nonseasonal conditional Gaussian ARIMA with anchored level forecasts and marginal
//     innovation variance.
//
// E. Indicators: readiness, arithmetic and recovery (10 packages)
//
//   - ema: Exact affine exponential levels with explicit SMA or First seed and recoverable streams.
//   - atr: Exact signed-candle True Range and Wilder ATR with independent readiness and replay.
//   - donchian: Observed rolling high/low channels with explicit current-bar policy and exact
//     midpoint.
//   - vwap: Exact anchored price-volume means with positive-weight readiness and transactional
//     reset.
//   - sessionlevels: Previous completed session high, low and final close over explicit
//     observed-index partitions, with a shared readiness mask.
//   - rsi: Wilder gain/loss oscillator over ordered finite positive prices, batch and checkpointable
//     stream.
//   - macd: SMA-seeded fast/slow Appel EMAs, signed Line/Signal/Histogram and separate masks.
//   - mfi: Rolling volume-weighted typical-price flow over P transitions, compensated sums.
//   - obv: Cumulative signed current-candle base volume with compensated accumulation.
//   - ad: Chaikin close-location-weighted current-candle base volume with compensated accumulation.
//
// # Performance and trade-offs (concise, per area)
//
// Complexity includes the selected representation, preparation, requested artifacts
// and numeric policy; these summaries are not universal timing guarantees.
//
// A. Topology, paths, capacity and discrete optimization
//
//   - core: Sorted publication and cloning have costs; keep topology stable throughout algorithms.
//   - builder: Separate fixture creation from measured kernels; generated graph size determines
//     setup and storage.
//   - gridgraph: For W by H cells and d neighbors, component/expansion work is O(W H d), with O(W H)
//     workspace; graph conversion also stores edges.
//   - bfs: Linear traversal O(V+E) before ordering, filters, hooks and publication costs.
//   - dfs: Linear traversal O(V+E), excluding graph-ordering and hook overhead; account for
//     traversal state.
//   - dijkstra: Priority-queue work plus validation, sorted graph reads, snapshot and exact-key
//     arithmetic; target methods do not promise a universal speedup.
//   - mst: Kruskal O(E log E+E alpha(V)); edge-frontier Prim O(E log E). Steiner includes terminal
//     routing/closure and expansion; walking depends on tree size.
//   - flow: Algorithm-dependent augmentation and residual construction; Edmonds-Karp O(V E^2), Dinic
//     O(V^2 E) general textbook bounds for effective networks, excluding adapters.
//   - matrix: Dense O(R C) storage; Floyd-Warshall O(V^3); factor construction, basis-solve
//     conditioning and residuals cost more than repeated triangular solves. Economy SVD depends on
//     sweeps and min/max dimensions.
//   - match: Account for input/subset validation and copies plus the selected engine; exact and
//     greedy workloads differ.
//   - tsp: Held-Karp O(n^2 2^n) time, exponential memory; branch-and-bound remains exponential in
//     worst case; local search is bounded by its policy.
//
// B. Time, features and alignment
//
//   - series: Linear validation/transforms and owning aligned outputs; number of explicit windows
//     depends on stride/length.
//   - acf: O(n(m+1)) time and O(n+m) additional storage for maximum lag m.
//   - pacf: Direct lag products O(n(m+1)) plus O(m^2) recursion; no FFT assumption.
//   - paa: Overlap traversal rather than an all-pairs matrix; output size w, arithmetic cost
//     includes exact accumulation.
//   - sax: Compression plus threshold lookup and O(w) word comparison; indexing/refinement are
//     separate application costs.
//   - dtw: General O(n m) DP work (window may reduce visited cells), multivariate local cost adds
//     feature work; rolling rows vs full path/matrix storage; top-K cost depends on candidates and
//     certified pruning.
//   - tfsax: Full exact scan is the reference; shortlist adds encoding/ranking plus DTW refinement.
//     Compare on frozen plans and workloads.
//
// C. Statistics, regression, hidden states and inference
//
//   - stats: Linear observations; merge cost depends on fixed state; feature fit/transform scans
//     rows by columns.
//   - ols: Tall QR factor work plus diagnostics; prediction uses frozen coefficients, not another
//     fit.
//   - ridge: Augmented system increases rows by predictor count; measure construction, conditioning
//     and solves separately.
//   - pca: Covariance formation and checked eigenspectrum dominate fit; inference scales with
//     retained components.
//   - hmm: Dense transitions give O(T K^2) filtering/decoding work plus emissions; smoothing/EM
//     require history and iterations.
//   - lstm: Each active step evaluates four input/recurrent affine gates; optional sequence
//     retention costs B T H, with per-batch state B H.
//
// D. Stochastic computation and econometrics
//
//   - rng: Constant-size stream state; fill work linear in draws, not a shared global RNG.
//   - bootstrap: Work/storage scale with requested index count; publish complete rows only.
//   - montecarlo: Per-path model work plus ordered vector reduction; retained samples/quantiles
//     require budgeted raw storage and sorting.
//   - gbm: Linear interval work/output; terminal facade need not retain every endpoint; model/RNG
//     costs explicit.
//   - jumpdiffusion: Linear grid traversal plus bounded Poisson work and normal draws; preserve RNG
//     consumption order.
//   - gjrgarch: Linear filtering/horizon work; fit multiplies evaluation cost by optimizer work and
//     active-set checks.
//   - arima: Fit depends on order, observations and optimizer; forecast horizon memory is linear,
//     not a dense covariance guarantee.
//
// E. Indicators: readiness, arithmetic and recovery
//
//   - ema: Two linear batch passes, O(n) output, fixed 34-limb Push with no internal heap
//     allocation; snapshot serialization separate.
//   - atr: Linear batch, O(n) output; fixed 34-limb Push with no internal heap allocation; exact
//     unsigned seed continuation.
//   - donchian: O(n) total, O(n) outputs and O(min(P,n)) rings; masked-only history allocates no
//     rings, single insert may pop O(P).
//   - vwap: Two linear batch scans, O(n) output, fixed 67-limb Push with no internal heap
//     allocation; bounded decimal continuation.
//   - sessionlevels: O(n+s) batch including validation, O(n) owning output, O(1) scalar
//     workspace; four independent output buffers, no per-session allocation.
//   - rsi: Linear batch and fixed-state Push; no internal heap allocation in documented hot path.
//   - macd: Linear batch, fixed-state streaming; no hidden history retention.
//   - mfi: Linear total window updates with periodic rebasing and O(P) rolling state where
//     allocated; owning O(n) batch output.
//   - obv: Linear batch, fixed-state Push with no internal heap allocation.
//   - ad: Linear batch and fixed-state streaming; no previous-close history needed.
//
// # Scope boundaries and known limitations
//
// A. Topology, paths, capacity and discrete optimization
//
//   - core: Declare directed, weighted, loop, parallel and mixed-edge policy. Graph weights are
//     finite; structural pointers may alias live records. An undirected edge retains its stored
//     endpoints. Locking is not snapshot isolation.
//   - builder: BuildGraph validates options and executes constructors in order; check setup errors,
//     preserve seed and shape, and do not infer correctness from one generated family.
//   - gridgraph: Declare land threshold and neighborhood (4/8); distinguish minimum conversions from
//     weighted route cost. Grid coordinates and graph IDs are separate representations.
//   - bfs: Visited means discovered/enqueued; Order means processed. Honor depth/filter/hooks and
//     partial-result contracts; weak components are not SCCs. Weighted cost is not hop count.
//   - dfs: Order is finish, not discovery. A cycle witness is not exhaustive cycle enumeration.
//     Require a DAG for topological order and respect partial-order validity.
//   - dijkstra: Keep unknown, unreachable (+Inf), disabled tracking and broken witnesses distinct.
//     DistanceTo/ShortestPathTo run full SSSP; target PathResult retains original EdgeIDs. Validate
//     finite nonnegative costs, heuristic consistency, exact target cutoff and walls.
//   - mst: Strict MST never silently becomes MSF. Finite negative edges can be valid for MST, not
//     KMB. SteinerResult network edges and WalkResult repeated passages have different objectives
//     and costs; preserve original IDs.
//   - flow: Finite capacity uses explicit epsilon policy; undirected edges become two independent
//     directed arcs, parallel arcs aggregate. MaxFlowResult separates completed cut evidence
//     from partial interruption; the input graph is not mutated.
//   - matrix: Preserve finite zero edges with explicit policy and +Inf absence; never export metric
//     closure as original edges. View aliases; Clone detaches; read zero-element transform
//     exceptions. FactorLU/QR/Cholesky are distinct from historical decompositions; SVD is not PCA
//     or a solve. Respect rank, condition, residual, convergence and finite-domain diagnostics.
//   - match: Complete finite nonnegative approximately symmetric costs; no triangle inequality
//     required. Preserve original index pairs, subset order and raw cost; never mutate TSP adjacency
//     or silently fall back to Greedy.
//   - tsp: Tour cost is not shortest path or MST cost. Christofides 1.5 requires symmetric complete
//     metric input and exact Blossom; Greedy does not certify it. Inspect incumbent, timeout,
//     exactness and optimality separately.
//
// B. Time, features and alignment
//
//   - series: Keep sample indices, missing whole slots, timestamps and Valid masks separate; no
//     interpolation or implicit resampling. Do not compress gaps into false adjacency.
//   - acf: Fit one full-window mean or use explicit raw origin. LagAdjusted can exceed unit
//     magnitude and lose Toeplitz PSD; no clipping, no causal prefix claim, FFT or significance
//     test.
//   - pacf: Explicit full-window centering; preserve separate coefficient rows and reject
//     nonpositive prediction variance, no epsilon repair. Not automatic AR fit or confidence bands.
//   - paa: Every source observation contributes by overlap; no tail loss, padding or unweighted
//     integer-bin shortcut. Returned averages/segments are detached.
//   - sax: Right-bin equality and encoding-policy identity matter; compare compatible words only.
//     MINDIST is not a certified DTW lower bound; identity digest is not authentication.
//   - dtw: Local cost, window, penalty, memory and path artifacts are explicit. Exact Search has its
//     own certified LB_Keogh domain; neither generic DTW nor SAX MINDIST inherits that bound. DTW is
//     not generally metric.
//   - tfsax: No TFSAX codec or proven symbolic DTW pruning. Complete and Exact are independent;
//     heuristic shortlist stays non-exact at observed recall 1. Preserve temporal provenance,
//     strict-ID metrics and NA.
//
// C. Statistics, regression, hidden states and inference
//
//   - stats: Exact represented-input mean and scaled second moments are distinct. Respect ddof,
//     constant-column policy, transactional update/merge and train-only transforms; one owner per
//     mutable accumulator.
//   - ols: Explicit intercept or exact zero origin; rank/shape/condition diagnostics are not
//     repaired by normal equations or pseudoinverse fallback. Fit preprocessing on training only.
//   - ridge: Penalty applies to optimization slopes, never intercept. Preserve physical predictions
//     and explicit scaling, numerical rank and checked objective; no hidden Lambda=0 OLS mode.
//   - pca: CovarianceEigen is implemented; ThinSVD remains unsupported on the reviewed contract.
//     Verify spectrum/PSD/rank, compare degenerate subspaces, no holdout recentering/refit or
//     automatic SVD fallback.
//   - hmm: Filter/FilterStep are causal; smoothing/decoding use full supplied sequence. Preserve
//     zero-probability support, log-domain overflow distinction and independent training boundaries;
//     ModelID is not source authentication.
//   - lstm: No training or prediction head. Keep gate/bias order, exact-zero masked output,
//     unchanged masked state, ModelID-bound detached Hidden/Cell, numeric error instead of clipping.
//     Check error before state reuse.
//
// D. Stochastic computation and econometrics
//
//   - rng: One mutable stream per owner; immutable factory; preserve raw draw order, normal spare
//     and version identity. Not cryptographic randomness.
//   - bootstrap: Returns indices, not statistics or confidence intervals. Preserve paired
//     observations, unbiased bounded integers, per-replicate RNG identity, memory admission and
//     complete-row interruption semantics.
//   - montecarlo: One stream per logical path, not worker; 2W scheduling bound, canonical reduction,
//     byte admission; failed run exposes only contiguous complete prefix, no quantiles. Sampler owns
//     model, payoffs and concurrency safety.
//   - gbm: Positive finite initial price, matching units, no implicit annualization; zero volatility
//     consumes no entropy. Preserve safe log-growth publication and distinguish completed prefix
//     from terminal success; no calibration or measure selection.
//   - jumpdiffusion: Use arithmetic jump compensator expm1(m+s^2/2), not log mean; exact discrete
//     bounded Poisson inversion/PTRS, no Gaussian count fallback. Intensity=0 delegates to GBM.
//     Interrupted intervals are not terminal payoffs.
//   - gjrgarch: Variance for r[t] is fixed before observing r[t]; negative shocks affect next
//     variance. Respect stationarity and convergence diagnostics; unfinished fit is not an approved
//     model; variance is not VaR.
//   - arima: Condition on burn-in max(p,q) of differenced observations; strict Schur roots, no
//     clipping or exact diffuse MLE claim. Preserve historical anchors; variance excludes parameter
//     uncertainty.
//
// E. Indicators: readiness, arithmetic and recovery
//
//   - ema: Finite signed values, 1<=P<MaxInt, no default; SMA ready at P-1, First at 0. Exact seed
//     and RN64(((P-1)*prev+2*x)/(P+1)); P1 canonical +0. Keep bounded exact SeedSum; never
//     substitute MACD's private kernel or rounded alpha.
//   - atr: Validate finite low<=close<=high. TR0=RN64(H-L); later prior-close gaps. Seed P rounded
//     TR values then RN64(((P-1)*prev+TR)/P); P1 identity. Positive periods, named default 14,
//     Options{} invalid. TR never masked by ATR warmup.
//   - donchian: Include [i-P+1,i] ready P-1; exclude [i-P,i-1] ready P. Preserve latest equal
//     extrema bits, one nearest-even midpoint, validate even masked/excluded rows. No default,
//     Stream or Snapshot; MaxInt period legal. All failures nil; width arithmetic external.
//   - vwap: All finite signed prices, nonnegative finite weights, explicit first anchor. Reset
//     before row; zero-weight prefix +0/false; later zero weight retains value/count. Exact weights
//     2^-1074 and products 2^-2148, one nearest-even ratio. No implicit HLC3, period or calendar.
//   - sessionlevels: Only final interval may be incomplete; first session masked. Earliest
//     equal extrema and exact final close, signed finite H/L/C. Any nil price channel
//     fails; non-nil empty prices with no sessions succeed. Context, nil, shape, session
//     structure, global nonfinite before finite geometry; all errors return nil.
//     Daily/weekly wrappers require explicit manifests, not inferred time.
//   - rsi: Period counts changes, not candles; warmup mask is not an oscillator zero. Keep shared
//     batch/stream recurrence and flat-price convention; reject invalid rows without state advance.
//   - macd: Signal starts from ready Line, not warmup zeros; preserve signed outputs and distinct
//     readiness. MACD's private recurrence is not ema's exact affine ABI. Keep batch/stream parity
//     and snapshot identity.
//   - mfi: Preserve zero-volume slots and canonical typical-price arithmetic; reject invalid OHLCV;
//     keep rolling ring and rebasing state in checkpoint. No fabricated first transition.
//   - obv: First bar is valid zero regardless of its volume; compare successive closes and keep
//     zero-volume continuity and Neumaier correction. No implicit session reset from time.
//   - ad: First bar contributes; flat candle contributes zero, no division by flat range or
//     overflowing 2*Close. Keep Neumaier correction and source-bound replay distinct from OBV's
//     seed.
//
// # Streaming continuation and causal composition
//
// EMA and ATR retain exact bounded seed state; VWAP retains exact weights and
// products across explicit anchors. Their versioned continuation laws must not
// be substituted by approximate accumulators. EMA uses a signed seed, ATR an
// unsigned seed and previous close (each decimal <=660 bytes); VWAP retains two
// sums (each <=1300 bytes), with fixed magnitude/count/convexity constraints.
// SessionLevels and Donchian have no stream/snapshot. SessionLevels needs an
// admitted complete predecessor for ready levels, not a saved incremental state.
// It validates all supplied bars, including masked and incomplete intervals,
// and leaves calendar coverage, corrections and as-of availability to callers.
// For Donchian, validate the complete supplied history even
// for excluded or masked rows. It uses a different batch error precedence from
// ATR and VWAP. RSI, MACD, MFI, OBV and AD have their own shared batch/stream
// arithmetic and checkpoint ABI. Read package-specific phase rules and source
// admission requirements. Output masks, count and state fingerprints do not
// establish event availability or trade execution. Keeping one packed Donchian
// view retains its allocation; copying a result struct does not deep-copy slices.
// Snapshot serialization, JSON, custom context and I/O costs are distinct from
// internal allocation claims. No allocation arithmetic check guarantees physical
// memory availability or hard real-time latency.
//
// # Where to go next
//
//   - README.md: package map and working entry points.
//   - docs/README.md: guide index and import boundaries.
//   - docs/TUTORIAL.md: learning path and data-composition diagrams.
//   - docs/FAQ_&_TIPS.md: troubleshooting and result interpretation.
//   - docs/lvlath_UES.md: engineering and verification standard.
//   - {package}/doc.go and exported GoDoc: formal implemented contract.
//   - docs/{PACKAGE}.md: mathematics, examples and pitfalls; MATRICES.md and
//     GRID_GRAPH.md are the matrix and gridgraph guides; DTW_SEARCH.md extends DTW.
//   - AGENTS.md and llms.txt: operational rules and machine-oriented discovery.
package main
