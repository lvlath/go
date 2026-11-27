# AGENTS.md

## Project identity and authority

- Repository and Go module: `github.com/lvlath/go`.
- Minimum declared Go version: 1.23.4; verify the selected `go.mod`.
- License: AGPL-3.0-only. One module; algorithm packages own their APIs.
- Purpose: deterministic, contract-driven algorithm engineering.

This file is operational guidance, not a replacement for the
[Universal Engineering Standard](docs/lvlath_UES.md), package GoDoc, source,
tests or executable examples. Read the current exported API, implementation,
package charter, tests and guide together. Existing behavior is evidence, not
permission to preserve a mathematical bug; disagreements require an explicit
contract decision and regression proof. Never infer an API from a roadmap or
an older release. Do not silently prefer one conflicting source.

## Package map

The following families are a navigation convention, not import layering.
`core` and `matrix` own foundational graph and dense representations;
independent scalar kernels should not import either for cosmetic reuse.

### A. Topology, paths, capacity and discrete optimization (11)

- [`core`](core/doc.go): Deterministic capability-driven graph topology, stable IDs and traversal surfaces.
- [`builder`](builder/doc.go): Deterministic graph constructors and numeric datasets for experiments.
- [`gridgraph`](gridgraph/doc.go): Rectangular lattice connectivity, island analysis, 0-1 expansion and conversion to core.Graph.
- [`bfs`](bfs/doc.go): Unweighted shortest-hop traversal, deterministic paths and weak components.
- [`dfs`](dfs/doc.go): Depth-first forest, finish order, cycle witnesses and topological sort.
- [`dijkstra`](dijkstra/doc.go): Full SSSP and four target methods: Dijkstra, bidirectional, consistent A*, balanced bidirectional A*.
- [`mst`](mst/doc.go): Prim/Kruskal MST/MSF, KMB graph Steiner approximation, closed or fixed-end tree-covering walks.
- [`flow`](flow/doc.go): Canonical MaxFlow (Dinic, Edmonds-Karp, Ford-Fulkerson) with MaxFlowResult residual, cut and interruption evidence.
- [`matrix`](matrix/doc.go): Row-major dense algebra, graph adapters, APSP, stable statistics, pivoted solves and direct economy SVD.
- [`match`](match/doc.go): Minimum-weight perfect matching on an ordered even matrix subset; exact-or-error Blossom and explicit Greedy.
- [`tsp`](tsp/doc.go): Matrix-backed TSP/ATSP, exact Held-Karp and completed Branch-and-Bound, Christofides and local search.

### B. Time, features and alignment (7)

- [`series`](series/doc.go): Exact integer-nanosecond grids, causal lag/difference/returns, complete index windows.
- [`acf`](acf/doc.go): Direct fixed-window sample autocorrelation with common or lag-adjusted denominator and pair counts.
- [`pacf`](pacf/doc.go): Common-n Durbin-Levinson partial autocorrelation and order-wise prediction-error variance.
- [`paa`](paa/doc.go): Fractional equal-duration compression with exact rational segment geometry and correctly rounded means.
- [`sax`](sax/doc.go): Immutable symbolic encoders over fractional PAA, explicit normalization and compatible Euclidean MINDIST.
- [`dtw`](dtw/doc.go): Scalar, cost-matrix and multivariate alignment; bounded classifications and exact fixed-length squared-scalar top-K search.
- [`tfsax`](tfsax/doc.go): Research-only exact absolute-cost DTW oracle versus classical SAX TopM retrieval.

### C. Statistics, regression, hidden states and inference (6)

- [`stats`](stats/doc.go): Stable scalar moments, Chan merges, immutable fitted column standardization.
- [`ols`](ols/doc.go): Full-rank least squares by column-pivoted QR, physical-unit coefficients and frozen prediction.
- [`ridge`](ridge/doc.go): Positive-Lambda L2 regression via protected augmented pivoted QR, including wide/collinear designs.
- [`pca`](pca/doc.go): Immutable training covariance eigenbasis, projection and reconstruction in original units.
- [`hmm`](hmm/doc.go): Categorical and diagonal-Gaussian hidden Markov models, filtering, smoothing, Viterbi and multi-sequence MAP-EM.
- [`lstm`](lstm/doc.go): Frozen single-layer unidirectional PyTorch IFGO recurrent inference over masked batch-first tensors.

### D. Stochastic computation and econometrics (7)

- [`rng`](rng/doc.go): Version-pinned SHA-256/PCG-DXSM logical-path streams, open uniforms and cached normals.
- [`bootstrap`](bootstrap/doc.go): Moving, Circular and Stationary dependent-data block-index ensembles.
- [`montecarlo`](montecarlo/doc.go): Bounded-parallel independent vector sampling with ordered moments and optional exact empirical quantiles.
- [`gbm`](gbm/doc.go): Exact-in-distribution lognormal endpoint paths on explicit elapsed-time grids.
- [`jumpdiffusion`](jumpdiffusion/doc.go): Experimental Merton compound-Poisson lognormal jumps on GBM grids.
- [`gjrgarch`](gjrgarch/doc.go): Stationary asymmetric GJR-GARCH(1,1), causal variance, expected-variance forecasts and Gaussian QMLE.
- [`arima`](arima/doc.go): Fixed nonseasonal conditional Gaussian ARIMA with anchored level forecasts and marginal innovation variance.

### E. Indicators: readiness, arithmetic and recovery (11)

- [`ema`](ema/doc.go): Exact affine exponential levels with explicit SMA or First seed and recoverable streams.
- [`atr`](atr/doc.go): Exact signed-candle True Range and Wilder ATR with independent readiness and replay.
- [`donchian`](donchian/doc.go): Observed rolling high/low channels with explicit current-bar policy and exact midpoint.
- [`vwap`](vwap/doc.go): Exact anchored price-volume means with positive-weight readiness and transactional reset.
- [`sessionlevels`](sessionlevels/doc.go): Exact previous completed session high, low and final close over explicit partitions, with shared readiness.
- [`drawdown`](drawdown/doc.go): Correctly rounded sampled equity depth, exact maximum witness and first recovery; batch-only.
- [`rsi`](rsi/doc.go): Wilder gain/loss oscillator over ordered finite positive prices, batch and checkpointable stream.
- [`macd`](macd/doc.go): SMA-seeded fast/slow Appel EMAs, signed Line/Signal/Histogram and separate masks.
- [`mfi`](mfi/doc.go): Rolling volume-weighted typical-price flow over P transitions, compensated sums.
- [`obv`](obv/doc.go): Cumulative signed current-candle base volume with compensated accumulation.
- [`ad`](ad/doc.go): Chaikin close-location-weighted current-candle base volume with compensated accumulation.

## Repository-wide laws

1. **Mathematical contracts first.** Define domain, objective, result, errors, and guarantees before implementation; do not substitute weaker mathematics or change a valid algorithm to satisfy a wrong test.
2. **Explicit policy.** Graph capabilities, weights, windows, seeds, estimator conventions, memory, matching and solver choices are declared; no hidden fallback.
3. **Deterministic evidence.** Preserve observable order and tie rules. A witness, mask, residual, bound or convergence diagnostic is meaningful only under its documented contract.
4. **Numeric domains.** Finite graph weights, real zero, absence, unreachable `+Inf`, invalid input, overflow and warmup are distinct. Use the owning package's numerical policy, not a universal epsilon.
5. **Errors are protocol.** Use `errors.Is`; check errors before consuming a result, including any explicitly supported partial or diagnostic artifact.
6. **Ownership and concurrency.** Respect aliasing, detached outputs, immutable models and single-owner streams. Thread-safe storage is not snapshot isolation.
7. **Temporal causality.** Preserve time grids, original indices and validity; no inferred observations, future data or holdout refit. Fixed-window estimators are not automatically causal.
8. **Reproducible stochastic and stateful work.** Keep versioned logical RNG identities, deterministic reductions and exact continuation where promised; structural fingerprints are not authentication.
9. **Bounded work.** Account for preparation, validation, allocation bytes, requested artifacts, interruption, and actual kernel complexity. Do not advertise universal speed, OOM recovery or real-time guarantees.
10. **Documentation parity.** Source, public GoDoc, UES, examples, tests, benchmarks and guides must agree. Verify the actual selected version; distinguish a proposal from an implemented feature.

## Change workflow

1. Identify the exact revision, package and observable contract. Read `go.mod`, the UES, package `doc.go`, relevant guide, public API, validators, results and tests. Inspect actual dependencies before modifying them.
2. Reproduce a defect with an independent mathematical or behavioral oracle. Establish the correct domain, error precedence, ownership, determinism, cancellation and result law before proposing code. Separate a contract change from an internal refactor.
3. Prefer the smallest coherent change. Reuse existing `core`/`matrix` operations when their contracts match; do not force graph or dense storage into independent scalar algorithms. Keep canonical facades and thin wrappers; no duplicated kernels or invented options/sentinels.
4. Validate dimensions, bytes, finite-domain input and arithmetic before publication where practical. Preserve error atomicity, exact/approximate distinction, explicit interruption status, and documented output identity. Never silently clamp, regularize, resample or fall back.
5. Preserve compatible behavior, exported names and stable error identity. Document intentional breaking changes before implementation. Do not rewrite correct mathematics to satisfy an erroneous expected value.
6. Add independent validation, normal, degenerate, adversarial, regression and metamorphic tests. Exercise deterministic cancellation and actual allocation/ownership laws. Update examples and all affected documentation; do not delete useful unrelated material.
7. Benchmark the relevant workload, including preparation where applicable. Report actual results and limitations; no unsupported speed, coverage or CI claims.

## Go style and verification discipline

- English GoDoc for every exported symbol; meaningful private-kernel comments for non-obvious arithmetic, state transitions and bounds. Use `gofmt`, idiomatic names and checked errors.
- No ordinary-input panic/recover, mutable global policy or RNG, reflect-based testing, test-only production exports, hidden unsafe shortcuts or casual lint suppressions. Do not add dependencies or nested modules without an explicit contract review.
- Use external `package <name>_test` and the owning public API for package contract tests. Reusable fixtures/oracles belong in `test_helper_test.go` where that is the package convention; follow existing UES-compatible filenames rather than inventing duplicates. Check fixture setup. Compare errors with `errors.Is`, exact bits only where promised, and tolerance only with justified scale/conditioning.
- Keep independent oracles, deterministic ties, insertion-order permutations, witnesses, cancellation, input/result ownership, resource admission and race tests appropriate to the package. Never weaken a valid assertion merely to pass CI. Preserve the required 92% overall and per-executable-file coverage where enforced; noninstrumented files are N/A. A coverage number is not a correctness proof.
- Keep benchmarks' setup outside the measured loop, use fixed workloads/seeds and `-benchmem`; record toolchain, OS/architecture, CPU, shape, options and command. Include adapter costs in end-to-end regimes, not in mislabeled kernel comparisons.

## Required verification

Run the relevant package while iterating, then the repository gates on the
required toolchain. If a tool or environment is unavailable, report it.

```bash
gofmt -w <changed-go-files>
go test ./...
go test -race ./...
go vet ./...
golangci-lint run
govulncheck ./...
```

For benchmark-sensitive changes use `go test -run '^$' -bench . -benchmem ./path/to/package`.
For coverage use a fresh instrumented profile and the package's documented gate
(e.g. VWAP's `LVLATH_VWAP_COVERPROFILE` is an absolute path; do not reuse a stale
profile). Compile complete documentation examples and compare checked outputs.
Do not silently refresh frozen source fixtures or label engineered data as
exchange-audited. Keep live-network, authentication and execution outside kernels.

## Package-specific cautions

Read each owning package for full validation order, numeric bounds, status and
snapshot grammar. The list below states the highest-risk contract distinctions,
not every private field or branch. Apply the shared laws to all packages.

### A. Topology, paths, capacity and discrete optimization

- **`core`:** Declare directed, weighted, loop, parallel and mixed-edge policy. Graph weights are finite; structural pointers may alias live records. An undirected edge retains its stored endpoints. Locking is not snapshot isolation.
- **`builder`:** BuildGraph validates options and executes constructors in order; check setup errors, preserve seed and shape, and do not infer correctness from one generated family.
- **`gridgraph`:** Declare land threshold and neighborhood (4/8); distinguish minimum conversions from weighted route cost. Grid coordinates and graph IDs are separate representations.
- **`bfs`:** Visited means discovered/enqueued; Order means processed. Honor depth/filter/hooks and partial-result contracts; weak components are not SCCs. Weighted cost is not hop count.
- **`dfs`:** Order is finish, not discovery. A cycle witness is not exhaustive cycle enumeration. Require a DAG for topological order and respect partial-order validity.
- **`dijkstra`:** Keep unknown, unreachable (+Inf), disabled tracking and broken witnesses distinct. DistanceTo/ShortestPathTo run full SSSP; target PathResult retains original EdgeIDs. Validate finite nonnegative costs, heuristic consistency, exact target cutoff and walls.
- **`mst`:** Strict MST never silently becomes MSF. Finite negative edges can be valid for MST, not KMB. SteinerResult network edges and WalkResult repeated passages have different objectives and costs; preserve original IDs.
- **`flow`:** Finite capacity uses explicit epsilon policy; undirected edges become independent directed arcs, parallel arcs aggregate. Preserve residual/cut and Partial semantics; no input mutation or task orchestration.
- **`matrix`:** Preserve finite zero edges with explicit policy and +Inf absence; never export metric closure as original edges. View aliases; Clone detaches; read zero-element transform exceptions. FactorLU/QR/Cholesky are distinct from historical decompositions; SVD is not PCA or a solve. Respect rank, condition, residual, convergence and finite-domain diagnostics.
- **`match`:** Complete finite nonnegative approximately symmetric costs; no triangle inequality required. Preserve original index pairs, subset order and raw cost; never mutate TSP adjacency or silently fall back to Greedy.
- **`tsp`:** Tour cost is not shortest path or MST cost. Christofides 1.5 requires symmetric complete metric input and exact Blossom; Greedy does not certify it. Inspect incumbent, timeout, exactness and optimality separately.

### B. Time, features and alignment

- **`series`:** Keep sample indices, missing whole slots, timestamps and Valid masks separate; no interpolation or implicit resampling. Do not compress gaps into false adjacency.
- **`acf`:** Fit one full-window mean or use explicit raw origin. LagAdjusted can exceed unit magnitude and lose Toeplitz PSD; no clipping, no causal prefix claim, FFT or significance test.
- **`pacf`:** Explicit full-window centering; preserve separate coefficient rows and reject nonpositive prediction variance, no epsilon repair. Not automatic AR fit or confidence bands.
- **`paa`:** Every source observation contributes by overlap; no tail loss, padding or unweighted integer-bin shortcut. Returned averages/segments are detached.
- **`sax`:** Right-bin equality and encoding-policy identity matter; compare compatible words only. MINDIST is not a certified DTW lower bound; identity digest is not authentication.
- **`dtw`:** Local cost, window, penalty, memory and path artifacts are explicit. Exact Search has its own certified LB_Keogh domain; neither generic DTW nor SAX MINDIST inherits that bound. DTW is not generally metric.
- **`tfsax`:** No TFSAX codec or proven symbolic DTW pruning. Complete and Exact are independent; heuristic shortlist stays non-exact at observed recall 1. Preserve temporal provenance, strict-ID metrics and NA.

### C. Statistics, regression, hidden states and inference

- **`stats`:** Exact represented-input mean and scaled second moments are distinct. Respect ddof, constant-column policy, transactional update/merge and train-only transforms; one owner per mutable accumulator.
- **`ols`:** Explicit intercept or exact zero origin; rank/shape/condition diagnostics are not repaired by normal equations or pseudoinverse fallback. Fit preprocessing on training only.
- **`ridge`:** Penalty applies to optimization slopes, never intercept. Preserve physical predictions and explicit scaling, numerical rank and checked objective; no hidden Lambda=0 OLS mode.
- **`pca`:** CovarianceEigen is implemented; ThinSVD remains unsupported on the reviewed contract. Verify spectrum/PSD/rank, compare degenerate subspaces, no holdout recentering/refit or automatic SVD fallback.
- **`hmm`:** Filter/FilterStep are causal; smoothing/decoding use full supplied sequence. Preserve zero-probability support, log-domain overflow distinction and independent training boundaries; ModelID is not source authentication.
- **`lstm`:** No training or prediction head. Keep gate/bias order, exact-zero masked output, unchanged masked state, ModelID-bound detached Hidden/Cell, numeric error instead of clipping. Check error before state reuse.

### D. Stochastic computation and econometrics

- **`rng`:** One mutable stream per owner; immutable factory; preserve raw draw order, normal spare and version identity. Not cryptographic randomness.
- **`bootstrap`:** Returns indices, not statistics or confidence intervals. Preserve paired observations, unbiased bounded integers, per-replicate RNG identity, memory admission and complete-row interruption semantics.
- **`montecarlo`:** One stream per logical path, not worker; 2W scheduling bound, canonical reduction, byte admission; failed run exposes only contiguous complete prefix, no quantiles. Sampler owns model, payoffs and concurrency safety.
- **`gbm`:** Positive finite initial price, matching units, no implicit annualization; zero volatility consumes no entropy. Preserve safe log-growth publication and distinguish completed prefix from terminal success; no calibration or measure selection.
- **`jumpdiffusion`:** Use arithmetic jump compensator expm1(m+s^2/2), not log mean; exact discrete bounded Poisson inversion/PTRS, no Gaussian count fallback. Intensity=0 delegates to GBM. Interrupted intervals are not terminal payoffs.
- **`gjrgarch`:** Variance for r[t] is fixed before observing r[t]; negative shocks affect next variance. Respect stationarity and convergence diagnostics; unfinished fit is not an approved model; variance is not VaR.
- **`arima`:** Condition on burn-in max(p,q) of differenced observations; strict Schur roots, no clipping or exact diffuse MLE claim. Preserve historical anchors; variance excludes parameter uncertainty.

### E. Indicators: readiness, arithmetic and recovery

- **`ema`:** Finite signed values, 1<=P<MaxInt, no default; SMA ready at P-1, First at 0. Exact seed and RN64(((P-1)*prev+2*x)/(P+1)); P1 canonical +0. Keep bounded exact SeedSum; never substitute MACD's private kernel or rounded alpha. Snapshot v1 `ema/affine-rn/v1`: exact signed seed in 2^-1074 units, canonical decimal <=660 bytes and Count-based bound, zero outside SMA warmup; match Options and preserve canonical Current. Failed Push/Restore changes nothing.
- **`atr`:** Validate finite low<=close<=high. TR0=RN64(H-L); later prior-close gaps. Seed P rounded TR values then RN64(((P-1)*prev+TR)/P); P1 identity. Positive periods, named default 14, Options{} invalid. TR never masked by ATR warmup. Snapshot v1 `atr/wilder-rn/v1`: previous close and exact unsigned seed in 2^-1074 units, canonical decimal <=660 bytes and Count-based bound. Retain per-row nonfinite-before-geometry validation and independent TR. Failed Push/Restore rolls back.
- **`donchian`:** Include [i-P+1,i] ready P-1; exclude [i-P,i-1] ready P. Preserve latest equal extrema bits, one nearest-even midpoint, validate even masked/excluded rows. No default, Stream or Snapshot; MaxInt period legal. All failures nil; width arithmetic external. Context/options/shape, then global nonfinite priority over remembered finite ErrBar. Mixed nil/nonempty matches both ErrShape and ErrNilInput. Packed disjoint cap-limited numeric channels share one allocation; keep phase-local polling. No streaming continuation is implemented.
- **`vwap`:** All finite signed prices, nonnegative finite weights, explicit first anchor. Reset before row; zero-weight prefix +0/false; later zero weight retains value/count. Exact weights 2^-1074 and products 2^-2148, one nearest-even ratio. No implicit HLC3, period or calendar. Snapshot v1 `vwap/exact-weighted/v1`: canonical decimal sums <=1300 bytes, V>=0, V<=Count*Umax, |W|<=V*Umax; Umax=(2^53-1)<<2045, bit bounds 2162/4260. Count includes zero-weight rows; reset can restart at Count=1 after MaxUint64. Restore derives Current from sums. Row-order errors (finite before negative volume within row), explicit first anchor and mixed-shape dual sentinel must be preserved; constructed stream required.
- **`sessionlevels`:** Keep `Compute` canonical; `ComputeDefault`, `ComputeDaily`, and `ComputeWeekly` only construct Options{} and delegate. No calendar inference, period, Stream or Snapshot. Require non-nil equal-length finite H/L/C with low<=close<=high; nil before shape; valid non-nil empty prices plus empty sessions succeeds. Validate nonempty adjacent [Start,End) intervals covering [0,n), only final incomplete; ErrSession precedes price checks. Global ErrNonFinite beats remembered first finite ErrBar; every failure returns nil. First interval +0/false, final incomplete inherits predecessor. Copy exact final close and earliest equal extrema bits (not Donchian's latest tie). Keep O(n+s) time/O(n) owning output/O(1) workspace, independent buffers, 1024-element phase-local context polls and IsNil pointer check. Preserve fixed-manifest revision horizon and admissible prefix parity; no synthetic bars, truncated complete predecessor, or source-availability claim. Keep external tests, oracle, wrappers, five checked offline examples, standalone scenario and uppercase guide aligned.
- **`drawdown`:** Keep `Compute` canonical; `ComputeMaximum` and `ComputeCurrentDepth` are full-batch scalar projections (O(n) transient output), not distinct kernels or streaming. Require nonempty finite equity, first >0 and later >=0; later signed zero is total loss. Context, shape, then ascending index finite-before-sign checks. All failures return nil Result; scalar wrappers return +0 with the unchanged error. RN64((P-E)/P) on exact binary64 inputs: Sterbenz near peak, bounded integer quotient/remainder for deep loss, exact cross-products for witness comparison. Never choose a witness from rounded Depth, including collisions at 1. Strict new peaks and strict exact improvements retain earliest equal peak/trough. Recovery is first later equity >= selected peak, -1 if absent; no drawdown witness 0/0/0. Maximum copies Depth[TroughIndex]; no warmup, no Options, no Stream/Snapshot. Keep 1024-element phase-local polls, final publication check, detached Depth, O(n) time/O(1) auxiliary workspace and reflect-free IsNil. NAV/cashflow treatment, sampling, intrabar paths, timestamps and thresholds remain external. Depth is prefix invariant; final summary is retrospective. Keep external rational oracle, rounding and cancellation regressions, five offline scenarios and uppercase guide aligned.
- **`rsi`:** Period counts changes, not candles; warmup mask is not an oscillator zero. Keep shared batch/stream recurrence and flat-price convention; reject invalid rows without state advance. Keep the versioned scalar state, gain/loss seed, valid-mask transition and exact batch/Push parity in one environment.
- **`macd`:** Signal starts from ready Line, not warmup zeros; preserve signed outputs and distinct readiness. MACD's private recurrence is not ema's exact affine ABI. Keep batch/stream parity and snapshot identity. Keep independent Line and Signal/Histogram readiness, seed counts and signed finite state; no shared-state race or hidden reinitialization.
- **`mfi`:** Preserve zero-volume slots and canonical typical-price arithmetic; reject invalid OHLCV; keep rolling ring and rebasing state in checkpoint. No fabricated first transition. Keep canonical typical price, exact transition count, detached ring, two Neumaier accumulators and deterministic rebasing; all-zero-flow ready value is 50.
- **`obv`:** First bar is valid zero regardless of its volume; compare successive closes and keep zero-volume continuity and Neumaier correction. No implicit session reset from time. Keep prior close, cumulative compensation and first-candle zero; preserve checkpoint and Count contracts.
- **`ad`:** First bar contributes; flat candle contributes zero, no division by flat range or overflowing 2*Close. Keep Neumaier correction and source-bound replay distinct from OBV's seed. Keep first-bar contribution, cumulative compensation and package-specific batch validation order; do not invent a previous-close state.

## Documentation parity and AI-generated changes

Keep the package `doc.go`, adjacent per-symbol GoDoc, uppercase guide, examples,
tests and benchmark claims synchronized. Cross-package changes also update
README, Tutorial, FAQ, root `doc.go`, `llms.txt` and this file as appropriate.
README explains purpose/composition; Tutorial teaches; FAQ diagnoses; root
GoDoc summarizes; llms.txt indexes; AGENTS protects implementation contracts.
Preserve unrelated examples and existing anchors. Diagrams distinguish actual
imports from caller-side data composition. Use the real source paths (matrix
uses `docs/MATRICES.md`, gridgraph `docs/GRID_GRAPH.md`).

Before accepting AI-generated code, check every symbol and import against the
selected revision, math and edge cases against independent evidence, error
identity and state/ownership laws against the owning contract, and test/lint
claims against actual output. Do not publish a proposed API or partial model as
implemented/approved. Keep private credentials and application-specific source
records out of public fixtures and documentation.
