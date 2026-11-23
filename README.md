<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  lvlath - Repository README

  Purpose:
    This README is the public face of lvlath. It presents the library as a
    deterministic, contract-driven Go toolkit for graph algorithms, flow networks,
    dense graph algebra, routing, exact timestamp grids, causal time-series primitives, Wilder RSI batch/stream filtering, SMA-seeded MACD, compensated MFI, compensated cumulative OBV, Chaikin A/D, fixed-window ACF and Durbin-Levinson PACF, fractional PAA,
    symbolic SAX encoding, research-only TFSAX exact-DTW/SAX retrieval audits, dependent-data block-index resampling, path-indexed parallel Monte Carlo simulation, exact-in-distribution GBM and Merton jump-diffusion paths, causal GJR-GARCH volatility, fixed-order conditional Gaussian ARIMA with anchored forecasts, frozen IFGO LSTM recurrent inference, alignment, and reproducible algorithm engineering. It should give a new reader immediate confidence in
    what lvlath can solve, how packages compose, where the formal contracts live,
    and why this repository is engineered beyond ad-hoc algorithm snippets.

  License:
    lvlath is licensed under AGPL-3.0-only. See LICENSE.
-->

![img.png](img.png)

---

```text
┌─────────────────────────────────┐
│  o       o            o    o    │
│  ║ o   o ║   o───╖  o─╫─o  ║    │
│  ║  \ /  ║   o───╢    ║    ╟──╖ │
│  ╙o  o   ╙o  ╙───o    ╙o   o  o │
└─────────────────────────────────┘
```

---

[![Go Reference](https://pkg.go.dev/badge/github.com/lvlath/go.svg)](https://pkg.go.dev/github.com/lvlath/go)
[![CI](https://github.com/lvlath/go/actions/workflows/go.yml/badge.svg?branch=main)](https://github.com/lvlath/go/actions/workflows/go.yml)
[![Codecov](https://codecov.io/github/lvlath/go/graph/badge.svg?token=QMUZPAO34Y)](https://codecov.io/github/lvlath/go)
[![Go version](https://img.shields.io/badge/Go-%3E%3D1.23.4-00ADD8?logo=go&logoColor=white)](https://go.dev/)
[![License: AGPL-3.0-only](https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg)](LICENSE)

[![GitHub contributors](https://img.shields.io/github/contributors/lvlath/go.svg)](https://github.com/lvlath/go/graphs/contributors)
[![GitHub issues](https://img.shields.io/github/issues/lvlath/go.svg)](https://github.com/lvlath/go/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/lvlath/go.svg)](https://github.com/lvlath/go/commits/main)
[![GitHub release](https://img.shields.io/github/release/lvlath/go.svg)](https://github.com/lvlath/go/releases)



---

# lvlath

**lvlath** is a pure-Go algorithm engineering library for graphs, flows, dense graph algebra, routing, exact timestamp grids, causal series transforms, fractional PAA, symbolic SAX time-series signatures, TFSAX research-only retrieval comparisons, stable statistical moments, fitted feature scaling, sample autocorrelation and partial autocorrelation, Wilder RSI, SMA-seeded MACD, compensated OHLCV Money Flow Index, cumulative On-Balance Volume, Chaikin Accumulation/Distribution, covariance PCA, causal Hidden Markov Models, version-pinned PCG-DXSM stochastic streams, time-series alignment, reproducible parallel Monte Carlo drivers, exact-in-distribution geometric Brownian motion and Merton jump-diffusion paths, asymmetric stationary GJR-GARCH conditional volatility, fixed-order conditional Gaussian ARIMA level forecasts, deterministic single-layer IFGO LSTM inference, and reproducible test fixtures.

It is built for people who need algorithmic tools they can explain in a code review, trust in CI, compose across packages, and debug from deterministic artifacts instead of guessing why an answer changed.

**The core promise:**

> Same validated inputs, same explicit policy, same algorithm - same result contract and failure class.

`lvlath` is not just "BFS, DFS, Dijkstra, and friends." It is a small ecosystem where:

* `core.Graph` gives deterministic, thread-safe topology with explicit capabilities;
* traversal packages publish meaningful result artifacts, not incidental internal state;
* weighted routing keeps `+Inf`, unknown targets, disabled path tracking, and unreachable targets separate;
* `matrix` turns topology into dense numeric objects without erasing zero-weight or metric-closure semantics, and provides stable covariance, reusable pivoted linear-system solvers, and a separate rectangular economy SVD via one-sided Jacobi;
* `series` checks exact timestamp grids and exposes mask-aware causal transformations without interpolation;
* `paa` compresses every sample of a finite series into exact fractional equal-duration averages, without dropped tails or padding;
* `sax` quantizes fractional PAA to compatible symbolic words with Euclidean MINDIST; it never promotes that bound to a DTW pruning certificate;
* `tfsax` supplies an **unapproved research gate** for independent exact-DTW versus classical SAX TopM retrieval evaluation, synthetic stress data and temporal leakage audits; it does **not** supply a new TFSAX encoder or proven DTW pruning;
* `stats` provides scaled Welford/Chan moments and immutable training-only feature standardization without refit;
* `ols` fits full-rank original-unit linear models through pivoted QR, with an explicit zero-origin policy and frozen predictions;
* `ridge` fits strongly correlated or wide linear models with a caller-chosen positive L2 penalty, unpenalized intercept and stable augmented QR coordinates;
* `pca` fits validated covariance principal components and retains immutable training means/scales for leakage-free latent projection and reconstruction;
* `hmm` provides typed categorical and diagonal-Gaussian hidden states with strictly causal filtering, retrospective smoothing, global Viterbi paths, and multi-sequence MAP-EM;
* `acf` computes scaled direct sample autocorrelation with an explicit common versus lag-adjusted estimator convention and exact available-pair counts;
* `pacf` computes common-denominator Durbin-Levinson partial autocorrelation and order-wise linear prediction-error variances, with explicit whole-window centering;
* `rsi` computes causal Wilder price-strength factors with batch/stream bit parity in one environment and detached restart checkpoints;
* `macd` computes signed EMA convergence/divergence, signal and histogram series with independent readiness masks, atomic zero-allocation Push and detached checkpoint replay;
* `mfi` computes compensated rolling OHLCV flow with zero-volume slot preservation, bounded MFI ratios, transactional streaming and detached restart state;
* `obv` accumulates signed current-candle base volume from successive closes, with first-bar zero, Neumaier cancellation protection, batch/stream bit parity and transactional checkpoints;
* `ad` accumulates close-location-weighted current-candle base volume, includes the first bar, and protects its signed balance with Neumaier compensation and versioned checkpoints;
* `bootstrap` constructs reproducible dependent-data index ensembles with unbiased bounded integers, checked memory budgets and complete-row cancellation;
* `rng` supplies local pinned PCG-DXSM128 streams, strict 52-bit open uniforms, cached Box-Muller normals and logical-path replay;
* `montecarlo` orchestrates independent seeded simulation paths with bounded worker scheduling, canonical vector moments, exact empirical quantiles and auditable interruption prefixes;
* `gbm` simulates positive exact-in-distribution lognormal paths on irregular explicitly unit-typed grids, with robust log-growth arithmetic and an error-safe completed-prefix protocol;
* `jumpdiffusion` adds compensated compound-Poisson lognormal jumps over GBM without hidden Gaussian count approximations, RNG-order drift or partial-payoff publication;
* `gjrgarch` provides immutable asymmetric return-variance models, strictly causal filtering, expected-variance forecasts and constrained Gaussian QMLE;
* `arima` fits a fixed nonseasonal conditional Gaussian ARIMA law, validates strict Schur roots and predicts anchored levels with marginal innovation variances;
* `lstm` evaluates one immutable PyTorch-ordered IFGO layer over checked batch-first finite tensors, with zero-padded masks, detached streaming states and exact overflow classification;
* `match` exposes exact-or-error weighted Blossom and explicit last-pivot Greedy over ordered matrix subsets, without importing TSP or mutating adjacency;
* `flow`, `mst`, `tsp`, `dtw`, `gridgraph`, and `builder` packages extend the same discipline into optimization, alignment, fixtures, and examples.

---

## Why lvlath exists

A graph or numeric-algorithm library becomes dangerous when the demo works, but the production system loses ordering, topology meaning, numeric semantics, or error identity.

The common failures are familiar:

* Go map iteration leaks into traversal order;
* directed, undirected, mixed, loop, and multi-edge semantics are guessed instead of modeled;
* zero-weight edges disappear inside adjacency matrices;
* `+Inf` is replaced with "large enough" constants and later becomes a fake route;
* BFS is used for weighted routing because it "looks like pathfinding";
* Dijkstra, MST, TSP, and DTW are treated as interchangeable shortest-cost tools;
* DFS discovery order is confused with finish order;
* max-flow examples print only a number and hide residual-network semantics;
* Christofides is claimed without checking metric assumptions or matching strength;
* DTW path output is requested from a memory mode that intentionally discarded history;
* SAX labels are mistaken for compatible thresholds, or Euclidean MINDIST is misused as an exact-DTW pruning bound;
* examples panic, print unstable maps, or silently rely on fictional APIs;
* documentation explains textbook theory but not the actual package contract;
* high-offset variance disappears through raw-moment cancellation, or inference refits on holdout data;
* hidden-state filters are accidentally replaced with future-conditioned smoothing, or independent EM training sequences are concatenated;
* lag-adjusted autocorrelations are clamped to [-1,1], or fed to a PSD-dependent PACF solver without checking their estimator convention;
* centered and uncentered PACF windows are mixed, their previous-order coefficients are updated in place, or a singular prediction variance is silently epsilon-clamped;
* RSI warmup placeholders are read as genuine oscillator zeros, invalid quotes advance stream state, or a config hash is mistaken for checkpoint authentication;
* MACD is seeded from the first price instead of initial SMA, Signal is seeded with invalid Line zeros, or a checkpoint is replayed from the wrong source offset.
* a zero-volume MFI bar is dropped, shifting the rolling window, or a configuration hash is mistaken for authenticating mutable checkpoint content.
* OBV incorrectly includes the seed candle volume, loses low-order volume after a large cancellation, or uses a public snapshot Config digest as an integrity tag.
* Chaikin A/D incorrectly starts with OBV's zero seed, forms 2*Close near MaxFloat64, divides by a flat spread, or drops a low-order volume correction.
* Monte Carlo worker count silently changes floating-point totals, a blocked low-index path grows an unbounded queue, or a partial result is mistaken for a complete ensemble.
* GBM silently applies annual coefficients to elapsed seconds, overflows a representable `S*exp(increment)` result, or turns a failed endpoint into a fake epsilon-clamped price.
* Merton jump simulation treats log-jump mean as the arithmetic drift compensator, substitutes Gaussian Poisson counts, or prices an incomplete path as maturity.
* neural model exports reorder IFGO gates, omit recurrent biases, treat padding as valid state updates, or silently saturate an overflowed affine expression.

`lvlath` is built against those failure modes. It is an algorithm engineering library: each package publishes a narrow mathematical contract, validates its input domain, returns meaningful result artifacts, and keeps deterministic behavior observable.

### The repository-level laws

1. **Determinism is a package contract**

   `core` provides stable graph surfaces. Algorithm packages either preserve that order or define their own deterministic tie-breaks: BFS processing order, DFS finish order, Dijkstra strict improvement, MST edge ordering, DTW backtracking, TSP matrix order and solver tie-breaks.

2. **Capabilities are declared before topology is trusted**

   Directedness, weighting, loops, multi-edges, and mixed-edge behavior are graph capabilities, not comments. Invalid topology should be rejected at the write or validation boundary, not discovered after an algorithm has already published nonsense.

3. **Every non-trivial algorithm publishes a result artifact**

   BFS `Result`, DFS `Result`, Dijkstra `Result`, MST `Result`, TSP `Result`, DTW `Result`, residual flow graphs, adjacency matrices, incidence matrices, and metric closures are caller-owned artifacts with domain meaning. They are not incidental internal state.

4. **Errors are protocol, not prose**

   Public errors are meant for `errors.Is`. Error text is diagnostic only. A caller must be able to distinguish "unknown target," "known unreachable," "path not tracked," "disconnected strict MST," "invalid matching policy," and "time-limited incumbent."

5. **Policy must be explicit**

   Filters, cutoffs, thresholds, forest mode, metric closure, local search, matching engine, DTW window, memory mode, slope penalty, and TSP time limits are explicit policy choices. No package should silently swap to weaker mathematics because the selected one is inconvenient.

6. **Numeric semantics are not interchangeable**

   `0`, `+Inf`, `NaN`, negative weights, missing edges, zero-cost edges, unreachable states, and local costs mean different things in different packages. `matrix`, `dtw`, `mst`, `dijkstra`, and `tsp` intentionally enforce different numeric laws because they solve different problems.

7. **Graph algorithms and dense algebra compose without erasing meaning**

   `matrix` can convert topology into adjacency, incidence, metric closure, statistics, and factorization workflows. It must preserve zero-weight and `+Inf` semantics instead of flattening them into convenient but false dense arrays.

8. **Time-index contracts must not invent observations**

   `series` checks signed-nanosecond timestamp intervals, keeps sample indices separate from clock time, and publishes validity masks rather than filling missing samples.

9. **Fitted statistics and learned coordinate systems must remain training-only**

   `stats` separates an exact binary-input mean sum from compensated scaled second moments, checks errors before publishing mutations, and freezes fitted feature parameters. PCA similarly freezes its covariance basis, means and optional scales; neither package may recompute training coordinates on a heldout batch. Constant training columns follow explicit policy.

10. **Documentation, examples, tests, and source must describe the same library**

`{package}/doc.go` is the implemented API contract. `docs/{PACKAGE}.md` teaches the mathematics and operational pitfalls. Examples demonstrate real package behavior. Tests defend the contract. None of them should invent future APIs or convenient lies.

11. **ACF estimators and time-window causality are not interchangeable**

    `acf` fits one mean to its entire fixed window. `CommonDenominator` is Toeplitz-PSD-compatible in exact arithmetic, whereas `LagAdjusted` can exceed unit magnitude and need not be PSD. Neither estimator is a causal streaming feature or a statistical significance test.

12. **Cumulative volume has an explicit origin and a protected recurrence**

    `obv` starts with a valid zero regardless of first-candle volume, preserves all observed closes and Neumaier compensation, and rejects arithmetic failure without advancing the stream. Upstream systems, not OBV, establish time continuity and checkpoint authenticity.

13. **Chaikin A/D uses current-candle geometry and contributes from index zero**

    `ad` computes a bounded close-location multiplier from finite price differences and accumulates the current candle's signed base volume, including the first bar. Flat candles contribute zero without division. Errors never publish a partial batch or mutate a rejected Stream; source continuity and checkpoint authentication remain external.

14. **Parallel stochastic paths have a fixed identity and a canonical reduction order**

    `montecarlo` derives one `rng.Factory.Stream(Seed,pathID)` per logical trial, not per worker. The coordinator alone advances path-ordered `stats.Moments` under a `2W` in-flight window; exact quantiles are optional and fully budgeted. Failed runs publish only a contiguous prefix and no quantiles. The model and its statistical assumptions remain with the caller.

15. **GBM time dimensions and path completion cannot be inferred**

    `gbm` requires matching declared years/days/seconds, finite positive input prices and an exact-origin increasing grid. It computes model transitions in log-growth coordinates, consumes no entropy at zero volatility, and never publishes rounded zero or infinity as success. A runtime failure returns a completed prefix plus error, **not** a valid terminal scenario. It neither annualizes the model nor validates its suitability for observed markets.

16. **Jump-model drift and entropy are explicit contracts**

    `jumpdiffusion` distinguishes the log-jump mean `m` from the arithmetic
    price compensator `k=expm1(m+s^2/2)`, uses bounded discrete Poisson
    inversion/PTRS rather than Gaussian count approximation, and delegates
    `Intensity=0` to GBM for exact path/draw parity. It never turns an
    interrupted interval into a complete terminal payoff; the model remains
    experimental until its parameters and use case are independently validated.

17. **Conditional volatility is not a look-ahead price scenario**

    `gjrgarch` assigns `h[t]` before seeing `r[t]`, updates from the prior residual,
    and estimates parameters with an explicit deterministic QMLE start. A forecast
    variance is not a VaR tail quantile; a nonconverged `FitResult` is not an
    approved model. Feed cadence, training/holdout separation and model review
    belong to the application.

18. **ARIMA conditional likelihood is not exact diffuse Kalman MLE**

    `arima` conditions on the first `max(p,q)` differenced observations,
    profiles innovation variance only from later residuals, rejects unit
    roots rather than clipping them, and reconstructs integrated levels
    using historical anchors. Marginal innovation variance is not full
    predictive parameter uncertainty or a risk quantile.

19. **Recurrent inference separates model identity, numerical computation, and decision policy**

    `lstm` imports finite IFGO weights into an immutable model with a canonical
    numerical fingerprint. Masked timesteps never modify hidden/cell state;
    canceled or failed inference returns no partial result. The caller owns
    validated preprocessing, model provenance, trained heads and alert policy.

20. **Partial autocorrelation has one explicit estimator, not a hidden AR model**

    `pacf` uses biased/common-`n` autocovariance and a single Durbin-Levinson
    recursion. Centering is caller-controlled and uses the full supplied window.
    Two coefficient rows protect mirrored recurrence operands; nonpositive
    prediction variance and nonrepresentable arithmetic fail without a partial
    result. PACF supports diagnostic order screening, not automatic AR fitting
    or statistical significance claims.

---

## Installation

```bash
go get github.com/lvlath/go@latest
```

Requires Go 1.23.4 or newer. Pure Go. No cgo. No external runtime dependencies.

---

## Repository structure

```text
lvlath/
├── core/                  # deterministic graph substrate
│   └── doc.go             # package-level API contract
│
├── bfs/                   # unweighted hop traversal and weak components
├── dfs/                   # post-order traversal, cycle witnesses, topological sort
├── dijkstra/              # full SSSP + Dijkstra/A* target-search family
├── mst/                   # Prim/Kruskal MST/MSF, KMB Steiner networks, tree-covering walks
├── flow/                  # Ford-Fulkerson, Edmonds-Karp, Dinic
├── dtw/                   # temporal alignment, bounded outcomes and exact window search
├── series/                # exact timestamp grids, causal transforms, complete index windows
├── paa/                   # fractional PAA, exact rational geometry, detached averages
├── sax/                   # symbolic PAA words, SHA-256 policy identity, Euclidean MINDIST
├── tfsax/                 # research-only exact-DTW/SAX comparison; no TFSAX encoder or certified pruning
├── stats/                 # scaled Welford/Chan moments and immutable fitted scaling
├── ols/                   # QR regression, full-rank gate, frozen physical coefficients
├── ridge/                 # positive-Lambda Tikhonov, balanced augmented QR, immutable fit
├── pca/                   # certified covariance PCA, frozen projection and reconstruction
├── hmm/                   # categorical/Gaussian HMM, causal filter, smoother, Viterbi, MAP EM
├── acf/                   # direct sample ACF, common/lag-adjusted modes, pair counts
├── pacf/                  # biased autocovariance, Durbin-Levinson PACF, prediction variance
├── rsi/                   # Wilder RSI batch and single-owner streaming with versioned snapshots
├── macd/                  # SMA-seeded EMA MACD, aligned masks, transactional stream and snapshots
├── mfi/                   # compensated rolling OHLCV Money Flow Index, stream and snapshots
├── obv/                   # compensated cumulative On-Balance Volume, stream and checkpoints
├── ad/                    # Chaikin A/D from current-candle geometry, compensated cumulative stream
├── bootstrap/             # versioned Moving/Circular/Stationary block-index ensembles
├── rng/                   # pinned SHA256LE/PCG-DXSM streams, open uniforms and cached normals
├── montecarlo/            # bounded parallel paths, ordered moments, exact sample quantiles
├── gbm/                   # exact lognormal GBM paths, explicit units, positive-price prefix law
├── jumpdiffusion/         # experimental Merton Poisson jumps, compensated drift, PTRS
├── gjrgarch/              # causal asymmetric return variance, forecast and Gaussian QMLE
├── arima/                 # conditional Gaussian ARIMA, anchored level/variance forecasts
├── lstm/                  # immutable IFGO forward inference, masked batches, stateful chunks
├── gridgraph/             # 2D lattice graph generation
├── matrix/                # dense graph algebra, stable statistics, pivoted factors, solves and economy SVD
├── match/                 # independent minimum-weight perfect matching: Blossom and Greedy
├── tsp/                   # tours, Christofides adapter over match, local search, exact routing
├── builder/               # deterministic fixtures and graph generators
│
├── docs/
│   ├── TUTORIAL.md
│   ├── CORE.md
│   ├── BFS.md
│   ├── DFS.md
│   ├── DIJKSTRA.md
│   ├── MST.md
│   ├── FLOW.md
│   ├── DTW.md
│   ├── DTW_SEARCH.md
│   ├── SERIES.md
│   ├── STATS.md
│   ├── OLS.md
│   ├── RIDGE.md
│   ├── PCA.md
│   ├── HMM.md
│   ├── ACF.md
│   ├── PACF.md
│   ├── MACD.md
│   ├── MFI.md
│   ├── OBV.md
│   ├── AD.md
│   ├── BOOTSTRAP.md
│   ├── RNG.md
│   ├── MONTECARLO.md
│   ├── GBM.md
│   ├── JUMPDIFFUSION.md
│   ├── GJRGARCH.md
│   ├── ARIMA.md
│   ├── LSTM.md
│   ├── PAA.md
│   ├── SAX.md
│   ├── TFSAX.md           # research gate, exact/heuristic boundaries, provenance, optional JSON artifacts
│   ├── GRID_GRAPH.md
│   ├── MATRICES.md     # canonical complete matrix reference
│   ├── MATCH.md
│   ├── TSP.md
│   ├── lvlath_UES.md
│   └── FAQ_&_TIPS.md
│
├── examples/              # runnable scenario programs
├── CONTRIBUTING.md        # contribution workflow and quality gates
├── LICENSE
├── README.md
└── go.mod
```

### Two documentation layers

```text
lvlath/{package}/doc.go
  ↓
  The implemented API contract:
  exported types, options, errors, complexity, ownership, and package guarantees.

lvlath/docs/{PACKAGE}.md
  ↓
  The learning and specification layer:
  mathematics, proofs, pseudocode, diagrams, pitfalls, scenarios, and package selection guidance.
```

---

## Package map: how the system composes

```text
                                      ┌──────────────────────────────────────┐
                                      │              core.Graph              │
                                      │ deterministic topology + capability  │
                                      │ directed / weighted / loops / multi  │
                                      └───────────────────┬──────────────────┘
                                                          │
        ┌───────────────────────────────┬─────────────────┼────────────────┬───────────────────────────────┐
        │                               │                 │                │                               │
  hop / structure                 weighted cost      connectivity      capacity flow                   dense algebra
        │                               │                 │                │                               │
┌───────▼────────┐              ┌───────▼────────┐ ┌──────▼──────┐ ┌───────▼────────┐              ┌───────▼────────┐
│ bfs            │              │ dijkstra       │ │ mst         │ │ flow           │              │ matrix         │
│ hop layers,    │              │ non-negative   │ │ MST/MSF/ST  │ │ max-flow,      │              │ adjacency,     │
│ paths, weak    │              │ shortest paths,│ │ explicit    │ │ min-cut,       │              │ incidence,     │
│ components     │              │ +Inf states    │ │ forest mode │ │ residual graph │              │ APSP, stats    │
└───────┬────────┘              └───────┬────────┘ └──────┬──────┘ └───────┬────────┘              └───────┬────────┘
        │                               │                 │                │                               │
┌───────▼────────┐                      │                 │                │                       ┌───────▼────────┐
│ dfs            │                      │                 │                │                       │ tsp            │
│ finish order,  │                      │                 │                │                       │ matrix-backed  │
│ cycle witness, │                      │                 │                │                       │ tours: exact,  │
│ topo sort      │                      │                 │                │                       │ approximate,   │
└────────────────┘                      │                 │                │                       │ local search   │
                                        │                 │                │                       └────────────────┘
                                        │                 │                │
                           ┌────────────▼──────────┐      │     ┌──────────▼──────────┐
                           │ gridgraph             │      │     │ builder             │
                           │ generated lattice     │      │     │ deterministic       │
                           │ topology for routing  │      │     │ fixtures, examples, │
                           │ and traversal demos   │      │     │ benchmarks          │
                           └───────────────────────┘      │     └─────────────────────┘
                                                          │
                                             ┌────────────▼────────────┐
                                             │ dtw                     │
                                             │ deterministic temporal  │
                                             │ alignment for scalar,   │
                                             │ cost-matrix, and        │
                                             │ multivariate sequences  │
                                             └─────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ series                                     │
                                   │ exact grids / causal masks / index ranges  │
                                   │ standard library; no graph dependencies    │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ paa                                        │
                                   │ rational fractional boundaries / means     │
                                   │ standard library; no graph dependencies    │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ sax                                        │
                                   │ fractional PAA -> symbolic words           │
                                   │ Euclidean MINDIST; NOT a DTW lower bound   │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ bootstrap                                  │
                                   │Moving/Circular/Stationary index ensembles  │
                                   │versioned RNG / exact mapping / partial rows│
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ acf                                        │
                                   │ fixed-window scaled direct sample ACF      │
                                   │ common/lag-adjusted; not causal streaming  │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ pacf                                       │
                                   │ biased common-n / Durbin-Levinson          │
                                   │ partial lag correlation / V_m by order     │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ arima                                      │
                                   │ conditional ARMA on d-th differences       │
                                   │ Schur roots / anchored level forecasts     │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ lstm                                       │
                                   │ frozen IFGO recurrent inference            │
                                   │ masked batches / detached chunk State      │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ rsi                                        │
                                   │ causal Wilder changes / ready mask         │
                                   │ zero-allocation Push / versioned snapshot  │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ macd                                       │
                                   │ SMA-seeded Appel EMA, signed MACD features │
                                   │ 3 aligned masks / atomic Push / checkpoint │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ mfi                                        │
                                   │ typical-price flow * volume, P transitions │
                                   │ compensated ring / masks / snapshots       │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ obv                                        │
                                   │ close-signed cumulative traded base volume │
                                   │ Neumaier correction / atomic stream replay │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ ad                                         │
                                   │ close-location-weighted base volume        │
                                   │ Neumaier / first bar / atomic replay       │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ stats                                      │
                                   │ stable Welford/Chan moments, fitted scaling│
                                   │ matrix.Matrix input, immutable inference   │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ ols                                        │
                                   │ pivoted QR, full rank, frozen prediction   │
                                   │ physical beta/intercept; train-only scale  │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ ridge                                      │
                                   │ positive Lambda / equilibrated augmented QR│
                                   │ wide + collinear, physical beta/intercept  │
                                   └────────────────────────────────────────────┘

                                   ┌────────────────────────────────────────────┐
                                   │ pca                                        │
                                   │ sample covariance -> verified eigenbasis   │
                                   │ train-only scaling / frozen reconstruction │
                                   └────────────────────────────────────────────┘

Composition rule:
 - core owns topology.
 - matrix owns dense graph/numeric artifacts.
 - dtw owns temporal alignment artifacts.
 - series owns exact-grid validation and causal array artifacts.
 - paa owns fractional step-function compression and exact rational boundaries.
 - sax owns compatible symbolic encoding and Euclidean lower-bound dissimilarity.
 - stats owns stable scalar moments and immutable matrix feature scaling.
 - bootstrap owns versioned block-index replicates; values and inference stay with callers.
 - rng owns version-pinned path-specific pseudo-random streams; it does not schedule trials.
 - gbm owns one exact-in-distribution geometric Brownian motion trajectory on
  explicitly unit-matched elapsed-time grids; stochastic draws come from rng,
  payoffs and calibration come from the caller, and ensemble reduction comes
  from montecarlo.
 - lstm owns immutable IFGO single-layer inference, checked batch-first
   tensors, local valid lengths, exact-zero padding and model-identified
   Hidden/Cell continuation. It neither trains coefficients nor outputs a
   calibrated task-specific prediction.
 - arima owns fixed-order conditional Gaussian mean estimation, strict Schur AR/MA
   admissibility, original-level anchor recovery and linear-memory marginal
   innovation variances; cadence, order choice, holdout review and release are external.
 - gjrgarch owns deterministic stationary conditional-return variance, strictly
  lagged posterior filtering, expected-variance horizons and constrained
  Gaussian QMLE. It estimates no price path, risk quantile or trading policy.
 - jumpdiffusion owns an explicitly selected Merton compound-Poisson model
  on the same GBM grid and parameters: lambda=0 delegates to gbm, while
  positive lambda uses bounded discrete Poisson inversion/PTRS and a
  compensated or uncompensated jump-drift convention. It does not calibrate
  jump risk, predict crashes, or choose a probability measure.
 - montecarlo owns the bounded parallel driver, canonical path-ID reduction,
  optional exact quantiles and prefix-only interruption semantics; it does not model a process.
 - acf owns fixed-window sample autocorrelation, explicit estimator conventions,
  and detached coefficients with exact pair counts; it is not a causal transform.
 - pacf owns fixed-window partial autocorrelation from common-n covariances
  and Durbin-Levinson prediction variances; it neither fits AR models nor
  publishes significance bands, and it is not a causal stream.
 - rsi owns causal Wilder gain/loss smoothing, validity masks and single-owner
  checkpointed streams; timestamps and market decisions remain with callers.
 - mfi owns volume-weighted typical-price transitions, compensated rolling
  sums, the MFI validity mask and the detached ring checkpoint; market feed
  timestamps, base-unit conversion and security remain caller responsibilities.
 - obv owns cumulative signed-volume calculation from successive closes, a
  first-candle zero origin and two-component compensated state; source offsets,
  volume-unit conversion and checkpoint authentication remain external.
 - ad owns current-candle close-location-weighted volume, first-bar inclusion
  and a two-component compensated state without previous-close history; the
  caller owns timestamp integrity, replay offsets and checkpoint authentication.
 - macd owns SMA-seeded fast/slow price EMAs, signed Signal/Histogram, aligned
  readiness masks and transactional streaming; feed identity and decisions are external.
 - ols owns full-rank regression artifacts, intercept policy and frozen prediction.
 - ridge owns positive-Lambda regression artifacts, unpenalized intercept, and QR equilibration.
 - pca owns covariance-based orthogonal factors, strict spectrum admission,
  and immutable train-only reconstruction coordinates.
 - hmm owns immutable hidden-state probability models, causal filtering,
  retrospective smoothing, joint-path decoding, and multi-sequence EM.
 - mst / flow / tsp own optimization artifacts.
 - builder and gridgraph produce reproducible input worlds.
```

---

The stochastic branch of the architecture is deliberately separate from graph traversal:

```text
 application Sampler                  rng.Factory
          |                     Stream(seed,pathID)
          |                             |
          +-------- bounded workers ----+
                           |
               indexed path outcomes
                           |
             single ordered coordinator
                   /             \
             stats.Moments    optional Dense matrix
                   \             /
                 *montecarlo.Result
                           |
                    application policy
```


A separate **conditional-variance branch** consumes historical returns, not
GBM/Merton price-path RNG draws:

```text
positive prices --series.LogReturns/Valid--> ordered returns
                                               |
                     frozen explicit Parameters/Initial
                                               |
                                     gjrgarch.New / Fit
                                               |
                              immutable Model + causal State
                                               |
                     ForecastNextVariance / ForecastVariance
                                               |
                       application-owned holdout/risk review
```

`gjrgarch.Fit` can return an unfinished candidate with `ErrNotConverged`;
neither an immutable model identity nor a decreasing training objective
approves risk-model deployment. Details follow in the
[GJR-GARCH section](#gjr-garch-stationary-causal-volatility-and-asymmetric-downside-response).

## Package roles and strengths

| Package         | What it gives you                                                                                                                              | Concrete strength                                                                                                          | Typical scenario                                                                                               |
|:----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|
| `core`          | Thread-safe in-memory graph with deterministic `Vertices`, `Edges`, and `Neighbors`; explicit directed/weighted/loop/multi/mixed capabilities. | Prevents unstable map order and invalid topology from leaking into algorithms.                                             | Service maps, routing graphs, dependency graphs, test fixtures.                                                |
| `bfs`           | Unweighted shortest-hop traversal, path reconstruction, weak components, hooks, filters, partial results.                                      | Distinguishes discovery (`Visited`) from processing (`Order`) and preserves useful partial state.                          | Blast radius, dependency waves, crawler frontiers, weak island discovery.                                      |
| `dfs`           | DFS forest, post-order, cycle witnesses, topological sorting, hooks, filters, cancellation.                                                    | Makes finish order explicit and returns deterministic cycle witnesses instead of unstable recursion artifacts.             | Release plans, DAG validation, lock/resource cycle auditing.                                                   |
| `dijkstra`      | Full-SSSP Dijkstra plus target Dijkstra, bidirectional Dijkstra, consistent A*, balanced bidirectional A*.                                     | Deterministic original `EdgeIDs`, detached `PathResult`, exact target cutoff/keys and certified stopping.                  | Weighted routing, mixed multigraph logistics, grids, network failover.                                         |
| `mst`           | Prim/Kruskal exact MST/MSF; original-edge KMB graph Steiner approximation; closed/fixed-end tree-edge covering walks.                          | Shared deterministic Kruskal/DSU selection, original IDs, independent tree validation and distinct network/travel costs.   | Fiber/chiplet routing backbones, terminal-only connections, inspection walks.                                  |
| `flow`          | Ford-Fulkerson, Edmonds-Karp, and Dinic over `core.Graph`, returning max flow and residual graph.                                              | Preserves residual semantics and supports algorithm selection from simple to high-throughput.                              | Capacity planning, traffic engineering, assignment models, min-cut analysis.                                   |
| `dtw`           | General Dynamic Time Warping, bounded classification, certified exact top-K fixed-length squared-scalar window search.                         | Aligns sequences that share a pattern but differ in speed or local timing.                                                 | Sensors, gestures, audio contours, time-series similarity.                                                     |
| `series`        | Exact signed-nanosecond grid checks, causal lag/difference/returns, and complete index windows.                                                | Detached validity masks, checked integer intervals, and explicit whole-slot gaps without interpolation.                    | Telemetry preparation, causal features, window scheduling, signal-quality gates.                               |
| `paa`           | Fractional PAA with exact, unsimplified rational segment geometry and a correctly rounded final binary64 mean.                                 | Never discards remainder samples or fabricates zeros; results are detached.                                                | SAX preparation, telemetry compression, fixed-dimensional sequence features.                                   |
| `sax`           | Fixed-length symbolic words, pinned Gaussian/custom thresholds, immutable encoders and Euclidean MINDIST.                                      | Right-bin equality, SHA-256 policy identity, detached results; no false DTW pruning claim.                                 | Telemetry signatures, approximate candidate ranking, shape-change screening.                                   |
| `tfsax`         | Research-only exact-DTW versus prepared SAX ranking, deterministic stress families and temporal provenance gates.                              | Explicit `Complete`/`Exact`, strict recall/NA, frozen in-memory plan; no production encoder or certified LB.               | Scientific retrieval audits, independently seeded synthetic comparison, missed-best detection.                 |
| `ols`           | Full-column-rank least-squares regression, immutable original-unit coefficients and QR diagnostics.                                            | Explicit intercept/scale-only origin policies, rank refusal, training SSE, no heldout refit.                               | CAPM factors, sensor calibration, energy-load forecasting.                                                     |
| `ridge`         | Tikhonov L2 regression with mandatory Lambda>0 and immutable physical-unit inference.                                                          | Stable balanced augmented Householder QR, p>n, duplicated predictors, unpenalized intercept and checked objective.         | Correlated multi-factor hedges, wide optical sensors, seasonal AR(12).                                         |
| `rng`           | Version-pinned SHA-256/PCG-DXSM128/64 streams, strict uniform midpoints, cached Box-Muller normals and batch facades.                          | Immutable factory, logical path isolation, no hot-loop allocation and exact raw consumption.                               | Monte Carlo paths, risk simulation, sensor fault injection, repeatable stress experiments.                     |
| `montecarlo`    | Independent multidimensional stochastic experiments, bounded worker scheduling, ordered Welford moments and type-7 quantiles.                  | Bitwise worker-count parity on complete deterministic runs, checked optional-buffer admission and strict partial prefixes. | Option/risk valuation, structural fatigue screening, microgrid resilience, repeatable fault experiments.       |
| `gbm`           | Exact-in-distribution GBM price paths on caller-specified irregular grids, with explicit years/days/seconds.                                   | Avoids spurious `S*exp`/`sigma²` overflows; detached prefix/errors, zero RNG draws at sigma zero.                          | European-option paths, synthetic tick stress, FX-liability scenarios and deterministic escalation.             |
| `gjrgarch`      | Immutable GJR-GARCH(1,1) causal variance, forecast and Gaussian QMLE with explicit optimizer diagnostics.                                      | Causal lag indexing, sign-dependent shock, stationarity margin and deterministic KKT/BFGS.                                 | Next-day equity risk, collateral monitoring, model-risk audit and volatility term structure.                   |
| `arima`         | Fixed-order conditional Gaussian ARIMA fit and anchored level forecasts.                                                                       | Schur-stable AR/MA, deterministic optimizer, historical anchors, `O(H)` horizon variance.                                  | GDP index planning, backlog staffing, liquidity spreads and industrial energy scheduling.                      |
| `lstm`          | Frozen single-layer IFGO LSTM inference on finite batch-first tensors with optional zero-padded outputs.                                       | Private copied weights, canonical ModelID, exact overflow gating, detached Hidden/Cell and full/chunk replay.              | Fleet telemetry embeddings, order-book features and replayable sensor packets (externally validated model).    |
| `jumpdiffusion` | Merton compound-Poisson lognormal jump-diffusion on explicit GBM grids.                                                                        | `lambda=0` GBM parity, bounded discrete PTRS, drift compensator, partial prefix with errors.                               | Scenario-driven collateral gap stress, jump-sensitive European options, credit event sensitivity.              |
| `bootstrap`     | Reproducible Moving/Circular/Stationary index resampling for dependent series with per-replicate streams and exact bounded integers.           | Detached complete-row results, memory admission and cancellation without interrupted-row publication.                      | Portfolio risk sensitivity, paired serial regression, telemetry alarm calibration.                             |
| `acf`           | Direct fixed-window sample autocorrelation with two explicit denominator conventions and raw/centered origins.                                 | Scaled compensated products, no lag-specific recentering or clipping; caller-owned lag counts.                             | Residual audits, compressor cycles, vibration periodicity, signal pitch.                                       |
| `pacf`          | Finite-window partial autocorrelation and order-by-order prediction variance from biased common-n covariance.                                  | Stable centering, two coefficient rows, detached curves and conservative numeric refusal.                                  | Candidate AR-order screening, compressor residuals, sensor memory, order-flow diagnostics.                     |
| `rsi`           | Deterministic Wilder RSI over finite-positive prices: batch mask, streaming Push and snapshot continuation.                                    | One shared scalar recurrence, explicit warmup, atomic failed Push, detached checkpoints, no signal thresholds.             | Portfolio momentum-factor extraction, feed monitoring, authenticated hot-failover analytics.                   |
| `macd`          | SMA-seeded fast/slow EMAs, signed Line/Signal/Histogram, batch and single-owner streaming.                                                     | Shared kernel, separate warmup masks, bitwise batch/stream parity and detached state snapshots.                            | Trend-feature pipelines, bar-feed monitoring, replayable analytics failover.                                   |
| `mfi`           | Volume-weighted typical-price Money Flow Index over a fixed transition window.                                                                 | Neumaier ring, deterministic rebase, zero-volume retention, atomic Push and detached snapshots.                            | OHLCV research features, market-feed reconciliation and checkpointed streaming analytics.                      |
| `obv`           | Cumulative On-Balance Volume based on successive closing prices.                                                                               | First-bar zero, compensated signed volumes, transactional zero-allocation Push and deterministic snapshots.                | Feed-health audits, volume divergence monitoring and replayable multi-instrument screening.                    |
| `ad`            | Chaikin close-location-weighted accumulation of current-candle base volume.                                                                    | First-bar contribution, stable bounded multiplier, Neumaier cancellation, error-atomic streaming and two-float snapshots.  | Intraday volume-pressure review, vendor OHLC quality gates, multi-asset research features and failover replay. |
| `hmm`           | Typed finite-state categorical and diagonal-Gaussian probability models; causal filtering, smoothing, Viterbi, multi-sequence Baum-Welch.      | Explicit log-domain evidence, immutable model IDs, no future leakage in filtering, no cross-session transition counts.     | Live stress monitoring, machine regimes, independent clickstream sessions.                                     |
| `pca`           | Immutable covariance-based principal components, train-only projection and original-unit reconstruction.                                       | Bounded Jacobi, independent PSD/eigenpair checks, detached loadings and no holdout recentering.                            | Yield-curve factors, turbine residual screening, multichannel telemetry compression.                           |
| `stats`         | Stable scalar mean/variance, mergeable Welford/Chan states, train-only fitted feature standardization.                                         | Exact binary-input mean and scaled second moments, detached fitted metadata, immutable inference without refit.            | Telemetry aggregation, distributed metrics, ML feature preparation, anomaly-score calibration.                 |
| `gridgraph`     | 2D lattice graph generation with neighborhood and obstacle-style workflows.                                                                    | Avoids manual wiring for pathfinding maps and teaching graphs.                                                             | Grid routing, maps, demos, benchmark fixtures.                                                                 |
| `matrix`        | Dense graph algebra, scaled covariance/Pearson/L2, reusable pivoted LU/QR/Cholesky solves, and rectangular economy SVD.                        | True residuals, immutable factors, independent SVD column scales, honest nonconvergence and detached result modes.         | Graph features, calibration, RF measurements, sensor and return-matrix compression.                            |
| `tsp`           | Tour optimization toolkit: practical approximation/local search/exact small-instance strategies.                                               | Bridges exactness and practicality for route planning.                                                                     | Delivery tours, inspection routes, metric routing experiments.                                                 |
| `builder`       | Deterministic graph and data generators.                                                                                                       | Produces reproducible examples, tests, and benchmarks.                                                                     | Golden tests, performance fixtures, tutorials.                                                                 |


### Series: grid validation before causal features

Use `series` when order, representable timestamp intervals, missing grid slots,
or future-free transformations matter. It is not an alternative DTW alignment
engine: `series` preserves each supplied sample index; `dtw` may align different
indices under an explicit warping policy.

```go
package main

import (
	"errors"
	"fmt"

	"github.com/lvlath/go/series"
)

func main() {
	timesNS := []int64{0, 10, 30}
	observations := []float64{2, 4, 8}

	grid, err := series.Validate(timesNS, observations, series.GridOptions{
		CadenceNS: 10,
		AllowGaps: true,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	changes, err := series.Difference(observations, 1)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("gaps:", grid.Gaps)
	fmt.Println("differences:", changes.Values, changes.Valid)

	_, err = series.Validate(timesNS, observations, series.GridOptions{CadenceNS: 10})
	fmt.Println("strict grid rejected:", errors.Is(err, series.ErrGap))
}
```

Output:

```text
gaps: [{1 1}]
differences: [0 2 4] [false true true]
strict grid rejected: true
```

A missing timestamp is **not** inserted into `observations`. The difference
at the last supplied index crosses that gap; whether to allow it is an
application decision. Invalid warmup zeros are distinguished by `Valid`,
not by a NaN sentinel. See [`docs/SERIES.md`](docs/SERIES.md) for exact error
precedence, numerical stability, allocation bounds, and ownership guarantees.


### PAA: fractional compression without losing observations

`github.com/lvlath/go/paa` maps finite uniformly sampled `[]float64` observations into a chosen number of **equal-duration**, not floor-divided, output segments. Its rational `Segment` boundaries keep partial contributions at every fractional split. The result owns separate `Values` and `Segments` slices and records the original sample count.

| Need                                                    | API                                                | Why                                                                           |
|:--------------------------------------------------------|:---------------------------------------------------|:------------------------------------------------------------------------------|
| Retain means, exact segment geometry and length         | `paa.Transform(values, segments)`                  | Canonical `*paa.Result` for safe composition with SAX and lower-bound proofs. |
| Reduce one half-open interval without unchecked slicing | `paa.TransformRange(values, start, end, segments)` | The returned boundaries are **local to the selected interval**.               |
| Produce a value-only feature vector                     | `paa.Means(values, segments)`                      | Projects only means; uses the same canonical kernel, but discards geometry.   |

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/paa"
)

func main() {
	result, err := paa.Transform([]float64{0, 2, 4, 6, 8}, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("PAA: %.1f, %.1f; boundary: %d/%d\n", result.Values[0], result.Values[1], result.Segments[0].RightNumerator, result.Segments[0].Denominator)
	// Output: PAA: 1.6, 6.4; boundary: 5/2
}
```

This is not a calendar-aware resampler: validate timestamps and reject/handle missing grid cells using `series` first. PAA requires `1<=segments<=n`, rejects nonfinite observations, publishes no partial results, and preserves all source interval measure. The signed-zero identity `segments==n` is a detached bitwise copy; all other means use exact fixed-point weighted accumulation followed by nearest-even binary64 rounding. No new Options interface or `matrix` dependency is necessary.

See [`docs/PAA.md`](docs/PAA.md), [`paa/doc.go`](paa/doc.go), and the runnable `go run ./examples/paa_window` scenario.

### SAX: compact symbolic signatures with an explicit Euclidean bound

`github.com/lvlath/go/sax` turns finite, uniformly sampled scalar windows into **fixed-length, fixed-alphabet SAX words**. Unlike PAA, which preserves the averaged numeric values, SAX also quantizes them into unsigned bin indexes and attaches a content-derived `EncodingID` so words are compared only under compatible mathematical policies. The package reuses the **same exact fractional PAA kernel**; it does not drop remainder samples or create a second averaging implementation.

| Need                              | Public surface                                                       | Why it matters                                                                                                       |
|:----------------------------------|:---------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------|
| One normalized window             | `sax.EncodeGaussian(values, segments, alphabet)`                     | Window-local population Z-score, pinned Gaussian quantiles; constant input is rejected by default.                   |
| Preserve absolute units           | `sax.EncodeRaw(values, segments, alphabet)`                          | Identity transform; Gaussian bins are used unchanged, without any claim that the input is Gaussian.                  |
| Calibrate physical thresholds     | `sax.EncodeRawWithBreakpoints(values, segments, breakpoints, label)` | Explicit custom bins with right-boundary equality; label is audit-only.                                              |
| Reuse a fixed policy              | `sax.NewEncoder(opts)` → `encoder.Encode(values)`                    | Immutable copied thresholds and SHA-256 policy identity; detached words per call.                                    |
| Compare compatible symbolic words | `encoder.Mindist(left, right)`                                       | Euclidean lower-bound dissimilarity with fractional factor `sqrt(n/w)`, after independent shape and identity checks. |
| Occasional pairwise bound         | `sax.MindistValues(left, right, opts)`                               | Canonical encode-encode-compare wrapper; no index or recall guarantee.                                               |

For equal-length, equally segmented, compatibly transformed windows, the mathematical invariant in **exact arithmetic** is:

$$ \operatorname{MINDIST}(\hat Q,\hat C)\le D_{\mathrm{PAA}}(\bar Q,\bar C)\le D_{\mathrm{Euclidean}}(Q^*,C^*). $$

![SAX Euclidean lower-bound inequality](https://latex.codecogs.com/svg.image?%5Cmathrm%7BMINDIST%7D%5Cle%20D_%7B%5Cmathrm%7BPAA%7D%7D%5Cle%20D_%7B%5Cmathrm%7BEuclidean%7D%7D%28Q%5E%2A%2CC%5E%2A%29)


The last distance is measured in the **selected transformed space**: original values for `Raw`, separately window-normalized values for `ZPopulation`. The implementation uses float64 and does not provide a directed-rounding numerical pruning certificate. MINDIST is **not a metric on symbol strings**; adjacent but distinct bins can have zero gap.

A runnable, auditable CPU saturation example uses custom percentage boundaries rather than pretending raw percentages follow a Gaussian distribution:

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/sax"
)

func main() {
	baseline := []float64{35, 35, 35, 35, 35, 35, 35, 35}
	saturated := []float64{35, 35, 35, 35, 90, 90, 90, 90}
	opts := sax.Options{
		Segments:      4,
		Alphabet:      5,
		Normalization: sax.Raw,
		Breakpoints:   []float64{20, 40, 60, 80},
		BreakpointID:  "cpu-percent-v1",
	}
	encoder, err := sax.NewEncoder(opts)
	if err != nil {
		fmt.Println(err)
		return
	}
	normal, err := encoder.Encode(baseline)
	if err != nil {
		fmt.Println(err)
		return
	}
	abnormal, err := encoder.Encode(saturated)
	if err != nil {
		fmt.Println(err)
		return
	}
	bound, err := encoder.Mindist(normal, abnormal)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("words: %v vs %v\n", normal.Symbols, abnormal.Symbols)
	fmt.Printf("Euclidean lower bound: %.0f\n", bound)
	// Output:
	// words: [1 1 1 1] vs [1 1 4 4]
	// Euclidean lower bound: 80
}
```

**Critical DTW boundary:** Euclidean SAX MINDIST must **never** be used to prune arbitrary DTW candidates. For `Q=[0,0,1]` and `C=[0,1,1]` with `[0.25,0.75]` thresholds, MINDIST is `0.5` while unconstrained DTW is `0`. SAX can rank an **approximate shortlist** for later `dtw.Align` evaluation, but global DTW nearest-neighbor correctness is not certified; compare retrieval `Recall@K` against exhaustive DTW if it matters. For an exact fixed-length DTW search use the separately certified `dtw.Search` contract.

The immutable Encoder is safe for concurrent inference with stable inputs and Results; `Result.Symbols` and optional `Result.PAA` are detached. Constant Z windows use explicit `RejectConstant` or `ZeroWord` policy (which makes **normalized coordinates**, not necessarily symbol indexes, zero). A matching `EncodingID` proves mathematical policy compatibility, not authenticity of externally mutable Result data.

See [`docs/SAX.md`](docs/SAX.md) for the full fractional Jensen proof, symbol-bin convention, encoding hash serialization, error precedence, numerical limitations, performance and independent regression oracles. Five runnable synthetic operational examples are provided in [`sax/example_test.go`](sax/example_test.go).

### TFSAX: exact-DTW/SAX retrieval research without inventing an encoder

`github.com/lvlath/go/tfsax` is a **research-only** package. It asks whether a future rigorously specified symbolic representation can outperform the existing SAX filter at a precommitted strict-ID recall target. The name **TFSAX does not currently define a mathematical encoder**. The package therefore provides *no* production `Fit`/`Encode`, no certified symbolic DTW pruning, and no claimed empirical victory. `Run` correctly rejects a missing four-way experiment instead of returning fictitious measurements.

| Research question                                                            | Current Go surface                                              | Guarantee                                                                                     |
|:-----------------------------------------------------------------------------|:----------------------------------------------------------------|:----------------------------------------------------------------------------------------------|
| What are the full-universe nearest neighbors under the accepted scalar DTW?  | `ScanExact`, `RunExactReference`                                | Exact full scan, fixed endpoints, absolute-cost `dtw.Align`, sorted `(DTW, CandidateID)`      |
| Can classical SAX rank a smaller set of candidates for exact DTW refinement? | `PrepareSAXIndex` + `SAXIndex.Search`, `RunAvailableComparison` | Heuristic TopM, with `Diagnostics.Exact=false`; empirical Recall@K only                       |
| Can an unproven lower bound certify DTW exclusions?                          | `ScanLBConservative`, `LBKeoghAbsolute`                         | Diagnostic only; exact pruning refuses with `ErrUncertifiedBound`                             |
| Is the proposed TFSAX algorithm defined well enough to deploy?               | `Definition`, `ValidateDefinition`, `Run`                       | **NOT ADMITTED** without formal mathematics, independent reference and full four-arm evidence |

**Use the library entirely in Go:** `Protocol`, `Thresholds`, `Definition` and `Experiment` are typed Go values. Neither `protocol.json` nor `definition.json` is loaded by a search or comparison call. Both files are *optional versioned reproducibility records*, not foreign-language dependencies; the `json:"..."` tags affect only serialization by Go's standard library and preserve stable historical field names/hashes. Deleting or changing this record ABI would require a separately versioned migration, not a silent cleanup.

```go
// Run inside a function with ctx context.Context and ordinary Go imports.
corpus, err := tfsax.GenerateFamily(tfsax.FamilyMotif, 701, 24, 24)
if err != nil { return err }
independent, err := tfsax.GenerateFamily(tfsax.FamilyMotif, 1701, 1, 24)
if err != nil { return err }
query := tfsax.Query{Values: independent[0].Values, TopK: 3}
exact, err := tfsax.ScanExact(ctx, corpus, query, 4)
if err != nil { return err }
approx, err := tfsax.ScanSAXWithOptions(ctx, corpus, query,
    tfsax.SAXSearchOptions{Window: 4, Shortlist: 10, Segments: 6, Alphabet: 8})
if err != nil { return err }
recall := tfsax.RecallAtK(approx.Candidates, exact.Candidates)
// recall is an empirical observation; approx.Diagnostics.Exact remains false.
_ = recall
```

The snippet assumes the stated imports, `ctx context.Context` and a function returning `error`. The [complete runnable, no-JSON scenario](docs/TFSAX.md#41-fully-in-memory-no-files-no-marshaling-executable-go-example) demonstrates the whole experiment; run the file-free [`examples/tfsax_go_only`](examples/tfsax_go_only/main.go) scenario with `go run ./examples/tfsax_go_only`, and see [`tfsax/example_test.go`](tfsax/example_test.go) for additional stress and temporal-leakage tests. All representative demonstrations use deterministic **synthetic** observations, not recorded exchange prices or deployed plant alarm thresholds.

A valid study freezes raw Train/Calibration/Test intervals before creating overlapping windows, checks full supports and label horizons using `ValidateWindowRecords`, retains independent query seeds and exact-bit corpus/query digests, and reports `RecallAtK`, missed-best and the **worst** family. `Complete=true` only reports procedural completion; `Diagnostics.Exact=true` reports mathematical global exactness. Empty exact targets yield NA, not fabricated perfect recall. Timing and memory results must be measured separately for index build, cached queries and real retained allocations; `SAXIndex.IndexBytes()` counts only ID+symbol payload.

**Research admission thresholds are requirements, not achievements:** zero certified false exclusions; Recall@K at least 0.99 per predeclared non-adversarial family; missed-best rate at most 0.01; at least 2× median speedup with at most 2× **measured** index memory compared with SAX. Until independent TFSAX definition, safe bound and four-arm results exist, the decision remains **NOT ADMITTED**. See [`docs/TFSAX.md`](docs/TFSAX.md), [`tfsax/doc.go`](tfsax/doc.go), and the [TFSAX Tutorial](docs/TUTORIAL.md#11v-tfsax-research-gate-exact-dtw-truth-sax-ranking-and-reproducible-evidence).

### Stats: stable reductions and train-only feature calibration

Use `github.com/lvlath/go/stats` when the application needs **numerically stable one-dimensional moments**, deterministic merging of separately processed partitions, or an **immutable feature standardizer fitted only on training observations**.

| Need                                 | API                                                         | Contract highlight                                               |
|:-------------------------------------|:------------------------------------------------------------|:-----------------------------------------------------------------|
| Stable population/sample statistics  | `stats.SummarizePopulation` / `stats.SummarizeSample`       | Compensated Welford `M2` and exact binary mean sum               |
| Streaming and partitioned statistics | `stats.Moments.Update` / `Merge` / `Summary`                | Error-atomic mutation; scaled Chan merge; checked `uint64` count |
| Train-only feature normalization     | `stats.FitStandardizer`                                     | One fitted mean/scale per matrix column, immutable model         |
| Explicit population/sample fitting   | `stats.FitPopulationStandardizer` / `FitSampleStandardizer` | Fixed DDOF, caller-selected constant policy                      |
| Heldout inference and reconstruction | `Model.Transform` / `InverseTransform`                      | No refit, detached dense outputs, stable affine arithmetic       |

The two numeric layers are deliberate: `Moments` accumulates a **compensated scaled** approximation of the second central moment, while a bounded exact binary sum protects the final mean against arbitrary cancellation of finite binary64 inputs. Therefore an intermediate `M2` outside the `float64` exponent range does not automatically make a *final* variance unrepresentable. Variance remains approximate; no universal error bound is claimed.

This short example demonstrates two independent contracts: sample variance on a large offset and the difference between a constant training feature and a novel heldout value.

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/stats"
)

func main() {
	batch := []float64{1e12 + 1, 1e12 + 2, 1e12 + 3}
	result, err := stats.SummarizeSample(batch)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("sample variance: %.0f\n", result.Variance)

	training, err := matrix.NewDense(3, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	for row := 0; row < 3; row++ {
		if err := training.Set(row, 0, 5); err != nil {
			fmt.Println(err)
			return
		}
	}
	fit, err := stats.FitPopulationStandardizer(training, stats.UnitScaleConstant)
	if err != nil {
		fmt.Println(err)
		return
	}
	heldout, err := matrix.NewDense(1, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := heldout.Set(0, 0, 7); err != nil {
		fmt.Println(err)
		return
	}
	standardized, err := fit.Model.Transform(heldout)
	if err != nil {
		fmt.Println(err)
		return
	}
	novel, err := standardized.At(0, 0)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("heldout constant-feature deviation: %.0f\n", novel)

	// Output:
	// sample variance: 1
	// heldout constant-feature deviation: 2
}
```

`ScaleOptions{}` defaults to population variance and `RejectConstant`; `UnitScaleConstant` must be selected explicitly when retaining a constant feature. Fitted parameter slices (`Means`, `Scales`, `Constant`) are detached from model internals; the model may be used concurrently for inference as long as input matrices remain stable for the call. The mutable `Moments` accumulator is **not** concurrent-safe without external synchronization.

For the full derivations, overflow cases, error precedence, ownership, ASCII diagrams, and integration with `series` masks or multivariate DTW, read [`docs/STATS.md`](docs/STATS.md) and [`stats/doc.go`](stats/doc.go). The five canonical executable examples live in [`stats/example_test.go`](stats/example_test.go).

### Matrix: stable cross-feature statistics and reusable linear solvers

`github.com/lvlath/go/matrix` is both the graph-to-dense representation layer and the numerical layer for full-rank linear systems. Its finite-only statistical kernels are distinct from graph distance matrices that use `+Inf` as an absence marker. For **sample** statistics, rows are observations and columns are features; `Covariance`/`Correlation` require at least two rows, even when the matrix has no feature columns.

| Need                                              | Public API                              | Essential contract                                                                  |
|:--------------------------------------------------|:----------------------------------------|:------------------------------------------------------------------------------------|
| Demean many feature columns                       | `matrix.CenterColumns`                  | Scale-aware mean; detached nonempty result; zero-element no-op aliases input        |
| Compute sample cross-feature variability          | `matrix.Covariance`                     | Divides by `n-1` before restoring large scales; bit-identical symmetric cells       |
| Compare correlated features across extreme scales | `matrix.Correlation`                    | Direct Pearson; constant columns have zero row and diagonal                         |
| Unit-length finite feature rows                   | `matrix.NormalizeRowsL2`                | Scaled sum of squares; reports original row norms; true overflow returns error      |
| Solve a square full-rank system                   | `matrix.FactorLU`                       | Deterministic row-pivoted `PA=LU`; reusable and detached                            |
| Fit a tall full-column-rank model                 | `matrix.FactorQR` / `SolveLeastSquares` | Column-pivoted Householder; no normal-equation or minimum-norm fallback             |
| Solve a symmetric SPD system                      | `matrix.FactorCholesky`                 | Explicit symmetry tolerance; positive definiteness checked during factorization     |
| Decompose a rectangular measurement matrix        | `matrix.SVDDefault` / `SVD`             | One-sided Jacobi economy modes; no Gram normal equations; diagnostic nonconvergence |

For every successful new factor solve, `SolveResult` includes a detached solution, numerical rank, one original-coordinate L2 residual per right-hand side, and a **measured** 1-norm condition diagnostic. LU and Cholesky report `kappa_1(A)` of their accepted matrices; QR reports `kappa_1(R)`. Building this diagnostic uses basis-vector solves and is not free. Reusing a factor for successive RHS batches avoids repeatedly paying the factorization cost.

The three current **solve** domains are intentionally strict. QR rejects wide and rank-deficient inputs; Cholesky rejects matrices outside the accepted symmetry/SPD domain; LU reports numerical rank loss. None silently switches to SVD, ridge, diagonal jitter or another objective. Separately, `matrix.SVD` is implemented for rectangular decomposition; it does not return a least-squares solution or activate `pca.ThinSVD`. Existing `LUDecompose`/`QRDecompose` are **legacy unpivoted and square-only** interfaces with distinct output conventions.

```go
package main

import (
	"context"
	"fmt"

	"github.com/lvlath/go/matrix"
)

func main() {
	design, err := matrix.NewDense(3, 2)
	if err != nil {
		fmt.Println("design:", err)
		return
	}
	if err := design.Fill([]float64{1, 0, 1, 1, 1, 2}); err != nil {
		fmt.Println("design data:", err)
		return
	}
	rhs, err := matrix.NewDense(3, 1)
	if err != nil {
		fmt.Println("rhs:", err)
		return
	}
	if err := rhs.Fill([]float64{8, 11, 14}); err != nil {
		fmt.Println("rhs data:", err)
		return
	}
	fit, err := matrix.SolveLeastSquares(context.Background(), design, rhs, matrix.FactorOptions{})
	if err != nil {
		fmt.Println("least squares:", err)
		return
	}
	fmt.Printf("unknowns=%d, rank=%d, rhs=%d\n", fit.Solution.Rows(), fit.Rank, len(fit.ResidualNorms))

	// Output: unknowns=2, rank=2, rhs=1
}
```

For larger runnable scenarios with checked errors, see [`docs/MATRICES.md`](docs/MATRICES.md#108-go-usage-examples) and [`matrix/example_test.go`](matrix/example_test.go). For why exact binary-input residuals can be expensive, and where `ErrNaNInf`, `ErrOverflow`, `ErrRankDeficient`, `ErrAsymmetry`, and `ErrNotPositiveDefinite` differ, see the [solver reference](docs/MATRICES.md#1013-error-classification-and-decision-table).

#### Economy SVD: compression and singular-direction diagnostics

`matrix.SVD(ctx, observations, matrix.SVDOptions{})` directly decomposes a
finite `m x n` input into a detached **economy** `U[m,k]`, `SingularValues[k]`
and `VT[k,n]`, `k=min(m,n)`. The zero-valued options policy selects 100 full
Jacobi sweeps and normalized correlation tolerance `8*2^-52`. Every nonzero
working column retains its own binary exponent, protecting widely separated
independent singular directions from global-scale underflow. The kernel
does not construct `A^T*A` or a full large orthogonal basis.

| Need                               | API                                           | Interpretation                              |
|:-----------------------------------|:----------------------------------------------|:--------------------------------------------|
| First direct economy decomposition | `matrix.SVDDefault(ctx, A)`                   | Canonical default; all k modes are returned |
| Tune normalized correlation        | `matrix.SVDWithTolerance(ctx, A, t)`          | `t` is convergence, not rank truncation     |
| Bound full Jacobi sweeps           | `matrix.SVDWithMaxIterations(ctx, A, sweeps)` | Budget, not rotation count                  |
| Set both policies                  | `matrix.SVD(ctx, A, matrix.SVDOptions{...})`  | Exact canonical result/error contract       |

The `Rank` field counts relative numerical modes; it never truncates the
singular-value slice. A returned `ErrNotConverged` may carry a **diagnostic-only** result with `Converged=false`; never consume those factors as an
approved orthogonal model. Actual physical output overflow returns
`ErrOverflow` with nil. `IsNil()` is pointer presence, not convergence.

**Practical applications:** a twelve-window, six-channel turbine telemetry
experiment quantifies whether two latent directions retain its observed
energy; a five-pulse, eighteen-frequency RF calibration shows the `5 x 18`
economy dual; a 24-day, eight-asset synthetic return fixture demonstrates
three-mode energy retention without constructing normal equations. None of
these demonstrations establishes production alarm thresholds, trading
returns, or identifiability.

```sh
go test ./matrix -run 'ExampleSVD_' -count=1
```

The five executable Go examples are in [`matrix/example_test.go`](matrix/example_test.go).
For the transpose proof, numeric rank, scaled column correlations,
orthogonal completion, errors and honest limitations, read the canonical
[SVD reference](docs/MATRICES.md#1016-rectangular-economy-svd-direct-singular-mode-analysis)
and the [matrix FAQ](docs/FAQ_%26_TIPS.md#6-matrix-faq).

### OLS: stable factor regression with an immutable prediction model

`github.com/lvlath/go/ols` solves **finite, unweighted, full-column-rank
ordinary least squares** by delegating to the existing pivoted Householder
`matrix.SolveLeastSquares` kernel. It provides a regression-specific layer over
`matrix` (numerical solver) and `stats` (population feature scales), without
forming or inverting `XᵀX` or adding a hidden pseudoinverse fallback.

| Need                                      | Public API                                  | Model and failure law                                                     |
|:------------------------------------------|:--------------------------------------------|:--------------------------------------------------------------------------|
| Affine baseline with intercept            | `ols.FitDefault(ctx,x,y)`                   | Explicit ones column, centered predictors, returned physical coefficients |
| Affine model with uneven predictor scales | `ols.FitStandardized(ctx,x,y)`              | Train DDOF=0 scaling; heldout never refits                                |
| Physically mandatory zero origin          | `ols.FitThroughOrigin(ctx,x,y)`             | Exactly positive-zero intercept, no centering                             |
| Origin-constrained and uneven scales      | `ols.FitThroughOriginStandardized(ctx,x,y)` | Scale-only transform, never hidden affine correction                      |
| Custom relative QR rank policy            | `ols.Fit(ctx,x,y,ols.Options{...})`         | Full-rank gate; invalid option/rank/overflow have distinct sentinels      |

A successful `*ols.FitResult` contains `Model`, `Rank`, `ResidualSumSquares`
(training **sum**, not mean squared error) and `ConditionEstimate` (the QR
factor's `kappa_1(R)`, not a statistical confidence measure). `Model.Coefficients()`
returns detached slopes in original feature units and an intercept scalar;
`Model.Predict(ctx,future)` uses those **frozen** parameters without learning
new means or variances. Independent concurrent predictions are safe with
stable input matrices. An `ols.FitResult` also implements the `core.Nilable`
`IsNil()` pointer-presence convention.

```go
package main

import (
    "context"
    "fmt"

    "github.com/lvlath/go/matrix"
    "github.com/lvlath/go/ols"
)

func main() {
    // Each training row is temperature (C) and production-shift count.
    training, err := matrix.NewDense(6, 2)
    if err != nil {
        fmt.Println("training:", err)
        return
    }
    if err = training.Fill([]float64{
        15, 2,
        22, 4,
        20, 3,
        28, 5,
        17, 6,
        30, 1,
    }); err != nil {
        fmt.Println("training data:", err)
        return
    }
    energyKWh := []float64{148, 175, 164, 192, 183, 163}

    fit, err := ols.FitStandardized(context.Background(), training, energyKWh)
    if err != nil {
        fmt.Println("regression:", err)
        return
    }

    future, err := matrix.NewDense(1, 2)
    if err != nil {
        fmt.Println("holdout:", err)
        return
    }
    if err = future.Fill([]float64{26, 3}); err != nil {
        fmt.Println("holdout data:", err)
        return
    }
    prediction, err := fit.Model.Predict(context.Background(), future)
    if err != nil {
        fmt.Println("prediction:", err)
        return
    }
    fmt.Printf("rank=%d, forecast=%.2f kWh\n", fit.Rank, prediction[0])
}
```

OLS **does not** compute p-values, confidence intervals, automatically remove
collinear factors, support weighted/Ridge regression, or guarantee heldout
accuracy. Finance and telemetry may have dependent residuals, so a low training
SSE is not a causal or IID-based inference claim. The five scenario-driven
executable examples include CAPM, industrial calibration, origin-constrained
charging, duplicate-feature rejection and chronological energy-demand holdout.
See [`docs/OLS.md`](docs/OLS.md), [`ols/doc.go`](ols/doc.go) and
[`ols/example_test.go`](ols/example_test.go).

### Ridge: positive-Lambda regularization of correlated and wide factors

`github.com/lvlath/go/ridge` solves a **different statistical objective** from OLS:

![Ridge loss with unpenalized intercept](https://latex.codecogs.com/svg.image?J%28b%2C%5Ctheta%29%3D%5Cfrac%7B1%7D%7B2n%7D%5Csum_%7Bi%3D1%7D%5En%28y_i-b-z_i%5E%5Ctop%5Ctheta%29%5E2%2B%5Cfrac%7B%5Clambda%7D%7B2%7D%5C%7C%5Ctheta%5C%7C_2%5E2%2C%5Cqquad%5Clambda%3E0.)

The intercept `b` is never penalized. `Standardize=true` uses `stats` population scales (`DDOF=0`) while the public `Model.Coefficients` always returns physical-unit coefficients. `NoIntercept=true` forbids centering, so an all-zero physical observation predicts exactly zero. Unlike OLS, `ridge` admits duplicated and very wide original feature sets (`p>n`) when augmented QR accepts their numerical rank.

| Need                                | Public API                                           | Mathematical policy                                  |
|-------------------------------------|------------------------------------------------------|------------------------------------------------------|
| Standard affine shrinkage           | `ridge.FitWithLambda(ctx,x,y,lambda)`                | Physical slopes penalized; free intercept            |
| Scaled correlated factors           | `ridge.FitStandardized(ctx,x,y,lambda)`              | Penalize optimization-coordinate slopes              |
| Physical zero-origin law            | `ridge.FitThroughOrigin(ctx,x,y,lambda)`             | No intercept or centering                            |
| Zero origin with population scaling | `ridge.FitThroughOriginStandardized(ctx,x,y,lambda)` | Scale-only, no implicit affine offset                |
| Custom rank policy                  | `ridge.Fit(ctx,x,y,ridge.Options{...})`              | Mandatory Lambda; `RankTolerance` controls QR cutoff |

The implementation constructs an augmented matrix `[1,Z;0,sqrt(n*Lambda)*I_p]` (or `[Z;sqrt(n*Lambda)*I_p]` without intercept), then applies a **loss-preserving QR preparation**: zero-response penalty rows first, penalized slopes before the intercept, and positive column equilibration. The QR solution is mapped back before publishing `beta`, `SSE`, `Penalty` and `Objective`. This protects extremely small nonzero slopes that a naive Householder RHS update can round to `-0` at high Lambda, without forming a normal equation, changing the penalty or silently falling back to OLS.

```go
package main

import (
    "context"
    "fmt"

    "github.com/lvlath/go/matrix"
    "github.com/lvlath/go/ridge"
)

func main() {
    // Two perfectly correlated synthetic signals plus an independent one.
    train, err := matrix.NewDense(5, 3)
    if err != nil { fmt.Println("matrix:", err); return }
    if err = train.Fill([]float64{
        1, 1, 10,
        2, 2, 12,
        3, 3, 11,
        4, 4, 16,
        5, 5, 14,
    }); err != nil { fmt.Println("training:", err); return }

    fit, err := ridge.FitStandardized(
        context.Background(), train, []float64{3.1, 4.9, 6.8, 9.4, 11.2}, 0.2,
    )
    if err != nil { fmt.Println("fit:", err); return }

    future, err := matrix.NewDense(2, 3)
    if err != nil { fmt.Println("future:", err); return }
    if err = future.Fill([]float64{6, 6, 17, 2, 2, 9}); err != nil {
        fmt.Println("future data:", err)
        return
    }
    forecast, err := fit.Model.Predict(context.Background(), future)
    if err != nil { fmt.Println("forecast:", err); return }
    fmt.Printf("forecasted points=%d objective=%.4f\n", len(forecast), fit.Objective)
}
```

The model learns nothing from future rows. `Lambda` is always caller-controlled and **must be positive and finite**; use `ols` explicitly for unregularized fits. `ResidualSumSquares` is the *original-unit training sum*, while `Penalty` uses *optimization-coordinate* slopes and `Objective=SSE/(2*n)+Penalty`. Treat training diagnostics as model-fit metadata, never as validation or live-market performance claims.

For the Householder cancellation counterexample `Lambda=1e100`, complete arithmetic derivation, strict error precedence, scalar/wide oracles and six operational executable examples, read [`docs/RIDGE.md`](docs/RIDGE.md), [`ridge/doc.go`](ridge/doc.go), [`ridge/example_test.go`](ridge/example_test.go), and [`examples/ridge/main.go`](examples/ridge/main.go).

### PCA: frozen orthogonal factors and auditable reconstruction

`pca` learns orthogonal latent directions from **training rows**, then projects new observations using frozen means, optional training population scales, and canonicalized loading vectors. It is the unsupervised complement of `ols` and `ridge`: those packages fit target responses, while PCA models predictor covariance and reconstructs measured features.

**What makes this PCA implementation deliberately conservative:**

* `pca.Fit(ctx, x, pca.Options{})` uses `matrix.Covariance` plus `matrix.EigenSym`, with a bounded Jacobi rotation budget and an independent eigenpair-accuracy gate. `pca.ThinSVD` is a reserved unsupported PCA enum **even though `matrix.SVD` exists separately**; there is no automatic fallback or center-model bridge.
* `FitResult` contains a privately owned immutable `Model`, retained sorted `Eigenvalues`, non-renormalized `ExplainedVarianceRatio`, full detected `Rank`, and the direct training `ReconstructionResidual` in centered/scaled coordinates. `FitResult.IsNil()` implements `core.Nilable`.
* `Model.Transform`, `InverseTransform`, and `Reconstruct` use frozen training parameters and independent per-call buffers; they never recenter a holdout batch or allow a getter to mutate fitted loadings.
* The solver refuses materially negative eigenvalues, invalid symmetry, inaccurate eigenpairs, nonfinite values, overflow and nonconvergence with distinct package sentinels. A covariance-basis numerical rank is **not** an SVD-rank certificate for poorly conditioned data.

A typical application builds **train → calibration → holdout**, fits `k` using training data only, calibrates fault thresholds outside PCA, and evaluates holdout reconstruction against a training-only baseline. Useful examples include yield-curve factor stress, six-channel turbine vibration monitoring, 12-lead ECG denoising, grid-feeder telemetry compression, and mixed-unit water-quality measurements. These executable fixtures are deterministic synthetic engineering demonstrations, not verified market, medical or plant datasets.

For complete executable code, see [`pca/example_test.go`](pca/example_test.go) and [`examples/pca/main.go`](examples/pca/main.go). For formulas, complexity, numeric limits, and exact API/error precedence, read [`docs/PCA.md`](docs/PCA.md).

**Choose by objective, not by the word "factor":** `pca.FitWithComponents` compresses measured features without a response variable; `ols.FitDefault` estimates a full-rank linear response; `ridge.FitStandardized` estimates a penalized response when predictors are correlated or wide. PCA is neither a substitute for a regression target nor a forecasting method.

### Weighted shortest paths: choose the result domain

`github.com/lvlath/go/dijkstra` now exposes **two distinct API families**, not four variants of the same return type:

| Need                                             | Public API                                                            | Result                                                            |
|:-------------------------------------------------|:----------------------------------------------------------------------|:------------------------------------------------------------------|
| All distances from one source                    | `dijkstra.Dijkstra(g, sourceID, opts...)`                             | `*dijkstra.Result` with distance map and optional predecessor map |
| One target, no heuristic                         | `dijkstra.Search(g, sourceID, targetID, opts...)`                     | `*dijkstra.PathResult` with original `EdgeIDs`                    |
| One target, two search directions                | `dijkstra.Bidirectional(g, sourceID, targetID, opts...)`              | `*dijkstra.PathResult` with certified stopping                    |
| One target with a **consistent** target estimate | `dijkstra.AStar(g, sourceID, targetID, hT, opts...)`                  | `*dijkstra.PathResult`                                            |
| Two **consistent** directional estimates         | `dijkstra.BidirectionalAStar(g, sourceID, targetID, hT, hS, opts...)` | `*dijkstra.PathResult`                                            |

`dijkstra.DistanceTo` and `dijkstra.ShortestPathTo` remain legacy **full-SSSP wrappers**, not target early-exit optimizations. `Search` returns only one original-cost route witness and does not expose finalized all-vertex distances. It uses one `core.Clone` per query; the legacy runner retains its live-graph traversal semantics.

Target methods require **exactly representable binary64 stored costs and heuristic keys**; this stricter arithmetic law is not the legacy rounded-distance policy. Reachable results include original edge IDs, while known-unreachable targets return `Distance=+Inf`, `Reachable=false` and nil error. Consistent A* callbacks are verified against all admitted oriented arcs before settlement; arbitrary admissible-only heuristics are not accepted.


### MST, graph Steiner, and covering walks: three distinct results

`github.com/lvlath/go/mst` now supports three related but different objectives:

| Objective                                               | Public call                                | Result / guarantee                                                                                 |
|:--------------------------------------------------------|:-------------------------------------------|:---------------------------------------------------------------------------------------------------|
| Cheapest full-vertex spanning structure                 | `mst.Kruskal(g)` / `mst.Prim(g, root)`     | `*mst.Result`, exact MST; explicit `mst.WithForest()` for MSF                                      |
| Cheap network connecting only named terminals           | `mst.SteinerTree(g, terminals, opts...)`   | `*mst.SteinerResult`, KMB **approximation** of graph Steiner using original `core.Edge.ID` values  |
| Cheapest route covering every edge of one supplied tree | `mst.SteinerWalk(network, startID, endID)` | `*mst.WalkResult`, closed when endpoints agree, fixed-end otherwise; ordered repeated original IDs |

`SteinerTree` requires an undirected weighted graph with finite **nonnegative** edge weights, unlike ordinary Prim/Kruskal, which accept finite negative weights. It validates every original edge (even loops and unrelated islands), permits disconnected components with no required terminal, and publishes only one connected acyclic selected network. `SteinerRecomputePaths` is the default memory mode; `SteinerRetainPaths` retains O(kV) predecessor state to avoid selected-path SSSP recomputations. A network's `TotalWeight` counts **unique installed edges**; a walk's `Distance` counts **every actual passage**.

---

### ACF: explicit sample autocorrelation over a fixed observation window

`github.com/lvlath/go/acf` analyzes serial dependence of a **single fixed,
complete, equally spaced** scalar window. It computes one globally centered
sample ACF by default, or an explicitly requested lag-adjusted or uncentered
second-moment estimator. Unlike the causal transforms in `series`, the ACF is
a fitted **whole-window** statistic: adding observations can change the fitted
mean and every previously computed lag.

| Need                            | Public call                                       | Critical interpretation                                           |
|:--------------------------------|:--------------------------------------------------|:------------------------------------------------------------------|
| Default centered sample ACF     | `acf.ComputeDefault(values,maxLag)`               | Shared `S0` denominator; exact-arithmetic Toeplitz PSD-compatible |
| Adjust for actual pair count    | `acf.ComputeLagAdjusted(values,maxLag)`           | Multiplies by `n/(n-k)`; may exceed `[-1,1]`, not generally PSD   |
| Preserve a physical zero origin | `acf.ComputeUncentered(values,maxLag)`            | Raw second moment; `Result.Mean=0` is a mode marker               |
| Raw moment and pair correction  | `acf.ComputeUncenteredLagAdjusted(values,maxLag)` | Both policies explicit; no hidden centering                       |
| Full explicit control           | `acf.Compute(values,maxLag,acf.Options{...})`     | The sole canonical kernel; no FFT fallback                        |

```go
package main

import (
    "fmt"

    "github.com/lvlath/go/acf"
)

func main() {
    readings := []float64{1, 2, 3}
    common, err := acf.ComputeDefault(readings, 2)
    if err != nil {
        fmt.Println("common:", err)
        return
    }
    adjusted, err := acf.ComputeLagAdjusted(readings, 2)
    if err != nil {
        fmt.Println("adjusted:", err)
        return
    }
    fmt.Printf("lag-2 common=%+.1f adjusted=%+.1f pairs=%d\n",
        common.Values[2], adjusted.Values[2], adjusted.PairCounts[2])
    // Output: lag-2 common=-0.5 adjusted=-1.5 pairs=1
}
```

The example demonstrates why **clamping adjusted values is mathematically
wrong**. Zero-lag `Values[0]` is exactly `1` for a successful result;
`PairCounts[k]=n-k`; `Result.Denominator` identifies the convention used.
Both result slices are detached, `Result.Clone()` deep-copies them, and
`Result.IsNil()` implements the `core.Nilable` pointer-presence law.
A centered constant is `ErrZeroVariance`; an uncentered nonzero constant
has a valid raw second-moment ACF. `errors.Is` distinguishes invalid options,
nil/empty input, nonfinite observations, zero energy and unrepresentable output.

The direct kernel uses normalized deviations, Neumaier-compensated products
and `math.FMA` mean reconstruction to avoid avoidable intermediate overflow.
Its current cost is **O(n*(maxLag+1))** time and **O(n+maxLag)** space: choose a
scientifically meaningful lag horizon. Exact-arithmetic PSD for the common
estimator is **not** a binary64 PSD certificate, and the package does not
supply PACF, FFT, confidence intervals, stationarity tests or p-values; use the
separate `pacf` package for common-denominator partial lag dependence.

For six executable operational examples (industrial cycles, finance, pitch,
bearing impacts, calibration and standby power), see
[`acf/example_test.go`](acf/example_test.go); for incident retry periodicity,
run `go run ./examples/acf`. The [ACF reference](docs/ACF.md) contains the
Gram proof, strict error precedence, numeric implementation details and a
mask-aware `series` integration example.

---

### PACF: partial lag dependence and Durbin–Levinson prediction variance

`github.com/lvlath/go/pacf` measures how much **additional linear lag
structure** remains at order `m` after accounting for lags `1..m-1`. It
computes a fixed-window, biased sample autocovariance with **one common
`n` denominator** for every lag, then solves the Yule–Walker prediction
system by Durbin–Levinson recursion. This is not the `acf.LagAdjusted`
estimator, Burg, or an OLS-based PACF.

| Need                    | Public API                                               | Meaning                                           |
|:------------------------|:---------------------------------------------------------|:--------------------------------------------------|
| Explicit lag and origin | `pacf.Compute(ctx,x,pacf.Options{MaxLag:L,Center:true})` | Canonical calculation, inclusive `L<n`            |
| Mean-adjusted PACF      | `pacf.ComputeCentered(ctx,x,L)`                          | Fits one full-window mean                         |
| Physical-zero PACF      | `pacf.ComputeUncentered(ctx,x,L)`                        | Raw second moments, **no centering**              |
| One coefficient         | `pacf.ComputeAtLag(ctx,x,m,true)`                        | Projects full recursion through order `m`         |
| Independent result      | `result.Clone()` / `result.IsNil()`                      | Detached curves, nil-safe `core.Nilable` behavior |

The exact small fixture below exposes the difference between ordinary and
partial autocorrelation at lag two. A practical AR-order investigation
should instead use a sufficiently long, complete training window and
independent chronological holdout evaluation.

```go
package main

import (
    "context"
    "fmt"

    "github.com/lvlath/go/pacf"
)

func main() {
    // Three centered readings give gamma=[2/3,0,-1/3].
    profile, err := pacf.ComputeCentered(context.Background(), []float64{1, 2, 3}, 2)
    if err != nil {
        fmt.Println("PACF:", err)
        return
    }
    fmt.Printf("samples=%d PACF(1)=%.1f PACF(2)=%.1f V(2)=%.1f\n",
        profile.Count, profile.Values[1], profile.Values[2],
        profile.PredictionVariance[2])
    // Output: samples=3 PACF(1)=0.0 PACF(2)=-0.5 V(2)=0.5
}
```

The zero `pacf.Options{}` deliberately means **uncentered lag zero**, unlike
the centered default of `acf.Options{}`. On success, `Values[0]` is exactly
`1`, both output slices are independently owned and have `L+1` elements,
`Count=n`, and every `PredictionVariance[m]` is finite and positive in
squared input units. `Result.Clone()` deep-copies both slices;
`IsNil()` safely detects typed nil pointers. The package never retains
or modifies inputs and does not publish a partially valid Result on error.

Numerical safeguards include globally scaled compensated covariance scans,
two disjoint coefficient rows (preventing in-place mirror corruption),
the factored update `V_m=V_(m-1)*(1-|kappa|)*(1+|kappa|)` and binary-exponent
physical-variance reconstruction. A zero selected energy gives
`ErrZeroVariance`; a failed recursion gives `ErrSingular`; nonrepresentable
arithmetic/output gives `ErrOverflow`. Classify through `errors.Is`.
A near-singular finite sample can fail conservatively in binary64; the
kernel does **not** epsilon-clamp or secretly change estimator. Nil or
pre-canceled context and invalid input follow the documented precedence.

PACF is a **whole-window descriptive** statistic. The population PACF of
stationary AR(p) cuts off beyond p, but a finite-sample profile is noisy:
no exact sample cutoff, p-value, automatic `arima.Order` or confidence
interval is promised. `PredictionVariance` is *in-sample linear prediction
error variance*, not held-out forecast risk. Computation costs
**`O(n*(L+1)+L²)`** time and **`O(L+1)`** extra space.

For numerical proofs, edge-case contracts, and production usage see
[`docs/PACF.md`](docs/PACF.md); for the tutorial see
[the PACF foundations](docs/TUTORIAL.md#11u-pacf-foundations-partial-dependence-and-prediction-variance);
for executable real-domain synthetic workflows see
[`pacf/example_test.go`](pacf/example_test.go) and `go run ./examples/pacf`.

---

### RSI: one Wilder recurrence for batch factors and online price streams

`github.com/lvlath/go/rsi` computes **Wilder Relative Strength Index** over
finite, strictly positive closing prices, using one numerical transition for
both whole histories and a single-owner stream. It measures relative smoothed
upward/downward **price changes**; it does not issue trade signals, choose
thresholds or infer a timestamp grid.

| Need                                   | API                                                             | Contract to preserve                                                |
|----------------------------------------|-----------------------------------------------------------------|---------------------------------------------------------------------|
| Indexed factor values with warmup mask | `rsi.Compute(closes, rsi.Options{Period:p})`                    | `*rsi.Result{Values,Valid}`; first valid at index `P`               |
| Standard 14-change factor              | `rsi.ComputeDefault(closes)`                                    | Wrapper; needs 15 closes before valid output                        |
| Explicit-period factor                 | `rsi.ComputePeriod(closes, p)`                                  | Wrapper; no duplicated formula                                      |
| Final point of a full history          | `rsi.Latest(closes, opts)`                                      | `(value,valid,error)`; O(n) time **and temporary memory**           |
| Online zero-allocation recurrence      | `rsi.NewPeriod(p)` / `rsi.NewDefault()` -> `Stream.Push(close)` | O(1) time/state, 0 allocs per Push after New                        |
| Ordered-feed failover                  | `Stream.Snapshot()` / `Restore(state)`                          | Versioned, detached scalar state; externally authenticate transport |

In this deterministic synthetic mark sequence, period two first becomes valid
at index two. The consumer keeps the mask, instead of treating a warmup zero
as a numerical measurement:

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/rsi"
)

func main() {
	closes := []float64{100, 102, 104, 103, 101, 100, 102, 106}
	const period = 2
	history, err := rsi.ComputePeriod(closes, period)
	if err != nil {
		fmt.Println("RSI:", err)
		return
	}
	firstValid := -1
	for i, valid := range history.Valid {
		if valid {
			firstValid = i
			break
		}
	}
	latest := len(history.Values) - 1
	fmt.Printf("first-valid=%d latest-valid=%t latest-rsi=%.2f\n",
		firstValid, history.Valid[latest], history.Values[latest])
}
```

```text
first-valid=2 latest-valid=true latest-rsi=90.11
```

A successful short history has **all `Valid=false`**, while a valid falling
history can genuinely have RSI=0 and a valid flat history has RSI=50.
`ErrNonFinite` rejects NaN/infinity; `ErrDomain` rejects nonpositive marks;
`ErrOverflow` rejects unrepresentable arithmetic; `ErrStateVersion` protects
checkpoint shape/configuration and uninitialized streams. A failed `Push`
leaves the previous close, averages and Count unchanged. Streams cannot be
mutated concurrently without synchronization. No path silently clamps a
numeric failure into `[0,100]`.

**Operational boundary:** `series` can validate a known timestamp cadence,
while exchange feed logic must check source sequence IDs. On a genuine gap, the caller
explicitly `Reset`s and rebuilds warmup; RSI never invents missing closes.
`Snapshot.Config` is SHA-256 of the algorithm ID and normalized period,
**not** a MAC of mutable state. Authenticate serialized checkpoints and
persist exchange offsets externally to avoid double-consuming a close after
failover. Do not reuse a stream across instruments or periods.

Read [`docs/RSI.md`](docs/RSI.md) for stable-ratio mathematics, exact error
precedence, ownership, ABI and integration examples; [`rsi/doc.go`](rsi/doc.go)
for the formal charter; [`rsi/example_test.go`](rsi/example_test.go) for six
executable operational stories (multi-period factors, collateral marks, flat
FX marks, authenticated failover, order-book sequence gaps and new-instrument
warmup). Example prices are deterministic fixtures, not authenticated exchange
history or evidence of strategy performance.

---

### MACD: SMA-seeded EMA trend separation with deterministic batch/stream replay

`github.com/lvlath/go/macd` computes three **signed, index-aligned** indicator
series from an ordered sequence of strictly positive finite closes. Its
Fast and Slow EMAs each use the arithmetic mean of their first `P` closes as
the seed; the Signal EMA is seeded only from the first `Signal` *valid* MACD
Line observations. It uses the conventional Appel EMA coefficient `2/(P+1)`,
**not** RSI's Wilder `1/P` recurrence. No first-price seed shortcut or
alternative hidden algorithm is selected by a wrapper.

| Requirement                          | Public API                                                      | Contract boundary                                                     |
|--------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------|
| Complete batch with explicit periods | `macd.Compute(closes, macd.Options{Fast:5, Slow:35, Signal:5})` | Canonical `*macd.Result`, three detached `Values`/`Valid` pairs       |
| Conventional 12/26/9                 | `macd.ComputeDefault(closes)`                                   | Same recurrence; first Line index 25, first Signal/Histogram index 33 |
| Three standalone period integers     | `macd.ComputeWithPeriods(closes, 5, 35, 5)`                     | Option assembly only; no second EMA kernel                            |
| Continuous stream                    | `macd.NewDefault()` / `macd.NewWithPeriods(5,35,5)` then `Push` | O(1) mutable state, 0 heap allocations in Push after initialization   |
| Controlled transfer between workers  | `Stream.Snapshot()` / `Stream.Restore(state)`                   | Detached state, version and period identity, transactional validation |

Every Result series has length `len(closes)`. Unready numeric cells contain
`+0` with `Valid=false`; a **valid** zero histogram is a different state.
Histogram readiness exactly matches Signal readiness, while Line can be
ready earlier. `Signal=1` makes Signal equal Line and Histogram identically
zero from the first valid Line onward. `Result.IsNil()` follows the
`core.Nilable` pointer-presence convention, not a readiness test.

This runnable example checks index alignment without interpreting histogram
values as orders. Its prices are synthetic and deterministic:

```go
package main

import (
    "fmt"

    "github.com/lvlath/go/macd"
)

func main() {
    closes := make([]float64, 40)
    for index := range closes {
        closes[index] = 100 + float64(index)
    }
    result, err := macd.ComputeDefault(closes)
    if err != nil {
        fmt.Println("macd:", err)
        return
    }
    firstLine, firstSignal, ready := -1, -1, 0
    for index := range closes {
        if result.Line.Valid[index] && firstLine < 0 {
            firstLine = index
        }
        if result.Signal.Valid[index] {
            ready++
            if firstSignal < 0 {
                firstSignal = index
            }
        }
    }
    fmt.Printf("line=%d signal=%d usable=%d\n", firstLine, firstSignal, ready)
    // Output:
    // line=25 signal=33 usable=7
}
```

**Operational boundary:** `Compute` processes complete batches in O(n) time
and O(n) result space. `Stream.Push` is error-atomic and needs O(1) time
and memory; a failed price, count or computed output cannot partially
advance its state. A Stream belongs to one goroutine and one
instrument/cadence. Missing bars, source offsets, duplicate sequence IDs,
corporate-action adjustments and downstream execution rules belong to the
caller. `Snapshot.Config` is a SHA-256 fingerprint of the **version and
periods**, not an authentication tag for the state; authenticate persisted
checkpoints and resume with the first unseen bar. These guarantees are
independent of, and do not imply, trading profitability.

Read [`docs/MACD.md`](docs/MACD.md) for mathematical and numeric proofs,
[`macd/doc.go`](macd/doc.go) for the package charter,
[`macd/example_test.go`](macd/example_test.go) for five reproducible
operational examples, and the [MACD Tutorial](docs/TUTORIAL.md#11k-macd-foundations-sma-seeded-ema-signed-momentum-and-checkpoint-replay)
for a checked failover pipeline. The [MACD FAQ](docs/FAQ_%26_TIPS.md#13j-macd-faq--sma-seeding-signed-values-batchstream-replay-and-feature-safety)
covers warmup, signed arithmetic, error precedence and checkpoint trust.

---

### MFI: compensated money flow across exactly the last P OHLCV transitions

`github.com/lvlath/go/mfi` combines **typical-price direction** with the
**current candle's traded base-asset volume**. It returns a bounded `[0,100]`
feature and an independently meaningful readiness mask. Unlike RSI (Wilder
price changes) and MACD (SMA-seeded price EMAs), this indicator uses volume,
with a fixed **transition** window rather than a smoothing recurrence.

| Need                          | Public call                                              | Engineering contract                                                   |
|-------------------------------|----------------------------------------------------------|------------------------------------------------------------------------|
| Five-column historical OHLCV  | `mfi.Compute(bars, mfi.Options{Period:14})`              | Canonical `*mfi.Result{Values,Valid}`, no input alias                  |
| Standard 14-transition window | `mfi.ComputeDefault(bars)`                               | First valid index **14** (15 candles), no short-history error          |
| Explicit period               | `mfi.ComputePeriod(bars,p)`                              | Same kernel; `p=0` selects 14, maximum `1<<20`                         |
| Row-oriented market candles   | `mfi.ComputeCandles(candles,opts)`                       | `[]mfi.Bar` adapter with `O(n)` temporary transposition                |
| Final historical value        | `mfi.Latest(bars,opts)`                                  | `(value,valid,error)` from full batch; **not** an incremental shortcut |
| Continuous feed               | `mfi.NewDefault()` / `mfi.NewPeriod(p)` then `Push(bar)` | Compensated rolling ring, single owner, atomic failures                |
| Hot failover                  | `Snapshot()` / `Restore(state)`                          | Detached ring/corrections, versioned ABI; authenticate data externally |

Every genuine candle advances the window, **including** a zero-volume candle
or one whose typical price has not changed. The first candle only seeds the
previous typical price; `Period+1` accepted bars are required for readiness.
On an all-zero-flow window MFI is exactly `50`; a positive-only window is
`100` and a negative-only window is `0`. `Valid` distinguishes a legitimate
zero from the pre-warmup zero placeholder.

The implementation protects finite `float64` arithmetic with a High-anchored
typical-price mean, a bounded MFI ratio, two Neumaier-compensated rolling
accumulators, scheduled full-ring replay every `Period` transitions, and at
most one extra negative-drift replay per epoch. There is **no** arbitrary
negative clamp, silent data repair or hidden unbounded-precision mode.
Unrepresentable raw flow/sums return `ErrOverflow` without a partial result
or mutated streaming state.

A short, complete example uses a deterministic synthetic OHLCV record stream:

```go
package main

import (
    "fmt"

    "github.com/lvlath/go/mfi"
)

func main() {
    candles := []mfi.Bar{
        {Open: 100, High: 101, Low: 99, Close: 100, Volume: 100},
        {Open: 100, High: 103, Low: 99, Close: 102, Volume: 150},
        {Open: 102, High: 103, Low: 100, Close: 101, Volume: 0},
        {Open: 101, High: 101.5, Low: 98, Close: 99, Volume: 170},
        {Open: 99, High: 102, Low: 98, Close: 101, Volume: 180},
        {Open: 101, High: 103, Low: 100, Close: 102, Volume: 130},
    }
    history, err := mfi.ComputeCandles(candles, mfi.Options{Period: 3})
    if err != nil {
        fmt.Println("MFI:", err)
        return
    }
    firstValid := 3
    last := len(history.Values) - 1
    fmt.Printf("first-valid=%t latest-valid=%t latest=%.2f\n",
        history.Valid[firstValid], history.Valid[last], history.Values[last])
    // Output: first-valid=true latest-valid=true latest=64.90
}
```

The example is synthetic and illustrates **indicator computation**, not
trading advice. For production feed handling, the caller validates timestamps,
source offsets and base volume units. A missing candle requires an explicit
`Reset` and new warmup; it must not be confused with an observed zero-volume
bar. The mutable `Stream` cannot be shared between goroutines without external
synchronization. `Snapshot.Config` hashes only algorithm identity and period,
so authenticate all serialized checkpoint fields and persist the source offset
externally before failover.

For independent analytic oracles, error precedence, numerical proofs,
Snapshot ABI and memory/latency limits, read [`docs/MFI.md`](docs/MFI.md)
and [`mfi/doc.go`](mfi/doc.go). The five executable scenarios in
[`mfi/example_test.go`](mfi/example_test.go) cover cross-asset review,
continuous volume pressure, authenticated failover, timestamp gaps and
zero-volume ingestion reconciliation. For a runnable primary/standby
pipeline: `go run ./examples/mfi`.

---

### OBV: compensated cumulative signed volume with deterministic streaming

`github.com/lvlath/go/obv` computes **cumulative signed traded base volume**:
a bar's entire current volume is added when its closing price rises versus the
previous accepted close, subtracted when it falls and ignored when closes are
equal. The initial candle establishes the price baseline only; the first
valid OBV output is **exactly +0**, independent of its volume. Every valid
candle produces a valid output, including observed zero-volume and flat-close
bars. OBV has **no warmup window** and is not a bounded ratio.

| Need                 | Public call                                              | Contract                                                     |
|----------------------|----------------------------------------------------------|--------------------------------------------------------------|
| Columnar history     | `obv.Compute(obv.Bars{...})`                             | Strict five-column OHLCV, detached complete Values and Valid |
| Decoded candle rows  | `obv.ComputeCandles([]obv.Bar{...})`                     | Same kernel without five temporary transposition arrays      |
| Continuous feed      | `obv.New()` or `var stream obv.Stream`, then `Push(bar)` | O(1) transactional step, zero heap allocations               |
| New epoch            | `stream.Reset()`                                         | The next accepted candle seeds a new baseline at zero        |
| Failover             | `Snapshot()` / `Restore(state)`                          | Preserves sum **and** compensation; authenticate externally  |
| Nil-state convention | `result.IsNil()`                                         | Checks pointer presence, not `Valid[0]` or slice lengths     |

Unlike `mfi`, OBV uses consecutive **closing-price direction** and a cumulative
balance rather than a rolling window of typical-price money flow. It retains a
Neumaier compensation component: an up-volume of `1e16`, then `+1`, then a
matching down-volume of `1e16` produces a final OBV of `1`, not zero. When an
addition or accepted observation count cannot be represented, `Push` returns
`ErrOverflow` and **does not mutate** the prior close, sum, compensation or
count. A failing batch returns `(nil,error)`, not a partial result.

A minimal batch example is independently checkable by hand:

```go
package main

import (
    "fmt"

    "github.com/lvlath/go/obv"
)

func main() {
    candles := []obv.Bar{
        {Open: 10, High: 10, Low: 10, Close: 10, Volume: 5},
        {Open: 11, High: 11, Low: 11, Close: 11, Volume: 3},
        {Open: 11, High: 11, Low: 11, Close: 11, Volume: 8},
        {Open: 9, High: 9, Low: 9, Close: 9, Volume: 2},
    }
    result, err := obv.ComputeCandles(candles)
    if err != nil {
        fmt.Println("OBV:", err)
        return
    }
    fmt.Printf("values=%v valid=%v\n", result.Values, result.Valid)
    // Output: values=[0 3 3 1] valid=[true true true true]
}
```

**Operations:** The caller must verify source order, missing bars, session
boundaries and the meaning of volume units. No price epsilon, time-gap
interpolation, quote/base conversion, institutional-order identification or
trading threshold is performed by the package. One mutable `Stream` belongs
to one goroutine; use detached snapshots for controlled handoff.
`Snapshot.Config` is a SHA-256 **compatibility fingerprint of the public
algorithm ID**, *not* an HMAC of mutable state. Authenticate serialized state
and bind the external feed offset before deploying failover.

For mathematical derivations, error precedence, overflow invariants,
versioned ABI and practical caveats, see [`docs/OBV.md`](docs/OBV.md) and
[`obv/doc.go`](obv/doc.go). Five independently runnable application scenarios
are in [`obv/example_test.go`](obv/example_test.go); a stand-alone
offset-aware gateway replay is available at `go run ./examples/obv`.

### AD: Chaikin close-location-weighted volume with compensated replay

`github.com/lvlath/go/ad` computes the **Chaikin Accumulation/Distribution
Line**. For each OHLCV candle, the sign and fraction of its **current
base volume** come from where Close lies between Low and High, rather than
from a comparison to the previous close. The first candle contributes
immediately, and every successful `Valid` entry (including index zero) is
`true`. Unlike `mfi`, A/D has **no rolling window or warmup** and is **not
bounded to `[0,100]`**. Unlike `obv`, it never stores a previous Close.

The stable multiplier uses `left=Close-Low`, `right=High-Close`, and
`spread=High-Low`; its value is `(left-right)/spread` for positive spread
and exact zero on a valid flat candle. The implementation never forms
`2*Close-High-Low`, which can overflow near `math.MaxFloat64`. It sums
signed contributions with Neumaier compensation and tries a
cancellation-first rebase before reporting an unrepresentable total.

| Need                    | Public API                                   | Contract                                                       |
|-------------------------|----------------------------------------------|----------------------------------------------------------------|
| Five OHLCV columns      | `ad.Compute(ad.Bars{...})`                   | Strict non-nil equal lengths, detached Values/Valid, O(n)      |
| Already-decoded candles | `ad.ComputeCandles([]ad.Bar{...})`           | Same numeric kernel; no five-column transposition              |
| Final batch A/D         | `ad.Latest(bars)` / `ad.LatestCandles(rows)` | Delegates to full batch; **O(n) time and memory**              |
| Continuous feed         | `ad.New()` / `ad.Stream{}`, then `Push(bar)` | O(1) transactional step, 0 steady-state heap allocs            |
| Explicit new epoch      | `stream.Reset()`                             | Zero origin before the next contributing first bar             |
| Standby continuation    | `Snapshot()` / `Restore(state)`              | `[sum,compensation]`, Count, fixed v1 identity, detached state |
| Nilable result          | `result.IsNil()`                             | True only for nil `*Result`, not for valid numeric zero        |

This five-bar numerical oracle verifies both the **first-bar contribution**
and flat/midpoint behavior without a hypothetical signal policy:

```go
package main

import (
    "fmt"

    "github.com/lvlath/go/ad"
)

func main() {
    candles := []ad.Bar{
        {Open: 1.5, High: 2, Low: 1, Close: 2, Volume: 3},
        {Open: 1.5, High: 2, Low: 1, Close: 1, Volume: 2},
        {Open: 1, High: 1, Low: 1, Close: 1, Volume: 100},
        {Open: 1.5, High: 2, Low: 1, Close: 1.5, Volume: 100},
        {Open: 3, High: 4, Low: 2, Close: 3.5, Volume: 8},
    }
    result, err := ad.ComputeCandles(candles)
    if err != nil {
        fmt.Println("invalid candles:", err)
        return
    }
    fmt.Println(result.Values)
    fmt.Println(result.Valid)
    // Output:
    // [3 1 1 1 5]
    // [true true true true true]
}
```

All OHLCV fields, including Open, are checked for finite positive prices,
correct candle geometry and finite nonnegative Volume. In batch mode the
error order is **Nil -> Empty -> Shape -> all NonFinite -> all Domain ->
Overflow**; a rejected call publishes **no partial Result**. `Push` and
`Restore` reject invalid operations without changing mutable state. The
snapshot carries `Algorithm="ad/chaikin-neumaier/v1"`, `Count`, an empty
`IntState` and `FloatState=[sum,compensation]` (exactly two float64
coordinates). `Config` is a SHA-256 **compatibility fingerprint of the
public algorithm identifier**, not authentication of checkpoint data.

**Operations:** Use an upstream source to establish contiguous timestamps,
instrument identity, stable base-volume units, feed offsets and replay policy.
A missing bar is not a recorded zero-volume bar; `Reset()` selects a **new**
cumulative origin, whose balance must not be compared naively to an old epoch.
Streams are single-owner, while batch results and snapshots are detached.
A/D is a numerical feature, **not** evidence of identified buyer activity
and not an automatically calibrated price/divergence signal.

For the mathematical bounds, compensated-rebase pseudocode, error precedence
and versioned replay contract, read [`docs/AD.md`](docs/AD.md) and
[`ad/doc.go`](ad/doc.go). Six deterministic executable stories are in
[`ad/example_test.go`](ad/example_test.go); the standalone CSV and explicit
cadence/failover scenario is in [`examples/ad/main.go`](examples/ad/main.go).

### Bootstrap: reproducible dependent-data resampling without hiding statistics

`github.com/lvlath/go/bootstrap` produces **only observation indexes**. It is
designed for workflows where independent observation shuffling would lose
important serial structure: portfolio tail risk, chronologically paired
regression data, industrial vibration episodes or service telemetry bursts.
The package intentionally does not fit models, materialize values, choose
block length or calculate a confidence interval.

| Objective                                           | API / `Plan.Kind`                              | Source behavior                                           |
|:----------------------------------------------------|:-----------------------------------------------|:----------------------------------------------------------|
| Respect the observed source boundary                | `SampleMovingIndices` / `MovingBlocks`         | Draw fixed contiguous, nonwrapping blocks                 |
| Treat the source as cyclic                          | `SampleCircularIndices` / `CircularBlocks`     | Draw fixed blocks with cyclic continuation                |
| Generate geometric runs of expected length B        | `SampleStationaryIndices` / `StationaryBlocks` | Restart with exact probability `1/B`                      |
| Persist one fully specified reproducible experiment | `SampleIndices(ctx,rows,Plan)`                 | Every replicate uses a separate Seed + logical row stream |

The canonical result is `*bootstrap.Result`: `Indices [][]int`, `Completed`
and versioned `RNG AlgorithmID`. Returned rows are detached and caller-owned.
**A non-nil result can accompany cancellation** and then contains *only
completed rows*, never the interrupted row. `Result.IsNil()` implements
`core.Nilable` but cannot replace checking `err` and `Completed`.

```go
package main

import (
    "context"
    "fmt"

    "github.com/lvlath/go/bootstrap"
)

func main() {
    // Training-only daily percentage returns from a synthetic volatility burst.
    returns := []float64{
        0.3, -0.2, 0.5, -0.6, -1.8, -1.4,
        -1.1, 0.2, 0.4, -0.3, 0.1, -0.2,
    }
    plan := bootstrap.Plan{
        Kind: bootstrap.StationaryBlocks,
        Length: len(returns), BlockLength: 4,
        Replicates: 80, Seed: 0,
    }
    result, err := bootstrap.SampleIndices(context.Background(), len(returns), plan)
    if err != nil {
        fmt.Println("resampling:", err)
        return
    }

    // Every statistic is computed *outside* the sampler. The same source
    // index could also select an aligned predictor row or a second asset.
    worstLosses := make([]float64, result.Completed)
    for r, row := range result.Indices {
        worst := -returns[row[0]]
        for _, index := range row[1:] {
            loss := -returns[index]
            if loss > worst {
                worst = loss
            }
        }
        worstLosses[r] = worst
    }
    fmt.Printf("scenario paths=%d, loss estimates=%d, replay ID set=%t\n",
        result.Completed, len(worstLosses), result.RNG != "")
    // Output: scenario paths=80, loss estimates=80, replay ID set=true
}
```

**Dependence, not false certainty.** A stationary resampled index process
has a uniform invariant distribution, but that does not make a
nonstationary return process stationary or guarantee valid VaR/ES confidence
intervals. The number of time steps, expected block length, chronological
training selection and statistical assumptions remain application policy.
The 256-MiB logical output ceiling checks native-`int` overflow first;
`ErrOverflow`, `ErrBudget`, `ErrInvalidOption` and the context errors retain
distinct `errors.Is` classifications. Unbiased bounded integers use exact
threshold-based rejection, never naive modulo mapping.

**Concrete cross-package bridge:** `series` checks the timestamp grid;
`bootstrap` samples positions in an approved training window; the caller
gathers the *same* index row for the corresponding `matrix` X rows and y
responses before fitting each `ols`/`ridge` replicate. For an ACF workflow,
block-resampled data can be inspected with `acf.ComputeDefault`, but it does
**not** automatically represent a zero-autocorrelation null distribution or
Newey-West HAC covariance. None of these procedures may leak future holdout
observations into the resampled training population.

Read [`docs/BOOTSTRAP.md`](docs/BOOTSTRAP.md) for full mathematics, exact RNG
word consumption, replay identity, memory admission and cancellation laws;
[`bootstrap/doc.go`](bootstrap/doc.go) for the package contract;
[`bootstrap/example_test.go`](bootstrap/example_test.go) for five executable
operational examples; and `go run ./examples/bootstrap` for the standalone
99% portfolio tail-risk sensitivity review. The demonstration inputs are
synthetic and deterministic, not certified live market or plant measurements.

---

### RNG: pinned stochastic streams, not process-global randomness

`github.com/lvlath/go/rng` is the shared **numeric stream protocol** for
reproducible Monte Carlo and fault-injection experiments. It uses an immutable
`Factory`, a fresh single-owner `RNG` per logical trajectory and pinned
`pcg-dxsm128-sha256le-u52-boxmuller-v1` seed/draw semantics. It is **not** a
cryptographic PRNG, independence proof, stochastic model or replacement for
`bootstrap.SampleIndices` without a separately accepted migration.

| Need                           | Public API                                                             | Contract                                                     |
|--------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Persistent stream for one path | `rng.NewFactory().Stream(seed,pathID)` or `rng.NewStream(seed,pathID)` | One fresh mutable object, one goroutine owner                |
| Strict open uniform            | `stream.UniformOpen()`                                                 | One raw word, `2^-53 ≤ U ≤ 1−2^-53`                          |
| Standard-normal shock          | `stream.Normal()`                                                      | Box-Muller cosine first; sine spare cached                   |
| Caller-owned random vectors    | `factory.FillUint64`, `FillUniformOpen`, `FillNormal`                  | New stream per batch call; writes ascending indices          |
| Exact conversion test          | `rng.UniformOpenFromUint64(raw)`                                       | Pure raw-word mapping, including forced endpoints            |
| Replay identity                | `factory.Version()`                                                    | Pin with seed, logical pathID, input revision and math model |

```go
package main

import (
    "fmt"
    "math"

    "github.com/lvlath/go/rng"
)

func main() {
    factory := rng.NewFactory()
    path := factory.Stream(0, 42) // Seed zero and logical path 42 are valid.
    shock := path.Normal()
    eventCode := path.Uint64()     // Raw draw preserves a pending Gaussian spare.
    secondShock := path.Normal()   // Returns cached sine with no additional raw draw.

    fmt.Println("finite shocks:", !math.IsNaN(shock) && !math.IsNaN(secondShock))
    fmt.Printf("maintenance-event-code: %02x\n", byte(eventCode))
    fmt.Println("version:", factory.Version())
    // Output:
    // finite shocks: true
    // maintenance-event-code: ea
    // version: pcg-dxsm128-sha256le-u52-boxmuller-v1
}
```

One nontrivial simulation should use `pathID` as the **logical sample number**.
Do not seed by worker ID, share a mutable stream between workers, or reduce
concurrent worker-local floats in completion order. When orchestrating trials
manually, store outcomes by logical path and reduce in ascending index order;
prefer `montecarlo.Run` for a ready bounded scheduler and reducer. Factory is
concurrent-safe; individual RNG streams are not.

For six executable operational examples (fulfillment-load stress, GBM option
pricing, circular-latency blocks, industrial temperature sensor faults, Gaussian
cache interleaving and six-channel batch telemetry), see
[`rng/example_test.go`](rng/example_test.go). For bounded concurrent
execution and canonical cross-worker moments, use the `montecarlo.Run` driver
rather than worker-local float64 accumulation. The complete RNG proof,
consumption table and integration restrictions are in [`docs/RNG.md`](docs/RNG.md)
and [`rng/doc.go`](rng/doc.go).

---

### Monte Carlo: independent paths, deterministic parallel reduction and exact quantiles

`github.com/lvlath/go/montecarlo` is a **model-independent parallel experiment driver**. Your `Sampler` defines a fixed-dimensional financial, industrial or infrastructure scenario; the driver creates one version-pinned `rng.Factory.Stream(Seed,pathID)` per logical trial, schedules a bounded pool of workers, and reduces results in **strict ascending path-ID order** through `stats.Moments`.

| Need                                               | Public facade                                                 | Contract                                                                       |
|:---------------------------------------------------|:--------------------------------------------------------------|:-------------------------------------------------------------------------------|
| Complete configuration and reproducible experiment | `montecarlo.Run(ctx,plan,sampler)`                            | Canonical `*Result`; validates options, dimension and memory before `Sample`   |
| Mean/sample variance/SE without raw paths          | `montecarlo.Estimate(ctx,paths,seed,sampler)`                 | Serial default, no `Paths × d` matrix                                          |
| Parallel mean-only estimation                      | `montecarlo.EstimateParallel(ctx,paths,seed,workers,sampler)` | Fixed per-path streams, single ordered reducer                                 |
| Empirical percentiles of every simulated path      | `montecarlo.RunWithQuantiles(ctx,plan,levels,sampler)`        | Continuous type-7 quantiles; raw values stored privately even if `Samples=nil` |
| Explicit path-by-feature audit table               | `montecarlo.RunWithSamples(ctx,plan,sampler)`                 | Detached matrix, path-ID row order and strict partial prefix                   |

**Why the separation matters:** workers may finish in any order, but floating-point moments are always updated by the coordinator in logical order. Full successful outputs are bitwise reproducible across worker counts on the same platform/backend when the sampler is itself deterministic and concurrency-safe. The dispatcher stays within `[nextReduce,nextReduce+2W)`, so a slow path zero cannot cause an unbounded completed-results queue. `W` is the effective worker count, not the total number of requested paths.

**Statistical and memory laws:** sample variance uses Bessel's `n-1`; `MeanStandardError` measures uncertainty of the estimated mean under independent paths, *not* uncertainty in the model itself. Exact empirical quantiles require one `Paths*d` raw float64 buffer, a `Paths` scratch column and a `q*d` output. `MaxSamplesBytes=0` selects a **64 MiB limit on those optional buffers**, not on the entire process or caller sampler. Preflight rejects overflow/insufficient budget before sampling.

The following complete example models an illustrative **one-hour microgrid capacity shortfall**, returning `[unservedMW,shortfallIndicator]`. It estimates average unmet load, the probability of any unmet load, and the empirical 95th-percentile unmet load. The demand and supply distributions are explicit model assumptions, not observations from a real utility or a validated dispatch optimizer.

```go
package main

import (
    "context"
    "fmt"
    "math"

    "github.com/lvlath/go/montecarlo"
    "github.com/lvlath/go/rng"
)

type gridHour struct{}

func (gridHour) Dimension() int { return 2 }

func (gridHour) Sample(ctx context.Context, _ uint64, random rng.RNG) ([]float64, error) {
    if err := ctx.Err(); err != nil {
        return nil, err
    }
    demandMW := math.Max(0, 240+18*random.Normal())
    availableMW := math.Max(0, 245+12*random.Normal())
    unservedMW := math.Max(0, demandMW-availableMW)
    shortfall := 0.0
    if unservedMW > 0 {
        shortfall = 1
    }
    return []float64{unservedMW, shortfall}, nil
}

func main() {
    result, err := montecarlo.RunWithQuantiles(
        context.Background(),
        montecarlo.Plan{Paths: 20_000, Seed: 20261118, Workers: 4},
        []float64{0.95}, gridHour{},
    )
    if err != nil {
        fmt.Println("simulation:", err)
        return
    }
    if !result.Complete {
        fmt.Println("incomplete simulation")
        return
    }
    p95, err := result.Quantiles.At(0, 0)
    if err != nil {
        fmt.Println("quantile:", err)
        return
    }
    fmt.Printf("mean shortfall=%.2f MW; p95=%.2f MW; outage fraction=%.1f%%\n",
        result.Mean[0], p95, 100*result.Mean[1])
}
```

`Result.Completed` counts only the committed prefix `[0,Completed)`. A sampler/context error returns a non-nil incomplete `Result` after work has begun; quantiles are **nil on every failure**, and moment vectors are nil with fewer than two committed paths or if final moment materialization fails. Even `Completed==Requested` does not prove success: inspect `err` and `Complete`. A sampler panic is **not** recovered by the package; an uncaught worker panic has Go's normal process-terminating behavior.

For applied models, see the five checked examples in [`montecarlo/example_test.go`](montecarlo/example_test.go): five-asset option pricing, two-pass VaR/Expected Shortfall, fatigue cracks, microgrid reserve stress, and interrupted calibration. Correlated assets require an explicitly modeled and validated factor inside the caller-owned `Sampler`; the driver does not infer covariance. The full mathematical and engineering contract is in [`docs/MONTECARLO.md`](docs/MONTECARLO.md) and [`montecarlo/doc.go`](montecarlo/doc.go). Neither the Monte Carlo SE nor a replayable numeric result certifies a financial or physical model's correctness.

---

### GBM: exact-in-distribution lognormal paths with explicit physical units

`github.com/lvlath/go/gbm` simulates **one positive geometric Brownian motion trajectory** over a caller-owned nonuniform time grid. It consumes one standard-normal draw per stochastic interval from `rng.RNG`, accumulates logarithmic growth in extended-exponent arithmetic and publishes only positive finite `float64` prices. The initial price is retained exactly as the conversion anchor; positive subnormals are legal, while a rounded zero or infinity is `ErrOverflow`, **not** a fabricated epsilon-clamped value.

| Need                                    | Public API                                            | What the result means                                                 |
|:----------------------------------------|:------------------------------------------------------|:----------------------------------------------------------------------|
| Full explicit-grid GBM path             | `gbm.Simulate(ctx,p,grid,random)`                     | `*Result{Times,Values,Complete}`; complete or valid prefix with error |
| Scalar construction with explicit units | `gbm.SimulatePath(ctx,s0,mu,sigma,unit,times,random)` | Same canonical kernel and error law                                   |
| One-horizon European terminal model     | `gbm.SimulateTerminal(ctx,p,horizon,random)`          | `[0,horizon]` or initial-only `[0]` at zero horizon                   |
| Reproducible standalone scenario        | `gbm.SimulateSeeded(ctx,p,grid,seed,pathID)`          | A fresh `rng.NewStream` addressed by logical path ID                  |

**Why the numeric policy matters.** Neither `S_previous*exp(logIncrement)` nor an upfront `sigma*sigma` is safe across the full finite `float64` parameter domain. The kernel computes `sigma*sqrt(dt)` and the Itô correction in wide arithmetic and avoids premature level-domain overflow, including cases with representable final prices. The fixed-precision implementation is **not** a proof of correctly rounded exact-real results for every possible finite input. Its exact-in-distribution statement concerns the model transition law, assuming normal draws, not arbitrary precision or empirical realism.

**Time is an explicit model dimension.** `gbm.TimeYears`, `TimeDays` and `TimeSeconds` are labels with strict equality between `Parameters.Unit` and `TimeGrid.Unit`. Drift is `1/Unit`; volatility is `1/sqrt(Unit)`. No hidden 252-session calendar, year fraction or seconds conversion occurs. Grid values are elapsed, finite, begin at zero and increase strictly. `[0]` is valid and consumes no Gaussian draw; `sigma==0` consumes no RNG entropy on any grid but still requires a non-nil stream.

**Runtime failure is not successful early maturity.** Invalid inputs and pre-canceled contexts return `nil, error`. After preflight, interrupted or numerically failed steps return only the detached completed prefix, with `Complete=false` and a classified error. `Result.IsNil()` implements pointer-presence semantics through `core.Nilable`; it does not prove horizon completion. A Monte Carlo adapter must return an error rather than pricing an incomplete GBM path.

The following example produces a replayable, synthetic one-year path with event-specific checkpoints; the printed metadata is deterministic without implying that the sampled endpoint is a verified market forecast:

```go
package main

import (
    "context"
    "fmt"

    "github.com/lvlath/go/gbm"
)

func main() {
    params := gbm.Parameters{
        Initial: 100, Drift: 0.03, Volatility: 0.22, Unit: gbm.TimeYears,
    }
    grid := gbm.TimeGrid{
        Values: []float64{0, 0.05, 0.25, 0.75, 1}, Unit: gbm.TimeYears,
    }
    path, err := gbm.SimulateSeeded(context.Background(), params, grid, 20261118, 17)
    if err != nil {
        fmt.Println("simulation:", err)
        return
    }
    if path == nil || !path.Complete {
        fmt.Println("incomplete path")
        return
    }
    fmt.Printf("checkpoints=%d first=%.0f horizon=%.0f year\n",
        len(path.Values), path.Values[0], path.Times[len(path.Times)-1])
    // Output:
    // checkpoints=5 first=100 horizon=1 year
}
```

`gbm` deliberately does **not** calibrate a model, certify lognormal market behavior, choose a financial probability measure, implement portfolio correlation or price a derivative. To price a dividend-paying European option under a caller-selected risk-neutral model, pass `Drift=r-q`, use the stream supplied to `montecarlo.Sampler.Sample`, simulate a complete terminal path, discount the payoff **outside GBM**, and let `montecarlo.Run` perform ordered sample reduction. For a path-dependent payoff, provide the observation grid explicitly; unobserved continuous-time barrier crossings are not resolved by endpoint simulation.

The five executable examples cover irregular synthetic ticks, risk-neutral call pricing with Monte Carlo worker replay, five-year FX reserve stress, zero-entropy deterministic procurement costs and explicit annual-to-seconds conversion. See [`docs/GBM.md`](docs/GBM.md) for complete mathematics, numerical limits, error precedence, diagrams and applied examples; [`gbm/doc.go`](gbm/doc.go) for the source contract; [`gbm/example_test.go`](gbm/example_test.go) and [`examples/gbm/main.go`](examples/gbm/main.go) for tested operational programs.

---

### Jump Diffusion: compensated Merton shocks over explicit GBM time grids

`github.com/lvlath/go/jumpdiffusion` simulates one **compound-Poisson lognormal
jump-diffusion trajectory** on a caller-defined grid. It reuses
`gbm.Parameters` and `gbm.TimeGrid`, gets all randomness from the path-owned
`rng.RNG`, and exposes the familiar detached
`*Result{Times,Values,Complete}`. It is an **experimental, assumption-driven
scenario model**: neither a fitted tail distribution nor a crash predictor.

The distinction from ordinary GBM is mathematically important. With
`N~Poisson(lambda*dt)`, log-jump mean `m` and log-jump deviation `s`, the
aggregate log jump is `N*m+sqrt(N)*s*Z`. In `Compensated` mode (the default),
the deterministic proportional drift subtracts

$$
  \lambda k,\qquad k=\operatorname{expm1}(m+s^2/2),
$$

![Merton compensator](https://latex.codecogs.com/svg.image?%5Cdisplaystyle%20%5Clambda%5Coperatorname%7Bexpm1%7D%28m%2Bs%5E2%2F2%29)

so the ideal model retains `E[S(t)]=S0*exp(mu*t)`. **`m` is not `k`**.
`Uncompensated` instead has expected growth
`S0*exp((mu+lambda*k)*t)`. These are explicit modeling conventions, not
automatic risk-neutral calibration.

| Need                                     | Actual public facade                                        | Important law                                               |
|:-----------------------------------------|:------------------------------------------------------------|:------------------------------------------------------------|
| One full irregular jump-aware trajectory | `jumpdiffusion.Simulate(ctx,p,grid,random)`                 | Canonical kernel; valid completed prefix on runtime failure |
| Model plus existing time slice           | `jumpdiffusion.SimulatePath(ctx,p,times,random)`            | Times are in `p.Diffusion.Unit`                             |
| Only the expiry (European payoff)        | `jumpdiffusion.SimulateTerminal(ctx,p,T,random)`            | `[0,T]` or `[0]`, no intrastep monitoring                   |
| Replay one logical scenario              | `jumpdiffusion.SimulateSeeded(ctx,p,grid,seed,pathID)`      | New path-owned RNG                                          |
| Replay one logical expiry                | `jumpdiffusion.SimulateTerminalSeeded(ctx,p,T,seed,pathID)` | No global RNG or implicit calendar                          |

**Why Poisson accuracy matters.** For `a=lambda*dt`, product inversion handles
`0<a<30`, and bounded Hörmann PTRS handles `30<=a<=2^40`; there is no
Gaussian approximation of the integer count. Accepted counts are limited
to `2^53-1`, and each sampling interval has at most `2^20` attempts.
The RNG sequence is diffusion `Normal` (if `sigma>0`), Poisson
`UniformOpen`, then one aggregate-jump `Normal` (only if `N>0 && s>0`).
With `Intensity=0` the facade delegates directly to `gbm.Simulate` for
**bitwise path and RNG-consumption parity**.

**Numerical and operational boundaries.** The implementation accumulates
anchored log growth with extended precision, accepts finite positive
subnormal prices, and returns classified `ErrOverflow`, `ErrDomain`,
`ErrNonFinite`, `ErrBudget` or context errors without publishing an
unfinished point. Model/grid unit mismatches are rejected; intensity is
measured per `Diffusion.Unit`. An invalid input produces nil result, while
a failure after valid preflight returns a completed prefix with
`Complete=false`. `Result.IsNil()` implements the `core.Nilable` pointer
protocol but is **not** a completeness test. A `montecarlo.Sampler` must
return an error for any incomplete path rather than silently price the last
published mark.

A small reproducible collateral-risk fixture demonstrates how to use the
canonical seeded facade without relying on any wall clock:

```go
package main

import (
    "context"
    "fmt"

    "github.com/lvlath/go/gbm"
    "github.com/lvlath/go/jumpdiffusion"
)

func main() {
    params := jumpdiffusion.Parameters{
        Diffusion: gbm.Parameters{
            Initial: 60_000, Drift: 0.03, Volatility: 0.65,
            Unit: gbm.TimeYears,
        },
        Intensity: 2, LogJumpMean: -0.20, LogJumpStdDev: 0.10,
        Convention: jumpdiffusion.Compensated,
    }
    grid := gbm.TimeGrid{
        Values: []float64{0, 1.0 / 365, 7.0 / 365, 30.0 / 365, 90.0 / 365},
        Unit:   gbm.TimeYears,
    }
    path, err := jumpdiffusion.SimulateSeeded(
        context.Background(), params, grid, 2026, 17,
    )
    if err != nil {
        fmt.Println("simulation:", err)
        return
    }
    if path == nil || !path.Complete {
        fmt.Println("incomplete scenario")
        return
    }
    fmt.Printf("points=%d initial=%.0f days=%.0f\n",
        len(path.Values), path.Values[0], 365*path.Times[len(path.Times)-1],
    )
    // Output:
    // points=5 initial=60000 days=90
}
```

This printed metadata is deterministic; the sampled path is **not** a
validated frequency of real flash crashes. Five operational examples in
[`jumpdiffusion/example_test.go`](jumpdiffusion/example_test.go) cover
flash-crash collateral controls, European call/put Monte Carlo with
put–call parity, PTRS high-intensity event flow, path-ID audit replay and
credit-event reserves. See [`docs/JUMPDIFFUSION.md`](docs/JUMPDIFFUSION.md)
for formula derivations, algorithm branches, complete error precedence,
benchmarks and attribution; [`jumpdiffusion/doc.go`](jumpdiffusion/doc.go)
for the package charter; and [the FAQ](docs/FAQ_%26_TIPS.md#13p-jump-diffusion-faq-merton-compensation-poisson-counts-and-replay)
for model-selection and numerical pitfalls.

---

### GJR-GARCH: stationary causal volatility and asymmetric downside response

`github.com/lvlath/go/gjrgarch` estimates and forecasts the **conditional
variance of returns**, not a price trajectory or a fully specified tail-risk
distribution. Its immutable `Model` uses a strictly lagged asymmetric shock:


![Causal GJR-GARCH variance recurrence](https://latex.codecogs.com/svg.image?h_%7Bt%2B1%7D%3D%5Comega%2B%28%5Calpha%2B%5Cgamma%5Cmathbf%7B1%7D_%7B%5C%7B%5Cepsilon_t%3C0%5C%7D%7D%29%5Cepsilon_t%5E2%2B%5Cbeta%20h_t)

Here `epsilon[t]=r[t]-Mean`, `h[t]` is assigned *before* observing `r[t]`,
`Omega>0`, `Alpha>=0`, `Beta>=0`, `Alpha+Gamma>=0`, and symmetric-innovation
persistence `rho=Alpha+Beta+Gamma/2<1`. **Gamma may be negative**. Negative
shocks influence the **next** variance; they cannot change the variance already
assigned to their own observation.

| Need                               | Public API                                           | Honest interpretation                                                                             |
|------------------------------------|------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Validate and freeze parameters     | `gjrgarch.New(parameters)`                           | Immutable, model-identified stationary law                                                        |
| Start from the first observation   | `model.InitialState(firstReturn,h0)`                 | Posterior `(epsilon[0],h[0],Steps=1)`; zero State is invalid                                      |
| Update one observation             | `model.Step(state,newReturn)`                        | Uses the **old** residual for new variance; successful hot path allocates no slice                |
| Filter a fixed sample              | `model.Filter(ctx,returns,h0)`                       | Detached `[h0,...,h[T-1]]`; no future leakage or partial output                                   |
| Use long-run initialization        | `model.FilterStationary(ctx,returns)`                | Explicit `Omega/(1-rho)` starting variance                                                        |
| Predict next-period variance       | `model.ForecastNextVariance(state)`                  | Exact known-shock first horizon                                                                   |
| Predict an H-period variance curve | `model.ForecastVariance(state,H)`                    | One known shock, then `Omega+rho*h` conditional expectations                                      |
| Fit with explicit initial model    | `gjrgarch.Fit(ctx,returns,Options{Initial:...})`     | Deterministic projected BFGS; error and convergence must both be checked                          |
| Audit a fixed Gaussian candidate   | `gjrgarch.EvaluateGaussianQMLE(ctx,returns,p,floor)` | Negative average loss and five-coordinate **analytic gradient**, not a holdout-continuation score |

A minimal stress comparison using the same prior conditional variance:

```go
package main

import (
	"fmt"
	"math"

	"github.com/lvlath/go/gjrgarch"
)

func main() {
	model, err := gjrgarch.New(gjrgarch.Parameters{
		Omega: 0.00002, Alpha: 0.08, Gamma: 0.14, Beta: 0.72,
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	down, err := model.InitialState(-0.03, 0.00030)
	if err != nil {
		fmt.Println(err)
		return
	}
	up, err := model.InitialState(0.03, 0.00030)
	if err != nil {
		fmt.Println(err)
		return
	}
	downVariance, err := model.ForecastNextVariance(down)
	if err != nil {
		fmt.Println(err)
		return
	}
	upVariance, err := model.ForecastNextVariance(up)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("downside_greater=%t down_sigma=%.2f%% up_sigma=%.2f%%\n",
		downVariance > upVariance,
		100*math.Sqrt(downVariance), 100*math.Sqrt(upVariance))
	// Output:
	// downside_greater=true down_sigma=2.08% up_sigma=1.75%
}
```

For a real market-risk pipeline, first validate timestamps and obtain returns
from `series.LogReturns` **using its `Valid` mask**. Feed the training period to
`Fit` with an economically justified starting point; on `ErrNotConverged`, a
non-nil `FitResult` is only a diagnostic candidate. **Never** approve a model
from training likelihood or convergence alone. For a chronological held-out
log-score, continue from the last training `State`, forecast before observing
the next return, then call `Step`; `EvaluateGaussianQMLE` uses a fresh stationary
start and does not provide this continuation.

The Gaussian likelihood is quasi-likelihood. It does not certify VaR/ES,
regulatory margin levels, parameter uncertainty or market adequacy. See
[`docs/GJRGARCH.md`](docs/GJRGARCH.md) for KKT/gradient mathematics, numeric
constraints, model-ID semantics, errors and deployment evidence;
[`gjrgarch/doc.go`](gjrgarch/doc.go) for the canonical charter;
[`gjrgarch/example_test.go`](gjrgarch/example_test.go) for five operational
examples; and [`examples/gjrgarch/main.go`](examples/gjrgarch/main.go) for
prices-to-returns monitoring with explicit fail-closed model release.

---

### ARIMA: fixed-order conditional Gaussian forecasting with anchored levels

`github.com/lvlath/go/arima` estimates a **specified** univariate
nonseasonal ARIMA `(p,d,q)` under conditional Gaussian likelihood, then
forecasts in **original level units**. It is separate from `gjrgarch`:
ARIMA models the conditional mean of differenced observations with a fixed
innovation variance, whereas GJR-GARCH models time-varying asymmetric
conditional return variance. Neither package silently fits the other.

![ARIMA differenced-space equation](https://latex.codecogs.com/svg.image?y_t%3D%5CDelta%5Ed%20x_t%3Dc%2B%5Csum_%7Bi%3D1%7D%5Ep%5Cphi_i%20y_%7Bt-i%7D%2B%5Cvarepsilon_t%2B%5Csum_%7Bj%3D1%7D%5Eq%5Ctheta_j%5Cvarepsilon_%7Bt-j%7D)

Here `Drift=c` is the **intercept after differencing**, not necessarily
the mean level or a constant growth rate. The first `max(p,q)` differenced
observations condition the residual recursion; they are not part of the
SSE/likelihood. The AR and MA polynomials must be strictly stationary and
invertible in differenced space; unit roots are rejected, never clipped.

| Requirement             | Public API                                | Key contract                                                      |
|-------------------------|-------------------------------------------|-------------------------------------------------------------------|
| Fixed-order calibration | `arima.Fit(ctx,values,Options{...})`      | `p,q=0..8`, `d=0..2`; `Options{}` means white noise without drift |
| Differenced intercept   | `arima.FitWithDrift(ctx,values,order)`    | Fit `c`, whose long-run differenced mean is `c/(1-sum(phi))`      |
| Exactly zero intercept  | `arima.FitWithoutDrift(ctx,values,order)` | Fix `c=0`; no hidden recentering                                  |
| Stationary ARMA         | `arima.FitARMA(ctx,values,p,q,drift)`     | `d=0` convenience facade                                          |
| Integrated white noise  | `arima.FitIntegrated(ctx,values,d,drift)` | ARIMA(0,d,0), analytical estimation                               |
| Validated model         | `arima.New(arima.Parameters{...})`        | Immutable coefficients, strict AR/MA Schur checks                 |
| Horizon forecast        | `model.Forecast(ctx,history,H)`           | Anchored `Mean`, marginal `Variance` and `Horizon`                |
| Next observation        | `model.ForecastNext(ctx,history)`         | Same canonical forecast with H=1                                  |

For `d=1` the level origin is the last observed value; for `d=2` the
origin additionally retains the last observed first difference. The
horizon variance is accumulated from the **integrated impulse response**
with `O(H)` working memory, without a dense forecast covariance matrix.
Both levels and variances are detached, and an immutable model supports
independent concurrent forecasting on stable histories.

A minimal operating example projects three days of a reviewed incident
backlog. It deliberately chooses an analytical integrated model to avoid
requiring a nonlinear optimizer for this introduction:

```go
package main

import (
    "context"
    "fmt"

    "github.com/lvlath/go/arima"
)

func main() {
    dailyBacklog := []float64{
        200, 204, 207, 211, 215, 218, 222,
        225, 229, 233, 236, 240, 244,
    }
    fit, err := arima.FitIntegrated(context.Background(), dailyBacklog, 1, true)
    if err != nil {
        fmt.Println("fit:", err)
        return
    }
    forecast, err := fit.Model.Forecast(context.Background(), dailyBacklog, 3)
    if err != nil {
        fmt.Println("forecast:", err)
        return
    }
    fmt.Printf("day+3=%.1f tickets; innovation variance=%.2f tickets^2\n",
        forecast.Mean[2], forecast.Variance[2])
    // Output: day+3=255.0 tickets; innovation variance=0.67 tickets^2
}
```

This deterministic sample is an **illustrative fixture**, not an observed
service ledger. Callers validate timestamp cadence, choose their model order,
score a chronological holdout and control the decision threshold. `Fit` may
return a last accepted candidate **with `ErrNotConverged`**; using it in
production requires an explicit failed-fit review policy. `IsNil()` checks
only pointer presence and `Model.ID()` is a content fingerprint, not
authenticated data provenance.

Full mathematics, independent oracles, error precedence, Schur conventions,
uncertainty limits and examples: [`docs/ARIMA.md`](docs/ARIMA.md),
[`arima/doc.go`](arima/doc.go), [`arima/example_test.go`](arima/example_test.go),
[`examples/arima/main.go`](examples/arima/main.go),
[the tutorial](docs/TUTORIAL.md#11s-arima-foundations-conditional-likelihood-differencing-and-anchored-forecasts),
and [the FAQ](docs/FAQ_%26_TIPS.md#13r-arima-faq-conditional-gaussian-fit-schur-constraints-and-anchored-forecasts).

### HMM: causal regime filtering, retrospective smoothing and reproducible EM

`github.com/lvlath/go/hmm` models **hidden operating regimes** behind categorical events or finite multichannel telemetry. Its two immutable model families share the same transition mathematics but have distinct, typed observation contracts:

| Need                                     | API                                           | Contract / safe interpretation                                             |
|:-----------------------------------------|:----------------------------------------------|:---------------------------------------------------------------------------|
| Categorical hidden regimes               | `hmm.NewCategorical(CategoricalSpec)`         | `K×A` emission probabilities, `uint32` symbols in `[0,A)`                  |
| Independent Gaussian feature coordinates | `hmm.NewGaussian(GaussianSpec)`               | `K×D` means and **positive** variances in feature²                         |
| Single live observation                  | `Model.FilterStep(ctx, prev, symbolOrVector)` | New detached causal checkpoint; `prev` is never modified                   |
| Batch last state only                    | `Model.FilterDefault(ctx, observations)`      | `FilterResult.Last` plus cumulative log evidence; **no** historical matrix |
| Causal trajectory                        | `Model.FilterWithHistory(ctx, observations)`  | T×K `P(z_t \| x_0:t)`, invariant under future append                       |
| Retrospective regime attribution         | `Model.Smooth(ctx, observations)`             | T×K `P(z_t \| x_0:T-1)`, **uses future evidence**                          |
| One globally consistent path             | `Model.Decode(ctx, observations)`             | Maximum-joint path, not per-row posterior maxima                           |
| Learn from independent sessions/cycles   | `hmm.FitCategorical` / `hmm.FitGaussian`      | Explicit initial parameters, no hidden RNG, no cross-session transitions   |

**The critical causal boundary:** a smoothed posterior is valid for *postmortem analysis*, but it must not be shown as the risk probability known to an online controller at the earlier timestamp. The causal filter combines the current emission with **only** the previous filtered state. Viterbi optimizes one entire state path and therefore answers a third question, not a different formatting of the same probabilities.

```text
           previous FilterState + current observation
                         |
                  FilterStep (causal) ------------------> live decisions
                         |
                 log posterior / ModelID

           complete recorded observation sequence
                         |
                 +-------+---------+
                 |                 |
              Smooth             Decode
          retrospective       most likely joint
           per-time belief     state trajectory

   independent training sequences -> Fit* -> diagnostics + immutable model
                         |
              NO transition across sequence boundaries
```

The stochastic kernel evaluates the log-domain Forward/Backward and Viterbi recurrences through `LogSumExp`, including an explicit all-`-Inf` case. It never fixes impossible categorical events by inventing epsilon probability. Gaussian observations can have positive log-density; arithmetic overflow is distinguished from impossible categorical support. The `FilterState.ModelID` binds checkpoints to the exact parameter revision, and all five result types implement `core.Nilable.IsNil()`.

**Training is explicit and auditable:** `Pseudocount=lambda` corresponds to symmetric Dirichlet MAP regularization (`alpha=1+lambda`) on stochastic rows, and EM monitors the **penalized** objective, not naive data likelihood. `VarianceFloor` constrains Gaussian **variance in squared units** via `max(estimate,floor)`, not by adding jitter. `ErrNotConverged` and a rejected objective step can return the last accepted model with diagnostic status; this is **not** implicit successful training. Input/context errors return nil results. Use `errors.Is` for all sentinel classifications.

Operational examples span clickstream conversion intent, live market stress detection, industrial machine phase decoding, retrospective alarm attribution, and independent-session training. They are deterministic synthetic fixtures, not pre-calibrated real-world classifiers. Feature masks and timestamps are handled by `series`; train-only scaling by `stats` or a frozen application transformer; categorical encoders must maintain an explicit stable symbol dictionary. Nothing in the package silently refits preprocessing during inference.

See [`docs/HMM.md`](docs/HMM.md) for formal log-domain equations, EM counts and penalty derivations, memory/error laws and anti-leakage counterexamples. The package constitution is [`hmm/doc.go`](hmm/doc.go); executable examples are [`hmm/example_test.go`](hmm/example_test.go); a standalone scenario is [`examples/hmm/main.go`](examples/hmm/main.go), where installed.

---

### LSTM: immutable IFGO inference with masked batches and stateful continuation

`github.com/lvlath/go/lstm` evaluates a **previously trained**, single-layer,
unidirectional `float64` LSTM. It uses the PyTorch **i/f/g/o** gate layout,
strictly finite numeric inputs, a versioned SHA-256 parameter identity and a
single deterministic forward kernel. The package does **not** train weights,
load Python checkpoints, fit feature scales, implement neural output heads,
or produce a calibrated risk or fault probability on its own.

| Need                                    | Public API                                                     | Important behavior                                                                            |
|:----------------------------------------|:---------------------------------------------------------------|:----------------------------------------------------------------------------------------------|
| Import trusted numerical weights        | `lstm.Import(lstm.Weights{...})`                               | Owns copies of Input `[4H,D]`, Recurrent `[4H,H]`, combined Bias `[4H]`; freezes `Model.ID()` |
| Run all steps with zero initial state   | `model.InferDefault(ctx, input)`                               | Full batch, final-only result; no Sequence buffer                                             |
| Extract final state from a masked batch | `model.InferFinal(ctx, input, lengths, initial)`               | Masked steps never update either Hidden or Cell                                               |
| Extract every valid recurrent feature   | `model.InferSequence(ctx, input, lengths, initial)`            | `[B,T,H]` Sequence; padded positions are exactly zero                                         |
| Continue an ordered session             | `model.InferChunk(ctx, chunk, localLengths, previous, opts)`   | Uses both `previous.Final` matrices and verifies `ModelID`                                    |
| Select the full behavior explicitly     | `model.Infer(ctx, input, lengths, initial, lstm.Options{...})` | Canonical facade; all wrappers delegate to it                                                 |

```text
trusted, externally validated numeric weights
             |
          Import --------------> immutable *lstm.Model / SHA-256 ID
                                      |
caller-normalized Tensor3 [B,T,D] ----+--> Infer [IFGO gates, finite checks]
          lengths [B] -------------------> |                 |
                                           |                 +--> Final [B,H]: Hidden + Cell
                                           +--> Sequence [B,T,H] (optional, zero padding)
                                                                        |
                              previous.Final --> next InferChunk -------+
```

The numerical contract separates **invalid input** (`ErrNonFinite`) from
**unrepresentable arithmetic** (`ErrOverflow`). A computed infinite affine
value is never passed to sigmoid as if it were a valid saturated gate.
`lengths` describes valid steps **within the current chunk**; the full
input buffer, including masked cells, is checked for NaN/Inf. Model weights
and published result matrices are detached. One immutable Model supports
concurrent calls over stable read-only inputs, with independent workspaces.

A compact first-use example (illustrative, **synthetic** weights):

```go
package main

import (
	"context"
	"fmt"

	"github.com/lvlath/go/lstm"
	"github.com/lvlath/go/matrix"
)

func main() {
	// H=1, D=1; IFGO gates have four input and four recurrent rows.
	wi, err := matrix.NewDense(4, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	wr, err := matrix.NewDense(4, 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := wi.Fill([]float64{0.2, 0.1, 0.3, -0.1}); err != nil {
		fmt.Println(err)
		return
	}
	model, err := lstm.Import(lstm.Weights{
		Input: wi, Recurrent: wr, Bias: []float64{0, 0, 0.05, 0}, Hidden: 1,
	})
	if err != nil {
		fmt.Println(err)
		return
	}

	// Second row is masked after its first observation; all padding remains finite.
	input := lstm.Tensor3{Batch: 2, Time: 3, Features: 1,
		Data: []float64{0.2, 0.3, 0.4, 0.5, 0, 0}}
	result, err := model.InferSequence(context.Background(), input, []int{3, 1}, nil)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("sequence shape: %dx%dx%d\n", result.Sequence.Batch, result.Sequence.Time, result.Sequence.Features)
	fmt.Println("second-row padding zero:", result.Sequence.Data[4] == 0 && result.Sequence.Data[5] == 0)
	fmt.Println("final state shape:", result.Final.Hidden.Rows(), result.Final.Hidden.Cols())
	// Output:
	// sequence shape: 2x3x1
	// second-row padding zero: true
	// final state shape: 2 1
}
```

For model deployment, the application retains fixed training-time feature
order/scaling, timestamp quality, artifact authenticity, chronological
holdout evidence, and a separately trained downstream head or decision
policy. The numeric `ModelID` matches parameter bytes; it is **not** a
digital signature. Chunk replay can be bitwise identical within the same
Go kernel, while PyTorch CPU `float64` compatibility is verified against a
**pinned numerical fixture** at `1e-12` absolute tolerance, not promised
bit-for-bit across every PyTorch backend.

See [`docs/LSTM.md`](docs/LSTM.md) for the complete ABI/overflow/masking
contracts, [`lstm/doc.go`](lstm/doc.go) for package-level laws,
[`lstm/example_test.go`](lstm/example_test.go) for six executable scenarios,
the four-device telemetry example in [`lstm/example_test.go`](lstm/example_test.go),
and [the Tutorial](docs/TUTORIAL.md#11t-lstm-foundations-frozen-ifgo-inference-and-masked-streaming)
for masked streaming from a frozen model.

---

## Current contract documentation

| Layer                          | File                                             | Purpose                                                                                                                          |
|:-------------------------------|:-------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|
| Tutorial                       | [`docs/TUTORIAL.md`](docs/TUTORIAL.md)           | First guided entry into graph/matrix concepts, package roles, examples, and best practices.                                      |
| Core spec                      | [`docs/CORE.md`](docs/CORE.md)                   | Graph model, capability law, deterministic enumeration, locks, topology invariants.                                              |
| BFS spec                       | [`docs/BFS.md`](docs/BFS.md)                     | Hop-distance math, BFS result semantics, partial results, weak components.                                                       |
| DFS spec                       | [`docs/DFS.md`](docs/DFS.md)                     | Post-order semantics, cycle witnesses, DFS forest, topological sort.                                                             |
| Shortest paths spec            | [`docs/DIJKSTRA.md`](docs/DIJKSTRA.md)           | Full SSSP vs four target methods, consistency proofs, balanced potentials, exact arithmetic and runnable scenarios.              |
| MST spec                       | [`docs/MST.md`](docs/MST.md)                     | Exact Prim/Kruskal, KMB Steiner approximation, tree-covering walks, original IDs, numeric/error contracts.                       |
| Flow spec                      | [`docs/FLOW.md`](docs/FLOW.md)                   | Max-flow/min-cut math, residual networks, Ford-Fulkerson, Edmonds-Karp, Dinic.                                                   |
| DTW spec                       | [`docs/DTW.md`](docs/DTW.md)                     | Dynamic programming, numeric classification, windows, penalties and path recovery.                                               |
| DTW exact search               | [`docs/DTW_SEARCH.md`](docs/DTW_SEARCH.md)       | Bounded classification, fixed-length top-K, certified pruning, budgets and worker law.                                           |
| Fractional PAA                 | [`docs/PAA.md`](docs/PAA.md)                     | Exact step-function geometry, output ownership, nearest-even arithmetic, wrappers and integration.                               |
| Symbolic SAX                   | [`docs/SAX.md`](docs/SAX.md)                     | Canonical words, Euclidean MINDIST proof, versioned identity, errors and false-DTW-pruning counterexample.                       |
| TFSAX research gate            | [`docs/TFSAX.md`](docs/TFSAX.md)                 | Exact DTW truth versus heuristic SAX shortlists, temporal provenance, empirical metrics and scientifically honest non-admission. |
| Series primitives              | [`docs/SERIES.md`](docs/SERIES.md)               | Exact integer grids, causal transforms, result masks, numeric safeguards, index windows.                                         |
| Deterministic RNG streams      | [`docs/RNG.md`](docs/RNG.md)                     | Pinned PCG-DXSM128/64, SHA-256 state mapping, strict uniform, Gaussian spare and replay.                                         |
| Monte Carlo driver             | [`docs/MONTECARLO.md`](docs/MONTECARLO.md)       | Path-keyed replay, bounded dispatch, vector moments, exact empirical quantiles, memory limits and prefix failures.               |
| GBM exact lognormal paths      | [`docs/GBM.md`](docs/GBM.md)                     | Model equations, unit dimensions, input precedence, robust price publication and Monte Carlo bridge.                             |
| GJR-GARCH conditional variance | [`docs/GJRGARCH.md`](docs/GJRGARCH.md)           | Strict causality, asymmetric shock mathematics, stationary forecasts, QMLE gradient, KKT projection and release policy.          |
| ARIMA mean/level forecasting   | [`docs/ARIMA.md`](docs/ARIMA.md)                 | Conditional Gaussian fit, burn-in, Schur roots, anchor recovery, impulse variance and operational limits.                        |
| Frozen IFGO LSTM inference     | [`docs/LSTM.md`](docs/LSTM.md)                   | Combined PyTorch biases, tensor/length masks, exact overflow, ModelID, streaming replay and CPU-double oracle boundaries.        |
| Merton jump-diffusion          | [`docs/JUMPDIFFUSION.md`](docs/JUMPDIFFUSION.md) | Compound Poisson log-jump law, compensation, bounded PTRS, RNG replay, errors and source attribution.                            |
| Stable statistics              | [`docs/STATS.md`](docs/STATS.md)                 | Welford/Chan, exact mean sum, scaled M2, fitted scaling, sentinels, immutable inference.                                         |
| Stable Ridge regression        | [`docs/RIDGE.md`](docs/RIDGE.md)                 | Positive-Lambda augmented QR, strict intercept policy, wide/collinear designs, nonzero tiny coefficients and frozen prediction.  |
| Dependent index bootstrap      | [`docs/BOOTSTRAP.md`](docs/BOOTSTRAP.md)         | Moving/Circular/Stationary laws, exact integer mapping, replay, budget and partial rows.                                         |
| Sample autocorrelation         | [`docs/ACF.md`](docs/ACF.md)                     | Common/lag-adjusted ACF, full-window centering, Toeplitz PSD proof and numeric limitations.                                      |
| Partial autocorrelation        | [`docs/PACF.md`](docs/PACF.md)                   | Common-n covariance, Durbin-Levinson recursion, V_m, stable arithmetic and error precedence.                                     |
| Wilder RSI                     | [`docs/RSI.md`](docs/RSI.md)                     | Stable ratio math, exact seed/warmup, batch/stream equivalence, sentinels and checkpoint ABI.                                    |
| SMA-seeded MACD                | [`docs/MACD.md`](docs/MACD.md)                   | Appel EMA seed/recurrence, signed values, aligned masks, batch/stream parity, checkpoint safety.                                 |
| Compensated MFI                | [`docs/MFI.md`](docs/MFI.md)                     | Typical-price OHLCV flow, transition windows, Neumaier rebase, warmup masks and versioned state.                                 |
| Cumulative OBV                 | [`docs/OBV.md`](docs/OBV.md)                     | First-bar zero, signed close comparisons, Neumaier compensation, atomic Push and snapshot trust model.                           |
| Chaikin A/D                    | [`docs/AD.md`](docs/AD.md)                       | First-bar weighted contribution, stable close-location geometry, Neumaier rebase, global errors and fixed snapshot ABI.          |
| Finite-state HMM               | [`docs/HMM.md`](docs/HMM.md)                     | Causal log-domain filtering, retrospective smoothing, Viterbi, multi-sequence MAP-EM, error and identity laws.                   |
| Fitted covariance PCA          | [`docs/PCA.md`](docs/PCA.md)                     | Train-only orthogonal projection, checked covariance eigenpairs, numerical rank and leakage-safe reconstruction.                 |
| Stable OLS regression          | [`docs/OLS.md`](docs/OLS.md)                     | Pivoted Householder QR, explicit intercept, physical coefficients, SSE, frozen Model and holdout pitfalls.                       |
| Grid spec                      | [`docs/GRID_GRAPH.md`](docs/GRID_GRAPH.md)       | Grid/lattice graph modeling and pathfinding-oriented construction.                                                               |
| Matrix spec                    | [`docs/MATRICES.md`](docs/MATRICES.md)           | Canonical dense/graph/statistics/factorization/economy-SVD reference; this is the authoritative matrix guide.                    |
| Matching spec                  | [`docs/MATCH.md`](docs/MATCH.md)                 | Exact-or-error weighted Blossom, Greedy, original-index subset mapping, cost and ownership laws.                                 |
| TSP spec                       | [`docs/TSP.md`](docs/TSP.md)                     | Tour optimization, exact vs approximate methods, metric assumptions.                                                             |
| Engineering standard           | [`docs/lvlath_UES.md`](docs/lvlath_UES.md)       | Repository engineering standard and quality expectations.                                                                        |
| FAQ                            | [`docs/FAQ_&_TIPS.md`](docs/FAQ_%26_TIPS.md)     | Troubleshooting, common pitfalls, usage tips.                                                                                    |
| Contribution guide             | [`CONTRIBUTING.md`](CONTRIBUTING.md)             | Branching, tests, linting, coverage, PR rules.                                                                                   |

---

## Quick start A: graph "lvlath" - shapes, components, cycles, matrix view

This is the first visual example: a graph is not only "nodes and edges"; it is a controllable topology that can be inspected, transformed, and projected into matrix form.

The old `lvlath` letter sketch is preserved, but the implementation style is updated to current error-returning constructors and deterministic package contracts.

```text
Six disconnected glyph components:

   l₁            v           l₂            a               t             h
  top                       top         tail──top         top           top
   │                         │                 │           |             │
 middle      left right   middle    middleL──circle  left-cross-right  middle──right
   │            \ /          │          │      │           │             │      │
 base──tail    center      base──tail  baseL──base       base──tail    base  baseR

Expected component count: 6
Expected cycle count:     1   // the internal cycle in the letter "a"
```

```go
package main

import (
	"context"
	"fmt"
	"sort"

	"github.com/lvlath/go/bfs"
	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

func main() {
	g, err := core.NewGraph()
	if err != nil {
		fmt.Println(err)
		return
	}

	// Static example data. Ignoring AddEdge errors is acceptable here only because
	// all vertices, weights, and capabilities are predetermined. In production,
	// check every returned error.

	// l₁
	_, _ = g.AddEdge("l1_top", "l1_middle", 0)
	_, _ = g.AddEdge("l1_middle", "l1_base", 0)
	_, _ = g.AddEdge("l1_base", "l1_tail_base_right", 0)

	// v
	_, _ = g.AddEdge("v_left", "v_center", 0)
	_, _ = g.AddEdge("v_center", "v_right", 0)

	// l₂
	_, _ = g.AddEdge("l2_top", "l2_middle", 0)
	_, _ = g.AddEdge("l2_middle", "l2_base", 0)
	_, _ = g.AddEdge("l2_base", "l2_tail_base_right", 0)

	// a: the only cyclic glyph.
	_, _ = g.AddEdge("a_tail_top_left", "a_top", 0)
	_, _ = g.AddEdge("a_top", "a_circle_middle", 0)
	_, _ = g.AddEdge("a_circle_middle", "a_circle_base", 0)
	_, _ = g.AddEdge("a_circle_base", "a_circle_base_left", 0)
	_, _ = g.AddEdge("a_circle_base_left", "a_circle_middle_left", 0)
	_, _ = g.AddEdge("a_circle_middle_left", "a_circle_middle", 0)

	// t
	_, _ = g.AddEdge("t_top", "t_cross", 0)
	_, _ = g.AddEdge("t_left", "t_cross", 0)
	_, _ = g.AddEdge("t_right", "t_cross", 0)
	_, _ = g.AddEdge("t_base", "t_cross", 0)
	_, _ = g.AddEdge("t_base", "t_tail_base_right", 0)

	// h
	_, _ = g.AddEdge("h_top", "h_middle", 0)
	_, _ = g.AddEdge("h_middle", "h_base", 0)
	_, _ = g.AddEdge("h_middle", "h_right", 0)
	_, _ = g.AddEdge("h_right", "h_base_right", 0)

	components, err := bfs.Components(context.Background(), g)
	if err != nil {
		fmt.Println(err)
		return
	}

	cycles := g.EdgeCount() - g.VertexCount() + components.Count
	fmt.Println("components:", components.Count)
	fmt.Println("cyclomatic:", cycles)

	// Deterministic induced view of the letter "a".
	aVertices := map[string]bool{
		"a_tail_top_left":     true,
		"a_top":               true,
		"a_circle_middle":     true,
		"a_circle_base":       true,
		"a_circle_base_left":  true,
		"a_circle_middle_left": true,
	}

	sub, err := core.NewGraph()
	if err != nil {
		fmt.Println(err)
		return
	}
	for id := range aVertices {
		_ = sub.AddVertex(id)
	}
	for _, e := range g.Edges() {
		if aVertices[e.From] && aVertices[e.To] {
			_, _ = sub.AddEdge(e.From, e.To, 0)
		}
	}

	mOpts, _ := matrix.NewMatrixOptions(
		matrix.WithUnweighted(),
		matrix.WithAllowLoops(),
	)

	am, err := matrix.NewAdjacencyMatrix(sub, mOpts)
	if err != nil {
		fmt.Println(err)
		return
	}
	degree, err := am.DegreeVector()
	if err != nil {
		fmt.Println(err)
		return
	}

	ids := make([]string, 0, len(am.VertexIndex))
	for id := range am.VertexIndex {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		fmt.Printf("degree[%s]=%.0f\n", id, degree[am.VertexIndex[id]])
	}
}
```
[![Go Playground](https://img.shields.io/badge/Go_Playground-Examplу_№1-blue?logo=go)](https://go.dev/play/p/K1gwwipdNhD)

What this demonstrates:

* `core` gives safe, deterministic topology construction and inspection.
* `bfs.Components` gives weak component discovery without hand-written map iteration.
* The cyclomatic number `|E| - |V| + components` exposes the single cycle in `a`.
* `matrix.NewAdjacencyMatrix` turns a selected subgraph into dense graph algebra.

---

## Quick start B: Hexagram - one weighted graph, many algorithmic views

The classic weighted Hexagram example is the compact demonstration of `lvlath` as a multi-package toolkit: BFS-style shells on an unweighted view, Dijkstra weighted routing, MST backbone extraction, and TSP preparation through a metric matrix.

```text
                               [A]
                              / | \
                  (C-H:9)   3/  |7 \4   (D-F:7)
        (B-G:7)         \   /   |   \   /         (E-G:9)
             [B]────3────[C]──5─┼────[D]────3────[E]
                \       / | \   |  /  | \       /
                6\    7/  | 5\  | /4  |  \6    /7
                  \   /   |   \ |/    |   \   /
                   [F]──3─┼────[G]──5─┼────[H]
                  /   \   |9  / | \   |8  /   \
                2/    6\  |  /4 |  \6 |  /5    \6
                /       \ | /   |   \ | /       \
             [I]────5────[J]──8─┼────[K]────1────[L]
        (I-G:8)         /   \   |8  /   \          (L-G:7)
                  (J-H:7)   2\  |  /3   (K-F:8)
                              \ | /
                               [M]
```

Use this kind of graph when you want one fixture that is complex enough for real algorithms but still small enough to reason about visually.

```go
package main

import (
	"context"
	"fmt"
	"sort"

	"github.com/lvlath/go/bfs"
	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/mst"
)

func main() {
	g, err := core.NewGraph(core.WithWeighted())
	if err != nil {
		fmt.Println(err)
		return
	}

	edges := []struct {
		u, v string
		w    float64
	}{
		{"A", "C", 3}, {"A", "G", 7}, {"A", "D", 4},
		{"B", "C", 3}, {"B", "G", 7}, {"B", "F", 6},
		{"C", "F", 7}, {"C", "J", 9}, {"C", "G", 5}, {"C", "H", 9}, {"C", "D", 5},
		{"D", "F", 7}, {"D", "G", 4}, {"D", "K", 8}, {"D", "H", 6}, {"D", "E", 3},
		{"E", "G", 9}, {"E", "H", 7},
		{"F", "G", 3}, {"F", "K", 8}, {"F", "J", 6}, {"F", "I", 2},
		{"H", "G", 5}, {"H", "L", 6}, {"H", "K", 4}, {"H", "J", 7},
		{"I", "G", 8}, {"I", "J", 5},
		{"J", "G", 4}, {"J", "K", 8}, {"J", "M", 2},
		{"K", "G", 6}, {"K", "L", 1}, {"K", "M", 3},
		{"L", "G", 7},
		{"M", "G", 8},
	}
	for _, e := range edges {
		_, _ = g.AddEdge(e.u, e.v, e.w)
	}

	// BFS is unweighted by contract, so create an unweighted topology view.
	unweighted := core.UnweightedView(g)
	bfsResult, err := bfs.BFS(unweighted, "A")
	if err != nil {
		fmt.Println(err)
		return
	}

	oneHop := make([]string, 0)
	for id, depth := range bfsResult.Depth {
		if depth == 1 {
			oneHop = append(oneHop, id)
		}
	}
	sort.Strings(oneHop)
	fmt.Println("one-hop-from-A:", oneHop)

	// Dijkstra answers cost-sensitive routing.
	d, err := dijkstra.Dijkstra(g, "I", dijkstra.WithPathTracking())
	if err != nil {
		fmt.Println(err)
		return
	}
	distL, _ := d.DistanceTo("L")
	pathL, _ := d.PathTo("L")
	fmt.Println("I-to-L-cost:", distL)
	fmt.Println("I-to-L-path:", pathL)

	// MST gives the cheapest connected backbone.
	mstResult, err := mst.Kruskal(g)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("mst-weight:", mstResult.TotalWeight)

	// Matrix metric closure prepares an all-pairs distance surface for routing or TSP.
	opts, err := matrix.NewMatrixOptions(matrix.WithUndirected(), matrix.WithWeighted())
	if err != nil {
		fmt.Println(err)
		return
	}
	closure, err := matrix.BuildMetricClosure(g, opts)
	if err != nil {
		fmt.Println(err)
		return
	}
	_ = closure
	_ = context.Background()
}
```
[![Go Playground](https://img.shields.io/badge/Go_Playground-Examplу_№1-blue?logo=go)](https://go.dev/play/p/brRONKLM5-9)

---

## Quick start C: weighted service network - routing + flow + matrix policy

This is the practical infrastructure example: a directed weighted network where edge weights can represent latency/cost for routing, and another capacity graph can represent throughput for flow analysis.

```text
Weighted service graph for routing and matrix analysis:

 [edge] ────2────▶ [auth] ──1──▶ [profile] ──2──▶ [db]
    │                │             ▲              │
    │3               │4            │              │0  real zero-weight replication
    ▼                ▼             │              ▼
  [api] ───4────▶ [search] ───2────┘           [backup]
    │                │                            ▲
    │2               │3                           │1
    ▼                ▼                            │
 [cache] ◀──1──── [index] ────────5───────▶ [archive]

Key checks:
  - Dijkstra: cheapest route from edge to db.
  - Matrix: preserve zero-weight db -> backup.
  - Metric closure: publish all-pairs shortest distances.
  - Flow: compute capacity from ingress to storage tier on a related network.
```

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/dijkstra"
	"github.com/lvlath/go/flow"
	"github.com/lvlath/go/matrix"
)

func main() {
	routes, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		fmt.Println(err)
		return
	}

	// Static example data only; check errors in production.
	_, _ = routes.AddEdge("edge", "auth", 2)
	_, _ = routes.AddEdge("edge", "api", 3)
	_, _ = routes.AddEdge("auth", "profile", 1)
	_, _ = routes.AddEdge("auth", "search", 4)
	_, _ = routes.AddEdge("api", "search", 4)
	_, _ = routes.AddEdge("api", "cache", 2)
	_, _ = routes.AddEdge("search", "profile", 2)
	_, _ = routes.AddEdge("search", "index", 3)
	_, _ = routes.AddEdge("index", "cache", 1)
	_, _ = routes.AddEdge("index", "archive", 5)
	_, _ = routes.AddEdge("profile", "db", 2)
	_, _ = routes.AddEdge("db", "backup", 0)
	_, _ = routes.AddEdge("archive", "backup", 1)

	dr, err := dijkstra.Dijkstra(routes, "edge", dijkstra.WithPathTracking())
	if err != nil {
		fmt.Println(err)
		return
	}
	cost, _ := dr.DistanceTo("db")
	path, _ := dr.PathTo("db")
	fmt.Println("route-edge-db-cost:", cost)
	fmt.Println("route-edge-db-path:", path)

	matrixOpts, err := matrix.NewMatrixOptions(
		matrix.WithDirected(),
		matrix.WithWeighted(),
		matrix.WithPreserveZeroWeights(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	adj, err := matrix.NewAdjacencyMatrix(routes, matrixOpts)
	if err != nil {
		fmt.Println(err)
		return
	}
	db := adj.VertexIndex["db"]
	backup := adj.VertexIndex["backup"]
	zeroReplication, err := adj.Mat.At(db, backup)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("zero-replication-weight:", zeroReplication)

	closure, err := matrix.BuildMetricClosure(routes, matrixOpts)
	if err != nil {
		fmt.Println(err)
		return
	}
	_ = closure

	capacity, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		fmt.Println(err)
		return
	}
	_, _ = capacity.AddEdge("ingress", "auth", 8)
	_, _ = capacity.AddEdge("ingress", "api", 10)
	_, _ = capacity.AddEdge("auth", "profile", 5)
	_, _ = capacity.AddEdge("api", "search", 7)
	_, _ = capacity.AddEdge("profile", "storage", 6)
	_, _ = capacity.AddEdge("search", "storage", 8)
	_, _ = capacity.AddEdge("api", "cache", 4)
	_, _ = capacity.AddEdge("cache", "storage", 3)

	flowOpts := flow.DefaultOptions()
	maxFlow, residual, err := flow.Dinic(capacity, "ingress", "storage", flowOpts)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("max-flow-ingress-storage:", maxFlow)
	fmt.Println("residual-vertices:", residual.VertexCount())
}
```
[![Go Playground](https://img.shields.io/badge/Go_Playground-Examplу_№1-blue?logo=go)](https://go.dev/play/p/9uHM7u2HD88)

---

## Design notes

### Determinism

Determinism is not "sort something at the end." It is a cross-package contract.

* `core` owns deterministic graph surfaces: vertices, edges, neighbors.
* `bfs` preserves neighbor order and defines observable traversal state through result fields.
* `dfs` defines returned `Order` as finish order and publishes deterministic cycle witnesses.
* `dijkstra` uses deterministic frontier behavior and strict-improvement updates so equal-cost alternatives do not randomly overwrite witnesses.
* `mst` applies `(Weight, Edge.ID)` order to Kruskal selection, uses a deterministic Prim heap, expands KMB terminal links to original edge identities, and validates caller-owned trees before a covering walk.
* `flow` builds and consumes residual state through deterministic algorithm policy.
* `matrix` uses fixed row-major loops and deterministic graph-adapter publication; LU/QR pivot ties prefer the smallest original index.
* `stats` visits scalar samples in input order and matrix cells row-major; a caller fixes the merge-tree order when grouping floating-point moments.
* `ols` validates rows and features ascending, inherits QR deterministic pivot tie rules and predicts with original columns in the given order.
* `dtw` runs row-major DP and backtracks ties as diagonal, then vertical/up, then horizontal/left.
* `series` scans samples in index order, reports gaps and ranges in that order, and preserves length and validity masks.
* `tsp` uses matrix order, deterministic solver tie-breaks, explicit seed-governed local search, and result fields that report exactness, timeout, and approximation status.
* `builder` and `gridgraph` produce reproducible input topologies instead of hiding randomness inside examples.

The purpose is practical: stable examples, golden tests, CI logs, witness paths, component roots, DP paths, MST edges, tours, and failure classes.


### Error handling

Prefer:

```go
if errors.Is(err, dijkstra.ErrTargetNotFound) {
    // unknown target: not the same as known but unreachable
}
```

Avoid:

```go
if strings.Contains(err.Error(), "target") {
    // brittle and not part of the public contract
}
```

### Weighted vs unweighted algorithms

Do not choose by package popularity. Choose by mathematical objective.

```text
Question                                      Correct primitive
────────────────────────────────────────────────────────────────────────────
Fewest edges / hop layers?                     bfs
Deep traversal / finish order / cycles?        dfs
Cheapest non-negative route from one source?   dijkstra
Cheapest acyclic connected backbone?           mst.Kruskal / mst.Prim
Cheapest approximate terminal-only network?    mst.SteinerTree
Cover every installed tree edge and finish?    mst.SteinerWalk
Maximum feasible throughput?                   flow
All-pairs shortest distances?                  matrix.BuildMetricClosure
Dense topology/covariance/spectral features?   matrix
Stable mean/variance and partition merges?     stats
Fit full-rank factor regression?               ols
Fit regularized correlated/wide regression?    ridge
Train-only standardization / no refit?         stats
Exact timestamp grid / causal sample features? series
Rolling typical-price money flow?              mfi
Close-change-signed cumulative volume?         obv
Candle-close-location cumulative volume?       ad
Symbolic Euclidean screening of windows?       sax
Temporal alignment with phase drift?           dtw
Closed tour through every vertex?              tsp
Generated lattice topology?                    gridgraph
Repeatable benchmark/test fixtures?            builder
```

Important distinctions:

* BFS minimizes edge count, not cost.
* Dijkstra minimizes non-negative path cost, not throughput.
* MST minimizes selected backbone cost, not route distance.
* Flow maximizes capacity, not shortest path.
* Metric closure creates a distance artifact, not original graph topology.
* DTW aligns time-warped sequences, not graph routes.
* `series` operates on supplied sample indices and exact nanosecond checks; it does not fill missing timestamp slots.
* `stats` aggregates finite observations and transforms features with parameters learned on training data only; it does not interpret series validity masks or refit.
* `ols` uses pivoted QR to fit finite, full-rank linear models and returns frozen original-unit coefficients; it never substitutes normal equations, drops rank-deficient factors or refits at inference.
* `ridge` uses loss-preserving augmented QR with mandatory positive Lambda and no intercept penalty; its physical Model remains immutable even when the original design is collinear or wide.
* `sax` quantizes the same fractional PAA geometry into symbolic words; its Euclidean MINDIST is not a metric on words or a certified bound for arbitrary DTW.
* TSP solves a closed tour over a complete finite cost model, not sparse reachability.

---

### Matrix semantics

`matrix` intentionally does not pretend that every dense numeric array means the same thing.

```text
normal adjacency:       0 can mean "no edge"
zero-preserving mode:   finite 0 can mean "real zero-cost edge"; +Inf means absence
metric closure:         cell means shortest-path distance, not an original edge
incidence matrix:       columns are edge identities, not vertex-pair weights
```

### Concurrency

`core` is thread-safe for graph storage and controlled mutation. Algorithm packages should be run against topology that is stable for the duration of the call unless the package explicitly documents otherwise. `matrix.Dense` does not contain locks; concurrent writes require external synchronization. The `montecarlo` coordinator owns mutable `stats.Moments` while workers must receive independent RNG streams and a concurrency-safe `Sampler`; full-run numerical parity does not make an arbitrary sampler safe for shared mutation. Successfully built LU/QR/Cholesky factors retain detached immutable storage and support concurrent `Solve` with private per-call workspaces when RHS inputs remain stable.

---

## Choosing the right package

### By question type

| You ask                                                                                        | Use                                                                | Why                                                                                                   |
|:-----------------------------------------------------------------------------------------------|:-------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|
| "Which vertices are within `k` hops?"                                                          | `bfs.BFS` with depth policy                                        | Hop layers are unweighted traversal.                                                                  |
| "Which weak islands exist?"                                                                    | `bfs.Components`                                                   | Component membership is not a route-cost problem.                                                     |
| "What dependency order is valid?"                                                              | `dfs.TopologicalSort`                                              | DAG ordering is finish-order semantics.                                                               |
| "Where is the cycle?"                                                                          | `dfs.DetectCycles`                                                 | You need a deterministic witness, not all simple cycles.                                              |
| "What is the cheapest route from this source?"                                                 | `dijkstra.Dijkstra`                                                | Non-negative weighted shortest path.                                                                  |
| "Which targets are outside a runtime budget?"                                                  | `dijkstra.WithMaxDistance`                                         | Cutoff policy without deleting topology.                                                              |
| "Which links form the cheapest connected backbone?"                                            | `mst.MinimumSpanningTree`, `mst.Kruskal`, `mst.Prim`               | MST/MSF solves acyclic connectivity, not routing.                                                     |
| "What if the graph is disconnected but I still need per-component backbones?"                  | `mst.WithForest`                                                   | Forest mode is explicit, not a hidden fallback.                                                       |
| "How do I connect only a required set of vertices?"                                            | `mst.SteinerTree`                                                  | KMB approximates a nonnegative graph Steiner tree using original EdgeIDs.                             |
| "How do I cover all edges of the installed tree from A to B?"                                  | `mst.SteinerWalk`                                                  | Iterative optimal covering of the given tree, with repeated passage IDs.                              |
| "What is the max source-to-sink capacity?"                                                     | `flow.Dinic` or `flow.EdmondsKarp`                                 | Flow algorithms reason over residual capacity.                                                        |
| "How do I validate exact sample cadence and missing slots?"                                    | `series.Validate`                                                  | Signed-nanosecond grid checks; no interpolation.                                                      |
| "How do I compute causal lag, differences, or returns?"                                        | `series.Lag` / `Difference` / `SimpleReturns` / `LogReturns`       | Length-preserving values and validity masks.                                                          |
| "How do I compute Wilder relative strength on an ordered positive price feed?"                 | `rsi.ComputePeriod` / `rsi.NewPeriod`                              | Aligned Values+Valid or single-owner zero-allocation streaming; caller handles gaps.                  |
| "How do fast and slow price trends diverge, and what is their smoothed separation?"            | `macd.ComputeDefault` / `macd.NewDefault`                          | Three signed aligned series with readiness masks and transactional single-owner streaming.            |
| "Which typical-price directions received more traded volume in the last P transitions?"        | `mfi.ComputeDefault` / `mfi.NewDefault`                            | Bounded OHLCV MFI, explicit warmup validity and compensated rolling state.                            |
| "What is cumulative volume attached to rising versus falling closes since an explicit origin?" | `obv.ComputeCandles` / `obv.New`                                   | Unbounded compensated OBV, valid from the first candle, with no window or options.                    |
| "How does close location inside each candle affect cumulative volume pressure?"                | `ad.ComputeCandles` / `ad.New`                                     | Unbounded compensated A/D, first bar contributes, no previous close or warmup.                        |
| "How do I compress a window without dropping fractional tails?"                                | `paa.Transform` / `paa.TransformRange`                             | Exact fractional PAA, checked integer geometry and a detached result.                                 |
| "How do I create comparable symbolic time-series signatures?"                                  | `sax.NewEncoder` / `sax.Encode`                                    | Fixed-alphabet symbols, compatible policy hash, detached results.                                     |
| "Can SAX MINDIST prune arbitrary DTW candidates?"                                              | No                                                                 | Euclidean-only lower bound; shortlist retrieval must report Recall@K.                                 |
| "How do I measure repeat cycles or fixed-window serial dependence?"                            | `acf.ComputeDefault` / `acf.Compute`                               | Integer-lag ACF with explicit denominator and pair counts; not causal streaming.                      |
| "Which lag adds linear dependence after controlling for all shorter lags?"                     | `pacf.ComputeCentered` / `pacf.ComputeAtLag`                       | Biased common-n PACF and order-wise prediction variance, not a p-value or fitted AR order.            |
| "How do I enumerate only complete index windows?"                                              | `series.WindowRanges`                                              | Ordered half-open ranges without partial tails.                                                       |
| "How do I compare two jittery sensor signatures?"                                              | `dtw.Align`                                                        | Scalar DTW aligns timing drift.                                                                       |
| "How do I align model-provided frame costs?"                                                   | `dtw.AlignCostMatrix`                                              | Caller owns the local-cost surface.                                                                   |
| "How do I align multivariate sequences?"                                                       | `dtw.AlignMatrix`                                                  | Rows are time steps, columns are features.                                                            |
| "How do I calculate stable population/sample variance?"                                        | `stats.Summarize`                                                  | Stable central moments and explicit DDOF denominator.                                                 |
| "How do I combine independent streaming statistics?"                                           | `stats.Moments.Update` / `Merge`                                   | Error-atomic, count-checked Welford/Chan partition reductions.                                        |
| "How do I scale test features using only training data?"                                       | `stats.FitStandardizer` followed by `Model.Transform`              | Frozen per-column means and scales, no inference refit.                                               |
| "How do I filter a hidden operating regime from live sequential evidence?"                     | `hmm.CategoricalModel.FilterStep` / `hmm.GaussianModel.FilterStep` | Causal posterior, no look-ahead; explicit model identity and prior.                                   |
| "How do I retrospectively explain state changes across an incident?"                           | `hmm.Smooth` / `hmm.Decode`                                        | Noncausal per-step posterior or one maximum-joint path, respectively.                                 |
| "How do I train independent hidden-regime sessions without fabricated transitions?"            | `hmm.FitCategorical` / `hmm.FitGaussian`                           | Per-sequence prior reset and within-sequence xi only; MAP objective diagnostics.                      |
| "How do I reduce correlated feature dimensions without a target y?"                            | `pca.FitWithComponents` / `pca.FitStandardizedWithComponents`      | Frozen training PCA basis; k cannot exceed detected covariance rank.                                  |
| "How do I check a sensor batch against a learned normal subspace?"                             | `pca.Model.Reconstruct` + independently calibrated residual policy | Holdout reconstruction uses no batch refit; alerts are application policy.                            |
| "How do I fit stable factors when columns are duplicated or p>n?"                              | `ridge.FitStandardized` / `ridge.Fit`                              | Explicit positive-Lambda slope penalty, unpenalized intercept, frozen physical coefficients.          |
| "How do I fit reusable linear-factor coefficients with an intercept?"                          | `ols.FitDefault` / `ols.FitStandardized`                           | Original-unit beta and b, training SSE, no heldout refit.                                             |
| "How do I force a physical zero intercept?"                                                    | `ols.FitThroughOrigin` / `FitThroughOriginStandardized`            | No feature centering; b exactly +0 even when scaling.                                                 |
| "How do I fit a full-rank tall calibration model?"                                             | `matrix.SolveLeastSquares` / `FactorQR`                            | Pivoted Householder QR, rank refusal and residual diagnostics.                                        |
| "How do I inspect singular modes in a tall or wide measurement panel?"                         | `matrix.SVDDefault` / `matrix.SVD`                                 | Economy one-sided Jacobi; full singular spectrum; no Gram-based rank shortcut.                        |
| "How do I solve a reusable SPD coupling system?"                                               | `matrix.FactorCholesky`                                            | Symmetry/SPD validation, measured condition and detached solutions.                                   |
| "How do I compute topology features?"                                                          | `matrix.NewAdjacencyMatrix`, `NewIncidenceMatrix`, `DegreeVector`  | Matrix artifacts preserve topology policy.                                                            |
| "How do I compute all-pairs route distances?"                                                  | `matrix.BuildMetricClosure`                                        | Floyd-Warshall distance artifact with `+Inf` semantics.                                               |
| "How do I pair an even subset of sensor, taxi or relay endpoints?"                             | `match.Perfect` / `match.GreedyPerfect`                            | Blossom solves exact MWPM; Greedy is a deliberate weaker pivot heuristic.                             |
| "How do I solve a complete inspection/delivery tour?"                                          | `tsp.SolveMatrix`                                                  | TSP kernels consume matrix cost models.                                                               |
| "How do I solve TSP from graph topology?"                                                      | `tsp.SolveGraph`                                                   | Graph is adapted at the facade boundary, then solved as a matrix problem.                             |
| "How do I get exact TSP for a small instance?"                                                 | `tsp.ExactHeldKarp` / `tsp.BranchAndBound` policy                  | Exactness is explicit and guarded.                                                                    |
| "How do I get a metric symmetric approximation?"                                               | `tsp.Christofides` + `tsp.BlossomMatch`                            | The `1.5` ratio requires exact matching and metric assumptions.                                       |
| "How do I create reproducible stochastic simulation paths?"                                    | `rng.NewFactory().Stream(seed, pathID)`                            | PCG-DXSM pinned streams; logical path IDs, never worker IDs.                                          |
| "How do I parallelize independent Monte Carlo paths with ordered, reproducible moments?"       | `montecarlo.Run` / `EstimateParallel`                              | One RNG per path, bounded 2W window, single ordered reducer, optional budgeted quantiles.             |
| "How do I generate a positive lognormal price path on an irregular grid?"                      | `gbm.Simulate` / `SimulateSeeded`                                  | Model units are explicit, price publication is finite, runtime failures retain only completed points. |
| "How should the next variance react to a negative return shock?"                               | `gjrgarch.FilterStationary` / `Step` / `ForecastVariance`          | Strictly lagged conditional variance with leverage and stationary expected-variance horizons.         |
| "How do I predict future original levels with a fixed ARIMA order?"                            | `arima.Fit` / `Model.Forecast`                                     | Conditional innovations, strict roots, original-level anchors, and marginal innovation variance.      |
| "How do I extract frozen recurrent features from unequal-length telemetry sessions?"           | `lstm.Import` / `Model.InferSequence`                              | IFGO weights, explicit local masks, zero-padded Sequence and detached Final.                          |
| "How do I resume one LSTM inference across ordered chunks?"                                    | `Model.InferFinal` / `Model.InferChunk`                            | Reuse both Hidden/Cell, verify ModelID and recompute per-chunk lengths.                               |
| "How do I model sudden positive or negative price jumps on that GBM grid?"                     | `jumpdiffusion.Simulate` / `SimulateSeeded`                        | Compensated Merton log-jump model, bounded discrete Poisson counts and explicit model limitations.    |
| "How do I build repeatable graphs?"                                                            | `builder` / `gridgraph`                                            | Generated fixtures should be deterministic.                                                           |

### Minimum-weight perfect matching: `match` is not `tsp`

`match.Perfect` solves complete undirected minimum-weight pairing on an **even
ordered subset** of original `matrix.Matrix` indices. The result contains
canonical original-index `Pairs`, raw checked `Cost`, algorithm identity and
`Optimal` metadata. It does **not** plan a route or change graph topology.
The independent Blossom numerical engine is exact-or-error; there is no
fallback to Greedy. `match.GreedyPerfect` uses the existing deterministic
last-pivot/swap-delete heuristic and cannot claim global optimality.

```go
// Complete symmetric cost model with three possible matching costs: 5, 4, 101.
dist, err := matrix.NewDense(4, 4)
if err != nil {
    return err
}
if err := dist.Fill([]float64{
    0, 2, 2, 1,
    2, 0, 100, 2,
    2, 100, 0, 3,
    1, 2, 3, 0,
}); err != nil {
    return err
}
exact, err := match.Perfect(dist, []int{0, 1, 2, 3})
if err != nil {
    return err
}
fmt.Println(exact.Pairs, exact.Cost) // [{0 2} {1 3}] 4
```

This fragment belongs inside a function returning `error`; add the imports
`fmt`, `github.com/lvlath/go/matrix` and `github.com/lvlath/go/match`.
For a complete executable 20-task transport handoff, 12-node radio pairing
and MST/odd-degree/Eulerian integration, run
`go test ./match -run '^ExamplePerfect_' -count=1` and see
[`docs/MATCH.md`](docs/MATCH.md) and [`match/example_test.go`](match/example_test.go).
The suite explicitly demonstrates Greedy cost **101** on the four-node trap
versus exact cost **4**. The TSP Christofides adapter validates the detached
match and owns all multigraph mutation: `tsp -> match -> matrix`, never
`match -> tsp`. The 1.5 Christofides theorem additionally requires a
symmetric **metric** TSP matrix; matching alone does not establish metricity.

### By failure mode you need to avoid

| Risk                                                                                           | lvlath answer                                                                                                |
|:-----------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|
| Map-order flakiness in examples/tests                                                          | `core` deterministic surfaces and package tie-break laws.                                                    |
| Invalid topology reaches algorithms                                                            | Graph capability validation and package-level input validators.                                              |
| BFS used for weighted routing                                                                  | Separate BFS hop semantics from Dijkstra weighted semantics.                                                 |
| DFS order misread as discovery order                                                           | DFS `Result.Order` is documented finish order.                                                               |
| Unknown target treated as unreachable                                                          | Dijkstra keeps target-domain errors separate from `+Inf` distances.                                          |
| Runtime route policy deletes graph structure                                                   | Dijkstra thresholds/cutoffs model the run without mutating topology.                                         |
| Disconnected MST silently becomes forest                                                       | `mst.WithForest` is explicit; strict MST returns disconnection.                                              |
| Terminal-only optimization confused with full MST                                              | `mst.SteinerTree` uses required terminals and original edges; it is an approximation, not an exact MST.      |
| Network cost mistaken for inspection-route cost                                                | `SteinerResult.TotalWeight` counts unique links; `WalkResult.Distance` counts actual passages.               |
| Repeated walk EdgeIDs removed as duplicates                                                    | `SteinerWalk` repeats original IDs by design while covering every selected tree edge.                        |
| Non-finite MST weights corrupt sorting/heap behavior                                           | MST rejects `NaN`, `+Inf`, `-Inf` before kernel execution.                                                   |
| Flow output hides residual state                                                               | Flow result includes residual-network semantics.                                                             |
| DTW path requested from rolling memory                                                         | DTW memory/path policy is explicit; path needs full accumulated storage.                                     |
| Missing timestamps silently treated as samples                                                 | `series.Validate` reports whole-slot gaps; no values or time slots are synthesized.                          |
| PAA fractional remainder dropped or padded with zeros                                          | `paa.Transform` uses integer exact-overlap geometry; neither truncation nor padding occurs.                  |
| SAX breakpoint labels mistaken for encoding identity                                           | `EncodingID` hashes actual threshold content and numeric policy; `BreakpointID` is audit-only.               |
| Euclidean SAX MINDIST incorrectly used to prune DTW                                            | Do not prune; the two distance objectives differ and a positive MINDIST may accompany zero DTW.              |
| An undefined TFSAX score promoted to production or a two-arm study presented as four-arm proof | `tfsax.Run` refuses; only available exact/SAX arms are measured, and heuristic `Exact` stays false.          |
| LagAdjusted clipped to [-1,1] or treated as Toeplitz PSD                                       | Preserve the estimator convention; use CommonDenominator for exact-arithmetic PSD compatibility.             |
| Block-bootstrap path silently replaced by IID modulo-biased shuffling                          | `bootstrap` uses exact rejection, explicit block topology, per-replicate streams and versioned metadata.     |
| Worker-tied seeds or shared mutable pseudo-random streams                                      | Use `rng.Factory.Stream(seed,uint64(logicalPath))` and reduce path results by stable index.                  |
| Worker-count-dependent Monte Carlo means/variance or an unbounded pending queue                | `montecarlo` uses path-ID streams, coordinator-only `stats.Moments`, and a `2W` issue window.                |
| A partial Monte Carlo prefix interpreted as a completed risk estimate                          | Check `err`/`Complete`; `Completed` is a contiguous prefix, quantiles nil on failure.                        |
| GBM annualized drift silently applied to a seconds grid                                        | `gbm` rejects unit mismatches; callers must convert drift and volatility separately.                         |
| Representable GBM price rejected through intermediate `S*exp` overflow                         | Anchored log-growth and wide diffusion arithmetic defer conversion to a safe publication boundary.           |
| GBM underflow replaced by a fake positive epsilon or partial payoff                            | Zero/Inf prices yield `ErrOverflow`; only completed prefixes survive, never successful maturity substitutes. |
| Current GJR residual used to compute its own conditional variance                              | The observation at t contributes only to h[t+1]; never update h[t] from epsilon[t].                          |
| GJR likelihood silently clamps variance below the floor                                        | Reject infeasible candidates; clipping computed h[t] changes both loss and analytic gradient.                |
| GJR optimizer candidate treated as successfully fitted                                         | Check `err==nil` AND `Diagnostics.Converged`; `IsNil` is only pointer presence.                              |
| ARIMA drift `c` mistaken for stationary mean or long-run price growth                          | `c` is the differenced intercept; when stationary, mean `c/(1-sum(phi))`.                                    |
| Integrated ARIMA forecasts accidentally start at level/slope zero                              | Anchor `d=1` on last level, `d=2` on last level and last first difference.                                   |
| Conditional SSE includes burn-in or uses raw `n` for variance                                  | First `max(p,q)` differenced observations condition only; profile over `EffectiveN`.                         |
| ARIMA root boundary silently clipped or nonconverged fit deployed                              | Reject Schur unit roots; classify `ErrNotConverged` and retain only a diagnostic candidate.                  |
| LSTM gate rows or combined biases are imported in the wrong ABI                                | Require i/f/g/o contiguous H-row blocks and precombined input+recurrent bias; verify golden parity.          |
| LSTM padding updates state or repeats the last hidden feature                                  | Masked steps must skip both states; Sequence tails are exactly zero, while Final keeps last valid state.     |
| LSTM affine overflow is hidden behind sigmoid saturation                                       | Reject exact unrepresentable affines as ErrOverflow; do not invent a finite gate.                            |
| Horizon covariance materialized or uncertainty interpreted as complete                         | Linear impulse-response marginal variance excludes parameter uncertainty and joint horizon covariance.       |
| Merton log-jump mean `m` used as the compensator `k`                                           | Compensated Merton drift uses `lambda*expm1(m+s^2/2)`, never `lambda*m`.                                     |
| Poisson count replaced by a Gaussian at high intensity                                         | Use bounded PTRS at mean>=30, not a continuous approximation.                                                |
| An incomplete jump path silently priced as a maturity scenario                                 | Reject `!Complete` in `montecarlo.Sampler`; retain diagnostic prefix only.                                   |
| Full-window ACF mistaken for a causal streaming feature                                        | Fit only a fixed permitted window; appending observations can revise every lag.                              |
| PACF lag-adjusted covariance, in-place coefficients or epsilon variance clamp                  | Use common-n covariance, disjoint coefficient rows, and `ErrSingular` instead of fabricated V_m.             |
| Warmup zero mistaken for a computed zero                                                       | `series.TransformResult.Valid` marks computed cells; retain it with `Values`.                                |
| MFI zero-volume or unchanged-price bars silently dropped                                       | Retain every observed transition slot; a missing candle is a separate feed-integrity error.                  |
| MFI checkpoint Config mistaken for integrity proof                                             | Authenticate the entire Snapshot and persist source offset; Config is only an algorithm/period identity.     |
| First OBV candle incorrectly contributes its volume                                            | Seed the first accepted Close only; its valid OBV must be exactly positive zero.                             |
| OBV correction is lost after a large opposite-signed volume                                    | Preserve sum and compensation across Push and Snapshot/Restore; do not flatten them.                         |
| OBV checkpoint Config is treated as proof of authentic feed history                            | Authenticate the entire snapshot and its external source cursor; Config hashes a public algorithm ID.        |
| A/D bar zero is dropped or the multiplier forms 2*Close near MaxFloat64                        | Include the first weighted contribution; use bounded distances and the flat-range branch.                    |
| A/D loses low-order corrections or accepts an unauthenticated checkpoint                       | Preserve both Neumaier coordinates; authenticate snapshot plus source epoch/offset externally.               |
| DTW over-warps a signal                                                                        | Use window and slope penalty deliberately.                                                                   |
| Feature scale dominates multivariate DTW                                                       | Normalize before `dtw.AlignMatrix`; it uses squared L2 row cost.                                             |
| High-offset variance vanishes or intermediate `M2` overflows                                   | `stats.Moments` uses scaled Welford/Chan; exact binary sum preserves the final mean.                         |
| Holdout data refits a normalizer or constant features vanish                                   | `stats.Standardizer` freezes training parameters; `UnitScaleConstant` preserves novel deviations.            |
| PCA loadings drift after observing a holdout batch                                             | `pca.Model.Transform` retains train-only origin, scales and loading basis; it does not refit.                |
| Unconverged Jacobi result or negative eigenvalue is hidden                                     | PCA rejects failed eigenpair/PSD checks and never silently substitutes ThinSVD.                              |
| Zero-weight edge disappears in matrix adjacency                                                | Use zero-preserving matrix policy; absence becomes `+Inf`.                                                   |
| Metric closure exported as fake topology                                                       | Matrix refuses metric-closure export as ordinary graph topology.                                             |
| TSP receives sparse unresolved `+Inf` matrix                                                   | Use metric closure only when indirect travel is domain-valid; final kernels need finite completeness.        |
| Christofides ratio claimed with greedy matching                                                | `tsp.BlossomMatch` is required for the formal `1.5` ratio.                                                   |
| Branch-and-Bound timeout is mistaken for optimality                                            | Inspect TSP `Result.TimedOut` and TSP `Result.Optimal`.                                                      |
| Local-search TSP treated as proof                                                              | 2-opt/3-opt improve tours; they do not certify global optimum.                                               |
| Randomized tests/benchmarks drift                                                              | Use fixed seeds or deterministic `builder` fixtures.                                                         |
| Docs describe fictional APIs                                                                   | `{package}/doc.go`, `docs/{PACKAGE}.md`, examples, and tests must stay synchronized.                         |
| QR rank failure silently treated as direct SVD                                                 | QR stays full-column-rank only; use standalone `matrix.SVD` for decomposition, not an implicit solve.        |
| SVD nonconvergence used as a successful factorization                                          | Reject `ErrNotConverged` for downstream computation; optional returned factors are diagnostic only.          |

---

## Documentation and contribution workflow

Documentation quality is part of the project standard. A package is considered mature when it has:

1. `{package}/doc.go` with implemented API, options, errors, and complexity.
2. `docs/{PACKAGE}.md` with math, proofs/invariants, pseudocode, diagrams, examples, and pitfalls.
3. Scenario-driven examples in `examples/` or `example_test.go`.
4. Validation, medium, and special tests.
5. Benchmarks where performance matters.
6. CI passing: `go test`, `go vet`, and `golangci-lint`.

For contribution rules, branch flow, tests, linting, and coverage expectations, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## License

This project is licensed under **GNU Affero General Public License v3.0 only** (`AGPL-3.0-only`).

Important for server-side and enterprise usage: AGPL-3.0 requires that if you run this software on a server and allow users to interact with it over a network, you must make the corresponding source code available under the license terms.

If you want to use `lvlath` in a proprietary environment without AGPL source-disclosure obligations, contact the maintainer for commercial licensing terms:

**katalvlaran@gmail.com**

---

## Support

* GitHub Issues: `github.com/lvlath/go/issues`
* API reference: `pkg.go.dev/github.com/lvlath/go`
* Maintainer contact: `katalvlaran@gmail.com`

---

**lvlath** - deterministic graph algorithms, stable statistics, fitted feature scaling, dense algebra, and engineering-grade contracts for Go developers who want reproducible answers instead of accidental behavior.
