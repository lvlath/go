// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo provides a reproducible parallel driver for independent
// multidimensional stochastic experiments. The package is not a stochastic
// model: Sampler owns every equation, domain assumption and physical process.
//
// Domain and public contract:
//   - Sampler exposes Dimension and Sample(ctx,pathID,rng.RNG).
//   - Plan fixes the trial count, per-path seed, concurrency and memory policy.
//   - Result publishes a completed contiguous prefix, summary moments and
//     optionally an owned Samples matrix and exact type-7 Quantiles matrix.
//   - Run is the canonical facade; Estimate, EstimateParallel,
//     RunWithQuantiles and RunWithSamples delegate to Run without another kernel.
//
// Determinism and statistical laws:
//   - Every logical pathID receives rng.Factory.Stream(Plan.Seed,pathID).
//     Factory.Version is recorded for reproducible replay.
//   - Workers may finish in arbitrary order. Exactly one coordinator updates
//     stats.Moments by ascending pathID and then ascending feature index.
//   - Complete results are bitwise identical across worker counts under a
//     deterministic concurrency-safe sampler and the same platform/backend.
//   - SampleVariance uses ddof=1. MeanStandardError is the square root of
//     variance divided by sample count, evaluated without avoidable underflow.
//     Standard error represents sampling precision, not model uncertainty.
//   - Exact quantiles use h=(n-1)*p and sign-aware linear interpolation of
//     adjacent sorted order statistics separately for each feature.
//
// Shape, options and memory:
//   - Paths>=2, Workers>=0 (zero means one), Dimension()>0 fixed per run.
//   - Quantile levels are finite, unique, strictly ascending within [0,1].
//   - Raw storage exists only for requested RetainSamples or exact Quantiles.
//     MaxSamplesBytes=0 means 64 MiB. Preflight counts one Paths*d raw buffer,
//     one Paths sort scratch column and one len(Quantiles)*d output buffer.
//   - The dispatch window is [nextReduce,nextReduce+2W), W=min(max(1,Workers),
//     Paths). Pending outcomes and worker-owned copied samples use O(W*d)
//     additional memory, independent of total Paths in mean-only mode.
//   - Retained partial prefixes expose Completed*d cells but retain their
//     originally budgeted full backing allocation, without making a copy.
//
// Error precedence and partial results:
//   - Validation order is context, Plan options, sampler presence, Dimension,
//     checked memory sizing and budget, then computation.
//   - Invalid preflight returns (nil,error). Terminal sampler/context/numeric
//     failure returns (Result,error), Complete=false and only committed paths
//     [0,Completed); a later finished path never fills a lower-ID hole.
//   - On Completed<2, all moment vectors are nil; on any failure, Quantiles
//     is nil. Retained Samples is nil for Completed=0.
//   - ErrInvalidOption, ErrNilInput, ErrShape, ErrNonFinite, ErrDomain,
//     ErrOverflow and ErrBudget are stable errors.Is targets; sampler/context
//     causes are returned without rewriting their identity.
//
// Ownership and concurrency:
//   - Plan.Quantiles is copied; result vectors and matrices are detached from
//     sampler buffers. The caller owns public data and may modify it after Run.
//   - The Sampler must allow concurrent Sample calls if Workers>1 and must not
//     asynchronously modify a previously returned output slice.
//   - A mutable rng.RNG belongs to one path invocation; worker scheduling never
//     contributes to stream derivation. There are no coordinator locks or shared
//     mutable reduction states across worker goroutines.
//   - A Sampler that ignores context may delay shutdown. A panic inside Sample
//     is not recovered: an unrecovered panic in a worker terminates normal
//     Go process execution. No reflect, panic or recover is used by this package.
//
// Non-goals:
//   - No pricing model, portfolio loss distribution, physical crack law,
//     importance sampling, quasi-Monte-Carlo scheme, confidence model,
//     approximate quantile fallback or cross-platform bitwise guarantee.
//
// Complexity:
//   - Without exact quantiles: O(Paths*d) driver work plus sampler cost;
//     O(d+W*d) streaming memory if no samples are retained.
//   - With exact quantiles: O(d*Paths*log(Paths)) sorting time and
//     O(Paths*d+Paths+q*d+W*d) memory; q=len(Quantiles).
//
// AI-Hints:
//   - Never merge worker-local float64 totals or schedule beyond the 2W window.
//   - Never interpret a low Monte Carlo SE as proof that a financial or physical
//     model is correctly calibrated. Prefer explicit model validation outside
//     this computational driver and persist the RNG version with experiment IDs.
package montecarlo
