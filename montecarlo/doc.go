// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo executes reproducible, bounded-parallel Monte Carlo
// experiments over independently generated finite, fixed-dimensional vectors.
// It is a simulation driver and reduction engine, not a financial, physical,
// probabilistic, or domain-specific stochastic model.
//
// Domain and public surface:
//   - Sampler defines Dimension() and Sample(ctx, pathID, rng.RNG). Application
//     code supplies the model, its parameters, units, and draw-consumption law.
//   - Plan specifies Paths, Seed, Workers, RetainSamples, Quantiles, and the
//     MaxSamplesBytes admission limit. Seed=0 is a valid reproducible seed.
//   - Run is the canonical facade. Estimate, EstimateParallel, RunWithQuantiles,
//     and RunWithSamples are policy-setting wrappers over that same Run kernel;
//     they neither change its mathematics nor discard its Result artifact.
//   - Result records Requested, Completed, Seed, RNG, Complete, optionally the
//     retained Samples matrix and exact empirical Quantiles, and sample Mean,
//     SampleVariance, and MeanStandardError vectors when representable.
//   - (*Result).IsNil implements core.Nilable; it reports nil pointer identity,
//     not completion status, sample sufficiency, or absence of an error.
//
// Shape and input law:
//   - Paths must be at least two. Workers must be nonnegative; zero selects one
//     worker. Effective W=min(Paths,max(1,Workers)), after checked conversion.
//   - Dimension() is called once before any Sample; d must be positive. Each
//     successful Sample must return precisely d finite float64 components.
//   - Quantiles is empty or a strictly increasing sequence of distinct finite
//     probabilities p in [0,1]. Levels are never silently sorted or deduplicated.
//   - A nil context is ErrInvalidOption. An already cancelled context returns
//     its context error without calling Dimension or Sample.
//   - A nil or recognized core.Nilable-style typed-nil Sampler is ErrNilInput.
//     An arbitrary typed-nil implementation without IsNil cannot be detected
//     without introspection; such a value must not be supplied by callers.
//
// Determinism and concurrency law:
//   - For logical path ID i, a worker creates a fresh stream exclusively with
//     rng.Factory.Stream(Plan.Seed,i); it never shares that mutable RNG with
//     another path. The Result records rng.Factory.Version() for replay.
//   - Workers may complete out of order, but only one coordinator applies
//     stats.Moments.Update for path IDs 0,1,... and columns 0,1,... in that order.
//     There is no worker-local moment merge, shared accumulator, or reduction lock.
//   - Full successful results are bitwise reproducible across worker counts
//     on the same platform, implementation, and RNG revision, PROVIDED that
//     Sample is itself deterministic, concurrency-safe, and independent of
//     scheduling, wall-clock time, and external mutable state.
//   - Cancellation timing, the number of paths committed before interruption,
//     and the first observed competing error are not worker-count invariant.
//   - Each worker copies a returned sample slice before sending its outcome.
//     Callers must not mutate Plan.Quantiles during Run, and Samplers must not
//     mutate returned slices asynchronously or share mutable RNG instances.
//
// Bounded scheduling and memory law:
//   - The coordinator issues only IDs in [nextReduce,nextReduce+2*W), capped
//     by Paths. Until the first missing ID is committed, that window cannot
//     advance. Pending results remain O(W*d), even if path zero is slow.
//   - Without retention or quantiles, no Paths-by-d sample matrix is allocated;
//     additional driver memory is O(d+W*d), excluding Sampler-owned memory.
//   - With RetainSamples or Quantiles, exactly one Paths-by-d raw sample backing
//     is reserved. Quantiles additionally require a Paths-element sorting
//     column and a len(Quantiles)-by-d output matrix.
//   - The checked admission formula is 8*Paths*d + I(q>0)*8*Paths + 8*q*d
//     bytes, where q=len(Quantiles). MaxSamplesBytes=0 selects 64 MiB.
//     ErrOverflow covers unrepresentable sizes; ErrBudget covers a request
//     exceeding the configured buffer ceiling, both before any Sample call.
//   - This limit is NOT a bound on goroutine stacks, stats.Moments, channel/map
//     metadata, Sampler allocations, process RSS, or OS allocation failure.
//   - Partial retained Samples exposes Completed rows without duplicating the
//     full budgeted Dense backing. That original Paths-by-d backing remains
//     resident until the partial matrix is released; Clone allocates an
//     independent Completed-by-d matrix if compact ownership is required.
//
// Numeric and quantile law:
//   - One stats.Moments per feature computes a protected mean and scaled Welford
//     central moment in canonical path order. Variance is SAMPLE variance with
//     Bessel correction (ddof=1); it is not population variance.
//   - MeanStandardError[j]=sqrt(SampleVariance[j])/sqrt(Completed), avoiding
//     avoidable underflow of variance/Completed. It estimates sampling error
//     of the mean under independent paths, NOT uncertainty of model assumptions,
//     calibrated parameters, or a confidence bound on physical outcomes.
//   - Exact empirical quantiles use independently sorted coordinate samples
//     and the type-7 index h=(Paths-1)*p, i=floor(h), f=h-i. The result is a
//     floating-point interpolation of the adjacent order statistics, not an
//     approximation by histogram, t-digest, or subsampling.
//   - Opposite-sign endpoints use (1-f)*a+f*b instead of a+f*(b-a), whose
//     intermediate difference may overflow. Same-sign endpoints use the
//     bounded difference form; p=0 and p=1 preserve the observed extremes.
//   - All public scalar moments and quantiles must be finite. Unrepresentable
//     final variance or other required numeric output yields ErrOverflow,
//     even when some other statistic (for example, a median) is representable.
//
// Errors and partial-result law:
//   - Validation precedence is: context -> Plan options -> Sampler presence
//     -> Dimension -> checked memory sizes and budget -> execution.
//   - Preflight rejection returns (nil,error) without sampling. Once work has
//     started, a runtime, sampler, or context failure returns (*Result,error)
//     with Complete=false and the successfully reduced contiguous prefix
//     [0,Completed). Successful later IDs beyond any missing path are excluded.
//   - If Completed<2, Mean, SampleVariance and MeanStandardError are nil. On a
//     successful summary with Completed>=2, all three vectors have length d;
//     failure of final numeric materialization may leave them nil instead.
//   - Quantiles is nil on every unsuccessful or incomplete run and whenever
//     no quantile levels were requested. QuantileLevels copies requested levels
//     even if a started run fails; Samples is nil if retention is off or
//     Completed=0. Complete is the authoritative success flag: Completed==Paths
//     alone does not guarantee valid final moments or requested quantiles.
//   - ErrInvalidOption, ErrNilInput, ErrShape, ErrNonFinite, ErrDomain,
//     ErrBudget, and ErrOverflow are stable errors.Is classifications. Context
//     and Sampler error identity is preserved; dependency errors are joined
//     when mapped to a package sentinel. Error text is not a protocol.
//   - On an observed terminal failure, new dispatch stops, the child context
//     is canceled, accepted jobs are drained, and workers are joined. A Sampler
//     ignoring cancellation may delay return indefinitely. No arbitrary Go
//     goroutine can be forcibly stopped by this package.
//
// Ownership and unsupported guarantees:
//   - Successful result slices and matrices are detached from Sample buffers
//     and from the caller's Plan.Quantiles. Published artifacts are caller-owned;
//     concurrent mutation of shared public matrices/slices is not synchronized.
//   - No model fitting, importance sampling, quasi-Monte-Carlo, cross-platform
//     bitwise guarantee, automatic confidence interval, tail-risk estimator,
//     random-number security guarantee, or streaming exact quantiles without
//     sample materialization is provided by this package.
//   - The package never calls panic, recover, or reflect. A panic inside a
//     Sample worker is NOT converted into an error or a zero vector. Since Go
//     cannot transfer such a panic to the Run caller without recovery, an
//     unhandled worker panic has the normal process-terminating Go behavior;
//     pool cleanup is NOT guaranteed in that situation.
//
// Complexity:
//   - Mean-only: O(Paths*d + sampler work) total CPU work and O(d+W*d)
//     additional driver memory. Retention adds O(Paths*d) persistent memory.
//   - With q>0 exact quantiles: O(Paths*d*log(Paths) + sampler work) total CPU
//     work and O(Paths*d+Paths+q*d+W*d) space; sorting uses one column buffer.
//
// AI-Hints:
//   - Never replace coordinator-owned ordered updates with per-worker sums or
//     tie RNG stream identity to worker count; both break worker-count parity.
//   - Never infer the full experiment succeeded from Completed==Requested or
//     infer memory safety from MaxSamplesBytes alone; inspect Complete/error,
//     and budget Sampler/RNG/worker allocations separately.
//   - Retain the seed, RNG version, model revision, input calibration and
//     parameter assumptions alongside results for meaningful experiment replay.
//   - See docs/MONTECARLO.md and montecarlo/example_test.go for exact result
//     contracts, sign-safe quantiles, interruption handling, and domain examples.
package montecarlo
