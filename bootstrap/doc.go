// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap provides reproducible block-bootstrap index resampling for
// dependent observations through the canonical SampleIndices facade.
//
// Domain and public artifacts:
//   - Plan fixes the method, source-independent output row length, block length,
//     number of replicates, and explicit uint64 Seed.
//   - Kind selects MovingBlocks, CircularBlocks, or StationaryBlocks.
//   - Result owns completed [][]int index rows, Completed count and RNG revision.
//   - The convenience facades SampleMovingIndices, SampleCircularIndices and
//     SampleStationaryIndices delegate to the same validated canonical method.
//
// Mathematical contracts:
//   - MovingBlocks draws starts uniformly in [0,rows-B] and copies B adjacent
//     source positions; a final partial block is truncated, never wrapped.
//   - CircularBlocks draws starts uniformly in [0,rows) and advances cyclically
//     within each fixed block; the final block is truncated.
//   - StationaryBlocks starts uniformly and restarts independently at each next
//     position with exact probability 1/B; otherwise it advances cyclically.
//   - For B=1, all three methods consume one uniform index draw per output
//     position and therefore return identical rows for identical streams.
//     StationaryBlocks consumes no redundant Bernoulli words.
//   - All methods require 1<=B<=rows; Stationary B is an expectation rather
//     than a maximum run length.
//
// Randomness and numerical policy:
//   - The accepted rng.NewFactory supplies a separate Stream(Seed,uint64(r))
//     for each zero-based replicate r; zero Seed is reproducible and not defaulted.
//   - Uniform integer selection uses rejection with uint64 threshold
//     (-bound)%bound before modulo mapping, avoiding systematic modulo bias.
//   - Accepted remainders are strictly below the validated native-int bound;
//     a bounded digit reconstruction avoids unchecked uint64-to-int narrowing
//     without string conversions or changing uniformity and draw consumption.
//   - Result.RNG combines factory.Version() with /bootstrap-index-v1,
//     identifying generator version, stream derivation, and draw consumption.
//     Distinct logical streams do not prove cryptographic independence.
//   - Dimensions are integer counts, not floating quantities: NaN, Inf, rounding,
//     convergence thresholds and value-domain numeric policies do not apply.
//
// Validation and error order:
//   - Nil context -> ErrInvalidOption; already-canceled context -> ctx.Err().
//   - Unknown Kind, rows<1, Length<1, Replicates<1, B<1 or B>rows ->
//     ErrInvalidOption, before any output buffer allocation or RNG creation.
//   - Checked native-int byte arithmetic -> ErrOverflow if unrepresentable.
//   - Representable output above MaxResultBytes (256 MiB) -> ErrBudget.
//   - Classification uses errors.Is for sentinels and the original context errors.
//
// Cancellation and ownership:
//   - A rejected plan or preexisting cancellation returns (nil, error).
//   - During sampling, each row is constructed privately; a canceled row is
//     discarded, previously completed rows remain published, and ctx.Err()
//     accompanies the Result. Completed always equals len(Indices).
//   - The facade checks context after the last row too. A late cancellation
//     retains all finished rows but still returns the native context error.
//   - If canceled during the first row, Result.Indices is a non-nil empty slice.
//   - Success returns exactly Replicates completed, caller-owned, detached rows.
//   - No original observation values, matrices or mutable RNG workspaces are
//     stored in Result; callers own synchronization if they later mutate it.
//
// Concurrency:
//   - One SampleIndices call generates rows in increasing logical index order.
//   - Each stream is owned by exactly one row; concurrent calls use independent
//     factories and output slices. There is no shared mutable global sampler.
//   - Inputs are value-copy plans. Concurrent mutation of caller-owned results
//     requires explicit caller synchronization.
//
// Non-goals and statistical limits:
//   - The package never gathers observation values, fits models, estimates
//     means/quantiles/standard errors/CI, or chooses a block length.
//   - Correct resampling indices alone do not establish valid inference under
//     arbitrary dependence, nonstationarity or train/test contamination.
//   - The package does not impose an ACF/Ljung-Box null hypothesis: sampling
//     autocorrelated history does not in itself remove autocorrelation.
//   - Block-index resampling is not the analytic Newey-West HAC estimator.
//   - Newey-West covariance estimation is a different analytic operation;
//     block-bootstrap distributions may complement but do not implement it.
//
// Complexity:
//   - For R replicate rows of L indices, expected Time O(R*L) and output
//     Space O(R*L). Rejection sampling has no bounded worst-case word count;
//     context checks also occur inside every draw attempt.
//   - The 256-MiB cap includes logical row data and outer slice headers, not
//     Go allocator metadata, RSS or a guarantee against system-wide OOM.
//   - All methods use separate row allocations; no interrupted row is retained
//     after returning a partial completion prefix.
//
// AI-Hints:
//   - Never replace exact rejection with u%bound or rational restart with
//     float64 threshold sampling; both alter the intended numerical protocol.
//   - Never share a stream across replicates or key streams by goroutine index.
//   - Never publish a partial row, mask cancellation as successful short output,
//     or interpret Result.IsNil() as a completion predicate.
//   - Reproducing results requires the same RNG revision, Seed, Plan, original
//     observation selection and draw-consumption rules. Do not use wall time.
//   - Repeated calls with the same Seed restart replicate numbering at zero;
//     this API does not implement a cross-call replicate offset.
package bootstrap
