// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo exposes one canonical experiment facade and thin wrappers.
// This file delegates every public mode to the same validated driver kernel.
package montecarlo

import "context"

// Run executes independent, reproducible paths and reduces their vector outputs.
// Every path owns a stream derived only from Seed and pathID, while one coordinator
// updates moments in strict ascending path order independent of scheduling.
//
// Implementation:
//   - Stage 1: validate context, options, sampler presence, dimension and budget.
//   - Stage 2: execute bounded worker sampling and single-owner ordered reduction.
//   - Stage 3: publish detached summaries and optional retained samples/quantiles.
//
// Behavior highlights:
//   - At most 2*min(max(1,Workers),Paths) paths are assigned ahead of nextReduce.
//   - Exact quantiles require budgeted raw vectors even without retained Samples.
//   - Complete=false and an error can carry only the valid prefix [0,Completed).
//
// Inputs:
//   - ctx: non-nil context; cancellation is propagated without changing cause.
//   - plan: value-copy experiment policy; its Quantiles are read-only during Run.
//   - sampler: fixed-positive-dimension deterministic path model; concurrent-safe
//     if Workers>1 and responsive to ctx cancellation.
//
// Returns:
//   - *Result: complete experiment or precisely defined partial prefix after work.
//   - error: nil for success; preflight failures always return nil result.
//
// Errors:
//   - ErrInvalidOption: invalid context, paths, workers, dimension or levels.
//   - ErrNilInput: absent or recognized typed-nil sampler.
//   - ErrBudget/ErrOverflow: checked allocation sizes before first Sample.
//   - ErrShape/ErrNonFinite: malformed Sample output, excluded from prefix.
//   - ErrDomain/ErrOverflow: dependency or final statistic failure.
//   - Original sampler or context error, preserved for errors.Is.
//
// Determinism:
//   - RNG streams keyed by (Seed,pathID); reduction only pathIDs 0..Paths-1.
//   - Full successes are bitwise equal for any workers on one platform/backend.
//
// Complexity:
//   - Mean-only: O(Paths*d + sampler work), Space O(d+Workers*d).
//   - Exact quantiles: O(Paths*d*log(Paths) + sampler work), Space O(Paths*d+q*d).
//
// Notes:
//   - MeanStandardError concerns Monte Carlo sampling only, not model risk.
//   - A sampler panic in a worker is an unrecovered Go panic, not a Result error.
//   - Arbitrary sampler code that ignores cancellation can delay Run indefinitely.
//
// AI-Hints:
//   - Never parallelize stats.Moments updates or tie stream IDs to goroutine slots.
//   - Use RetainSamples only when individual paths are needed; quantiles have
//     their own exact, budgeted storage even when Samples remains nil.
func Run(ctx context.Context, plan Plan, sampler Sampler) (*Result, error) {
	configuration, err := validate(ctx, plan, sampler)
	if err != nil {
		return nil, err
	}
	return drive(ctx, configuration, sampler)
}

// Estimate runs a mean/variance-only experiment with serial default execution.
// It avoids materializing path vectors while keeping an explicit replay seed.
//
// Implementation:
//   - Stage 1: assemble Plan with Paths, Seed and zero-value optional modes.
//   - Stage 2: delegate all validation and sampling to Run.
//   - Stage 3: return the canonical Result without dropping diagnostics.
//
// Behavior highlights:
//   - Exactly equivalent to Run(ctx,Plan{Paths:paths,Seed:seed},sampler).
//
// Inputs:
//   - ctx: non-nil, cancellable execution context.
//   - paths: number of trials (>=2); seed: full uint64 replay identifier.
//   - sampler: deterministic model producing finite fixed-dimensional vectors.
//
// Returns: full or partial Result and the unmodified Run error policy.
// Errors: identical to Run with the constructed zero-mode Plan.
// Determinism: one worker; ascending path-ID reduction.
// Complexity: Time O(Paths*d+sampler work), Space O(d) plus bounded worker state.
// Notes: no samples or quantiles are retained; use Run for resource policy options.
// AI-Hints: do not treat SE as confidence in an incorrectly specified model.
func Estimate(ctx context.Context, paths, seed uint64, sampler Sampler) (*Result, error) {
	return Run(ctx, Plan{Paths: paths, Seed: seed}, sampler)
}

// EstimateParallel runs a mean/variance-only experiment with explicit concurrency.
// It is useful for large independent models where raw paths need not be returned.
//
// Implementation:
//   - Stage 1: build Plan from paths, seed and worker count without raw modes.
//   - Stage 2: delegate all option/error handling and execution to Run.
//   - Stage 3: publish the same canonical Result and optional partial prefix.
//
// Behavior highlights:
//   - Workers=0 remains serial; negative workers are rejected by Run.
//
// Inputs:
//   - ctx: active non-nil context; paths: trials >=2; seed: uint64 replay seed.
//   - workers: requested concurrency, >=0; sampler: concurrent-safe if >1.
//
// Returns: same Result and error semantics as Run with the constructed Plan.
// Errors: exactly those of Run, including context/sampler errors via errors.Is.
// Determinism: path streams and reductions are unchanged by workers.
// Complexity: Time O(Paths*d+sampler work), Space O(d+Workers*d).
// Notes: for quantiles or retention use an explicit Plan rather than this shortcut.
// AI-Hints: increasing Workers never changes the statistical reduction order.
func EstimateParallel(ctx context.Context, paths, seed uint64, workers int,
	sampler Sampler,
) (*Result, error) {
	return Run(ctx, Plan{Paths: paths, Seed: seed, Workers: workers}, sampler)
}

// RunWithQuantiles runs a Plan with explicit empirical quantile levels.
// It preserves the plan's retention and budget settings and replaces only
// the quantile level field, without changing the interpolation convention.
//
// Implementation:
//   - Stage 1: value-copy Plan, replacing Quantiles with the passed levels.
//   - Stage 2: let Run validate strict ascending levels and resource preflight.
//   - Stage 3: return a canonical Result with detached QuantileLevels and Quantiles.
//
// Behavior highlights:
//   - Nil/empty levels disable quantiles by the same rule as Plan.Quantiles.
//   - Sample retention remains opt-in through the caller's Plan.RetainSamples.
//
// Inputs:
//   - ctx: cancellable non-nil context; plan: value-copy policy.
//   - levels: read-only strictly ascending finite [0,1] probabilities.
//   - sampler: fixed-dimensional model with per-path RNG ownership.
//
// Returns: canonical Result, with exact quantiles on complete success only.
// Errors: same sentinels and precedence as Run with the amended Plan.
// Determinism: sorted column order and path-indexed streams follow Run.
// Complexity: Time O(d*Paths*log(Paths)+sampler work), Space O(Paths*d+q*d).
// Notes: MaxSamplesBytes still applies; this wrapper never chooses an approximation.
// AI-Hints: use Quantiles={0.01,0.05} for lower-tail P&L, not a VaR sign convention.
func RunWithQuantiles(ctx context.Context, plan Plan, levels []float64,
	sampler Sampler,
) (*Result, error) {
	plan.Quantiles = levels
	return Run(ctx, plan, sampler)
}

// RunWithSamples enables publication of the path-by-feature sample matrix.
// It preserves the caller's remaining Plan options, including quantile levels
// and the memory ceiling, and never returns a matrix on preflight failure.
//
// Implementation:
//   - Stage 1: value-copy Plan and enable RetainSamples.
//   - Stage 2: delegate all checked allocation and computation to Run.
//   - Stage 3: publish the full matrix or only the committed prefix on interruption.
//
// Behavior highlights:
//   - The returned matrix is detached from the sampler's transient buffers.
//
// Inputs:
//   - ctx: active non-nil context; plan: experiment with explicit memory ceiling.
//   - sampler: deterministic fixed-dimensional path generator.
//
// Returns: canonical Result with Samples when Completed>0; nil on preflight error.
// Errors: same sentinels/cause preservation as Run with RetainSamples=true.
// Determinism: ascending path rows, unchanged moments and stream derivation.
// Complexity: Time O(Paths*d+sampler work), Space O(Paths*d+Workers*d);
//
//	quantiles add sorting time and scratch when already requested in plan.
//
// Notes: even a short interrupted prefix retains the budgeted full Dense allocation.
// AI-Hints: choose EstimateParallel if full raw paths are unnecessary.
func RunWithSamples(ctx context.Context, plan Plan, sampler Sampler) (*Result, error) {
	plan.RetainSamples = true
	return Run(ctx, plan, sampler)
}
