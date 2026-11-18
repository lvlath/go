// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo defines the public experiment carriers and ownership laws.
// This file fixes the sampler's input contract and every Result field invariant.
package montecarlo

import (
	"context"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
	"github.com/lvlath/go/rng"
)

// Sampler generates a fixed-dimensional random trial for a logical path.
// The driver supplies a path-specific mutable RNG and a cancellation context;
// the sampler owns the physical model and must not depend on worker assignment.
//
// Implementation:
//   - Stage 1: Dimension fixes d before any path is dispatched.
//   - Stage 2: Sample(ctx,pathID,Stream(seed,pathID)) produces one vector.
//   - Stage 3: a worker detaches and validates the returned slice before reduction.
//
// Behavior highlights:
//   - Same inputs/stream and model state must yield bitwise-identical samples.
//   - Concurrent Sample calls must be safe when Plan.Workers > 1.
//
// Inputs:
//   - ctx: cancellation signal, not a source of random state or wall time.
//   - pathID: uint64 logical scenario index in [0,Paths), not a worker ID.
//   - random: single-path owned RNG; must not escape to other goroutines.
//
// Returns:
//   - []float64: d finite components; error: any domain or context failure.
//
// Errors:
//   - Sampler-defined errors are preserved through errors.Is on Run.
//   - The driver classifies invalid length/finiteness separately after return.
//
// Determinism:
//   - Only Seed/pathID and the fixed rng.Factory.Version select the stream.
//
// Complexity:
//   - Dimension is expected O(1); Sample cost is entirely model-specific.
//
// Notes:
//   - Do not asynchronously mutate a returned slice after Sample returns.
//   - Typed-nil custom implementations should expose IsNil() bool to the driver.
//
// AI-Hints:
//   - Do not hold or share random across Sample invocations or goroutines.
//   - Returning NaN/Inf as an unavailable outcome is invalid; return an error.
type Sampler interface {
	// Dimension declares the immutable positive vector dimension of each Sample.
	Dimension() int

	// Sample generates one logical path, returning either a finite vector or error.
	Sample(ctx context.Context, pathID uint64, random rng.RNG) ([]float64, error)
}

// Plan specifies every observable and resource-related experiment option.
// Its zero-value defaults apply only to Workers (one) and MaxSamplesBytes (64 MiB);
// zero Paths is invalid and seed zero is an ordinary replayable experiment seed.
//
// Implementation:
//   - Stage 1: Run validates context, path count, workers and sorted quantile levels.
//   - Stage 2: dimension/budget preflight resolves immutable internal configuration.
//   - Stage 3: Run executes without mutating or retaining caller-owned Plan slices.
//
// Behavior highlights:
//   - RetainSamples enables a full Samples matrix, Quantiles enable exact sorting.
//   - MaxSamplesBytes limits optional raw/sort/quantile float64 backing buffers.
//
// Inputs:
//   - Paths: uint64 >=2; Seed: any uint64; Workers: >=0 with zero meaning one.
//   - Quantiles: finite unique ascending values in [0,1], nil/empty means none.
//   - MaxSamplesBytes: byte ceiling, zero selects 64 MiB.
//
// Returns: the Run result retains the seed and a detached copy of levels.
// Errors: ErrInvalidOption, ErrOverflow and ErrBudget from preflight.
// Determinism: fields fix all driver policy; stream and path orders never use W.
// Complexity: validation O(q); raw storage O(Paths*d) only in requested modes.
// Notes: mutating Plan.Quantiles concurrently with Run is unsupported.
// AI-Hints: exact quantiles require raw storage even when RetainSamples is false.
type Plan struct {
	// Paths is the number of logical paths (minimum two); IDs are 0..Paths-1.
	Paths uint64

	// Seed selects version-pinned per-path streams; zero is a valid reproducible seed.
	Seed uint64

	// Workers is requested parallelism; zero means serial, negative is invalid.
	Workers int

	// RetainSamples publishes a detached Completed-by-d matrix when Completed>0.
	RetainSamples bool

	// Quantiles specifies strict ascending, unique, finite type-7 probabilities.
	Quantiles []float64

	// MaxSamplesBytes limits raw sample/sort/output backing storage (0 => 64 MiB).
	MaxSamplesBytes uint64
}

// Result reports the strictly reduced contiguous prefix of stochastic paths.
// A failed computation may still return a non-nil Result with Complete=false;
// invalid preflight/budget instead returns nil Result and the corresponding error.
//
// Implementation:
//   - Stage 1: metadata and detached QuantileLevels are fixed before worker launch.
//   - Stage 2: the coordinator commits only successive path IDs starting at zero.
//   - Stage 3: sample moments and optional quantiles are published at completion.
//
// Behavior highlights:
//   - Completed is a prefix length, never the count of arbitrary finished jobs.
//   - Mean/SampleVariance/MeanStandardError are nil when Completed<2.
//   - Quantiles is nil on every incomplete or failed experiment.
//   - Samples is nil unless retention is enabled and at least one path committed.
//
// Inputs: Run produces Results; exported fields are caller-owned after return.
// Returns: IsNil classifies pointer absence, not statistical completeness.
// Errors: Run separately returns the first observed sampler/context/numeric error.
// Determinism: completed successes reduce in pathID order, coordinate order ascending.
// Complexity: vectors O(d); Samples O(Completed*d) visible with up to Paths*d
//
//	allocated storage on partial retained runs; Quantiles O(q*d).
//
// Notes: sample mutations after return do not modify frozen moments/quantiles.
// AI-Hints: MeanStandardError describes Monte Carlo sampling precision, not model risk.
type Result struct {
	// Requested is the original Path count, regardless of interruption.
	Requested uint64

	// Completed counts only paths [0,Completed) successfully reduced in order.
	Completed uint64

	// Samples is the visible Completed-by-d matrix if retention was enabled.
	Samples matrix.Matrix

	// Mean is the sample mean of each coordinate; nil for fewer than two paths.
	Mean []float64

	// SampleVariance is Bessel-corrected (ddof=1) coordinate variance.
	SampleVariance []float64

	// MeanStandardError is sqrt(sample variance)/sqrt(Completed) per feature.
	MeanStandardError []float64

	// Quantiles is a len(QuantileLevels)-by-d matrix on complete success only.
	Quantiles matrix.Matrix

	// QuantileLevels is a detached copy of requested ascending probability levels.
	QuantileLevels []float64

	// Seed records the exact base random seed used by the per-path factory.
	Seed uint64

	// RNG records Factory.Version() to make replay identity explicit.
	RNG string

	// Complete is true only after every path and requested statistic succeeded.
	Complete bool
}

// IsNil reports whether the Result pointer itself is absent.
// It implements core.Nilable and is safe to invoke on a typed-nil *Result.
//
// Implementation:
//   - Stage 1: compare the receiver with nil, without reading any result fields.
//   - Stage 2: return the immutable identity classification.
//   - Stage 3: perform no allocation or state mutation.
//
// Behavior highlights:
//   - An allocated Result with Completed==0 is still non-nil.
//
// Returns: true exactly for a nil *Result; otherwise false.
// Errors: none; a nil receiver is valid for this query.
// Determinism: pure pointer comparison, independent of sample state.
// Complexity: Time O(1), Space O(1).
// Notes: IsNil does not indicate Complete or whether moment vectors are populated.
// AI-Hints: check errors and Complete separately from Result.IsNil.
func (r *Result) IsNil() bool {
	return r == nil
}

var _ core.Nilable = (*Result)(nil)
