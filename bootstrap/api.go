// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap provides a deterministic public facade for block-bootstrap
// index sampling. All application-level statistics remain with the caller.
package bootstrap

import (
	"context"

	"github.com/lvlath/go/rng"
)

// bootstrapMappingVersion fixes integer rejection, draw order, block transitions,
// and single-replicate publication semantics independently of the RNG algorithm.
const bootstrapMappingVersion = "/bootstrap-index-v1"

// SampleIndices returns reproducible block-bootstrap observation indices.
// Each row is generated from a separate versioned RNG stream keyed by Seed and
// its zero-based replicate index; only complete rows are returned to the caller.
//
// Implementation:
//   - Stage 1: Validate context, policy and native-int output budget before allocation.
//   - Stage 2: Create the accepted RNG factory and fill independent private rows
//     through exactly one of Moving, Circular or Stationary block kernels.
//   - Stage 3: Publish each row atomically; return completed rows plus ctx.Err()
//     if interrupted after computation began.
//
// Behavior highlights:
//   - MovingBlocks uses nonwrapping fixed blocks; CircularBlocks wraps fixed blocks.
//   - StationaryBlocks uses exact rational Bernoulli restarts with mean block B.
//   - StationaryBlocks with B=1 performs IID index draws without Bernoulli draws.
//   - Seed=0 is a fully explicit deterministic seed, never a random default.
//
// Inputs:
//   - ctx: non-nil context; a canceled context wins over invalid plan options.
//   - rows: count of original observations, at least 1, not the output length.
//   - plan: by-value policy with Length>=1, Replicates>=1, 1<=BlockLength<=rows.
//
// Returns:
//   - *Result: owns all Indices rows; success has Completed==plan.Replicates.
//   - error: nil on success, or an error classifying validation, budget or context.
//   - Before computation: nil Result on every validation/cancellation failure.
//   - During computation: non-nil Result with only fully completed rows and ctx.Err().
//
// Errors:
//   - ErrInvalidOption: nil context, invalid method or nonpositive/oversized B.
//   - ErrOverflow: native-int size arithmetic overflows before allocation.
//   - ErrBudget: logical result plus row headers exceeds MaxResultBytes.
//   - ctx.Err(): context cancellation or deadline, preserving native error identity.
//
// Determinism:
//   - Row r always uses factory.Stream(plan.Seed, uint64(r)), regardless of
//     execution scheduling or requested Replicates; rows commit in ascending r.
//   - Sampling word consumption is bound to the advertised AlgorithmID.
//
// Complexity:
//   - Expected Time O(Replicates*Length) and Space O(Replicates*Length);
//     rejection sampling has no finite worst-case draw count, but is cancelable.
//
// Notes:
//   - No observation values are copied; clients gather or slice their own data.
//   - No shared streams or mutable global RNG state are used by this facade.
//   - Concurrent callers are independent; caller mutation of returned Indices
//     across goroutines must be externally synchronized.
//   - Valid index resampling does not establish valid confidence intervals under
//     arbitrary nonstationarity, dependence or temporal data leakage.
//
// AI-Hints:
//   - Always inspect the error even when Result is non-nil: cancellation is partial.
//   - Persist (RNG, Seed, Kind, Length, BlockLength, Replicates) for exact replay.
//   - Do not replace factory.Stream(seed, r) with a shared RNG per call.
//   - Do not reinterpret moving blocks as IID observations or change word order.
func SampleIndices(ctx context.Context, rows int, plan Plan) (*Result, error) {
	// Context precedence is explicit: even an invalid plan cannot hide cancellation.
	if err := validateContext(ctx); err != nil {
		return nil, err
	}
	if err := validatePlan(rows, plan); err != nil {
		return nil, err
	}

	// This factory must come from the accepted shared RNG dependency; creating
	// a local generator or defaulting Seed changes the published replay law.
	// Version identifies its primitive and stream-derivation protocol.
	factory := rng.NewFactory()
	algorithm := AlgorithmID(string(factory.Version()) + bootstrapMappingVersion)

	return sampleIndicesValidated(ctx, rows, plan, algorithm, func(seed, pathID uint64) wordStream {
		return factory.Stream(seed, pathID)
	})
}

// sampleIndicesValidated owns generation and publication for one admitted plan.
// It receives a stream-constructor closure so the facade holds the only RNG adapter.
//
// Implementation:
//   - Stage 1: Prepare one checked-capacity result slice.
//   - Stage 2: For each r, allocate one private row and initialize Stream(Seed,r).
//   - Stage 3: Dispatch to one kernel, recheck cancellation, then append the row.
//
// Inputs:
//   - ctx, rows, plan: fully validated by the canonical public facade.
//   - algorithm: fixed generator/mapping identity for replay metadata.
//   - newStream: returns one exclusive stream for a seed and logical replicate ID.
//
// Returns:
//   - *Result: completed rows only, including an allocated empty slice after a
//     computation-time cancellation before the first completed row.
//
// Errors:
//   - ctx.Err() from a block, rejection loop, or final publication gate.
//
// Determinism:
//   - Always attempts rows in r=0,1,... order; all streams are local to one row.
//
// Complexity:
//   - Expected Time O(R*L), Space O(R*L); row allocations are bounded in advance.
//
// AI-Hints:
//   - Never append before fill returns AND context is rechecked.
//   - Do not retain the interrupted row or replace errors with partial success.
func sampleIndicesValidated(
	ctx context.Context,
	rows int,
	plan Plan,
	algorithm AlgorithmID,
	newStream func(seed, pathID uint64) wordStream,
) (*Result, error) {
	result := &Result{
		Indices: make([][]int, 0, plan.Replicates),
		RNG:     algorithm,
	}

	// pathID starts at zero and advances once per committed row; by induction it
	// equals uint64(replicate) without an unchecked signed-to-unsigned conversion.
	pathID := uint64(0)
	for replicate := 0; replicate < plan.Replicates; replicate++ {
		// Cancellation after admission is a partial-result outcome, even at r=0.
		if err := ctx.Err(); err != nil {
			return result, err
		}

		// A distinct row backing array is never attached until fill succeeds.
		row := make([]int, plan.Length)
		stream := newStream(plan.Seed, pathID)

		var err error
		switch plan.Kind {
		case MovingBlocks:
			err = fillMoving(ctx, stream, row, rows, plan.BlockLength)
		case CircularBlocks:
			err = fillCircular(ctx, stream, row, rows, plan.BlockLength)
		case StationaryBlocks:
			err = fillStationary(ctx, stream, row, rows, plan.BlockLength)
		default:
			// Defensive guard against future enum changes bypassing validation.
			return result, ErrInvalidOption
		}
		if err != nil {
			return result, err
		}

		// The final check closes the window between the last write and publish.
		if err = ctx.Err(); err != nil {
			return result, err
		}
		result.Indices = append(result.Indices, row)
		result.Completed++
		pathID++
	}

	// If cancellation is observed after the final commit, retain complete rows
	// while reporting the context failure rather than silently claiming success.
	if err := ctx.Err(); err != nil {
		return result, err
	}
	return result, nil
}

// SampleMovingIndices draws fixed contiguous, nonwrapping observation blocks.
// It is a convenience wrapper around SampleIndices with Kind=MovingBlocks.
//
// Implementation:
//   - Stage 1: Construct an explicit Plan using the supplied dimensions and Seed.
//   - Stage 2: Delegate validation, RNG and sampling to SampleIndices.
//   - Stage 3: Return the unchanged result and error, including partial cancellation.
//
// Inputs:
//   - ctx: active non-nil context; rows: number of source observations.
//   - length: target observations per replicate; blockLength: contiguous B in [1,rows].
//   - replicates: number of rows >=1; seed: explicit uint64 including zero.
//
// Returns:
//   - *Result with complete, detached rows or a classified error/partial result.
//
// Errors:
//   - Exactly those of SampleIndices for a MovingBlocks Plan.
//
// Determinism:
//   - Same word consumption and row order as the canonical moving-plan call.
//
// Complexity:
//   - Expected Time O(replicates*length), Space O(replicates*length).
//
// Notes:
//   - Explicitly named arguments avoid constructing Plan for a one-off call.
//
// AI-Hints:
//   - Use SampleIndices when storing a Plan for reproducibility or batch control.
func SampleMovingIndices(ctx context.Context, rows, length, blockLength, replicates int, seed uint64) (*Result, error) {
	return SampleIndices(ctx, rows, Plan{
		Kind: MovingBlocks, Length: length, BlockLength: blockLength,
		Replicates: replicates, Seed: seed,
	})
}

// SampleCircularIndices draws fixed blocks whose indices may wrap at rows-1.
// This named wrapper delegates without changing any canonical sampling rule.
//
// Implementation:
//   - Stage 1: Build an explicit CircularBlocks Plan.
//   - Stage 2: Delegate context, size and RNG policy to SampleIndices.
//   - Stage 3: Return its detached Result or classified cancellation unchanged.
//
// Inputs:
//   - ctx: non-nil context; rows: source observation count >=1.
//   - length: target row length; blockLength: B in [1,rows].
//   - replicates: count >=1; seed: explicit deterministic uint64.
//
// Returns:
//   - *Result of independent logical replicate rows, or error/partial result.
//
// Errors:
//   - Exactly those of SampleIndices for CircularBlocks.
//
// Determinism:
//   - The source wraps only inside blocks; restart starts are sampled per block.
//
// Complexity:
//   - Expected Time O(replicates*length), Space O(replicates*length).
//
// Notes:
//   - Use paired input data (X,y) with the same row indices when resampling pairs.
//
// AI-Hints:
//   - Never sample regression responses separately from their feature rows.
func SampleCircularIndices(ctx context.Context, rows, length, blockLength, replicates int, seed uint64) (*Result, error) {
	return SampleIndices(ctx, rows, Plan{
		Kind: CircularBlocks, Length: length, BlockLength: blockLength,
		Replicates: replicates, Seed: seed,
	})
}

// SampleStationaryIndices draws stationary index paths with random block lengths.
// BlockLength is the mean geometric run length, not a fixed maximum length.
//
// Implementation:
//   - Stage 1: Build a StationaryBlocks Plan from explicit input parameters.
//   - Stage 2: Delegate exact Bernoulli and stream policy to SampleIndices.
//   - Stage 3: Return the unchanged Result and error, preserving partial rows.
//
// Inputs:
//   - ctx: active non-nil context; rows: positive source size.
//   - length: output row length; meanBlockLength: expected run length in [1,rows].
//   - replicates: positive row count; seed: explicit uint64 including 0.
//
// Returns:
//   - Detached *Result or a classified error/partial result.
//
// Errors:
//   - Exactly those of SampleIndices for StationaryBlocks.
//
// Determinism:
//   - First index uniform; subsequent positions use exact restart before draw.
//
// Complexity:
//   - Expected Time O(replicates*length), Space O(replicates*length).
//
// Notes:
//   - B=1 is IID index sampling, intentionally without Bernoulli RNG draws.
//
// AI-Hints:
//   - Do not claim unconditional validity of VaR/ES confidence intervals.
func SampleStationaryIndices(ctx context.Context, rows, length, meanBlockLength, replicates int, seed uint64) (*Result, error) {
	return SampleIndices(ctx, rows, Plan{
		Kind: StationaryBlocks, Length: length, BlockLength: meanBlockLength,
		Replicates: replicates, Seed: seed,
	})
}
