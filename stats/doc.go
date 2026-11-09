// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package stats implements numerically stable univariate moments, deterministic
// partition merging, and immutable fitted feature standardization.
// It connects finite scalar observations and matrix.Matrix feature batches
// without conflating accumulation, training, and inference.
//
// Domain scope:
//
//   - Moments collects finite float64 observations in a mutable, zero-value-ready
//     accumulator. Update uses Welford's centered recurrence; Merge uses the
//     Chan-Golub-LeVeque parallel-combination recurrence.
//   - Summary reports Count, Mean, Variance, Min, and Max with either population
//     (ddof=0) or sample (ddof=1) variance. Summarize is the batch facade;
//     SummarizePopulation and SummarizeSample select explicit denominators.
//   - FitStandardizer learns one mean and scale per matrix column. Each row is
//     one training observation and each column is one feature. Population and
//     sample fitting wrappers delegate to the same kernel.
//   - Standardizer.Transform and Standardizer.InverseTransform operate on new
//     matrices using frozen training parameters; neither operation performs fit.
//   - All successful public result structures contain independently owned
//     artifacts. Summary, ScaleFitResult, and Standardizer implement core.Nilable.
//
// Why the distinction matters:
//
//   - Large baselines with small spreads require a centered reduction: subtracting
//     two nearly equal E[x*x] and E[x]*E[x] values can erase the variance.
//   - Finite observations can produce an intermediate second central moment M2
//     larger than MaxFloat64 while the divided variance is representable.
//   - The same finite data may be appropriate for fitting but not for inference
//     refitting. Transforming heldout samples with parameters learned from those
//     samples leaks information into a model's evaluation.
//   - A feature constant during training may take a novel value at inference;
//     the unit-scale policy must preserve that deviation rather than erase it.
//
// Non-goals:
//
//   - No rolling removal, time-weighted moments, missing-value interpolation,
//     quantiles, medians, covariance matrices, or correlation estimation.
//   - No implicit series timestamp validation, grid repair, validity-mask
//     filtering, streaming window scheduling, or direct series dependency.
//   - No train/holdout partitioning, persistence, versioning, or inferred
//     per-feature semantic units. The caller owns these decisions.
//   - No configurable epsilon for declaring a column constant; no implicit
//     standardizer refit, adaptive transform, or fallback to another scaler.
//   - No thread-safe mutable Moments and no snapshot isolation for mutable
//     input matrix.Matrix implementations.
//
// Public entry points:
//
//   - NewMoments() *Moments
//   - (*Moments).Update(value float64) error
//   - (*Moments).Merge(other *Moments) error
//   - (*Moments).Summary(ddof int) (*Summary, error)
//   - Summarize(values []float64, ddof int) (*Summary, error)
//   - SummarizePopulation(values []float64) (*Summary, error)
//   - SummarizeSample(values []float64) (*Summary, error)
//   - FitStandardizer(x matrix.Matrix, opts ScaleOptions) (*ScaleFitResult, error)
//   - FitPopulationStandardizer(x matrix.Matrix, policy ConstantPolicy)
//     (*ScaleFitResult, error)
//   - FitSampleStandardizer(x matrix.Matrix, policy ConstantPolicy)
//     (*ScaleFitResult, error)
//   - (*Standardizer).Transform(x matrix.Matrix) (matrix.Matrix, error)
//   - (*Standardizer).InverseTransform(z matrix.Matrix) (matrix.Matrix, error)
//
// Result and state semantics:
//
//   - A zero-value Moments is a valid empty accumulator. NewMoments allocates
//     the same state; no constructor call is required for stack usage.
//   - Summary.Count is the committed uint64 observation count. Mean is the
//     correctly rounded arithmetic mean of the original binary64 observations;
//     Min and Max are selected original observations. Variance is a finite,
//     nonnegative materialization of a compensated approximate central moment.
//   - ScaleFitResult.Model owns its training means, scales, and constant flags.
//     ScaleFitResult.Means, Scales, and Constant are separate detached slices.
//     Mutating public metadata cannot change the model's stored parameters.
//   - A Standardizer is immutable after successful fit. A zero-valued, unfitted
//     Standardizer is non-nil but cannot perform inference; IsNil reports only
//     whether the receiver pointer is nil, not whether it has been fitted.
//   - Methods that return a result alongside an error return nil on failure;
//     no partial Summary, fitted model, or output matrix is published.
//
// Welford's sequential mathematical law:
//
//   - For n committed observations with mean mu and unnormalized second
//     central moment M2, add a finite x with delta = x - mu:
//
//     n'   = n + 1
//     mu'  = mu + delta/n'
//     M2'  = M2 + delta*delta*n/n'
//
//   - The last expression is equivalent in exact arithmetic to
//     M2 + delta*(x-mu'), but it makes the nonnegative correction explicit.
//
//   - The first observation establishes mu=x, M2=0, Min=Max=x.
//
//   - No public operation evaluates E[x*x]-E[x]*E[x] as its variance kernel.
//
// Chan-Golub-LeVeque's partition mathematical law:
//
//   - For nonempty independent accumulators A and B, let n=nA+nB and
//     delta=muB-muA. Then:
//
//     muAB = muA + delta*nB/n
//     M2AB = M2A + M2B + delta*delta*nA*nB/n
//
//   - Merge first checks that nA+nB fits uint64. Its cross term uses scaled
//     arithmetic rather than a potentially overflowing uint64 nA*nB product
//     or an overflowing float64 delta*delta product.
//
//   - An empty source is the identity; an empty destination adopts its source
//     state by value. Self-merge is always ErrInvalidOption, even when empty.
//
//   - Different partition groupings need not agree bit-for-bit in M2:
//     floating-point addition is not associative. Fix the merge tree when
//     reproducibility of the approximation matters.
//
// Arithmetic-domain and representability laws:
//
//   - A fixed-width signed binary superaccumulator retains the exact sum of
//     finite input float64 values in units of 2^-1074. Its 34 uint64 words
//     accommodate any committed sample count up to MaxUint64. The final mean
//     is rounded once using integer/rational materialization; Update/Merge
//     do not allocate arbitrary-precision objects per observation.
//   - The Welford/Chan center and M2 use a compensated scaled representation
//     of the form (hi+lo)*2^exponent. This avoids premature exponent-range
//     overflow, but the central moment remains floating-point approximate.
//     It is not an exact rational second moment and carries no universal
//     relative-error certificate for every possible adversarial sequence.
//   - Summary computes M2/(Count-ddof) before conversion to float64.
//     For example, [-1e154,0,1e154] has sample variance near 1e308 even
//     though its unnormalized M2 is near 2e308 and cannot fit in float64.
//   - [1e308,1e308] has mean 1e308 and zero variance without overflow.
//   - A mathematically positive variance can underflow to zero in Summary's
//     float64 result. That rounded zero is not proof of a constant feature.
//   - FitStandardizer tests its internal M2 against zero before rounding the
//     variance and takes the square root in scaled arithmetic. A finite scale
//     can therefore exist even when the unsquared variance exceeds MaxFloat64.
//   - Inputs and published numeric outputs must be finite. An unrepresentable
//     final mean, variance, positive fitted scale, or inference value produces
//     ErrOverflow, not a +Inf success value or a hidden substitute.
//
// Degrees-of-freedom and constant-column policy:
//
//   - Only ddof=0 (population denominator n) and ddof=1 (sample denominator
//     n-1) are supported. Count must be strictly greater than ddof.
//   - ScaleOptions{} means DDOF=0 and RejectConstant. Option validation is
//     explicit and precedes any matrix inspection.
//   - RejectConstant returns ErrZeroVariance for a zero-M2 training column.
//   - UnitScaleConstant retains its training mean, sets scale to exactly 1,
//     and marks the corresponding Constant flag true. A novel x is mapped
//     to z=(x-trainingMean)/1, not unconditionally to zero.
//   - Nonconstant features have finite, strictly positive fitted scales
//     sqrt(M2/(trainingRows-DDOF)). No epsilon or arbitrary lower bound
//     silently changes this classification.
//
// Matrix shape and inference laws:
//
//   - Fit requires Rows>=1 and Cols>=1; a matrix of either zero extent
//     returns ErrEmptyInput. A sample fit also requires Rows>1.
//   - Transform/InverseTransform require Cols equal to the number of fitted
//     features. Rows may be zero: a 0 x d input returns a detached 0 x d
//     *matrix.Dense, not nil and not an ErrEmptyInput failure.
//   - Transform computes z=(x-mu)/scale. The common path uses a finite
//     float64 difference; when x-mu overflows, scaled subtraction/division
//     is used instead of publishing infinity.
//   - InverseTransform computes x=FMA(z,scale,mu), with one fused multiply-add
//     per cell. The fused operation can avoid false overflow of a separate
//     z*scale intermediate when the final sum is finite.
//   - Both directions validate all input cells in row-major order before
//     allocating their output, then read and transform cells in row-major
//     order. This is not a snapshot: the input must remain unchanged for
//     the entire call, and custom Matrix.At errors may be returned directly.
//   - Output is allocated using matrix.NewDense. A *matrix.Dense input has
//     O(1) At access, but stats does not access its private backing storage;
//     a custom Matrix implementation is valid when it satisfies the interface.
//
// Error-classification law:
//
//   - ErrInvalidOption: unsupported ddof or ConstantPolicy; self-merge.
//   - ErrNilInput: nil Moments receiver, nil source, nil slice, nil/unfitted
//     Standardizer, or matrix nil recognized through matrix.ValidateNotNil.
//   - ErrEmptyInput: non-nil empty input slice; training matrix with zero rows
//     or zero columns. Zero-row inference is not an empty-input error.
//   - ErrShape: negative/unrepresentably large matrix dimensions or a
//     transformed matrix with a different number of feature columns.
//   - ErrNonFinite: any NaN, +Inf, or -Inf observation or input matrix cell.
//   - ErrInsufficientSamples: Count<=ddof or trainingRows<=DDOF.
//   - ErrZeroVariance: zero-M2 training feature under RejectConstant.
//   - ErrOverflow: a uint64 count overflow or unrepresentable final numeric
//     result. Intermediate scaled M2 above MaxFloat64 is not an error.
//   - Matrix.At, Matrix.Set, matrix.NewDense, and matrix nil-validation
//     failures preserve their underlying errors for errors.Is; matrix nil
//     validation is additionally wrapped with stats.ErrNilInput.
//
// Validation precedence is operation-specific and observable:
//
//   - Summarize: ddof -> nil slice -> empty slice -> ordered Update scan
//     (including finite checks) -> Summary sufficiency and materialization.
//   - Moments.Update: nil receiver -> nonfinite value -> count overflow ->
//     candidate-state arithmetic and single commit.
//   - Moments.Merge: nil receiver -> self-merge -> nil source -> uint64
//     count overflow -> empty-identity or combined-state publication.
//   - Moments.Summary: ddof -> nil receiver -> insufficient count ->
//     finite materialization. It does not mutate the accumulator.
//   - FitStandardizer: options -> nil/typed-nil -> shape/size -> empty shape
//     -> ordered Matrix.At/finite scan -> sample sufficiency -> feature-domain
//     classification and scaling -> result publication.
//   - Transform/InverseTransform: nil/unfitted model -> nil/typed-nil matrix
//     -> shape/feature count -> full Matrix.At/finite scan -> allocation and
//     row-major computation. Failure never returns a partial matrix.
//
// Ownership and concurrency:
//
//   - Moments stores a bounded value state; Update and Merge mutate only the
//     receiver after successful validation. Errors leave it unchanged.
//   - Merge does not modify its source and does not share dynamically mutable
//     state with it. Neither concurrent Update/Update nor concurrent
//     Update/Merge on the same accumulators is supported. Use one accumulator
//     per worker and merge only after workers finish; synchronize externally.
//   - Fit reads caller-owned matrix input without modifying or retaining it.
//     No input deep copy is taken; concurrent source mutations are unsupported.
//   - The fitted Standardizer owns private parameter copies. Concurrent
//     Transform/InverseTransform calls are safe when each input matrix is
//     stable for the duration of the call; every output has private storage.
//   - ScaleFitResult's public metadata slices are detached from the model,
//     but public slices themselves remain caller-mutable. IsNil does not
//     validate fields, content, or fitted state.
//
// Determinism and partial-result law:
//
//   - Batch samples are visited from index 0 upward. Fit and inference scan
//     rows from 0 upward, visiting columns 0..d-1 within each row.
//   - Fit finalizes features in ascending column index; the first encountered
//     failure at a given stage is reported. No unordered maps participate.
//   - Merge uses an explicitly caller-chosen partition order; do not equate
//     floating-point M2 results across differently grouped merge trees.
//   - Update and Merge are transactional with respect to returned errors;
//     successful inference always returns a detached result and failures
//     return nil instead of a partially computed result.
//
// Complexity and allocation:
//
//   - NewMoments: O(1) time and space.
//   - Update: O(1) fixed-width state arithmetic, no heap allocation in
//     the scalar hot loop; the exact sum uses at most 34 uint64 limbs.
//   - Merge: O(1) bounded-width work and storage, including 34-word addition.
//   - Summary: O(1) in observation count; bounded ~2176-bit integer/rational
//     materialization may allocate during final mean conversion.
//   - Summarize: O(n) time, O(1) accumulator space.
//   - FitStandardizer: O(n*d) time, O(d) accumulator/model storage for n rows
//     and d columns; matrix.At may have implementation-specific extra cost.
//   - Transform/InverseTransform: O(n*d) time and O(n*d) owning output space,
//     with two deterministic input scans and O(1) additional scalar workspace.
//   - None of the asymptotic statements guarantees allocation success for
//     a shape that is mathematically representable but exceeds available RAM.
//
// Operational guidance:
//
//   - For scalar batches, use SummarizePopulation or SummarizeSample; for
//     streaming or distributed reductions, use Moments.Update and Merge.
//   - For feature matrices, fit only on the training split. Keep the
//     resulting Model and apply it unchanged to validation, test, and live
//     data. Store model-training provenance at the application boundary.
//   - To consume series.TransformResult, select only entries whose Valid
//     mask is true and apply your own timestamp/gap policy first. stats
//     never interprets warmup placeholder zeros as invalid automatically.
//   - For multivariate DTW, standardize comparable features before calling
//     dtw.AlignMatrix when units should not dominate squared L2 cost.
//   - For a walkthrough, mathematical derivations, diagrams, and executable
//     scenarios, see docs/STATS.md and stats/example_test.go.
//
// AI-Hints:
//
//   - Never replace the centered Welford/Chan M2 with raw sum-of-squares
//     subtraction; a test on small values will not reveal cancellation.
//   - Never cast uint64 partition counts to float64 before forming the
//     scaled correction or compute nA*nB as uint64 before division.
//   - Never materialize M2 or variance into float64 before division/sqrt
//     when the published statistic or scale may remain representable.
//   - Do not infer mathematical constancy from a rounded Summary.Variance
//     equal to zero; do not invent an epsilon threshold in fitted scaling.
//   - Keep model parameters private and copy exported metadata. Refit on
//     heldout data is leakage; so is updating a model during Transform.
//   - Do not replace InverseTransform's FMA with a separate multiplication
//     and addition. It may overflow where the fused result is finite.
//   - Do not use reflect for typed-nil matrix detection. The established
//     matrix.ValidateNotNil/core.Nilable convention is the validation layer.
//   - Do not infer concurrency safety for Moments from immutable-model
//     inference. Shared mutable accumulators require explicit synchronization.
package stats
