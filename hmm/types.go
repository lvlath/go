// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm defines the typed public contracts for finite-state hidden Markov models.
// This boundary specifies the ownership of parameters, causal snapshots, and diagnostic results.
package hmm

import "github.com/lvlath/go/matrix"

// Common specifies the stochastic hidden-state chain shared by both emission families.
// A chain has K>=1 states and transitions act from row i to successor column j.
// Implementation:
//   - Stage 1: Constructors validate K and the K-by-K transition shape.
//   - Stage 2: Finite nonnegative rows are checked against unit sum tolerance.
//   - Stage 3: Canonical normalized copies become private model parameters.
//
// Behavior highlights:
//   - Exact zeros remain zeros; no smoothing is implicit in a constructor.
//
// Inputs:
//   - Prior: length K distribution for the first hidden state, sum 1.
//   - Transition: K-by-K row-stochastic matrix in row-major mathematical order.
//
// Returns:
//   - A value specification, not a model; pass it to NewCategorical/NewGaussian.
//
// Errors:
//   - ErrNilInput, ErrShape, ErrNonFinite, ErrProbability from constructors.
//
// Determinism:
//   - All entries are copied in ascending state and transition order.
//
// Complexity:
//   - Constructor validation/copying costs O(K^2) time and space.
//
// Notes:
//   - Callers retain ownership; model construction does not alias their buffers.
//
// AI-Hints:
//   - A[i,j] means P(next=j|current=i), not its transpose.
type Common struct {
	Prior      []float64     // P(z_0=i), length K; nonnegative and normalized to one.
	Transition matrix.Matrix // P(z_t=j|z_{t-1}=i), K-by-K row-stochastic.
}

// CategoricalSpec describes a model of finite-alphabet observations.
// Every emission row represents P(symbol=a|state=k), never a numeric sensor reading.
// Implementation:
//   - Stage 1: Validate Common and emission shape K-by-A, A>=1.
//   - Stage 2: Validate probabilities and normalize permitted row residuals.
//   - Stage 3: Copy parameters into an immutable CategoricalModel.
//
// Behavior highlights:
//   - Zero emission probability makes that symbol impossible in its state.
//
// Inputs:
//   - Common: hidden Markov chain; Emission: K-by-A row-stochastic matrix.
//
// Returns:
//   - A constructor specification whose matrices remain caller-owned.
//
// Errors:
//   - ErrNilInput, ErrShape, ErrNonFinite, ErrProbability.
//
// Determinism:
//   - Symbols correspond to columns 0..A-1, in that fixed order.
//
// Complexity:
//   - Construction costs O(K^2+K*A) time and space.
//
// Notes:
//   - The alphabet cannot grow after model construction.
//
// AI-Hints:
//   - Convert real-world event labels to stable uint32 symbol IDs before inference.
type CategoricalSpec struct {
	Common   Common        // Shared immutable K-state Prior and Transition specification.
	Emission matrix.Matrix // K-by-A row-stochastic P(symbol=a | hidden state=k).
}

// GaussianSpec describes independent Gaussian features conditional on a hidden state.
// Each state carries D means and D strictly positive diagonal variances in feature-squared units.
// Implementation:
//   - Stage 1: Validate Common and both K-by-D parameter shapes.
//   - Stage 2: Reject nonfinite values and nonpositive variances.
//   - Stage 3: Copy immutable GaussianModel parameters.
//
// Behavior highlights:
//   - No missing-feature imputation or full covariance is performed.
//
// Inputs:
//   - Means: K-by-D feature means; Variances: K-by-D squared-feature variances.
//
// Returns:
//   - A specification for NewGaussian or FitGaussian.
//
// Errors:
//   - ErrNilInput, ErrShape, ErrNonFinite, ErrProbability.
//
// Determinism:
//   - Feature coordinates retain their original caller-supplied order.
//
// Complexity:
//   - Construction costs O(K^2+K*D) time and space.
//
// Notes:
//   - FitGaussian applies an additional initial VarianceFloor constraint.
//
// AI-Hints:
//   - Standardize data externally and freeze that transform before fitting.
type GaussianSpec struct {
	Common    Common        // Shared immutable K-state Prior and Transition specification.
	Means     matrix.Matrix // K-by-D finite expected feature values in physical units.
	Variances matrix.Matrix // K-by-D strictly positive variances in corresponding feature^2 units.
}

// chain stores canonical hidden-state parameters shared by both model variants.
// After construction its fields form a read-only value graph with stable ModelID identity.
//
// Implementation:
//   - Stage 1: Set k from the validated positive Prior length.
//   - Stage 2: Copy normalized K-prior and K-by-K transition values into private flat storage.
//   - Stage 3: Attach the complete-model hash after validating the emission family.
//
// Behavior highlights:
//   - A flattened transition at i*k+j means P(z_next=j | z_current=i).
//
// Inputs:
//   - k: hidden-state count; prior: K values; transition: K*K row-major values; id: SHA-256 identifier.
//
// Returns:
//   - An internal model component, never a publicly mutable result.
//
// Errors:
//   - Constructors return ErrShape/ErrProbability/etc. before publishing invalid chains.
//
// Determinism:
//   - Every chain entry is stored in ascending state/column order.
//
// Complexity:
//   - Storage O(K^2+K); construction validation O(K^2).
//
// Notes:
//   - The host model also stores a separate immutable categorical or Gaussian emission array.
//
// AI-Hints:
//   - Do not use the chain identifier before including emissions in the full hash.
type chain struct {
	k          int       // Positive number of hidden states, shared by all parameter arrays.
	prior      []float64 // Detached normalized P(z_0=i), indexed by state i.
	transition []float64 // Detached normalized A[i*k+j], current state i to successor j.
	id         string    // Complete model-family and emission-aware canonical SHA-256 identity.
}

// CategoricalModel stores an immutable canonical finite-alphabet HMM.
// Concurrent inference is safe when callers do not mutate their supplied observations.
// Implementation:
//   - Stage 1: Constructor validates and copies stochastic rows.
//   - Stage 2: Parameters and algorithm revision are cryptographically hashed.
//   - Stage 3: Inference creates per-call workspaces without model mutation.
//
// Behavior highlights:
//   - Internal flat matrices avoid interface dispatch in O(T*K^2) kernels.
//
// Inputs:
//   - Construct with NewCategorical(CategoricalSpec).
//
// Returns:
//   - A model pointer whose parameters are not exposed for mutation.
//
// Errors:
//   - Constructor errors are described in CategoricalSpec.
//
// Determinism:
//   - Fixed transition/state iteration order, with explicit Viterbi tie policy.
//
// Complexity:
//   - Storage O(K^2+K*A); inference costs depend on the invoked method.
//
// Notes:
//   - Independent concurrent calls use distinct scratch memory.
//
// AI-Hints:
//   - Refit to obtain a new model and ModelID; do not patch FilterState IDs.
type CategoricalModel struct {
	chain    chain     // Privately owned stochastic transitions and prior.
	alphabet int       // Fixed A>=1 emission symbols; valid indices are [0,A).
	emission []float64 // Private normalized K-by-A categorical masses in state-major order.
}

// GaussianModel stores immutable parameters of a diagonal-Gaussian HMM.
// Each feature is conditionally independent given the state, not unconditionally independent.
// Implementation:
//   - Stage 1: Validate and deep-copy means, variances, and chain.
//   - Stage 2: Hash the canonical complete parameter configuration.
//   - Stage 3: Use private read-only parameters in inference workspaces.
//
// Behavior highlights:
//   - Gaussian observations may yield positive log-density values.
//
// Inputs:
//   - Construct with NewGaussian(GaussianSpec).
//
// Returns:
//   - An immutable model pointer safe for independent parallel inference.
//
// Errors:
//   - Constructor errors are described in GaussianSpec.
//
// Determinism:
//   - State-major then feature-major evaluation order is fixed.
//
// Complexity:
//   - Storage O(K^2+K*D); per-emission evaluation O(K*D).
//
// Notes:
//   - Gaussian features must be finite; no NaN-as-missing convention.
//
// AI-Hints:
//   - Variances represent squared units, not standard deviations.
type GaussianModel struct {
	chain     chain     // Privately owned stochastic transitions and prior.
	dimension int       // Fixed D>=1 observed feature coordinates per time step.
	means     []float64 // Private K-by-D state means in unchanged caller feature units.
	variances []float64 // Private K-by-D positive feature-squared variance values.
}

// FilterState is the caller-owned checkpoint after exactly Steps causal observations.
// The posterior contains log P(z_t|x_0:t), not unnormalized forward masses.
// Implementation:
//   - Stage 1: FilterStep validates an incoming checkpoint when prev is non-nil.
//   - Stage 2: It evaluates the next emission and performs normalized log recursion.
//   - Stage 3: It returns a new detached snapshot without changing prev.
//
// Behavior highlights:
//   - Impossible states have -Inf log mass; at least one state must be possible.
//
// Inputs:
//   - ModelID: canonical model hash; Steps>=1 for a restored checkpoint.
//   - LogPosterior: K normalized log probabilities, LogSumExp approximately zero.
//
// Returns:
//   - A detached state value produced by FilterStep or FilterResult.Last.
//
// Errors:
//   - ErrStateVersion for mismatched/corrupt checkpoints; ErrOverflow on Steps overflow.
//
// Determinism:
//   - Fixed state ordering, independent of future observations.
//
// Complexity:
//   - State storage O(K); validation O(K).
//
// Notes:
//   - The exported slice is mutable by callers; FilterStep itself never mutates it.
//
// AI-Hints:
//   - A Go struct copy shares its LogPosterior slice; deep-copy before editing snapshots.
type FilterState struct {
	ModelID       string    // Canonical SHA-256 identity of the originating model.
	Steps         uint64    // Number of observations already incorporated, starting at one.
	LogPosterior  []float64 // Normalized log probabilities in state index order.
	LogLikelihood float64   // Finite cumulative marginal log evidence of the prefix.
}

// FilterOptions controls materialization of the historical causal posterior matrix.
// The default omits history and keeps streaming workspace independent of T.
// Implementation:
//   - Stage 1: Read the ReturnHistory flag without implicit conversion.
//   - Stage 2: Always compute the same normalized causal recursion.
//   - Stage 3: Optionally materialize detached T-by-K probability rows.
//
// Behavior highlights:
//   - ReturnHistory=false leaves FilterResult.Posterior nil.
//
// Inputs:
//   - ReturnHistory: whether to retain each time-indexed causal posterior.
//
// Returns:
//   - A value passed to either model's Filter method.
//
// Errors:
//   - This boolean-only option structure has no invalid values.
//
// Determinism:
//   - The flag affects retention only, never the likelihood calculation.
//
// Complexity:
//   - History requires O(T*K) additional result space, otherwise O(K) workspace.
//
// Notes:
//   - A full posterior history can dominate memory on long streams.
//
// AI-Hints:
//   - Prefer FilterDefault for only the last state and FilterWithHistory for plots.
type FilterOptions struct {
	// ReturnHistory retains a detached T-by-K matrix of causal posterior probabilities.
	// False is the memory-saving default; it does not change final evidence or Last.
	ReturnHistory bool
}

// FilterResult contains the final causal checkpoint and optional probability history.
// Its Last and LogLikelihood represent the same processed prefix.
// Implementation:
//   - Stage 1: Iterate observations with normalized log messages.
//   - Stage 2: Accumulate per-step log evidence without multiplying probabilities.
//   - Stage 3: Publish detached Last and optional Posterior.
//
// Behavior highlights:
//   - Posterior is T-by-K and stores probabilities, not log probabilities.
//
// Inputs:
//   - Produced by Filter; the history policy comes from FilterOptions.
//
// Returns:
//   - Last.Steps=T and LogLikelihood=Last.LogLikelihood on success.
//
// Errors:
//   - Any filter failure returns nil rather than a partially valid result.
//
// Determinism:
//   - A fixed prefix always produces the same history prefix.
//
// Complexity:
//   - Result storage O(K) without history, O(T*K) with history.
//
// Notes:
//   - All returned data belongs to the caller, not the model.
//
// AI-Hints:
//   - Never treat this causal history as the retrospectively smoothed probability.
type FilterResult struct {
	Last          FilterState   // Final detached causal checkpoint with exactly T observations.
	Posterior     matrix.Matrix // Optional T-by-K probability history; nil when history is disabled.
	LogLikelihood float64       // Finite total log evidence, equal to Last.LogLikelihood.
}

// SmoothResult contains the full retrospective posterior after Forward-Backward.
// Every row conditions on all T observations, including those in the future of that row.
// Implementation:
//   - Stage 1: Compute log emissions and normalized forward messages.
//   - Stage 2: Compute scaled backward messages and gamma probabilities.
//   - Stage 3: Return a detached T-by-K posterior matrix.
//
// Behavior highlights:
//   - No EM transition pair table is retained for smoothing alone.
//
// Inputs:
//   - Produced by Smooth from an entire nonempty observation sequence.
//
// Returns:
//   - Posterior: T-by-K normalized probabilities, LogLikelihood: log P(X).
//
// Errors:
//   - Invalid/impossible input returns nil and its corresponding sentinel.
//
// Determinism:
//   - Fixed reverse-time and state-major traversal orders.
//
// Complexity:
//   - Time O(T*K^2+T*K*D), space O(T*K+T*D) for Gaussian data.
//
// Notes:
//   - Do not use retrospective scores to claim causal decisions.
//
// AI-Hints:
//   - Compare Filter and Smooth on historical data to measure future-evidence effects.
type SmoothResult struct {
	Posterior     matrix.Matrix // Required detached T-by-K retrospective gamma probabilities.
	LogLikelihood float64       // Finite log evidence of the complete observation sequence.
}

// DecodeResult contains the globally most likely joint hidden-state path.
// Its States are obtained by Viterbi backtracking, not independent marginal maxima.
// Implementation:
//   - Stage 1: Accumulate maximum log joint scores for each state.
//   - Stage 2: Record deterministic predecessor choices.
//   - Stage 3: Backtrack one full T-step state trajectory.
//
// Behavior highlights:
//   - Ties select the lowest predecessor, then lowest terminal index.
//
// Inputs:
//   - Produced by Decode from one nonempty observed sequence.
//
// Returns:
//   - States: length T; LogJointProbability: log P(z*,X).
//
// Errors:
//   - Impossible sequences fail instead of returning an arbitrary zero path.
//
// Determinism:
//   - Strict greater-than comparisons preserve the first index in ties.
//
// Complexity:
//   - Time O(T*K^2+T*K*D); space O(T*K).
//
// Notes:
//   - The returned path slice is detached from the model and workspace.
//
// AI-Hints:
//   - Do not interpret exp(LogJointProbability) as posterior probability of each state.
type DecodeResult struct {
	States              []int   // Caller-owned length-T maximum-joint-probability state path.
	LogJointProbability float64 // Log P(States, observations) of this entire Viterbi path.
}

// FitOptions controls explicit deterministic Baum-Welch parameter estimation.
// The model starts from caller-provided parameters and never uses random initialization.
// Implementation:
//   - Stage 1: Validate nonnegative finite options and apply zero defaults.
//   - Stage 2: Maximize the corresponding penalized EM auxiliary function.
//   - Stage 3: Check objective monotonicity and return diagnostics.
//
// Behavior highlights:
//   - Pseudocount lambda corresponds to symmetric Dirichlet alpha=1+lambda.
//
// Inputs:
//   - MaxIterations: EM upper bound, zero defaults to 100.
//   - Tolerance: nonnegative relative objective-change threshold, zero defaults to 1e-6.
//   - Pseudocount: nonnegative Dirichlet penalty weight, zero means no penalty.
//   - VarianceFloor: squared-feature lower variance bound, zero defaults to 1e-8.
//
// Returns:
//   - A validated value used by FitCategorical or FitGaussian.
//
// Errors:
//   - ErrInvalidOption for negative or nonfinite option fields.
//
// Determinism:
//   - Fixed sequence, time, state, and feature iteration order.
//
// Complexity:
//   - O(1) validation; full fitting cost depends on sequence lengths and iterations.
//
// Notes:
//   - Initial Gaussian variances must already satisfy the floor.
//
// AI-Hints:
//   - Lambda penalizes categorical probabilities, not Gaussian means or variances.
type FitOptions struct {
	MaxIterations int     // Accepted EM M-step limit; 0 selects 100, negative is invalid.
	Tolerance     float64 // Relative penalized-objective improvement threshold; 0 selects 1e-6.
	Pseudocount   float64 // Dirichlet alpha-1 >= 0; 0 disables the penalty exactly.
	VarianceFloor float64 // Gaussian M-step variance lower bound in feature^2; 0 selects 1e-8.
}

// Diagnostics records the accepted iterations of penalized EM estimation.
// Objective includes the initial score followed by each accepted parameter update.
// Implementation:
//   - Stage 1: Evaluate the supplied initial model and its occupancy.
//   - Stage 2: Append only accepted monotone objective values.
//   - Stage 3: Report the final expected occupancy and convergence status.
//
// Behavior highlights:
//   - len(Objective)=Iterations+1 for a valid accepted model.
//
// Inputs:
//   - Produced by either Fit function; no caller initialization is required.
//
// Returns:
//   - Occupancy: expected state visits; its sum matches observations in accepted data.
//
// Errors:
//   - ErrNotConverged may accompany a valid Model and Converged=false.
//
// Determinism:
//   - Occupancy and objective accumulation follow input order.
//
// Complexity:
//   - Storage O(Iterations+K).
//
// Notes:
//   - Objective is penalized and need not track raw data likelihood monotonically.
//
// AI-Hints:
//   - Treat a nonconverged result as diagnostics, not automatically production-ready.
type Diagnostics struct {
	Iterations int       // Number of accepted M-steps; max MaxIterations.
	Converged  bool      // True only after a nonnegative improvement meets Tolerance.
	Objective  []float64 // Initial penalized score followed by each accepted score; length Iterations+1.
	Occupancy  []float64 // Final expected state visits across all sequences, length K.
}

// CategoricalFitResult publishes a fitted categorical model and training diagnostics.
// A nonconverged fit can return this last accepted model together with ErrNotConverged.
// Implementation:
//   - Stage 1: Validate observations and the initial model.
//   - Stage 2: Accept only monotone penalized EM steps.
//   - Stage 3: Return an immutable model and independently owned diagnostics.
//
// Behavior highlights:
//   - The number of states and alphabet remain fixed during training.
//
// Inputs:
//   - Produced by FitCategorical, never constructed by the fitting engine's caller.
//
// Returns:
//   - Model and Diagnostics, or nil on validation/cancellation errors.
//
// Errors:
//   - ErrNotConverged or ErrNumerical may accompany the last accepted result.
//
// Determinism:
//   - Training order and parameter copy order are fixed.
//
// Complexity:
//   - Training runtime O(iterations * total sequence length * K^2).
//
// Notes:
//   - The fitted model is safe for independent concurrent inference.
//
// AI-Hints:
//   - Check both the error sentinel and Diagnostics.Converged before deployment.
type CategoricalFitResult struct {
	Model       *CategoricalModel // Immutable last accepted candidate, even on ErrNotConverged.
	Diagnostics Diagnostics       // Caller-owned fit progress, occupancy, and convergence status.
}

// GaussianFitResult publishes a fitted diagonal-Gaussian model and diagnostics.
// Every accepted variance respects the configured floor in squared feature units.
// Implementation:
//   - Stage 1: Validate data, initial means, variances, and floor.
//   - Stage 2: Optimize weighted moments under a constrained variance domain.
//   - Stage 3: Publish only verified accepted parameters and objective history.
//
// Behavior highlights:
//   - Dead states preserve their previous Gaussian emission parameters.
//
// Inputs:
//   - Produced by FitGaussian with explicit initial model parameters.
//
// Returns:
//   - Model and Diagnostics, or nil on input or context failure.
//
// Errors:
//   - ErrNotConverged/ErrNumerical may accompany the last valid result.
//
// Determinism:
//   - Sequences and features are accumulated in input order.
//
// Complexity:
//   - O(iterations * total length * (K^2+K*D)) time.
//
// Notes:
//   - No full covariance matrix is fitted; only diagonal entries are learned.
//
// AI-Hints:
//   - A variance floor is a constrained bound, not an epsilon added to every variance.
type GaussianFitResult struct {
	Model       *GaussianModel // Immutable last accepted constrained candidate on fit-partial errors.
	Diagnostics Diagnostics    // Caller-owned penalized objective and final occupancy evidence.
}

// IsNil reports whether the FilterResult pointer is nil, including typed-nil interfaces.
// It does not inspect optional fields or the logical completeness of a non-nil result.
// Implementation:
//   - Stage 1: Compare the pointer to nil without dereferencing it.
//   - Stage 2: Return the identity result without allocations.
//   - Stage 3: Preserve the core.Nilable semantics.
//
// Behavior highlights:
//   - Allocated zero-value results are not considered nil.
//
// Inputs:
//   - r: possibly nil result receiver.
//
// Returns:
//   - bool: true exactly when r==nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure pointer equality.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Safe when the typed nil receiver is stored in an interface.
//
// AI-Hints:
//   - Use IsNil for interface presence, not as an algorithm success validator.
func (r *FilterResult) IsNil() bool { return r == nil }

// IsNil reports whether the SmoothResult pointer is nil, including typed-nil interfaces.
// It is a pointer-presence check rather than a posterior validation operation.
// Implementation:
//   - Stage 1: Compare r against nil.
//   - Stage 2: Return the comparison.
//   - Stage 3: Avoid traversing any matrix.
//
// Behavior highlights:
//   - An allocated zero result is non-nil.
//
// Inputs:
//   - r: potentially typed-nil receiver.
//
// Returns:
//   - bool: true iff r==nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure pointer equality.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Never dereferences a nil receiver.
//
// AI-Hints:
//   - Inspect Posterior separately when checking a successful Smooth call.
func (r *SmoothResult) IsNil() bool { return r == nil }

// IsNil reports whether the DecodeResult pointer is nil, including typed-nil interfaces.
// It does not determine whether a path exists in an allocated result.
// Implementation:
//   - Stage 1: Compare pointer to nil.
//   - Stage 2: Return the comparison directly.
//   - Stage 3: Preserve caller ownership of States.
//
// Behavior highlights:
//   - A zero-value DecodeResult pointer is not nil.
//
// Inputs:
//   - r: potentially typed-nil receiver.
//
// Returns:
//   - bool: true iff r==nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure pointer equality.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - No path allocation or validation occurs.
//
// AI-Hints:
//   - Check Decode errors before consuming States.
func (r *DecodeResult) IsNil() bool { return r == nil }

// IsNil reports whether the CategoricalFitResult pointer is nil, including typed nil.
// It does not treat nonconvergence or a missing fitted model as nil pointer identity.
// Implementation:
//   - Stage 1: Compare r to nil.
//   - Stage 2: Return the comparison without dereferencing.
//   - Stage 3: Preserve the nilable convention.
//
// Behavior highlights:
//   - An allocated zero result remains non-nil.
//
// Inputs:
//   - r: potentially nil fit result.
//
// Returns:
//   - bool: true iff r==nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure pointer equality.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Nonconvergence can legitimately return a non-nil result.
//
// AI-Hints:
//   - Inspect the error and Converged flag before adopting a fitted model.
func (r *CategoricalFitResult) IsNil() bool { return r == nil }

// IsNil reports whether the GaussianFitResult pointer is nil, including typed nil.
// The check is independent of the availability or convergence of fitted parameters.
// Implementation:
//   - Stage 1: Compare r to nil.
//   - Stage 2: Return the comparison directly.
//   - Stage 3: Avoid touching the underlying model.
//
// Behavior highlights:
//   - An allocated zero result is non-nil.
//
// Inputs:
//   - r: potentially nil fit result.
//
// Returns:
//   - bool: true iff r==nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure pointer equality.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Check Diagnostics.Converged separately from IsNil.
//
// AI-Hints:
//   - Avoid accepting an unconverged variance model without application review.
func (r *GaussianFitResult) IsNil() bool { return r == nil }
