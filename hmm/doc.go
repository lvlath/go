// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package hmm implements typed finite-state hidden Markov models (HMMs).
// It separates causal filtering, retrospective smoothing, joint-path decoding,
// and deterministic multi-sequence Baum-Welch training under explicit log-domain
// numerical and ownership contracts.
//
// # Domain and model families
//
// A hidden state z_t belongs to [0,K); transitions A[i,j] encode
// P(z_t=j|z_(t-1)=i). Prior[i] is P(z_0=i). Both are finite,
// nonnegative stochastic simplexes with absolute row-sum tolerance 1e-12.
// Constructors normalize only accepted residuals on private copies: exact zeros
// remain exact zeros. K>=1. A model has exactly one emission family:
//
//   - CategoricalModel: K-by-A row-stochastic symbol probabilities, A>=1,
//     observations are uint32 symbol indices in [0,A).
//   - GaussianModel: K-by-D means and strictly positive diagonal variances,
//     D>=1; each time step has D finite coordinates in consistent feature units.
//     Variances are in squared feature units, not standard deviations.
//
// Full-covariance Gaussians, missing-feature imputation, time-grid reconstruction,
// discriminative labels, duration models, and automatic state selection are outside
// this package. State indices are arbitrary labels: their interpretations are
// application-owned and may permute between separately fitted models.
//
// # Four distinct mathematical operations
//
// Filtering computes alpha_t(j)=P(z_t=j | x_0,...,x_t). It never reads a
// future observation. FilterStep(ctx,nil,x) begins at the model prior; otherwise
// it requires a model-matching, normalized checkpoint. FilterDefault omits the
// history matrix, keeping live workspace O(K). FilterWithHistory returns a
// detached T-by-K matrix of causal probabilities. Every successful FilterResult
// includes a final FilterState with cumulative marginal log likelihood.
//
// Smoothing computes gamma_t(j)=P(z_t=j | x_0,...,x_(T-1)) through a separate
// forward-backward pass. Unlike filtering, gamma_t can change when future
// observations are appended. Smooth always returns the complete T-by-K gamma
// matrix. Its workspace is O(T*K), not the O(T*K*K) xi transition table used
// exclusively during parameter fitting.
//
// Viterbi decoding finds the single maximum-joint-probability hidden state path.
// It is not equivalent to maximizing a filtered or smoothed posterior at each
// time independently. Exact equal-score alternatives choose the lowest
// predecessor index, then the lowest final state index. DecodeResult contains
// the detached state path and its maximum joint log probability.
//
// Training fits explicit initial CategoricalSpec/GaussianSpec values over one
// or more independent, nonempty sequences. Baum-Welch accumulates gamma at
// each sequence's first observation into the prior, and xi only across that
// sequence's internal adjacent steps. It never joins the end of one sequence
// with the start of the next; no random initialization or reseeding occurs.
//
// # Numerical law
//
// Forward, backward and Viterbi combine logarithms of probabilities, using
// LogSumExp instead of direct long products. A zero transition or categorical
// emission has log mass -Inf. LogSumExp(all -Inf) is exactly -Inf, and a
// categorical observation with no remaining possible state returns
// ErrImpossibleSequence, never an artificial positive probability or NaN.
//
// A Gaussian emission is a probability *density* and can exceed 1. Its log
// density, and thus accumulated LogLikelihood, may be positive. Finite input
// producing an unrepresentable Gaussian log density is ErrOverflow, not an
// impossible sequence. Floating-point addition order is fixed and documented;
// the implementation does not claim correctly rounded arbitrary-precision
// results or an absence of roundoff at extreme scales.
//
// # EM objective and constrained M-step
//
// FitOptions{} selects MaxIterations=100, Tolerance=1e-6, Pseudocount=0, and
// VarianceFloor=1e-8 in feature-squared units. Pseudocount=lambda>=0 means
// a symmetric Dirichlet MAP penalty alpha=1+lambda on the prior, transition
// rows and categorical emission rows; Gaussian means and variances have no
// Dirichlet penalty. The monitored objective is
//
//	log P(all sequences | theta) + lambda * sum(log(simplex entries)).
//
// Lambda=0 disables this penalty, even for exact zero probabilities (no
// 0*log(0)). Lambda>0 and an initial zero simplex cell may produce initial
// objective -Inf, which may legitimately improve to a finite candidate.
// Gaussian variance updates project onto the constrained domain by
// max(estimatedVariance,VarianceFloor), not by adding epsilon. FitGaussian
// rejects an initial variance below that floor with ErrInvalidOption.
//
// Zero-occupancy states retain their previous Gaussian emission parameters;
// zero-count categorical/transition rows retain their previous values at
// lambda=0. No state is silently removed. The M-step uses fixed sequence,
// time, state, and feature traversal order, not a worker-dependent reduction.
// Diagnostics.Objective includes the initial score and each accepted M-step;
// Iterations counts accepted steps and Occupancy reports final expected visits.
//
// On MaxIterations exhaustion, Fit returns the last accepted model together
// with ErrNotConverged and Converged=false. On a declining penalized objective,
// Fit returns its last accepted model together with ErrNumerical. These are
// diagnostic partial-result exceptions, not successful fits; applications
// must inspect errors.Is and Diagnostics.Converged before deployment. Invalid
// inputs and context cancellation return nil results instead.
//
// # Input validation, ownership and identity
//
// New public operations check context (when present), options, nil/typed-nil,
// dimensions, nonfinite values, domain constraints, then numerical computation.
// Context==nil is ErrInvalidOption; a canceled context returns ctx.Err().
// Model parameters are deep-copied and immutable after construction.
// The constructors hash normalized canonical parameters, model family,
// dimensions, and algorithm version into ModelID. An external FilterState
// must match that ID, have Steps>=1, have K log posterior entries whose
// LogSumExp is within 1e-12 of zero, and have finite cumulative evidence.
// A MaxUint64 Steps counter produces ErrOverflow before increment.
//
// FilterStep never changes prev, even on failure; on success it returns an
// independent checkpoint. Returned histories, paths and fit diagnostics are
// caller-owned. A *FilterState value-copy does share the LogPosterior backing
// array: callers must deep-copy that slice if they intend to modify a saved
// checkpoint. ModelID is an identity check, not an authenticated or durable
// binary state serialization protocol. Do not mutate model input matrices
// concurrently with a call that reads them.
//
// All five result types implement core.Nilable via IsNil: the nilness of the
// receiver pointer, not logical model validity or success of a computation.
// Errors are package-owned sentinels classified with errors.Is, never by
// matching the diagnostic error string. There are no calls to panic, recover,
// or reflect in this package.
//
// # Complexity and concurrency
//
// With K states, A categorical symbols, D Gaussian features, T time steps,
// and S independent sequences, dense transitions require O(T*K*K) time for
// inference. Categorical emissions add O(T*K); Gaussian emissions add
// O(T*K*D). Filtering without history needs O(K) live workspace for a
// categorical sequence (Gaussian batch input also copies T*D feature values);
// filtering with history, smoothing and Viterbi backpointers require O(T*K).
// EM adds a private (T_s-1)*K*K xi table per processed sequence; its peak
// workspace is proportional to the longest sequence rather than the sum of
// all such tables, plus model/count buffers and Gaussian input copies.
//
// Models are immutable and independent inference calls use separate scratch
// memory. Concurrent inference is safe if caller-owned observations remain
// unchanged during their calls. Fit operations do not mutate caller inputs,
// and mutable returned results require external synchronization if shared.
//
// # Usage guidance
//
// Use FilterStep for live regime signals, FilterWithHistory to plot causal
// trajectories, Smooth for retrospective attribution, Decode for a single
// globally consistent state path, and Fit* for explicit multi-sequence MAP
// estimation. Timestamp cadence and missing-grid policy belong in series;
// feature standardization belongs in stats or an application-fixed transform;
// symbolization (for example SAX) must use a stable, explicit alphabet map.
//
// See docs/HMM.md for derivations, error examples, a causal-versus-smoothed
// counterexample, operational scenarios, and integration with the wider lvlath
// ecosystem. Executable usage lives in hmm/example_test.go.
package hmm
