// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

/*
Package gbm simulates geometric Brownian motion (GBM) by exact-in-distribution
lognormal transitions on a caller-supplied, possibly irregular time grid.
It produces positive price paths for scenario analysis without calibrating a
model, selecting a financial probability measure, or scheduling Monte Carlo
experiments.

# Mathematical charter

The modeled process satisfies

	dS(t) = mu*S(t)*dt + sigma*S(t)*dW(t),     S(0) > 0.

For each positive interval dt = t[i]-t[i-1], a caller-owned rng.RNG supplies
one standard-normal Z[i]. The conditional transition is

	deltaLogS = mu*dt - 0.5*(sigma*sqrt(dt))^2 + sigma*sqrt(dt)*Z[i].

The transitions are exact in distribution for constant mu and sigma, assuming
independent standard-normal draws. They are not Euler-Maruyama updates.
Floating-point calculations and the distribution of a custom RNG remain
separate sources of approximation. GBM is a model assumption, not evidence
that observed prices, returns, or shocks follow its distribution.

# Public surface and result law

Simulate(ctx,p,grid,random) is the canonical facade. SimulatePath builds the
same explicit Parameters and TimeGrid from scalar arguments. SimulateTerminal
selects a one-step [0,horizon] grid (or [0] when horizon is zero). SimulateSeeded
creates a fresh rng.NewStream(seed,pathID) for a replayable standalone path.
All three wrappers call Simulate; none implements a different transition kernel.

Parameters fixes Initial, Drift, Volatility and Unit. TimeGrid carries elapsed
timestamps and Unit. Result publishes Times, Values and Complete. On success,
len(Times) == len(Values) == len(grid.Values), the initial observation is
included and Complete is true. Result.IsNil implements core.Nilable by testing
pointer presence only; it does not imply that a trajectory is complete.

# Units and shape

TimeYears, TimeDays and TimeSeconds are the only accepted TimeUnit values.
Parameters.Unit must equal TimeGrid.Unit exactly. Drift is measured per Unit;
Volatility is measured per square root of Unit. A timestamp is elapsed time,
not a Unix epoch, calendar date, session index or implicit trading day.
No conversion by 252, 365, 365.25, or an exchange calendar occurs inside gbm.
An application changing time units must explicitly convert both coefficients
and every timestamp consistently.

TimeGrid.Values must be non-nil, nonempty and finite, start at exactly zero,
and increase strictly. The single-point grid [0] is valid, returns Initial
without evaluating a transition and consumes no random draws. Duplicate or
reversed observations are never sorted, deduplicated or interpolated.

# Numeric and entropy policy

Initial must be finite and strictly positive, Drift finite (including negative),
and Volatility finite and nonnegative. A non-nil rng.RNG is required even when
Volatility is zero. Recognized typed-nil generators are rejected when their
implementation exposes IsNil() bool; arbitrary typed-nil implementations
without that optional method cannot be identified without reflection.

For Volatility == 0, each endpoint follows the analytic exponential relative
to Initial, and no RNG method is called. For Volatility > 0, each attempted
stochastic interval calls random.Normal() exactly once, in timestamp order.
Uint64 and UniformOpen are never called by gbm. A nonfinite Normal draw is
ErrNonFinite; no attempt is made to infer a distribution from one draw.

The canonical kernel accumulates logarithmic growth relative to the unchanged,
exact float64 Initial. Fixed-precision, extended-exponent arithmetic computes
time differences, sqrt(dt), drift, Ito correction and Gaussian shock without
first squaring sigma as float64. Price conversion uses a guarded float64 fast
region or an extended-range exponential near dangerous magnitudes. It does not
update prices through S_previous*exp(deltaLogS), and it never floors a small
positive price to an arbitrary epsilon. Positive subnormal float64 prices
are valid. A newly computed zero or infinite price is ErrOverflow.

The wide arithmetic has a fixed significand precision. The returned float64
is a numerical approximation; correctly rounded exact-real GBM prices for
all possible finite inputs and identical last bits across platforms are NOT
guaranteed. Inputs must remain finite; an unrepresentable final price is not
silently converted to zero, infinity, an omitted observation or a usable path.

# Error precedence and cancellation

Simulate checks, in order: nil or canceled context; valid/matching units;
missing generator (including recognized typed nil); nil/empty grid;
nonfinite grid origin and zero origin; later timestamps in ascending order
(nonfinite then strict order at each index); nonfinite parameters; and physical
parameter domains. Preflight errors always return nil Result, including a
context canceled before simulation begins.

ErrInvalidOption covers nil context and invalid/mismatched units.
ErrNilInput covers a nil time slice or missing RNG. ErrEmptyInput covers an
allocated empty time slice. ErrOrder covers a nonzero origin, duplicate times
or decreasing times. ErrNonFinite covers nonfinite input data or a Normal draw.
ErrDomain covers nonpositive Initial or negative Volatility. ErrOverflow
covers an unrepresentable new price. Context errors retain ctx.Err() identity.
Callers must classify errors with errors.Is rather than comparing error text.

After validation, the simulator publishes [0] and checks ctx.Err() before
each additional step. A cancellation noticed at that boundary, a nonfinite
Normal draw or an unrepresentable new price returns a non-nil Result containing
exactly the completed prefix, including [0], with Complete == false and an
error. The failed step is never appended to Times or Values, and len(Times)
always equals len(Values). There is no RNG rollback. In particular, a Normal
call already in progress is not interruptible by gbm: cancellation is next
observed at a later step boundary. A one-point path needs no step iteration.

A montecarlo.Sampler must return an error whenever Simulate fails or its
Result is incomplete; it must never reduce a prefix terminal price as a full-
horizon payoff. The Monte Carlo driver owns path-ID scheduling, per-path RNG
streams and ordered statistical reductions, not this package.

# Ownership, concurrency and determinism

Parameters is copied by value. The input grid slice is borrowed read-only and
never retained; concurrent caller mutation of the input is unsupported. Both
Result slices are detached, caller-owned and have identical lengths. There is
no package-level mutable state. Each simulation owns its arithmetic workspace.
Simulate can run concurrently with stable shared Parameters/TimeGrid values
when each call has a different RNG stream; a single RNG stream must never be
used concurrently by multiple paths or goroutines.

Validation scans timestamps in supplied order. Normal draws follow ascending
step index. Every successfully converted price is committed as a time/value
pair, and no unfinished step is visible. Replaying identical float64 inputs,
RNG version, seed and logical path ID in the same environment reproduces the
same sequence. Monte Carlo worker count is not an RNG seed or path identifier.

# Complexity and limits

For n timestamps, validation and full-path simulation take O(n) operations and
allocate O(n) returned float64 storage. Arithmetic scratch is O(1) in n at
fixed implementation precision. SimulateTerminal has O(1) path length and
output size. Extended-precision square roots and exponential conversion may
carry substantial constant-factor CPU/allocation costs, especially at high
volatility or extreme prices; O(n) is not a throughput guarantee.

No parameter estimation, financial-calendar model, hidden annualization,
volatility clustering, jumps, trading rules, option payoff reduction,
portfolio correlation, stochastic-volatility model or adaptive error control
is performed. Time-unit conversion and calibration belong in caller code;
path aggregation and statistics belong in montecarlo.

# AI-Hints

  - Never replace anchored log-growth evolution with S_previous*exp(increment)
    or sigma*sigma*dt in float64: both can fail on representable prices.
  - Do not combine TimeYears model coefficients with TimeSeconds grid values;
    convert drift and volatility externally using different powers of time.
  - Check err AND Complete before using Values[len(Values)-1] as a horizon
    observation. A partial result is valid evidence only for its completed prefix.
  - Never clamp a rounded zero to SmallestNonzeroFloat64 or resample a failed
    stochastic step to hide ErrOverflow or ErrNonFinite.
  - Never share a mutable rng.RNG across Monte Carlo worker goroutines; use
    the driver-provided stream, or new seed/pathID streams for standalone paths.
  - Use docs/GBM.md and gbm/example_test.go for the executable API, models,
    dimensional analysis, full error matrix and operational examples.
*/
package gbm
