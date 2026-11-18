// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

/*
Package gbm simulates Geometric Brownian Motion (GBM) on an explicitly unit-typed,
possibly irregular time grid using exact-in-distribution conditional lognormal
increments. It does not discretize the stochastic differential equation with
Euler-Maruyama; floating-point rounding and RNG quality remain relevant.

Domain charter:

The model is dS(t) = mu*S(t)*dt + sigma*S(t)*dW(t), S(0)>0. A step of
length dt>0 adds mu*dt - 0.5*(sigma*sqrt(dt))^2 + sigma*sqrt(dt)*Z
to the log price, with an independent standard-normal draw Z supplied by rng.RNG.
The simulator retains cumulative log growth relative to the unchanged initial
price; numerical operations must not prematurely evaluate exp(a large increment).

TimeGrid.Values must be nonnil, nonempty, finite, start at zero, and be strictly
increasing. A one-point grid [0] is valid. Only TimeYears, TimeDays, and
TimeSeconds exist; Parameters.Unit and TimeGrid.Unit must match exactly.
Drift has units 1/Unit and Volatility has units 1/sqrt(Unit). No conversion
between seconds, days, years, or trading sessions is performed.

Numeric policy:

Initial is finite and strictly positive; Drift is finite; Volatility is finite
and nonnegative. Stochastic steps consume exactly one finite Normal() value,
including when its value is zero. With Volatility=0, no RNG method is called,
but the RNG argument must still be nonnil. Extended-exponent arithmetic avoids
spurious intermediate sigma^2*dt and price-exponential overflow. Final positive
subnormal float64 prices are legal; a zero or infinite computed price raises
ErrOverflow. No arbitrary positive epsilon floor is applied. Fixed-precision
arithmetic is deterministic, not an arbitrary-precision exact real solver.

Error and partial-result law:

A nil context causes ErrInvalidOption; a context canceled before work returns
its original error and nil Result. Invalid units, missing data, order violations,
nonfinite inputs and physical-domain violations also return nil Result. A context
interrupted between steps, nonfinite Normal draw, or unrepresentable final price
returns an error and a nonnil Result containing only completed observations,
including [0]. Both slices always have equal length and Complete is false.
Successful paths are complete and have exactly len(TimeGrid.Values) entries.
Callers must not present an incomplete prefix as a terminal-horizon sample to
montecarlo.Sampler; instead the adapter must return its original error.

Ownership and concurrency:

Time grids are borrowed read-only during the call and never retained. Returned
Times and Values are detached caller-owned slices. No mutable global RNG or
state is used. Each rng.RNG is mutable and belongs to one goroutine/path;
sharing a stream across concurrent simulations is unsupported. Concurrent
Simulate calls with immutable inputs and separate RNG streams are allowed.
No reflect, panic, or recover is used by the production package.

Canonical API and convenience facades:

Simulate is the only mathematical kernel entrance. SimulatePath constructs
explicitly unit-typed parameter/grid values, SimulateTerminal provides a
zero-or-one-step horizon, and SimulateSeeded constructs a path-specific stream.
They all delegate to Simulate without changing its mathematical law or errors.

Non-goals:

No market-session inference, automatic annualization, calibration, inference
of GBM model suitability, trading strategies, price constraints, risk-neutral
measure selection, payoff reduction, or Monte Carlo worker scheduling are
performed here. These are caller or neighboring-package responsibilities.

Complexity: O(n) time and returned memory for n timestamps, with fixed-size
wide-number scratch. Public wrappers have the same asymptotic cost, except
SimulateTerminal, which always uses O(1) time and memory.

AI-Hints:

  - Never update the trajectory by S_previous*exp(largeIncrement), nor silently
    convert annual drift into per-second drift.
  - Never clamp an underflowed path to an arbitrary epsilon or publish a
    nonfinite price as a successful sample.
  - Check both error and Complete before constructing any terminal payoff.
  - Use one RNG stream per logical pathID, not one mutable stream per worker.
*/
package gbm
