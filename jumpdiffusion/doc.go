// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package jumpdiffusion simulates Merton compound-Poisson lognormal jumps
// superimposed on geometric Brownian motion at caller-supplied observation times.
// It is a grid-observation scenario generator, not a calibrated tail-risk forecast.
//
// Domain and mathematics:
//   - The public model is Parameters{Diffusion: gbm.Parameters, Intensity, ...}.
//   - For each positive dt, N~Poisson(Intensity*dt) independently of Brownian Z.
//   - Conditional on N>0, the aggregate log jump is N*m+sqrt(N)*s*Zjump.
//   - Compensated removes Intensity*expm1(m+s^2/2) from proportional drift.
//   - Compensated E[S(t)]=Initial*exp(Drift*t); Uncompensated retains
//     E[S(t)]=Initial*exp((Drift+Intensity*k)*t), when expectations are finite.
//   - The identities describe the ideal model, not an empirical guarantee for
//     finitely many accepted paths, a selected market measure, or calibration.
//
// Shape and numeric law:
//   - A valid grid is nonnil, nonempty, finite, begins at zero and rises strictly.
//   - Unit must match gbm.Parameters.Unit; no automatic calendar conversion.
//   - Initial>0; volatility, intensity, and log-jump stddev are finite and >=0.
//   - Jump mean is any finite float64; a nonfinite jump coefficient is invalid
//     even when Intensity is zero and no jump draws are required.
//   - Poisson mean is constrained to [0,2^40] and count to [0,2^53-1].
//   - Every published price is positive and finite, including legal subnormals.
//   - Compensated evaluates k via Expm1 after a wide mean+half-variance sum;
//     Uncompensated and Intensity=0 skip unnecessary compensator arithmetic.
//   - The interval dt, lambda*dt, Ito scale and log growth use fixed-precision
//     wide arithmetic; final binary64 conversion is a separate checked stage.
//   - Compensator or price unrepresentability is an explicit ErrOverflow.
//   - A strictly positive mean rounding to binary64 zero is ErrOverflow,
//     not a jump-free step: the Poisson draw protocol is never bypassed.
//
// Sampling and determinism law:
//   - Intensity=0 delegates directly to gbm.Simulate, preserving bits and draw order.
//   - At positive intensity, per step: Normal(diffusion) only if volatility>0,
//     UniformOpen (Poisson) for every supported positive mean, and Normal(jump)
//     only if N>0 and the jump standard deviation>0.
//   - For 0<a<30 the count uses product inversion; for 30<=a<=2^40 it uses
//     Hoermann PTRS transformed rejection with a stable deviance/Stirling
//     log-PMF, never a rounded-normal approximation.
//   - Each PTRS attempt consumes two UniformOpen draws in that order; an
//     independent jump Normal is drawn only for N>0 and positive jump stddev.
//   - At sigma=0 the diffusion Normal is not called. Poisson calls remain
//     required when Intensity>0 even if m=s=0 makes jump prices unchanged.
//   - Every sampling attempt is context-checked and the total attempts per
//     interval never exceed 2^20; no Gaussian approximation is used.
//   - Fixed seed/pathID, same RNG/kernel versions, and same floating inputs
//     reproduce the exact traversal and publication order on one backend.
//
// Error precedence:
//   - Nil/pre-canceled context first, then units/convention, RNG/grid presence,
//     empty grid, timestamp scan (first offending index), coefficient finiteness,
//     physical domains, and compensated k representability.
//   - Every interval checks its Poisson mean before its diffusion Normal draw.
//     A runtime error after valid preflight returns only the previously
//     committed time/price pairs and never publishes a failed candidate.
//
// Errors and partial-result law:
//   - Validation failure returns nil Result with ErrInvalidOption, ErrNilInput,
//     ErrEmptyInput, ErrOrder, ErrNonFinite, ErrDomain, or ErrOverflow.
//   - A nil context is ErrInvalidOption; a pre-canceled context returns ctx.Err().
//   - Runtime cancellation, ErrBudget, ErrDomain, ErrNonFinite and ErrOverflow
//     return only completed timestamps/prices in a non-nil Result, Complete=false.
//   - Published Times/Values remain equal-length and detached from grid.Values;
//     a failed step is discarded, while its consumed RNG draws are not restored.
//   - Errors must be classified via errors.Is, never by parsing error text.
//
// Public API and result semantics:
//   - Simulate is the canonical facade; SimulatePath, SimulateTerminal,
//     SimulateSeeded and SimulateTerminalSeeded delegate without changing
//     mathematics or error identity. A zero horizon requests only [0].
//   - *Result implements core.Nilable through nil-safe IsNil; IsNil reports
//     pointer absence rather than successful completion.
//   - Under Intensity=0, the GBM Result is wrapped and applicable GBM error
//     causes remain discoverable by errors.Is.
//
// Ownership and concurrency law:
//   - Model parameters are passed by value; the input grid is read-only during calls.
//   - Returned result slices are caller-owned and do not alias input arrays.
//   - Independent calls are safe with distinct streams; one mutable rng.RNG
//     must be owned by exactly one goroutine at any given moment.
//   - No reflection, panic, recover, global RNG, time source, or hidden locks.
//
// Non-goals:
//   - Intrastep jump timestamps, individual jump marks, Hawkes or stochastic
//     intensities, mixture tails, market calibration, and stochastic volatility.
//   - Brownian-bridge barrier crossing and exact continuous-time barrier monitoring.
//   - Theoretical martingale growth does not imply finite-sample unbiasedness
//     after path failures, truncation, selection or externally imposed clamps.
//   - Financial advice, validated crash prediction, or risk-neutral measure selection.
//
// Source and licensing:
//   - The PTRS implementation is native Go. Its Hoermann proposal and squeeze
//     parameterization were cross-checked against NumPy 2.2.0, which is not a
//     runtime, module, or cgo dependency. See THIRD_PARTY_NOTICES.txt for
//     source attribution and retained BSD-3-Clause license text.
//
// Complexity and API:
//   - Simulate validates n timestamps in O(n), allocates O(n) detached output,
//     and uses O(1) working memory with fixed-precision wide arithmetic.
//   - Worst-case time O(n*2^20) due to bounded rejection attempts; typical
//     PTRS count sampling has a small expected number of iterations.
//   - SimulatePath, SimulateTerminal, SimulateSeeded, and
//     SimulateTerminalSeeded delegate to Simulate;
//     they never introduce an alternate mathematical kernel.
//
// AI-Hints:
//   - Never substitute log-jump mean m for arithmetic jump compensator k.
//   - Never approximate high-intensity Poisson counts with rounded normals.
//   - Never pool one mutable rng.RNG across concurrent montecarlo.Sample calls.
//   - Treat Complete=false as a useful diagnostic prefix, not a terminal payoff.
//   - Preserve the Poisson threshold, PTRS draw order, and numeric guards
//     when refactoring; changing them invalidates replay of seeded scenarios.
//   - In montecarlo.Sampler, use the supplied path RNG and reject incomplete
//     trajectories rather than creating an independent stream in Sample.
package jumpdiffusion
