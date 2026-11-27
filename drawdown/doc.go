// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package drawdown computes maximum drawdown of observed equity and recovers
// one exact witness episode for deterministic risk auditing.
//
// This file fixes the package charter, numerical laws, and operational boundaries.
//
// # What and why
//
// The metric provides an explainable, path-dependent risk summary for sampled
// portfolio histories, simulation paths, and independently audited backtests.
// It measures the loss from an attained running peak, not return volatility.
// A terminal gain can coexist with a severe earlier drawdown.
//
// # Domain and public contract
//
// Compute accepts a nonempty ordered equity history with a positive initial
// observation and finite nonnegative later observations. Result.Depth is defined
// at every index, in [0,1], with canonical positive zero and no warmup mask.
// Result.Maximum, PeakIndex, TroughIndex and RecoveryIndex summarize the witness.
// Compute is canonical. ComputeMaximum projects the historical Maximum;
// ComputeCurrentDepth projects the last Depth element of the admitted prefix.
// Both delegate to Compute, preserve error precedence, and return +0 with error
// on failure (not a valid measurement). Both still allocate the full O(n) profile.
// There are no options, implicit modes, Stream or Snapshot.
//
// # Mathematical and ordering laws
//
// At index t, P is the running maximum through t and E is equity[t]. The
// published depth is RN64((P-E)/P) with round-to-nearest, ties-to-even, applied
// to the exact binary64 inputs. Near the peak subtraction is exact by Sterbenz;
// deeper losses use fixed-width integer quotient/remainder rounding on a 2^-53
// grid, with an even integer complement preserving midpoint parity. Products of
// normalized 53-bit mantissas compare residual E/P ratios without rounding or
// floating-point overflow. Exact equal depths retain the earliest trough; the
// peak index is the first occurrence of its running peak. Rounded collisions
// are not mathematical ties. Maximum copies Depth[TroughIndex] bit for bit.
//
// The recovery index is the first later observation >= the selected peak, or
// -1 when absent. With no decrease below the running peak the result is +0 and
// witness 0/0/0. Later zero equity represents total loss. A positive equity
// may also produce a rounded depth of 1; the exact comparator distinguishes it.
//
// # Errors and cancellation
//
// Context precedes shape, then values are checked by increasing index:
// finite check before domain at each index. Nil context gives ErrNilContext;
// nil/empty equity ErrInsufficient; NaN/Inf ErrNonFinite; finite nonpositive
// first or negative later equity ErrEquity. Use errors.Is. Cancellation returns
// ctx.Err(); polling occurs at most every 1024 observations and before publication.
// After starting, concurrent input error and cancellation follow first detection.
// Every error returns nil Result. Values are validated as the first private pass
// progresses; no intermediate output can escape. Arithmetic overflow is excluded
// by the domain and fixed-width bounds; normal runtime memory exhaustion is not
// converted into a domain error. Context polling is an iteration bound, not a
// wall-clock latency guarantee. Typed-nil or misbehaving custom contexts are
// outside the usable-context contract; cancellation after the final successful
// check cannot retract publication.
//
// # Ownership and concurrency
//
// No input mutation, retained buffers or shared state. Callers own the detached
// depth slice. A shallow Result copy aliases that slice; copy the data explicitly
// for independent mutation. Concurrent independent calls are safe if inputs stay read-only
// during each call; concurrent mutation of input or a shared Result is unsupported.
// Result.IsNil is safe for a nil receiver and checks pointer presence only; it
// structurally satisfies core.Nilable without importing core.
//
// # Complexity and limits
//
// One ascending profile/validation pass and one optional recovery suffix pass
// give O(n) time, O(n) output and O(1) auxiliary space. The exact integer path
// uses a bounded number of machine words; no per-observation allocation occurs
// on success. Depth is prefix invariant; Maximum and recovery are retrospective
// summaries of the supplied history, not causal features for earlier decisions.
// Repeated batch calls over every growing prefix require quadratic total work;
// a scalar wrapper does not change this fact.
//
// # Non-goals and AI-Hints
//
// Supply already prepared equity or external cashflow-adjusted NAV in one unit.
// Sampling, currencies, deposits, leveraged insolvency, intrabar excursions,
// calendar durations, risk limits, Ulcer Index and strategy decisions are external.
// Donchian, ATR, VWAP and SessionLevels are observed price features, not account
// equity. An external ledger maps executions and cashflows into a consistently
// sampled NAV before this package is invoked; there is no import dependency.
// Do not use rounded Depth as the exact witness comparator or treat a deposit as
// investment return. Do not interpret -1 recovery as a calendar interval.
// For a rounded trailing limit use the current depth, not historical Maximum;
// exact rational threshold certification is not part of the API.
package drawdown
