// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm provides deterministic, single-layer unidirectional float64 LSTM
// inference from explicitly supplied immutable weights. It evaluates the PyTorch
// IFGO ABI and returns detached recurrent states for batched or streamed sequences.
//
// # Scope and non-goals
//
// Import accepts matrix.Matrix input-to-hidden and recurrent coefficients and a
// combined bias. It never reads Python checkpoints, executes deserializers, trains
// weights, differentiates, chooses feature scaling, calibrates task-specific heads,
// or supports GPU, projections, bidirectionality and stacked LSTM layers.
//
// # Shapes and recurrence
//
// The batch-first Tensor3 has [B,T,D] geometry, B>=1, D>=1 and T>=0. Weight
// matrices have [4H,D] and [4H,H] geometry; Bias has 4H elements. Gate groups
// are strictly i, f, g, o, each with H rows. For each valid timestep:
//
//	a = Input*x + Recurrent*hprev + Bias
//	i = sigmoid(a_i); f = sigmoid(a_f); g = tanh(a_g); o = sigmoid(a_o)
//	c = f*cprev + i*g; h = o*tanh(c)
//
// Exactly zero active timesteps preserve the complete previous state. Each
// lengths[b] is local to its input chunk; nil lengths activates every timestep.
// Padding positions never update hidden/cell values and are exactly positive
// zero in returned Sequence. All input samples, including padding, are finite-checked.
//
// # Numerical and determinism laws
//
// Affine terms are reduced in ascending Input then Recurrent column order,
// for ascending batch, time and IFGO row indices. Compensated binary64
// arithmetic handles the normal path. Rare overflow/cancellation and extreme
// boundary values are recomputed as exact rational sums to distinguish a
// finite cancellation from an exact value outside float64 range. The latter
// returns ErrOverflow before nonlinearities; finite underflow uses float64
// rounding. The sigmoid has separate exponent-safe sign branches.
//
// Identical inputs and model within the same execution backend yield identical
// outputs. Full-vs-chunked inference is bitwise identical when initial states,
// local mask lengths, per-row chronological sample order and backend match.
// Compatibility with external PyTorch CPU libraries is anchored by numerical
// golden tests with a pinned fixture; universal bitwise PyTorch parity is not claimed.
//
// # Errors and precedence
//
// The canonical validation order is context (nil then canceled), nil model,
// tensor dimensions, exact Data length, local lengths, initial state ModelID
// and shape, then full numeric validation, followed by computation. Initial
// matrix values are finite-checked during state copying before input scanning.
// ErrInvalidOption, ErrNilInput, ErrShape, ErrStateVersion, ErrNonFinite and
// ErrOverflow are distinct sentinel classes; use errors.Is to inspect them.
// Cancellation and arithmetic failure always return a nil *Result.
//
// # Ownership and concurrency
//
// Import snapshots all numeric weights and hashes canonical little-endian
// float64 words (with -0 normalized to +0). Model.ID is immutable; no mutable
// global workspace, cache or lock is used. Concurrent Infer calls are safe when
// caller-owned input/state matrices are not concurrently mutated. Each call
// owns private cell/hidden workspace and publishes detached final matrices;
// callers may mutate successful results without modifying the Model.
//
// # Memory and performance
//
// Import costs O(H*(D+H)) time and space. Each Infer uses
// O(B*T*D+B*T*H*(D+H)) time, O(B*H+H) workspace, plus O(B*T*H) optional
// Sequence output and O(B*H) detached Final storage. Exact rational rescue
// is a rare input-dependent higher-cost path. The matrix.Matrix interface
// is accessed only during Import/state ingest/publication; the recurrent
// kernel reads locally owned contiguous row-major slices.
//
// # AI-Hints
//
//   - Never reinterpret PyTorch gate indices as IFOG or insert a silent transform.
//   - Never treat an overflowed affine sum as a legitimate saturated gate.
//   - Never propagate a hidden output into masked sequence positions.
//   - Always continue streams with both Final.Hidden and Final.Cell and the ModelID.
//   - Keep application-level feature scaling, model training and decision policy
//     outside this numerical inference package.
package lstm
