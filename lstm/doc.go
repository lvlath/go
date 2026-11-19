// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package lstm implements deterministic, single-layer, unidirectional LSTM
// inference from already trained float64 weights in PyTorch IFGO gate order.
// It publishes detached recurrent states and optional per-timestep features;
// no training, parameter estimation, or application decision policy occurs here.
//
// # Domain and non-goals
//
// Model is an immutable, imported input-to-hidden/recurrent layer. Weights
// carries the untrusted-by-default numerical import payload; Tensor3 carries
// a batch-first observation tensor; State carries both recurrent coordinates
// h and c plus the model identity; and Result is the sole canonical inference
// artifact. Infer and its convenience wrappers use one recurrent kernel.
//
// The package does not implement training, autodiff, optimizers, dropout,
// projections, bidirectionality, multiple layers, GPU execution, a prediction
// head, feature normalization, timestamp validation, a mutable streaming
// service, Python checkpoint loading, or arbitrary code deserialization.
// Trusted conversion of external training artifacts into finite matrices is
// the caller's responsibility. No graph-core arithmetic enters the kernel.
//
// # Shape and weight ABI
//
// B denotes batch size, T the supplied timesteps per batch row, D the feature
// width, and H the hidden/cell width. The tensor is contiguous row-major
// [B,T,D], with index (b*T+t)*D+d. B>=1, D>=1, T>=0 and every checked
// dimension product must equal the length of Data. The input D must match
// the imported model's input width; T=0 is legal and needs no Data storage.
//
// Import requires Input[4H,D], Recurrent[4H,H], Bias[4H], H>=1. The four
// consecutive H-row blocks are exactly input gate i, forget gate f, cell
// candidate g, and output gate o, in that order. Bias contains the already
// combined PyTorch input and recurrent biases, bias_ih_l0+bias_hh_l0.
// The Go State matrices are [B,H]; PyTorch single-layer state tensors have
// an additional leading layer dimension, which an external adapter removes.
//
// # Recurrent step and masked sequences
//
// For every active timestep, let a = Input*x + Recurrent*hPrev + Bias, with
// gate blocks ai, af, ag, ao. The kernel evaluates:
//
//	i = sigmoid(ai); f = sigmoid(af); g = tanh(ag); o = sigmoid(ao)
//	c = f*cPrev + i*g
//	h = o*tanh(c)
//
// All four preactivation blocks use the same previous hidden vector. Gates
// are computed before the previous hidden state is overwritten. The stable
// sigmoid evaluates 1/(1+exp(-x)) for x>=0 and exp(x)/(1+exp(x)) for x<0.
// It is applied only after a finite affine preactivation has been established.
//
// The lengths mask is nil (all T steps valid) or has exactly B entries in
// [0,T]. At t>=lengths[b], neither h nor c advances, and an enabled
// Result.Sequence contains exactly positive zero rather than the last h.
// All input.Data cells, including masked padding, must still be finite.
// Zero valid steps (including T=0) preserve the previous State, or the
// initial zero State when initial==nil.
//
// # Numeric domain and deterministic ordering
//
// Caller-provided weights, observations, and initial states must contain
// finite binary64 values; NaN and either infinity return ErrNonFinite.
// Ordinary affine evaluation uses a fixed-order compensated float64 sum.
// If intermediate overflow or a large boundary value could conceal a
// different exact outcome, the kernel recomputes the same products and sum
// as exact rationals of the binary64 operands. An exact magnitude larger
// than math.MaxFloat64 returns ErrOverflow, even if final float64 rounding
// could misleadingly produce a finite MaxFloat64. The cell transition uses
// an equivalent exact boundary classification for extreme finite values.
// Computed infinity is never reinterpreted as an activated sigmoid gate.
//
// Traversal order is ascending batch index, active timestep, IFGO row,
// input column and recurrent column; cell/hidden features then update in
// ascending hidden index. No map iteration, parallel reduction, RNG, or
// mutable global workspace is used. Equal weights, inputs, and execution
// backend therefore yield equal observable results. A sequence split into
// ordered chunks is bitwise replay-equivalent to a full call in the same Go
// kernel, provided the initial state, inputs, and per-chunk local masks match.
// This is not a universal bitwise parity guarantee for arbitrary PyTorch
// CPU backends; the pinned PyTorch double fixture uses an absolute tolerance.
//
// # Public invocation modes
//
// Import validates and copies the numerical weights. Model.ID exposes their
// versioned canonical fingerprint. Model.Infer(ctx,input,lengths,initial,opts)
// is the canonical public facade. InferDefault selects all steps, zero
// initial state and final-only output. InferFinal accepts masks and initial
// state but does not materialize a Sequence. InferSequence materializes a
// zero-padded [B,T,H] Sequence. InferChunk continues from previous.Final
// of a nonnil prior Result, with the same output choice as Infer.
//
// Option ReturnSequence=false leaves Sequence as the entire zero Tensor3
// value; Final.Hidden and Final.Cell are still always returned as new
// [B,H] matrices. For streaming, each lengths[b] describes the number of
// active timesteps in the CURRENT chunk, never the cumulative stream length.
// If global row length is L, a chunk starts at s and has width C, its local
// length is min(C,max(0,L-s)). Keep both Hidden and Cell in every continuation.
//
// # Error protocol and cancellation
//
// Infer checks nil/canceled context first, then nil Model, tensor dimensions
// and checked Data length, local lengths, and initial state. An initial
// ModelID mismatch returns ErrStateVersion before inspecting its matrices.
// Initial matrix presence, [B,H] shapes and finite values are validated
// during detached ingestion; the complete input.Data scan follows. Only
// then are optional output storage and the recurrent kernel evaluated.
// Missing required storage returns ErrNilInput; malformed geometry returns
// ErrShape; wrong masks or nil context return ErrInvalidOption. Dependency
// matrix errors preserve both a package sentinel and the underlying cause
// through errors.Join. Use errors.Is, never error-message matching.
//
// Cancellation is cooperatively checked before validation, periodically
// during full-data scanning and long gate computations, and before final
// publication. A failed import returns nil Model; failed/canceled inference
// returns nil *Result and an error, never a usable partial state. This
// differs from packages which deliberately publish completed prefixes.
//
// # Memory ownership and concurrency
//
// Import copies all finite coefficients into private row-major []float64
// buffers, normalizing -0 to +0, and never retains source matrices or Bias.
// Model.ID is a lowercase SHA-256 of the ASCII prefix "lstm/ifgo-f64/v1",
// LE64(D), LE64(H), and row-major little-endian IEEE-754 words of Input,
// Recurrent and Bias in that order. The ID fingerprints parameter bytes under
// this encoding; it is NOT artifact authenticity, provenance, or permission.
//
// Each call owns its hidden/cell workspace and returns detached Final
// matrices and optional Sequence data. A Model is safe for concurrent
// read-only inference without locks. Caller mutation of input tensors,
// initial states, or a previous Result while another call reads them is
// unsupported. Published results can be mutated after return without
// altering the model. A nil *Result supports IsNil() per core.Nilable;
// an allocated Result with zero fields is not proof of successful inference.
//
// # Complexity and performance
//
// Import uses O(H*(D+H)) time and owned parameter space. Infer validates
// O(B*T*D) input values and copies O(B*H) initial state values; the normal
// recurrence costs O(V*H*(D+H)), where V is the number of valid timesteps
// across rows. It allocates O(B*H+H) transient workspace, O(B*H) detached
// Final storage, and optionally O(B*T*H) zero-initialized Sequence storage.
// No heap buffer is allocated per normal recurrent step. Exceptional
// big.Rat rescue has input-dependent, higher time and memory costs.
// Checked integer products prevent geometry overflow before allocation;
// operating-system out-of-memory failures are not converted into errors.
//
// # AI-Hints
//
//   - Never change gate order to IFOG, even if another framework uses it.
//   - Never apply sigmoid to an overflowed affine value to hide ErrOverflow.
//   - Never treat padded input zero as a validity mask; use lengths explicitly.
//   - Do not replace masked outputs with the last hidden value; use Final
//     to access the last valid hidden/cell state.
//   - Keep both previous Hidden and Cell and the exact ModelID when chunking.
//   - Do not infer a failure probability, diagnosis, or trading signal from
//     uncalibrated hidden features: a validated head and application-specific
//     decision policy are separate responsibilities.
//   - Do not expose mutable Dense backing storage; copying at Import allows
//     the kernel to use private contiguous slices without unsafe/reflection.
//
// See lstm/example_test.go for runnable examples and docs/LSTM.md for
// equations, wrapper selection, PyTorch parity, masking, error precedence,
// resource behavior, and production troubleshooting.
package lstm
