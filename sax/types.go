// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File: types.go
// Role: Immutable encoder and caller-owned public Result contracts.
// AI-Hint: Nilable detects typed-nil Result pointers, not malformed non-nil structures.

package sax

// Normalization selects the coordinate system used by the PAA and MINDIST kernels.
// Both sequences must use the same mode for the symbolic Euclidean lower bound
// to have a mathematical interpretation; the mode participates in EncodingID.
//
// Behavior highlights:
//   - ZPopulation normalizes each window separately; Raw preserves amplitudes.
//   - The zero value is ZPopulation, not an unspecified automatic mode.
//
// Determinism:
//   - Stable uint8 codes are serialized into EncodingID without reinterpretation.
//
// Notes:
//   - Changing a mode is a change of mathematical space, not a speed setting.
//
// AI-Hints:
//   - Never compare Raw and ZPopulation results by their Symbols alone.
//   - Window-local Z scores are not a global fit and cannot include future data.
type Normalization uint8

const (
	// ZPopulation fits one population mean and standard deviation to the
	// current observation window (divisor n); nonconstant windows are scaled
	// to Z scores, while constant windows follow ConstantPolicy.
	ZPopulation Normalization = iota

	// Raw leaves every finite input value unchanged before fractional PAA.
	// Result.Mean=0 and Result.StdDev=1 identify this identity transformation.
	Raw
)

// ConstantPolicy determines the representation of constant ZPopulation windows.
// A constant is detected from equality of finite input values, not by using
// an arbitrary variance tolerance that could erase genuine small variation.
//
// Behavior highlights:
//   - Raw mode ignores this behavior during encoding, but the policy remains
//     part of the mathematical EncodingID for consistent configuration identity.
//
// Notes:
//   - Underflow of a nonzero standard deviation is not a constant window.
//
// AI-Hints:
//   - ZeroWord controls the normalized vector, not numeric symbol indices;
//     the symbol for zero depends on the selected breakpoints.
type ConstantPolicy uint8

const (
	// RejectConstant refuses a constant population-Z window with
	// ErrZeroVariance. It is the safe default for normalized pattern comparison.
	RejectConstant ConstantPolicy = iota

	// ZeroWord maps a constant population-Z window to a vector of exact +0.0
	// coordinates, sets Result.Degenerate and records its original mean.
	// It does not promise that the resulting symbol indices are zero.
	ZeroWord
)

// Options defines SAX's mathematical encoding policy and optional diagnostic output.
// A successful NewEncoder copies the user-supplied thresholds, so changing
// Options or its Breakpoints later cannot mutate the configured model.
//
// Implementation:
//   - Stage 1: validate explicit segment/alphabet/mode and custom breakpoint laws.
//   - Stage 2: copy custom boundaries or select pinned Gaussian quantiles.
//   - Stage 3: compute mathematical EncodingID, excluding diagnostic flags.
//
// Behavior highlights:
//   - Zero-value Options is invalid; use DefaultOptions or explicit fields.
//   - Nil Breakpoints chooses pinned Gaussian quantiles for Alphabet.
//   - Non-nil custom Breakpoints require exactly Alphabet-1 finite, strictly
//     increasing values and a nonempty audit-only BreakpointID.
//   - ReturnPAA controls output materialization, not Symbols or EncodingID.
//
// Inputs:
//   - Segments: positive PAA segment count w; at Encode time w must not exceed n.
//   - Alphabet: number of symbolic bins a in [2,256].
//   - Normalization: Raw or ZPopulation for the current observation window.
//   - ConstantPolicy: RejectConstant or ZeroWord when using ZPopulation.
//   - Breakpoints: optional ascending binary64 boundaries, read-only for constructor.
//   - BreakpointID: user audit label, never a compatibility/authentication key.
//   - ReturnPAA: whether Result contains detached aggregated segment values.
//
// Returns:
//   - The value is a configuration description, not an initialized Encoder.
//
// Errors:
//   - NewEncoder reports ErrInvalidOption or ErrShape before reading values.
//
// Determinism:
//   - Boundaries retain their supplied binary64 values; negative zero is
//     serialized as positive zero for the configuration identity.
//
// Complexity:
//   - Validation/copy costs O(a) time and space (Gaussian cache initialization
//     is a separate, one-time operation).
//
// Notes:
//   - There is no owned input source and no hidden default for a zero Options.
//
// AI-Hints:
//   - ReturnPAA is a presentation option; excluding it from the identity is
//     intentional, whereas excluding Normalization would break comparability.
//   - BreakpointID is human-readable provenance, not proof of threshold content.
type Options struct {
	// Segments is the required number w of equal-duration PAA intervals.
	// It must be positive and must not exceed the source window length n.
	Segments int

	// Alphabet is the number a of symbolic bins; only [2,256] is supported.
	// Each successful symbol is an unsigned index in [0,a).
	Alphabet int

	// Normalization chooses window-local population Z scores or Raw identity.
	// This mathematical policy participates in EncodingID.
	Normalization Normalization

	// ConstantPolicy applies when a ZPopulation window has exact zero variance.
	// Even in Raw mode the configured policy participates in EncodingID.
	ConstantPolicy ConstantPolicy

	// Breakpoints is nil for pinned Gaussian quantiles, or exactly a-1
	// finite, strictly ascending custom boundaries; NewEncoder copies it.
	Breakpoints []float64

	// BreakpointID is a nonempty audit label required for custom boundaries.
	// It is not a comparison key, hash input, or cryptographic authenticator.
	BreakpointID string

	// ReturnPAA requests a detached diagnostic PAA slice in each Result.
	// It does not influence the symbolic word or its mathematical EncodingID.
	ReturnPAA bool
}

// Encoder is an immutable model that transforms windows into compatible SAX words.
// Thresholds, policies and canonical identity are private; a constructor copies
// the caller's breakpoint slice before any inference request is accepted.
//
// Implementation:
//   - Stage 1: NewEncoder validates and freezes the policy.
//   - Stage 2: Encode allocates a per-call normalization/PAA workspace.
//   - Stage 3: Mindist reads immutable thresholds and validated caller results.
//
// Behavior highlights:
//   - An uninitialized zero-value Encoder is invalid; use NewEncoder.
//
// Returns:
//   - *Encoder: an immutable model with no exposed mutation methods.
//
// Errors:
//   - ErrInvalidOption from constructor or an uninitialized receiver.
//   - ErrNilInput from a nil *Encoder receiver.
//
// Determinism:
//   - Inference scans input sample order, then increasing PAA segment indices.
//
// Complexity:
//   - Retained model storage O(a); construction time O(a), plus the first
//     process-wide pinned-quantile decode when using Gaussian breakpoints.
//
// Notes:
//   - Concurrent Encode and Mindist calls are safe if caller-owned inputs and
//     Result fields are not modified concurrently with those calls.
//
// AI-Hints:
//   - Reuse one Encoder for many same-policy windows; one-shot Encode creates
//     and hashes a new model for every call.
type Encoder struct {
	segments       int
	alphabet       int
	normalization  Normalization
	constantPolicy ConstantPolicy
	returnPAA      bool
	breakpoints    []float64
	breakpointID   string
	encodingID     string
}

// Result carries a SAX word and the metadata required to validate comparison.
// Every successful result is detached: the caller owns Symbols and optional
// PAA independently of the Encoder, input observations and other results.
//
// Implementation:
//   - Stage 1: Encode validates the input, computes its normalized PAA values.
//   - Stage 2: encode each PAA value into an unsigned bin index in [0,Alphabet).
//   - Stage 3: publish copied metadata and optional diagnostic PAA coordinates.
//
// Behavior highlights:
//   - len(Symbols)==Segments, 1<=Segments<=OriginalLength, 2<=Alphabet<=256.
//   - PAA is nil unless ReturnPAA was requested, otherwise len(PAA)==Segments.
//   - Raw: Mean=0, StdDev=1, Degenerate=false; ZPopulation: per-window fit.
//   - ZPopulation + ZeroWord on a constant: Mean=constant, StdDev=0,
//     Degenerate=true; the zero-coordinate bin need not be symbol index zero.
//
// Returns:
//   - Caller-owned mutable artifact; no partial Result on an Encode error.
//
// Errors:
//   - Result does not itself return errors; Mindist validates mutable fields
//     and reports ErrShape, ErrNonFinite, or ErrIncompatible if corrupted.
//
// Determinism:
//   - Symbols and PAA use ascending segment index order on a fixed time grid.
//
// Complexity:
//   - Retained space O(w); IsNil uses constant time and space.
//
// Notes:
//   - EncodingID is a SHA-256 mathematical compatibility identifier, not a
//     cryptographic proof of authenticity of this publicly mutable structure.
//   - BreakpointID is informational; it may differ between compatible results.
//   - Results must not be modified while another goroutine reads them.
//
// AI-Hints:
//   - Do not discard OriginalLength: n/w is part of the distance scale.
//   - A Result with manually forged EncodingID cannot be authenticated by Mindist.
type Result struct {
	// Symbols contains exactly Segments indices in [0,Alphabet), in time order.
	Symbols []uint16
	// PAA optionally stores the transformed window's segment means; detached.
	PAA []float64
	// OriginalLength is n, the number of source samples in the encoded window.
	OriginalLength int
	// Segments is w, the number of equally spaced fractional PAA cells.
	Segments int
	// Alphabet is a, the number of symbolic quantization bins in [2,256].
	Alphabet int
	// Mean is the fitted window mean in ZPopulation, or 0 for Raw.
	Mean float64
	// StdDev is the fitted population sigma in ZPopulation, or 1 for Raw.
	StdDev float64
	// BreakpointID is an audit label; comparison ignores its text.
	BreakpointID string
	// EncodingID is the deterministic SHA-256 of mathematical configuration.
	EncodingID string
	// NormalizationVersion pins the window-normalization numerical algorithm.
	NormalizationVersion string
	// Degenerate identifies a constant ZPopulation window encoded under ZeroWord.
	Degenerate bool
}

// IsNil reports whether the Result receiver is a nil pointer.
// The predicate implements core.Nilable structurally without reflection or
// conflating an uninitialized but non-nil Result with nil.
//
// Implementation:
//   - Stage 1: compare the receiver with nil without dereferencing its fields.
//   - Stage 2: return the comparison unchanged.
//
// Behavior highlights:
//   - Typed-nil *Result inside core.Nilable reports true; &Result{} reports false.
//
// Inputs:
//   - r: possibly nil receiver; no ownership or metadata constraints.
//
// Returns:
//   - bool: true exactly when r == nil.
//
// Errors:
//   - None; this method does not allocate, mutate, or panic.
//
// Determinism:
//   - Pure pointer comparison independent of observable Result content.
//
// Complexity:
//   - Time O(1), extra space O(1).
//
// Notes:
//   - IsNil is not a shape/semantic validator and is safe under concurrent calls.
//
// AI-Hints:
//   - Never replace this with len(r.Symbols)==0: zero-value Results are non-nil.
func (r *Result) IsNil() bool {
	return r == nil
}
