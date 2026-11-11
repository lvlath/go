// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File: validators.go
// Role: Centralized policy, result-shape, identity and numerical metadata admission.
// AI-Hint: A forged EncodingID never bypasses independent Result validation.

package sax

import "math"

// finite reports whether a binary64 value belongs to the package's numeric
// input domain. Both infinities and NaN are invalid, but signed zero is valid.
// Complexity: time O(1), space O(1); the predicate does not allocate.
func finite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

// validateOptions admits only policies that unambiguously determine SAX words.
// It checks scalar parameters before inspecting optional user thresholds,
// preserving documented ErrInvalidOption-before-ErrShape precedence.
//
// Implementation:
//   - Stage 1: validate segments, alphabet, and both enumeration domains.
//   - Stage 2: if custom thresholds are supplied, check their exact length.
//   - Stage 3: require a label, finite boundaries and strict monotonicity.
//
// Errors:
//   - ErrInvalidOption: scalars, absent label, or invalid threshold contents.
//   - ErrShape: custom threshold count differs from Alphabet-1.
//
// Complexity:
//   - Time O(a), extra space O(1); nil thresholds short-circuit in O(1).
//
// AI-Hints:
//   - A custom slice with len==0 is not equivalent to nil Gaussian thresholds.
//   - Do not sort caller data: it would silently redefine threshold identity.
func validateOptions(opts Options) error {
	if opts.Segments < 1 || opts.Alphabet < minAlphabet || opts.Alphabet > maxAlphabet ||
		opts.Normalization > Raw || opts.ConstantPolicy > ZeroWord {
		return ErrInvalidOption
	}
	if opts.Breakpoints == nil {
		return nil
	}
	if len(opts.Breakpoints) != opts.Alphabet-1 {
		return ErrShape
	}
	if opts.BreakpointID == "" {
		return ErrInvalidOption
	}
	for index, boundary := range opts.Breakpoints {
		if !finite(boundary) || (index > 0 && !(boundary > opts.Breakpoints[index-1])) {
			return ErrInvalidOption
		}
	}
	return nil
}

// validateEncoder protects exported methods from nil and zero-value receivers.
// A normally constructed Encoder has all internal fields initialized and
// immutable; this check does not rehash or recopy thresholds on every call.
//
// Errors:
//   - ErrNilInput: receiver is nil.
//   - ErrInvalidOption: uninitialized internal model (e.g. &Encoder{}).
//
// Complexity:
//   - Time O(1), extra space O(1).
//
// AI-Hints:
//   - Do not instantiate Encoder with a struct literal; use NewEncoder.
func (e *Encoder) validateEncoder() error {
	if e == nil {
		return ErrNilInput
	}
	if e.segments < 1 || e.alphabet < minAlphabet || e.alphabet > maxAlphabet ||
		len(e.breakpoints) != e.alphabet-1 || e.encodingID == "" ||
		e.normalization > Raw || e.constantPolicy > ZeroWord {
		return ErrInvalidOption
	}
	return nil
}

// validateResult checks the independent shape and finite-domain invariants of
// a caller-owned Result without trusting its purported EncodingID value.
// It intentionally does not attempt to authenticate mutable Result fields.
//
// Implementation:
//   - Stage 1: reject nil and incompatible slice dimensions or symbol codes.
//   - Stage 2: validate optional PAA coordinates and finite fitted metadata.
//   - Stage 3: require nonempty identity/version fields for later comparison.
//
// Errors:
//   - ErrNilInput: nil Result pointer.
//   - ErrShape: malformed length, symbol code or negative StdDev.
//   - ErrNonFinite: non-finite PAA/Mean/StdDev.
//   - ErrIncompatible: missing encoding identity/version.
//
// Complexity:
//   - Time O(w), extra space O(1), scanning only public Result data.
//
// AI-Hints:
//   - SHA-256 strings supplied by callers are untrusted metadata, not proof
//     that the Symbols or PAA were created by a genuine Encoder.
func validateResult(result *Result) error {
	if result == nil {
		return ErrNilInput
	}
	if result.OriginalLength < 1 || result.Segments < 1 ||
		result.Segments > result.OriginalLength ||
		result.Alphabet < minAlphabet || result.Alphabet > maxAlphabet ||
		len(result.Symbols) != result.Segments ||
		(result.PAA != nil && len(result.PAA) != result.Segments) {
		return ErrShape
	}
	for _, symbol := range result.Symbols {
		if int(symbol) >= result.Alphabet {
			return ErrShape
		}
	}
	for _, value := range result.PAA {
		if !finite(value) {
			return ErrNonFinite
		}
	}
	if !finite(result.Mean) || !finite(result.StdDev) {
		return ErrNonFinite
	}
	if result.StdDev < 0 {
		return ErrShape
	}
	if result.EncodingID == "" || result.NormalizationVersion == "" {
		return ErrIncompatible
	}
	return nil
}

// validateResultForEncoder verifies the fitted metadata under the actual
// mathematical policy and checks optional PAA against published bin symbols.
// A matching EncodingID alone cannot waive these independent field checks.
//
// Implementation:
//   - Stage 1: validate the declared normalization-version contract.
//   - Stage 2: enforce raw identity or population-Z degeneracy invariants.
//   - Stage 3: enforce exact-zero PAA for degenerate Z windows, then check
//     every optional PAA coordinate against its published bin symbol.
//
// Errors:
//   - ErrIncompatible: unknown normalization computation version.
//   - ErrShape: inconsistent mean/sigma/degeneracy or diagnostic symbol relation,
//     including a nonzero PAA coordinate in a constant ZeroWord window.
//
// Complexity:
//   - Time O(w log a) if PAA exists, otherwise O(1); extra space O(1).
//
// AI-Hints:
//   - A false Degenerate value is not harmless metadata; it changes how the
//     zero-standard-deviation result should be interpreted by downstream code.
func (e *Encoder) validateResultForEncoder(result *Result) error {
	if result.NormalizationVersion != normalizationVersion {
		return ErrIncompatible
	}
	if e.normalization == Raw {
		if result.Mean != 0 || result.StdDev != 1 || result.Degenerate {
			return ErrShape
		}
	} else if result.Degenerate {
		if e.constantPolicy != ZeroWord || result.StdDev != 0 {
			return ErrShape
		}
	} else if result.StdDev <= 0 {
		// A nonconstant window cannot be published with an underflowed sigma.
		return ErrShape
	}
	for i, value := range result.PAA {
		// The constant-policy kernel normalizes every source sample to exact
		// zero. Its PAA must therefore contain zeros as well, even when a
		// forged nonzero value happens to quantize to the same symbol.
		if result.Degenerate && value != 0 {
			return ErrShape
		}
		if quantize(value, e.breakpoints) != result.Symbols[i] {
			return ErrShape
		}
	}
	return nil
}
