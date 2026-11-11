// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File: api.go
// Role: Canonical one-shot SAX facade and ergonomically narrow policy wrappers.
// AI-Hint: Wrappers delegate to the single Encoder kernel; they never duplicate math.

package sax

// defaultAlphabet is the conventional eight-symbol Gaussian alphabet used by
// DefaultOptions. It is a policy default, never a fallback for invalid input.
const defaultAlphabet = 8

// DefaultOptions constructs an explicit, safe population-normalized SAX policy.
// The supplied segment count remains mandatory; this constructor never changes
// an invalid count to a silently chosen default.
//
// Implementation:
//   - Stage 1: assign Segments from the caller without changing its value.
//   - Stage 2: select the eight-bin pinned Gaussian, population-Z policy.
//   - Stage 3: leave ReturnPAA disabled for a lightweight symbolic result.
//
// Behavior highlights:
//   - RejectConstant is the default; a constant window needs explicit ZeroWord.
//   - Zero-value Options is invalid because dimensions must be caller-selected.
//
// Inputs:
//   - segments: integer number w of PAA cells, required w>=1 at use time.
//
// Returns:
//   - Options: a value with default mathematical and presentation policies.
//
// Errors:
//   - None directly; NewEncoder reports ErrInvalidOption for segments<=0.
//
// Determinism:
//   - Pure fixed field construction, independent of platform state.
//
// Complexity:
//   - Time O(1), extra space O(1).
//
// Notes:
//   - Callers may change any field before passing Options to Encode.
//
// AI-Hints:
//   - Changing Alphabet or Normalization after DefaultOptions must still be
//     validated by the constructor; do not interpret defaults as final checks.
func DefaultOptions(segments int) Options {
	return Options{
		Segments:       segments,
		Alphabet:       defaultAlphabet,
		Normalization:  ZPopulation,
		ConstantPolicy: RejectConstant,
	}
}

// Encode constructs one SAX model and encodes a single finite observation window.
// This is the canonical one-shot facade; repeated-window workloads should
// construct an Encoder once to reuse its frozen policy and cached breakpoints.
//
// Implementation:
//   - Stage 1: validate Options and freeze the matching mathematical identity.
//   - Stage 2: validate and transform values through Encoder.Encode.
//   - Stage 3: return the fully detached Result or nil with its classified error.
//
// Behavior highlights:
//   - Options always precedes nil/source/shape validation.
//   - The same exact fractional PAA and quantization kernel serves all wrappers.
//
// Inputs:
//   - values: non-nil, nonempty, finite samples, read-only for this call.
//   - opts: explicit segment/alphabet/mode/threshold/diagnostic configuration.
//
// Returns:
//   - *Result: independently owned Symbols and optional PAA; nil on failure.
//   - error: package-owned sentinels; PAA root errors remain accessible via errors.Is.
//
// Errors:
//   - ErrInvalidOption/ErrShape during option admission; ErrNilInput,
//     ErrEmptyInput, ErrInvalidOption (w>n), ErrNonFinite during input admission;
//     ErrZeroVariance in population fitting; ErrOverflow for numeric failure.
//
// Determinism:
//   - Identical to Encoder.Encode: input scan then increasing segment index.
//
// Complexity:
//   - Time O(n+w log a+a), extra space O(n+w+a), including model creation;
//     one-time Gaussian cache decoding is paid only by the first constructor.
//
// Notes:
//   - No partial result is returned on any error; concurrent input mutation
//     is unsupported, and the input slice is never retained by the Encoder.
//
// AI-Hints:
//   - Use one Encoder for many windows; repeated one-shot calls rehash policy.
//   - Euclidean MINDIST cannot prove exact DTW pruning correctness.
func Encode(values []float64, opts Options) (*Result, error) {
	encoder, err := NewEncoder(opts)
	if err != nil {
		return nil, err
	}
	return encoder.Encode(values)
}

// EncodeGaussian encodes a population-Z window with pinned Gaussian thresholds.
// This wrapper makes both SAX dimensions explicit while using the same strict
// RejectConstant and diagnostic-output defaults as DefaultOptions.
//
// Implementation:
//   - Stage 1: build DefaultOptions(segments) and set the requested alphabet.
//   - Stage 2: pass unchanged observations and options to canonical Encode.
//   - Stage 3: return its complete Result and error identity.
//
// Behavior highlights:
//   - Gaussian quantiles are pinned; the mode is always ZPopulation.
//
// Inputs:
//   - values: finite read-only input window of length n>=segments.
//   - segments: w in [1,n]; alphabet: a in [2,256].
//
// Returns:
//   - *Result: detached normalized word; nil on error.
//
// Errors:
//   - Identical to Encode for the constructed explicit population-Z policy.
//
// Determinism:
//   - Input order and right-bin equality are unchanged from Encode.
//
// Complexity:
//   - Time O(n+w log a+a), extra space O(n+w+a).
//
// Notes:
//   - RejectConstant is intentional; use Encode with ZeroWord if required.
//
// AI-Hints:
//   - Do not apply this wrapper to constant windows expecting a zero-word;
//     the default correctly returns ErrZeroVariance.
func EncodeGaussian(values []float64, segments, alphabet int) (*Result, error) {
	opts := DefaultOptions(segments)
	opts.Alphabet = alphabet
	return Encode(values, opts)
}

// EncodeRaw encodes finite observations without window-local normalization.
// Gaussian quantiles are still used as fixed bin boundaries; this wrapper
// is useful when the input is already in suitable standardized units.
//
// Implementation:
//   - Stage 1: assemble explicit defaults with the requested dimensions.
//   - Stage 2: choose Raw identity coordinates, then call canonical Encode.
//   - Stage 3: return its fully detached word without diagnostic PAA.
//
// Behavior highlights:
//   - Mean=0 and StdDev=1 signal identity transformation, not a fitted window.
//
// Inputs:
//   - values: finite read-only observations in user-selected physical units.
//   - segments: w in [1,n]; alphabet: a in [2,256].
//
// Returns:
//   - *Result: detached Raw symbolic word; nil on failure.
//
// Errors:
//   - ErrInvalidOption, ErrNilInput, ErrEmptyInput, ErrNonFinite,
//     ErrShape (from invalid options), or ErrOverflow from underlying arithmetic.
//
// Determinism:
//   - Input order and ascending pinned thresholds; right-bin equality.
//
// Complexity:
//   - Time O(n+w log a+a), extra space O(w+a) for Raw, excluding PAA internals.
//
// Notes:
//   - Gaussian bin frequencies need not be equiprobable for unnormalized data.
//
// AI-Hints:
//   - For CPU percentages, rates and other domain units, prefer calibrated
//     thresholds via EncodeRawWithBreakpoints rather than standard-normal bins.
func EncodeRaw(values []float64, segments, alphabet int) (*Result, error) {
	opts := DefaultOptions(segments)
	opts.Alphabet = alphabet
	opts.Normalization = Raw
	return Encode(values, opts)
}

// EncodeRawWithBreakpoints encodes a Raw window with calibrated domain boundaries.
// Alphabet is inferred as len(breakpoints)+1; all thresholds are cloned in
// NewEncoder and the label is retained for audit, never trusted for identity.
//
// Implementation:
//   - Stage 1: build explicit Raw Options with the derived alphabet size.
//   - Stage 2: delegate strict finite/sorted/shape validation to NewEncoder.
//   - Stage 3: return the canonical Encode result without any hidden fitting.
//
// Behavior highlights:
//   - Each value equal to a threshold enters its right-hand bin.
//   - An absent or empty breakpoint slice is invalid for this custom wrapper.
//
// Inputs:
//   - values: read-only finite physical observations, n>=segments.
//   - segments: w>=1; breakpoints: 1..255 strictly ascending finite boundaries.
//   - breakpointID: nonempty human audit label, not part of EncodingID.
//
// Returns:
//   - *Result: detached Raw word; nil on failure.
//
// Errors:
//   - ErrInvalidOption for empty/oversized thresholds or invalid policy,
//     ErrShape/ErrInvalidOption for malformed breakpoints, plus Encode errors.
//
// Determinism:
//   - Threshold order is caller-provided; code is the count of thresholds<=v.
//
// Complexity:
//   - Time O(n+w log a+a), extra space O(w+a), including copied thresholds.
//
// Notes:
//   - Do not use the audit label as a key to compare separately built words.
//
// AI-Hints:
//   - Calibrate thresholds in the same units as values before indexing;
//     an unrelated breakpoints array cannot be fixed by changing the label.
func EncodeRawWithBreakpoints(values []float64, segments int, breakpoints []float64, breakpointID string) (*Result, error) {
	// A custom alphabet must have at least two and at most 256 bins. Guard
	// before len+1 to avoid an integer overflow on adversarial slice lengths.
	if len(breakpoints) < 1 || len(breakpoints) >= maxAlphabet {
		return nil, ErrInvalidOption
	}
	opts := Options{
		Segments:      segments,
		Alphabet:      len(breakpoints) + 1,
		Normalization: Raw,
		Breakpoints:   breakpoints,
		BreakpointID:  breakpointID,
	}
	return Encode(values, opts)
}

// MindistValues computes Euclidean SAX MINDIST for two compatible source windows.
// It is a convenience projection for occasional pairwise comparisons, not a
// DTW pruning certificate; for many candidates reuse NewEncoder and pre-encode.
//
// Implementation:
//   - Stage 1: validate and freeze opts in a single private Encoder.
//   - Stage 2: encode left then right under exactly the same numerical policy.
//   - Stage 3: validate both results through Encoder.Mindist and return distance.
//
// Behavior highlights:
//   - ReturnPAA is respected, although its artifacts are not returned here.
//   - Different source lengths are classified by Mindist as ErrIncompatible.
//
// Inputs:
//   - left, right: read-only, finite, nonempty sample windows of equal length.
//   - opts: full explicit SAX policy, including optional custom thresholds.
//
// Returns:
//   - float64: nonnegative Euclidean lower bound; 0 on any error.
//   - error: original classification from construction, encoding or comparison.
//
// Errors:
//   - ErrInvalidOption, ErrShape, ErrNilInput, ErrEmptyInput, ErrNonFinite,
//     ErrZeroVariance, ErrIncompatible, or ErrOverflow as applicable.
//
// Determinism:
//   - Source windows are read left then right; MINDIST scans increasing bins.
//
// Complexity:
//   - Time O(n_left+n_right+w log a+a), extra space O(n_left+n_right+w+a).
//
// Notes:
//   - No output Result is retained; repeated comparisons should reuse Encoder.
//
// AI-Hints:
//   - An exact DTW search cannot discard a candidate because this function
//     returned a positive distance. Use DTW-specific certified lower bounds.
func MindistValues(left, right []float64, opts Options) (float64, error) {
	encoder, err := NewEncoder(opts)
	if err != nil {
		return 0, err
	}
	leftResult, err := encoder.Encode(left)
	if err != nil {
		return 0, err
	}
	rightResult, err := encoder.Encode(right)
	if err != nil {
		return 0, err
	}
	return encoder.Mindist(leftResult, rightResult)
}
