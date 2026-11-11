// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// File: encoder.go
// Role: Immutable policy hashing, stable window fitting and canonical PAA encoding.
// AI-Hint: Never replace fractional paa.Transform or silently change fit arithmetic.

package sax

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"math"
	"sort"

	"github.com/lvlath/go/paa"
)

const (
	// normalizationVersion pins the arithmetic path for fitted population Z
	// scores. It participates in EncodingID and is published with every result.
	normalizationVersion = "zpopulation-scaled-twopass-v1"

	// encodingDomain separates the SAX mathematical identifier from hashes of
	// other contracts. The NUL byte prevents ambiguous domain concatenation.
	encodingDomain = "lvlath-sax-encoding-v1\x00"
)

// NewEncoder validates Options and freezes an immutable symbolic encoding model.
// Its SHA-256 identity covers dimensions, numerical policies, a normalization
// version, and the exact canonical binary64 content of every threshold.
//
// Implementation:
//   - Stage 1: validate all numeric and shape policy fields before allocation.
//   - Stage 2: clone custom breakpoints or copy the pinned Gaussian quantiles.
//   - Stage 3: serialize the mathematical configuration into SHA-256 and publish.
//
// Behavior highlights:
//   - The user-provided BreakpointID is an audit label, never a comparison key.
//   - ReturnPAA is a transport artifact, so it is excluded from EncodingID.
//   - Negative zero is serialized as positive zero; other bits are preserved.
//
// Inputs:
//   - opts: explicit validated policy; its Breakpoints slice is read-only for
//     the duration of construction and is never retained by reference.
//
// Returns:
//   - *Encoder: an independent immutable policy model; nil on failure.
//
// Errors:
//   - ErrInvalidOption: invalid dimensions/enums/custom threshold content.
//   - ErrShape: wrong number of custom thresholds; invalid shipped quantiles
//     are classified at the pinned-table loading boundary.
//
// Determinism:
//   - SHA-256 byte order: domain, Segments and Alphabet (uint64 little endian),
//     Normalization, ConstantPolicy, normalization version, optional table
//     version, then ascending threshold binary64 bits.
//
// Complexity:
//   - Time O(a), additional model space O(a), besides the first process-wide
//     O(32640) Gaussian cache initialization.
//
// Notes:
//   - EncodingID is compatibility, not authentication of a mutable Result.
//   - Separate encoder instances with equivalent mathematical config agree
//     even when audit labels or ReturnPAA differ.
//
// AI-Hints:
//   - Do not add BreakpointID to the compatibility key or remove threshold
//     content from it; equal labels do not imply equal mathematical bins.
//   - Do not update the table version without reviewing persisted SAX words.
func NewEncoder(opts Options) (*Encoder, error) {
	if err := validateOptions(opts); err != nil {
		return nil, err
	}

	// Resolve thresholds before calculating any model ID. Custom inputs must
	// be detached; pinnedBreakpoints also returns a detached alphabet row.
	var boundaries []float64
	if opts.Breakpoints == nil {
		var err error
		boundaries, err = pinnedBreakpoints(opts.Alphabet)
		if err != nil {
			return nil, err
		}
	} else {
		boundaries = append([]float64(nil), opts.Breakpoints...)
	}

	label := opts.BreakpointID
	if opts.Breakpoints == nil {
		// Gaussian tables have canonical provenance, not a user-selected label.
		label = quantileTableVersion
	}

	// Serialize only mathematical policy in a fixed order. Writing to SHA-256
	// cannot fail (hash.Hash.Write contract); bytes are never platform-native
	// struct layouts, pointers, or strings containing caller-defined labels.
	hasher := sha256.New()
	_, _ = hasher.Write([]byte(encodingDomain))
	// Both validated dimensions are nonnegative int values. Their canonical
	// eight-byte little-endian representation is independent of 32/64-bit Go int.
	writePositiveInt64LE(hasher, opts.Segments)
	writePositiveInt64LE(hasher, opts.Alphabet)
	_, _ = hasher.Write([]byte{byte(opts.Normalization), byte(opts.ConstantPolicy)})
	_, _ = hasher.Write([]byte(normalizationVersion))
	_, _ = hasher.Write([]byte{0})
	if opts.Breakpoints == nil {
		_, _ = hasher.Write([]byte(quantileTableVersion))
	}
	_, _ = hasher.Write([]byte{0})
	var word [8]byte
	for _, boundary := range boundaries {
		if boundary == 0 {
			// All IEEE-754 negative-zero bit patterns are made canonical here.
			boundary = 0
		}
		binary.LittleEndian.PutUint64(word[:], math.Float64bits(boundary))
		_, _ = hasher.Write(word[:])
	}

	return &Encoder{
		segments:       opts.Segments,
		alphabet:       opts.Alphabet,
		normalization:  opts.Normalization,
		constantPolicy: opts.ConstantPolicy,
		returnPAA:      opts.ReturnPAA,
		breakpoints:    boundaries,
		breakpointID:   label,
		encodingID:     hex.EncodeToString(hasher.Sum(nil)),
	}, nil
}

// writePositiveInt64LE serializes a validated nonnegative Go int as eight
// little-endian bytes without a platform-dependent machine-width conversion.
// Each emitted byte is explicitly masked, so higher 32-bit or 64-bit words
// produce the same byte sequence on their common nonnegative integer domain.
// Complexity: time O(1), extra space O(1); hash.Hash.Write cannot fail for SHA-256.
func writePositiveInt64LE(hasher hash.Hash, value int) {
	var encoded [8]byte
	for index := range encoded {
		encoded[index] = byte(value & 0xff)
		value >>= 8
	}
	_, _ = hasher.Write(encoded[:])
}

// Encode transforms one finite window into a SAX word under the frozen policy.
// It calls the canonical fractional paa.Transform kernel on the window's Raw
// values or window-local Z scores; no integer-block approximation is used.
//
// Implementation:
//   - Stage 1: validate receiver, source presence, size and all finite inputs.
//   - Stage 2: normalize in a detached workspace when required, then perform
//     weighted fractional PAA with the existing package-level kernel.
//   - Stage 3: quantize in time order and publish a detached Result/metadata.
//
// Behavior highlights:
//   - Constant population-Z windows are rejected or mapped to exact zeros.
//   - Raw is an identity transformation: Mean=0, StdDev=1, Degenerate=false.
//   - Equality to any boundary is always assigned to its right-hand bin.
//
// Inputs:
//   - values: non-nil, nonempty finite input samples, read-only for this call;
//     a window may not have fewer samples than the model's Segments count.
//
// Returns:
//   - *Result: detached Symbols and optionally detached normalized PAA values;
//     nil on every error, with no partially published artifact.
//
// Errors:
//   - ErrNilInput: nil receiver or nil input slice; ErrInvalidOption: invalid
//     receiver or Segments>n; ErrEmptyInput: zero length; ErrNonFinite: bad sample;
//     ErrZeroVariance: constant Z window under RejectConstant; ErrOverflow:
//     unrepresentable fit or PAA arithmetic (PAA root cause preserved).
//
// Determinism:
//   - Finite values checked in source order; PAA cells and symbols ascending.
//   - Every window is fitted independently, with no global/future statistics.
//
// Complexity:
//   - Time O(n+w log a), extra space O(n+w) in ZPopulation;
//     in Raw mode PAA storage and output are O(w).
//
// Notes:
//   - Input and model are never modified or retained by the result.
//   - Concurrent calls on the same immutable Encoder are safe if callers do
//     not concurrently mutate their source or returned Result fields.
//
// AI-Hints:
//   - Do not replace fractional PAA with floor(n/w) blocks, even when the
//     segment count does not divide source length evenly.
//   - SAX is not a safe pruning certificate for arbitrary DTW searches.
func (e *Encoder) Encode(values []float64) (*Result, error) {
	if err := e.validateEncoder(); err != nil {
		return nil, err
	}
	if values == nil {
		return nil, ErrNilInput
	}
	length := len(values)
	if length == 0 {
		return nil, ErrEmptyInput
	}
	if e.segments > length {
		return nil, ErrInvalidOption
	}

	// Do not inspect individual samples before the shape gate. The complete
	// finite-domain check precedes all normalization and PAA allocations.
	for _, observation := range values {
		if !finite(observation) {
			return nil, ErrNonFinite
		}
	}

	transformed := values
	mean, stddev := 0.0, 1.0
	degenerate := false
	if e.normalization == ZPopulation {
		var err error
		transformed, mean, stddev, degenerate, err = normalizeWindow(values, e.constantPolicy)
		if err != nil {
			return nil, err
		}
	}

	// Delegation ensures a single source of truth for fractional overlap,
	// nearest-even rounding and checked PAA geometry. Never reimplement it.
	aggregated, err := paa.Transform(transformed, e.segments)
	if err != nil {
		if errors.Is(err, paa.ErrOverflow) {
			return nil, fmt.Errorf("sax PAA: %w: %w", ErrOverflow, err)
		}
		return nil, err
	}

	// The w-element symbol slice belongs exclusively to the returned Result;
	// the shared Gaussian table and encoder threshold storage remain private.
	symbols := make([]uint16, e.segments)
	for segmentIndex, average := range aggregated.Values {
		if !finite(average) {
			return nil, ErrOverflow
		}
		symbols[segmentIndex] = quantize(average, e.breakpoints)
	}

	result := &Result{
		Symbols:              symbols,
		OriginalLength:       length,
		Segments:             e.segments,
		Alphabet:             e.alphabet,
		Mean:                 mean,
		StdDev:               stddev,
		BreakpointID:         e.breakpointID,
		EncodingID:           e.encodingID,
		NormalizationVersion: normalizationVersion,
		Degenerate:           degenerate,
	}
	if e.returnPAA {
		// paa.Transform already returns detached Values; transferring its slice
		// does not create an alias to the input, model or any other call.
		result.PAA = aggregated.Values
	}
	return result, nil
}

// quantize assigns one finite PAA mean to its right-continuous Gaussian or
// custom threshold bin. It is private because Symbols can only be interpreted
// together with a full Result and the Encoder's immutable configuration.
//
// Implementation:
//   - Stage 1: rely on Encode's finite data and ascending-boundary preconditions.
//   - Stage 2: binary search the FIRST boundary strictly greater than value.
//   - Stage 3: use that index, equal to the count of boundaries <= value.
//
// Behavior highlights:
//   - v==b[k] always enters bin k+1; v above the last threshold enters a-1.
//
// Inputs:
//   - value: finite binary64 coordinate; boundaries: a-1 sorted finite values.
//
// Returns:
//   - uint16: a valid symbolic index in [0,a); a<=256 by constructor law.
//
// Errors:
//   - None; domain/shape failures are rejected before entering this kernel.
//
// Determinism:
//   - Binary search uses the exact strict comparison b[i]>value.
//
// Complexity:
//   - Time O(log a), extra space O(1).
//
// AI-Hints:
//   - Replacing > with >= silently moves threshold equality into the wrong bin.
//
// nolint:gosec
func quantize(value float64, boundaries []float64) uint16 {
	index := sort.Search(len(boundaries), func(position int) bool {
		return boundaries[position] > value
	})
	// Constructor validation bounds index to [0,255]. The mask makes the
	// eight-bit bin domain explicit before representing it as a uint16 code.
	return uint16(index & (maxAlphabet - 1))
}

// normalizeWindow performs window-local population normalization in scaled units.
// Centering is performed after scaling to avoid overflow when subtracting large
// finite observations; corrected squares are accumulated before final Z scores.
//
// Implementation:
//   - Stage 1: detect exact constant windows, and select RejectConstant/ZeroWord.
//   - Stage 2: scale by the largest absolute value, subtract a fixed origin,
//     compute incremental centered mean and compensated population variance.
//   - Stage 3: compute finite metadata and divide normalized residuals by sigma.
//
// Behavior highlights:
//   - ZeroWord yields n exact zero coordinates, with a zero StdDev and flag.
//   - No numeric epsilon is used to reclassify nonconstant windows as constant.
//   - A genuinely positive sigma that underflows float64 is ErrOverflow,
//     NOT an apparently successful non-degenerate result with StdDev=0.
//
// Inputs:
//   - values: validated nonempty finite read-only observations, length n.
//   - policy: a validated ConstantPolicy, relevant to equal-valued windows.
//
// Returns:
//   - []float64: detached Z coordinates, including normalized constant zeros.
//   - mean/stddev: fitted in original value units, both finite on success.
//   - bool: whether the window was constant and ZeroWord was applied.
//
// Errors:
//   - ErrZeroVariance: an exactly constant RejectConstant window.
//   - ErrOverflow: unrepresentable numerical fit, including lost variation
//     during scaling or stddev underflow
//     from a nonconstant input or non-finite derived coordinates.
//
// Determinism:
//   - Fixed first-sample origin, ascending incremental mean updates, ascending
//     residual scans and compensated summation; no cross-window fitted state.
//
// Complexity:
//   - Time O(n), extra space O(n), one detached coordinate workspace.
//
// Notes:
//   - Numeric metadata is a representable binary64 summary; tiny finite
//     source deviations can underflow when converted to original sigma units.
//
// AI-Hints:
//   - Do not use an arbitrary variance epsilon, which would erase small but
//     meaningful changes; only exact identical samples use ZeroWord.
//   - Do not subtract two unscaled values near MaxFloat64 (possible overflow).
func normalizeWindow(values []float64, policy ConstantPolicy) ([]float64, float64, float64, bool, error) {
	length := len(values)
	constant := true
	scale := 0.0
	for _, observation := range values {
		if observation != values[0] {
			constant = false
		}
		scale = math.Max(scale, math.Abs(observation))
	}
	if constant {
		if policy == RejectConstant {
			return nil, 0, 0, false, ErrZeroVariance
		}
		// Floating +0.0 is the normalized coordinate, not necessarily bin zero.
		return make([]float64, length), values[0], 0, true, nil
	}

	// Scale first, then choose a local origin before centered reductions.
	// This avoids x-origin overflow even when input signs are opposite.
	centered := make([]float64, length)
	origin := values[0] / scale
	centerMean := 0.0
	for index, observation := range values {
		shift := observation/scale - origin
		centered[index] = shift
		centerMean += (shift - centerMean) / float64(index+1)
	}

	// Remove the fitted shift and find a stable residual scale. A zero result
	// after distinct inputs means scale arithmetic lost observable variation;
	// treating it as an exact constant would manufacture a false ZeroWord.
	residualScale := 0.0
	for index := range centered {
		centered[index] -= centerMean
		residualScale = math.Max(residualScale, math.Abs(centered[index]))
	}
	if residualScale == 0 {
		return nil, 0, 0, false, ErrOverflow
	}

	// Compensated squares in residual-scale units prevent squaring overflow.
	// The fixed-order summation also avoids a platform-dependent parallel tree.
	squares := 0.0
	correction := 0.0
	for _, residual := range centered {
		ratio := residual / residualScale
		term := ratio * ratio
		delta := term - correction
		next := squares + delta
		correction = (next - squares) - delta
		squares = next
	}
	populationRatio := math.Sqrt(squares / float64(length))
	if populationRatio == 0 || !finite(populationRatio) {
		return nil, 0, 0, false, ErrOverflow
	}

	// Calculate fitted metadata in input units. Converting a positive sigma
	// to zero would falsely advertise a nonconstant window as zero-variance.
	mean := (origin + centerMean) * scale
	stddev := (residualScale * populationRatio) * scale
	if !finite(mean) || !finite(stddev) || stddev == 0 {
		return nil, 0, 0, false, ErrOverflow
	}

	for index := range centered {
		centered[index] = (centered[index] / residualScale) / populationRatio
		if !finite(centered[index]) {
			return nil, 0, 0, false, ErrOverflow
		}
	}
	return centered, mean, stddev, false, nil
}
