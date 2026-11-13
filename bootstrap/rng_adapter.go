// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package bootstrap translates uniform 64-bit stream words into unbiased indices.
// The adapter owns neither a generator algorithm nor global random state.
package bootstrap

import "context"

// wordStream is the narrow source contract consumed by block sampling kernels.
// The owner must supply uniform 64-bit words and serialize all accesses per stream.
//
// Determinism:
//   - Draw order, including rejected words, is significant and versioned.
//
// AI-Hints:
//   - Do not replace this source with global math/rand or wall-clock randomness.
type wordStream interface {
	Uint64() uint64
}

// nativeIndexDigits maps a four-bit unsigned digit to the same native-int value.
// Its contents are fixed and are never modified; this is a decoding lookup,
// not shared RNG state, configuration, or an entropy source.
//
// AI-Hints:
//   - Keep the table ordered and read-only: its exact identity preserves the
//     integer mapping on 32-bit and 64-bit platforms without integer casts.
var nativeIndexDigits = [...]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

// uniformInteger returns an exact uniform integer in [0, bound) by rejection.
// For a uniform 64-bit source, every accepted residue has the same cardinality.
//
// Implementation:
//   - Stage 1: Check cancellation and validated positive bound; short-circuit bound=1.
//   - Stage 2: Reject uint64 words below the threshold 2^64 modulo bound.
//   - Stage 3: Reconstruct the bounded residue in native int without narrowing.
//
// Behavior highlights:
//   - For bound=1, returns 0 without requesting a random word.
//   - The modulo operation is applied only after the unbiased rejection step.
//   - A rejected word never advances the destination index position.
//
// Inputs:
//   - ctx: active non-nil context; checked for every attempted RNG word.
//   - stream: single-owner uniform uint64 source; no caller-side concurrent use.
//   - bound: strictly positive upper bound, no greater than native MaxInt.
//
// Returns:
//   - int: a sample in [0, bound); 0 with a non-nil error on failure.
//
// Errors:
//   - ctx.Err(): interruption, including during arbitrarily long rejection.
//   - ErrInvalidOption: nil context/source or nonpositive bound.
//
// Determinism:
//   - Consumes words in exact stream order until the first accepted word.
//
// Complexity:
//   - Expected Time O(1) on fixed-width machines, Space O(1). Each accepted
//     residue needs at most 16 nibble steps; rejection retries are unbounded.
//
// Notes:
//   - Unsigned negation computes -bound modulo 2^64 intentionally.
//   - The bound is a validated positive native int; accepted remainders are
//     strictly below bound and therefore mathematically fit native int.
//   - Uniformity is conditional on the source's word distribution.
//
// AI-Hints:
//   - Never replace rejection with u%bound: it biases every nonpower-of-two bound.
//   - Do not consume a word for bound=1; this is part of replay compatibility.
func uniformInteger(ctx context.Context, stream wordStream, bound int) (int, error) {
	if err := validateContext(ctx); err != nil {
		return 0, err
	}
	if bound <= 0 || stream == nil {
		return 0, ErrInvalidOption
	}
	if bound == 1 {
		return 0, nil
	}

	// Arithmetic is intentionally modulo 2^64; the threshold equals 2^64 % b.
	b := uint64(bound)
	threshold := (-b) % b

	for {
		// This check also ensures an unlucky sequence of rejected words is cancelable.
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		word := stream.Uint64()
		if word >= threshold {
			// The accepted interval is a multiple of b, so this residue is
			// exactly uniform. Since remainder < b <= native MaxInt, each
			// native-int digit and the assembled value are representable.
			return nativeIntFromRemainder(word % b), nil
		}
	}
}

// nativeIntFromRemainder reconstructs a proven-bounded uint64 residue as int.
// It avoids unsafe narrowing conversions without a string round-trip or heap
// allocation; each hexadecimal digit is selected from native-int constants.
//
// Implementation:
//   - Stage 1: Read from the fixed 16-entry native-int digit table.
//   - Stage 2: Extract low four-bit digits from least to most significant.
//   - Stage 3: Place each native-int digit at its original bit position.
//
// Behavior highlights:
//   - The caller guarantees remainder < bound <= MaxInt, so no sign bit is set.
//   - The zero residue requires no loop iterations and returns zero exactly.
//   - No uint64-to-int conversion or floating-point arithmetic occurs.
//
// Inputs:
//   - remainder: an accepted unsigned residue strictly below a positive int bound.
//
// Returns:
//   - int: the exact same nonnegative integer, reconstructed in native width.
//
// Errors:
//   - None; the caller proves the native integer domain before invocation.
//
// Determinism:
//   - Digits are combined from least-significant to most-significant position.
//
// Complexity:
//   - Time O(log_16(remainder+1)), Space O(1); at most 16 steps on 64-bit
//     and 8 steps on 32-bit platforms, with no heap allocation.
//
// Notes:
//   - The private lookup table is never modified by package code or callers.
//   - Go permits uint64 array indexes, so a masked nibble needs no type cast.
//
// AI-Hints:
//   - Do not replace this decoder with int(remainder): static analyzers can
//     flag that narrowing even when a preceding range check proves it safe.
//   - Do not change rejection or consume another stream word inside this helper.
func nativeIntFromRemainder(remainder uint64) int {
	const (
		bitsPerDigit = 4
		digitMask    = (1 << bitsPerDigit) - 1
	)

	// Indexing by uint64 is native Go syntax; the four-bit mask proves that
	// every access remains inside the fixed sixteen-entry digit table.
	var result int
	for shift := uint(0); remainder != 0; shift += bitsPerDigit {
		// Assemble each digit at its exact bit position. Since the original
		// residue is < MaxInt, neither the shift nor the OR can overflow.
		result |= nativeIndexDigits[remainder&digitMask] << shift
		remainder >>= bitsPerDigit
	}
	return result
}
