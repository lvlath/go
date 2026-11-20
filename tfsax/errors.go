// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax declares stable error identities for the experimental DTW research harness.
// This file is the sole definition of sentinel categories used with errors.Is.
package tfsax

import "errors"

// Sentinel-first error governance:
//   - Wrap identities with %w or errors.Join; message text is never machine protocol.
//   - Preserve underlying dtw/sax identities where an adapter adds local classification.
var (
	// ErrInvalidInput identifies missing or malformed corpus/query data.
	// It covers invalid shapes and frozen data mismatches without exposing sample values.
	ErrInvalidInput = errors.New("tfsax: invalid input")

	// ErrInvalidOption identifies invalid experiment or search policy.
	// It includes nil contexts, bad bounds, missing output paths and malformed schema options.
	ErrInvalidOption = errors.New("tfsax: invalid option")

	// ErrNonFinite identifies an invalid NaN/Inf sample or nonfinite heuristic score.
	// Finite input must never silently enter an exact distance or symbolic index.
	ErrNonFinite = errors.New("tfsax: non-finite sample")

	// ErrDuplicateID identifies a repeated stable candidate, query or membership identity.
	// Duplicate IDs never use a silent first-wins or last-wins resolution policy.
	ErrDuplicateID = errors.New("tfsax: duplicate ID")

	// ErrLeakage identifies crossed raw-time support or invalid train-only fit membership.
	// An untrusted provenance mismatch must block research measurement before it begins.
	ErrLeakage = errors.New("tfsax: temporal or membership leakage")

	// ErrUndefinedRepresentation identifies the missing reviewed symbolic algorithm definition.
	// A user-supplied Approved flag cannot manufacture a production encoder implementation.
	ErrUndefinedRepresentation = errors.New("tfsax: representation definition not approved")

	// ErrUncertifiedBound identifies an unavailable proof-safe DTW pruning rule.
	// Exhaustive finite test cases alone do not certify arbitrary real and float64 inputs.
	ErrUncertifiedBound = errors.New("tfsax: lower bound not certified")

	// ErrBudget identifies a predeclared experiment sample/cell budget exhaustion.
	// A guarded allocation failure is not converted into a panic-based success fallback.
	ErrBudget = errors.New("tfsax: resource budget exceeded")

	// ErrOverflow identifies unrepresentable floating-point arithmetic.
	// Exact DTW overflow remains distinct from structural unreachability.
	ErrOverflow = errors.New("tfsax: numerical overflow")
)
