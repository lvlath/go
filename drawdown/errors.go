// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package drawdown defines stable error categories for sampled equity audits.
// This file owns sentinel identity, independent of diagnostic text.
package drawdown

import "errors"

// Sentinel identities are shared by Compute and its scalar projections.
// Use errors.Is rather than matching text; indexed diagnostics may wrap these values.
var (
	// ErrNilContext indicates that Compute received a nil context interface.
	// This is checked before any data access; cancellation of a usable context
	// returns ctx.Err() instead. All facades preserve this classification.
	ErrNilContext = errors.New("drawdown: nil context")

	// ErrInsufficient indicates that the equity history contains no observations.
	// Nil and non-nil empty slices have identical error semantics, after context
	// admission and before accessing index zero or allocating the output.
	ErrInsufficient = errors.New("drawdown: insufficient equity observations")

	// ErrEquity indicates a finite nonpositive initial or negative later equity.
	// Finiteness is checked first at the same index. Initial signed zero fails;
	// later signed zero is valid total loss. No signed-equity model is inferred.
	ErrEquity = errors.New("drawdown: equity outside domain")

	// ErrNonFinite indicates that an equity observation is NaN or infinite.
	// It takes precedence over the equity domain error at that observation, but
	// not over an invalid earlier observation. The original index is diagnostic.
	ErrNonFinite = errors.New("drawdown: non-finite equity")
)
