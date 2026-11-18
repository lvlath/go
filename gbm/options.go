// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package gbm deliberately exposes no optional mathematical mode in this file.
// Initial, drift, volatility, time units, observation times, and RNG ownership
// are explicit parameters of the canonical Simulate contract. Introducing an
// unused Options value or an implicit annualization switch would create a
// second policy surface without adding legitimate GBM functionality.
package gbm
