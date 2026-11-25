// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package flow exposes one internal numeric guard to the external test suite.
// This test-build bridge contains no test cases and is excluded from production builds.
package flow

// ValidateCapacityForTest forwards to the private capacity validator in test builds only.
// It permits isolated invalid-edge fixtures without mutating published core.Graph edges.
// Implementation: Direct alias; no alternate validation or injected failure logic.
// Behavior highlights: The actual production validator owns all numeric semantics.
// Inputs: The edge and epsilon accepted by validateCapacity.
// Returns: The original validated capacity and error.
// Errors: Unmodified sentinel errors from validateCapacity.
// Determinism: Identical to the private validator.
// Complexity: No additional asymptotic time or space cost.
// Notes: The Go tool includes this symbol only when compiling the package for tests.
// AI-Hints: Never move this bridge into a non-test source file.
var ValidateCapacityForTest = validateCapacity
