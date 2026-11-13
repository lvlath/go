// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package rng pins the canonical algorithm version used for numeric replay.
// This file isolates the immutable compatibility identifier from implementation.
package rng

// algorithmVersion identifies the complete seed, raw, uniform and normal draw protocol.
// It must change if any existing method changes output or raw-word consumption.
const algorithmVersion = "pcg-dxsm128-sha256le-u52-boxmuller-v1"
