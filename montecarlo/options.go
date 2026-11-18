// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package montecarlo names the fixed policy constants used by its Plan.
// The public value-type Plan is the sole configuration surface: there is no
// hidden functional-option assembler or alternate numerical mode.
package montecarlo

const (
	// minimumPaths permits Bessel-corrected variance on every successful run.
	minimumPaths uint64 = 2

	// defaultWorkers chooses a serial experiment for a zero Workers field.
	defaultWorkers = 1

	// schedulerWindowFactor bounds accepted-but-unreduced paths to twice the
	// number of effective workers, including running and queued outcomes.
	schedulerWindowFactor = 2

	// bytesPerFloat64 is the exact backing storage cost of one scalar observation.
	bytesPerFloat64 uint64 = 8

	// defaultSamplesBytes bounds the optional full sample/sorting storage to 64 MiB.
	defaultSamplesBytes uint64 = 64 << 20
)
