// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 katalvlaran

package series

import "math"

// validateGrid enforces shape and finiteness before timestamp-domain rules.
// It never subtracts timestamps until representability is established.
func validateGrid(timesNS []int64, values []float64, opts GridOptions) (*ValidationResult, error) {
	if opts.CadenceNS < 0 {
		return nil, ErrInvalidOption
	}
	if timesNS == nil || values == nil {
		return nil, ErrNilInput
	}
	if len(timesNS) != len(values) {
		return nil, ErrShape
	}
	if len(timesNS) == 0 {
		return nil, ErrEmptyInput
	}
	if err := validateFinite(values); err != nil {
		return nil, err
	}

	result := &ValidationResult{Samples: len(values), Gaps: make([]Gap, 0)}
	for i := 1; i < len(timesNS); i++ {
		prev, next := timesNS[i-1], timesNS[i]
		if next <= prev {
			return nil, ErrOrder
		}
		// An increasing signed-int64 pair can overflow only when it crosses
		// from negative to positive with span greater than MaxInt64.
		if prev < 0 && next > 0 && next > math.MaxInt64+prev {
			return nil, ErrOverflow
		}
		delta := next - prev
		if opts.CadenceNS == 0 {
			continue
		}
		if delta%opts.CadenceNS != 0 {
			return nil, ErrGap
		}
		missing := delta/opts.CadenceNS - 1
		if missing > 0 {
			if !opts.AllowGaps {
				return nil, ErrGap
			}
			result.Gaps = append(result.Gaps, Gap{AfterIndex: i - 1, MissingCount: missing})
		}
	}
	return result, nil
}
