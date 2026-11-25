// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// The numerical example illustrates exact offset removal without treating
// rounded fitted metadata as an exact internal center.
package sax_test

import (
	"fmt"

	"github.com/lvlath/go/sax"
)

// ExampleEncodeGaussian_largeOffset compares a short aligned sensor signature
// with an exactly representable offset in its acquisition coordinates.
// Equal words illustrate offset removal, not proof of identical raw signals.
func ExampleEncodeGaussian_largeOffset() {
	base, err := sax.EncodeGaussian([]float64{0, 2, 6}, 3, 8)
	if err != nil {
		fmt.Println(err)
		return
	}
	shifted, err := sax.EncodeGaussian([]float64{1e16, 1e16 + 2, 1e16 + 6}, 3, 8)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(base.Symbols)
	fmt.Println(shifted.Symbols)
	// Output:
	// [1 3 7]
	// [1 3 7]
}
