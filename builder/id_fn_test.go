package builder_test

import (
	"testing"

	"github.com/lvlath/go/builder"
)

// assertPanics fails the test if the provided function does not panic.
// It recovers from a panic and marks the test as failed if none occurred.
func assertPanics(t *testing.T, fn func(), name string) {
	t.Helper()     // mark this function as a test helper
	defer func() { // set up a deferred function to recover from panic
		if r := recover(); r == nil { // if recover returns nil, no panic happened
			t.Errorf("%s: expected panic, but none occurred", name) // report failure
		}
	}()
	fn() // invoke the function under test, which should panic
}

// TestIDFns verifies each IDFn implementation both for correct outputs on valid inputs
// and for panics on invalid inputs. Uses table-driven subtests for clarity and completeness.
func TestIDFns(t *testing.T) {
	t.Parallel() // allow this test to run in parallel with other tests

	// define a table of test cases for all IDFn implementations
	tests := []struct {
		name        string       // subtest name
		fn          builder.IDFn // the ID function under test
		input       int          // input index to pass to the IDFn
		want        string       // expected output string (if no panic)
		shouldPanic bool         // whether the call should panic
	}{
		// DefaultIDFn: decimal conversion, never panics
		{"DefaultIDFn_zero", builder.DefaultIDFn, 0, "0", false},
		{"DefaultIDFn_multi", builder.DefaultIDFn, 123, "123", false},

		// SymbolIDFn: uppercase letters A–Z, panics out of range
		{"SymbolIDFn_min", builder.SymbolIDFn, 0, "A", false},
		{"SymbolIDFn_max", builder.SymbolIDFn, 25, "Z", false},
		{"SymbolIDFn_neg", builder.SymbolIDFn, -1, "", true},
		{"SymbolIDFn_tooHigh", builder.SymbolIDFn, 26, "", true},

		// AlphanumericIDFn: base-36 encoding, panics on negative
		{"AlphanumericIDFn_zero", builder.AlphanumericIDFn, 0, "0", false},
		{"AlphanumericIDFn_low", builder.AlphanumericIDFn, 10, "a", false},
		{"AlphanumericIDFn_high", builder.AlphanumericIDFn, 35, "z", false},
		{"AlphanumericIDFn_neg", builder.AlphanumericIDFn, -5, "", true},

		// ExcelColumnIDFn: Excel-style columns, panics on negative
		{"ExcelColumnIDFn_zero", builder.ExcelColumnIDFn, 0, "A", false},
		{"ExcelColumnIDFn_endSingle", builder.ExcelColumnIDFn, 25, "Z", false},
		{"ExcelColumnIDFn_startDouble", builder.ExcelColumnIDFn, 26, "AA", false},
		{"ExcelColumnIDFn_ZZ", builder.ExcelColumnIDFn, 701, "ZZ", false},
		{"ExcelColumnIDFn_AAA", builder.ExcelColumnIDFn, 702, "AAA", false},
		{"ExcelColumnIDFn_neg", builder.ExcelColumnIDFn, -1, "", true},

		// HexIDFn: hexadecimal encoding, panics on negative
		{"HexIDFn_zero", builder.HexIDFn, 0, "0", false},
		{"HexIDFn_ten", builder.HexIDFn, 10, "a", false},
		{"HexIDFn_neg", builder.HexIDFn, -2, "", true},
	}

	// iterate over each test case in the table
	var got string
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel() // allow subtests to run in parallel
			if tc.shouldPanic {
				// verify that the function panics for invalid input
				assertPanics(t, func() { tc.fn(tc.input) }, tc.name)
			} else {
				// call the IDFn and compare its output to the expected string
				got = tc.fn(tc.input)
				if got != tc.want {
					t.Errorf("%s: expected %q, got %q", tc.name, tc.want, got)
				}
			}
		})
	}
}
