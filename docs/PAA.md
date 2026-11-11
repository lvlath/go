<!--
SPDX-License-Identifier: AGPL-3.0-only
Copyright (C) 2025-2026 katalvlaran
File: docs/PAA.md
Package: github.com/lvlath/go/paa
Purpose: Exact fractional geometry, numeric policy, integration and verification.
Contract status: implemented Transform, TransformRange, and Means; no proposed API described as shipped.
Scope: scalar uniformly spaced time series, not irregular timestamps.
License: AGPL-3.0-only
-->

# PAA — fractional Piecewise Aggregate Approximation

## What / Why

`paa.Transform` compresses a finite, uniformly sampled time series of length `n` to exactly `w` equal-duration segments while retaining the full domain `[0,n)`. An observation `values[i]` occupies `[i,i+1)`. The output is a detached `*paa.Result` containing `Values`, exact rational `Segments`, and `OriginalLength`.

Unlike naive integer-window PAA, no remainder is dropped and no missing sample is padded with zero. The package is a stand-alone scalar primitive suitable for later SAX preprocessing and segment-based search. It does **not** compute symbolic distances, normalize series, or implement DTW.

## Exact mathematics

Define `f(t)=values[i]` on `[i,i+1)`. For `0<=j<w`, output segment `I_j=[j*n/w,(j+1)*n/w)`. Express both time grids in integer units of `1/w`:

- Original cell `i`: `[i*w,(i+1)*w)`.
- Output cell `j`: `[j*n,(j+1)*n)`.
- `overlap(i,j)=max(0,min((i+1)*w,(j+1)*n)-max(i*w,j*n))`.

Then the exact geometric average is

$$ \overline{x}_j=\sum_i x_i\frac{\operatorname{overlap}(i,j)}{n}. $$

The weights of contributing samples are nonnegative and add to one; therefore each mean is in the closed convex hull of its inputs. Since each sample cell is fully covered by the output cells, the weighted-integral law is

$$ \frac{n}{w}\sum_{j=0}^{w-1}\overline{x}_j = \sum_{i=0}^{n-1}x_i. $$

These are real-number identities. In `float64`, comparisons require scale-aware tolerances, not bitwise equality. ![Fractional PAA mean, CodeCogs](https://latex.codecogs.com/svg.image?%5Cbar%7Bx%7D_j%3D%5Csum_i%20x_i%5Cfrac%7B%5Cell_%7Bij%7D%7D%7Bn%7D)

### Fractional fixture

For `[0,2,4,6,8]` and `w=2`, the output is `[1.6,6.4]` and the boundaries are `[0/2,5/2), [5/2,10/2)`. The middle source cell `[2,3)` contributes equally to both output segments. The unsimplified denominator remains `2` even when a fraction could be simplified.

```text
sample:  |---0---|---2---|---4---|---6---|---8---|
PAA:     |--------1.6--------|--------6.4--------|
                             ^
                     t = 2.5 exact split
         [0/2, 5/2)           [5/2, 10/2)
```

## Algorithm and complexity

```text
validate w, nil, empty, w<=n, finite samples, checked n*w
check representable output allocation sizes; allocate Values[w], Segments[w]
if w == n:
    copy all original float64 values bit-for-bit into detached Values
    construct exact rational Segments; return Result
i=0, j=0, initialize fixed-width exact signed accumulator
while j < w:
    sample = [i*w,(i+1)*w)
    segment = [j*n,(j+1)*n)
    overlap = min(rights)-max(lefts)  // exact positive int64
    add exact binary mantissa(values[i])*overlap into signed integer bins
    if sample.right >= segment.right:
        divide exact signed numerator by n, round to nearest-even float64, publish Segment
        clear accumulator; j++
    if sample.right <= segment.right: i++
return Result
```

All scan iterations advance at least one pointer and there are at most `n+w-gcd(n,w)` nonzero intersections (no more than `n+w-1`). Time is `O(n+w)` including input validation, memory `O(w)` including result slices. No `n*w` matrix, sort, reflection, or `panic/recover` is involved.

The kernel uses a fixed 2304-bit unsigned **superaccumulator for each sign**, interpreting each float64 value as an exact binary rational and recording `mantissa*integerOverlap` in units of `2^-1074`. This is stronger than Kahan/Neumaier compensation: extreme cancellation cannot erase arbitrarily small residues. After exact signed subtraction and division by `n`, the result is rounded **once** to binary64, nearest with ties to even. This includes the fractional division remainder when the integer quotient already has exactly 53 significant bits: otherwise a result just above the subnormal/normal boundary can round in the wrong direction. The hull is checked rather than cosmetically clamped.

The fixed-width proof is concrete: for finite binary64, the magnitude in `2^-1074` units is below `2^2098`. Within one segment, the sum of the integer overlap weights is exactly `n`, and accepted `n` is at most `MaxInt64`. Thus each unsigned sign accumulator is below `2^2161`, well inside 2304 bits. No heap-allocated big integers or rational arithmetic appear in production. `math/big.Rat` is reserved for independent tests.

The identity partition `w=n` instead performs a detached bitwise copy, preserving signed zero (`-0`) without changing its numeric meaning. This fast path is still linear and is applied only after complete validation.

## Public API and failure laws

```go
package paa

type Segment struct {
    LeftNumerator, RightNumerator, Denominator int64
}
type Result struct {
    Values []float64
    Segments []Segment
    OriginalLength int
}
func Transform(values []float64, segments int) (*Result, error)
func TransformRange(values []float64, start, end, segments int) (*Result, error)
func Means(values []float64, segments int) ([]float64, error)
```

`Transform` validation precedence is exact: `segments<=0` => `ErrInvalidOption`; nil => `ErrNilInput`; non-nil empty => `ErrEmptyInput`; `segments>n` => `ErrInvalidOption`; any NaN/Inf => `ErrNonFinite`; integer geometry or checked output-allocation-size overflow => `ErrOverflow`; otherwise compute. Its successfully computed segment means are always finite and inside the hull. Defensive numeric overflow is still classified as `ErrOverflow`. On every error, the result is nil. Callers classify errors with `errors.Is`. `segments` is required and has no default.

### Checked range facade: `TransformRange`

`TransformRange(values, start, end, w)` computes exactly `Transform(values[start:end], w)` when its half-open range is valid, but checks the bounds *before slicing* to prevent caller-induced index panics. Its success result has `OriginalLength=end-start`; every `Segment` is measured **relative to the selected window**, never in the original array's global coordinates. It only reads and validates selected observations. An invalid NaN or infinity **outside** the selected range is irrelevant to that call.

The `TransformRange` precedence is: nonpositive `w` => `ErrInvalidOption`, before input presence; nil => `ErrNilInput`; empty original input => `ErrEmptyInput`; malformed bounds (`start<0`, `start>=end`, `end>len(values)`) or `w>end-start` => `ErrInvalidOption`; nonfinite **window** value => `ErrNonFinite`; checked geometry/allocation/arithmetic => `ErrOverflow`. Failure always yields nil `*Result`.

To convert a local fraction `k/w` to an absolute coordinate, add `start` in the same rational grid: `(start*w+k)/w`. Applications performing this conversion on huge indices must check the additional multiplication and addition for overflow; PAA intentionally does not publish absolute timestamps.

### Value-only projection: `Means`

`Means(values, w)` returns the same detached `Values` as `Transform` and preserves every validation rule, error identity, and rounding decision. It deliberately omits `Result.Segments` and `Result.OriginalLength`: choose it only when the consumer does not need exact sample-time provenance. It delegates to `Transform`; therefore the temporary `O(w)` geometry slice is still allocated before projection. No independent value-only kernel or new numerical mode is introduced.

Neither wrapper accepts optional `Options`: there is presently no independently meaningful policy that would justify a second algorithm or hidden default.

Integer geometry uses `int64` and checks `n*w<=MaxInt64` **before** multiplication. It also checks that the required `[]Segment` storage (`24*w` bytes) fits a platform `int` before allocation. No arbitrary numeric sample-count budget is introduced beyond these representability limits. System OOM is not promised to be recoverable, and no `panic`/`recover` policy is used.

`Values` and `Segments` are independently allocated and owned by the caller; no input is retained. `w=n` produces a detached, bit-preserving identity, including negative zero. Caller mutation of an input concurrently with a call is unsupported. Independent calls are safe and deterministic. For an empty or invalid request, all public methods return nil outputs and do not expose partially filled storage.

## Complete Go example

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/paa"
)

func main() {
	result, err := paa.Transform([]float64{0, 2, 4, 6, 8}, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("means: %.1f %.1f\n", result.Values[0], result.Values[1])
	fmt.Printf("first segment: %+v\n", result.Segments[0])
}
```

See `paa/example_test.go` for seven executable operational examples: rack cooling-temperature triage, end-of-shift latency degradation, energy-integral accounting, extreme-range numeric safety, detached audit snapshots, local shift windows, and aligned vibration-signature comparison. Unlike a peak detector, PAA publishes **averages** and does not guarantee that every instantaneous threshold crossing survives compression.

Existing runnable examples include `examples/paa/main.go` (`go run ./examples/paa`) and `examples/paa_window/main.go` (`go run ./examples/paa_window`). The additional self-contained `examples/paa_vibration_screen/main.go` (`go run ./examples/paa_vibration_screen`) separates a PAA-based sustained-trend review from a raw maximum-based safety alarm, preserving both the exact fractional source intervals and the original observations.

### Window and value-only example

```go
package main

import (
	"fmt"

	"github.com/lvlath/go/paa"
)

func main() {
	series := []float64{99, 0, 2, 4, 6, 8, 99}
	result, err := paa.TransformRange(series, 1, 6, 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	means, err := paa.Means(series[1:6], 2)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("local [%d/%d,%d/%d), means %.1f %.1f\n",
		result.Segments[0].LeftNumerator, result.Segments[0].Denominator,
		result.Segments[0].RightNumerator, result.Segments[0].Denominator,
		means[0], means[1])
	// Output: local [0/2,5/2), means 1.6 6.4
}
```

For an operational windowing example with output checks, run `go run ./examples/paa_window`.

## SAX, series, matrix and DTW boundaries

- **SAX:** should consume `Result.Values`, `Result.Segments`, and `Result.OriginalLength` directly while maintaining the same `n`, `w`, normalization and breakpoint policy. `Means` discards the segment coordinates and cannot by itself certify a SAX/MINDIST geometry. For two length-`n` series with a shared equal segmentation, Jensen gives `sqrt(n/w * sum_j (PAA(Q)_j-PAA(C)_j)^2) <= ||Q-C||_2` in exact arithmetic. Integer rounding of boundaries would invalidate that law. These inequalities are proved in exact arithmetic: ordinary floating-point evaluation of the scaled projected distance is **not automatically a conservative pruning certificate** near a cutoff. Strict threshold pruning needs a separately proved outward-rounding or interval-bound policy. SAX MINDIST requires its own proved quantization bound.
- **series:** a caller may supply one contiguous, uniformly sampled window from `series.WindowRanges`, using either `Transform(values[r.Start:r.End], w)` or checked `TransformRange(values, r.Start, r.End, w)`. In both cases the returned coordinates are local to that window; PAA does not infer or repair irregular timestamps or consume `Valid` masks implicitly.
- **matrix:** `Result.Values` (or the value-only `Means` output) may be explicitly copied into a matrix row by the caller, subject to `matrix`'s existing finite-value and ownership policies; no matrix dependency is introduced into the PAA kernel.
- **DTW:** no generic `LB_PAA <= DTW` or `LB_Keogh` claim follows automatically from this package. Path constraints and the precise proposed bound need their own proof and tests.

## Pitfalls and verification

Do not truncate `n/w`, pad zeros, round the rational Segment fields, use `float64` to decide geometry, multiply an extreme sample by an unnormalized integer overlap, or accept NaN/Inf. Do not treat `TransformRange` local coordinates as absolute parent-series coordinates, nor reconstruct a SAX lower bound from `Means` alone. Do not confuse exact segment coordinates with naive float64 averaging: this kernel also computes correctly-rounded binary64 segment means from the exact binary input values. For extreme-valued conservation tests, use rational sums to avoid overflowing the **test itself**.

External-package tests use an independent `math/big.Rat` oracle that enumerates all overlaps for small `n,w` (not the production sweep), and cover fractional cases, hull, conservation, exact rational Jensen inequalities on aligned series, fractional constant MaxFloat64, retained tail spikes, negative/positive extremes, bitwise nearest-even rounding at subnormal/normal transitions, detached output (including signed-zero identity), error precedence, range equivalence and parallel callers. Benchmarks cover `(n,w)=(1000,1),(1000,64),(1000,1000),(10^6,1),(10^6,64),(10^6,10^6)` plus range selection and value-only projection; reports include `ns/op`, `B/op`, and `allocs/op` without measuring fixture construction. `n*w` overflow under valid `w<=n` cannot be tested with a realistically allocated Go slice on usual CI machines; do not fabricate invalid slice headers.
