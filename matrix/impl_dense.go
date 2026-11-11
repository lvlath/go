// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package matrix - Dense storage (row-major) & safe accessors.
//
// Purpose:
//   - Provide a cache-friendly row-major buffer with the explicit index formula i*cols + j.
//   - Guarantee safety at the public surface: At/Set return errors instead of panicking.
//   - Keep algorithmic determinism (fixed loop orders, no map iteration).
//   - Support no-copy views (View) and copy-based submatrix extraction (Induced).
//   - Enforce a numeric policy (optional rejection of NaN/Inf) from a single source of truth.
//
// AI-Hints:
//   - Prefer fast-paths on *Dense in hot algebra (see impl_linear_algebra.go): operate on the flat data slice directly.
//   - Use View(r0,c0,h,w) to avoid copies for windows; mutations reflect in the base matrix.
//   - Use Induced(rows, cols) to materialize a submatrix (copy) for independent lifetime/shape.
//   - DefaultValidateNaNInf is on; insert only finite values unless you explicitly disable upstream.
//
// Complexity quicksheet:
//   - NewDense: O(r*c) zero-init; At/Set: O(1); Clone: O(r*c); View: O(1); Induced: O(r'*c').
package matrix

import (
	"fmt"
	"math"
	"strings"
)

// ---------- error context tags ----------

const (
	ctxAt     = "At"      // method tag used in error wrappers
	ctxSet    = "Set"     // method tag used in error wrappers
	ctxApply  = "Apply"   // method tag used in error wrappers
	ctxFill   = "Fill"    // method tag used in error wrappers
	ctxView   = "View"    // ctor tag for Dense.View
	ctxInduce = "Induced" // ctor/tag for Dense.Induced
)

// ---------- Formatting literals  ----------
const (
	_fmtRowOpen  = "["
	_fmtRowClose = "]\n"
	_fmtSep      = ", "
)

// denseErrorf wraps an error with a uniform Dense context and callsite indices.
// Attach method context and coordinates to a sentinel error for diagnostics.
//
// Implementation:
//   - Stage 1: format "Dense.<method>(row,col): %w".
//   - Stage 2: return wrapped error.
//
// Behavior highlights:
//   - Stable, human-friendly messages; preserves sentinel via %w.
//
// Inputs:
//   - method: context tag (ctxAt/ctxSet/ctxApply/...)
//   - row, col: coordinates
//   - err: sentinel (e.g., ErrOutOfRange, ErrNaNInf)
//
// Returns:
//   - error: wrapped with context
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Keep tags in constants for grep-ability and consistency.
//
// AI-Hints:
//   - Prefer to wrap at the nearest detection site for precise coordinates.
func denseErrorf(method string, row, col int, err error) error {
	return fmt.Errorf("Dense.%s(%d,%d): %w", method, row, col, err)
}

// validateValue validates v against the receiver's numeric policy.
// Implementation:
//   - Stage 1: If validateNaNInf is disabled, accept any value.
//   - Stage 2: Reject NaN and -Inf unconditionally when validation is enabled.
//   - Stage 3: Allow +Inf only when allowInfDistances is enabled.
//
// Behavior highlights:
//   - Single source of truth for all write paths (Set/Apply/View/Fill).
//   - validateNaNInf=false disables this scalar gate.
//   - NaN is never accepted when validation is enabled.
//   - -Inf is never accepted when validation is enabled.
//   - +Inf is accepted only when allowInfDistances is true.
//
// Inputs:
//   - v: candidate value to be stored.
//
// Returns:
//   - error: nil if acceptable; ErrNaNInf otherwise.
//
// Errors:
//   - ErrNaNInf for NaN, -Inf, or forbidden +Inf.
//
// Determinism:
//   - Pure function w.r.t. receiver flags; no side effects.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This function intentionally does NOT wrap errors with coordinates;
//     callers wrap at the nearest site for precise context.
//
// AI-Hints:
//   - Reuse this helper for any new write path to keep policy consistent.
//   - Do not inline this logic inside Set.
//   - Do not replace allowInfDistances with a second field.
func (m *Dense) validateValue(v float64) error {
	// Accept everything when validation is explicitly disabled.
	if !m.validateNaNInf {
		return nil
	}
	// Reject NaN and -Inf deterministically under validation.
	if isNaNOrNegInf(v) {
		return ErrNaNInf
	}
	// Allow +Inf only for distance-policy matrices.
	if math.IsInf(v, 1) && !m.allowInfDistances {
		return ErrNaNInf
	}

	// All other values are acceptable under the policy.
	return nil
}

// IsNil reports whether the receiver is a typed-nil *Dense and therefore must be
// treated as nil when stored inside the Matrix interface.
//
// Implementation:
//   - Stage 1: Compare the receiver pointer to nil.
//   - Stage 2: Return the result without dereferencing.
//
// Behavior highlights:
//   - Safe for typed-nil inside interfaces (no panic).
//   - Reflect-free nil detection used by ValidateNotNil via core.Nilable.
//
// Inputs:
//   - (receiver) *Dense: may be nil.
//
// Returns:
//   - bool: true iff receiver == nil.
//
// Errors:
//   - None.
//
// Determinism:
//   - Deterministic.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - This method must remain trivial; do not add deep validation here.
//
// AI-Hints:
//   - Implement the same pattern on any other pointer-backed Matrix types in this package.
func (m *Dense) IsNil() bool { return m == nil }

// Dense is a contiguous row-major matrix implementation.
// Implementation:
//   - Stage 1: Store shape (r,c) and data buffer (len=r*c).
//   - Stage 2: Enforce numeric policy on Set() (and any other write path).
//
// Behavior highlights:
//   - Deterministic indexing: idx = i*cols + j.
//   - Numeric policy is per-instance (no globals).
//
// Determinism:
//   - Deterministic for fixed inputs.
//
// Notes:
//   - validateNaNInf and allowInfDistances together define what Set() accepts.
//
// AI-Hints:
//   - Use WithAllowInfDistances only for distance-policy matrices (APSP/MetricClosure).
type Dense struct {
	// r (rows) is the number of rows in the matrix (>= 0).
	r int

	// c (columns) is the number of columns in the matrix (>= 0).
	c int

	// data stores row-major values: data[i*cols+j].
	data []float64

	// validateNaNInf rejects NaN and infinities on Set() when true.
	validateNaNInf bool

	// allowInfDistances permits +Inf on Set() when validateNaNInf is true.
	//
	// Contract:
	//   - NaN and -Inf remain rejected under validation.
	//   - +Inf is allowed only for semantic absence sentinels:
	//     distance matrices/APSP use +Inf as "no path";
	//     finite weighted adjacency may use +Inf as "no edge" when 0 is a valid edge weight.
	//
	// AI-Hints:
	//   - Do not use this flag as dirty-data mode.
	//   - Do not treat it as permission for NaN or -Inf.
	allowInfDistances bool
}

// Compile-time assertions for interface & fmt.Stringer conformance.
var (
	_ Matrix       = (*Dense)(nil) // *Dense implements our public Matrix interface
	_ fmt.Stringer = (*Dense)(nil)
)

// NewDense creates an r×c zero matrix using row-major storage.
//   - Public constructor for Dense with non-negative shape validation
//     and the default numeric policy.
//   - Zero-sized matrices (0×N, N×0, 0×0) are legal and represented
//     by a nil backing slice.
//
// Implementation:
//   - Stage 1: validate rows>=0, cols>=0 and checked rows*cols arithmetic.
//   - Stage 2: handle zero-sized matrices as a dedicated fast-path.
//   - Stage 3: allocate a contiguous zero-filled buffer for non-zero shapes.
//
// Behavior highlights:
//   - Never panics on user errors; returns sentinel errors instead.
//   - Zero-sized matrices are safe to pass into all Dense methods;
//     bounds checks will prevent any data access.
//   - Numeric policy (validateNaNInf) defaults from DefaultValidateNaNInf.
//
// Inputs:
//   - rows: number of rows (may be zero).
//   - cols: number of columns (may be zero).
//
// Returns:
//   - *Dense: newly allocated matrix with the requested shape.
//   - error : ErrInvalidDimensions when dimensions are negative or their product overflows int.
//
// Errors:
//   - ErrInvalidDimensions (shape contract violation).
//
// Determinism:
//   - Always allocates the same layout for given (rows, cols).
//   - For zero-sized shapes, data is guaranteed to be nil.
//
// Complexity:
//   - Time O(rows*cols) for non-zero shapes (allocation & zeroing).
//   - Time O(1) for zero-sized matrices.
//
// Notes:
//   - High-level facades (NewZeros, NewIdentity, APSP helpers) are
//     responsible for enforcing rows>0 && cols>0 where needed.
//   - This constructor is suitable for both user code and internal
//     low-level matrix creation.
//
// AI-Hints:
//   - Use NewDense whenever you need a concrete, owning Dense buffer.
//   - Zero-sized matrices are often useful as neutral elements in
//     pipelines; treat them as valid, not as errors.
func NewDense(rows, cols int) (*Dense, error) {
	// A checked element count rejects negative or overflowing shapes before
	// the runtime receives any slice length; zero-sized shapes remain legal.
	elements, err := matrixCheckedElementCount(rows, cols)
	if err != nil {
		return nil, err
	}

	// Zero-sized shapes are legal and represented by a nil buffer.
	if elements == 0 {
		return &Dense{
			r:                 rows,
			c:                 cols,
			data:              nil,                   // no backing storage needed
			validateNaNInf:    DefaultValidateNaNInf, // default numeric policy
			allowInfDistances: DefaultAllowInfDistances,
		}, nil
	}

	// Non-zero shape: allocate a contiguous flat buffer; make() zero-fills it.
	buf := make([]float64, elements)

	return &Dense{
		r:                 rows,
		c:                 cols,
		data:              buf,
		validateNaNInf:    DefaultValidateNaNInf,
		allowInfDistances: DefaultAllowInfDistances,
	}, nil
}

// newDenseWithPolicy constructs a Dense matrix with an explicit NaN/Inf policy.
// Implementation:
//   - Stage 1: Validate dimensions and checked row-major element count.
//   - Stage 2: Fast-path zero-sized shapes (nil buffer).
//   - Stage 3: Allocate row-major buffer.
//
// Behavior highlights:
//   - Hot-path friendly: avoids option parsing for internal builders.
//
// Inputs:
//   - rows, cols: matrix dimensions, must be >= 0.
//   - validateNaNInf: whether Dense.Set should reject NaN/±Inf values.
//   - allowInfDistances: allow +Inf on Set() when validateNaNInf is true.
//
// Returns:
//   - *Dense: allocated matrix.
//
// Errors:
//   - ErrInvalidDimensions: negative shape or product that exceeds MaxInt.
//
// Determinism:
//   - Deterministic allocation.
//
// Complexity:
//   - Time O(r*c), Space O(r*c).
//
// Notes:
//   - Intended for internal builders (Adjacency/Incidence/MetricClosure).
//
// AI-Hints:
//   - Use validateNaNInf=true and allowInfDistances=true for APSP/MetricClosure
//     distance matrices and finite weighted adjacency matrices that encode absence as +Inf.
func newDenseWithPolicy(rows, cols int, validateNaNInf, allowInfDistances bool) (*Dense, error) {
	// Use the identical shape arithmetic as NewDense; this constructor differs
	// only in its per-instance numeric policy, not in its memory safety rules.
	elements, err := matrixCheckedElementCount(rows, cols)
	if err != nil {
		return nil, err
	}

	// Represent zero-sized matrices with a nil backing slice (consistent with NewDense).
	if elements == 0 {
		return &Dense{
			r:                 rows,
			c:                 cols,
			data:              nil,               // explicit nil buffer for determinism
			validateNaNInf:    validateNaNInf,    // store resolved policy
			allowInfDistances: allowInfDistances, // store resolved policy
		}, nil
	}
	// Prepare: allocate row-major storage (runtime zero-fills).
	buf := make([]float64, elements)

	return &Dense{
		r:                 rows,
		c:                 cols,
		data:              buf,
		validateNaNInf:    validateNaNInf,
		allowInfDistances: allowInfDistances,
	}, nil
}

// NewPreparedDense allocates a Dense matrix and applies the resolved numeric policy options.
//
// Implementation:
//   - Stage 1: assemble Options through gatherOptions (no panic, no nil-option calls).
//   - Stage 2: allocate Dense via newDenseWithPolicy using only Dense-relevant fields.
//
// Behavior highlights:
//   - Backwards compatible for callers that pass no options.
//   - Uses the same option validation path as public facades.
//   - Dense observes numeric policy only:
//     validateNaNInf and allowInfDistances.
//   - Graph-adapter options are accepted but do not affect Dense storage layout.
//
// Inputs:
//   - rows, cols: matrix shape; zero-sized shapes are legal.
//   - opts: functional options.
//
// Returns:
//   - *Dense: allocated matrix with attached numeric policy.
//   - error: sentinel-preserving option or shape failure.
//
// Errors:
//   - ErrNilOption from option assembly.
//   - ErrNaNInf / ErrOutOfRange from numeric options.
//   - ErrInvalidDimensions for negative shape.
//
// Determinism:
//   - Stable left-to-right option application through gatherOptions.
//   - Deterministic allocation layout.
//
// Complexity:
//   - Time O(rows*cols), Space O(rows*cols).
//
// Notes:
//   - WithMetricClosure is accepted only insofar as finalizeOptions derives
//     allowInfDistances=true; Dense itself does not become a metric-closure matrix.
//
// AI-Hints:
//   - Do not manually call Option callbacks here.
//   - Do not ignore option errors.
//   - Use this constructor for semantic +Inf sentinels only: APSP/no-path matrices
//     or finite weighted adjacency matrices where +Inf means no edge.
func NewPreparedDense(rows, cols int, opts ...Option) (*Dense, error) {
	cfg, err := gatherOptions(opts...)
	if err != nil {
		return nil, err
	}

	// Finalize: allocate with resolved policy.
	return newDenseWithPolicy(rows, cols, cfg.validateNaNInf, cfg.allowInfDistances)
}

// Rows return the row count.
//
// Zero-shape law:
//   - Zero rows are legal.
//   - Nil receiver reports 0 to keep diagnostics/no-panic paths safe;
//     use ValidateNotNil when nil must be rejected.
//
// Complexity: O(1).
func (m *Dense) Rows() int {
	if m == nil {
		return 0
	}

	return m.r
}

// Cols returns the column count.
//
// Zero-shape law:
//   - Zero columns are legal.
//   - Nil receiver reports 0 to keep diagnostics/no-panic paths safe;
//     use ValidateNotNil when nil must be rejected.
//
// Complexity: O(1).
func (m *Dense) Cols() int {
	if m == nil {
		return 0
	}

	return m.c
}

// Shape packs Rows and Cols into a single call.
//
// Zero-shape law:
//   - Zero columns are legal.
//   - Nil receiver reports 0 and 0 to keep diagnostics/no-panic paths safe;
//     use ValidateNotNil when nil must be rejected.
//
// Complexity: O(1).
func (m *Dense) Shape() (rows, cols int) {
	if m == nil {
		return 0, 0
	}

	return m.r, m.c
}

// indexOf computes the row-major offset or returns ErrOutOfRange.
// Bounds-check (row,col) and compute flat offset for row-major storage.
//
// Implementation:
//   - Stage 1: validate 0 ≤ row < m.r and 0 ≤ col < m.c.
//   - Stage 2: compute row*m.c + col.
//
// Behavior highlights:
//   - Error is wrapped with the caller's method context.
//   - Returns a sentinel (ErrOutOfRange) without adding context; public
//     methods (At/Set) will wrap with coordinates and method name.
//
// Inputs:
//   - method: caller identifier (ctxAt/ctxSet/...)
//   - row, col: coordinates.
//
// Returns:
//   - (offset, nil) on success; (0, ErrOutOfRange) otherwise.
//
// Errors:
//   - ErrOutOfRange when indices are invalid
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Keep unexported to avoid accidental panics at public surface.
//
// AI-Hints:
//   - Reuse in At/Set to keep identical bound semantics.
func (m *Dense) indexOf(row, col int) (int, error) {
	if row < 0 || row >= m.r {
		return 0, ErrOutOfRange
	}
	if col < 0 || col >= m.c {
		return 0, ErrOutOfRange
	}

	// Row-major offset: i*c + j.
	return row*m.c + col, nil
}

// At returns the value at (row, col) or ErrOutOfRange. Safe element read at coordinates.
//
// Implementation:
//   - Stage 1: compute offset via indexOf (bounds check).
//   - Stage 2: load from flat buffer.
//
// Behavior highlights:
//   - Never panics on out-of-range; returns sentinel error.
//
// Inputs:
//   - row, col: zero-based indices.
//
// Returns:
//   - (value, nil) on success; (0, ErrOutOfRange) on invalid indices.
//
// Errors:
//   - ErrOutOfRange when out of bounds
//
// Determinism:
//   - Stable access cost; no allocations.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - Uses direct data[] to avoid double checking.
//
// AI-Hints:
//   - Prefer At in external code; internal hot paths may index directly.
func (m *Dense) At(row, col int) (float64, error) {
	if m == nil {
		return 0, denseErrorf(ctxAt, row, col, ErrNilMatrix)
	}

	off, err := m.indexOf(row, col)
	if err != nil {
		return 0, denseErrorf(ctxAt, row, col, err) // wrap with context
	}

	return m.data[off], nil
}

// Set stores v at (row, col) or returns an error (bounds or numeric policy).
// Safe element write with optional finite-only policy.
//
// Implementation:
//   - Stage 1: Validate bounds.
//   - Stage 2: Enforce numeric policy (NaN/-Inf always rejected under validation; +Inf conditional).
//   - Stage 3: Write into row-major storage.
//
// Behavior highlights:
//   - Strict, local validation; no hidden global toggles.
//   - Never panics; returns sentinel errors.
//   - Numeric policy is a per-instance flag preserved by Clone.
//
// Inputs:
//   - row, col: element coordinates.
//   - v      : value to store.
//
// Returns:
//   - nil on success; errors on invalid indices.
//
// Errors:
//   - ErrOutOfRange for bounds; ErrNaNInf for invalid numbers
//
// Determinism:
//   - Direct flat write; fixed order irrelevant here.
//
// Determinism:
//   - Stable, no side-effects beyond the cell.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - allowInfDistances affects ONLY +Inf acceptance under validation.
//   - Policy flag is carried by Clone/Induced/View (single source of truth).
//   - If you need to store NaN/±Inf for sanitizer tests, disable validation explicitly via WithNoValidateNaNInf().
//
// AI-Hints:
//   - Keep policy ON in production data flows; disable only in controlled ingestion.
//   - If you need to store NaN/±Inf for sanitizer tests, disable validation explicitly via WithNoValidateNaNInf().
func (m *Dense) Set(row, col int, v float64) error {
	// Validate: nil receiver must not panic on the public surface.
	if m == nil {
		return denseErrorf(ctxSet, row, col, ErrNilMatrix)
	}
	off, err := m.indexOf(row, col)
	if err != nil {
		return denseErrorf(ctxSet, row, col, err) // wrap with context
	}

	// Validate: enforce numeric policy via the single-source helper.
	if err = m.validateValue(v); err != nil {
		return denseErrorf(ctxSet, row, col, err) // keep coordinates for diagnostics
	}
	// Execute: direct flat write into row-major storage.
	m.data[off] = v

	return nil
}

// Fill overwrites the entire Dense buffer with the provided row-major slice.
// Enforces the receiver numeric policy (same rules as Set) to keep invariants consistent.
//
// Implementation:
//   - Stage 1: Validate receiver, shape invariants, and input length.
//   - Stage 2: Validate every value against the numeric policy in deterministic order.
//   - Stage 3: Copy into the internal row-major buffer.
//
// Behavior highlights:
//   - Policy-consistent: rejects NaN/-Inf always when validation is enabled; +Inf only if allowInfDistances=true.
//   - Deterministic: validates and copies in stable row-major order.
//   - Zero-sized matrices accept an empty slice and become a no-op.
//
// Inputs:
//   - data: flat row-major slice; len(data) must equal Rows()*Cols().
//
// Returns:
//   - error: nil on success.
//
// Errors:
//   - ErrNilMatrix: if receiver is nil.
//   - ErrInvalidDimensions: if len(data) mismatches shape or receiver invariants are broken.
//   - ErrNaNInf: if a value violates numeric policy.
//
// Determinism:
//   - Fixed k=0..N-1 traversal; stable failure coordinate.
//
// Complexity:
//   - Time O(r*c), Space O(1) extra.
//
// Notes:
//   - Use distance-policy matrices (WithAllowInfDistances) for APSP/unreachable sentinels (+Inf).
//   - Use raw-policy matrices (WithNoValidateNaNInf) for ingestion/sanitizer scenarios.
//
// AI-Hints:
//   - Prefer Fill for bulk deterministic ingestion; prefer Set for sparse updates.
//   - For distance matrices: allocate via NewPreparedDense(..., WithAllowInfDistances()).
//   - If you intentionally need to allow NaN/±Inf for controlled tests,
//     disable validation explicitly upstream (WithNoValidateNaNInf()).
func (m *Dense) Fill(data []float64) error {
	// Validate: nil receiver must not panic on the public surface.
	if m == nil {
		return ErrNilMatrix
	}

	// Validate: Dense shape must never be negative.
	if m.r < 0 || m.c < 0 {
		return ErrInvalidDimensions
	}

	// Prepare: compute the expected total element count.
	total := m.r * m.c

	// Validate: the input length must match the shape exactly.
	if len(data) != total {
		return ErrInvalidDimensions
	}

	// Fast-path: zero-area matrices are legal; no backing storage is required.
	if total == 0 {
		return nil
	}

	// Validate: enforce the exact same numeric policy as Set(), in deterministic order.
	// We map flat index k -> (row, col) for stable diagnostics.
	if m.validateNaNInf {
		var k int
		var row, col int
		for k = 0; k < total; k++ {
			if err := m.validateValue(data[k]); err != nil {
				// Convert linear index to coordinates in a deterministic way.
				row = k / m.c
				col = k - row*m.c
				return denseErrorf(ctxFill, row, col, err)
			}
		}
	}

	// Validate: internal invariant guard to avoid panics on malformed receivers.
	if len(m.data) != total {
		return ErrInvalidDimensions
	}

	// Execute: overwrite the entire buffer deterministically.
	copy(m.data, data)

	// Finalize: success.
	return nil
}

// Clone returns a deep copy (new buffer, same numeric policy).
// Produce an independent Dense with identical shape/data/policy.
//
// Implementation:
//   - Stage 1: allocate new buffer len==r*c.
//   - Stage 2: copy data and flags.
//
// Behavior highlights:
//   - Independence: mutations do not affect the original.
//
// Returns:
//   - Matrix: *Dense implementing Matrix.
//
// Determinism:
//   - Stable double loop cost reduced to single copy.
//
// Complexity:
//   - Time O(r*c), Space O(r*c).
//
// Notes:
//   - Returned dynamic type is *Dense.
//
// AI-Hints:
//   - For structural copy with transform, consider Apply on clone.
func (m *Dense) Clone() Matrix {
	// Validate: nil receiver clone should not panic; it has no meaningful result.
	if m == nil {
		return nil
	}
	// Prepare: preserve the "zero-sized => nil backing slice" invariant for determinism.
	var cp []float64
	if len(m.data) != 0 {
		cp = make([]float64, len(m.data)) // allocate same length
		copy(cp, m.data)                  // deep copy values
	}

	return &Dense{
		r:                 m.r,
		c:                 m.c,
		data:              cp,
		validateNaNInf:    m.validateNaNInf,    // preserve guard policy
		allowInfDistances: m.allowInfDistances, // critical for distance-policy matrices
	}
}

// String HUMAN-READABLE dump of rows for diagnostics.
// Implementation:
//   - Stage 1: iterate rows/cols deterministically.
//   - Stage 2: write values into strings.Builder with standard delimiters.
//
// Behavior highlights:
//   - Not for hot paths; intended for logs and debugging.
//
// Returns:
//   - string: multi-line representation of matrix.
//
// Determinism:
//   - Fixed traversal order.
//
// Complexity:
//   - Time O(r*c), Space O(r*c) for formatting.
//
// AI-Hints:
//   - For large matrices prefer printing a few rows/cols or summarize.
func (m *Dense) String() string {
	// Guard against nil receiver to avoid accidental panics in diagnostics paths.
	if m == nil {
		return "<nil Dense>\n"
	}

	var b strings.Builder
	var i, j, base int
	for i = 0; i < m.r; i++ { // iterate rows deterministically
		b.WriteString(_fmtRowOpen) // open row
		base = i * m.c
		for j = 0; j < m.c; j++ { // iterate cols
			b.WriteString(fmt.Sprintf("%g", m.data[base+j]))
			if j+1 < m.c {
				b.WriteString(_fmtSep) //separate values with comma + space
			}
		}
		b.WriteString(_fmtRowClose) // close row
	}

	return b.String()
}

// Induced materializes a copy submatrix using explicit index sets.
// Copy rows/cols at the given index lists (duplicates allowed).
//
// Implementation:
//   - Stage 1: handle zero-sized result (legal).
//   - Stage 2: allocate result with the same Dense numeric policy.
//   - Stage 3: nested loops with direct offset math; bounds-check each index.
//
// Behavior highlights:
//   - Numeric policy is preserved from the base (validateNaNInf and allowInfDistances).
//   - Duplicates in index sets are allowed (repeated rows/cols in the result).
//
// Inputs:
//   - rowsIdx: indices into [0..m.r).
//   - colsIdx: indices into [0..m.c).
//
// Returns:
//   - *Dense: independent copy with size len(rowsIdx)×len(colsIdx).
//
// Errors:
//   - ErrOutOfRange (index outside bounds).
//
// Determinism:
//   - Fixed nested loops i→j.
//
// Complexity:
//   - Time O(rp*cp), Space O(rp*cp).
//
// Notes:
//   - Zero-area returns legal Dense with zero-length buffer.
//
// AI-Hints:
//   - Use when the result must be independent (e.g., transform downstream).
func (m *Dense) Induced(rowsIdx, colsIdx []int) (*Dense, error) {
	// Validate: nil receiver must not panic; report a sentinel instead.
	if m == nil {
		return nil, ErrNilMatrix
	}
	rp := len(rowsIdx) // result rows
	cp := len(colsIdx) // result cols
	// Zero-area: legal Dense, shared policy
	if rp == 0 || cp == 0 {
		return &Dense{
			r:                 rp,
			c:                 cp,
			data:              nil, // deterministic nil buffer for zero-sized shape
			validateNaNInf:    m.validateNaNInf,
			allowInfDistances: m.allowInfDistances,
		}, nil
	}

	// Allocate the result with the same numeric policy as the base matrix.
	res, err := newDenseWithPolicy(rp, cp, m.validateNaNInf, m.allowInfDistances)
	if err != nil {
		return nil, err
	}

	// Deterministic double loop; direct offset math in both matrices.
	var i, j int
	var ri, cj int
	var src, dst int
	for i = 0; i < rp; i++ {
		ri = rowsIdx[i]
		if ri < 0 || ri >= m.r {
			return nil, fmt.Errorf("Dense.%s: row index %d: %w", ctxInduce, ri, ErrOutOfRange)
		}
		for j = 0; j < cp; j++ {
			cj = colsIdx[j]
			if cj < 0 || cj >= m.c {
				return nil, fmt.Errorf("Dense.%s: col index %d: %w", ctxInduce, cj, ErrOutOfRange)
			}
			// Direct linear index in source and destination.
			src = ri*m.c + cj // source offset in base
			dst = i*cp + j    // destination offset in result
			res.data[dst] = m.data[src]
		}
	}

	return res, nil
}

// Do visits each element (i,j) in row-major order and calls f(i,j,v).
// Read-only visitor; stops early when f returns false.
//
// Implementation:
//   - Stage 1: validate receiver and callback; bail out on nil.
//   - Stage 2: nested loops over rows and columns in row-major order.
//   - Stage 3: invoke f for each element and terminate early when it
//     returns false.
//
// Behavior highlights:
//   - Read-only with respect to the callback; no allocations.
//   - Deterministic traversal order: i from 0..Rows()-1, j from 0..Cols()-1.
//   - Safe for zero-sized matrices (callback is never invoked).
//
// Inputs:
//   - f: callback that receives coordinates and value; returns false
//     to signal early termination.
//
// Determinism:
//   - Fixed i→j iteration order for all shapes.
//
// Complexity:
//   - Time O(r*c) in the worst case (when f always returns true).
//   - Space O(1).
//
// Notes:
//   - For mutation, use Apply instead; Do is intended for aggregation,
//     statistics, and diagnostics.
//
// AI-Hints:
//   - Use Do to implement custom reductions (sums, norms, scans) without
//     allocating intermediate slices or matrices.
func (m *Dense) Do(f func(i, j int, v float64) bool) {
	// Guard against nil receiver or callback to avoid accidental panics.
	if m == nil || f == nil {
		return
	}

	var i, j, base int // predeclare loop counters and base offset
	var v float64      // temporary for current value

	for i = 0; i < m.r; i++ { // iterate rows deterministically
		base = i * m.c            // compute flat base offset for row i
		for j = 0; j < m.c; j++ { // iterate columns
			v = m.data[base+j] // read current element
			if !f(i, j, v) {   // invoke callback; stop if it returns false
				return // early exit requested by caller
			}
		}
	}
}

// Apply replaces each element with f(i,j,v) in-place.
// In-place map with policy enforcement and deterministic order.
//
// Implementation:
//   - Stage 1: nested loops - double for-loop over rows then cols; compute new value via f.
//   - Stage 2: compute new value; reject NaN/Inf if policy enabled.
//   - Stage 3: write back.
//
// Behavior highlights:
//   - Deterministic row-major order; no extra allocations.
//   - Respects validateNaNInf (rejects NaN/±Inf when enabled).
//   - Early error aborts; elements written before the error remain updated.
//
// Inputs:
//   - f: transformer from (i,j,v) to new value.
//
// Returns:
//   - error: ErrNaNInf when transformer produced non-finite (if policy ON).
//
// Determinism:
//   - Fixed i→j order; side effects are predictable.
//
// Complexity:
//   - Time O(r*c), Space O(1).
//
// Notes:
//   - For all-or-nothing semantics, transform into a clone and swap on success.
//
// AI-Hints:
//   - Keep transforms pure; avoid capturing external mutable state.
func (m *Dense) Apply(f func(i, j int, v float64) float64) error {
	// Validate: nil receiver must not panic on the public surface.
	if m == nil {
		return ErrNilMatrix
	}
	if f == nil {
		return ErrNilCallback
	}
	var i, j, base int // predeclare loop counters and base offset
	var v, nv float64  // old and new values

	for i = 0; i < m.r; i++ { // iterate rows
		base = i * m.c            // base offset for row i
		for j = 0; j < m.c; j++ { // iterate columns
			v = m.data[base+j] // read current value
			nv = f(i, j, v)    // compute new value
			// Validate: enforce numeric policy via the single-source helper.
			if err := m.validateValue(nv); err != nil {
				return denseErrorf(ctxApply, i, j, err) // wrap with coordinates
			}
			m.data[base+j] = nv // write back new value
		}
	}

	return nil // success
}

// View creates a no-copy window [r0:r0+rows, c0:c0+cols) over the same storage.
// Lightweight submatrix referencing the base buffer (shared storage).
//
// Implementation:
//   - Stage 1: validate window bounds; allow zero-area.
//   - Stage 2: return View with offsets.
//
// Behavior highlights:
//   - Writes via view reflect in base; policy is inherited.
//
// Inputs:
//   - r0,c0: top-left offsets; rows, cols: window size (≥0).
//
// Returns:
//   - *View or error.
//
// Errors:
//   - ErrBadShape when the window is invalid.
//
// Determinism:
//   - Constant-time creation; fixed access order in methods.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - View does not implement Matrix on purpose to avoid accidental copies in ops.
//
// AI-Hints:
//   - Use for sliding-window ops; copy only when lifetime must be independent.
func (m *Dense) View(r0, c0, rows, cols int) (*View, error) {
	// Validate: nil receiver must not panic; View cannot exist without a base.
	if m == nil {
		return nil, fmt.Errorf("Dense.%s(%d,%d,%d,%d): %w", ctxView, r0, c0, rows, cols, ErrNilMatrix)
	}

	// Dimensions of the window must be non-negative.
	if rows < 0 || cols < 0 {
		return nil, fmt.Errorf("Dense.%s(%d,%d,%d,%d): %w", ctxView, r0, c0, rows, cols, ErrInvalidDimensions)
	}

	// Offsets are indices; negative offsets are out-of-range.
	if r0 < 0 || c0 < 0 {
		return nil, fmt.Errorf("Dense.%s(%d,%d,%d,%d): %w", ctxView, r0, c0, rows, cols, ErrOutOfRange)
	}

	// Offsets beyond the base bounds are out-of-range.
	// NOTE: r0==m.r is allowed only when rows==0 (zero-height view), similarly for c0.
	if r0 > m.r || c0 > m.c {
		return nil, fmt.Errorf("Dense.%s(%d,%d,%d,%d): %w", ctxView, r0, c0, rows, cols, ErrOutOfRange)
	}

	// Window exceeding base bounds is out-of-range.
	// Use subtraction form to avoid integer overflow on r0+rows / c0+cols.
	if rows > m.r-r0 || cols > m.c-c0 {
		return nil, fmt.Errorf("Dense.%s(%d,%d,%d,%d): %w", ctxView, r0, c0, rows, cols, ErrOutOfRange)
	}
	return &View{
		base: m,    // share storage
		r0:   r0,   // top row in base
		c0:   c0,   // left col in base
		r:    rows, // view height
		c:    cols, // view width
	}, nil
}

// View is a non-owning window into a Dense (shared storage).
// Not implementing Matrix interface to avoid accidental copies in ops.
type View struct {
	base *Dense // underlying storage owner
	r0   int    // top-left row offset in base
	c0   int    // top-left col offset in base
	r    int    // view height
	c    int    // view width
}

// Rows return the number of rows in the view.
// Complexity: O(1).
func (v *View) Rows() int {
	if v == nil {
		return 0
	}

	return v.r
}

// Cols return the number of columns in the view.
// Complexity: O(1).
func (v *View) Cols() int {
	if v == nil {
		return 0
	}

	return v.c
}

// At reads element (i,j) in the view or returns ErrOutOfRange.
// Safe read within the view bounds; translates to base coordinates.
//
// Implementation:
//   - Stage 1: check 0≤i<r and 0≤j<c.
//   - Stage 2: return base.data[(r0+i)*base.c + (c0+j)].
//
// Behavior highlights:
//   - Never panics; returns sentinel on violation.
//
// Complexity:
//   - Time O(1), Space O(1).
func (v *View) At(i, j int) (float64, error) {
	// Validate: nil view or nil base must not panic.
	if v == nil || v.base == nil {
		return 0, fmt.Errorf("View.At(%d,%d): %w", i, j, ErrNilMatrix)
	}
	if i < 0 || i >= v.r || j < 0 || j >= v.c {
		return 0, fmt.Errorf("View.At(%d,%d): %w", i, j, ErrOutOfRange)
	}

	// Translate to base coordinates and load directly from the flat buffer.
	return v.base.data[(v.r0+i)*v.base.c+(v.c0+j)], nil
}

// Set writes (i,j) in the view and forwards numeric-policy enforcement to the base Dense.
//
// Implementation:
//   - Stage 1: Validate receiver/base and bounds in view coordinates.
//   - Stage 2: Validate value via base numeric policy.
//   - Stage 3: Translate coordinates and write-through into base storage.
//
// Behavior highlights:
//   - Deterministic: fixed checks and a single write.
//   - Policy-consistent: uses base.validateValue exactly once.
//
// Inputs:
//   - i, j: view-local coordinates.
//   - val : value to store.
//
// Returns:
//   - error: nil on success.
//
// Errors:
//   - ErrNilMatrix: if view or base is nil.
//   - ErrOutOfRange: if (i,j) is outside the view.
//   - ErrNaNInf: if value violates base numeric policy.
//
// Determinism:
//   - Stable check order; stable error cause.
//
// Complexity:
//   - Time O(1), Space O(1).
//
// Notes:
//   - View shares storage with base; mutations are visible in the base Dense.
//
// AI-Hints:
//   - Allocate distance-policy bases when the view must allow +Inf.
func (v *View) Set(i, j int, val float64) error {
	// Validate: nil view or nil base must not panic.
	if v == nil || v.base == nil {
		return fmt.Errorf("View.Set(%d,%d): %w", i, j, ErrNilMatrix)
	}

	// Validate: enforce view bounds first (more intuitive error for callers).
	if i < 0 || i >= v.r || j < 0 || j >= v.c {
		return fmt.Errorf("View.Set(%d,%d): %w", i, j, ErrOutOfRange)
	}

	// Validate: enforce base numeric policy exactly once.
	if err := v.base.validateValue(val); err != nil {
		return fmt.Errorf("View.Set(%d,%d): %w", i, j, err)
	}

	// Execute: translate to base coordinates and write-through.
	v.base.data[(v.r0+i)*v.base.c+(v.c0+j)] = val

	// Finalize: success.
	return nil
}

// Clone creates an independent owning Dense copy of only the view's window.
// Unlike the original View, the returned Matrix does not share storage with
// the parent Dense or any sibling views and preserves the parent's numeric policy.
//
// Implementation:
//   - Stage 1: reject nil/unbound view receivers without dereferencing the base.
//   - Stage 2: allocate exactly r*c elements and copy each view row with its stride.
//   - Stage 3: publish a Dense with the original window shape and numeric policy.
//
// Behavior highlights:
//   - A view's original coordinates are not carried into the clone: (0,0) is its corner.
//   - An empty but valid view produces a non-nil, zero-element Dense of the same shape.
//   - Inf/NaN validation and distance-sentinel settings are preserved from the base.
//
// Inputs:
//   - v: a view constructed by Dense.View; its base must not be concurrently mutated.
//
// Returns:
//   - Matrix: a freshly allocated *Dense with no backing-array alias, or nil for
//     a nil/unbound view receiver.
//
// Errors:
//   - None: Matrix.Clone has no error return; Dense.View already validates ranges.
//
// Determinism:
//   - Rows are copied ascending, with columns ascending within each row.
//
// Complexity:
//   - Time O(r*c), Space O(r*c) for an r-by-c view, not O(parent.rows*parent.cols).
//
// Notes:
//   - View itself remains a write-through alias; only Clone severs ownership.
//   - The returned interface uses the same numeric policy as Dense.Clone.
//
// AI-Hints:
//   - Do not implement Clone as `return v.base.Clone()`: that copies the wrong shape.
//   - Do not return v or share its base.data: QR and other callers may mutate clones.
func (v *View) Clone() Matrix {
	if v == nil || v.base == nil {
		return nil
	}

	var values []float64
	if v.r != 0 && v.c != 0 {
		values = make([]float64, v.r*v.c)
		for row := 0; row < v.r; row++ {
			// The parent is row-major but view rows may be strided by parent width.
			sourceStart := (v.r0+row)*v.base.c + v.c0
			destinationStart := row * v.c
			copy(values[destinationStart:destinationStart+v.c],
				v.base.data[sourceStart:sourceStart+v.c])
		}
	}
	return &Dense{
		r:                 v.r,
		c:                 v.c,
		data:              values,
		validateNaNInf:    v.base.validateNaNInf,
		allowInfDistances: v.base.allowInfDistances,
	}
}

// Compile-time check ensures View implements every Matrix method, including Clone.
var _ Matrix = (*View)(nil)
