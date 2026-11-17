// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package mfi defines deterministic, detached checkpoints for rolling MFI state.
// This file validates version, topology and numeric consistency before any mutation.
package mfi

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"strconv"
)

const (
	// snapshotAlgorithm fixes the recurrence and the public binary state layout.
	// Any incompatible change to either requires a new algorithm identifier.
	snapshotAlgorithm = "mfi/rolling-neumaier/v1"

	// These indices constitute the versioned FloatState and IntState wire format.
	snapshotFloatHeader = 5
	snapshotIntLength   = 3
	snapshotPrevious    = 0
	snapshotPositive    = 1
	snapshotNegative    = 2
	snapshotCorrPos     = 3
	snapshotCorrNeg     = 4
	snapshotUsed        = 0
	snapshotHead        = 1
	snapshotExtraRebase = 2
)

// snapshotWord encodes a nonnegative native ring coordinate as an unsigned ABI word.
// The caller supplies a validated period, occupancy or head index; reconstructing
// the value from its set bits gives an identical wire value on 32- and 64-bit Go.
//
// Implementation:
//   - Stage 1: start with an empty unsigned word and the least-significant bit.
//   - Stage 2: copy each set bit of the native nonnegative integer into that word.
//   - Stage 3: return the exact unsigned magnitude without signed narrowing.
//
// Behavior highlights:
//   - All actual callers provide values within [0, MaxPeriod], so at most 21 bits are read.
//
// Inputs:
//   - value: nonnegative, validated native ring coordinate or effective period.
//
// Returns:
//   - uint64: exact nonnegative integer value in the snapshot's fixed-width domain.
//
// Determinism:
//   - Bit weights are visited from least significant to most significant.
//
// Complexity:
//   - Time O(log(value+1)), bounded by 21 steps; Space O(1), no allocations.
//
// Notes:
//   - Call only with validated nonnegative values; this is an internal ABI encoder.
//
// AI-Hints:
//   - Preserve the numeric value and LE64 configuration encoding across architectures.
func snapshotWord(value int) uint64 {
	var word uint64
	weight := uint64(1)
	for value > 0 {
		if value&1 != 0 {
			word |= weight
		}
		value >>= 1
		weight <<= 1
	}
	return word
}

// configHash identifies one exact MFI algorithm and window configuration.
// The SHA-256 digest is a compatibility fingerprint, not message authentication.
//
// Implementation:
//   - Stage 1: encode the validated effective period as an eight-byte little-endian word.
//   - Stage 2: hash fixed ASCII algorithm identity followed by the period word.
//   - Stage 3: publish the lowercase hex digest.
//
// Behavior highlights:
//   - The digest does not include mutable ring or Count state.
//
// Inputs:
//   - period: already validated positive period bounded by MaxPeriod.
//
// Returns:
//   - string: lowercase SHA-256 hex of ASCII algorithm ID followed by LE64(period).
//
// Determinism:
//   - The byte sequence and digest encoding are fixed by snapshot ABI v1.
//
// Complexity:
//   - Time O(1), Space O(1) excluding a short digest-string allocation.
//
// Notes:
//   - External authenticity requires a separate MAC/signature over the entire checkpoint.
//
// AI-Hints:
//   - Never accept this compatibility hash as evidence of unmodified Snapshot payload.
func configHash(period int) string {
	var suffix [8]byte
	binary.LittleEndian.PutUint64(suffix[:], snapshotWord(period))
	hash := sha256.New()
	_, _ = hash.Write([]byte(snapshotAlgorithm))
	_, _ = hash.Write(suffix[:])
	return hex.EncodeToString(hash.Sum(nil))
}

// Snapshot exports every component needed for exact deterministic continuation.
// Physical ring order and Neumaier corrections are preserved rather than rebuilt,
// so a valid restored engine executes the same future transitions as the source.
//
// Implementation:
//   - Stage 1: return the zero Snapshot for nil or uninitialized receivers.
//   - Stage 2: copy the numeric prefix and both rings into a new float slice.
//   - Stage 3: publish a distinct integer slice and the versioned configuration hash.
//
// Behavior highlights:
//   - The result is detached even when rings have wrapped and been rebased.
//   - Config identifies compatibility; it cannot authenticate caller-provided state.
//
// Inputs:
//   - s: nil or uninitialized receiver returns zero Snapshot without dereference.
//
// Returns:
//   - Snapshot: detached scalar metadata and arrays, or zero Snapshot for invalid receiver.
//
// Determinism:
//   - State fields and ring slots are copied in the fixed v1 order.
//
// Complexity:
//   - Time O(Period), Space O(Period) newly allocated snapshot storage.
//
// Notes:
//   - Concurrency with Push, Reset or Restore on the same engine is unsupported.
//   - Protect externally stored state with a MAC/signature over all serialized fields.
//
// AI-Hints:
//   - Do not discard compensation or rotate the rings when creating a checkpoint.
func (s *Stream) Snapshot() Snapshot {
	if !s.ready() {
		return Snapshot{}
	}

	// Preserve the actual floating-point recurrence, not merely its final ratio.
	floats := make([]float64, snapshotFloatHeader+2*s.period)
	floats[snapshotPrevious] = s.previous
	floats[snapshotPositive] = s.positive.sum
	floats[snapshotNegative] = s.negative.sum
	floats[snapshotCorrPos] = s.positive.correction
	floats[snapshotCorrNeg] = s.negative.correction
	copy(floats[snapshotFloatHeader:snapshotFloatHeader+s.period], s.pos)
	copy(floats[snapshotFloatHeader+s.period:], s.neg)

	var extra uint64
	if s.extraRebaseUsed {
		extra = 1
	}
	return Snapshot{
		Algorithm:  snapshotAlgorithm,
		Config:     configHash(s.period),
		Count:      s.count,
		FloatState: floats,
		IntState:   []uint64{snapshotWord(s.used), snapshotWord(s.head), extra},
	}
}

// checkedSnapshotIndex decodes a snapshot ring coordinate into the native int domain.
// Snapshot slots are uint64 for a stable wire format, while Stream indexes use int;
// parsing after the period ceiling check keeps conversion safe on 32-bit targets.
//
// Implementation:
//   - Stage 1: reject every coordinate outside the public ring memory ceiling.
//   - Stage 2: convert the bounded decimal encoding with native-int overflow checks.
//   - Stage 3: return the usable index or the stable checkpoint error sentinel.
//
// Behavior highlights:
//   - Corrupt larger-than-period fields cannot alias valid slots by modulo wrapping.
//
// Inputs:
//   - value: untrusted v1 wire coordinate, never an unchecked slice index.
//
// Returns:
//   - int: exact native index in [0, MaxPeriod] on success.
//   - error: ErrStateVersion for an out-of-range or unrepresentable coordinate.
//
// Errors:
//   - ErrStateVersion: input is above the public resource ceiling or fails native-int decoding.
//
// Determinism:
//   - Base-ten encoding and strconv.Atoi have no platform-dependent rounding.
//
// Complexity:
//   - Time O(log(MaxPeriod)), Space O(log(MaxPeriod)) temporary conversion data;
//     this helper runs only during Restore, not in the streaming hot path.
//
// Notes:
//   - The caller must still verify that the decoded index belongs to its period.
//
// AI-Hints:
//   - Avoid direct unsigned-to-int casts when reading externally supplied state.
//   - Do not use a modulo reduction: it would silently accept corrupted cursors.
func checkedSnapshotIndex(value uint64) (int, error) {
	if value > MaxPeriod {
		return 0, ErrStateVersion
	}

	// Atoi validates the native int limit as part of the conversion.
	index, err := strconv.Atoi(strconv.FormatUint(value, 10))
	if err != nil {
		return 0, ErrStateVersion
	}
	return index, nil
}

// snapshotCursor validates count-to-ring relationships before native-int decoding.
// All bounds follow from accepted transition count and the stream's fixed period.
// It rejects flag states unreachable through the canonical scheduled rebase policy.
//
// Implementation:
//   - Stage 1: reject a mismatched version, configuration digest or array shape.
//   - Stage 2: derive accepted transition count, expected used slots and cursor modulo period.
//   - Stage 3: reject impossible epoch flags before bounded native-int decoding.
//
// Behavior highlights:
//   - Count zero has no transitions; Count one is only the initial price seed.
//
// Inputs:
//   - state: untrusted, caller-owned Snapshot with a fixed public wire format.
//
// Returns:
//   - used, head: checked native-int ring coordinates on success.
//   - extra: admissible epoch flag that records one unscheduled rebase.
//   - error: ErrStateVersion on inconsistent metadata.
//
// Errors:
//   - ErrStateVersion: incorrect version/digest/shape/cursor/epoch or unsafe index.
//
// Determinism:
//   - Checks fixed fields before deriving transition positions; no mutable state changes.
//
// Complexity:
//   - Time O(1) in the bounded period domain, Space O(log MaxPeriod) for coordinate parsing.
//
// Notes:
//   - Only a receiver constructed through New may call this helper.
//
// AI-Hints:
//   - Do not trust head independently of Count or accept an extra rebase on a scheduled boundary.
func (s *Stream) snapshotCursor(state Snapshot) (used, head int, extra bool, err error) {
	period := snapshotWord(s.period)
	if state.Algorithm != snapshotAlgorithm || state.Config != configHash(s.period) ||
		len(state.FloatState) != snapshotFloatHeader+2*s.period || len(state.IntState) != snapshotIntLength {
		return 0, 0, false, ErrStateVersion
	}

	// Count includes one seed-only bar. Do not underflow Count-1 at zero.
	transitions := uint64(0)
	if state.Count != 0 {
		transitions = state.Count - 1
	}
	expectedUsed := transitions
	if expectedUsed > period {
		expectedUsed = period
	}
	ints := state.IntState
	if ints[snapshotUsed] != expectedUsed || ints[snapshotHead] != transitions%period ||
		ints[snapshotExtraRebase] > 1 {
		return 0, 0, false, ErrStateVersion
	}

	// An extra drift rebase cannot arise before eviction, nor survive a scheduled rebase.
	// An emergency rebase requires a full window and cannot remain set at an
	// exact scheduled boundary, where projected resets the epoch flag.
	if ints[snapshotExtraRebase] == 1 && (transitions <= period || transitions%period == 0) {
		return 0, 0, false, ErrStateVersion
	}

	used, err = checkedSnapshotIndex(ints[snapshotUsed])
	if err != nil {
		return 0, 0, false, err
	}
	head, err = checkedSnapshotIndex(ints[snapshotHead])
	if err != nil {
		return 0, 0, false, err
	}
	return used, head, ints[snapshotExtraRebase] == 1, nil
}

// validateSnapshotRing checks the domain and placement of every physical slot.
// Unused slots must be zero and mutually exclusive sign contributions remain intact.
//
// Implementation:
//   - Stage 1: check seeded versus unseeded previous typical and finite prefix fields.
//   - Stage 2: ensure Count<=1 has no contributions or corrections.
//   - Stage 3: scan both physical rings for finite, nonnegative, mutually exclusive signs and unused zeros.
//
// Behavior highlights:
//   - A zero pair in an occupied slot is valid and must not be treated as unused.
//
// Inputs:
//   - floats: untrusted FloatState with an already validated fixed shape.
//   - period, used: checked ring size and candidate occupancy.
//   - count: accepted bar count, including the first seeding bar.
//
// Returns:
//   - error: nil only when the prefix and every ring entry meet the state domain.
//
// Errors:
//   - ErrStateVersion: invalid seeded state, nonfinite values, simultaneous signs or occupied unused slots.
//
// Determinism:
//   - Checks fixed prefix fields first and then physical slots in ascending index order.
//
// Complexity:
//   - Time O(Period), Space O(1), no allocation.
//
// Notes:
//   - This gate checks domain and placement; validateSnapshotTotals checks sums separately.
//
// AI-Hints:
//   - Do not skip zero-volume slots or infer ring occupancy from nonzero contribution counts.
func validateSnapshotRing(floats []float64, period, used int, count uint64) error {
	previous := floats[snapshotPrevious]
	if (count == 0 && previous != 0) ||
		(count != 0 && (!finite(previous) || previous <= 0)) {
		return ErrStateVersion
	}
	for i := 0; i < snapshotFloatHeader; i++ {
		if !finite(floats[i]) {
			return ErrStateVersion
		}
	}
	if count <= 1 && (floats[snapshotPositive] != 0 || floats[snapshotNegative] != 0 ||
		floats[snapshotCorrPos] != 0 || floats[snapshotCorrNeg] != 0) {
		return ErrStateVersion
	}

	positive := floats[snapshotFloatHeader : snapshotFloatHeader+period]
	negative := floats[snapshotFloatHeader+period:]
	// Physical slot order is stable; zero entries in used slots are allowed,
	// but every slot outside the warmup occupancy must remain empty.
	for slot := 0; slot < period; slot++ {
		pos, neg := positive[slot], negative[slot]
		if !finite(pos) || !finite(neg) || pos < 0 || neg < 0 || (pos > 0 && neg > 0) {
			return ErrStateVersion
		}
		if slot >= used && used < period && (pos != 0 || neg != 0) {
			return ErrStateVersion
		}
	}
	return nil
}

// referenceSum independently replays occupied slots in logical oldest-to-newest order.
// It computes both a compensated checksum and an upper magnitude scale without
// allocating, providing a bounded comparison against the saved recurrence.
//
// Implementation:
//   - Stage 1: choose oldest logical index (head when full, zero during warmup).
//   - Stage 2: add each occupied value in order with Neumaier and accumulate nonnegative magnitude.
//   - Stage 3: return both a compensated reference and its independent size bound.
//
// Behavior highlights:
//   - Does not modify the source ring or overwrite the saved compensation.
//
// Inputs:
//   - values: one finite nonnegative sign ring of length period.
//   - used: number of occupied slots, bounded by ring length.
//   - head: next overwrite index, used as oldest when the ring is full.
//
// Returns:
//   - compensated: independently replayed ring accumulator.
//   - float64: checked magnitude used for roundoff scale.
//   - error: ErrOverflow if replay or magnitude is not representable.
//
// Errors:
//   - ErrOverflow: compensated replay or magnitude sum exceeds finite float64.
//
// Determinism:
//   - Visits occupied logical slots from oldest to newest with fixed sign arithmetic.
//
// Complexity:
//   - Time O(used), Space O(1), no allocation.
//
// Notes:
//   - The caller translates a validation failure into ErrStateVersion.
//
// AI-Hints:
//   - Do not replace this independent replay with the saved sum; it would stop detecting corruption.
func referenceSum(values []float64, used, head int) (compensated, float64, error) {
	var result compensated
	magnitude := 0.0
	index := 0
	if used == len(values) {
		index = head
	}
	for i := 0; i < used; i++ {
		value := values[index]
		if err := result.add(value); err != nil {
			return compensated{}, 0, err
		}
		if magnitude > math.MaxFloat64-value {
			return compensated{}, 0, ErrStateVersion
		}
		magnitude += value
		index++
		if index == len(values) {
			index = 0
		}
	}
	return result, magnitude, nil
}

// nearSaved compares normalized binary64 sums against a conservative roundoff envelope.
// 64*2^-52*(Period+2) accounts for a bounded epoch's incremental evictions and
// insertions; it is a corruption screen, not evidence of adversarial authenticity.
//
// Implementation:
//   - Stage 1: reject nonfinite effective sums.
//   - Stage 2: normalize both values by the greatest saved/reference/magnitude scale.
//   - Stage 3: compare their normalized difference against the documented epoch bound.
//
// Behavior highlights:
//   - A zero magnitude requires both sums exactly zero, not an arbitrary epsilon.
//
// Inputs:
//   - saved: effective sum reconstructed from saved sum and correction.
//   - reference: effective sum from independent logical ring replay.
//   - magnitude: nonnegative upper magnitude of ring entries.
//   - period: effective window, controlling the bounded epoch tolerance.
//
// Returns:
//   - bool: true when normalized discrepancy is within 64*2^-52*(Period+2).
//
// Determinism:
//   - Uses fixed binary64 comparison order and a contract-defined bound.
//
// Complexity:
//   - Time O(1), Space O(1), zero allocation.
//
// Notes:
//   - A successful comparison is a numerical consistency screen, not authentication or reachability proof.
//
// AI-Hints:
//   - Do not use this relaxed Snapshot bound to clamp a negative live rolling sum.
func nearSaved(saved, reference, magnitude float64, period int) bool {
	if !finite(saved) || !finite(reference) {
		return false
	}
	scale := math.Max(math.Abs(saved), math.Abs(reference))
	scale = math.Max(scale, magnitude)
	if scale == 0 {
		return saved == 0 && reference == 0
	}
	const floatingPointUnitRoundoff = 0x1p-52
	const correctionSafetyMultiplier = 64
	boundFactor := correctionSafetyMultiplier * floatingPointUnitRoundoff * float64(period+2)
	return math.Abs(saved/scale-reference/scale) <= boundFactor
}

// validateSnapshotTotals verifies the effective saved sums against the occupied ring.
// It checks floating-point representability and rejects drift outside the specified
// roundoff envelope; it never silently replaces saved sums or their compensators.
//
// Implementation:
//   - Stage 1: independently sum occupied positive and negative physical ring arrays in logical order.
//   - Stage 2: evaluate both compensated references and both saved effective sums.
//   - Stage 3: enforce finiteness, nonnegativity and bounded normalized agreement.
//
// Behavior highlights:
//   - Saved Neumaier corrections remain unchanged after validation.
//
// Inputs:
//   - floats: already shape- and domain-validated FloatState.
//   - period, used, head: checked window length, occupancy and cursor.
//
// Returns:
//   - error: nil when both sign sums agree within the numerical checkpoint envelope.
//
// Errors:
//   - ErrStateVersion: replay, saved effective sum or comparison violates the state contract.
//
// Determinism:
//   - Checks positive ring before negative ring; each follows oldest-to-newest order.
//
// Complexity:
//   - Time O(Period), Space O(1), zero allocation.
//
// Notes:
//   - The future recurrence uses saved components; replay is only a validation witness.
//
// AI-Hints:
//   - Do not normalize/rebase the saved accumulator during Restore; continuation must retain its bits.
func validateSnapshotTotals(floats []float64, period, used, head int) error {
	positive := floats[snapshotFloatHeader : snapshotFloatHeader+period]
	negative := floats[snapshotFloatHeader+period:]
	posReference, posMagnitude, err := referenceSum(positive, used, head)
	if err != nil {
		return ErrStateVersion
	}
	negReference, negMagnitude, err := referenceSum(negative, used, head)
	if err != nil {
		return ErrStateVersion
	}

	posRecomputed, err := posReference.value()
	if err != nil {
		return ErrStateVersion
	}
	negRecomputed, err := negReference.value()
	if err != nil {
		return ErrStateVersion
	}
	// Keep saved compensation for eventual commit; this arithmetic is only
	// a consistency check against a separate oldest-to-newest replay.
	posSaved := floats[snapshotPositive] + floats[snapshotCorrPos]
	negSaved := floats[snapshotNegative] + floats[snapshotCorrNeg]
	if !finite(posSaved) || !finite(negSaved) || posSaved < 0 || negSaved < 0 ||
		!nearSaved(posSaved, posRecomputed, posMagnitude, period) ||
		!nearSaved(negSaved, negRecomputed, negMagnitude, period) {
		return ErrStateVersion
	}
	return nil
}

// Restore installs a compatible Snapshot without mutating state on failure.
// It validates the complete checkpoint before copying any ring slot, preserving
// the saved Neumaier recurrence exactly instead of restarting from recomputed sums.
//
// Implementation:
//   - Stage 1: classify nil/uninitialized receiver and verify schema/counters.
//   - Stage 2: validate all ring entries and recompute numeric consistency checks.
//   - Stage 3: atomically publish the detached ring copy and all scalar state.
//
// Behavior highlights:
//   - Count, used, head and extraRebaseUsed must describe a reachable epoch.
//   - Ring content is checked in physical order and sums in logical order.
//   - A failure leaves the previous state bit-for-bit intact.
//
// Inputs:
//   - state: caller-owned read-only snapshot, with exact v1 algorithm/config ABI.
//
// Returns:
//   - error: nil only if the complete state was validated and installed.
//
// Errors:
//   - ErrNilInput: nil Stream receiver.
//   - ErrStateVersion: uninitialized engine or invalid/incompatible checkpoint.
//
// Determinism:
//   - Validates slots in index order; checks both signs in fixed positive/negative order.
//   - Accepted state preserves exact ring ordering and both stored compensators.
//
// Complexity:
//   - Time O(Period), Space O(1) extra; copies into existing Stream ring storage.
//
// Notes:
//   - The engine's period cannot change; a compatible target is created with New.
//   - Caller must authenticate untrusted persisted bytes before invoking Restore.
//   - Concurrent use of the same Stream is unsupported.
//
// AI-Hints:
//   - Never commit one ring or a counter before all checksum and geometry checks pass.
//   - Do not replace saved compensation with a newly computed value on successful Restore.
func (s *Stream) Restore(state Snapshot) error {
	if s == nil {
		return ErrNilInput
	}
	if !s.ready() {
		return ErrStateVersion
	}

	used, head, extra, err := s.snapshotCursor(state)
	if err != nil {
		return err
	}
	if err = validateSnapshotRing(state.FloatState, s.period, used, state.Count); err != nil {
		return err
	}
	if err = validateSnapshotTotals(state.FloatState, s.period, used, head); err != nil {
		return err
	}

	// No allocation or error-producing operation remains after this commit boundary.
	floats := state.FloatState
	copy(s.pos, floats[snapshotFloatHeader:snapshotFloatHeader+s.period])
	copy(s.neg, floats[snapshotFloatHeader+s.period:])
	s.previous = floats[snapshotPrevious]
	s.positive = compensated{sum: floats[snapshotPositive], correction: floats[snapshotCorrPos]}
	s.negative = compensated{sum: floats[snapshotNegative], correction: floats[snapshotCorrNeg]}
	s.used, s.head, s.count, s.extraRebaseUsed = used, head, state.Count, extra
	return nil
}
