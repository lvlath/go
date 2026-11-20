// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax generates bounded, deterministic synthetic time-series stress corpora.
// This file owns the eight family kernels and byte-exact corpus provenance hashing.
package tfsax

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"hash"
	"math"

	"github.com/lvlath/go/rng"
)

const (
	// GeneratorVersion pins the family equations, RNG draw order and logical path assignment.
	// Any change to those laws requires a new version and independent fixture hashes.
	GeneratorVersion = "tfsax-synthetic-eight-families-v2"

	generatorPathStride = 1000000 // Distinct family path-ID blocks exceed the maximum corpus count.
	heavyTailThreshold  = 0.04    // Fixed rare-shock frequency in the illustrative stress generator.
	baselineNoiseScale  = 0.12    // Fixed additive normal perturbation amplitude.
)

// GenerateFamily produces a reproducible corpus for one declared stress family.
// Each path's samples depend only on seed, family number, stable ID and RNG version.
//
// Implementation:
//   - Stage 1: validate family, count, length and checked sample budget before allocation.
//   - Stage 2: allocate one independently seeded RNG stream and window per CandidateID.
//   - Stage 3: apply exactly one family transform in increasing sample-index order.
//
// Behavior highlights:
//   - FamilyMotif performs an unsigned cyclic phase rotation with no unsafe uint64-to-int cast.
//   - Constant, warp, noise, affine, plateau, trend, phase and motif scenarios remain separate.
//   - Plateau trajectories retain exactly flat intervals but are independent across paths.
//
// Inputs:
//   - family: one of eight named deterministic stress families.
//   - seed: unsigned reproducibility seed, including zero.
//   - count: number of windows in [1,100000].
//   - length: samples per window in [4,4096], with bounded total scalar count.
//
// Returns:
//   - []Candidate: newly allocated, independently owned windows with IDs [0,count).
//
// Errors:
//   - ErrBudget for invalid sizes or more than maxGeneratorSamples generated values.
//   - ErrInvalidOption for an unknown family.
//
// Determinism:
//   - Candidate IDs are ascending; RNG path is familyNumber*stride+ID.
//   - Family transforms consume streams in fixed sample order regardless of worker scheduling.
//
// Complexity:
//   - Time O(count*length), output space O(count*length), transient O(1) per stream.
//
// Notes:
//   - These are synthetic stress models, never actual exchange prices or clinical signals.
//   - Changing transforms or draw order requires a new generator version and fixture hash.
//
// AI-Hints:
//   - Never reseed by worker ID or silently replace uniform cyclic motif rotation with float sampling.
//   - Keep uint64 phase modulo arithmetic unsigned; casting its random source to int triggers G115.
func GenerateFamily(family Family, seed uint64, count, length int) ([]Candidate, error) {
	// Validate the divisor's positive finite range before checking the total sample
	// product. No count*length multiplication precedes the resource gate.
	if length < 4 || length > maxSeriesLength ||
		count < 1 || count > maxExperimentItems || count > maxGeneratorSamples/length {
		return nil, ErrBudget
	}
	familyID := familyNumber(family)
	if familyID == 0 {
		return nil, ErrInvalidOption
	}

	factory := rng.NewFactory()
	candidates := make([]Candidate, 0, count)
	lengthUnsigned := uint64(length) // Positive and <=4096 after the preceding explicit guard.
	var logicalID uint64
	for id := 0; id < count; id++ {
		// Increment the unsigned logical ID directly. No signed-to-unsigned cast
		// depends on the loop implementation or on an untrusted random word.
		pathID := familyID*generatorPathStride + logicalID
		stream := factory.Stream(seed, pathID)
		values := make([]float64, length)
		motifPosition := stream.Uint64() % lengthUnsigned
		slope := 0.25 + stream.UniformOpen()
		// Plateau geometry and phase depend on the logical trajectory, not on its
		// sample index. Compute them once so only samples perform numeric synthesis.
		plateauStart := length/3 + id%3 - 1
		if plateauStart < 0 {
			plateauStart = 0
		}
		plateauEnd := plateauStart + length/3
		if plateauEnd > length {
			plateauEnd = length
		}
		plateauLevel := 0.35 + 0.04*float64(id%7)
		plateauPhase := 0.11 * float64(id%11)
		for sampleIndex := range values {
			t := float64(sampleIndex) / float64(length-1)
			switch family {
			case FamilyConstant:
				if id%2 == 0 {
					values[sampleIndex] = float64(id % 3)
				} else if sampleIndex >= length/2 {
					values[sampleIndex] = 1
				}
			case FamilyHeavyTail:
				z := stream.Normal()
				values[sampleIndex] = math.Sin(2*math.Pi*t) + z*baselineNoiseScale
				if stream.UniformOpen() < heavyTailThreshold {
					values[sampleIndex] += math.Copysign(2+3*stream.UniformOpen(), z)
				}
			case FamilyWarp:
				warped := t * t * (3 - 2*t) // Smooth monotone remapping of the unit interval.
				values[sampleIndex] = math.Sin(4*math.Pi*warped) + 0.05*stream.Normal()
			case FamilySlope:
				if t < 0.5 {
					values[sampleIndex] = slope * t
				} else {
					values[sampleIndex] = slope*0.5 - (t-0.5)*slope*1.5
				}
			case FamilyMotif:
				// Increment modulo length in uint64: no lossy or unsafe integer narrowing.
				motif := math.Sin(6 * math.Pi * float64(motifPosition) / float64(length))
				values[sampleIndex] = motif + 0.15*stream.Normal()
				motifPosition++
				if motifPosition == lengthUnsigned {
					motifPosition = 0
				}
			case FamilyPhase:
				values[sampleIndex] = math.Sin(4*math.Pi*t + float64(id%9)*math.Pi/9)
			case FamilyAffine:
				values[sampleIndex] = (1+float64(id%5))*math.Sin(2*math.Pi*t) + float64(id%7)
			case FamilyPlateau:
				// Each trajectory contains an exactly flat interval, but has its own
				// baseline phase, plateau level, placement and independent sensor noise.
				// The previous identical-across-ID family would artificially inflate recall.
				if sampleIndex >= plateauStart && sampleIndex < plateauEnd {
					values[sampleIndex] = plateauLevel
				} else {
					values[sampleIndex] = math.Sin(4*math.Pi*t+plateauPhase) + 0.03*stream.Normal()
				}
			}
		}
		candidates = append(candidates, Candidate{ID: CandidateID(logicalID), Values: values})
		logicalID++
	}
	return candidates, nil
}

// familyNumber assigns stable unsigned RNG-domain identifiers to known families.
// Unknown names map to zero, which cannot become a valid generator family.
func familyNumber(family Family) uint64 {
	switch family {
	case FamilyConstant:
		return 1
	case FamilyHeavyTail:
		return 2
	case FamilyWarp:
		return 3
	case FamilySlope:
		return 4
	case FamilyMotif:
		return 5
	case FamilyPhase:
		return 6
	case FamilyAffine:
		return 7
	case FamilyPlateau:
		return 8
	default:
		return 0
	}
}

// knownFamily reports whether a family participates in the frozen stress universe.
// The lookup shares the stable path-numbering source of truth.
func knownFamily(family Family) bool { return familyNumber(family) != 0 }

// CorpusHash commits to candidate order, IDs, lengths and exact float64 sample bits.
// Identical ordered universes produce equal SHA-256 digests across repeated executions.
//
// Implementation:
//   - Stage 1: validate uniform lengths, distinct IDs and finite samples.
//   - Stage 2: serialize each ID, length and sample into little-endian uint64 words.
//   - Stage 3: hash the full ordered stream and publish lowercase SHA-256 hexadecimal.
//
// Behavior highlights:
//   - Negative zero and positive zero retain different bit-level provenance.
//
// Inputs:
//   - candidates: read-only caller-owned ordered universe; not retained.
//
// Returns:
//   - SHA-256 digest of exact data bytes, or empty string and classified validation error.
//
// Errors:
//   - ErrInvalidInput for missing/jagged corpus; ErrDuplicateID for collisions.
//   - ErrNonFinite for any NaN/Inf sample.
//
// Determinism:
//   - Serialization order is ascending corpus index and ascending sample index.
//
// Complexity:
//   - Time O(N*L); O(N) validation IDs, O(1) hash workspace.
//
// Notes:
//   - A hash detects changed content only when compared to a trusted frozen reference.
//
// AI-Hints:
//   - Do not sort candidates before hashing: input order is part of the declared universe.
func CorpusHash(candidates []Candidate) (string, error) {
	if err := validateCorpusShape(candidates); err != nil {
		return "", err
	}
	return corpusHashValidated(candidates), nil
}

// corpusHashValidated is the canonical byte serializer after corpus admission.
// It avoids ValidateCorpus->CorpusHash recursion and repeated O(N*L) validation.
func corpusHashValidated(candidates []Candidate) string {
	h := sha256.New()
	for _, candidate := range candidates {
		writeHashWord(h, uint64(candidate.ID))
		writeHashWord(h, uint64(len(candidate.Values)))
		for _, sample := range candidate.Values {
			writeHashWord(h, math.Float64bits(sample))
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// writeHashWord writes one fixed-width little-endian word to a SHA hash.
// SHA-256's Write never returns an error; the buffer is not retained.
func writeHashWord(h hash.Hash, word uint64) {
	var bytes [8]byte
	binary.LittleEndian.PutUint64(bytes[:], word)
	_, _ = h.Write(bytes[:])
}
