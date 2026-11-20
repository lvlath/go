// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax defines reproducible versioned research records and JSON boundaries.
// This file fixes the protocol/definition schema and avoids implicit experiment defaults.
package tfsax

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
)

const (
	protocolSchema   = "lvlath-tfsax-protocol-v1"   // Experiment schema for frozen budgets and comparison policy.
	definitionSchema = "lvlath-tfsax-definition-v1" // Definition record schema, not an encoder version.
	defaultSeed      = 29                           // Reproducible seed; no global RNG is consulted.
	defaultLength    = 32                           // Fixture sample count per scalar window.
	defaultCorpus    = 64                           // Fixture number of candidate windows.
	defaultQueries   = 8                            // Fixture number of independently identified queries.
	defaultTopK      = 3                            // Fixture nearest-neighbor target.
	defaultShortlist = 16                           // Fixture approximate full-DTW refinement budget.
	defaultWindow    = 4                            // Fixture Sakoe-Chiba radius in samples.
	defaultMaxCells  = 1 << 24                      // Entire experiment budget of exact DP cells.
)

// Definition records the required mathematical metadata for a proposed representation.
// Approved is an externally reviewed metadata flag, never proof of an implemented encoder.
//
// Implementation:
//   - Stage 1: collect source, sample domain, transforms and exact scoring semantics.
//   - Stage 2: validate required fields if Approved is asserted.
//   - Stage 3: serialize for audit without creating Fit, Encode or Search operations.
//
// Behavior highlights:
//   - An unresolved record may leave fields blank, but may not enable production dispatch.
//
// Inputs:
//   - All strings are specification prose/provenance, not executable formulas.
//
// Returns:
//   - Struct may be serialized; no mathematical algorithm is inferred from it.
//
// Errors:
//   - ValidateDefinition returns ErrInvalidOption for an incompatible or incomplete record.
//
// Determinism:
//   - JSON field order is fixed by exported struct declaration order.
//
// Complexity:
//   - Validation O(total string bytes), serialization O(total record bytes).
//
// Notes:
//   - A complete JSON record is not an independently audited implementation/proof certificate.
//
// AI-Hints:
//   - Never interpret Approved=true as authorization to dispatch to a nonexistent encoder.
type Definition struct {
	Schema        string `json:"schema"`         // Versioned record layout, not a production codec version.
	Name          string `json:"name"`           // Unambiguous author-defined full name or unresolved label.
	Source        string `json:"source"`         // Author/source provenance or explicitly new proposal.
	Domain        string `json:"domain"`         // Uni-/multivariate sample domain and units.
	TimeGrid      string `json:"time_grid"`      // Permitted lengths, timestamps and sampling grid.
	Normalization string `json:"normalization"`  // Centering, scales, DDOF and training ownership.
	Segmentation  string `json:"segmentation"`   // Exact segmentation and feature extraction method.
	Alphabet      string `json:"alphabet"`       // Symbol types, bins and output cardinality.
	Boundaries    string `json:"boundaries"`     // Quantile/boundary ownership and equality convention.
	OutputShape   string `json:"output_shape"`   // Symbol word dimensions and metadata.
	MissingPolicy string `json:"missing_policy"` // Finite, constant and missing-value behavior.
	Score         string `json:"score"`          // Exact heuristic or metric and dimensional units.
	Ordering      string `json:"ordering"`       // Comparison ties and deterministic ordering.
	PruningProof  string `json:"pruning_proof"`  // Proof/counterexamples and numerical certificate status.
	Ownership     string `json:"ownership"`      // Aliasing, thread-safety and version guarantees.
	Approved      bool   `json:"approved"`       // Research review metadata, not automatic dispatch permission.
}

// Thresholds fixes independently declared empirical criteria before any measurement.
// They are engineering admission requirements, not proven properties of TFSAX.
//
// Implementation:
//   - Validate all four finite ratios/proportions as part of the complete Protocol before benchmarking.
//
// Behavior highlights:
//   - RecallMin and MissedBestMax are proportions; SpeedupMin and MemoryRatioMax ratios.
//
// Inputs:
//   - Four finite thresholds validated by ValidateProtocol.
//
// Returns:
//   - Value fields precommitting test criteria; they are not fitted parameters.
//
// Errors:
//   - ValidateProtocol returns ErrInvalidOption for nonfinite or out-of-range thresholds.
//
// Determinism:
//   - Values are immutable-for-the-run and hashed in JSON protocol identity.
//
// Complexity:
//   - O(1) storage.
//
// Notes:
//   - The runner does not automatically certify performance merely because these values exist.
//
// AI-Hints:
//   - Changing gates after looking at test results requires a new experiment and holdout.
type Thresholds struct {
	RecallMin      float64 `json:"recall_min"`       // Per-declared non-adversarial family, default >=0.99.
	MissedBestMax  float64 `json:"missed_best_max"`  // Missed exact-first ID rate, default <=0.01.
	SpeedupMin     float64 `json:"speedup_min"`      // Median latency improvement vs SAX, default >=2.
	MemoryRatioMax float64 `json:"memory_ratio_max"` // Index-memory ratio vs SAX, default <=2.
}

// Protocol freezes the exact target contract and experiment resource limits.
// It is checked before creating a corpus, index or measurement.
//
// Implementation:
//   - Stage 1: bind scalar window length, corpus/query counts and TopK.
//   - Stage 2: freeze Sakoe-Chiba radius, shortlist size and temporal gap policy.
//   - Stage 3: validate work budget and empirical gates before computation.
//
// Behavior highlights:
//   - Window=-1 means unrestricted; all other nonnegative windows constrain |i-j|.
//   - MaxCells bounds all exact baseline DP cells, not just one query's workload.
//
// Inputs:
//   - Seed: a uint64 replay identifier used with independent logical RNG paths.
//
// Returns:
//   - A versioned value record; it does not own corpus samples.
//
// Errors:
//   - ValidateProtocol returns ErrInvalidOption or ErrBudget.
//
// Determinism:
//   - Equal protocol records and RNG implementation versions replay equal generated series.
//
// Complexity:
//   - Validation O(1), storage O(1).
//
// Notes:
//   - This schema cannot encode an absent TFSAX formula or prove train-only fitting.
//
// AI-Hints:
//   - Pin the protocol hash before experiments and never compare incompatible DTW cost policies.
type Protocol struct {
	Schema     string     `json:"schema"`      // Fixed layout revision.
	Seed       uint64     `json:"seed"`        // Deterministic RNG seed.
	Length     int        `json:"length"`      // Scalar samples per candidate/query.
	CorpusSize int        `json:"corpus_size"` // Number of unique candidates.
	QueryCount int        `json:"query_count"` // Number of stable query IDs.
	TopK       int        `json:"top_k"`       // Requested exact/heuristic neighbor count.
	Shortlist  int        `json:"shortlist"`   // SAX full-DTW refine budget.
	Window     int        `json:"window"`      // Sakoe-Chiba radius; -1 is unrestricted.
	Purge      int        `json:"purge"`       // Raw support separation, >=Length.
	Embargo    int        `json:"embargo"`     // Extra post-split isolation in samples.
	Horizon    int        `json:"horizon"`     // Label horizon beyond feature support.
	MaxCells   uint64     `json:"max_cells"`   // Q*N*Length^2 DP-cell work limit.
	Thresholds Thresholds `json:"thresholds"`  // Precommitted research acceptance gates.
}

// DefaultProtocol returns a bounded illustrative research protocol.
// Every default is explicit and subject to ValidateProtocol before execution.
//
// Implementation:
//   - Stage 1: fix schema, seed, dimensions and computational budget.
//   - Stage 2: set conservative temporal separation and predeclared success gates.
//
// Behavior highlights:
//   - Defaults do not imply a defined TFSAX algorithm or empirical result.
//
// Inputs:
//   - No arguments: uses pinned illustrative seed, sample counts, gap policy and admission criteria.
//
// Returns:
//   - Protocol with no borrowed references or hidden mutable state.
//
// Errors:
//   - None from construction; ValidateProtocol checks edited values.
//
// Determinism:
//   - Pure constant-value assembly.
//
// Complexity:
//   - O(1) time and space.
//
// Notes:
//   - A caller must select and freeze actual experiment counts before benchmarking.
//
// AI-Hints:
//   - Store a JSON digest before running; passing a zero-value Protocol is invalid.
func DefaultProtocol() Protocol {
	return Protocol{
		Schema: protocolSchema, Seed: defaultSeed, Length: defaultLength,
		CorpusSize: defaultCorpus, QueryCount: defaultQueries, TopK: defaultTopK,
		Shortlist: defaultShortlist, Window: defaultWindow, Purge: defaultLength,
		MaxCells:   defaultMaxCells,
		Thresholds: Thresholds{RecallMin: 0.99, MissedBestMax: 0.01, SpeedupMin: 2, MemoryRatioMax: 2},
	}
}

// UnresolvedDefinition returns an explicit block on undefined symbolic research.
// Missing scientific metadata remains absent rather than being filled with guessed theory.
//
// Implementation:
//   - Stage 1: select the versioned research schema.
//   - Stage 2: publish an unresolved identity with Approved=false.
//
// Behavior highlights:
//   - Approved is always false; unavailable mathematical clauses remain empty rather than implied.
//
// Inputs:
//   - No arguments: returns a declared unresolved research record only.
//
// Returns:
//   - Standalone record that can be serialized without authorizing any encoder.
//
// Errors:
//   - None.
//
// Determinism:
//   - Pure constant-value construction.
//
// Complexity:
//   - O(1) time and space.
//
// Notes:
//   - The returned struct is safe to copy and edit, but a reviewed metadata record cannot instantiate an absent encoder.
//
// AI-Hints:
//   - Do not replace this record with made-up time-frequency/trend formulas.
func UnresolvedDefinition() Definition {
	return Definition{Schema: definitionSchema, Name: "TFSAX (unresolved)", Approved: false}
}

// ValidateDefinition checks that an asserted definition has complete provenance metadata.
// Passing this check never authorizes production inference or proves a DTW lower bound.
//
// Implementation:
//   - Stage 1: confirm the schema and a non-empty identity.
//   - Stage 2: require all scientific policy fields if Approved is asserted.
//
// Behavior highlights:
//   - Unresolved records may omit scientific clauses; Approved records must provide every named field.
//
// Inputs:
//   - d: a read-only Definition with an exact matching schema label and a nonempty name.
//
// Returns:
//   - nil for a well-formed record; not an encoder certification.
//
// Errors:
//   - ErrInvalidOption when the record is malformed or falsely asserts completion.
//
// Determinism:
//   - Fixed ordered field checks; no map or reflection in application code.
//
// Complexity:
//   - O(1) field checks, O(total string length) for non-empty trimming.
//
// Notes:
//   - Independent review, implementation and full empirical gates are still required.
//
// AI-Hints:
//   - Keep the independent encoder gate blocked even for an Approved JSON record.
func ValidateDefinition(d Definition) error {
	if d.Schema != definitionSchema || d.Name == "" {
		return ErrInvalidOption
	}
	if d.Approved && (d.Source == "" || d.Domain == "" || d.TimeGrid == "" ||
		d.Normalization == "" || d.Segmentation == "" || d.Alphabet == "" ||
		d.Boundaries == "" || d.OutputShape == "" || d.MissingPolicy == "" ||
		d.Score == "" || d.Ordering == "" || d.PruningProof == "" || d.Ownership == "") {
		return ErrInvalidOption
	}
	return nil
}

// DigestJSON returns a SHA-256 hash of Go's deterministic JSON encoding of a value.
// This permits audit comparisons but does not authenticate unknown authors or algorithms.
//
// Implementation:
//   - Stage 1: encode the provided value with encoding/json.
//   - Stage 2: hash the complete serialized byte sequence and hex-encode it.
//
// Behavior highlights:
//   - The digest uses compact serialization; indentation of on-disk JSON is not part of this identity.
//
// Inputs:
//   - value: JSON-serializable record; caller retains ownership.
//
// Returns:
//   - Lowercase SHA-256 hexadecimal digest, 64 characters on success.
//
// Errors:
//   - Original JSON marshal errors, e.g. unsupported type or NaN/Inf.
//
// Determinism:
//   - Same Go JSON representation yields equal digests; maps have sorted keys.
//
// Complexity:
//   - Time O(serialized bytes), extra memory O(serialized bytes).
//
// Notes:
//   - File whitespace and indentation differ from compact JSON digest.
//
// AI-Hints:
//   - Hash the frozen semantic record, not wall-clock durations or mutable process state.
func DigestJSON(value any) (string, error) {
	payload, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(payload)
	return hex.EncodeToString(hash[:]), nil
}

// ParseProtocol decodes one strict JSON protocol document and validates its gates.
// Unknown fields and trailing JSON values are errors rather than silently ignored.
//
// Implementation:
//   - Stage 1: reject empty source and decode with unknown-field rejection.
//   - Stage 2: confirm EOF after exactly one JSON document.
//   - Stage 3: call ValidateProtocol before publishing the record.
//
// Behavior highlights:
//   - Unknown object keys and additional JSON documents are rejected, then normal budget/threshold validation runs.
//
// Inputs:
//   - raw: caller-owned JSON document bytes; not retained.
//
// Returns:
//   - Validated Protocol value, or zero value on error.
//
// Errors:
//   - ErrInvalidInput for malformed or extra JSON; ErrInvalidOption/ErrBudget for policy.
//
// Determinism:
//   - JSON parsing and field validation follow fixed document and schema laws.
//
// Complexity:
//   - O(len(raw)) time and temporary memory for decoding.
//
// Notes:
//   - Use only to import a versioned audit document; native Go callers can construct and validate Protocol values directly.
//
// AI-Hints:
//   - Never use json.Unmarshal alone for an externally provided research plan.
func ParseProtocol(raw []byte) (Protocol, error) {
	if len(raw) == 0 {
		return Protocol{}, ErrInvalidInput
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var record Protocol
	if err := decoder.Decode(&record); err != nil {
		return Protocol{}, errors.Join(ErrInvalidInput, err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return Protocol{}, ErrInvalidInput
	}
	if err := ValidateProtocol(record); err != nil {
		return Protocol{}, err
	}
	return record, nil
}

// WriteJSON publishes one indented JSON artifact with same-directory atomic replacement.
// It prevents partially written output from being mistaken for a completed protocol.
//
// Implementation:
//   - Stage 1: serialize and append a deterministic terminal newline.
//   - Stage 2: create and fsync a mode-0600 temporary file in the target directory.
//   - Stage 3: close and atomically rename into the requested destination.
//
// Behavior highlights:
//   - An on-disk audit export is opt-in and never influences an already constructed Protocol or retrieval result.
//
// Inputs:
//   - path: destination path; its parent directory must already exist.
//   - value: JSON-serializable read-only object; no input mutation.
//
// Returns:
//   - nil after successful rename; no returned file descriptor.
//
// Errors:
//   - JSON marshal errors or original filesystem errors for create/write/sync/rename.
//
// Determinism:
//   - Complete payload bytes are deterministic; temporary filename is not serialized.
//
// Complexity:
//   - O(serialized bytes) time and temporary memory.
//
// Notes:
//   - Atomic rename is same-directory; cross-process durability of the directory is FS-dependent.
//
// AI-Hints:
//   - Do not stream partial JSON directly into a final protocol filename.
func WriteJSON(path string, value any) error {
	payload, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	return writeAtomically(path, append(payload, '\n'))
}

// writeAtomically owns a same-directory temporary file until successful rename.
// File write, sync and close failures are reported; removal on failure is best-effort.
func writeAtomically(path string, payload []byte) error {
	if path == "" {
		return ErrInvalidOption
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".tfsax-*")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(file.Name()) }()
	if err := file.Chmod(0600); err != nil {
		_ = file.Close()
		return err
	}
	if _, err := file.Write(payload); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(file.Name(), path)
}
