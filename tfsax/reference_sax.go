// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2025-2026 katalvlaran

// Package tfsax adapts the existing PAA/SAX encoder to a DTW ranking-only baseline.
// This file owns the immutable prepared SAX index and its exact DTW refinement facade.
package tfsax

import (
	"context"

	"github.com/lvlath/go/sax"
)

const (
	defaultSAXSegments = 8                                     // Explicit research convenience default, requires length>=8.
	defaultSAXAlphabet = 8                                     // Standard eight-bin Gaussian SAX vocabulary.
	saxRankingVersion  = "sax-gaussian-mindist-dtw-ranking-v1" // Heuristic for DTW; never an exact-DTW certificate.
)

// SAXSearchOptions describes one explicit SAX ranking and DTW refinement policy.
// It is a convenience value; the search never guesses a segment count from the corpus.
//
// Implementation:
//   - One-shot facade validates each parameter and creates a prepared index before a single Search.
//
// Behavior highlights:
//   - Segments must be in [1,window length], Alphabet in [2,256].
//   - Shortlist must be between 1 and the number of candidates.
//
// Inputs:
//   - Window: DTW radius (-1 unrestricted, >=0 banded).
//   - Shortlist: number of candidate windows receiving full-DTW refinement.
//   - Segments: fractional PAA cells per window.
//   - Alphabet: Gaussian SAX symbol bins.
//
// Returns:
//   - Policy value specifying both symbolic ranking and exact-DTW refinement stages.
//
// Errors:
//   - Invalid values are classified by ScanSAXWithOptions or PrepareSAXIndex.
//
// Determinism:
//   - Exact fixed fields are passed unchanged to the baseline adapter.
//
// Complexity:
//   - O(1) configuration storage.
//
// Notes:
//   - A prepared SAXIndex is preferable when many queries share one frozen corpus.
//
// AI-Hints:
//   - Changing TopM is not the same as changing query.TopK.
type SAXSearchOptions struct {
	Window    int // Inclusive Sakoe-Chiba radius, -1 unrestricted.
	Shortlist int // Number of candidates to refine with exact DTW.
	Segments  int // Number of PAA cells in each SAX word.
	Alphabet  int // Number of symbol bins; [2,256].
}

// DefaultSAXSearchOptions supplies a disclosed, explicit eight-bin SAX policy.
// It uses the supplied shortlist and band without silently modifying either value.
//
// Implementation:
//   - Stage 1: set fixed Gaussian segment and alphabet defaults.
//   - Stage 2: preserve caller-specified window/TopM values exactly.
//
// Behavior highlights:
//   - The supplied band and shortlist are not silently clamped to corpus size.
//
// Inputs:
//   - window: DTW radius (-1 unrestricted); shortlist: caller-selected full-DTW refine count.
//
// Returns:
//   - Independent SAXSearchOptions value; no model or corpus allocation.
//
// Errors:
//   - None here; invalid values fail when passed to a scan.
//
// Determinism:
//   - Pure constant assembly.
//
// Complexity:
//   - O(1) time and memory.
//
// Notes:
//   - Validate the chosen shortlist against actual corpus size when the search executes.
//
// AI-Hints:
//   - The default requires at least eight samples per query; shorter inputs need explicit options.
func DefaultSAXSearchOptions(window, shortlist int) SAXSearchOptions {
	return SAXSearchOptions{Window: window, Shortlist: shortlist, Segments: defaultSAXSegments, Alphabet: defaultSAXAlphabet}
}

// SAXIndex stores a detached corpus and its immutable baseline SAX words.
// Repeated queries reuse the same pinned Gaussian thresholds and encoded windows.
//
// Implementation:
//   - Stage 1: PrepareSAXIndex validates and deep-copies every corpus window.
//   - Stage 2: construct one canonical sax.Encoder and encode each candidate once.
//   - Stage 3: Search encodes only the query, ranks by sax.Mindist and runs full DTW.
//
// Behavior highlights:
//   - SAX MINDIST is a Euclidean bound in SAX space, NOT an arbitrary-DTW lower bound.
//   - IndexBytes reports ID+symbol payload, not runtime RSS or raw source corpus storage.
//
// Inputs:
//   - Created by PrepareSAXIndex from a read-only, uniform-length finite candidate corpus.
//
// Returns:
//   - Immutable prepared index safe for concurrent read-only searches with disjoint inputs.
//
// Errors:
//   - Constructors classify malformed corpus/options; Search propagates DTW/SAX errors.
//
// Determinism:
//   - Candidate storage preserves original corpus order, ties sort by CandidateID.
//
// Complexity:
//   - Persistent retained space O(N*(L+segments)), not just IndexBytes payload.
//
// Notes:
//   - Model, words, and raw samples are private and detached from caller-owned buffers.
//
// AI-Hints:
//   - Do not report IndexBytes as measured resident memory in the research admission gate.
//   - Never expose internal sax.Result pointers: they are mutable public structures.
type SAXIndex struct {
	encoder  *sax.Encoder
	corpus   []Candidate
	words    []*sax.Result
	length   int
	segments int
	alphabet int
	digest   string
}

// PrepareSAXIndex builds one immutable, detached baseline SAX corpus index.
// Gaussian breakpoints are pinned by sax.NewEncoder, not fitted from held-out windows.
//
// Implementation:
//   - Stage 1: validate finite unique-ID fixed-length corpus and symbolic dimensions.
//   - Stage 2: clone every raw window and encode with population-Z plus ZeroWord constants.
//   - Stage 3: retain private immutable model and words, and commit to a corpus digest.
//
// Behavior highlights:
//   - Constants use an explicit ZeroWord policy to keep the full exact candidate universe.
//   - Per-window Z is a specified observation transform, not fitting global train parameters.
//
// Inputs:
//   - corpus: read-only finite equal-length unique-ID windows, no retained caller aliases.
//   - segments: number of fractional PAA cells, [1,L].
//   - alphabet: number of Gaussian SAX bins, [2,256].
//
// Returns:
//   - *SAXIndex with detached data, or nil on any error.
//
// Errors:
//   - ErrInvalidInput/ErrNonFinite/ErrDuplicateID from corpus validation.
//   - ErrInvalidOption for invalid dimensions; original SAX error identity is preserved.
//
// Determinism:
//   - Copies and encodes candidates in original corpus order.
//
// Complexity:
//   - Time O(N*(L+segments*log(alphabet))) plus one model creation;
//   - Space O(N*(L+segments)+alphabet), excluding temporary normalized buffers.
//
// Notes:
//   - Immutable reads are concurrent-safe; callers must not modify shared Query.Values.
//
// AI-Hints:
//   - Use a prepared index for repeated queries; one-shot ScanSAX includes model build time.
//   - Mindist is a ranking score only: never certify it as a DTW pruning lower bound.
func PrepareSAXIndex(corpus []Candidate, segments, alphabet int) (*SAXIndex, error) {
	if err := ValidateCorpus(corpus, ""); err != nil {
		return nil, err
	}
	length := len(corpus[0].Values)
	if segments < 1 || segments > length || alphabet < 2 || alphabet > 256 {
		return nil, ErrInvalidOption
	}
	options := sax.DefaultOptions(segments)
	options.Alphabet = alphabet
	options.ConstantPolicy = sax.ZeroWord
	encoder, err := sax.NewEncoder(options)
	if err != nil {
		return nil, err
	}
	index := &SAXIndex{
		encoder: encoder, length: length, segments: segments, alphabet: alphabet,
		corpus: make([]Candidate, 0, len(corpus)), words: make([]*sax.Result, 0, len(corpus)),
	}
	for _, candidate := range corpus {
		// Detach raw values as well as symbols so later external corpus mutation cannot
		// change the baseline's exact refinement target while its word stays frozen.
		copyValues := append([]float64(nil), candidate.Values...)
		word, err := encoder.Encode(copyValues)
		if err != nil {
			return nil, err
		}
		index.corpus = append(index.corpus, Candidate{ID: candidate.ID, Values: copyValues})
		index.words = append(index.words, word)
	}
	index.digest = corpusHashValidated(index.corpus)
	return index, nil
}

// IndexBytes reports the canonical SAX symbol and CandidateID payload byte count.
// It intentionally excludes Go container overhead, allocator fragmentation and raw corpus.
//
// Implementation:
//   - Stage 1: handle nil receiver without dereference.
//   - Stage 2: multiply the validated count by 2*segments plus 8 ID bytes.
//
// Behavior highlights:
//   - Counts only serialized IDs and symbol indices, not the raw windows required for refinement.
//
// Inputs:
//   - index: possibly nil *SAXIndex receiver; nil or uninitialized returns zero.
//
// Returns:
//   - uint64 payload bytes; zero for nil/uninitialized index.
//
// Errors:
//   - None; construction bounds prevent multiplication overflow.
//
// Determinism:
//   - Pure arithmetic on index dimensions.
//
// Complexity:
//   - O(1) time and space.
//
// Notes:
//   - Not suitable as proof of an index resident-memory ratio acceptance gate.
//
// AI-Hints:
//   - Measure actual allocated bytes separately from this serialized payload model.
func (index *SAXIndex) IndexBytes() uint64 {
	if index == nil || index.encoder == nil || index.segments < 1 {
		return 0
	}
	// The constructor admits strictly positive segments. A slice length is
	// always nonnegative; both conversions are widening on supported targets.
	return uint64(len(index.corpus)) * (uint64(index.segments)*2 + 8)
}

// CorpusDigest returns the ordered SHA-256 identity of the index-owned detached corpus.
// It helps correlate offline measurements with the exact candidate universe.
//
// Implementation:
//   - Return the stored ordered-corpus digest computed once during preparation.
//
// Behavior highlights:
//   - The result is stable even if caller-owned input arrays are later mutated.
//
// Inputs:
//   - index: possibly nil *SAXIndex; nil returns an empty string.
//
// Returns:
//   - Lowercase digest, or empty string for nil/uninitialized receiver.
//
// Errors:
//   - None; digest read performs no hashing or filesystem I/O.
//
// Determinism:
//   - Derived once from frozen IDs, lengths and sample bits during preparation.
//
// Complexity:
//   - O(1) time; the digest string is immutable.
//
// Notes:
//   - A matching hash authenticates neither the data source nor timestamps without a trusted frozen reference.
//
// AI-Hints:
//   - Compare this against a trusted external corpus manifest before a benchmark.
func (index *SAXIndex) CorpusDigest() string {
	if index == nil {
		return ""
	}
	return index.digest
}

// Search performs SAX MINDIST ranking followed by exact-DTW TopM refinement.
// It reuses prepared immutable words and never treats their symbolic score as DTW.
//
// Implementation:
//   - Stage 1: validate context, query shape and explicit band/refinement budget.
//   - Stage 2: encode query once, compare to every cached word using sax.Encoder.Mindist.
//   - Stage 3: sort symbolic scores, DTW-refine TopM and detach the exact-distance TopK IDs.
//
// Behavior highlights:
//   - The reported Exact flag is false because MINDIST does not certify DTW exclusions.
//   - Numeric ties in both ranking and exact refinement use the smaller CandidateID.
//
// Inputs:
//   - ctx: non-nil active context; checked before rankings and DTW calls.
//   - query: read-only finite L-sample vector and positive TopK.
//   - window: -1 unrestricted, otherwise >=0 Sakoe-Chiba radius.
//   - shortlist: integer TopM in [1,number of indexed candidates].
//
// Returns:
//   - Detached *RetrievalResult with independent CandidateIDs or nil on error.
//
// Errors:
//   - ErrInvalidOption for nil/uninitialized index, bad band or shortlist/context.
//   - ErrInvalidInput/ErrNonFinite for query shape or samples; SAX/DTW errors propagate.
//   - ctx.Err for cancellation; no usable partial result is published.
//
// Determinism:
//   - Index order for scoring, (MINDIST,ID) for ranking, (DTW,ID) for final output.
//
// Complexity:
//   - Time O(L + N*segments + N log N + TopM*L^2 + TopM log TopM).
//   - Space O(N+TopM+L) per search; no corpus re-encoding or input aliasing.
//
// Notes:
//   - Index is immutable and safe to share among concurrent Search calls.
//
// AI-Hints:
//   - Mindist's Euclidean mathematical property does not authorize exact DTW pruning.
//   - For latency comparisons prepare the index outside the timed search loop.
func (index *SAXIndex) Search(ctx context.Context, query Query, window, shortlist int) (*RetrievalResult, error) {
	if ctx == nil {
		return nil, ErrInvalidOption
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if index == nil || index.encoder == nil || len(index.corpus) == 0 ||
		window < -1 || shortlist < 1 || shortlist > len(index.corpus) {
		return nil, ErrInvalidOption
	}
	if err := ValidateQuery(query, index.length); err != nil {
		return nil, err
	}
	return index.searchValidated(ctx, query, window, shortlist)
}

// searchValidated performs query encoding, SAX scoring and refinement after facade validation.
// Experiment runners use it to exclude shared input validation from timed kernels.
func (index *SAXIndex) searchValidated(ctx context.Context, query Query, window, shortlist int) (*RetrievalResult, error) {
	word, err := index.encoder.Encode(query.Values)
	if err != nil {
		return nil, err
	}
	ranked := make([]scored, 0, len(index.words))
	for indexPosition, candidateWord := range index.words {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		score, err := index.encoder.Mindist(word, candidateWord)
		if err != nil {
			return nil, err
		}
		if !finite(score) || score < 0 {
			return nil, ErrOverflow
		}
		ranked = append(ranked, scored{id: index.corpus[indexPosition].ID, distance: score, index: indexPosition})
	}
	sortScores(ranked)
	return refineRanked(ctx, index.corpus, query, window, shortlist, saxRankingVersion, ranked)
}

// ScanSAXWithOptions performs one complete SAX-build plus ranking/refinement run.
// It is a convenience facade for one query, not a warmed prepared-index benchmark.
//
// Implementation:
//   - Stage 1: validate the context before model construction.
//   - Stage 2: prepare a detached SAXIndex using declared dimensions.
//   - Stage 3: search once using the selected band/TopM budget.
//
// Behavior highlights:
//   - One-shot execution includes index allocation and corpus copying, unlike a warmed prepared-index search.
//
// Inputs:
//   - ctx: non-nil active context; other inputs are read-only for the call.
//   - corpus: complete candidate universe with uniform finite windows.
//   - query: finite same-length observations and positive TopK.
//   - options: named segment, alphabet, window and shortlist policy.
//
// Returns:
//   - Detached *RetrievalResult or nil and classified error.
//
// Errors:
//   - ErrInvalidOption for invalid configuration/context; corpus, SAX, DTW errors preserved.
//
// Determinism:
//   - Same policy as SAXIndex.Search; model preparation is deterministic.
//
// Complexity:
//   - Includes O(N*L) copying and O(N*L) preprocessing plus Search cost.
//
// Notes:
//   - For repeated queries prefer PrepareSAXIndex and SAXIndex.Search.
//
// AI-Hints:
//   - This wrapper includes index build time; do not compare it to a warmed exact scan.
func ScanSAXWithOptions(ctx context.Context, corpus []Candidate, query Query, options SAXSearchOptions) (*RetrievalResult, error) {
	if ctx == nil {
		return nil, ErrInvalidOption
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if options.Window < -1 {
		return nil, ErrInvalidOption
	}
	index, err := PrepareSAXIndex(corpus, options.Segments, options.Alphabet)
	if err != nil {
		return nil, err
	}
	return index.Search(ctx, query, options.Window, options.Shortlist)
}

// ScanSAX preserves the explicit positional SAX ranking facade for existing callers.
// It delegates to the same prepared-index kernel used by ScanSAXWithOptions.
//
// Implementation:
//   - Stage 1: assemble the named policy from the explicit positional arguments.
//   - Stage 2: delegate to ScanSAXWithOptions without alternative mathematics.
//
// Behavior highlights:
//   - The four positional numeric values are translated directly into SAXSearchOptions without hidden defaults.
//
// Inputs:
//   - ctx, corpus, query: same read-only inputs as ScanSAXWithOptions.
//   - window, shortlist, segments, alphabet: explicit integer SAX/DTW policies.
//
// Returns:
//   - Detached *RetrievalResult, or nil with exactly the delegated failure.
//
// Errors:
//   - Same identities as ScanSAXWithOptions.
//
// Determinism:
//   - Exact identity and tie behavior of the canonical SAX search kernel.
//
// Complexity:
//   - Includes one index build and one Search; see ScanSAXWithOptions.
//
// Notes:
//   - Use the named-options facade when call sites benefit from explicit field semantics.
//
// AI-Hints:
//   - Treat this as heuristic ranking only, even when a SAX distance is numerically zero.
func ScanSAX(ctx context.Context, corpus []Candidate, query Query, window, shortlist, segments, alphabet int) (*RetrievalResult, error) {
	return ScanSAXWithOptions(ctx, corpus, query, SAXSearchOptions{
		Window: window, Shortlist: shortlist, Segments: segments, Alphabet: alphabet,
	})
}
