<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  lvlath - Contributing Guide

  Purpose:
    This guide defines how contributors propose, implement, test, document, and
    review changes to lvlath. It applies the lvlath Universal Engineering Standard
    (UES) to everyday contribution workflow: deterministic behavior, sentinel-first
    errors, synchronized public surfaces, contract-defending tests, honest examples,
    and regression-safe evolution.

  License:
    lvlath is licensed under AGPL-3.0-only. See LICENSE.
-->

# Contributing to lvlath

Thank you for contributing to **lvlath**.

This repository is not maintained as a pile of algorithm snippets. It is maintained as a deterministic, contract-driven Go library where code, tests, documentation, examples, and benchmarks must describe the same mathematical behavior.

The standard is simple to state and demanding to satisfy:

> A contribution is accepted only when it improves correctness, determinism, contract clarity, safety, documentation truth, or measurable engineering quality.

---

## 1. Before you start

Read these first:

1. [`README.md`](README.md) - public project map and package overview.
2. [`docs/TUTORIAL.md`](docs/TUTORIAL.md) - first guided learning path.
3. [`docs/lvlath_UES.md`](docs/lvlath_UES.md) - universal engineering standard for lvlath.
4. The package-local `{package}/doc.go` for the package you are changing.
5. The repository-level `docs/{PACKAGE}.md` for the same package.

The UES is not optional style guidance. It defines the contribution bar for public contracts, deterministic behavior, error governance, documentation synchronization, examples, tests, and benchmarks.

---

## 2. Core contribution laws

Every change must respect these laws.

### 2.1. Contract > implementation

Public behavior, sentinel errors, determinism, numeric policy, ownership, aliasing, concurrency assumptions, and partial-result semantics are part of the API contract.

Do not “optimize” or “simplify” code in a way that silently changes any of those rules.

### 2.2. Determinism by default

For the same inputs, options, graph/data state, and package version, observable results must remain stable:

* same ordering where ordering is contractual;
* same witness path/cycle when the package promises deterministic witnesses;
* same sentinel classification;
* same documented partial-result behavior;
* same matrix shape and numeric meaning.

### 2.3. Sentinel-first errors

Use `errors.Is` for classification. Never classify errors by string matching.

Good:

```go
if errors.Is(err, dijkstra.ErrTargetNotFound) {
	// stable protocol branch
}
```

Forbidden:

```go
if strings.Contains(err.Error(), "target") {
	// brittle; not a contract
}
```

### 2.4. One source of truth

There must be one canonical kernel for the actual mathematics. Public facades validate/apply options and delegate. Wrappers may reduce boilerplate, but they must not invent alternative mathematics.

### 2.5. Explicit policy only

No hidden fallback modes. No silent contract shifts. Any behavior that changes mathematical meaning must be explicit through an option, a named mode, or a documented type.

### 2.6. Tests defend mathematics

Tests exist to protect correct math and public contracts. Do not weaken the implementation just to satisfy a bad test. Fix the test and lock the true law in docs and regression tests.

### 2.7. Documentation is executable truth

`doc.go`, exported GoDoc, `docs/*.md`, examples, tests, and benchmarks must not describe different APIs or different contracts.

---

## 3. Development environment

### Requirements

* Go 1.23 or newer.
* Pure Go workflow: no cgo requirement.
* No external assertion frameworks such as `testify`.
* `golangci-lint` installed for local linting.

### Clone

```sh
git clone https://github.com/lvlath/go.git
cd lvlath
```

### Sync dependencies

```sh
go mod tidy
```

`lvlath` aims to stay lightweight and pure Go. Do not introduce dependencies casually. A dependency must be justified by clear engineering value, stable licensing, maintenance quality, and absence of a simpler standard-library solution.

---

## 4. Branching workflow

The repository uses a lightweight GitFlow-style process.

| Branch           | Role                                                        |
|:-----------------|:------------------------------------------------------------|
| `main`           | Stable branch representing released or release-ready state. |
| `v0.1.0-alpha`   | Active development branch for the current alpha line.       |
| `feature/<name>` | Focused feature branch.                                     |
| `fix/<name>`     | Focused bug-fix branch.                                     |
| `docs/<name>`    | Documentation-only branch.                                  |
| `test/<name>`    | Test-only or regression branch.                             |
| `hotfix/<name>`  | Urgent fix branch based on `main` when needed.              |

Create focused branches:

```sh
git checkout v0.1.0-alpha
git pull
git checkout -b feature/dijkstra-cutoff-contract
```

Keep PRs small enough to review deeply. A “small” PR is not measured only by lines changed; it is measured by whether one reviewer can validate the full contract impact.

---

## 5. Local verification commands

Run these before opening a PR.

```sh
gofmt -w .
go vet ./...
go test ./...
go test ./... -coverprofile=coverage.out
golangci-lint run
```

For packages with concurrency-sensitive code or recent lock/topology changes, also run:

```sh
go test -race ./...
```

For benchmark work:

```sh
go test ./... -run '^$' -bench . -benchmem
```

If a command fails, fix the cause before requesting review. Do not hide failure by narrowing the command unless the PR explicitly documents why a subset is being investigated.

---

## 6. Package architecture expectations

A non-trivial package should follow the UES layer model.

```text
{package}/
  api.go                 canonical facade + honest wrappers
  doc.go                 package charter and public laws
  errors.go              sentinel errors
  options.go             options and centralized option assembly
  types.go               public result/types/interfaces
  validators.go          validation and sanitization helpers
  impl_*.go              canonical kernels
  *_test.go              contract tests
  test_helpers_test.go   precise package-local test helpers
  example_test.go        stable executable package examples
  bench_test.go          shape-aware benchmarks where useful

docs/{PACKAGE}.md        repository-level tutorial/specification
```

Tiny packages may not need every file, but omissions should be intentional and should not blur facade/kernel/error/result responsibilities.

### Canonical facade rule

For non-trivial algorithm packages, the main public function should return a named result artifact.

Preferred shape:

```go
func Dijkstra(g *core.Graph, sourceID string, opts ...Option) (*Result, error)
```

Avoid canonical APIs like:

```go
func Solve(...) (map[string]float64, map[string]string, error) // too many parallel meanings
```

Wrappers are allowed when they are honest projections of the canonical result. They must document whether they force-enable a mode, discard fields, or perform a point query.

---

## 7. Adding or changing public API

Public API changes require synchronization across the full surface.

### 7.1. Required updates

When you add, remove, rename, or change any exported type/function/method/option/error, update:

1. source code;
2. package `doc.go`;
3. exported GoDoc;
4. `docs/{PACKAGE}.md`;
5. examples and scenario files;
6. tests;
7. benchmarks if the change affects performance or complexity;
8. README/Tutorial only if the public package story changes.

### 7.2. Public contract checklist

Document these where relevant:

* nil behavior;
* zero-shape or empty-input behavior;
* numeric policy for `NaN`, `+Inf`, `-Inf`, negative weights, and zero;
* graph capability requirements;
* deterministic order source and tie-breaks;
* sentinel errors and validation priority;
* ownership and aliasing;
* concurrency assumptions;
* partial-result behavior;
* complexity.

### 7.3. Compatibility aliases

If a compatibility alias is retained, docs must state:

* canonical name;
* alias name;
* whether the alias is legacy-only or first-class;
* whether it will be removed in a future major version.

Do not silently teach aliases as canonical in new examples.

---

## 8. Documentation standards

Documentation is part of the implementation surface.

### 8.1. `doc.go`

Each package `doc.go` should act as the package charter. It should define:

* domain scope;
* non-goals;
* public result artifacts;
* options and modes;
* error law;
* determinism law;
* numeric/topology policy;
* ownership and aliasing;
* concurrency assumptions;
* partial-result behavior;
* complexity.

Do not turn `doc.go` into a long tutorial. That belongs in `docs/{PACKAGE}.md`.

### 8.2. Exported GoDoc

Every exported symbol must have GoDoc that explains actual behavior. For complex symbols, include:

* what it does;
* why it exists;
* inputs and validation;
* return ownership;
* errors;
* determinism;
* complexity;
* edge cases;
* AI-Hints only when they prevent a real misuse pattern.

### 8.3. Repository docs

Each fundamental `docs/*.md` file should include a repository-comment header with:

* file/package reference;
* purpose;
* contract status;
* scope;
* license.

Repository docs should teach the real package using formulas, diagrams, pseudocode, pitfalls, examples, and practical recipes. They must not invent future APIs or stronger guarantees than the source provides.

### 8.4. Diagrams and formulas

Use ASCII, Mermaid, or formulas only when they clarify real semantics.

Good diagram goals:

* graph capability visualization;
* BFS/DFS order semantics;
* Dijkstra `+Inf` state distinctions;
* matrix row-major layout;
* residual network updates;
* benchmark topology shapes.

Decorative diagrams are noise and should be removed.

---

## 9. Testing standards

Tests must protect contracts, not merely execute lines.

### 9.1. Minimum test grid

Every non-trivial package should cover:

| Group          | Purpose                                            | Examples                                                                        |
|:---------------|:---------------------------------------------------|:--------------------------------------------------------------------------------|
| Validation     | Fail-fast input and policy errors.                 | nil graph, bad option, invalid weight, missing start.                           |
| Medium         | Real algorithm behavior on non-trivial structures. | weighted routing, cycle witness, dense algebra, max-flow network.               |
| Special        | Edge cases and regression traps.                   | zero-shape matrix, zero-weight edge, cancellation, mixed edges, partial result. |
| Determinism    | Stable order/tie behavior.                         | equal-cost Dijkstra witnesses, DFS cycle canonicalization, sorted components.   |
| Error protocol | Sentinel classification.                           | `errors.Is`, wrapped errors, joined/double-wrapped causes.                      |

### 9.2. Test helper policy

Use package-local helpers in `test_helpers_test.go` for repeated checks.

Useful helper families:

* `mustErrorIs`;
* numeric approximate comparisons;
* path/order comparison;
* matrix shape/cell assertions;
* nil-result behavior;
* cycle/witness canonical checks;
* deterministic map-domain checks.

Avoid `reflect.DeepEqual` when a domain-specific helper would produce clearer contract failures.

External assertion frameworks such as `testify` are forbidden.

### 9.3. Error tests

Good:

```go
if !errors.Is(err, matrix.ErrDimensionMismatch) {
	t.Fatalf("got %v, want ErrDimensionMismatch", err)
}
```

Forbidden:

```go
if !strings.Contains(err.Error(), "dimension") {
	t.Fatal("bad error")
}
```

### 9.4. Regression tests

Every bug fix needs an explicit regression anchor.

Name tests so the protected contract is visible:

```go
func TestBuildDenseAdjacency_PreservesMixedZeroWeightEdge(t *testing.T) { ... }
func TestDijkstra_EqualCostPathDoesNotOverwritePredecessor(t *testing.T) { ... }
func TestBFS_CancelReturnsPartialVisitedButStableOrder(t *testing.T) { ... }
```

### 9.5. Concurrent tests

Never call `t.Fatal`, `t.FailNow`, or similar methods from a non-owner goroutine. Use channels or deterministic aggregation.

---

## 10. Benchmark standards

Benchmarks must measure a real algorithmic regime, not setup noise.

### 10.1. Required benchmark shape

```go
func BenchmarkDinic_DenseLayeredNetwork(b *testing.B) {
	g := buildDenseLayeredNetwork(b)
	opts := flow.DefaultOptions()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := flow.Dinic(g, "S", "T", opts)
		if err != nil {
			b.Fatal(err)
		}
	}
}
```

### 10.2. Benchmark laws

* Build data before `b.ResetTimer()`.
* Check setup errors.
* Use fixed seeds for random data.
* Name the topology or regime being measured.
* Use `b.ReportAllocs()` for core and algorithmic packages.
* Do not print or allocate scaffolding inside the hot loop.
* Do not silently shrink benchmark shape because graph policy rejected inserts.

### 10.3. Good benchmark regimes

* sparse chain traversal;
* dense local competition;
* mixed-edge endpoint resolution;
* path-tracking overhead;
* cutoff/wall policy overhead;
* full traversal / forest restart;
* matrix fast-path vs fallback equivalence;
* residual graph pressure in flow;
* DTW full matrix vs rolling rows.

---

## 11. Example standards

Examples must compile, run, and teach real API usage.

### 11.1. Package examples

`example_test.go` should provide stable, deterministic examples for package-level contracts.

Default expectation for non-trivial algorithm packages:

* at least five meaningful examples;
* practical scenario names;
* `build -> algorithm -> consume` pipeline;
* deterministic `// Output:`;
* no fake future APIs;
* no unstable map printing;
* no time-flaky cancellation.

### 11.2. Scenario examples

`examples/*.go` may be larger and more story-driven. They should demonstrate package composition and real operational value.

Good examples explain:

* what the domain story is;
* why each public method is called;
* what output validates;
* what complexity regime applies;
* what not to copy into production shortcuts.

### 11.3. Static example data

For predetermined examples, it is acceptable to use:

```go
_, _ = g.AddEdge("A", "B", 1)
```

Only when a nearby comment explains that the shortcut is safe for static tutorial data and should not be copied into production without error checks.

---

## 12. Style and implementation rules

### 12.1. General Go style

* Use idiomatic Go names.
* Keep public names clear and stable.
* Prefer small helpers only when they clarify real structure.
* Avoid cleverness that hides the algorithm.
* Keep hot loops allocation-aware.
* Use `gofmt` and `go vet`.

### 12.2. Public names

| Concept                    | Naming                                                         |
|:---------------------------|:---------------------------------------------------------------|
| Constructor                | `NewXxx`                                                       |
| Option                     | `WithXxx`, `WithAllowXxx`, `WithNoXxx`                         |
| Error                      | `ErrXxx`                                                       |
| Result                     | `XxxResult`                                                    |
| Canonical algorithm facade | package-specific clear name, e.g. `Dijkstra`                   |
| Wrappers                   | explicit projection names, e.g. `DistanceTo`, `ShortestPathTo` |

### 12.3. No magic values

Name values that affect contract behavior:

```go
const defaultTolerance = 1e-10
const rootDepth = 0
const noDepthLimit = -1
```

Magic values hidden inside algorithm stages are not acceptable when they affect semantics.

### 12.4. Options

Options should be centrally assembled and validated before heavy allocation or kernel execution.

Invalid option input should return an error, not panic.

---

## 13. Package-specific reminders

### `core`

* Preserve deterministic graph surfaces.
* Do not weaken capability checks.
* Document lock/order behavior for topology changes.
* Be careful with metadata aliasing and clone semantics.

### `bfs`

* BFS is unweighted hop distance.
* Preserve visit/dequeue semantics.
* Partial results are intentional and must remain documented/tested.
* Avoid queue retention patterns such as `queue = queue[1:]`.

### `dfs`

* Returned `Order` is finish order.
* Partial `Order` must not be exposed as if complete.
* Cycle detection returns witnesses, not exhaustive simple-cycle enumeration unless explicitly documented.

### `dijkstra`

* Reject finite negative weights, `NaN`, and `-Inf` according to package policy.
* Preserve unknown target vs known unreachable target.
* Do not overwrite predecessor state on equal-cost candidates.
* Do not simplify mixed/undirected traversal to `edge.To`.

### `matrix`

* Preserve row-major shape laws.
* Treat zero-shape matrices as valid structural results.
* Do not confuse finite zero edge weight with absence.
* Do not export metric closure as original topology.
* Fast paths must match fallback behavior and error classification.

### `flow`

* Preserve residual capacity semantics.
* Rebuild or clone network state when a new independent run requires a clean residual model.
* Keep capacity validation and epsilon behavior explicit.

### `dtw`

* Keep memory mode and path recovery compatibility explicit.
* Full path recovery requires a mode that stores enough backtracking information.
* Window constraints must not silently claim valid alignment when no path exists.

### `tsp`

* State metric assumptions clearly.
* Distinguish exact small-instance methods from heuristics/approximations.
* Do not imply approximation guarantees outside their required conditions.

### `builder` and `gridgraph`

* Fixtures must be reproducible.
* Randomness requires fixed seeds.
* Generated topology should be described by shape, not by accidental output.

---

## 14. Pull request checklist

Before requesting review, verify:

### Contract

- [ ] The mathematical objective is unchanged or explicitly documented.
- [ ] Determinism source and tie-breaks remain valid.
- [ ] Sentinel error classification is preserved.
- [ ] Result ownership/partial-result behavior is documented.
- [ ] No hidden fallback or policy change was introduced.

### Code

- [ ] `gofmt -w .` run.
- [ ] `go vet ./...` passes.
- [ ] `go test ./...` passes.
- [ ] `golangci-lint run` passes.
- [ ] No external assertion framework introduced.
- [ ] No unnecessary dependency introduced.

### Tests

- [ ] Validation tests added/updated.
- [ ] Medium mathematical tests added/updated.
- [ ] Special/edge tests added/updated.
- [ ] Error checks use `errors.Is`.
- [ ] Regression test added for each bug fix.
- [ ] Fast-path and fallback equivalence tested when applicable.

### Documentation

- [ ] `doc.go` synchronized.
- [ ] Exported GoDoc synchronized.
- [ ] `docs/{PACKAGE}.md` synchronized.
- [ ] Examples compile and reflect current API.
- [ ] README/Tutorial updated if package story changed.

### Benchmarks

- [ ] Benchmark setup is outside timed loop.
- [ ] `b.ReportAllocs()` used where appropriate.
- [ ] Benchmark name describes a real regime.
- [ ] Setup errors are checked.

---

## 15. Reporting issues

A high-quality issue should include:

* package name;
* version or commit;
* Go version and OS;
* minimal reproduction;
* expected behavior;
* actual behavior;
* whether the issue is about correctness, determinism, documentation drift, performance, or API ergonomics;
* logs/output if relevant;
* links to affected docs or examples if this is a synchronization issue.

For algorithmic bugs, include a small graph or matrix whenever possible.

For determinism bugs, include two observed outputs from identical inputs.

For documentation bugs, state which file contradicts which source/API behavior.

---

## 16. Review philosophy

Review should be strict, technical, and respectful.

Reviewers should ask:

* Does this preserve the real mathematical contract?
* Is the public surface synchronized?
* Are failure classes branchable through sentinels?
* Is determinism explicit and tested?
* Are examples honest and useful?
* Are benchmarks measuring the claimed regime?
* Does this reduce ambiguity or create it?

A PR may be rejected even when tests pass if it weakens contract clarity, hides policy, or makes future behavior harder to reason about.

---

## 17. Licensing

By contributing, you agree that your contribution will be licensed under the project license: **AGPL-3.0-only**.

If your contribution contains material derived from another project, disclose it clearly and ensure the license is compatible before opening the PR.

---

## 18. Contact and support

* Issues: `github.com/lvlath/go/issues`
* Discussions: use GitHub Discussions when enabled.
* Maintainer contact: `katalvlaran@gmail.com`

---

## Project stewardship

lvlath/go is currently stewarded by @katalvlaran.

Changes to exported API, deterministic behavior, numeric semantics,
compatibility, or licensing require prior design discussion.

**lvlath contribution rule:** make the library more correct, more deterministic, more explainable, or more trustworthy. Everything else is noise.

