<!--
  lvlath - Repository Documentation

  Purpose:
    This document is the repository-level specification, tutorial, and contract map
    for lvlath/matrix. It explains the deterministic dense matrix model, row-major
    memory layout, numeric policy, graph-to-matrix adapters, metric-closure workflow,
    zero-shape law, statistical transforms, and operational patterns that downstream
    graph and math algorithms rely on.

  Contract status:
    - Matrix shape, zero-shape, and row-major storage rules described here are part
      of the matrix contract.
    - Deterministic loop-order and graph-adapter publication rules described here
      are part of the matrix contract.
    - Numeric policy rules for NaN, -Inf, and +Inf described here are part of the
      matrix contract.
    - Adjacency, incidence, zero-weight, and metric-closure semantics described here
      are part of the matrix contract.
    - Sentinel-error semantics described here are part of the matrix contract.
    - Any incompatible change must be explicit, documented, and versioned.

  Scope:
    - Deterministic row-major dense matrices and no-copy views.
    - Graph-to-matrix and matrix-to-graph transformations.
    - Dense algebra facades, APSP/metric closure, sanitization, comparison, and statistics.
    - Zero-shape behavior for structural matrix pipelines.
    - Allocation-aware patterns for reproducible computation.

  License:
    The lvlath repository is licensed under AGPL-3.0-only. See LICENSE.
-->

# 10. Matrices: Computational Engine & Graph Algebra

> **Package:** `lvlath/matrix` | **Focus:** Determinism, Cache Locality, Numeric Discipline, Graph-Semantics Fidelity

The `lvlath/matrix` package is the deterministic dense-algebra layer of `lvlath`. It turns graph topology into strict numeric artifacts, runs matrix kernels with reproducible traversal order, and preserves edge-case semantics that casual matrix helpers usually lose: zero-weight edges, empty shapes, incidence signs, `+Inf` unreachable sentinels, and caller-owned results.

This is not a bag of `[][]float64` snippets. It is a contract-driven matrix subsystem for algorithms that must be debuggable, deterministic, and mathematically explicit.

---

## 10.1. What & Why: The Architectural Paradigm

Why use `lvlath/matrix` instead of generic multidimensional slices, mutation-heavy math objects, or ad-hoc graph exporters?

### The Five Laws of Matrix

1. **Pure Functional Determinism**
    *   **The Problem:** Many libraries hide mutation behind method calls (`A.Mul(B)` mutates `A`) or rely on unstable map iteration while converting graph topology into numeric arrays.
    *   **The lvlath Way:** Public algebra facades such as `Product`, `Sum`, `Diff`, `T`, and `ScaleBy` publish new result artifacts instead of mutating their operands. Builders scan vertices and edges in stable order, and Floyd-Warshall updates only on strict improvement (`cand < current`) so equal-cost alternatives never create accidental tie drift.

2. **Hardware Sympathy Without API Leaks**
    *   **The Problem:** `[][]float64` scatters rows across the heap, increasing allocation count and destroying cache locality.
    *   **The lvlath Way:** `Dense` stores an entire matrix in one row-major `[]float64`. Hot kernels can unwrap `*Dense` and scan the flat buffer, while callers still use the safe `Matrix` interface.

3. **Safe Numeric Discipline**
    *   **The Problem:** Dirty floats (`NaN`, `+Inf`, `-Inf`) often leak into a pipeline and silently poison every downstream statistic or shortest-path result.
    *   **The lvlath Way:** `Dense` validates scalar writes through one gate. By default it rejects `NaN` and infinities. `+Inf` becomes legal only under an explicit distance/absence policy, where it means “no path” or “no edge”, never “unknown dirty value”.

4. **Graph-Semantics Fidelity**
    *   **The Problem:** Dense adjacency has one cell per vertex pair, so it can easily erase parallel edges, confuse zero-weight edges with absence, or export shortest-path distances as if they were literal graph edges.
    *   **The lvlath Way:** Adapters document and enforce their compression law: first-edge-wins or last-write-wins for dense adjacency, signed edge columns for incidence, `+Inf` absence for zero-weight preservation, and explicit refusal to export metric-closure matrices.

5. **Zero-Shape and Ownership Correctness**
    *   **The Problem:** Many libraries treat empty matrices as errors even when they are valid structural results, or return views that look like copies.
    *   **The lvlath Way:** `0×0`, `0×N`, and `N×0` are legal. `Clone` and `Induced` produce independent storage; `View` intentionally shares storage. Statistical metadata stays shape-correct even when there are no elements to process.

---

## 10.2. Architecture: Memory Model & Fast-Paths

To write high-performance matrix algorithms on top of `lvlath/matrix`, you must understand the storage law. A `Dense` matrix of size `R×C` is backed by a single `make([]float64, R*C)` allocation. Elements of each row are contiguous.

$$ \text{offset}(i, j) = i \times C + j $$

where `C = Cols()`.

> [!NOTE]
> **Why this matters:** Row-major storage turns matrix scans into predictable sequential memory reads. This is the difference between a cache-friendly kernel and a heap-fragmented collection of row slices.

### 10.2.1. Visualization A: Logical Matrix to Physical Slice

```text
      Logical 2D Matrix (3x3)                     Physical 1D Flat Slice (len=9)
      ┌────────┬────────┬────────┐
Row 0 │ A[0,0] │ A[0,1] │ A[0,2] │ ─────┐
      ├────────┼────────┼────────┤      │   ┌──── Row 0 ───┐  ┌──── Row 1 ───┐  ┌──── Row 2 ───┐
Row 1 │ A[1,0] │ A[1,1] │ A[1,2] │ ───┐ └─►[ 0,0 | 0,1 | 0,2 | 1,0 | 1,1 | 1,2 | 2,0 | 2,1 | 2,2 ]
      ├────────┼────────┼────────┤    │                               ▲                 ▲
Row 2 │ A[2,0] │ A[2,1] │ A[2,2] │ ─┐ └───────────────────────────────┘                 │
      └────────┴────────┴────────┘  └───────────────────────────────────────────────────┘
                                        contiguous rows maximize CPU prefetching
```

### 10.2.2. Visualization B: Zero-Shape Matrices

Zero-shape matrices are structural matrices with no addressable cells. They are not failed allocations.

```text
Dense(0,4)                                                Dense(4,0)
┌ no rows ┐                                               ┌──── empty row 0 ────┐
│         │  cols = 4                                     ├──── empty row 1 ────┤  cols = 0
└─────────┘  data = nil                                   ├──── empty row 2 ────┤  data = nil
             CenterColumns -> means len 4: [0 0 0 0]      ├──── empty row 3 ────┤  CenterRows -> means len 4
             At(0,0) -> ErrOutOfRange                     └────────────────────┘  NormalizeRowsL1 -> norms len 4
```

### 10.2.3. The Hardware-Sympathy Rules

1. **Cache locality:** Row scans read contiguous floats.
2. **Allocation efficiency:** `Dense` needs one flat allocation for non-zero area.
3. **Fast-path unwrapping:** Kernels may type-assert `Matrix` to `*Dense` and run flat loops.
4. **Semantic equivalence:** Fast-path and generic fallback must produce the same result contract.

---

## 10.3. Core API Surface & Math Formulation

### 10.3.1. Matrix Interface

```go
type Matrix interface {
	Rows() int
	Cols() int
	At(i, j int) (float64, error)
	Set(i, j int, v float64) error
	Clone() Matrix
}
```

Contract:

* `Rows` and `Cols` are `O(1)`;
* `At` returns `ErrOutOfRange` on invalid coordinates;
* `Set` returns `ErrOutOfRange` or numeric-policy errors;
* `Clone` returns an independent matrix.

### 10.3.2. Dense Constructors and Ownership

```go
func NewDense(rows, cols int) (*Dense, error)
func NewPreparedDense(rows, cols int, opts ...Option) (*Dense, error)

func (m *Dense) Fill(data []float64) error
func (m *Dense) Clone() Matrix
func (m *Dense) Induced(rowsIdx, colsIdx []int) (*Dense, error)
func (m *Dense) View(r0, c0, rows, cols int) (*View, error)
func (m *Dense) Apply(f func(i, j int, v float64) float64) error
func (m *Dense) Do(f func(i, j int, v float64) bool)
```

`Clone` and `Induced` own new buffers. `View` and `View` share base storage. `Apply` mutates in-place and is intentionally not all-or-nothing: earlier accepted writes remain if a later value violates policy.

### 10.3.3. Public Facades

| Group                  | Facades                                                                                          |
|:-----------------------|:-------------------------------------------------------------------------------------------------|
| Construction           | `NewZeros`, `NewIdentity`, `ZerosLike`, `IdentityLike`, `CloneMatrix`                            |
| Algebra                | `Sum`, `Diff`, `Product`, `HadamardProd`, `T`, `ScaleBy`, `MatVecMul`                            |
| Factorization          | `LUDecompose`, `QRDecompose`, `InverseOf`, `EigenSym`                                            |
| Graph adapters         | `BuildAdjacency`, `GraphFromAdjacency`, `AdjacencyToGraph`, `BuildMetricClosure`, `DegreeVector` |
| APSP                   | `APSPInPlace`, `MetricClosure`                                                                   |
| Sanitization / compare | `Clip`, `ReplaceInfNaN`, `AllClose`                                                              |
| Statistics             | `CenterColumns`, `CenterRows`, `NormalizeRowsL1`, `NormalizeRowsL2`, `Covariance`, `Correlation` |

### 10.3.4. Math Formulation Summary

The public API is intentionally thin: facades delegate to canonical kernels and preserve their validation, loop order, and allocation model.

| Facade API           | Operation             | Complexity             | Formulation & Notes                                                        |
|:---------------------|:----------------------|:-----------------------|:---------------------------------------------------------------------------|
| `Sum(A, B)`          | Matrix sum            | $O(R \cdot C)$         | $$ C_{ij} = A_{ij} + B_{ij} $$                                             |
| `Diff(A, B)`         | Matrix difference     | $O(R \cdot C)$         | $$ C_{ij} = A_{ij} - B_{ij} $$                                             |
| `HadamardProd(A, B)` | Element-wise product  | $O(R \cdot C)$         | $$ C_{ij} = A_{ij}B_{ij} $$                                                |
| `T(A)`               | Transposition         | $O(R \cdot C)$         | $$ C_{ij} = A_{ji} $$                                                      |
| `Product(A, B)`      | Matrix multiplication | $O(R \cdot N \cdot C)$ | $$ C_{ij} = \sum_k A_{ik}B_{kj} $$                                         |
| `ScaleBy(A, α)`      | Scalar scale          | $O(R \cdot C)$         | $$ C_{ij} = \alpha A_{ij} $$                                               |
| `MatVecMul(A, x)`    | Matrix-vector product | $O(R \cdot C)$         | $$ y_i = \sum_j A_{ij}x_j $$                                               |
| `LUDecompose(A)`     | LU factorization      | $O(N^3)$               | Deterministic LU facade; callers must handle singularity errors.           |
| `InverseOf(A)`       | Matrix inverse        | $O(N^3)$               | Delegates to the inverse kernel; singular matrices return sentinel errors. |
| `EigenSym(A,tol,k)`  | Symmetric eigensolver | $O(k \cdot N^3)$       | Delegates to deterministic Jacobi-style symmetric eigen decomposition.     |
| `APSPInPlace(D)`     | Floyd-Warshall        | $O(N^3)$               | $$ D_{ij} \leftarrow \min(D_{ij}, D_{ik}+D_{kj}) $$                        |
| `Covariance(X)`      | Sample covariance     | $O(R \cdot C^2)$       | $$ Cov = \frac{(X^c)^T X^c}{R-1} $$                                        |
| `Correlation(X)`     | Pearson correlation   | $O(R \cdot C^2)$       | $$ Corr = \frac{Z^T Z}{R-1} $$                                             |

> [!NOTE]
> **On determinants:** `lvlath/matrix` does not expose a determinant facade in the provided API. Use factorization-level workflows when you need structural information about singularity or invertibility.

---

## 10.4. Numeric Policy, Options & Configuration Visualized

Options are assembled left-to-right and finalized once by `NewMatrixOptions` / internal option gathering. They are runtime policy objects; they do not mutate graph topology.

```go
func NewMatrixOptions(opts ...Option) (Options, error)

func WithDirected() Option
func WithUndirected() Option
func WithWeighted() Option
func WithUnweighted() Option
func WithAllowMulti() Option
func WithDisallowMulti() Option
func WithAllowLoops() Option
func WithDisallowLoops() Option
func WithMetricClosure() Option

func WithValidateNaNInf() Option
func WithNoValidateNaNInf() Option
func WithAllowInfDistances() Option
func WithDisallowInfDistances() Option
func WithPreserveZeroWeights() Option
func WithAutoZeroWeights() Option
func WithEdgeThreshold(t float64) Option
func WithKeepWeights() Option
func WithBinaryWeights() Option
```

### 10.4.1. Numeric Admissibility

| Value        | Default Dense | `+Inf`-policy Dense |
|:-------------|:-------------:|:-------------------:|
| finite value |   accepted    |      accepted       |
| `NaN`        |   rejected    |      rejected       |
| `-Inf`       |   rejected    |      rejected       |
| `+Inf`       |   rejected    |      accepted       |

Finalization rules:

* `WithMetricClosure` forces `allowInfDistances=true`;
* `WithUnweighted` clears zero-weight preservation;
* `WithBinaryWeights` and `WithKeepWeights` are mutually exclusive;
* nil option slots return `ErrNilOption`.

### 10.4.2. Configuration Options: ON vs OFF

#### Directed vs Undirected

```text
Graph edge: A --5--> B

WithDirected                         WithUndirected
        A   B                                A   B
A     [ 0   5 ]                      A     [ 0   5 ]
B     [ 0   0 ]                      B     [ 5   0 ]

Directed writes one ordered cell. Undirected mirrors non-loop edges.
```

#### AllowMulti vs DisallowMulti

```text
Input edges in stable order:
  e1: A -> B weight 2
  e2: A -> B weight 8

WithAllowMulti                       WithDisallowMulti
        A   B                                A   B
A     [ 0   8 ]                      A     [ 0   2 ]
B     [ 0   0 ]                      B     [ 0   0 ]

Dense adjacency has one cell. AllowMulti means deterministic last-write-wins.
DisallowMulti means deterministic first-edge-wins.
```

#### AllowLoops vs DisallowLoops

```text
Input edge: A -> A weight 3

WithAllowLoops                       WithDisallowLoops
        A                                   A
A     [ 3 ]                         A     [ 0 ]

For +Inf-no-edge weighted adjacency, a missing loop remains +Inf in ordinary adjacency.
Metric closure normalizes distance diagonals later.
```

#### Weighted vs Unweighted

```text
Input edge: A -> B weight 42.5

WithWeighted                         WithUnweighted
        A     B                              A   B
A     [ 0   42.5 ]                   A     [ 0   1 ]
B     [ 0    0   ]                   B     [ 0   0 ]
```

#### PreserveZeroWeights

```text
Input edges:
  A -> B = 4
  A -> C = 0   real edge

Default weighted mixed profile selects +Inf absence automatically:
        A     B     C
A     [Inf    4     0]
B     [Inf   Inf   Inf]
C     [Inf   Inf   Inf]

All-zero weighted graphs need WithPreserveZeroWeights to avoid binary auto-degrade.
```

#### MetricClosure

```text
Normal adjacency                     Metric closure distance matrix
        A   B   C                            A      B      C
A     [ 0   5   0 ]                  A     [ 0      5     Inf ]
B     [ 0   0   2 ]                  B     [ Inf    0      2  ]
C     [ 0   0   0 ]                  C     [ Inf   Inf     0  ]

Metric closure is not adjacency anymore. ToGraph refuses it.
```

---

## 10.5. Graph-to-Matrix Adapters

Graph adapters convert string vertex IDs into integer coordinates. The low-level builders preserve the caller-provided vertex order; constructors that consume `core.Graph` rely on the deterministic order returned by `core`, while internal test helpers may defensively sort when canonical layouts are needed.

### 10.5.0. Conversion Pipeline

```text
 [core.Graph]              Stable Vertex Order               [matrix.AdjacencyMatrix]
 String IDs                Integer Mapping                   Dense Tensor Coordinates

  "Fiona"   ──────────►    0: "Alice"  ───────────────────► Row 0 / Col 0
  "Bob"     ──────────►    1: "Bob"    ───────────────────► Row 1 / Col 1
  "Alice"   ──────────►    2: "Fiona"  ───────────────────► Row 2 / Col 2

The adapter does not store vertex IDs inside cells. It stores numeric cells plus metadata.
```

### 10.5.1. Social Network Scenario

Consider a social platform where users send weighted directed messages.

```text
  [User Network Example]

           Fiona
       ↙︎    ↓    ↖
      ↓   ↗ Bob ↘︎  ↑
    Alice    ↓    Carol
      ↓     Eva     ↓
       ↘︎    ↑    ↙︎
           Dave
```

Edges:

* `Alice -> Bob = 5`
* `Alice -> Dave = 3`
* `Bob -> Carol = 2`
* `Bob -> Eva = 1`
* `Carol -> Dave = 4`
* `Carol -> Fiona = 6`
* `Dave -> Eva = 3`
* `Fiona -> Alice = 4`
* `Fiona -> Bob = 2`

### 10.5.2. Adjacency Matrix

Adjacency is a `V×V` matrix. It gives `O(1)` cell lookup and `O(V)` row scans for neighbors.

$$ A_{ij} =
\begin{cases}
w_{ij}, & (i,j) \in E \\
0, & \text{absent under classic encoding} \\
+\infty, & \text{absent under zero-preserving weighted encoding}
\end{cases} $$

#### Example: Weighted Adjacency (6×6)

|           | Alice | Bob | Carol | Dave | Eva | Fiona |
|-----------|:-----:|:---:|:-----:|:----:|:---:|:-----:|
| **Alice** | 0 | 5 | 0 | 3 | 0 | 0 |
| **Bob**   | 0 | 0 | 2 | 0 | 1 | 0 |
| **Carol** | 0 | 0 | 0 | 4 | 0 | 6 |
| **Dave**  | 0 | 0 | 0 | 0 | 3 | 0 |
| **Eva**   | 0 | 0 | 0 | 0 | 0 | 0 |
| **Fiona** | 4 | 2 | 0 | 0 | 0 | 0 |

Complexity:

* build: $O(V^2 + E)$ time, $O(V^2)$ space;
* lookup: $O(1)$;
* neighbors: $O(V)$ row scan.

### 10.5.3. Incidence Matrix

Incidence is a `V×E'` matrix where columns are effective edges after loop/multi filtering. It is the correct representation when edge identity matters.

$$ B_{u,k} = -1,\quad B_{v,k}=+1 \quad \text{for directed } u \to v $$

$$ B_{u,k} = +1,\quad B_{v,k}=+1 \quad \text{for undirected } u-v $$

$$ B_{u,k}=+2 \quad \text{for undirected self-loop } u-u $$

Directed self-loops are skipped because `-1` and `+1` cancel in the same row.

#### Example: Social Network Incidence (6×9)

Labeling edges $e_1$ to $e_9$ in the listed directed order:

|         | e₁ | e₂ | e₃ | e₄ | e₅ | e₆ | e₇ | e₈ | e₉ |
|---------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Alice   | -1 | 0  | -1 | 0  | 0  | 0  | 0  | 0  | +1 |
| Bob     | +1 | -1 | 0  | -1 | 0  | 0  | +1 | 0  | 0  |
| Carol   | 0  | +1 | 0  | 0  | -1 | -1 | 0  | 0  | 0  |
| Dave    | 0  | 0  | +1 | 0  | +1 | 0  | 0  | -1 | 0  |
| Eva     | 0  | 0  | 0  | +1 | 0  | 0  | 0  | +1 | 0  |
| Fiona   | 0  | 0  | 0  | 0  | 0  | +1 | -1 | 0  | -1 |

Complexity:

* build: $O(V + E)$ metadata/filtering plus $O(V \cdot E')$ dense storage;
* vertex incidence row copy: $O(E')`;
* endpoint lookup: $O(1)`.

---

## 10.6. Metric Closure & Floyd-Warshall

Metric closure converts direct adjacency into all-pairs shortest-path distances.

Distance matrix contract:

$$ D_{ii} = 0 $$

$$ D_{ij}=+\infty \quad \text{means no path from } i \text{ to } j $$

Relaxation law:

$$ D_{ij} \leftarrow \min(D_{ij}, D_{ik}+D_{kj}) $$

The implementation writes only on strict improvement:

$$ D_{ik}+D_{kj} < D_{ij} $$

Equal-cost alternatives do not overwrite existing values.

```text
Routing graph:

A ──5──▶ B ──2──▶ C
│        │        │
│        └─10──▶ D
└─12──▶ C ──2──▶ D

Initial distances:
        A     B     C     D
A       0     5    12    Inf
B      Inf    0     2     10
C      Inf   Inf    0      2
D      Inf   Inf   Inf     0

After APSP:
        A     B     C     D
A       0     5     7      9
B      Inf    0     2      4
C      Inf   Inf    0      2
D      Inf   Inf   Inf     0
```

Negative-cycle detection scans the final diagonal:

$$ \exists i: D_{ii} < -\epsilon \Rightarrow ErrNegativeCycle $$

`ToGraph` on metric closure returns `ErrMatrixNotImplemented`, because shortest-path distances are derived facts, not original edges.

---

## 10.7. Statistical Transforms & Zero-Shape Law

### 10.7.1. Centering

Column mean and centering:

$$ \mu_j = \frac{1}{r}\sum_{i=0}^{r-1} X_{ij} $$

$$ X^c_{ij}=X_{ij}-\mu_j $$

Row mean and centering:

$$ \rho_i = \frac{1}{c}\sum_{j=0}^{c-1} X_{ij} $$

$$ X^r_{ij}=X_{ij}-\rho_i $$

```text
X = 3×2 sensor readings

        temp  load
row0     10    2
row1     14    6
row2     16    7

Column means:
        temp = 13.333..., load = 5

CenterColumns subtracts those means from each column.
```

Zero-shape law:

* `0×N`: column means length `N`, all zero;
* `N×0`: row means length `N`, all zero;
* returned matrix is the original structural input because there are no elements to transform.

### 10.7.2. Row Normalization

L1 norm:

$$ \|x_i\|_1 = \sum_j |X_{ij}| $$

L2 norm:

$$ \|x_i\|_2 = \sqrt{\sum_j X_{ij}^2} $$

Scale law:

$$ s_i =
\begin{cases}
1/\|x_i\|, & \|x_i\| > 0 \\
1, & \|x_i\| = 0
\end{cases} $$

Degenerate rows remain unchanged.

```text
Traffic distribution row:
  [ 2, 3, 5 ]

L1 norm = 10
NormalizeRowsL1 -> [ 0.2, 0.3, 0.5 ]

Degenerate row:
  [ 0, 0, 0 ]
NormalizeRowsL1 -> unchanged [ 0, 0, 0 ]
```

### 10.7.3. Covariance and Correlation

Sample covariance:

$$ Cov = \frac{(X^c)^T X^c}{r-1} $$

Sample standard deviation:

$$ std_j = \sqrt{\frac{\sum_i (X^c_{ij})^2}{r-1}} $$

Pearson correlation:

$$ Corr = \frac{Z^T Z}{r-1} $$

Degenerate columns use inverse scale `0`, so their correlation row/column becomes zero.

Shape law:

* if `cols == 0`, covariance and correlation return valid `0×0`;
* if `cols > 0`, sample statistics require `rows >= 2`.

---

## 10.8. Go Usage Examples

### 10.8.1. Social Network Matrix Workflow

This example demonstrates the updated contracts: zero-weight preservation, loop policy, adjacency neighbors, incidence shape, metric-closure export refusal, and zero-feature covariance.

```go
package main

import (
	"errors"
	"fmt"
	"log"
	"math"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

func main() {
	g, err := core.NewGraph(core.WithDirected(true), core.WithWeighted(), core.WithLoops())
	if err != nil {
		log.Fatalf("new graph: %v", err)
	}

	// Example setup only: construction errors are intentionally ignored below to keep
	// the scenario readable. Do not repeat this shortcut in production code.
	_, _ = g.AddEdge("Alice", "Bob", 5)
	_, _ = g.AddEdge("Alice", "Dave", 3)
	_, _ = g.AddEdge("Bob", "Carol", 2)
	_, _ = g.AddEdge("Bob", "Eva", 1)
	_, _ = g.AddEdge("Carol", "Dave", 4)
	_, _ = g.AddEdge("Carol", "Fiona", 6)
	_, _ = g.AddEdge("Dave", "Eva", 3)
	_, _ = g.AddEdge("Fiona", "Alice", 4)
	_, _ = g.AddEdge("Fiona", "Bob", 2)
	_, _ = g.AddEdge("Grace", "Heidi", 0) // real zero-weight edge
	_, _ = g.AddEdge("Heidi", "Heidi", 0) // directed loop kept by adjacency when loops are allowed

	opts, err := matrix.NewMatrixOptions(
		matrix.WithDirected(),
		matrix.WithWeighted(),
		matrix.WithAllowLoops(),
		matrix.WithPreserveZeroWeights(),
	)
	if err != nil {
		log.Fatalf("matrix options: %v", err)
	}

	adj, err := matrix.NewAdjacencyMatrix(g, opts)
	if err != nil {
		log.Fatalf("adjacency: %v", err)
	}

	grace := adj.VertexIndex["Grace"]
	heidi := adj.VertexIndex["Heidi"]
	alice := adj.VertexIndex["Alice"]

	zeroEdge, err := adj.Mat.At(grace, heidi)
	if err != nil {
		log.Fatalf("read Grace->Heidi: %v", err)
	}
	absent, err := adj.Mat.At(grace, alice)
	if err != nil {
		log.Fatalf("read Grace->Alice: %v", err)
	}

	neighbors, err := adj.Neighbors("Grace")
	if err != nil {
		log.Fatalf("neighbors: %v", err)
	}

	inc, err := matrix.NewIncidenceMatrix(g, opts)
	if err != nil {
		log.Fatalf("incidence: %v", err)
	}
	erows, err := inc.VertexCount()
	if err != nil {
		log.Fatalf("incidence rows: %v", err)
	}
	ecols, err := inc.EdgeCount()
	if err != nil {
		log.Fatalf("incidence cols: %v", err)
	}

	metricOpts, err := matrix.NewMatrixOptions(
		matrix.WithDirected(),
		matrix.WithWeighted(),
		matrix.WithAllowLoops(),
		matrix.WithPreserveZeroWeights(),
		matrix.WithMetricClosure(),
	)
	if err != nil {
		log.Fatalf("metric options: %v", err)
	}
	closure, err := matrix.NewAdjacencyMatrix(g, metricOpts)
	if err != nil {
		log.Fatalf("metric closure: %v", err)
	}
	_, exportErr := closure.ToGraph()

	zeroFeatures, err := matrix.NewZeros(8, 0)
	if err != nil {
		log.Fatalf("zero features: %v", err)
	}
	cov, means, err := matrix.Covariance(zeroFeatures)
	if err != nil {
		log.Fatalf("covariance: %v", err)
	}

	fmt.Printf("Grace->Heidi weight: %.1f\n", zeroEdge)
	fmt.Printf("Grace->Alice absent: %t\n", math.IsInf(absent, 1))
	fmt.Printf("Grace neighbors: %v\n", neighbors)
	fmt.Printf("incidence shape: %dx%d\n", erows, ecols)
	fmt.Printf("metric export refused: %t\n", errors.Is(exportErr, matrix.ErrMatrixNotImplemented))
	fmt.Printf("zero-feature covariance: %dx%d means=%d\n", cov.Rows(), cov.Cols(), len(means))

	// Output:
	// Grace->Heidi weight: 0.0
	// Grace->Alice absent: true
	// Grace neighbors: [Heidi]
	// incidence shape: 8x10
	// metric export refused: true
	// zero-feature covariance: 0x0 means=0
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-Matrix_Social_Network-blue?logo=go)](https://go.dev/play/p/gp2o_7vl-YK)

### 10.8.2. Operations Lab: LU, APSP, DegreeVector, Normalization, Correlation

This scenario combines algebraic and graph workflows: a small calibration solve, an in-place Floyd-Warshall distance matrix, a graph-derived degree vector, normalized routing shares, and feature correlation.

```go
package main

import (
	"fmt"
	"log"
	"math"

	"github.com/lvlath/go/core"
	"github.com/lvlath/go/matrix"
)

func main() {
	// 1) LU decomposition for a calibration matrix.
	A, err := matrix.NewDense(2, 2)
	if err != nil {
		log.Fatalf("dense: %v", err)
	}
	_ = A.Set(0, 0, 4)
	_ = A.Set(0, 1, 3)
	_ = A.Set(1, 0, 6)
	_ = A.Set(1, 1, 3)

	L, U, err := matrix.LUDecompose(A)
	if err != nil {
		log.Fatalf("LU: %v", err)
	}
	fmt.Printf("LU shapes: L=%dx%d U=%dx%d\n", L.Rows(), L.Cols(), U.Rows(), U.Cols())

	// 2) Direct distance matrix for Floyd-Warshall.
	D, err := matrix.NewZeros(4, 4, matrix.WithAllowInfDistances())
	if err != nil {
		log.Fatalf("distance matrix: %v", err)
	}
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if i != j {
				_ = D.Set(i, j, math.Inf(1))
			}
		}
	}
	_ = D.Set(0, 1, 3)
	_ = D.Set(0, 2, 12)
	_ = D.Set(1, 2, 4)
	_ = D.Set(1, 3, 10)
	_ = D.Set(2, 3, 2)

	if err = matrix.APSPInPlace(D); err != nil {
		log.Fatalf("APSP: %v", err)
	}
	shortest, _ := D.At(0, 3)
	fmt.Printf("shortest 0->3: %.0f\n", shortest)

	// 3) Graph adjacency degree vector.
	g, err := core.NewGraph(core.WithDirected(true), core.WithWeighted())
	if err != nil {
		log.Fatalf("graph: %v", err)
	}
	_, _ = g.AddEdge("hub", "api", 2)
	_, _ = g.AddEdge("hub", "db", 3)
	_, _ = g.AddEdge("api", "cache", 5)
	_, _ = g.AddEdge("db", "archive", 7)

	opts, err := matrix.NewMatrixOptions(matrix.WithDirected(), matrix.WithWeighted())
	if err != nil {
		log.Fatalf("options: %v", err)
	}
	adj, err := matrix.NewAdjacencyMatrix(g, opts)
	if err != nil {
		log.Fatalf("adjacency: %v", err)
	}
	deg, err := matrix.DegreeVector(adj)
	if err != nil {
		log.Fatalf("degree vector: %v", err)
	}
	fmt.Printf("degree vector length: %d\n", len(deg))

	// 4) Normalize routing shares by row.
	R, err := matrix.NewDense(2, 3)
	if err != nil {
		log.Fatalf("routing dense: %v", err)
	}
	_ = R.Set(0, 0, 2)
	_ = R.Set(0, 1, 3)
	_ = R.Set(0, 2, 5)
	_ = R.Set(1, 0, 0)
	_ = R.Set(1, 1, 0)
	_ = R.Set(1, 2, 0)

	Y, norms, err := matrix.NormalizeRowsL1(R)
	if err != nil {
		log.Fatalf("normalize: %v", err)
	}
	y00, _ := Y.At(0, 0)
	fmt.Printf("row0 norm %.0f first share %.1f row1 norm %.0f\n", norms[0], y00, norms[1])

	// 5) Correlation over feature columns.
	X, err := matrix.NewDense(4, 3)
	if err != nil {
		log.Fatalf("features dense: %v", err)
	}
	values := []float64{
		1, 10, 5,
		2, 20, 5,
		3, 30, 5,
		4, 40, 5,
	}
	if err = X.Fill(values); err != nil {
		log.Fatalf("fill features: %v", err)
	}
	corr, _, stds, err := matrix.Correlation(X)
	if err != nil {
		log.Fatalf("correlation: %v", err)
	}
	fmt.Printf("corr shape: %dx%d degenerate std: %.0f\n", corr.Rows(), corr.Cols(), stds[2])

	// Output:
	// LU shapes: L=2x2 U=2x2
	// shortest 0->3: 9
	// degree vector length: 5
	// row0 norm 10 first share 0.2 row1 norm 0
	// corr shape: 3x3 degenerate std: 0
}
```

[![Go Playground](https://img.shields.io/badge/Go_Playground-Matrix_LU_APSP_DegreeVector_Normalization_Correlation-blue?logo=go)](https://go.dev/play/p/ITfz4eDMR9c)

---

## 10.9. Laws of Ownership, Concurrency & Publication

### Ownership

* Allocation-producing algebra and statistics functions return caller-owned matrices.
* `Clone` deep-copies `Dense` storage and preserves numeric policy.
* `Induced` materializes an independent copied submatrix.
* `View` returns a shared window. Mutations through the view are visible in the base matrix.

### Concurrency

`Dense` does not contain locks. Concurrent read-only access is safe when no goroutine mutates the matrix. Concurrent writes or read/write races require external synchronization.

Safe read-style operations include shape reads and `At` when the matrix is not being mutated. Unsafe concurrent operations include `Set`, `Fill`, `Apply`, and writes through `View`.

### Publication on Failure

Construction and transform functions return `nil` result artifacts with an error when validation or kernel execution fails. In-place functions may mutate before returning an error if the contract explicitly permits partial writes, as with `Apply`.

---

## 10.10. Pitfalls & Best Practices (Architectural Mastery)

> [!IMPORTANT]
> **1. Strict Error Policy: check sentinels, never parse strings.**
> Matrix errors are wrapped with operation context. Use `errors.Is(err, matrix.ErrOutOfRange)`, `errors.Is(err, matrix.ErrNaNInf)`, `errors.Is(err, matrix.ErrDimensionMismatch)`, and related sentinels. Error text is for diagnostics, not branching.

> [!CAUTION]
> **2. Zero is not always absence.**
> In classic adjacency, `0` means no edge. In zero-preserving weighted adjacency, finite `0` is a real edge and `+Inf` means no edge. Always interpret cells using the adjacency encoding selected by options.

> [!IMPORTANT]
> **3. `+Inf` is semantic, not dirty data.**
> Use `WithAllowInfDistances` or `WithMetricClosure` only when `+Inf` means “no path” or “no edge”. Do not use it to smuggle invalid measurements through statistics. Sanitize dirty data with `ReplaceInfNaN` and domain-specific replacement values.

> [!WARNING]
> **4. Do not run Floyd-Warshall on raw `0/weight` adjacency.**
> A raw `0` would be interpreted as a zero-cost path if passed directly as a distance. Use metric-closure builders or distance initialization so off-diagonal absence becomes `+Inf` and diagonal entries become `0`.

> [!CAUTION]
> **5. Dense adjacency is not an edge ledger.**
> It has one cell per vertex pair. If you need edge-column identity, parallel-edge visibility, or incidence algebra, use `IncidenceMatrix` instead of trying to infer identity from adjacency values.

> [!NOTE]
> **6. Choose copy vs view deliberately.**
> Use `Induced` or `Clone` when downstream code needs independent lifetime. Use `View` only for intentional shared windows. Treat `View.Set` as a write into the base matrix.

> [!IMPORTANT]
> **7. Respect zero-shape matrices.**
> Do not reject `0×N` or `N×0` just because they contain no elements. They are valid structural results for pipelines, especially feature extraction and empty graph transformations.

> [!NOTE]
> **8. Avoid eager materialization in hot loops.**
> Algebra facades allocate result matrices. For repeated transformations, preallocate staging matrices with `ZerosLike`, use `Apply` for in-place transformations when partial-write semantics are acceptable, and keep `*Dense` operands in hot paths to unlock fast loops.

> [!IMPORTANT]
> **9. Resolve options once near the boundary.**
> `Options` are the policy snapshot. Build them once with `NewMatrixOptions`, check the error, and pass the resolved value through builders. This prevents split-brain behavior between adjacency, incidence, and export steps.

> [!CAUTION]
> **10. Do not export derived distances as topology.**
> Metric closure answers “what is the shortest distance?”, not “which edge exists?”. `ToGraph` refuses it intentionally with `ErrMatrixNotImplemented`.

---

**lvlath/matrix**: deterministic dense algebra, precise numeric policy, and graph semantics without ambiguity.

> Next: [11. Traveling Salesman (TSP) ->](TSP.md)
