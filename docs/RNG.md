<!--
  SPDX-License-Identifier: AGPL-3.0-only
  Copyright (C) 2025-2026 katalvlaran

  File: docs/RNG.md
  Package reference: github.com/lvlath/go/rng
  Purpose: reproducible pseudo-random stream contracts, numerical proofs,
    ownership, exact draw consumption, integration and operational examples.
  Contract status: implemented package reference for lvlath-spec-20-v1.
  Scope: pinned PCG-DXSM128/64, SHA-256 state derivation, UniformOpen,
    Box-Muller cache, immutable Factory and additive buffer facades.
  License: AGPL-3.0-only; see ../LICENSE. The adapted PCG kernel additionally
    carries the Go authors' BSD attribution in ../rng/UPSTREAM_LICENSE.txt.
-->

# RNG — Version-pinned PCG-DXSM streams for reproducible experiments

> One logical simulation, one stream. The random path belongs to the **experiment**, never to the worker that happens to execute it.

`github.com/lvlath/go/rng` supplies deterministic pseudo-random streams for risk simulations, sensor fault injection, stochastic optimizers, and dependent-data resampling. It pins its own 128-bit PCG-DXSM kernel rather than delegating sequence generation to a possibly changing Go standard-library revision. It is **not cryptographic randomness**.

The `rng` package has no business-domain strategy: it does not price an option, sample a block index, select a volatility model, implement Kalman filtering, or estimate confidence intervals. Consumers own those choices. This separation lets different applications share the *same exact stream protocol*.

## Contents

1. [Where RNG fits](#1-where-rng-fits)
2. [Public API and ownership](#2-public-api-and-ownership)
3. [Pinned seed derivation](#3-pinned-seed-derivation)
4. [128-bit recurrence and DXSM](#4-128-bit-recurrence-and-dxsm)
5. [The 52-bit strictly open uniform](#5-the-52-bit-strictly-open-uniform)
6. [Box-Muller cache and exact consumption](#6-box-muller-cache-and-exact-consumption)
7. [Batch facades and nilability](#7-batch-facades-and-nilability)
8. [Parallel simulation without scheduling leakage](#8-parallel-simulation-without-scheduling-leakage)
9. [Executable operational scenarios](#9-executable-operational-scenarios)
10. [Integration contracts and limitations](#10-integration-contracts-and-limitations)
11. [Verification and performance](#11-verification-and-performance)
12. [Engineering pitfalls and FAQ](#12-engineering-pitfalls-and-faq)

---

## 1. Where RNG fits

```text
                      application-defined experiment
                                  |
                    (seed, logical pathID, ABI)
                                  |
                     +------------v-------------+
                     | immutable rng.Factory    |
                     | Stream / Fill* / Version |
                     +---+----------+-----------+
                         |          |
                   pathID=0    pathID=1       ... pathID=N
                         |          |                   |
                   mutable RNG mutable RNG        mutable RNG
                         |          |                   |
                      scenario-specific math and observables
                         |          |                   |
                   result[0]  result[1]          result[N]
                         +----------+-------------------+
                                    |
                       indexed deterministic reduction
                                    |
                         replayable aggregate result
```

* `Factory` is immutable and concurrent-safe; a nil receiver still works.
* Every `Stream` returns a new mutable generator; **one stream must have one goroutine owner**.
* `seed=0` is an ordinary reproducible seed, not an instruction to randomize.
* `pathID` is the logical trial/replicate/trajectory number. It must not depend on CPU, worker assignment, goroutine scheduling, or process startup order.
* Replaying the raw stream also requires the *same method-call order*. Uniform and normal calls consume different numbers of raw words.
* A deterministic floating-point aggregate additionally requires a fixed reduction order.

## 2. Public API and ownership

The `RNG` interface is intentionally **exactly three methods**:

```go
type RNG interface {
    Uint64() uint64
    UniformOpen() float64
    Normal() float64
}

type Factory struct{}

func NewFactory() *Factory
func NewStream(seed, pathID uint64) RNG
func (f *Factory) Stream(seed, pathID uint64) RNG
func (f *Factory) Version() string
func (f *Factory) IsNil() bool

func (f *Factory) FillUint64(seed, pathID uint64, dst []uint64)
func (f *Factory) FillUniformOpen(seed, pathID uint64, dst []float64)
func (f *Factory) FillNormal(seed, pathID uint64, dst []float64)

func UniformOpenFromUint64(raw uint64) float64
```

`NewStream`, `Fill*`, `Factory.IsNil`, and `UniformOpenFromUint64` are additive conveniences. They **do not** change the original interface, the derivation, any raw draws, or the pinned `Version()` value.

| API                        | State / consumption                    | Memory and ownership           | Typical purpose                       |
|----------------------------|----------------------------------------|--------------------------------|---------------------------------------|
| `NewFactory()`             | no draws                               | stateless immutable factory    | share one factory in orchestration    |
| `f.Stream(seed,pathID)`    | fresh state at position 0              | one independent mutable object | simulate one logical trajectory       |
| `NewStream(seed,pathID)`   | exactly `NewFactory().Stream(...)`     | fresh mutable object           | simplest one-off stream               |
| `f.Version()`              | no draws                               | constant string                | persist replay metadata               |
| `f.IsNil()`                | no draws                               | bool; true for nil pointer     | structural `core.Nilable` convention  |
| `s.Uint64()`               | 1 PCG step                             | no draw-time allocation        | raw protocol, controlled adapters     |
| `s.UniformOpen()`          | 1 raw word                             | strict `(0,1)` float64         | logarithms, inverse transforms        |
| `s.Normal()`               | 2 raw words on miss, 0 on hit          | private cached sine            | standard-normal shocks                |
| `f.FillUint64(...)`        | fresh stream, one raw step per cell    | writes caller-owned slice      | reproducible raw batch                |
| `f.FillUniformOpen(...)`   | fresh stream, one raw step per cell    | writes caller-owned slice      | reproducible uniform batch            |
| `f.FillNormal(...)`        | fresh stream; 2 raw per generated pair | writes caller-owned slice      | standard-normal vector                |
| `UniformOpenFromUint64(x)` | **zero** stream draws                  | pure float64 conversion        | exact endpoint verification / interop |

The `Factory` methods and pure conversion do not return errors: every `uint64` input is valid; nil/empty batch destinations are valid no-ops. The package does **not** expose context, Options, error sentinels, state snapshot, reset, or a `Result` artifact. The corresponding estimator-style interfaces would create fictional states and additional contract obligations without solving a real RNG problem.

**Nilability distinction.** `(*Factory).IsNil()` follows the pointer-presence convention (`nil` pointer → `true`, allocated zero Factory → `false`), but nil factory receivers remain functional. The private concrete stream implements optional `IsNil()` structurally; that method is intentionally *not* added to `RNG`, so existing custom RNGs remain source-compatible. There is no public result structure to which `Result.IsNil()` could meaningfully apply.

## 3. Pinned seed derivation

The SHA-256 preimage is **exactly 38 bytes**:

```text
byte index     00.....................21 22........29 30........37
              +------------------------+------------+------------+
              | 22 ASCII domain bytes  | seed LE64  | path LE64  |
              +------------------------+------------+------------+
              "lvlath/rng/pcg-dxsm-v1"   uint64       uint64
                          |
                       SHA-256
                          |
                  32 digest bytes
                +----------+----------+--------------------+
                | 0:8      | 8:16     | 16:32              |
                +----------+----------+--------------------+
                | hi LE64  | lo LE64  | unused for state   |
                +----------+----------+--------------------+
```

Formally,

$$
D=\operatorname{SHA256}(\mathrm{ASCII}(\texttt{lvlath/rng/pcg-dxsm-v1})\;\|\;\mathrm{LE64}(seed)\;\|\;\mathrm{LE64}(pathID)).
$$

`hi = LE64(D[0:8])` and `lo = LE64(D[8:16])`. There is no NUL separator and no extra length prefix. The following literal known-answer vectors are pinned as external tests; hexadecimal values must never be regenerated from the implementation under test:

| seed, pathID           | initial hi         | initial lo         | first `Uint64`     | second `Uint64`    |
|------------------------|--------------------|--------------------|--------------------|--------------------|
| 0, 0                   | `959f7cde48806db1` | `f075daaa9e93c7fe` | `de616b2b746d339a` | `8351bd653a53bc01` |
| 0, 1                   | `13acb9e304cdcdb7` | `4212ef527710d91e` | `6bd7526a9750e27d` | `bd45572342df7046` |
| 1, 0                   | `b4710acc1507334a` | `3dbed5e0dcdf52df` | `22fabff5ca8f6db0` | `2e9409e6e6c2bca2` |
| max uint64, max uint64 | `9040449b8608bb02` | `259a7f6793588e02` | `df0d2e5fdaa45333` | `24cf2f9c39b42ed0` |

`SHA-256` here is a fixed **state derivation** and domain separator, not an assertion that PCG outputs are secure or that arbitrary path pairs have mathematically disjoint streams. The state space has only 128 bits while `(seed,pathID)` ranges over 128 bits; digest truncation does not constitute a collision-free or disjoint-period proof.

## 4. 128-bit recurrence and DXSM

Let `S = (hi << 64) | lo`, with the following fixed 128-bit words:

```text
Multiplier M = 0x2360ed051fc65da44385df649fccf645
Increment  C = 0x5851f42d4c957f2d14057b7ef767814f
DXSM factor = 0xda942042e4dd58b5
```

The state transition is the linear congruential recurrence

$$
S_{t+1}=(S_t\cdot M+C)\bmod 2^{128}.
$$

![State recurrence](https://latex.codecogs.com/svg.image?S_%7Bt%2B1%7D%3D%28S_t%5Ccdot%20M%2BC%29%5Cbmod%202%5E%7B128%7D.)

Because every multiplication is modulo `2^128`, a 64-bit implementation needs only the low×low product, the two cross products, and the carry from adding the low increment. Writing `M = M_hi·2^64+M_lo` gives

$$
\begin{aligned}
q_{hi},q_{lo} &= \mathrm{Mul64}(lo,M_{lo}),\\
newLo,carry &= \mathrm{Add64}(q_{lo},C_{lo},0),\\
newHi &= q_{hi}+hi\cdot M_{lo}+lo\cdot M_{hi}+C_{hi}+carry\pmod{2^{64}}.
\end{aligned}
$$

The `hi*M_hi*2^128` term vanishes modulo `2^128`. DXSM mixes **the updated** (`newHi`,`newLo`) words in exactly this order:

```text
h = newHi
h = h XOR (h >> 32)
h = h * 0xda942042e4dd58b5       // unsigned mod 2^64
h = h XOR (h >> 48)
h = h * (newLo OR 1)             // unsigned mod 2^64
return h
```

The `lo|1` multiplier is intentionally odd. Any of the following changes are new algorithms, not refactors: mixing the old state, swapping the high/low state order, changing the increment, dropping a carry, using 64-bit rather than 128-bit state progression, or applying a different output mixer.

The local implementation is adapted from the PCG source of **Go 1.23.4 `math/rand/v2`**, with the original Go authors' BSD license notice preserved in `rng/UPSTREAM_LICENSE.txt`. Reproducibility does not depend on later Go toolchains preserving the same stdlib PCG output.

## 5. The 52-bit strictly open uniform

For a raw word `x`, write `k = x >> 12`, thus `0 ≤ k ≤ 2^52−1`. The mandated conversion is:

$$
U=(\operatorname{float64}(k)+\tfrac12)2^{-52}.
$$

**Proof of strict openness.** Every integer `k` in that range converts exactly to binary64. Every midpoint `k+0.5`, including `2^52−0.5`, is also exactly representable. Multiplication by the binary power `2^-52` is exact. Consequently,

$$
U_{min}=2^{-53},\qquad U_{max}=1-2^{-53},\qquad 0<U<1.
$$

![Strict open uniform endpoints](https://latex.codecogs.com/svg.image?U_%7Bmin%7D%3D2%5E%7B-53%7D%2C%5Cqquad%20U_%7Bmax%7D%3D1-2%5E%7B-53%7D%2C%5Cqquad%200%3CU%3C1.)

No rejection branch, rounding fix, or second draw is required. `UniformOpen` consumes **exactly one** `Uint64()`, regardless of the raw word. `UniformOpenFromUint64` exposes the same pure conversion for forced endpoint tests; it consumes **zero** words.

**Why not 53 bits?** The analogous 53-bit expression fails at its top code: binary64 rounds `2^53−0.5` to even `2^53`; scaling gives `1.0`. This is fatal for an inverse exponential/normal transform relying on `log(1-u)` or a strict open interval.

```go
fmt.Println(rng.UniformOpenFromUint64(0) == 0x1p-53)
fmt.Println(rng.UniformOpenFromUint64(^uint64(0)) == math.Nextafter(1, 0))
// true
// true
```

Here `fmt`, `math` and `rng` refer to the standard library and `github.com/lvlath/go/rng` imports in a complete Go program. See `rng/uniform_test.go` for a fully executable black-box regression.

## 6. Box-Muller cache and exact consumption

When `hasSpare == false`, `Normal()` draws `u1=UniformOpen()` and `u2=UniformOpen()`, computes

$$
r=\sqrt{-2\ln u_1},\qquad \theta=2\pi u_2,
$$

and produces

$$
z_0=r\cos\theta,\qquad z_1=r\sin\theta.
$$

It **returns `z0`** and stores `z1` in `spare` with `hasSpare=true`. On the next `Normal()` it returns `spare`, clears the flag, and consumes **no** additional raw words. `Uint64()` and `UniformOpen()` deliberately leave the Gaussian cache untouched.

| Sequence        | Raw consumption | Cache afterward | Meaning                      |
|-----------------|----------------:|-----------------|------------------------------|
| `Normal()`      |               2 | present (`z1`)  | generates one pair           |
| `Uint64()`      |               1 | still present   | consumes next raw event code |
| `Normal()`      |               0 | absent          | returns original cached `z1` |
| `UniformOpen()` |               1 | absent          | next uniform raw word        |
| `Normal()`      |               2 | present         | generates next pair          |

`Normal → Uint64 → Normal` therefore consumes exactly **3** raw words, not 5, while preserving the originally paired second Gaussian value.

The radius is always finite because `u1` is bounded below by `2^-53`. `sin`, `cos`, `sqrt` and `log` are ordinary transcendental floating-point functions: identical normal values are replayable within one platform/toolchain implementation, but **cross-platform last-bit equality is not promised**. The distribution is a numerical simulation approximation, not a proof of cryptographic strength or tail-fit quality.

### State checkpoint is deliberately not an API

Persisting only `hi/lo` would be wrong: a stream with a pending sine variate would resume with a different next `Normal()` after restore. Any future checkpoint design must version **all** of `{hi,lo,spare,hasSpare}` and the math implementation's compatibility scope. No such Snapshot, `MarshalBinary`, or reseed API is exposed in v1.

## 7. Batch facades and nilability

The additive batch methods are deterministic projections of `Stream`. They write into caller-owned slices and do not retain pointers:

```go
factory := rng.NewFactory()

shocks := make([]float64, 256)
factory.FillNormal(2026, 17, shocks)

// For this independent replay, the first three elements match the same stream.
stream := factory.Stream(2026, 17)
for i := 0; i < 3; i++ {
    fmt.Println(math.Float64bits(shocks[i]) == math.Float64bits(stream.Normal()))
}
// true
// true
// true
```

For `FillNormal(n)` the conceptual raw consumption is `2*ceil(n/2)`, although the temporary stream is discarded on return. If `n` is odd, its unused sine spare is discarded with it. Two consecutive calls to `FillNormal(seed,pathID,chunk)` **restart** at position zero. This is not a continuing stream and must not be used as one.

For a continuous logical trajectory across chunks:

```go
stream := factory.Stream(2026, 17)
for i := range firstChunk {
    firstChunk[i] = stream.Normal()
}
for i := range secondChunk {
    secondChunk[i] = stream.Normal() // Preserves the spare across chunks.
}
```

The same principle applies to `FillUint64` and `FillUniformOpen`: one fresh stream per nonempty batch invocation, filled by ascending slice index. A nil or empty buffer is accepted and left untouched. There is no separate batch RNG kernel, no implicit parallelism and no automatic memory allocation for the destination.

### Nil pointer is not invalid factory configuration

```go
var factory *rng.Factory
fmt.Println(factory.IsNil())  // true: pointer is nil
fmt.Println(factory.Version())
fmt.Println(factory.Stream(0, 0) != nil) // true: nil receiver is valid
```

`Factory.IsNil()` is an optional implementation of the `core.Nilable` pointer-presence convention. The private stream also has optional nilability, while `RNG` deliberately remains a **three-method interface** to preserve custom implementations. There is no `rng.Result` and no error-valued partial return to describe.

## 8. Parallel simulation without scheduling leakage

**Correct:** each path writes exactly one result at an index determined by its logical number; completion order is irrelevant.

```go
factory := rng.NewFactory()
const paths = 10000
payoffs := make([]float64, paths)

var group sync.WaitGroup
for worker := 0; worker < 8; worker++ {
    group.Add(1)
    go func(workerID int) {
        defer group.Done()
        for path := workerID; path < paths; path += 8 {
            pathRNG := factory.Stream(0, uint64(path))
            payoffs[path] = pathRNG.Normal() // One write per logical path.
        }
    }(worker)
}
group.Wait()

var total float64
for _, payoff := range payoffs { // Fixed reduction order, regardless of scheduling.
    total += payoff
}
_ = total
```

**Incorrect:** seed by worker ID, share one stream among goroutines, or sum inside concurrently scheduled workers and expect bitwise float64 identity.

The reproducibility key for an experiment includes at least: algorithm `Version`, seed, logical path assignment, call sequence, simulation parameters, input data revision, and aggregation order. A seed alone is insufficient. Independent worker scheduling does not imply identical libm bits across architectures or independence of random paths.

## 9. Executable operational scenarios

The repository contains six executable `// Output:` examples in [`rng/example_test.go`](../../../../Downloads/lvlath_rng_phase20_v2/rng/example_test.go), all using only the public API:

| Example                         | Operational problem                                                                       | Contract regime                                                  |
|---------------------------------|-------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| `ExampleFactory_Stream`         | Eight fulfillment zones × 24 hours of synthetic demand and capacity stress                | Fixed logical entity IDs; complete ledger replay                 |
| `ExampleFactory_parallelPaths`  | 16,384 dividend-adjusted European call GBM paths; independent Black-Scholes sanity        | Same payoff and reduction under 1 vs 8 workers                   |
| `ExampleFactory_blockBootstrap` | 512 circular index resamples of 32 quarter-hour latency observations                      | Preserves within-block adjacency; power-of-two *unbiased* starts |
| `ExampleFactory_sensorNoise`    | 1,000-step industrial temperature observer with Gaussian noise and raw packet-loss events | Causal update; interleaved raw/normal draw protocol              |
| `ExampleFactory_normalCache`    | Risk-event replay with Gaussian pair separated by an event word                           | Analytic sine/cosine oracle and exact raw position               |
| `ExampleFactory_FillNormal`     | Six channels × 256 preallocated sensor disturbances                                       | Fresh batch matches a canonical stream bit-for-bit               |

The separate [`examples/rng/main.go`](../../../../Downloads/lvlath_rng_phase20_v2/examples/rng/main.go) simulates **100,000** GBM terminal-value paths. It compares one worker with eight workers, reduces by logical index, prints the Monte Carlo estimate, and evaluates the independent Black-Scholes closed-form reference:

```bash
go run ./examples/rng
go test ./rng -run '^Example' -count=1 -v
```

All observations are **deterministic synthetic engineering fixtures**. The scenarios demonstrate reproducibility and implementation contracts, not validated market models, production Kalman tuning, certified confidence bounds or financial advice.

## 10. Integration contracts and limitations

### RNG versus bootstrap: do not silently rewrite an existing sampler

`bootstrap.SampleIndices(ctx, rows, plan)` has an existing public replay identifier (`Result.RNG`), block-index rules, row-completion/cancellation law and exact bounded-integer mapping. Merely changing its random source to `rng.Factory.Stream(plan.Seed,uint64(replicate))` can change all published index sequences and replay identities. A future integration requires a separate consumer contract audit, golden vectors and a deliberate migration/version decision. **Adding `rng` does not claim that the existing bootstrap implementation has already migrated.**

If the source population has a power-of-two size, `word & (size-1)` gives an unbiased start index; this is why the standalone 32-element demonstration is valid. For arbitrary bounds, `word%bound` is biased unless `bound` divides `2^64`; the domain consumer must implement and test exact rejection sampling. `rng` v1 does not expose a bounded-integer method.

### Financial Monte Carlo and stochastic differential equations

`Normal()` supplies unit normal shocks; application code specifies drift, diffusion, discretization, dividends, interest rates, variance models and numerical error checks. A conventional exact-terminal GBM path is:

$$
S_T=S_0\exp\left((r-q-\tfrac12\sigma^2)T+\sigma\sqrt{T}\,Z\right),\qquad Z\sim N(0,1).
$$

![GBM terminal price](https://latex.codecogs.com/svg.image?S_T%3DS_0%5Cexp%5Cleft%28%28r-q-%5Ctfrac12%5Csigma%5E2%29T%2B%5Csigma%5Csqrt%7BT%7D%5C%2CZ%5Cright%29%2C%5Cqquad%20Z%5Csim%20N%280%2C1%29.)

For time-stepped GBM, Heston or Merton models, each logical path should retain one RNG across steps. Changing how many uniforms/normals are consumed or how many steps are simulated changes the path sequence and must be recorded with simulation parameters. `rng` does not prove that the chosen financial model or discretization is correct.

### Sensor faults, Kalman-style observers, and causal inference

A controller can use the stream to inject `measurement = trueValue + sigma*stream.Normal()` and a raw event code for deterministic telemetry dropouts. The observer/filter itself must remain causal. Do not confuse a reproduced input noise path with a validated estimator or a guarantee that real faults are normally distributed.

### Custom RNGs and consumer validation

The `RNG` interface is deliberately minimal so clients can substitute fakes or alternative implementations in their own tests. These custom implementations might return invalid uniforms or `NaN` normals. A consumer that requires `0<U<1` or finite normal samples must validate **its incoming values** at that boundary; the factory's numeric guarantees apply only to streams it creates.

## 11. Verification and performance

| Suite                  | What it protects                                                                          |
|------------------------|-------------------------------------------------------------------------------------------|
| `seed_vectors_test.go` | four literal SHA seed/state and raw-output vectors, including seed 0 and MaxUint64        |
| `pcg_test.go`          | independent `math/big.Int` unsigned 128-bit recurrence and DXSM for 20 steps              |
| `uniform_test.go`      | forced raw zero/max endpoint conversion, unsafe 53-bit counterexample, single-word parity |
| `normal_test.go`       | cosine-first result, cached sine, raw/uniform interleaving and fixed-seed moment smoke    |
| `factory_test.go`      | 100 path IDs under concurrent scheduling, replay, nil-receiver behavior and no aliasing   |
| `batch_test.go`        | raw/uniform/normal batch parity at empty, odd, even and large lengths; nilability         |
| `alloc_test.go`        | 1,000-call allocation regression for each stateful hot method                             |
| `source_audit_test.go` | forbidden production imports and direct panic/recover calls via Go AST                    |
| `example_test.go`      | six operational, deterministic `// Output:` programs                                      |
| `bench_test.go`        | separate creation cost, single-draw cost and amortized batch cost                         |

A million fixed-seed normal observations form an inexpensive **statistical smoke test** for mean, variance and skewness. It is *not* a proof of full distribution quality, absence of stream collisions, or cryptographic security.

The core stream uses `O(1)` state. Its draws are `O(1)` time with **zero explicit allocations** after creation. A batch wrapper performs `O(n+1)` work and holds `O(1)` auxiliary state but deliberately writes `n` caller-provided elements. Creating a fresh stream may allocate; do not conflate this with hot-loop allocations.

Recommended repo-level gates:

```bash
gofmt -w rng/*.go examples/rng/main.go
go test ./rng -count=1
go test ./rng -race -count=1
go test ./rng -cover -count=1
go test -run '^$' -bench . -benchmem ./rng
go vet ./...
go test ./...
golangci-lint run
govulncheck ./...
```

Record the actual toolchain, CPU and checked git SHA. A successful standalone test under Go 1.23.2 is **not equivalent** to passing a repository's declared Go 1.23.4+ CI. If lint/go vulnerability tools are unavailable, report **NOT RUN**, never PASS.

## 12. Engineering pitfalls and FAQ

**Does the factory prove independent streams?** No. SHA-256 maps fixed identifiers to reproducible starting states; it does not prove non-overlap of sequences. Validate application statistical assumptions independently.

**Can I use `Stream(seed,workerID)`?** Not when worker assignment changes across runs. Always address a logical trial/replicate/path. Worker slots only decide which paths they execute.

**Can I call the same stream concurrently?** No. There is no internal locking. Give each goroutine its own path stream, or synchronize access if shared consumption order is intentionally part of your model (which usually defeats scheduling independence).

**Why is seed zero valid?** All `uint64` values encode as ordinary little-endian fields. No zero → clock conversion or fallback exists.

**Why do raw draws between normals matter?** They advance the PCG state but do not invalidate an already cached sine variate. The later cached `Normal` must consume **zero** words, so raw consumption is predictable.

**Can I resume after a batch call?** Not by calling the same `Fill*` method again. Every batch starts a fresh stream. Use one retained `RNG` to span chunks, especially for an odd number of normal draws.

**Is a uniform value ever exactly zero or one?** No for the factory producer; the range is `[2^-53,1-2^-53]`. The pure conversion lets tests force both worst raw codes directly.

**Can I use `UniformOpen` to select an arbitrary integer index?** Not if unbiased integer sampling is a hard contract. The midpoint mapping has finite bins, and naive scaling/modulo can introduce bias for arbitrary bounds. Implement exact bounded integer rejection in the appropriate consumer.

**Can I use it to generate secrets?** No. PCG-DXSM is a fast numerical PRNG, not a cryptographic construction. SHA-256 seeding does not change this fact.

**What guarantees cross-platform replay?** The unsigned 128-bit recurrence and input hashing are portable. Last bits of the Box-Muller transcendental functions and derived floating point finance calculations may vary across platforms/libm implementations.

**Should I add an RNG Result or Options structure?** No, not without a real new mathematical mode or fallible operation. `Stream` is an ownership boundary for mutable state. `core.Nilable` applies to relevant pointer artifacts, not a requirement to invent fake successful-fit results.

**Where do I begin?** Use [`rng/doc.go`](../../../../Downloads/lvlath_rng_phase20_v2/rng/doc.go) for the formal charter, `rng/example_test.go` for six executable operational workflows, [`docs/TUTORIAL.md`](TUTORIAL.md) for first-use integration, and [`docs/FAQ_&_TIPS.md`](FAQ_%26_TIPS.md) for troubleshooting.
