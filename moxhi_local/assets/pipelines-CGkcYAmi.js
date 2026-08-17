import{H as N,d as A,S as U,c as Y,E as $,f as te,B as K,V as se,R as J,F as Z,L as Q,A as P,D as ee,P as ne,e as re}from"./constants-CSVWRdrh.js";const oe=`// GEMM: Y[m,n] = Σ_k X[m,k] · W[k,n] (+ B[n]), optional SiLU.
// X: [M,K] row-major, W: [K,N] row-major, Y: [M,N] row-major.
// With the WT flag, W is stored TRANSPOSED — [N,K] row-major — and the W
// index becomes n·K + k (the tied-embedding LM head reads shared.weight
// [24000,448] directly, no transposed copy).
// Simple correctness-first kernel — one invocation per output element,
// f32 accumulation regardless of storage type. Tuning happens in Task 17.
//
// Template placeholders (substituted by buildShader in pipelines.js — note:
// do NOT write literal placeholder syntax in comments, the substitution is a
// dumb string replace over the whole file):
//   ENABLE_F16      the f16 enable directive when T or OUT_T is f16, else empty
//   T               storage type of X/W/B (f16|f32)
//   OUT_T           storage type of Y (f16|f32)
//   WG              workgroup size in x (default 64)
//   IF_BIAS/IF_SILU/IF_WT conditional blocks
{{ENABLE_IMMEDIATE}}
{{ENABLE_F16}}

struct Dims { M: u32, K: u32, N: u32, _pad: u32 }

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> dims: Dims;
@group(0) @binding(1) var<storage, read> X: array<{{T}}>;
@group(0) @binding(2) var<storage, read> W: array<{{T}}>;
@group(0) @binding(3) var<storage, read> B: array<{{T}}>;
@group(0) @binding(4) var<storage, read_write> Y: array<{{OUT_T}}>;

@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Keep B in the shader interface (and thus in the auto bind group layout)
  // even when the bias block is compiled out — a 4-byte dummy buffer is
  // bound in that case.
  _ = &B;
  let n = gid.x;
  let m = gid.y;
  if (n >= dims.N || m >= dims.M) { return; }
  var acc: f32 = 0.0;
  let xoff = m * dims.K;
  for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
    var wIdx = k * dims.N + n; // W [K,N] row-major
    {{IF_WT}}wIdx = n * dims.K + k; // W transposed: [N,K] row-major{{/IF_WT}}
    acc = acc + f32(X[xoff + k]) * f32(W[wIdx]);
  }
  {{IF_BIAS}}acc = acc + f32(B[n]);{{/IF_BIAS}}
  {{IF_SILU}}acc = acc / (1.0 + exp(-acc));{{/IF_SILU}}
  Y[m * dims.N + n] = {{OUT_T}}(acc);
}
`,ie=`// GEMV-style GEMM for SMALL M (decode-step projections, M = batch ≤ ~16).
// Same math/semantics as gemm.wgsl: Y[m,n] = Σ_k X[m,k]·W[k,n] (+B[n]),
// optional SiLU, f32 accumulation. Chosen by dispatchGemm({flags.gemv}).
//
// Why a separate kernel (Task 17 profile data): at M=1..8 the plain kernel
// launches only N threads, each walking the whole K serially → latency-bound
// (~150ns/iter chains, e.g. fc2 268µs at b1); and in WT mode adjacent threads
// read 2-byte elements 896B apart → ~1/64 cache-line utilization (lm_head
// 1918µs at b8). Here a workgroup of TK·TN threads computes a tile of outputs
// with K split across TK lanes + a shared-memory tree reduction: TK× shorter
// dependency chains, TK·(TN)× more threads in flight, and coalesced loads.
//
// Two layouts (exactly one of WT / NWT):
//   WT  — W stored [N,K] row-major (lm_head / shared.weight). K%4 == 0
//         required: X and W are bound as vec4 arrays and the lane loop walks
//         K/4 vec4s. tid = o·TK + lane (lane fastest) → the TK lanes of one
//         output read CONSECUTIVE vec4s of the W row (fully coalesced).
//         TN outputs per workgroup; grid (ceil(N/TN), ceil(M/MT)): a
//         workgroup serves up to MT rows of X from one L1-resident W tile —
//         without this every batch row re-streams the whole W from DRAM
//         (lm_head at b8: 8×21.5MB).
//   NWT — W stored [K,N] row-major (all other projections). N%4 == 0
//         required: each thread owns a QUAD of 4 consecutive n via vec4 W
//         row loads; X[k] is a scalar broadcast. tid = lane·TN + oq (quad
//         fastest) → adjacent threads read adjacent vec4s within a k-row
//         (coalesced); lanes stride k by TK. TN quads (4·TN outputs) per
//         workgroup; grid (ceil(N/(4·TN)), M).
//
// STORE_KV (decode self_qkv, either layout): the projection output is a
// fused q|k|v row [M, 3·H·D]; the epilogue additionally scatters the k and v
// outputs into the [B, Lmax, H·D] K/V caches at decode position t —
// replacing the separate kv_append dispatch (Task 17: per-dispatch fixed
// overhead dominates the b1 step). Values are rounded through the same
// storage-type conversion as Y, so parity with the old copy-from-Y kv_append
// is bit-exact within a layout (WT vs NWT differ in accumulation order).
// Mutually exclusive with WQ8 (both claim binding 5; dispatch enforces).
//
// Out-of-range tail outputs are computed on clamped indices (uniform control
// flow for the barriers) and simply not stored.
//
// Template placeholders (buildShader in pipelines.js):
//   ENABLE_F16      f16 enable directive when T or OUT_T is f16
//   T / OUT_T       storage types (f16|f32 — vec4<f32> is core, no fallback
//                   split needed; the plain gemm.wgsl path remains the
//                   unvectorized fallback)
//   WG              workgroup size — MUST equal TK·TN (dispatchGemm enforces)
//   TK              k-lanes per output (power of two)
//   TN              outputs (WT) / output quads (NWT) per workgroup
//   MT              WT only: X rows served per workgroup (W-tile reuse)
//   IF_BIAS/IF_SILU/IF_WT/IF_NWT conditional blocks
//   IF_WQ8/IF_WQF   WT only: int8 W8A16 weights / float weights. WQ8 binds W
//                   as u32 words (4×i8 along K, quantizeQ8Rows layout) with
//                   per-N f32 scales at binding 5, applied in the epilogue
//                   BEFORE bias — same contract as gemm_tiled2's WQ8 path.
//                   Mutually exclusive with STORE_KV (dispatch enforces).
//   IF_SG/IF_NOSG   WT only: subgroup reduction (flags.sg) — the TK lanes of
//                   one output are consecutive tids, so a subgroupShuffleDown
//                   chain folds them with ZERO barriers (the shared-memory
//                   tree costs log2(TK) barriers per MT row). Requires a
//                   TK-slice to never straddle a subgroup: dispatch gates on
//                   TK ≤ ctx.subgroupMinSize. Reduction order differs →
//                   tolerance-equal, not bit-equal, vs the tree.
{{ENABLE_IMMEDIATE}}
{{ENABLE_F16}}
{{ENABLE_SG}}

struct Dims { M: u32, K: u32, N: u32, _pad: u32{{IF_STORE_KV}}, t: u32, Lmax: u32, _p1: u32, _p2: u32{{/IF_STORE_KV}} }

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> dims: Dims;
{{IF_WT}}
@group(0) @binding(1) var<storage, read> X: array<vec4<{{T}}>>;
{{IF_WQF}}
@group(0) @binding(2) var<storage, read> W: array<vec4<{{T}}>>;
{{/IF_WQF}}
{{IF_WQ8}}
@group(0) @binding(2) var<storage, read> W: array<u32>; // 4×i8 along K
{{/IF_WQ8}}
{{/IF_WT}}
{{IF_NWT}}
@group(0) @binding(1) var<storage, read> X: array<{{T}}>;
@group(0) @binding(2) var<storage, read> W: array<vec4<{{T}}>>;
{{/IF_NWT}}
@group(0) @binding(3) var<storage, read> B: array<{{T}}>;
@group(0) @binding(4) var<storage, read_write> Y: array<{{OUT_T}}>;
{{IF_STORE_KV}}
@group(0) @binding(5) var<storage, read_write> Kc: array<{{T}}>;
@group(0) @binding(6) var<storage, read_write> Vc: array<{{T}}>;
{{/IF_STORE_KV}}
{{IF_WQ8}}
@group(0) @binding(5) var<storage, read> S: array<f32>; // per-N scales
{{/IF_WQ8}}

const TK: u32 = {{TK}}u;
const TN: u32 = {{TN}}u;
const WG: u32 = {{WG}}u;

{{IF_WT}}
const MT: u32 = {{MT}}u;
{{IF_NOSG}}
var<workgroup> red: array<f32, WG>;
{{/IF_NOSG}}

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = &B; // keep B in the auto layout even when bias is compiled out
  let tid = lid.x;
  let lane = tid % TK;   // lane fastest → coalesced W reads
  let o = tid / TK;
  let n = wid.x * TN + o;
  let nc = min(n, dims.N - 1u); // clamp tail (store is guarded below)
  let K4 = dims.K / 4u;
  let wbase = nc * K4;
  // m loop: the same W tile (L1-hot after the first row) is swept for up to
  // MT rows. The break is uniform across the workgroup (wid-derived), so the
  // barriers below stay in uniform control flow. red is reused per row: safe,
  // because between barriers each thread only touches red slots it owns.
  for (var mm = 0u; mm < MT; mm = mm + 1u) {
    let m = wid.y * MT + mm;
    if (m >= dims.M) { break; }
    let xbase = m * K4;
    var acc = vec4<f32>(0.0);
    for (var i = lane; i < K4; i = i + TK) {
      {{IF_WQF}}
      acc = acc + vec4<f32>(X[xbase + i]) * vec4<f32>(W[wbase + i]);
      {{/IF_WQF}}
      {{IF_WQ8}}
      // One u32 = 4 raw int8 q values along K, sign-extended by shifts;
      // the per-N scale is applied once in the epilogue.
      let word = bitcast<i32>(W[wbase + i]);
      acc = acc + vec4<f32>(X[xbase + i]) * vec4<f32>(
        f32((word << 24u) >> 24u),
        f32((word << 16u) >> 24u),
        f32((word << 8u) >> 24u),
        f32(word >> 24u));
      {{/IF_WQ8}}
    }
    {{IF_NOSG}}
    red[tid] = acc.x + acc.y + acc.z + acc.w;
    workgroupBarrier();
    for (var s = TK / 2u; s > 0u; s = s >> 1u) {
      if (lane < s) { red[tid] = red[tid] + red[tid + s]; }
      workgroupBarrier();
    }
    let vr = red[tid];
    {{/IF_NOSG}}
    {{IF_SG}}
    // Butterfly over the TK-slice: after step s, lanes < s hold sums of
    // [lane, lane+2s). Lanes ≥ TK/2 read across the slice edge — harmless,
    // they never feed lane 0. No barriers, no shared memory.
    var vr = acc.x + acc.y + acc.z + acc.w;
    for (var s = TK / 2u; s > 0u; s = s >> 1u) {
      vr = vr + subgroupShuffleDown(vr, s);
    }
    {{/IF_SG}}
    if (lane == 0u && n < dims.N) {
      var v = vr;
      {{IF_WQ8}}v = v * S[n]; // dequantize before bias{{/IF_WQ8}}
      {{IF_BIAS}}v = v + f32(B[n]);{{/IF_BIAS}}
      {{IF_SILU}}v = v / (1.0 + exp(-v));{{/IF_SILU}}
      Y[m * dims.N + n] = {{OUT_T}}(v);
      {{IF_STORE_KV}}
      // Fused q|k|v row: outputs in [HD, 2HD) are k, [2HD, 3HD) are v.
      // Same cache layout as the NWT quad scatter (kv_append contract).
      let HD = dims.N / 3u;
      if (n >= HD) {
        let off = n - HD;
        let dstBase = (m * dims.Lmax + dims.t) * HD;
        if (off < HD) {
          Kc[dstBase + off] = {{T}}(v);
        } else {
          Vc[dstBase + (off - HD)] = {{T}}(v);
        }
      }
      {{/IF_STORE_KV}}
    }
  }
}
{{/IF_WT}}

{{IF_NWT}}
var<workgroup> red: array<vec4<f32>, WG>;

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = &B;
  let tid = lid.x;
  let m = wid.y;
  let oq = tid % TN;     // quad fastest → coalesced W reads within a k-row
  let lane = tid / TN;
  let N4 = dims.N / 4u;
  let q = wid.x * TN + oq;
  let qc = min(q, N4 - 1u); // clamp tail (store is guarded below)
  let xbase = m * dims.K;
  var acc = vec4<f32>(0.0);
  for (var k = lane; k < dims.K; k = k + TK) {
    acc = acc + f32(X[xbase + k]) * vec4<f32>(W[k * N4 + qc]);
  }
  red[tid] = acc;
  workgroupBarrier();
  for (var s = TK / 2u; s > 0u; s = s >> 1u) {
    if (lane < s) { red[tid] = red[tid] + red[tid + s * TN]; }
    workgroupBarrier();
  }
  if (lane == 0u && q < N4) {
    var v = red[tid];
    let n0 = q * 4u;
    {{IF_BIAS}}
    v = v + vec4<f32>(f32(B[n0]), f32(B[n0 + 1u]), f32(B[n0 + 2u]), f32(B[n0 + 3u]));
    {{/IF_BIAS}}
    {{IF_SILU}}v = v / (vec4<f32>(1.0) + exp(-v));{{/IF_SILU}}
    let ybase = m * dims.N + n0;
    Y[ybase] = {{OUT_T}}(v.x);
    Y[ybase + 1u] = {{OUT_T}}(v.y);
    Y[ybase + 2u] = {{OUT_T}}(v.z);
    Y[ybase + 3u] = {{OUT_T}}(v.w);
    {{IF_STORE_KV}}
    // Fused q|k|v row: quads in [HD, 2HD) are k, [2HD, 3HD) are v. A quad
    // never straddles slices (HD % 4 == 0). Cache layout matches kv_append:
    // dst = (b·Lmax + t)·HD + (h·D + d), with m = batch row.
    let HD = dims.N / 3u;
    if (n0 >= HD) {
      var off = n0 - HD;
      let dstBase = (m * dims.Lmax + dims.t) * HD;
      if (off < HD) {
        Kc[dstBase + off] = {{T}}(v.x);
        Kc[dstBase + off + 1u] = {{T}}(v.y);
        Kc[dstBase + off + 2u] = {{T}}(v.z);
        Kc[dstBase + off + 3u] = {{T}}(v.w);
      } else {
        off = off - HD;
        Vc[dstBase + off] = {{T}}(v.x);
        Vc[dstBase + off + 1u] = {{T}}(v.y);
        Vc[dstBase + off + 2u] = {{T}}(v.z);
        Vc[dstBase + off + 3u] = {{T}}(v.w);
      }
    }
    {{/IF_STORE_KV}}
  }
}
{{/IF_NWT}}
`,ue=`// Tiled GEMM: Y[m,n] = Σ_k X[m,k] · W[k,n] (+ B[n]), optional SiLU — the
// large-M sibling of gemm.wgsl (encoder GEMMs, decode lm_head at big batch).
// X: [M,K] row-major, W: [K,N] row-major (WT flag: [N,K] row-major), Y: [M,N].
//
// Each workgroup computes a BM×BN output tile: the K dimension is walked in
// BK-wide slices staged through workgroup memory (Xs/Ws, f32), and each of
// the (BM/4)·(BN/4) threads accumulates a FIXED 4×4 register subtile — 16
// FMAs per 8 shared-memory reads instead of gemm.wgsl's 1 FMA per 2 global
// reads. f32 accumulation regardless of storage type.
//
// The 4×4 subtile is deliberately NOT templated: it lives in four vec4<f32>
// accumulators with fully static indexing. A first version used
// array<f32, TM*TN> with loop indices — Tint/DXC kept it in scratch memory
// and the kernel came out SLOWER than the naive one (862ms vs 598ms encoder
// b64). Static vec4 registers are the whole point of the tile.
//
// Constraints: BM % 4 == 0, BN % 4 == 0, workgroup size = (BM/4)·(BN/4)
// (dispatchGemmTiled computes it), Xs+Ws = (BM+BN)·BK·4 bytes ≤ workgroup
// storage limit (16 KiB default). M/N/K tails are guarded: OOB loads stage
// 0, OOB stores are skipped.
//
// Template placeholders (see buildShader in pipelines.js):
//   ENABLE_F16, T, OUT_T, WG          as in gemm.wgsl (WG = thread count)
//   BM, BN, BK                        tile geometry (u32 literals)
//   IF_BIAS / IF_SILU / IF_WT         conditional blocks
{{ENABLE_IMMEDIATE}}
{{ENABLE_F16}}

struct Dims { M: u32, K: u32, N: u32, _pad: u32 }

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> dims: Dims;
@group(0) @binding(1) var<storage, read> X: array<{{T}}>;
@group(0) @binding(2) var<storage, read> W: array<{{T}}>;
@group(0) @binding(3) var<storage, read> B: array<{{T}}>;
@group(0) @binding(4) var<storage, read_write> Y: array<{{OUT_T}}>;

const BM = {{BM}}u;
const BN = {{BN}}u;
const BK = {{BK}}u;
const THREADS = (BM / 4u) * (BN / 4u);

// Xs is stored TRANSPOSED — Xs[kk][row] — so the inner loop's 4-row read is
// one contiguous run; Ws[kk][col] likewise makes the 4-col read contiguous.
var<workgroup> Xs: array<f32, BK * BM>;
var<workgroup> Ws: array<f32, BK * BN>;

fn storeRow(m: u32, n0: u32, v: vec4<f32>) {
  if (m >= dims.M) { return; }
  let base = m * dims.N;
  if (n0 + 3u < dims.N) {
    Y[base + n0] = {{OUT_T}}(v.x);
    Y[base + n0 + 1u] = {{OUT_T}}(v.y);
    Y[base + n0 + 2u] = {{OUT_T}}(v.z);
    Y[base + n0 + 3u] = {{OUT_T}}(v.w);
    return;
  }
  // N tail: per-lane guards.
  if (n0 < dims.N) { Y[base + n0] = {{OUT_T}}(v.x); }
  if (n0 + 1u < dims.N) { Y[base + n0 + 1u] = {{OUT_T}}(v.y); }
  if (n0 + 2u < dims.N) { Y[base + n0 + 2u] = {{OUT_T}}(v.z); }
}

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  // Keep B in the auto bind-group layout even when the bias block is
  // compiled out — a 4-byte dummy buffer is bound in that case.
  _ = &B;
  let rowBase = wid.y * BM;
  let colBase = wid.x * BN;
  // This thread's 4×4 subtile origin inside the workgroup tile.
  let tRow = (lid / (BN / 4u)) * 4u;
  let tCol = (lid % (BN / 4u)) * 4u;

  var acc0 = vec4<f32>(0.0); // row tRow+0, cols tCol..tCol+3
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  for (var k0: u32 = 0u; k0 < dims.K; k0 = k0 + BK) {
    // Cooperative stage: Xs[BK][BM] (transposed) and Ws[BK][BN], linear
    // thread sweep, zero-fill out-of-bounds lanes (M/K/N tails).
    for (var i: u32 = lid; i < BK * BM; i = i + THREADS) {
      let kk = i / BM;
      let row = i % BM;
      var v: f32 = 0.0;
      if (rowBase + row < dims.M && k0 + kk < dims.K) {
        v = f32(X[(rowBase + row) * dims.K + k0 + kk]);
      }
      Xs[i] = v;
    }
    for (var i: u32 = lid; i < BK * BN; i = i + THREADS) {
      let kk = i / BN;
      let col = i % BN;
      var v: f32 = 0.0;
      if (colBase + col < dims.N && k0 + kk < dims.K) {
        var wIdx = (k0 + kk) * dims.N + colBase + col; // W [K,N] row-major
        {{IF_WT}}wIdx = (colBase + col) * dims.K + k0 + kk; // W transposed: [N,K]{{/IF_WT}}
        v = f32(W[wIdx]);
      }
      Ws[i] = v;
    }
    workgroupBarrier();

    for (var kk: u32 = 0u; kk < BK; kk = kk + 1u) {
      let xb = kk * BM + tRow;
      let wb = kk * BN + tCol;
      let xr = vec4<f32>(Xs[xb], Xs[xb + 1u], Xs[xb + 2u], Xs[xb + 3u]);
      let wr = vec4<f32>(Ws[wb], Ws[wb + 1u], Ws[wb + 2u], Ws[wb + 3u]);
      acc0 = fma(vec4<f32>(xr.x), wr, acc0);
      acc1 = fma(vec4<f32>(xr.y), wr, acc1);
      acc2 = fma(vec4<f32>(xr.z), wr, acc2);
      acc3 = fma(vec4<f32>(xr.w), wr, acc3);
    }
    workgroupBarrier();
  }

  {{IF_BIAS}}
  // min-clamped reads: OOB cols load a dummy lane that storeRow never writes.
  let nHi = dims.N - 1u;
  let bv = vec4<f32>(
    f32(B[min(colBase + tCol, nHi)]),
    f32(B[min(colBase + tCol + 1u, nHi)]),
    f32(B[min(colBase + tCol + 2u, nHi)]),
    f32(B[min(colBase + tCol + 3u, nHi)]));
  acc0 = acc0 + bv;
  acc1 = acc1 + bv;
  acc2 = acc2 + bv;
  acc3 = acc3 + bv;
  {{/IF_BIAS}}
  {{IF_SILU}}
  acc0 = acc0 / (vec4<f32>(1.0) + exp(-acc0));
  acc1 = acc1 / (vec4<f32>(1.0) + exp(-acc1));
  acc2 = acc2 / (vec4<f32>(1.0) + exp(-acc2));
  acc3 = acc3 / (vec4<f32>(1.0) + exp(-acc3));
  {{/IF_SILU}}

  storeRow(rowBase + tRow, colBase + tCol, acc0);
  storeRow(rowBase + tRow + 1u, colBase + tCol, acc1);
  storeRow(rowBase + tRow + 2u, colBase + tCol, acc2);
  storeRow(rowBase + tRow + 3u, colBase + tCol, acc3);
}
`,de=`// Tiled GEMM v2 — gemm_tiled.wgsl with the memory system fixed:
//   * X and W are bound as vec4 arrays: staging loads are one coalesced
//     8-byte (f16) vec4 per thread instead of v1's stride-K scalar sweeps.
//   * Xs/Ws live as vec4 shared arrays: the inner loop issues 2 vec4 LDS
//     reads per kk instead of v1's 8 scalar reads (v1 profiled ~7.9 TFLOPS
//     on fc1 — LDS instruction bound, not FMA bound).
//   * Optional 8×4 register subtile (IF_TM8): halves threads, doubles the
//     FLOPs per Ws read. Static vec4 accumulators only — array accumulators
//     spill under Tint/DXC (see gemm_tiled.wgsl header; that lesson cost a
//     1.4× regression before it was learned).
//
// Y[m,n] = Σ_k X[m,k]·W[k,n] (+B[n]), optional SiLU. X: [M,K] row-major,
// W: [K,N] row-major (WT: [N,K] row-major), Y: [M,N].
//
// Eligibility (dispatchGemmTiled routes ineligible shapes to v1):
//   K % 4 == 0            (vec4 X reads; also W reads when WT)
//   N % 4 == 0 unless WT  (vec4 W reads along N)
//   BK % 4 == 0, BM % 4 == 0, BN % TN == 0
// M tails ARE handled (OOB rows stage zero / skip store); K and N never
// straddle a vec4 thanks to the %4 constraints, so tail guards are per-vec4.
//
// W8A16 int8 mode (IF_WQ8 — lm_head, decode FFN): W is int8 [N,K] rows
// packed 4-per-u32 along K (quantizeQ8Rows layout); the per-N f32 scales get
// their own binding (5) so bias/SiLU stay available (FFN fc1). Raw q values
// are unpacked with sign-extending shifts and accumulated in f32 exactly
// like float weights; the per-column scale factors out of the dot product
// and multiplies the accumulators in the epilogue BEFORE bias — so
// dequantized results are bit-identical to a float GEMM over q·scale.
// Halves the W traffic of the f16 kernels.
//
// Fused argmax mode (IF_ARGMAX — the lm_head): the [M, 24000] f32 logits
// never touch global memory. Each thread applies final_logits_bias plus the
// repetition penalty to its register tile and keeps a per-(row, column-quad)
// running max; the workgroup folds its BN columns to ONE (val, idx) partial
// per row, and argmax_reduce.wgsl folds the ceil(N/BN) partials per row and
// runs argmax_penalty's token/done/bitmask epilogue. Same f32 ops in the same
// lexicographic (val, lowest idx) order as the unfused store+scan path, so
// the picked tokens are bit-identical — while the logits store, the argmax
// re-read, and the per-row bias re-read all disappear (~37MB/step at B=128).
//
// Template placeholders (see buildShader in pipelines.js):
//   ENABLE_F16, T, OUT_T, WG   as in gemm.wgsl (WG = (BM/TM)·(BN/4))
//   BM, BN, BK                 tile geometry (u32 literals)
//   IF_TM8                     8-row register subtile (else 4-row)
//   IF_WQ8 / IF_WQF            int8-packed W + scales / float W + bias
//   IF_BIAS / IF_SILU / IF_WT  as in gemm_tiled.wgsl
//   IF_STORE_Y / IF_ARGMAX     store Y (default) / fused-argmax epilogue
//   PENALTY / MASK_WORDS       argmax mode only: repetition penalty literal,
//                              bitmask u32 words per row
//   IF_SH16 / IF_SH32          shared tiles store the native {{T}} / f32.
//                              SH16 halves LDS bank traffic; f16-origin
//                              values round-trip f32→f16 exactly so results
//                              are bit-identical. ARGMAX requires SH32 (the
//                              pVal alias needs full f32 lanes).
//   IF_DBUF / IF_SBUF          two tile buffers ping-ponged with ONE barrier
//                              per K-slice (stage k0+BK into the idle buffer
//                              while the live one computes) / classic
//                              stage-barrier-compute-barrier single buffer
//   IF_SPLITK / KSL            split-K for starved small-N decode sites
//                              (fc2 at B=128: 7×2 = 14 workgroups): grid.z
//                              partitions K into KSL-sized (BK-aligned)
//                              ranges, each workgroup stores its RAW f32
//                              partial tile to PART [nz, M, N] — bias/SiLU
//                              move to gemm_reduce.wgsl, which folds the nz
//                              slices. Excludes WQ8/ARGMAX (guarded at
//                              dispatch; epilogues would double-apply).
{{ENABLE_IMMEDIATE}}
{{ENABLE_F16}}

struct Dims { M: u32, K: u32, N: u32, _pad: u32 }

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> dims: Dims;
@group(0) @binding(1) var<storage, read> X: array<vec4<{{T}}>>;
{{IF_WQF}}
@group(0) @binding(2) var<storage, read> W: array<vec4<{{T}}>>;
@group(0) @binding(3) var<storage, read> B: array<{{T}}>;
{{/IF_WQF}}
{{IF_WQ8}}
@group(0) @binding(2) var<storage, read> W: array<u32>; // 4×i8 along K
@group(0) @binding(3) var<storage, read> B: array<{{T}}>;
{{/IF_WQ8}}
{{IF_STORE_Y}}
@group(0) @binding(4) var<storage, read_write> Y: array<{{OUT_T}}>;
{{/IF_STORE_Y}}
{{IF_ARGMAX}}
// (val bitcast to u32, vocab idx) per [row, column-tile] — row-major [M, NT].
@group(0) @binding(4) var<storage, read_write> P: array<vec2<u32>>;
{{/IF_ARGMAX}}
{{IF_SPLITK}}
// Raw f32 partials per K-partition — [nz, M, N]; gemm_reduce folds them.
@group(0) @binding(4) var<storage, read_write> PART: array<f32>;
{{/IF_SPLITK}}
{{IF_WQ8}}
@group(0) @binding(5) var<storage, read> S: array<f32>; // per-N scales
{{/IF_WQ8}}
{{IF_ARGMAX}}
{{IF_WQF}}
@group(0) @binding(5) var<storage, read> LB: array<f32>;   // final_logits_bias [N]
@group(0) @binding(6) var<storage, read> SEEN: array<u32>; // rep bitmask [M·MASK_WORDS]
{{/IF_WQF}}
{{IF_WQ8}}
@group(0) @binding(6) var<storage, read> LB: array<f32>;   // final_logits_bias [N]
@group(0) @binding(7) var<storage, read> SEEN: array<u32>; // rep bitmask [M·MASK_WORDS]
{{/IF_WQ8}}
{{/IF_ARGMAX}}

const BM = {{BM}}u;
const BN = {{BN}}u;
const BK = {{BK}}u;
const TM = 4u{{IF_TM8}} * 2u{{/IF_TM8}}; // register subtile rows
const BM4 = BM / 4u;
const BN4 = BN / 4u;
const BK4 = BK / 4u;
const THREADS = (BM / TM) * BN4;

// Same transposed layouts as v1 (Xs[kk][row], Ws[kk][col]), vec4-packed
// along the second axis so the inner loop reads whole quads. ST is the
// shared-lane type (see IF_SH16 header note); NBUF the tile buffer count.
{{IF_SH16}}
alias ST = {{T}};
{{/IF_SH16}}
{{IF_SH32}}
alias ST = f32;
{{/IF_SH32}}
const NBUF = {{IF_DBUF}}2u{{/IF_DBUF}}{{IF_SBUF}}1u{{/IF_SBUF}};
var<workgroup> Xs: array<vec4<ST>, NBUF * BK * BM4>;
var<workgroup> Ws: array<vec4<ST>, NBUF * BK * BN4>;

{{IF_WQ8}}
// Sign-extend one packed u32 (4×i8 along K) to a vec4 of raw q values.
fn q8quad(word: i32) -> vec4<f32> {
  return vec4<f32>(
    f32((word << 24u) >> 24u),
    f32((word << 16u) >> 24u),
    f32((word << 8u) >> 24u),
    f32(word >> 24u));
}
{{/IF_WQ8}}

// Stage one BK-slice of X and W into tile buffer \`buf\`.
//
// PORTABILITY INVARIANT: every shared vec4 slot is written WHOLE by exactly
// one thread — each thread transposes a 4×4 register block (four vec4 global
// reads along K, four whole-vec4 shared stores). Per-component stores to one
// shared vec4 from different threads are legal WGSL (components are distinct
// memory locations) but miscompile on Metal into a whole-vector RMW: on an
// Apple M5 Max every v2 variant read back zeros/garbage (2026-07-06) until
// the staging was restructured. Do not reintroduce \`Xs[...][i] = scalar\`
// writes from multiple threads.
fn stageTiles(lid: u32, rowBase: u32, colBase: u32, k0: u32, buf: u32) {
  let K4 = dims.K / 4u; // K % 4 == 0 enforced by dispatch
  let xo = buf * (BK * BM4);
  let wo = buf * (BK * BN4);
  // Xs: 4-row × 4-k register transpose into the Xs[kk][row-quad] layout.
  // i → (rq, j) over BM4 × BK4 blocks; reads stay coalesced along K.
  for (var i: u32 = lid; i < BM4 * BK4; i = i + THREADS) {
    let rq = i / BK4;
    let j = i % BK4;
    let r0 = rowBase + 4u * rq;
    var v0 = vec4<f32>(0.0);
    var v1 = vec4<f32>(0.0);
    var v2 = vec4<f32>(0.0);
    var v3 = vec4<f32>(0.0);
    if (k0 + 4u * j < dims.K) {
      if (r0 < dims.M) { v0 = vec4<f32>(X[r0 * K4 + k0 / 4u + j]); }
      if (r0 + 1u < dims.M) { v1 = vec4<f32>(X[(r0 + 1u) * K4 + k0 / 4u + j]); }
      if (r0 + 2u < dims.M) { v2 = vec4<f32>(X[(r0 + 2u) * K4 + k0 / 4u + j]); }
      if (r0 + 3u < dims.M) { v3 = vec4<f32>(X[(r0 + 3u) * K4 + k0 / 4u + j]); }
    }
    let kk = 4u * j;
    Xs[xo + kk * BM4 + rq] = vec4<ST>(vec4<f32>(v0.x, v1.x, v2.x, v3.x));
    Xs[xo + (kk + 1u) * BM4 + rq] = vec4<ST>(vec4<f32>(v0.y, v1.y, v2.y, v3.y));
    Xs[xo + (kk + 2u) * BM4 + rq] = vec4<ST>(vec4<f32>(v0.z, v1.z, v2.z, v3.z));
    Xs[xo + (kk + 3u) * BM4 + rq] = vec4<ST>(vec4<f32>(v0.w, v1.w, v2.w, v3.w));
  }
  {{IF_WT}}
  // W [N,K] row-major: same 4-col × 4-k register transpose as Xs.
  for (var i: u32 = lid; i < BN4 * BK4; i = i + THREADS) {
    let cq = i / BK4;
    let j = i % BK4;
    let c0 = colBase + 4u * cq;
    var v0 = vec4<f32>(0.0);
    var v1 = vec4<f32>(0.0);
    var v2 = vec4<f32>(0.0);
    var v3 = vec4<f32>(0.0);
    if (k0 + 4u * j < dims.K) {
      if (c0 < dims.N) { v0 = vec4<f32>(W[c0 * K4 + k0 / 4u + j]); }
      if (c0 + 1u < dims.N) { v1 = vec4<f32>(W[(c0 + 1u) * K4 + k0 / 4u + j]); }
      if (c0 + 2u < dims.N) { v2 = vec4<f32>(W[(c0 + 2u) * K4 + k0 / 4u + j]); }
      if (c0 + 3u < dims.N) { v3 = vec4<f32>(W[(c0 + 3u) * K4 + k0 / 4u + j]); }
    }
    let kk = 4u * j;
    Ws[wo + kk * BN4 + cq] = vec4<ST>(vec4<f32>(v0.x, v1.x, v2.x, v3.x));
    Ws[wo + (kk + 1u) * BN4 + cq] = vec4<ST>(vec4<f32>(v0.y, v1.y, v2.y, v3.y));
    Ws[wo + (kk + 2u) * BN4 + cq] = vec4<ST>(vec4<f32>(v0.z, v1.z, v2.z, v3.z));
    Ws[wo + (kk + 3u) * BN4 + cq] = vec4<ST>(vec4<f32>(v0.w, v1.w, v2.w, v3.w));
  }
  {{/IF_WT}}
  {{IF_WNT}}
  // W [K,N] row-major: vec4 reads along N land directly on one Ws quad
  // (N % 4 == 0 enforced, so quads never straddle the boundary).
  let N4 = dims.N / 4u;
  for (var i: u32 = lid; i < BK * BN4; i = i + THREADS) {
    let kk = i / BN4;
    let c = i % BN4;
    var v = vec4<f32>(0.0);
    if (colBase + 4u * c < dims.N && k0 + kk < dims.K) {
      v = vec4<f32>(W[(k0 + kk) * N4 + colBase / 4u + c]);
    }
    Ws[wo + i] = vec4<ST>(v);
  }
  {{/IF_WNT}}
  {{IF_WQ8}}
  // W int8 [N,K] packed: one u32 = 4 raw q values along K, sign-extended
  // by shifts and staged UNSCALED (the per-N scale is an epilogue factor).
  // Same 4-col × 4-k register transpose as the WT path.
  for (var i: u32 = lid; i < BN4 * BK4; i = i + THREADS) {
    let cq = i / BK4;
    let j = i % BK4;
    let c0 = colBase + 4u * cq;
    var v0 = vec4<f32>(0.0);
    var v1 = vec4<f32>(0.0);
    var v2 = vec4<f32>(0.0);
    var v3 = vec4<f32>(0.0);
    if (k0 + 4u * j < dims.K) {
      if (c0 < dims.N) { v0 = q8quad(bitcast<i32>(W[c0 * K4 + k0 / 4u + j])); }
      if (c0 + 1u < dims.N) { v1 = q8quad(bitcast<i32>(W[(c0 + 1u) * K4 + k0 / 4u + j])); }
      if (c0 + 2u < dims.N) { v2 = q8quad(bitcast<i32>(W[(c0 + 2u) * K4 + k0 / 4u + j])); }
      if (c0 + 3u < dims.N) { v3 = q8quad(bitcast<i32>(W[(c0 + 3u) * K4 + k0 / 4u + j])); }
    }
    let kk = 4u * j;
    Ws[wo + kk * BN4 + cq] = vec4<ST>(vec4<f32>(v0.x, v1.x, v2.x, v3.x));
    Ws[wo + (kk + 1u) * BN4 + cq] = vec4<ST>(vec4<f32>(v0.y, v1.y, v2.y, v3.y));
    Ws[wo + (kk + 2u) * BN4 + cq] = vec4<ST>(vec4<f32>(v0.z, v1.z, v2.z, v3.z));
    Ws[wo + (kk + 3u) * BN4 + cq] = vec4<ST>(vec4<f32>(v0.w, v1.w, v2.w, v3.w));
  }
  {{/IF_WQ8}}
}

{{IF_STORE_Y}}
fn storeRow(m: u32, n0: u32, v: vec4<f32>) {
  if (m >= dims.M) { return; }
  let base = m * dims.N;
  if (n0 + 3u < dims.N) {
    Y[base + n0] = {{OUT_T}}(v.x);
    Y[base + n0 + 1u] = {{OUT_T}}(v.y);
    Y[base + n0 + 2u] = {{OUT_T}}(v.z);
    Y[base + n0 + 3u] = {{OUT_T}}(v.w);
    return;
  }
  // N tail (v1-fallback shapes never reach here, but WT allows N % 4 != 0).
  if (n0 < dims.N) { Y[base + n0] = {{OUT_T}}(v.x); }
  if (n0 + 1u < dims.N) { Y[base + n0 + 1u] = {{OUT_T}}(v.y); }
  if (n0 + 2u < dims.N) { Y[base + n0 + 2u] = {{OUT_T}}(v.z); }
}
{{/IF_STORE_Y}}
{{IF_SPLITK}}
const KSL = {{KSL}}u; // BK-aligned K range per grid.z partition
fn storePart(z: u32, m: u32, n0: u32, v: vec4<f32>) {
  if (m >= dims.M) { return; }
  let base = (z * dims.M + m) * dims.N + n0;
  if (n0 + 3u < dims.N) {
    PART[base] = v.x;
    PART[base + 1u] = v.y;
    PART[base + 2u] = v.z;
    PART[base + 3u] = v.w;
    return;
  }
  if (n0 < dims.N) { PART[base] = v.x; }
  if (n0 + 1u < dims.N) { PART[base + 1u] = v.y; }
  if (n0 + 2u < dims.N) { PART[base + 2u] = v.z; }
}
{{/IF_SPLITK}}
{{IF_ARGMAX}}
const NEG_MAX: f32 = -3.40282e38; // finite f32 lowest (WGSL has no inf literal)
const PENALTY: f32 = {{PENALTY}};
const MASK_WORDS: u32 = {{MASK_WORDS}}u;
// Per-row partials for the cross-thread fold: pVal aliases Xs, which is dead
// after the K loop (f32 values through f32 lanes are bit-preserving; ARGMAX
// requires SH32 so the lanes are full f32). pIdx must NOT alias Ws — small
// u32 indices are f32 denormal bit patterns, and a shared-memory round trip
// through f32 may flush them to zero.
// pVal slots are laid out [col-quad][row] (slot = c4·BM + row) so one
// thread's TM row-partials land in the SAME Xs vec4s and are stored whole —
// the staging portability invariant applies to this reuse too.
const_assert(BK * BM >= BM * BN4); // Xs lane capacity covers the BM·BN4 slots
var<workgroup> pIdx: array<u32, BM * BN4>;

fn penal(x: f32, seen: bool) -> f32 {
  if (!seen) { return x; }
  if (x > 0.0) { return x / PENALTY; }
  return x * PENALTY;
}

// One register row: bias + penalty + 4-lane max. Ascending lane order with
// strict > keeps the lowest-index tie-break (torch.argmax first-max) exact.
// A quad never straddles a bitmask word: n0 % 4 == 0 so (n0 & 31) <= 28.
// Writes pIdx (scalar array — per-slot stores are race-free) and RETURNS the
// partial value; the caller packs TM of them into whole Xs vec4 stores.
fn fusedPartial(m: u32, slot: u32, n0: u32, acc: vec4<f32>, lbq: vec4<f32>) -> f32 {
  var val: f32 = NEG_MAX;
  var idx: u32 = 0xffffffffu;
  if (m < dims.M && n0 < dims.N) {
    let x = acc + lbq;
    let bits = SEEN[m * MASK_WORDS + (n0 >> 5u)] >> (n0 & 31u);
    var v1 = NEG_MAX;
    var v2 = NEG_MAX;
    var v3 = NEG_MAX;
    if (n0 + 1u < dims.N) { v1 = penal(x.y, ((bits >> 1u) & 1u) == 1u); }
    if (n0 + 2u < dims.N) { v2 = penal(x.z, ((bits >> 2u) & 1u) == 1u); }
    if (n0 + 3u < dims.N) { v3 = penal(x.w, ((bits >> 3u) & 1u) == 1u); }
    val = penal(x.x, (bits & 1u) == 1u);
    idx = n0;
    if (v1 > val) { val = v1; idx = n0 + 1u; }
    if (v2 > val) { val = v2; idx = n0 + 2u; }
    if (v3 > val) { val = v3; idx = n0 + 3u; }
  }
  pIdx[slot] = idx;
  return val;
}
{{/IF_ARGMAX}}

fn biasQuad(n0: u32) -> vec4<f32> {
  // min-clamped reads: OOB cols load a dummy lane that storeRow never writes.
  let nHi = dims.N - 1u;
  return vec4<f32>(
    f32(B[min(n0, nHi)]),
    f32(B[min(n0 + 1u, nHi)]),
    f32(B[min(n0 + 2u, nHi)]),
    f32(B[min(n0 + 3u, nHi)]));
}

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  _ = &B; // keep B in the auto layout when the bias block is compiled out
  let rowBase = wid.y * BM;
  let colBase = wid.x * BN;
  let tRow = (lid / BN4) * TM;
  let tCol = (lid % BN4) * 4u;
  let r4 = tRow / 4u;
  let c4 = tCol / 4u;

  var acc0 = vec4<f32>(0.0); // row tRow+0, cols tCol..tCol+3
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);
  {{IF_TM8}}
  var acc4 = vec4<f32>(0.0);
  var acc5 = vec4<f32>(0.0);
  var acc6 = vec4<f32>(0.0);
  var acc7 = vec4<f32>(0.0);
  {{/IF_TM8}}

  {{IF_SPLITK}}
  let kBeg = wid.z * KSL;
  let kEnd = min(kBeg + KSL, dims.K);
  {{/IF_SPLITK}}
  {{IF_NOSPLITK}}
  let kBeg = 0u;
  let kEnd = dims.K;
  {{/IF_NOSPLITK}}
  {{IF_DBUF}}
  stageTiles(lid, rowBase, colBase, kBeg, 0u);
  workgroupBarrier();
  {{/IF_DBUF}}
  var buf = 0u;
  for (var k0: u32 = kBeg; k0 < kEnd; k0 = k0 + BK) {
    {{IF_DBUF}}
    // Prefetch the NEXT slice into the idle buffer while this one computes.
    // The single trailing barrier both publishes this prefetch for the next
    // iteration and retires the live buffer's reads before iteration
    // k0 + 2·BK overwrites it — safe with one barrier per slice.
    if (k0 + BK < kEnd) { stageTiles(lid, rowBase, colBase, k0 + BK, 1u - buf); }
    {{/IF_DBUF}}
    {{IF_SBUF}}
    stageTiles(lid, rowBase, colBase, k0, 0u);
    workgroupBarrier();
    {{/IF_SBUF}}
    let xo = buf * (BK * BM4);
    let wo = buf * (BK * BN4);
    for (var kk: u32 = 0u; kk < BK; kk = kk + 1u) {
      let wr = vec4<f32>(Ws[wo + kk * BN4 + c4]);
      let xr = vec4<f32>(Xs[xo + kk * BM4 + r4]);
      acc0 = fma(vec4<f32>(xr.x), wr, acc0);
      acc1 = fma(vec4<f32>(xr.y), wr, acc1);
      acc2 = fma(vec4<f32>(xr.z), wr, acc2);
      acc3 = fma(vec4<f32>(xr.w), wr, acc3);
      {{IF_TM8}}
      let xr2 = vec4<f32>(Xs[xo + kk * BM4 + r4 + 1u]);
      acc4 = fma(vec4<f32>(xr2.x), wr, acc4);
      acc5 = fma(vec4<f32>(xr2.y), wr, acc5);
      acc6 = fma(vec4<f32>(xr2.z), wr, acc6);
      acc7 = fma(vec4<f32>(xr2.w), wr, acc7);
      {{/IF_TM8}}
    }
    workgroupBarrier();
    {{IF_DBUF}}
    buf = 1u - buf;
    {{/IF_DBUF}}
  }

  {{IF_WQ8}}
  // Dequantize BEFORE bias: the per-N scale multiplies the whole column.
  let sHi = dims.N - 1u;
  let sn0 = colBase + tCol;
  let sv = vec4<f32>(
    S[min(sn0, sHi)], S[min(sn0 + 1u, sHi)], S[min(sn0 + 2u, sHi)], S[min(sn0 + 3u, sHi)]);
  acc0 = acc0 * sv;
  acc1 = acc1 * sv;
  acc2 = acc2 * sv;
  acc3 = acc3 * sv;
  {{IF_TM8}}
  acc4 = acc4 * sv;
  acc5 = acc5 * sv;
  acc6 = acc6 * sv;
  acc7 = acc7 * sv;
  {{/IF_TM8}}
  {{/IF_WQ8}}
  {{IF_BIAS}}
  let bv = biasQuad(colBase + tCol);
  acc0 = acc0 + bv;
  acc1 = acc1 + bv;
  acc2 = acc2 + bv;
  acc3 = acc3 + bv;
  {{IF_TM8}}
  acc4 = acc4 + bv;
  acc5 = acc5 + bv;
  acc6 = acc6 + bv;
  acc7 = acc7 + bv;
  {{/IF_TM8}}
  {{/IF_BIAS}}
  {{IF_SILU}}
  acc0 = acc0 / (vec4<f32>(1.0) + exp(-acc0));
  acc1 = acc1 / (vec4<f32>(1.0) + exp(-acc1));
  acc2 = acc2 / (vec4<f32>(1.0) + exp(-acc2));
  acc3 = acc3 / (vec4<f32>(1.0) + exp(-acc3));
  {{IF_TM8}}
  acc4 = acc4 / (vec4<f32>(1.0) + exp(-acc4));
  acc5 = acc5 / (vec4<f32>(1.0) + exp(-acc5));
  acc6 = acc6 / (vec4<f32>(1.0) + exp(-acc6));
  acc7 = acc7 / (vec4<f32>(1.0) + exp(-acc7));
  {{/IF_TM8}}
  {{/IF_SILU}}

  {{IF_STORE_Y}}
  storeRow(rowBase + tRow, colBase + tCol, acc0);
  storeRow(rowBase + tRow + 1u, colBase + tCol, acc1);
  storeRow(rowBase + tRow + 2u, colBase + tCol, acc2);
  storeRow(rowBase + tRow + 3u, colBase + tCol, acc3);
  {{IF_TM8}}
  storeRow(rowBase + tRow + 4u, colBase + tCol, acc4);
  storeRow(rowBase + tRow + 5u, colBase + tCol, acc5);
  storeRow(rowBase + tRow + 6u, colBase + tCol, acc6);
  storeRow(rowBase + tRow + 7u, colBase + tCol, acc7);
  {{/IF_TM8}}
  {{/IF_STORE_Y}}
  {{IF_SPLITK}}
  storePart(wid.z, rowBase + tRow, colBase + tCol, acc0);
  storePart(wid.z, rowBase + tRow + 1u, colBase + tCol, acc1);
  storePart(wid.z, rowBase + tRow + 2u, colBase + tCol, acc2);
  storePart(wid.z, rowBase + tRow + 3u, colBase + tCol, acc3);
  {{IF_TM8}}
  storePart(wid.z, rowBase + tRow + 4u, colBase + tCol, acc4);
  storePart(wid.z, rowBase + tRow + 5u, colBase + tCol, acc5);
  storePart(wid.z, rowBase + tRow + 6u, colBase + tCol, acc6);
  storePart(wid.z, rowBase + tRow + 7u, colBase + tCol, acc7);
  {{/IF_TM8}}
  {{/IF_SPLITK}}
  {{IF_ARGMAX}}
  // The K loop's trailing barrier already ordered every thread's Xs reads
  // before this reuse — fusedPartial may write pVal into Xs immediately.
  // Slot layout [col-quad][row]: this thread's TM partials are consecutive
  // rows of ONE column-quad, so they pack into whole Xs vec4 stores
  // (tRow % 4 == 0 by construction — the portability invariant again).
  let nHiL = dims.N - 1u;
  let n0 = colBase + tCol;
  let lbq = vec4<f32>(
    LB[min(n0, nHiL)], LB[min(n0 + 1u, nHiL)], LB[min(n0 + 2u, nHiL)], LB[min(n0 + 3u, nHiL)]);
  let pBase = c4 * BM + tRow;
  let p0 = fusedPartial(rowBase + tRow, pBase, n0, acc0, lbq);
  let p1 = fusedPartial(rowBase + tRow + 1u, pBase + 1u, n0, acc1, lbq);
  let p2 = fusedPartial(rowBase + tRow + 2u, pBase + 2u, n0, acc2, lbq);
  let p3 = fusedPartial(rowBase + tRow + 3u, pBase + 3u, n0, acc3, lbq);
  Xs[pBase / 4u] = vec4<ST>(vec4<f32>(p0, p1, p2, p3));
  {{IF_TM8}}
  let p4 = fusedPartial(rowBase + tRow + 4u, pBase + 4u, n0, acc4, lbq);
  let p5 = fusedPartial(rowBase + tRow + 5u, pBase + 5u, n0, acc5, lbq);
  let p6 = fusedPartial(rowBase + tRow + 6u, pBase + 6u, n0, acc6, lbq);
  let p7 = fusedPartial(rowBase + tRow + 7u, pBase + 7u, n0, acc7, lbq);
  Xs[pBase / 4u + 1u] = vec4<ST>(vec4<f32>(p4, p5, p6, p7));
  {{/IF_TM8}}
  workgroupBarrier();

  // Fold each row's BN4 column-quads (ascending c = ascending vocab ids, so
  // the explicit lower-idx tie clause and scan order both preserve first-max)
  // and emit the workgroup's one partial per row.
  let NT = (dims.N + BN - 1u) / BN;
  for (var r = lid; r < BM; r = r + THREADS) {
    var val: f32 = NEG_MAX;
    var idx: u32 = 0xffffffffu;
    for (var c = 0u; c < BN4; c = c + 1u) {
      let s = c * BM + r;
      let v = f32(Xs[s / 4u][s % 4u]);
      let i = pIdx[s];
      if (v > val || (v == val && i < idx)) { val = v; idx = i; }
    }
    if (rowBase + r < dims.M) {
      P[(rowBase + r) * NT + wid.x] = vec2<u32>(bitcast<u32>(val), idx);
    }
  }
  {{/IF_ARGMAX}}
}
`,ce=`// Unified scaled-dot-product attention — the ONE kernel for encoder self,
// decoder self, and cross attention (no transposes anywhere in the engine).
//
//   Q: element (row, h, d) at row*Q_STRIDE + Q_OFF + h*D + d, row = b*M + m.
//      With the defaults (Q_STRIDE = H·D, Q_OFF = 0) that is a plain
//      [B·M, H·D] row-major buffer.
//   K, V: element (b, j, h, d) at (b*L + j)*KV_STRIDE + K_OFF|V_OFF + h*D + d.
//      Defaults (KV_STRIDE = H·D, offsets 0) reproduce the [B, L, H, D]
//      cache layout: ((b*L + j)*H + h)*D + d. Non-default strides/offsets let
//      the kernel read q/k/v slices DIRECTLY from fused projection outputs
//      (encoder self: Q=K=V= fused qkv [B·S, 3·H·D] with stride 3·H·D and
//      offsets 0/H·D/2·H·D; cross: K=V= fused kv [B·S, 2·H·D]). K and V may
//      be bound to the same buffer (both read-only storage).
//   Y: always compact [B·M, H·D] row-major, regardless of Q layout.
//         L is the K/V buffer's position CAPACITY (stride) — the valid length
//         is selected by lenMode: 0 → t+1 (decoder self, uniform across
//         batch), 1 → lens[b] (encoder self / cross, right-padded batches).
//
// Q/K/V are bound as vec4 arrays (Task 17: 4× fewer loads in the phase-1 dot
// and phase-3 accumulation): D and every stride/offset (Q_STRIDE, Q_OFF,
// KV_STRIDE, K_OFF, V_OFF — and h·D by D%4) must be multiples of 4;
// dispatchAttention enforces this. Y stays a scalar array.
//
// Grid: dispatchWorkgroups(B·M, H) — wid.x = query row, wid.y = head. One
// workgroup computes one (row, head) output vector of D elements:
//   phase 0: stage q into shared qs4 (D4 = D/4 vec4s)
//   phase 1: scores_j = ATTN_SCALE·Σ_d qs·K, tree-reduce MAX (threads stride j)
//   phase 2: exponentiate in place, tree-reduce SUM → denom
//   phase 3: out_dq = Σ_j scores_j·V: thread (dq, jg) owns d-quad dq and sums
//            j ≡ jg (mod JT) into a shared partial; the first D4 threads fold
//            the JT partials and store. (Previously one thread per d with a
//            SERIAL j loop — the profiled decode hotspot of this kernel.)
// All math in f32 regardless of storage type; exp after max-subtraction;
// single division at the end.
//
// Template placeholders (buildShader in pipelines.js):
//   ENABLE_F16  the f16 enable directive when T is f16, else empty
//   T           storage type of Q/K/V/Y (f16|f32)
//   WG          workgroup size (128; must be a power of two, >= D)
//   H           heads (8)
//   D           head dim (56)
//   SCORES_CAP  shared scores capacity (352 — max valid length)
//   ATTN_SCALE  1/sqrt(D) as a full-precision literal
//   Q_STRIDE    per-row element stride of Q (default H·D = 448)
//   Q_OFF       element offset of the q slice within a Q row (default 0)
//   KV_STRIDE   per-position element stride of K/V (default H·D = 448)
//   K_OFF       element offset of the k slice within a K position (default 0)
//   V_OFF       element offset of the v slice within a V position (default 0)
//   IF_SG/IF_NOSG  subgroup reduction variant (flags.sg): subgroupMax/
//               subgroupAdd + elect + serial fold over per-subgroup partials
//               in two SEPARATE arrays (no red[] reuse). Besides the barrier
//               savings, this is the CORRECTNESS route on Adreno 7xx: its
//               driver miscompiles this kernel's tree-reduce idiom (wave-1
//               re-writes of red[] read as stale phase-1 values across a
//               barrier — 2026-07 Android probe rounds; the naked idiom in
//               isolation passes, so the trigger is contextual). Reduction
//               order differs from the tree → tolerance-equal, not bit-equal.
{{ENABLE_IMMEDIATE}}
{{ENABLE_F16}}
{{ENABLE_SG}}

struct Params {
  B: u32,        // batch
  M: u32,        // query rows per batch element (encoder S, else 1)
  L: u32,        // K/V position capacity (stride)
  lenMode: u32,  // 0: len = t+1, 1: len = lens[b]
  t: u32,        // decode step (lenMode 0 only)
  _pad0: u32, _pad1: u32, _pad2: u32,
}

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> params: Params;
@group(0) @binding(1) var<storage, read> Q: array<vec4<{{T}}>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<{{T}}>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<{{T}}>>;
// Always bound — a 4-byte dummy when lenMode == 0 (never indexed then).
@group(0) @binding(4) var<storage, read> lens: array<u32>;
@group(0) @binding(5) var<storage, read_write> Y: array<{{T}}>;

const H: u32 = {{H}}u;
const D: u32 = {{D}}u;
const WG: u32 = {{WG}}u;
const SCORES_CAP: u32 = {{SCORES_CAP}}u;
const ATTN_SCALE: f32 = {{ATTN_SCALE}};
const Q_STRIDE: u32 = {{Q_STRIDE}}u;
const Q_OFF: u32 = {{Q_OFF}}u;
const KV_STRIDE: u32 = {{KV_STRIDE}}u;
const K_OFF: u32 = {{K_OFF}}u;
const V_OFF: u32 = {{V_OFF}}u;
const D4: u32 = D / 4u;   // d-quads per head (D%4 == 0 enforced)
const JT: u32 = WG / D4;  // phase-3 j-lanes per d-quad

var<workgroup> qs4: array<vec4<f32>, D4>;      // staged query vector
var<workgroup> scores: array<f32, SCORES_CAP>; // raw scores, then exp values
{{IF_NOSG}}
var<workgroup> red: array<f32, WG>;            // tree-reduction scratch
{{/IF_NOSG}}
{{IF_SG}}
// One slot per subgroup, separate arrays per reduction (max, then sum) so
// the sum pass never overwrites slots the max readers still need. WG/4
// covers the spec-minimum subgroup size of 4.
var<workgroup> sgMax: array<f32, WG / 4u>;
var<workgroup> sgSum: array<f32, WG / 4u>;
{{/IF_SG}}
var<workgroup> part: array<vec4<f32>, WG>;     // phase-3 partial V sums

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>{{IF_SG}}, @builtin(subgroup_size) sgSize: u32{{/IF_SG}}) {
  // Uniform per workgroup (whole workgroup shares wid) — safe early return
  // before the first barrier.
  if (wid.x >= params.B * params.M) { return; }
  let row = wid.x;
  let h = wid.y;
  let b = row / params.M;
  let tid = lid.x;

  var len: u32;
  if (params.lenMode == 0u) { len = params.t + 1u; } else { len = lens[b]; }
  // Engine guarantees len <= SCORES_CAP; clamp anyway — cheap insurance
  // against out-of-bounds shared writes on a bad Params.
  len = min(len, SCORES_CAP);

  // Phase 0: stage q as D4 vec4s (D4=14 < WG → one quad per thread).
  let qoff4 = (row * Q_STRIDE + Q_OFF + h * D) / 4u;
  for (var i = tid; i < D4; i = i + WG) {
    qs4[i] = vec4<f32>(Q[qoff4 + i]);
  }
  workgroupBarrier(); // qs4 must be visible before phase-1 dot products

  // Phase 1: scores + running max. Threads stride j; idle threads (j >= len)
  // still contribute the neutral element -1e30 to the max reduction.
  var localMax: f32 = -1e30;
  for (var j = tid; j < len; j = j + WG) {
    let koff4 = ((b * params.L + j) * KV_STRIDE + K_OFF + h * D) / 4u;
    var dot4 = vec4<f32>(0.0);
    for (var i = 0u; i < D4; i = i + 1u) {
      dot4 = dot4 + qs4[i] * vec4<f32>(K[koff4 + i]);
    }
    let sc = (dot4.x + dot4.y + dot4.z + dot4.w) * ATTN_SCALE;
    scores[j] = sc;
    localMax = max(localMax, sc);
  }
{{IF_NOSG}}
  red[tid] = localMax;
  workgroupBarrier(); // red fully written (and scores writes made visible)
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { red[tid] = max(red[tid], red[tid + s]); }
    workgroupBarrier();
  }
  let rowMax = red[0];
  workgroupBarrier(); // all reads of red[0] done before phase 2 reuses red
{{/IF_NOSG}}
{{IF_SG}}
  // sgId assumes linear subgroup layout (tid/sgSize) — holds on
  // Metal/D3D12/Vulkan under Tint; the sg equiv gates catch a device where
  // it doesn't. subgroup ops sit in uniform control flow — outside the elect.
  let sgId = tid / sgSize;
  let nSg = (WG + sgSize - 1u) / sgSize;
  let m1 = subgroupMax(localMax);
  if (subgroupElect()) { sgMax[sgId] = m1; }
  workgroupBarrier(); // sgMax written (and scores writes made visible)
  var rowMax = sgMax[0];
  for (var i = 1u; i < nSg; i = i + 1u) { rowMax = max(rowMax, sgMax[i]); }
{{/IF_SG}}

  // Phase 2: exponentiate in place + sum. Idle threads contribute 0.
  var localSum: f32 = 0.0;
  for (var j = tid; j < len; j = j + WG) {
    let e = exp(scores[j] - rowMax);
    scores[j] = e;
    localSum = localSum + e;
  }
{{IF_NOSG}}
  red[tid] = localSum;
  workgroupBarrier(); // red fully written (and exp'd scores made visible)
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
    workgroupBarrier();
  }
  let denom = red[0];
  // No further barrier: red is not reused, and the exp'd scores reads below
  // were ordered by the reduction barriers above.
{{/IF_NOSG}}
{{IF_SG}}
  let s2 = subgroupAdd(localSum);
  if (subgroupElect()) { sgSum[sgId] = s2; }
  workgroupBarrier(); // sgSum written (and exp'd scores made visible)
  var denom = sgSum[0];
  for (var i = 1u; i < nSg; i = i + 1u) { denom = denom + sgSum[i]; }
{{/IF_SG}}

  // Phase 3: thread (dq = tid%D4, jg = tid/D4) accumulates d-quad dq over
  // j ≡ jg (mod JT); the first D4 threads fold the JT partials and store.
  // Y is always compact [B·M, H·D] even when Q is strided into a fused buffer.
  let dq = tid % D4;
  let jg = tid / D4;
  var acc = vec4<f32>(0.0);
  if (jg < JT) {
    for (var j = jg; j < len; j = j + JT) {
      let voff4 = ((b * params.L + j) * KV_STRIDE + V_OFF + h * D) / 4u;
      acc = acc + scores[j] * vec4<f32>(V[voff4 + dq]);
    }
  }
  part[tid] = acc; // threads with jg >= JT park a zero in an unread slot
  workgroupBarrier();
  if (tid < D4) {
    var out = vec4<f32>(0.0);
    for (var g = 0u; g < JT; g = g + 1u) { out = out + part[g * D4 + tid]; }
    out = out / denom;
    let yoff = row * H * D + h * D + tid * 4u;
    Y[yoff] = {{T}}(out.x);
    Y[yoff + 1u] = {{T}}(out.y);
    Y[yoff + 2u] = {{T}}(out.z);
    Y[yoff + 3u] = {{T}}(out.w);
  }
}
`,le=`// Blocked (flash-style) scaled-dot-product attention — ENCODER SELF/FUSED
// ONLY (lenMode 1, ragged lens; decode keeps attention.wgsl, whose M=1 gains
// nothing from query blocking).
//
// Why it exists: attention.wgsl runs one workgroup per (query row, head), so
// K/V of a (batch, head) pair are re-read from global memory once per query
// row — S× redundant traffic at encoder shapes (enc_profile after the tiled
// GEMM round: attention ≈ 50% of encoder time). Here one workgroup owns QB
// query rows of one (b, h): K/V tiles of JB positions are staged in shared
// memory ONCE and reused by all QB queries (global K/V traffic ÷ QB), with a
// streaming online softmax so no full score row is ever materialized.
//
// Layouts match attention.wgsl exactly (fused-QKV strides/offsets, vec4
// bindings, compact Y): see that header. Grid is (ceil(M/QB), H, B).
//
// Threads: WG = QB·D4 — thread (q = tid/D4, dq = tid%D4) owns query q's
// d-quad dq. Its V-sum lives in ONE static vec4<f32> register accumulator
// (array accumulators spill under Tint/DXC — see gemm_tiled.wgsl header).
// Per j-tile:
//   stage   Ks/Vs[JB·D4] cooperatively, native {{T}} (no precision change:
//           attention.wgsl converts the same f16 values at load)
//   scores  QB·jn dot products over the WG threads → p[] (raw, f32)
//   update  one thread per query: running max m, rescale = exp(m_old − m),
//           exp scores in place, running sum
//   accum   acc = acc·rescale[q] + Σ_j p[q][j]·Vs[j][dq]
// Store: acc / rowSum, guarded for the M tail block. All math f32.
//
// Template placeholders (buildShader in pipelines.js):
//   ENABLE_F16  f16 enable directive when T is f16, else empty
//   T           storage type of Q/K/V/Y (f16|f32)
//   QB          query rows per workgroup (workgroup size = QB·D/4)
//   JB          K/V tile positions staged per iteration
//   H, D, ATTN_SCALE, Q_STRIDE, Q_OFF, KV_STRIDE, K_OFF, V_OFF
//               as in attention.wgsl (D%4 == 0 enforced by dispatch)
//   IF_PACKED / IF_NOPACKED
//               encoder row-packing: with PACKED, Q/K/V/Y hold T = Σ lens
//               rows (no pad rows) and sequence b occupies rows
//               starts[b] .. starts[b]+len — binding 6 carries starts.
//               Query blocks entirely past len early-exit (uniform: len and
//               starts are read-only storage loads at workgroup-uniform
//               indices). NOPACKED keeps the padded b·M / b·L bases.
{{ENABLE_IMMEDIATE}}
{{ENABLE_F16}}

struct Params {
  B: u32,        // batch
  M: u32,        // query rows per batch element (encoder S)
  L: u32,        // K/V position capacity (stride)
  lenMode: u32,  // must be 1 (dispatch enforces); len = lens[b]
  t: u32,        // unused here
  _pad0: u32, _pad1: u32, _pad2: u32,
}

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> params: Params;
@group(0) @binding(1) var<storage, read> Q: array<vec4<{{T}}>>;
@group(0) @binding(2) var<storage, read> K: array<vec4<{{T}}>>;
@group(0) @binding(3) var<storage, read> V: array<vec4<{{T}}>>;
@group(0) @binding(4) var<storage, read> lens: array<u32>;
@group(0) @binding(5) var<storage, read_write> Y: array<{{T}}>;
{{IF_PACKED}}
@group(0) @binding(6) var<storage, read> starts: array<u32>; // packed row offsets [B]
{{/IF_PACKED}}

const H: u32 = {{H}}u;
const D: u32 = {{D}}u;
const QB: u32 = {{QB}}u;
const JB: u32 = {{JB}}u;
const ATTN_SCALE: f32 = {{ATTN_SCALE}};
const Q_STRIDE: u32 = {{Q_STRIDE}}u;
const Q_OFF: u32 = {{Q_OFF}}u;
const KV_STRIDE: u32 = {{KV_STRIDE}}u;
const K_OFF: u32 = {{K_OFF}}u;
const V_OFF: u32 = {{V_OFF}}u;
const D4: u32 = D / 4u;
const WG: u32 = QB * D4;

var<workgroup> Qs: array<vec4<f32>, QB * D4>;   // staged query block
var<workgroup> Ks: array<vec4<{{T}}>, JB * D4>; // staged K tile (native T)
var<workgroup> Vs: array<vec4<{{T}}>, JB * D4>; // staged V tile (native T)
var<workgroup> p: array<f32, QB * JB>;          // tile scores, then exp values
var<workgroup> rowMax: array<f32, QB>;          // running max per query
var<workgroup> rowSum: array<f32, QB>;          // running exp-sum per query
var<workgroup> rescale: array<f32, QB>;         // exp(oldMax - newMax) per tile

@compute @workgroup_size(WG) // const-expr: QB·D4 threads
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let qb0 = wid.x * QB; // first query row of this block (within the batch elt)
  let h = wid.y;
  let b = wid.z;
  let tid = lid.x;
  let q = tid / D4;  // this thread's query within the block
  let dq = tid % D4; // this thread's d-quad

  // Engine guarantees len <= L; clamp anyway (bad Params must not overrun L).
  let len = min(lens[b], params.L);
{{IF_PACKED}}
  // Nothing to store for a query block entirely past len — uniform early exit
  // (before any barrier; qb0/len are workgroup-uniform).
  if (qb0 >= len) { return; }
  let qBase = starts[b];
  let kvBase = starts[b];
  let mEnd = len; // valid query rows of this sequence
{{/IF_PACKED}}
{{IF_NOPACKED}}
  let qBase = b * params.M;
  let kvBase = b * params.L;
  let mEnd = params.M;
{{/IF_NOPACKED}}

  // Stage the query block: one owned quad per thread. Tail-block rows past
  // mEnd stage a duplicate of the last valid row — computed but never stored.
  let qRow = min(qb0 + q, mEnd - 1u);
  Qs[tid] = vec4<f32>(Q[((qBase + qRow) * Q_STRIDE + Q_OFF + h * D) / 4u + dq]);
  if (dq == 0u) {
    rowMax[q] = -1e30;
    rowSum[q] = 0.0;
  }
  var acc = vec4<f32>(0.0);
  workgroupBarrier(); // Qs/rowMax/rowSum visible before the first tile

  // len is workgroup-uniform (lens[b], b from workgroup_id) — barriers inside
  // the loop are in uniform control flow.
  for (var j0 = 0u; j0 < len; j0 = j0 + JB) {
    let jn = min(JB, len - j0);

    // Stage K/V tiles: JB·D4 quads each, cooperative. Slots past jn stage a
    // duplicate of the last valid position — never read below.
    for (var i = tid; i < JB * D4; i = i + WG) {
      let jj = i / D4;
      let dd = i % D4;
      let base = (kvBase + j0 + min(jj, jn - 1u)) * KV_STRIDE + h * D;
      Ks[i] = K[(base + K_OFF) / 4u + dd];
      Vs[i] = V[(base + V_OFF) / 4u + dd];
    }
    workgroupBarrier(); // tiles staged

    // Scores: p[sq][sj] = ATTN_SCALE · (q_sq · k_sj) for sj < jn.
    for (var i = tid; i < QB * JB; i = i + WG) {
      let sq = i / JB;
      let sj = i % JB;
      if (sj < jn) {
        var dot4 = vec4<f32>(0.0);
        for (var d4 = 0u; d4 < D4; d4 = d4 + 1u) {
          dot4 = dot4 + Qs[sq * D4 + d4] * vec4<f32>(Ks[sj * D4 + d4]);
        }
        p[i] = (dot4.x + dot4.y + dot4.z + dot4.w) * ATTN_SCALE;
      }
    }
    workgroupBarrier(); // raw scores written

    // Online softmax bookkeeping — one thread per query row (serial jn scan;
    // negligible next to the QB·JB·D score FLOPs above).
    if (tid < QB) {
      var m = rowMax[tid];
      for (var jj = 0u; jj < jn; jj = jj + 1u) {
        m = max(m, p[tid * JB + jj]);
      }
      rescale[tid] = exp(rowMax[tid] - m); // first tile: exp(-1e30 - m) = 0
      var s = rowSum[tid] * rescale[tid];
      for (var jj = 0u; jj < jn; jj = jj + 1u) {
        let e = exp(p[tid * JB + jj] - m);
        p[tid * JB + jj] = e;
        s = s + e;
      }
      rowMax[tid] = m;
      rowSum[tid] = s;
    }
    workgroupBarrier(); // exp'd scores + rescale visible

    // Accumulate this tile's V contribution into the register accumulator.
    acc = acc * rescale[q];
    for (var jj = 0u; jj < jn; jj = jj + 1u) {
      acc = acc + p[q * JB + jj] * vec4<f32>(Vs[jj * D4 + dq]);
    }
    workgroupBarrier(); // all reads of Ks/Vs/p done before the next staging
  }

  // Store — Y rows mirror Q rows; skip tail-block rows past mEnd.
  if (qb0 + q < mEnd) {
    let out = acc / rowSum[q];
    let yoff = (qBase + qb0 + q) * H * D + h * D + dq * 4u;
    Y[yoff] = {{T}}(out.x);
    Y[yoff + 1u] = {{T}}(out.y);
    Y[yoff + 2u] = {{T}}(out.z);
    Y[yoff + 3u] = {{T}}(out.w);
  }
}
`,fe=`// Add + LayerNorm (post-LN): Y[r,i] = gamma[i]·(v-μ)/√(σ²+EPS) + beta[i],
// where v = X[r,i] + R[r,i], μ is the row mean and σ² the row population
// variance (Σ(v-μ)²/D — second pass over shared memory, not sumsq-μ², which
// is cancellation-prone). f32 math regardless of storage type.
//
// One workgroup per row: dispatchWorkgroups(rows). WG threads each own
// ceil(D/WG) strided elements (D=448, WG=256 → ≤2 each).
//
// Template placeholders (buildShader in pipelines.js):
//   ENABLE_F16  the f16 enable directive when T is f16, else empty
//   T           storage type of X/R/gamma/beta/Y (f16|f32)
//   WG          workgroup size (256; must be a power of two)
//   D           row width (d_model, 448)
//   EPS         layernorm epsilon literal
//   IF_SG/IF_NOSG  subgroup reduction variant (flags.sg): subgroupAdd folds
//               each subgroup's partials without barriers → 2 barriers per
//               row instead of ~18. Reduction ORDER differs from the tree,
//               so results are tolerance-equal, not bit-equal.
{{ENABLE_IMMEDIATE}}
{{ENABLE_F16}}
{{ENABLE_SG}}

struct Params { rows: u32, _pad0: u32, _pad1: u32, _pad2: u32 }

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> params: Params;
@group(0) @binding(1) var<storage, read> X: array<{{T}}>;
@group(0) @binding(2) var<storage, read> R: array<{{T}}>;
@group(0) @binding(3) var<storage, read> gamma: array<{{T}}>;
@group(0) @binding(4) var<storage, read> beta: array<{{T}}>;
@group(0) @binding(5) var<storage, read_write> Y: array<{{T}}>;

const D: u32 = {{D}}u;
const WG: u32 = {{WG}}u;

var<workgroup> vbuf: array<f32, D>;
{{IF_NOSG}}
var<workgroup> scratch: array<f32, WG>;
{{/IF_NOSG}}
{{IF_SG}}
// One slot per subgroup; two arrays so pass 2 never overwrites slots pass 1
// readers still need (saves a barrier). WG/4 covers the spec-minimum
// subgroup size of 4.
var<workgroup> sgSum: array<f32, WG / 4u>;
var<workgroup> sgSq: array<f32, WG / 4u>;
{{/IF_SG}}

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>{{IF_SG}}, @builtin(subgroup_size) sgSize: u32{{/IF_SG}}) {
  // Uniform per workgroup (one workgroup per row), so returning here before
  // any barrier is safe: the whole workgroup exits together.
  if (wid.x >= params.rows) { return; }
  let base = wid.x * D;
  let tid = lid.x;

  // Load v = X + R into shared memory, accumulating this thread's partial sum.
  var sum: f32 = 0.0;
  for (var i = tid; i < D; i = i + WG) {
    let v = f32(X[base + i]) + f32(R[base + i]);
    vbuf[i] = v;
    sum = sum + v;
  }
{{IF_NOSG}}
  scratch[tid] = sum;
  workgroupBarrier();

  // Tree-reduce the partial sums → row sum in scratch[0].
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { scratch[tid] = scratch[tid] + scratch[tid + s]; }
    workgroupBarrier();
  }
  let mu = scratch[0] / f32(D);
  // All reads of scratch[0] must complete before phase 2 overwrites scratch.
  workgroupBarrier();

  // Second pass: sum of squared deviations → population variance.
  var sq: f32 = 0.0;
  for (var i = tid; i < D; i = i + WG) {
    let dev = vbuf[i] - mu;
    sq = sq + dev * dev;
  }
  scratch[tid] = sq;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { scratch[tid] = scratch[tid] + scratch[tid + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(scratch[0] / f32(D) + {{EPS}});
{{/IF_NOSG}}
{{IF_SG}}
  // sgId assumes linear subgroup layout (tid/sgSize) — holds on
  // Metal/D3D12/Vulkan under Tint; the sg kernel equiv gate catches a device
  // where it doesn't.
  let sgId = tid / sgSize;
  let nSg = WG / sgSize;
  // subgroupAdd must sit in subgroup-uniform control flow — outside the elect.
  let s1 = subgroupAdd(sum);
  if (subgroupElect()) { sgSum[sgId] = s1; }
  workgroupBarrier(); // also publishes vbuf for pass 2
  var total = 0.0;
  for (var i = 0u; i < nSg; i = i + 1u) { total = total + sgSum[i]; }
  let mu = total / f32(D);

  // Second pass: sum of squared deviations → population variance.
  var sq: f32 = 0.0;
  for (var i = tid; i < D; i = i + WG) {
    let dev = vbuf[i] - mu;
    sq = sq + dev * dev;
  }
  let s2 = subgroupAdd(sq);
  if (subgroupElect()) { sgSq[sgId] = s2; }
  workgroupBarrier();
  var total2 = 0.0;
  for (var i = 0u; i < nSg; i = i + 1u) { total2 = total2 + sgSq[i]; }
  let inv = inverseSqrt(total2 / f32(D) + {{EPS}});
{{/IF_SG}}

  for (var i = tid; i < D; i = i + WG) {
    Y[base + i] = {{T}}(f32(gamma[i]) * (vbuf[i] - mu) * inv + f32(beta[i]));
  }
}
`,pe=`// Fused projection + residual add + LayerNorm for the decode step's three
// LN-terminated sites (self_out+ln1, cross_out+ln2, fc2+ln3):
//   Y[m,:] = LN( f16round(X[m,:]·W + bias) + R[m,:] ) · gamma + beta
// One workgroup per row — LN needs the whole N=448 output row, which caps the
// GEMM parallelism at M workgroups. That starves the GPU at large batch (the
// GEMV kernel launches ~100× more threads), so this kernel is routed at
// SMALL B only, where the decode step is dispatch-overhead-bound and merging
// two dispatches into one is worth more than GEMM occupancy (25 → 19
// dispatches per step).
//
// Numerics match the unfused pair as closely as split kernels allow: the
// projection result is rounded through the storage type BEFORE the residual
// add — exactly the Y-buffer round trip the unfused path performs — and the
// LN is the same two-pass form (mean, then Σ(v−μ)²; not the
// cancellation-prone sumsq−μ²) in f32. Only reduction/accumulation ORDER
// differs (per-thread serial K here vs TK-lane trees there) — f32 ULP-level,
// gated by m3/golden like every kernel-routing change.
//
// Layouts: X [M,K] row-major, W [K,N] row-major (the original NWT .weight
// tensors — NOT the .wt copies), N % 4 == 0, K % 4 == 0 (both hold for all
// three sites: K = 448 | 1792, N = 448). The X row is staged into shared
// memory once (vec4 loads) so the K-loop never touches global X.
//
// Template placeholders (buildShader in pipelines.js):
//   ENABLE_F16   f16 enable directive when T is f16
//   T            storage type of all tensors (f16|f32); math is f32
//   WG           workgroup size (128; power of two)
//   KDIM         K as a const (shared X row size) — one pipeline per K
//   D            output row width N (= d_model 448)
//   EPS          layernorm epsilon literal
//   IF_SG/IF_NOSG  subgroup LN reduction (flags.sg): subgroupAdd folds each
//                subgroup's partials without barriers — 2 LN barriers per
//                row instead of ~16 (same shape as add_layernorm.wgsl).
//                Tolerance-equal, not bit-equal (reduction order).
{{ENABLE_IMMEDIATE}}
{{ENABLE_F16}}
{{ENABLE_SG}}

struct Params { M: u32, _pad0: u32, _pad1: u32, _pad2: u32 }

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> params: Params;
@group(0) @binding(1) var<storage, read> X: array<vec4<{{T}}>>;      // [M, K/4]
@group(0) @binding(2) var<storage, read> W: array<vec4<{{T}}>>;      // [K, N/4]
@group(0) @binding(3) var<storage, read> B: array<{{T}}>;            // [N]
@group(0) @binding(4) var<storage, read> R: array<{{T}}>;            // [M, N] residual
@group(0) @binding(5) var<storage, read> gamma: array<{{T}}>;        // [N]
@group(0) @binding(6) var<storage, read> beta: array<{{T}}>;         // [N]
@group(0) @binding(7) var<storage, read_write> Y: array<{{T}}>;      // [M, N]

const K: u32 = {{KDIM}}u;
const D: u32 = {{D}}u;
const D4: u32 = D / 4u;
const K4: u32 = K / 4u;
const WG: u32 = {{WG}}u;

var<workgroup> Xs: array<f32, K>;      // the staged input row
var<workgroup> vbuf: array<f32, D>;    // v = f16round(gemm) + residual
{{IF_NOSG}}
var<workgroup> scratch: array<f32, WG>;
{{/IF_NOSG}}
{{IF_SG}}
var<workgroup> sgSum: array<f32, WG / 4u>; // one slot per subgroup (min size 4)
var<workgroup> sgSq: array<f32, WG / 4u>;
{{/IF_SG}}

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>{{IF_SG}}, @builtin(subgroup_size) sgSize: u32{{/IF_SG}}) {
  // Uniform per workgroup (one workgroup per row) — safe early return.
  if (wid.x >= params.M) { return; }
  let m = wid.x;
  let tid = lid.x;

  // Stage the X row: one vec4 global read per thread-iteration.
  for (var i = tid; i < K4; i = i + WG) {
    let xv = vec4<f32>(X[m * K4 + i]);
    Xs[4u * i] = xv.x;
    Xs[4u * i + 1u] = xv.y;
    Xs[4u * i + 2u] = xv.z;
    Xs[4u * i + 3u] = xv.w;
  }
  workgroupBarrier();

  // Projection: each thread owns output quad(s) q, walking the full K with a
  // 2-way unrolled chain (K is even at both sites). W reads along N are
  // coalesced across threads within each k-row.
  for (var q = tid; q < D4; q = q + WG) {
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    for (var k = 0u; k < K; k = k + 2u) {
      acc0 = fma(vec4<f32>(Xs[k]), vec4<f32>(W[k * D4 + q]), acc0);
      acc1 = fma(vec4<f32>(Xs[k + 1u]), vec4<f32>(W[(k + 1u) * D4 + q]), acc1);
    }
    let n0 = 4u * q;
    let bq = vec4<f32>(f32(B[n0]), f32(B[n0 + 1u]), f32(B[n0 + 2u]), f32(B[n0 + 3u]));
    // Round through the storage type BEFORE the residual add — replicating
    // the unfused path's Y-buffer round trip bit-for-bit.
    let g = vec4<{{T}}>(acc0 + acc1 + bq);
    let rbase = m * D + n0;
    vbuf[n0] = f32(g.x) + f32(R[rbase]);
    vbuf[n0 + 1u] = f32(g.y) + f32(R[rbase + 1u]);
    vbuf[n0 + 2u] = f32(g.z) + f32(R[rbase + 2u]);
    vbuf[n0 + 3u] = f32(g.w) + f32(R[rbase + 3u]);
  }
  workgroupBarrier();

  // LayerNorm — same two-pass structure as add_layernorm.wgsl.
  var sum: f32 = 0.0;
  for (var i = tid; i < D; i = i + WG) { sum = sum + vbuf[i]; }
{{IF_NOSG}}
  scratch[tid] = sum;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { scratch[tid] = scratch[tid] + scratch[tid + s]; }
    workgroupBarrier();
  }
  let mu = scratch[0] / f32(D);
  workgroupBarrier(); // scratch[0] reads before phase 2 overwrites

  var sq: f32 = 0.0;
  for (var i = tid; i < D; i = i + WG) {
    let dev = vbuf[i] - mu;
    sq = sq + dev * dev;
  }
  scratch[tid] = sq;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { scratch[tid] = scratch[tid] + scratch[tid + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(scratch[0] / f32(D) + {{EPS}});
{{/IF_NOSG}}
{{IF_SG}}
  // sgId assumes linear subgroup layout (tid/sgSize) — see add_layernorm.
  let sgId = tid / sgSize;
  let nSg = WG / sgSize;
  let s1 = subgroupAdd(sum); // subgroup-uniform flow: outside the elect
  if (subgroupElect()) { sgSum[sgId] = s1; }
  workgroupBarrier();
  var total = 0.0;
  for (var i = 0u; i < nSg; i = i + 1u) { total = total + sgSum[i]; }
  let mu = total / f32(D);

  var sq: f32 = 0.0;
  for (var i = tid; i < D; i = i + WG) {
    let dev = vbuf[i] - mu;
    sq = sq + dev * dev;
  }
  let s2 = subgroupAdd(sq);
  if (subgroupElect()) { sgSq[sgId] = s2; }
  workgroupBarrier();
  var total2 = 0.0;
  for (var i = 0u; i < nSg; i = i + 1u) { total2 = total2 + sgSq[i]; }
  let inv = inverseSqrt(total2 / f32(D) + {{EPS}});
{{/IF_SG}}

  let ybase = m * D;
  for (var i = tid; i < D; i = i + WG) {
    Y[ybase + i] = {{T}}(f32(gamma[i]) * (vbuf[i] - mu) * inv + f32(beta[i]));
  }
}
`,me=`// Split-K epilogue: fold the nz raw f32 partial slices a split-K
// gemm_tiled2 dispatch left in PART [nz, M, N] into Y [M, N], applying the
// bias/SiLU epilogue the GEMM skipped (it must run on the FULL sum, once).
// Ascending-z summation is deterministic; the only numerical difference vs
// the unsplit kernel is the f32 re-association at the nz seams.
//
// IF_STORE_KV (split-K self_qkv): Y is the fused q|k|v row [M, 3·HD]; the
// k and v slices additionally scatter into the decode caches
// [M, Lmax, HD] at position dims.t — the SAME {{OUT_T}} value written to Y,
// so the caches stay bit-identical to Y's slices (the kv_append contract,
// exactly like the GEMV epilogue it replaces).
//
// Template placeholders: ENABLE_F16, T (bias storage type), OUT_T, WG,
// IF_BIAS, IF_SILU, IF_STORE_KV.
{{ENABLE_IMMEDIATE}}
{{ENABLE_F16}}

struct Dims { M: u32, N: u32, NZ: u32, t: u32, Lmax: u32, _p0: u32, _p1: u32, _p2: u32 }

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> dims: Dims;
@group(0) @binding(1) var<storage, read> PART: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<{{T}}>;
@group(0) @binding(3) var<storage, read_write> Y: array<{{OUT_T}}>;
{{IF_STORE_KV}}
@group(0) @binding(4) var<storage, read_write> Kc: array<{{OUT_T}}>;
@group(0) @binding(5) var<storage, read_write> Vc: array<{{OUT_T}}>;
{{/IF_STORE_KV}}

@compute @workgroup_size({{WG}})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &B; // keep B in the auto layout when the bias block is compiled out
  let i = gid.x;
  let total = dims.M * dims.N;
  if (i >= total) { return; }
  var acc: f32 = 0.0;
  for (var z: u32 = 0u; z < dims.NZ; z = z + 1u) {
    acc = acc + PART[z * total + i];
  }
  {{IF_BIAS}}
  acc = acc + f32(B[i % dims.N]);
  {{/IF_BIAS}}
  {{IF_SILU}}
  acc = acc / (1.0 + exp(-acc));
  {{/IF_SILU}}
  let v = {{OUT_T}}(acc);
  Y[i] = v;
  {{IF_STORE_KV}}
  let HD = dims.N / 3u;
  let m = i / dims.N;
  let n = i % dims.N;
  if (n >= HD) {
    let dstBase = (m * dims.Lmax + dims.t) * HD;
    if (n < 2u * HD) {
      Kc[dstBase + (n - HD)] = v;
    } else {
      Vc[dstBase + (n - 2u * HD)] = v;
    }
  }
  {{/IF_STORE_KV}}
}
`,ve=`// Token + position embedding: Y[r,i] = f32(table[id,i])·EMBED_SCALE
//                                     + f32(posEmbed[pos,i])   (pos NOT scaled)
// One workgroup per row: dispatchWorkgroups(nRows). Two modes, selected at
// build time (exactly one of SRC_IDS / DECODE):
//   SRC_IDS (encoder): ids is src token ids [B*S]; id = ids[r], pos = r % s
//                      (batch rows are consecutive: r = b*S + m). PAD rows
//                      embed the pad token normally — masking happens in
//                      attention, not here. With PACKED (encoder row-packing:
//                      pad rows dropped, T = Σ lens rows) each word carries
//                      its own position: ids[r] = (pos << 16) | id — id fits
//                      (VOCAB 24000 < 2^16) and the dispatch enforces
//                      S < 2^16.
//   DECODE:            ids is the token ring [T_max*B]; row r = batch index b;
//                      id = DECODER_START when t == 0, else ids[(t-1)*batch + r];
//                      pos = t.
//
// Template placeholders (buildShader in pipelines.js):
//   ENABLE_F16       the f16 enable directive when T is f16, else empty
//   T                storage type of table/posEmbed/Y (f16|f32)
//   WG               workgroup size (224 → 2 elements per thread at D=448)
//   D                row width (d_model, 448)
//   EMBED_SCALE      √d_model as a full-precision literal
//   IF_SRC_IDS / IF_DECODE  mode blocks
//   DECODER_START    decoder start token id (DECODE mode only)
{{ENABLE_IMMEDIATE}}
{{ENABLE_F16}}

struct Params { nRows: u32, t: u32, batch: u32, s: u32 }

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> params: Params;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> table: array<{{T}}>;
@group(0) @binding(3) var<storage, read> posEmbed: array<{{T}}>;
@group(0) @binding(4) var<storage, read_write> Y: array<{{T}}>;

const D: u32 = {{D}}u;
const WG: u32 = {{WG}}u;

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  // Uniform per workgroup (one workgroup per row) — safe early return.
  if (wid.x >= params.nRows) { return; }
  let r = wid.x;
{{IF_SRC_IDS}}
{{IF_PACKED}}
  let id = ids[r] & 0xffffu;
  let pos = ids[r] >> 16u;
{{/IF_PACKED}}
{{IF_NOPACKED}}
  let id = ids[r];
  let pos = r % params.s;
{{/IF_NOPACKED}}
{{/IF_SRC_IDS}}
{{IF_DECODE}}
  var id: u32 = {{DECODER_START}}u;
  if (params.t != 0u) { id = ids[(params.t - 1u) * params.batch + r]; }
  let pos = params.t;
{{/IF_DECODE}}
  let toff = id * D;
  let poff = pos * D;
  let base = r * D;
  for (var i = lid.x; i < D; i = i + WG) {
    Y[base + i] = {{T}}(f32(table[toff + i]) * {{EMBED_SCALE}} + f32(posEmbed[poff + i]));
  }
}
`,ge=`// Row scatter for encoder row-packing: packed activations [T, N] → padded
// [B·S, N], where T = Σ lens[b] and padded row (b, m < lens[b]) comes from
// packed row starts[b] + m. Padding rows are left untouched — the destination
// arena buffer is zero-initialized by WebGPU, so they read as zeros
// downstream (decoder cross-attention masks j ≥ len and never reads them).
//
// One workgroup per padded row, vec4 element copies (N % 4 == 0 enforced by
// the dispatch). Early returns are uniform: wid-derived plus read-only
// storage loads at workgroup-uniform indices; there are no barriers.
//
// Template placeholders (buildShader in pipelines.js):
//   ENABLE_F16  f16 enable directive when T is f16, else empty
//   T           storage type of X/Y (f16|f32)
//   WG          workgroup size
{{ENABLE_IMMEDIATE}}
{{ENABLE_F16}}

struct Params { B: u32, S: u32, N4: u32, _pad: u32 }

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> params: Params;
@group(0) @binding(1) var<storage, read> starts: array<u32>; // packed row offsets [B]
@group(0) @binding(2) var<storage, read> lens: array<u32>;   // sequence lengths [B]
@group(0) @binding(3) var<storage, read> X: array<vec4<{{T}}>>;
@group(0) @binding(4) var<storage, read_write> Y: array<vec4<{{T}}>>;

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  if (wid.x >= params.B * params.S) { return; }
  let b = wid.x / params.S;
  let m = wid.x % params.S;
  if (m >= lens[b]) { return; }
  let src = (starts[b] + m) * params.N4;
  let dst = wid.x * params.N4;
  for (var i = lid.x; i < params.N4; i = i + {{WG}}u) {
    Y[dst + i] = X[src + i];
  }
}
`,he=`// In-place live-row gather for decode compaction: row map[i] moves to row i
// for i < rows, inside the SAME storage buffer. copyBufferToBuffer cannot use
// one GPUBuffer as both source and destination, so this compute pass performs
// the move.
//
// The compaction invariant makes the ordering safe: map is strictly ascending,
// hence map[i] >= i. Rows move only toward lower indices. One workgroup walks
// destination rows in ascending order and a storage barrier separates rows;
// writing row i can therefore never clobber a source needed by a later row.
//
// Offsets and lengths are u32 counts rather than vec4 counts because repetition
// mask rows are not necessarily 16-byte aligned.
{{ENABLE_IMMEDIATE}}

struct Params { rows: u32, rowStride: u32, copyLen: u32, _pad: u32 }

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> params: Params;
@group(0) @binding(1) var<storage, read> map: array<u32>;
@group(0) @binding(2) var<storage, read_write> data: array<u32>;

@compute @workgroup_size({{WG}})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  for (var i = 0u; i < params.rows; i = i + 1u) {
    let src = map[i];
    if (src != i) {
      let srcBase = src * params.rowStride;
      let dstBase = i * params.rowStride;
      for (var k = lid.x; k < params.copyLen; k = k + {{WG}}u) {
        data[dstBase + k] = data[srcBase + k];
      }
    }
    storageBarrier();
  }
}
`,we=`// Decode-step MEGAKERNEL: one workgroup computes one batch row's ENTIRE
// decoder layer — 8 dispatches collapse into 1 (small-B latency: the b1 step
// is dispatch-overhead-bound, ~15µs of fixed cost per dispatch against
// ~5-10µs kernels). With IF_EMBED (layer 0) the decode embedding folds in
// too, so a b1 step becomes: mega L0 → mega L1 → lm_head → argmax.
//
// Stages (workgroupBarrier between each; all math f32; every stage boundary
// value is ROUNDED THROUGH THE STORAGE TYPE first — replicating the unfused
// path's activation-buffer round trips, the gemm_row_ln precedent):
//   S0  x ← embed (L0: ring/table/pos, matching embed.wgsl DECODE) or the
//       global hidden buffer X (L1)
//   S1  qkv = x·Wqkvᵀ + b; q stays in shared, k|v quads store to the caches
//       at position t (the same rounded value — the kv_append contract)
//   S2  self-attention over positions 0..t (phase structure and score/exp/
//       fold order mirror attention.wgsl; heads loop serially)
//   S3  x = LN1(x + self_out(attn))
//   S4  q = cross_q(x)
//   S5  cross-attention over lens[b] encoder positions (fused crossKV k|v)
//   S6  x = LN2(x + cross_out(attn))
//   S7  ffn = SiLU(fc1(x))   (SiLU in f32 BEFORE the f16 round, as gemv)
//   S8  X ← LN3(x + fc2(ffn))   (written back to the global hidden buffer)
//
// All projections read the ORIGINAL [K, N] row-major \`.weight\` tensors with
// the gemm_row_ln access pattern: thread q owns output quad q, and at each k
// the threads read CONSECUTIVE quads of W's k-row — fully coalesced. (The
// first version walked the transposed [N,K] copies, one row per thread —
// every load touched 32 distinct lines and the whole kernel ran at ~4.5GB/s,
// 4.3× SLOWER than the chain it replaced. One workgroup has only ~8 warps of
// latency-hiding; coalescing is everything here.) Four independent
// accumulators per k-quad keep 4 loads in flight per thread. Every tensor is
// addressed inside the ONE weights buffer via compile-time vec4-element
// offsets (…4 defines = byteOffset/8; manifest offsets are 256-aligned so /8
// is exact). One pipeline per layer.
//
// NOT bit-exact vs the unfused chain (accumulation/reduction ORDER differs
// at every site) — routed like every kernel change: m3/golden gates + the
// step-0 + divergence-rate equiv (mega_equiv), e2e A/B decides the batch
// threshold.
//
// Shared budget (16KB): (2·HD4 + TMP4)·16 + SCORES_CAP·4 + WG·4 — 13,184B
// for MoxhiMT-30 (448/1792), 16,256B for HachimiMT-60 (576/2304); checked at
// dispatch. tmp4 is max(FFN4, HD4 + WG) quads: the attention phase-3 partial
// scratch tmp4[HD4 .. HD4+WG) must fit even when the model's FFN is small
// (q lives in [0..HD4) for self, out4 for cross; the fold result lands in
// [0..HD4) only after all partial reads).
//
// Template placeholders (buildShader in pipelines.js):
//   ENABLE_F16, T (must be f16 — the .wt copies only exist as f16), WG (256)
//   ENABLE_SG + IF_SG/IF_NOSG   subgroup wgMax/wgSum (flags.sg) — see below
//   IF_EMBED / IF_NOEMBED   layer-0 embedding fold (TABLE4/POS4/EMBED_SCALE/
//                           DECODER_START live inside IF_EMBED)
//   H, D, FFN4, LMAX, SCORES_CAP, ATTN_SCALE, EPS
//   QKVW4 QKVB4 OUTW4 OUTB4 LN1G4 LN1B4 CQW4 CQB4 COW4 COB4 LN2G4 LN2B4
//   FC1W4 FC1B4 FC2W4 FC2B4 LN3G4 LN3B4   per-tensor vec4 offsets into W
//
// SG mode (flags.sg): wgMax/wgSum are where this kernel's barriers live —
// each tree call is 10 workgroupBarriers, and one layer makes 38 of them
// (2 per attention head × 2 sides × H, 2 per LN × 3), ~400 barriers per
// step per layer. That is exactly what the megakernel pays on Apple GPUs
// (Metal mega_sweep: mega LOSES b1 there while winning −12% on NVIDIA).
// With sg each call is subgroupMax/Add → one partial per subgroup → ONE
// barrier → serial fold over ≤ WG/4 partials, ~5× fewer barriers overall.
{{ENABLE_IMMEDIATE}}
{{ENABLE_SG}}
{{ENABLE_F16}}

struct Params {
  B: u32,   // batch rows (grid.x)
  t: u32,   // decode step (cache position; embed pos; self len = t+1)
  S: u32,   // encoder crossKV position capacity (padded S)
  _pad: u32,
}

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> params: Params;
@group(0) @binding(1) var<storage, read> W: array<vec4<{{T}}>>;    // whole weights buffer
@group(0) @binding(2) var<storage, read> ring: array<u32>;         // token ring
@group(0) @binding(3) var<storage, read_write> Kc: array<vec4<{{T}}>>; // [B, LMAX, H·D]
@group(0) @binding(4) var<storage, read_write> Vc: array<vec4<{{T}}>>;
@group(0) @binding(5) var<storage, read> CKV: array<vec4<{{T}}>>;  // [B·S, 2·H·D] fused k|v
@group(0) @binding(6) var<storage, read> lens: array<u32>;
@group(0) @binding(7) var<storage, read_write> X: array<vec4<{{T}}>>; // hidden [B, H·D]

const H: u32 = {{H}}u;
const D: u32 = {{D}}u;
const D4: u32 = D / 4u;            // quads per head
const HD4: u32 = H * D4;           // quads per d_model row
const QKV4: u32 = 3u * HD4;        // fused q|k|v quads
const FFN4: u32 = {{FFN4}}u;       // ffn quads (FFN/4)
const KQ_FFN: u32 = FFN4;          // fc2 K quads (K = FFN)
const LMAX: u32 = {{LMAX}}u;
const SCORES_MAX: u32 = {{SCORES_CAP}}u;
const ATTN_SCALE: f32 = {{ATTN_SCALE}};
const WG: u32 = {{WG}}u;
const JT: u32 = WG / D4;           // attention phase-3 j-lanes per d-quad
const TMP4: u32 = max(FFN4, HD4 + WG); // ffn row AND attn partial scratch fit

var<workgroup> xs4: array<vec4<f32>, HD4>;   // hidden state (residual base)
var<workgroup> tmp4: array<vec4<f32>, TMP4>; // q / vbuf / ffn / attn partials
var<workgroup> out4: array<vec4<f32>, HD4>;  // stage outputs
var<workgroup> scores: array<f32, SCORES_MAX>;
{{IF_NOSG}}
var<workgroup> red: array<f32, WG>;
{{/IF_NOSG}}
{{IF_SG}}
// One partial per subgroup, in TWO alternating slots of NSG_CAP (WG/4 covers
// the spec-minimum subgroup size 4). The alternation is what buys the single
// barrier per call: call N's fold reads slot A strictly before every thread
// passes call N+1's barrier (slot B), and call N+2's elect-writes to slot A
// happen strictly after it — so no trailing barrier is needed to protect
// reuse. sgId = tid/sgSize assumes the linear tid→subgroup layout (same bet
// as add_layernorm.wgsl; the equiv gates catch a violating backend).
const NSG_CAP: u32 = WG / 4u;
var<workgroup> red: array<f32, 2u * NSG_CAP>;
var<private> sgId: u32;
var<private> nSg: u32;
var<private> redSlot: u32 = 0u;
{{/IF_SG}}

fn wgMax(tid: u32, v: f32) -> f32 {
{{IF_SG}}
  let s1 = subgroupMax(v);
  let base = redSlot * NSG_CAP;
  if (subgroupElect()) { red[base + sgId] = s1; }
  workgroupBarrier();
  var r = red[base];
  for (var i = 1u; i < nSg; i = i + 1u) { r = max(r, red[base + i]); }
  redSlot = 1u - redSlot;
  return r;
{{/IF_SG}}
{{IF_NOSG}}
  red[tid] = v;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { red[tid] = max(red[tid], red[tid + s]); }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier(); // red[0] reads done before the next reduction reuses red
  return r;
{{/IF_NOSG}}
}

fn wgSum(tid: u32, v: f32) -> f32 {
{{IF_SG}}
  let s1 = subgroupAdd(v);
  let base = redSlot * NSG_CAP;
  if (subgroupElect()) { red[base + sgId] = s1; }
  workgroupBarrier();
  var r = red[base];
  for (var i = 1u; i < nSg; i = i + 1u) { r = r + red[base + i]; }
  redSlot = 1u - redSlot;
  return r;
{{/IF_SG}}
{{IF_NOSG}}
  red[tid] = v;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
{{/IF_NOSG}}
}

// One GEMV output quad, gemm_row_ln-style: thread computes outputs
// 4·n4 .. 4·n4+3 from the [K, N] row-major W — at each k, threads read
// consecutive quads of the k-row (coalesced across the workgroup). srcSel
// picks the shared source (0 = xs4, 1 = out4, 2 = tmp4); kq = K/4 source
// quads, nq = N/4 output quads (the W row stride). Four independent
// accumulators keep 4 loads in flight; the fold order is fixed
// (a0+a1)+(a2+a3). Returns f32 WITHOUT rounding — the caller rounds/routes.
fn gemvQuad(wOff: u32, bOff: u32, n4: u32, kq: u32, nq: u32, srcSel: u32) -> vec4<f32> {
  var a0 = vec4<f32>(0.0);
  var a1 = vec4<f32>(0.0);
  var a2 = vec4<f32>(0.0);
  var a3 = vec4<f32>(0.0);
  for (var k4 = 0u; k4 < kq; k4 = k4 + 1u) {
    var xq: vec4<f32>;
    if (srcSel == 0u) { xq = xs4[k4]; }
    else if (srcSel == 1u) { xq = out4[k4]; }
    else { xq = tmp4[k4]; }
    let kBase = wOff + (k4 << 2u) * nq + n4;
    a0 = fma(vec4<f32>(xq.x), vec4<f32>(W[kBase]), a0);
    a1 = fma(vec4<f32>(xq.y), vec4<f32>(W[kBase + nq]), a1);
    a2 = fma(vec4<f32>(xq.z), vec4<f32>(W[kBase + 2u * nq]), a2);
    a3 = fma(vec4<f32>(xq.w), vec4<f32>(W[kBase + 3u * nq]), a3);
  }
  return (a0 + a1) + (a2 + a3) + vec4<f32>(W[bOff + n4]);
}

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>{{IF_SG}}, @builtin(subgroup_size) sgSize: u32{{/IF_SG}}) {
  // Uniform per workgroup — safe early return before the first barrier.
  if (wid.x >= params.B) { return; }
  let b = wid.x;
  let tid = lid.x;
  let t = params.t;
{{IF_SG}}
  sgId = tid / sgSize;
  nSg = (WG + sgSize - 1u) / sgSize;
{{/IF_SG}}

  // ---- S0: hidden state into xs4 ----
{{IF_EMBED}}
  // embed.wgsl DECODE semantics: id = DECODER_START at t=0, else the ring
  // token; pos = t; y = f16round(table·EMBED_SCALE + pos_embed).
  var id: u32 = {{DECODER_START}}u;
  if (t != 0u) { id = ring[(t - 1u) * params.B + b]; }
  for (var i = tid; i < HD4; i = i + WG) {
    let e = vec4<f32>(W[{{TABLE4}}u + id * HD4 + i]) * {{EMBED_SCALE}}
          + vec4<f32>(W[{{POS4}}u + t * HD4 + i]);
    xs4[i] = vec4<f32>(vec4<{{T}}>(e));
  }
{{/IF_EMBED}}
{{IF_NOEMBED}}
  // Phony use: only the embed fold reads the ring, but the binding must stay
  // statically used or layout 'auto' drops @binding(2) and the bind group
  // (which always supplies it) fails validation — killing the whole submit.
  _ = ring[0];
  for (var i = tid; i < HD4; i = i + WG) {
    xs4[i] = vec4<f32>(X[b * HD4 + i]);
  }
{{/IF_NOEMBED}}
  workgroupBarrier();

  // ---- S1: fused qkv projection; q → tmp4[0..HD4), k|v quads → caches ----
  let kvBase = (b * LMAX + t) * HD4;
  for (var n4 = tid; n4 < QKV4; n4 = n4 + WG) {
    let g = vec4<{{T}}>(gemvQuad({{QKVW4}}u, {{QKVB4}}u, n4, HD4, QKV4, 0u));
    if (n4 < HD4) { tmp4[n4] = vec4<f32>(g); }
    else if (n4 < 2u * HD4) { Kc[kvBase + n4 - HD4] = g; }
    else { Vc[kvBase + n4 - 2u * HD4] = g; }
  }
  workgroupBarrier();

  // ---- S2: self-attention over positions 0..t (attention.wgsl phases) ----
  {
    let len = min(t + 1u, LMAX);
    for (var h = 0u; h < H; h = h + 1u) {
      let hq = h * D4;
      var lm: f32 = -1e30;
      for (var j = tid; j < len; j = j + WG) {
        let koff = (b * LMAX + j) * HD4 + hq;
        var dot4 = vec4<f32>(0.0);
        for (var i = 0u; i < D4; i = i + 1u) {
          dot4 = dot4 + tmp4[hq + i] * vec4<f32>(Kc[koff + i]);
        }
        let sc = (dot4.x + dot4.y + dot4.z + dot4.w) * ATTN_SCALE;
        scores[j] = sc;
        lm = max(lm, sc);
      }
      let rowMax = wgMax(tid, lm);
      var ls: f32 = 0.0;
      for (var j = tid; j < len; j = j + WG) {
        let e = exp(scores[j] - rowMax);
        scores[j] = e;
        ls = ls + e;
      }
      let denom = wgSum(tid, ls);
      let dq = tid % D4;
      let jg = tid / D4;
      var acc = vec4<f32>(0.0);
      if (jg < JT) {
        for (var j = jg; j < len; j = j + JT) {
          acc = acc + scores[j] * vec4<f32>(Vc[(b * LMAX + j) * HD4 + hq + dq]);
        }
      }
      tmp4[HD4 + tid] = acc; // partial scratch; q region [0..HD4) untouched
      workgroupBarrier();
      if (tid < D4) {
        var o = vec4<f32>(0.0);
        for (var g = 0u; g < JT; g = g + 1u) { o = o + tmp4[HD4 + g * D4 + tid]; }
        out4[hq + tid] = vec4<f32>(vec4<{{T}}>(o / denom));
      }
      workgroupBarrier(); // out4 + partial reads done before the next head
    }
  }

  // ---- S3: x = LN1(x + self_out(attn)); vbuf = tmp4 ----
  for (var n4 = tid; n4 < HD4; n4 = n4 + WG) {
    let g = vec4<{{T}}>(gemvQuad({{OUTW4}}u, {{OUTB4}}u, n4, HD4, HD4, 1u));
    tmp4[n4] = vec4<f32>(g) + xs4[n4];
  }
  workgroupBarrier();
  {
    var s: f32 = 0.0;
    for (var i = tid; i < HD4; i = i + WG) {
      let v = tmp4[i];
      s = s + v.x + v.y + v.z + v.w;
    }
    let mu = wgSum(tid, s) / f32(H * D);
    var sq: f32 = 0.0;
    for (var i = tid; i < HD4; i = i + WG) {
      let dv = tmp4[i] - vec4<f32>(mu);
      sq = sq + dot(dv, dv);
    }
    let inv = inverseSqrt(wgSum(tid, sq) / f32(H * D) + {{EPS}});
    for (var i = tid; i < HD4; i = i + WG) {
      let o = vec4<f32>(W[{{LN1G4}}u + i]) * (tmp4[i] - vec4<f32>(mu)) * inv
            + vec4<f32>(W[{{LN1B4}}u + i]);
      xs4[i] = vec4<f32>(vec4<{{T}}>(o));
    }
  }
  workgroupBarrier();

  // ---- S4: q = cross_q(x) → out4 ----
  for (var n4 = tid; n4 < HD4; n4 = n4 + WG) {
    out4[n4] = vec4<f32>(vec4<{{T}}>(gemvQuad({{CQW4}}u, {{CQB4}}u, n4, HD4, HD4, 0u)));
  }
  workgroupBarrier();

  // ---- S5: cross-attention over lens[b] encoder positions → tmp4[0..HD4) ----
  {
    let len = min(lens[b], SCORES_MAX);
    for (var h = 0u; h < H; h = h + 1u) {
      let hq = h * D4;
      var lm: f32 = -1e30;
      for (var j = tid; j < len; j = j + WG) {
        let koff = (b * params.S + j) * 2u * HD4 + hq; // k slice at offset 0
        var dot4 = vec4<f32>(0.0);
        for (var i = 0u; i < D4; i = i + 1u) {
          dot4 = dot4 + out4[hq + i] * vec4<f32>(CKV[koff + i]);
        }
        let sc = (dot4.x + dot4.y + dot4.z + dot4.w) * ATTN_SCALE;
        scores[j] = sc;
        lm = max(lm, sc);
      }
      let rowMax = wgMax(tid, lm);
      var ls: f32 = 0.0;
      for (var j = tid; j < len; j = j + WG) {
        let e = exp(scores[j] - rowMax);
        scores[j] = e;
        ls = ls + e;
      }
      let denom = wgSum(tid, ls);
      let dq = tid % D4;
      let jg = tid / D4;
      var acc = vec4<f32>(0.0);
      if (jg < JT) {
        for (var j = jg; j < len; j = j + JT) {
          // v slice at element offset H·D within the fused k|v position
          acc = acc + scores[j] * vec4<f32>(CKV[(b * params.S + j) * 2u * HD4 + HD4 + hq + dq]);
        }
      }
      tmp4[HD4 + tid] = acc; // fold results land in [0..HD4) only afterwards
      workgroupBarrier();
      if (tid < D4) {
        var o = vec4<f32>(0.0);
        for (var g = 0u; g < JT; g = g + 1u) { o = o + tmp4[HD4 + g * D4 + tid]; }
        tmp4[hq + tid] = vec4<f32>(vec4<{{T}}>(o / denom));
      }
      workgroupBarrier();
    }
  }

  // ---- S6: x = LN2(x + cross_out(attn)); vbuf = out4 ----
  for (var n4 = tid; n4 < HD4; n4 = n4 + WG) {
    let g = vec4<{{T}}>(gemvQuad({{COW4}}u, {{COB4}}u, n4, HD4, HD4, 2u));
    out4[n4] = vec4<f32>(g) + xs4[n4];
  }
  workgroupBarrier();
  {
    var s: f32 = 0.0;
    for (var i = tid; i < HD4; i = i + WG) {
      let v = out4[i];
      s = s + v.x + v.y + v.z + v.w;
    }
    let mu = wgSum(tid, s) / f32(H * D);
    var sq: f32 = 0.0;
    for (var i = tid; i < HD4; i = i + WG) {
      let dv = out4[i] - vec4<f32>(mu);
      sq = sq + dot(dv, dv);
    }
    let inv = inverseSqrt(wgSum(tid, sq) / f32(H * D) + {{EPS}});
    for (var i = tid; i < HD4; i = i + WG) {
      let o = vec4<f32>(W[{{LN2G4}}u + i]) * (out4[i] - vec4<f32>(mu)) * inv
            + vec4<f32>(W[{{LN2B4}}u + i]);
      xs4[i] = vec4<f32>(vec4<{{T}}>(o));
    }
  }
  workgroupBarrier();

  // ---- S7: ffn = SiLU(fc1(x)) → tmp4[0..FFN4) (SiLU in f32, then round) ----
  for (var n4 = tid; n4 < FFN4; n4 = n4 + WG) {
    var v = gemvQuad({{FC1W4}}u, {{FC1B4}}u, n4, HD4, FFN4, 0u);
    v = v / (vec4<f32>(1.0) + exp(-v));
    tmp4[n4] = vec4<f32>(vec4<{{T}}>(v));
  }
  workgroupBarrier();

  // ---- S8: X ← LN3(x + fc2(ffn)); vbuf = out4 ----
  for (var n4 = tid; n4 < HD4; n4 = n4 + WG) {
    let g = vec4<{{T}}>(gemvQuad({{FC2W4}}u, {{FC2B4}}u, n4, KQ_FFN, HD4, 2u));
    out4[n4] = vec4<f32>(g) + xs4[n4];
  }
  workgroupBarrier();
  {
    var s: f32 = 0.0;
    for (var i = tid; i < HD4; i = i + WG) {
      let v = out4[i];
      s = s + v.x + v.y + v.z + v.w;
    }
    let mu = wgSum(tid, s) / f32(H * D);
    var sq: f32 = 0.0;
    for (var i = tid; i < HD4; i = i + WG) {
      let dv = out4[i] - vec4<f32>(mu);
      sq = sq + dot(dv, dv);
    }
    let inv = inverseSqrt(wgSum(tid, sq) / f32(H * D) + {{EPS}});
    for (var i = tid; i < HD4; i = i + WG) {
      let o = vec4<f32>(W[{{LN3G4}}u + i]) * (out4[i] - vec4<f32>(mu)) * inv
            + vec4<f32>(W[{{LN3B4}}u + i]);
      X[b * HD4 + i] = vec4<{{T}}>(o);
    }
  }
}
`,be=`// Repetition-penalty + greedy argmax + token writeback — the kernel that keeps
// the decode loop GPU-resident: logits → next token with no CPU roundtrip.
//
// One workgroup per batch row b: dispatchWorkgroups(B).
//   scan:    thread strides v = tid, tid+WG, …: x = logits[b·V+v] + bias[v]
//            (final_logits_bias is part of the model's logits BEFORE the
//            penalty runs, matching HF); if v's bit is set in the row's seen
//            bitmask: x = x>0 ? x/PENALTY : x·PENALTY. Strict > keeps the
//            EARLIEST max within a thread (v ascends per thread), so the
//            lowest-index tie-break (torch.argmax first-max) holds locally.
//   reduce:  tree reduction over (val, idx) pairs; comparator prefers higher
//            val, then LOWER idx on exact equality — global lowest-index ties.
//   epilogue (thread 0 only):
//            wasDone → token = PAD, no done/bitmask update;
//            else: token = argmax; token == EOS → done[b] = 1 (the EOS token
//            itself IS written to the ring); the picked token's bit is ALWAYS
//            set for a not-already-done row — including the EOS pick (HF
//            appends eos to input_ids too; harmless, subsequent picks are PAD
//            and skip the mask). Ring slot: t·B + b (embed at step t+1 reads
//            ((t+1)-1)·B + b — the same slot).
//
// Races: thread 0 is the ONLY writer of done/tokens/bitmask for its row, and
// other rows' workgroups touch disjoint slots — no atomics needed. The
// reduction loop's final workgroupBarrier() orders the epilogue after every
// thread's scan, so thread 0's bitmask |= cannot race this dispatch's own
// bitmask reads (within one dispatch the |= wouldn't be visible to sibling
// workgroups anyway, but the barrier makes the intra-workgroup ordering
// explicit). Cross-dispatch visibility (step t's bit seen by step t+1) is
// guaranteed by WebGPU dispatch/submission ordering.
//
// Template placeholders (buildShader in pipelines.js):
//   WG          workgroup size (256; must be a power of two)
//   V           vocab size (24000)
//   EOS / PAD   special token ids (2 / 0)
//   PENALTY     repetition penalty as a literal (1.2)
//   MASK_WORDS  bitmask u32 words per row (V/32 = 750)

{{ENABLE_IMMEDIATE}}

struct Params { B: u32, t: u32, _pad0: u32, _pad1: u32 }

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> params: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;         // [B·V]
@group(0) @binding(2) var<storage, read> bias: array<f32>;           // [V]
@group(0) @binding(3) var<storage, read_write> bitmask: array<u32>;  // [B·MASK_WORDS]
@group(0) @binding(4) var<storage, read_write> done: array<u32>;     // [B]
@group(0) @binding(5) var<storage, read_write> tokens: array<u32>;   // ring [T_max·B]
{{IF_SHORT}}
// Shortlisted lm_head (tiled-unfused arm): logits/bias/bitmask are all in
// LOCAL column space (V = shortlist length); IDMAP maps the winning local
// index to a vocab id, and gmask keeps the vocab-space mask in sync for any
// full-vocab state that inherits this row (see argmax_reduce.wgsl).
@group(0) @binding(6) var<storage, read> IDMAP: array<u32>;          // [N_short]
@group(0) @binding(7) var<storage, read_write> gmask: array<u32>;    // [B·GMASK_WORDS]
{{/IF_SHORT}}

const V: u32 = {{V}}u;
const WG: u32 = {{WG}}u;
const EOS: u32 = {{EOS}}u;
const PAD: u32 = {{PAD}}u;
const PENALTY: f32 = {{PENALTY}};
const MASK_WORDS: u32 = {{MASK_WORDS}}u;
{{IF_SHORT}}
const GMASK_WORDS: u32 = {{GMASK_WORDS}}u;
{{/IF_SHORT}}
// Finite f32 lowest — WGSL has no infinity literal (and -1.0/0.0 is a
// const-expr error). Logits + bias are finite reals, so this is safely below.
const NEG_MAX: f32 = -3.40282e38;

var<workgroup> bestVal: array<f32, WG>;
var<workgroup> bestIdx: array<u32, WG>;

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  // Uniform per workgroup (one workgroup per row) — safe early return.
  if (wid.x >= params.B) { return; }
  let b = wid.x;
  let tid = lid.x;
  let lbase = b * V;
  let mbase = b * MASK_WORDS;

  // Scan: strided over v, ascending per thread.
  var val: f32 = NEG_MAX;
  var idx: u32 = 0xffffffffu;
  for (var v = tid; v < V; v = v + WG) {
    var x = logits[lbase + v] + bias[v];
    let seen = (bitmask[mbase + (v >> 5u)] >> (v & 31u)) & 1u;
    if (seen == 1u) {
      if (x > 0.0) { x = x / PENALTY; } else { x = x * PENALTY; }
    }
    // Strict >: earliest (lowest-v) max wins within this thread.
    if (x > val) { val = x; idx = v; }
  }
  bestVal[tid] = val;
  bestIdx[tid] = idx;
  workgroupBarrier();

  // Tree reduce: higher val wins; exact-equal vals break to the LOWER idx.
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      let ov = bestVal[tid + s];
      let oi = bestIdx[tid + s];
      if (ov > bestVal[tid] || (ov == bestVal[tid] && oi < bestIdx[tid])) {
        bestVal[tid] = ov;
        bestIdx[tid] = oi;
      }
    }
    // Final iteration's barrier also orders the epilogue below after ALL
    // threads' scans/reduction writes (see races note in the header).
    workgroupBarrier();
  }

  // Epilogue: single writer per row.
  if (tid == 0u) {
    let wasDone = done[b];
    var tok = bestIdx[0];
{{IF_SHORT}}
    let lidx = tok;
    tok = IDMAP[lidx];
{{/IF_SHORT}}
    if (wasDone == 1u) {
      tok = PAD;
    } else {
      if (tok == EOS) { done[b] = 1u; }
{{IF_SHORT}}
      bitmask[mbase + (lidx >> 5u)] = bitmask[mbase + (lidx >> 5u)] | (1u << (lidx & 31u));
      let gbase = b * GMASK_WORDS;
      gmask[gbase + (tok >> 5u)] = gmask[gbase + (tok >> 5u)] | (1u << (tok & 31u));
{{/IF_SHORT}}
{{IF_NOSHORT}}
      bitmask[mbase + (tok >> 5u)] = bitmask[mbase + (tok >> 5u)] | (1u << (tok & 31u));
{{/IF_NOSHORT}}
    }
    tokens[params.t * params.B + b] = tok;
  }
}
`,_e=`// Stage 2 of the FUSED lm_head argmax (gemm_tiled2.wgsl IF_ARGMAX): fold the
// per-workgroup (val, idx) partials — final_logits_bias and the repetition
// penalty were already applied in the GEMM epilogue — into the row's token,
// then run argmax_penalty.wgsl's done/EOS/bitmask/ring epilogue VERBATIM.
// One workgroup per batch row; NT = ceil(V/BN) partials per row (375 at
// BN=64), so this dispatch reads ~3KB per row where argmax_penalty re-read
// 96KB of logits plus 96KB of bias.
//
// Tie-breaking matches argmax_penalty exactly: partial j covers vocab ids
// [j·BN, (j+1)·BN) — ascending j = ascending, disjoint id ranges — and each
// partial already carries its tile's lowest-index max, so a strict > over the
// per-thread ascending stride plus the (higher val, then LOWER idx) tree
// comparator reproduce torch.argmax first-max semantics globally.
//
// Template placeholders (buildShader in pipelines.js):
//   WG          workgroup size (256; must be a power of two)
//   EOS / PAD   special token ids (2 / 0)
//   MASK_WORDS  bitmask u32 words per row (V/32 = 750)

{{ENABLE_IMMEDIATE}}

struct Params { B: u32, t: u32, NT: u32, _pad0: u32 }

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> params: Params;
@group(0) @binding(1) var<storage, read> P: array<vec2<u32>>;        // [B·NT] (bitcast f32 val, idx)
@group(0) @binding(2) var<storage, read_write> bitmask: array<u32>;  // [B·MASK_WORDS]
@group(0) @binding(3) var<storage, read_write> done: array<u32>;     // [B]
@group(0) @binding(4) var<storage, read_write> tokens: array<u32>;   // ring [T_max·B]
{{IF_SHORT}}
// Shortlisted lm_head: P carries LOCAL column indices; IDMAP maps them to
// vocab ids. Two masks stay in sync — \`bitmask\` (binding 2) is the
// LOCAL-space mask the fused gemm epilogue reads next step, \`gmask\` the
// vocab-space mask that survives routing changes (GEMV/full-vocab states
// created by compaction keep reading it).
@group(0) @binding(5) var<storage, read> IDMAP: array<u32>;          // [N_short]
@group(0) @binding(6) var<storage, read_write> gmask: array<u32>;    // [B·GMASK_WORDS]
{{/IF_SHORT}}

const WG: u32 = {{WG}}u;
const EOS: u32 = {{EOS}}u;
const PAD: u32 = {{PAD}}u;
const MASK_WORDS: u32 = {{MASK_WORDS}}u;
{{IF_SHORT}}
const GMASK_WORDS: u32 = {{GMASK_WORDS}}u;
{{/IF_SHORT}}
// Finite f32 lowest (WGSL has no infinity literal). Real partials hold finite
// logits, so only all-OOB tile partials carry this value — and never win.
const NEG_MAX: f32 = -3.40282e38;

var<workgroup> bestVal: array<f32, WG>;
var<workgroup> bestIdx: array<u32, WG>;

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  // Uniform per workgroup (one workgroup per row) — safe early return.
  if (wid.x >= params.B) { return; }
  let b = wid.x;
  let tid = lid.x;
  let base = b * params.NT;

  var val: f32 = NEG_MAX;
  var idx: u32 = 0xffffffffu;
  for (var j = tid; j < params.NT; j = j + WG) {
    let p = P[base + j];
    let v = bitcast<f32>(p.x);
    // Strict >: ascending j = ascending vocab ranges, so equal-val ties keep
    // the earliest partial = the lower vocab id.
    if (v > val) { val = v; idx = p.y; }
  }
  bestVal[tid] = val;
  bestIdx[tid] = idx;
  workgroupBarrier();

  // Tree reduce: higher val wins; exact-equal vals break to the LOWER idx.
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      let ov = bestVal[tid + s];
      let oi = bestIdx[tid + s];
      if (ov > bestVal[tid] || (ov == bestVal[tid] && oi < bestIdx[tid])) {
        bestVal[tid] = ov;
        bestIdx[tid] = oi;
      }
    }
    // Final iteration's barrier also orders the epilogue below after ALL
    // threads' scans/reduction writes (see argmax_penalty.wgsl races note).
    workgroupBarrier();
  }

  // Epilogue: single writer per row — identical to argmax_penalty.wgsl.
  if (tid == 0u) {
    let mbase = b * MASK_WORDS;
    let wasDone = done[b];
    var tok = bestIdx[0];
{{IF_SHORT}}
    let lidx = tok;
    tok = IDMAP[lidx];
{{/IF_SHORT}}
    if (wasDone == 1u) {
      tok = PAD;
    } else {
      if (tok == EOS) { done[b] = 1u; }
{{IF_SHORT}}
      bitmask[mbase + (lidx >> 5u)] = bitmask[mbase + (lidx >> 5u)] | (1u << (lidx & 31u));
      let gbase = b * GMASK_WORDS;
      gmask[gbase + (tok >> 5u)] = gmask[gbase + (tok >> 5u)] | (1u << (tok & 31u));
{{/IF_SHORT}}
{{IF_NOSHORT}}
      bitmask[mbase + (tok >> 5u)] = bitmask[mbase + (tok >> 5u)] | (1u << (tok & 31u));
{{/IF_NOSHORT}}
    }
    tokens[params.t * params.B + b] = tok;
  }
}
`;function ae(e={}){const r=e.defines??{};return{t:e.t??"f32",outT:e.outT??e.t??"f32",wg:e.wg??64,bias:!!e.bias,silu:!!e.silu,wt:!!e.wt,sg:!!e.sg,immediate:!!e.immediate,defines:Object.fromEntries(Object.keys(r).sort().map(t=>[t,r[t]]))}}function Be(e,r={}){const{t,outT:c,wg:n,bias:s,silu:f,wt:d,sg:u,immediate:a,defines:p}=ae(r),l={T:t,OUT_T:c,WG:String(n),ENABLE_F16:t==="f16"||c==="f16"?"enable f16;":"",ENABLE_SG:u?"enable subgroups;":"",ENABLE_IMMEDIATE:a?"requires immediate_address_space;":"",PARAM_BINDING:a?"":"@group(0) @binding(0) ",PARAM_ADDRESS:a?"immediate":"uniform"},i={BIAS:!!s,SILU:!!f,WT:!!d,SG:!!u,NOSG:!u};for(const[h,o]of Object.entries(p))typeof o=="boolean"?i[h.toUpperCase()]=o:l[h.toUpperCase()]=String(o);let m=e;for(let h=null;h!==m;)h=m,m=m.replace(/\{\{IF_([A-Z0-9_]+)\}\}([\s\S]*?)\{\{\/IF_\1\}\}/g,(o,v,w)=>{if(!(v in i))throw new Error(`buildShader: unknown conditional {{IF_${v}}}`);return i[v]?w:""});return m=m.replace(/\{\{([A-Z0-9_/]+)\}\}/g,(h,o)=>{if(!(o in l))throw new Error(`buildShader: unresolved placeholder {{${o}}}`);return l[o]}),m}const V=new WeakMap,ke=256,Se=256*1024,Te=64,X=new WeakMap,C=new WeakMap,H=new WeakSet;let ye=1;function j(e){let r=C.get(e);return r||(r=ye++,C.set(e,r)),r}function F(e){let r=X.get(e);return r||(r={bindGroups:new Map,bindGroupLimit:ke,activeUniformPools:0,uniformPoolOriginalBindGroupLimit:null,dummyStorage:null,uniformFrame:null,stats:{uniformBuffersCreated:0,uniformPoolBuffersCreated:0,uniformPoolBuffersDestroyed:0,uniformPoolFramesBegun:0,uniformPoolFramesFlushed:0,uniformPoolBlocks:0,uniformPoolBytes:0,uniformPoolBindGroupCacheHits:0,uniformPoolWarmBindGroupLookups:0,uniformPoolWarmBindGroupCacheHits:0,uniformPoolWarmBindGroupResets:0,uniformPoolGenerationInvalidations:0,uniformPoolCachePurges:0,dummyBuffersCreated:0,bindGroupsCreated:0,bindGroupCacheHits:0,bindGroupEvictions:0,bindGroupTargetedPurgeCalls:0,bindGroupTargetedPurges:0,immediateSets:0}},X.set(e,r)),r}function Ge(e){const r=F(e);return{...r.stats,bindGroupCacheSize:r.bindGroups.size,bindGroupCacheLimit:r.bindGroupLimit}}function Oe(e,{clearCache:r=!1}={}){const t=F(e);for(const c of Object.keys(t.stats))t.stats[c]=0;r&&t.bindGroups.clear()}function Re(e,r){if(!Number.isInteger(r)||r<0)throw new Error(`bind-group cache limit must be a non-negative integer, got ${r}`);const t=F(e);for(t.bindGroupLimit=r;t.bindGroups.size>r;)t.bindGroups.delete(t.bindGroups.keys().next().value),t.stats.bindGroupEvictions++}function Ke(e,{B:r,immediate:t}={}){return!!e&&t===!1&&Number.isInteger(r)&&r>=1&&r<=Te}function qe(e,{banks:r=2,bankBytes:t=Se,alignment:c=e?.limits?.minUniformBufferOffsetAlignment??256}={}){if(!Number.isInteger(r)||r<1)throw new Error(`uniform pool banks must be a positive integer, got ${r}`);if(!Number.isInteger(c)||c<16||c%16!==0)throw new Error(`uniform pool alignment must be a positive multiple of 16, got ${c}`);if(!Number.isInteger(t)||t<c)throw new Error(`uniform pool bankBytes must be an integer >= alignment, got ${t}`);t=Math.ceil(t/c)*c;const n=F(e),s={uniformBuffersCreated:n.stats.uniformBuffersCreated,bindGroupsCreated:n.stats.bindGroupsCreated,bindGroupCacheHits:n.stats.bindGroupCacheHits,pooledBindGroupCacheHits:n.stats.uniformPoolBindGroupCacheHits,warmBindGroupLookups:n.stats.uniformPoolWarmBindGroupLookups,warmBindGroupCacheHits:n.stats.uniformPoolWarmBindGroupCacheHits,warmBindGroupResets:n.stats.uniformPoolWarmBindGroupResets,generationInvalidations:n.stats.uniformPoolGenerationInvalidations,cachePurges:n.stats.uniformPoolCachePurges},f={highWater:n.bindGroupLimit},d=Array.from({length:r},(o,v)=>{const w=e.createBuffer({label:`uniform params bank ${v}`,size:t,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});return H.add(w),{buffer:w,cpu:new Uint32Array(t/4),cursor:0,blocks:0,busy:!1,flushed:!1,highWaterBytes:0,highWaterBlocks:0}});n.activeUniformPools===0&&(n.uniformPoolOriginalBindGroupLimit=n.bindGroupLimit),n.activeUniformPools++;const u=n.uniformPoolOriginalBindGroupLimit>0;n.stats.uniformPoolBuffersCreated+=d.length;const a=d.map(o=>j(o.buffer)),p=()=>{let o=0;for(const v of[...n.bindGroups.keys()])a.some(w=>v.includes(`|${w}@`))&&(n.bindGroups.delete(v),n.stats.uniformPoolCachePurges++,o++);return o};let l=!1;const i=()=>{if(l)throw new Error("uniform pool is destroyed")},m=o=>{if(!Number.isInteger(o)||o<0||o>=d.length)throw new Error(`uniform pool bank ${o} out of range 0..${d.length-1}`);return d[o]},h={begin(o){if(i(),n.uniformFrame)throw new Error("uniform pool frame already active");const v=m(o);if(v.busy)throw new Error(`uniform pool bank ${o} reused while busy`);return v.cursor=0,v.blocks=0,v.busy=!0,v.flushed=!1,n.uniformFrame={pool:h,bank:v,index:o,alignment:c,bankBytes:t,cacheBanks:d.length,cacheEnabled:u,cacheLimitStats:f,warm:v.highWaterBlocks>0},n.stats.uniformPoolFramesBegun++,o},flush(){i();const o=n.uniformFrame;if(!o||o.pool!==h)throw new Error("uniform pool has no active frame to flush");const{bank:v,index:w}=o,g=Math.ceil(v.cursor/4)*4;return g>0&&e.queue.writeBuffer(v.buffer,0,v.cpu.buffer,v.cpu.byteOffset,g),v.flushed=!0,v.highWaterBytes=Math.max(v.highWaterBytes,g),v.highWaterBlocks=Math.max(v.highWaterBlocks,v.blocks),n.uniformFrame=null,n.stats.uniformPoolFramesFlushed++,n.stats.uniformPoolBlocks+=v.blocks,n.stats.uniformPoolBytes+=g,{bank:w,blocks:v.blocks,usedBytes:g}},abort(){i();const o=n.uniformFrame;if(!o||o.pool!==h)throw new Error("uniform pool has no active frame to abort");o.bank.cursor=0,o.bank.blocks=0,o.bank.busy=!1,o.bank.flushed=!1,n.uniformFrame=null},release(o){i();const v=m(o);if(n.uniformFrame?.bank===v)throw new Error(`uniform pool bank ${o} released while its frame is active`);if(!v.busy||!v.flushed)throw new Error(`uniform pool bank ${o} released before a flushed submission`);v.busy=!1,v.flushed=!1},invalidateBindings(){if(i(),n.uniformFrame?.pool===h)throw new Error("uniform pool generation invalidated while its frame is active");const o=d.findIndex(v=>v.busy);if(o>=0)throw new Error(`uniform pool bank ${o} is busy during generation invalidation`);return n.stats.uniformPoolGenerationInvalidations++,p()},snapshot(){return{banks:d.length,bankBytes:t,alignment:c,bindGroupCacheLimit:n.bindGroupLimit,bindGroupCacheLimitHighWater:f.highWater,busyBanks:d.filter(o=>o.busy).length,highWaterBytes:Math.max(0,...d.map(o=>o.highWaterBytes)),highWaterBlocks:Math.max(0,...d.map(o=>o.highWaterBlocks)),transientUniformBuffersCreated:n.stats.uniformBuffersCreated-s.uniformBuffersCreated,bindGroupsCreated:n.stats.bindGroupsCreated-s.bindGroupsCreated,bindGroupCacheHits:n.stats.bindGroupCacheHits-s.bindGroupCacheHits,pooledBindGroupCacheHits:n.stats.uniformPoolBindGroupCacheHits-s.pooledBindGroupCacheHits,warmBindGroupLookups:n.stats.uniformPoolWarmBindGroupLookups-s.warmBindGroupLookups,warmBindGroupCacheHits:n.stats.uniformPoolWarmBindGroupCacheHits-s.warmBindGroupCacheHits,warmBindGroupResets:n.stats.uniformPoolWarmBindGroupResets-s.warmBindGroupResets,generationInvalidations:n.stats.uniformPoolGenerationInvalidations-s.generationInvalidations,bindGroupsPurged:n.stats.uniformPoolCachePurges-s.cachePurges}},destroy(){if(!l){if(n.uniformFrame?.pool===h){const o=n.uniformFrame.bank;o.busy=!1,o.flushed=!1,n.uniformFrame=null}p();for(const o of d)H.delete(o.buffer),o.busy=!1,o.flushed=!1,o.buffer.destroy();if(n.activeUniformPools--,n.activeUniformPools===0)for(n.bindGroupLimit=n.uniformPoolOriginalBindGroupLimit,n.uniformPoolOriginalBindGroupLimit=null;n.bindGroups.size>n.bindGroupLimit;)n.bindGroups.delete(n.bindGroups.keys().next().value),n.stats.bindGroupEvictions++;n.stats.uniformPoolBuffersDestroyed+=d.length,l=!0}}};return h}function B(e,r,t,c={}){let n=V.get(e);n||(n=new Map,V.set(e,n));const s=`${r}:${JSON.stringify(ae(c))}`;let f=n.get(s);if(f){if(f.source!==t)throw new Error(`pipeline cache key collision: ${s}`);return f.pipeline}const d=e.createShaderModule({label:s,code:Be(t,c)}),u=e.createComputePipeline({label:s,layout:"auto",compute:{module:d,entryPoint:"main"}});return n.set(s,{pipeline:u,source:t}),u}function z(e){return e.buffer?e:{buffer:e}}function Ee(e,r,t){F(e).stats.uniformBuffersCreated++;const c=Math.max(16,Math.ceil(t.length*4/16)*16),n=e.createBuffer({label:r,size:c,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:!0});return new Uint32Array(n.getMappedRange()).set(t),n.unmap(),n}function q(e){const r=F(e);return r.dummyStorage||(r.dummyStorage=e.createBuffer({label:"shared dummy storage",size:4,usage:GPUBufferUsage.STORAGE}),r.stats.dummyBuffersCreated++),r.dummyStorage}function T(e,r,t,c){if(c)return{resource:null,values:Uint32Array.from(t),scratch:[]};const n=F(e),s=n.uniformFrame;if(s){const d=Math.max(16,Math.ceil(t.length*4/16)*16),u=Math.ceil(s.bank.cursor/s.alignment)*s.alignment,a=u+d;if(a>s.bankBytes)throw new Error(`uniform pool bank ${s.index} overflow: need ${a} bytes, cap ${s.bankBytes}`);return s.bank.cpu.fill(0,u/4,a/4),s.bank.cpu.set(t,u/4),s.bank.cursor=a,s.bank.blocks++,s.cacheEnabled&&(n.bindGroupLimit=Math.max(n.bindGroupLimit,s.bank.blocks*s.cacheBanks),s.cacheLimitStats.highWater=Math.max(s.cacheLimitStats.highWater,n.bindGroupLimit)),{resource:{buffer:s.bank.buffer,offset:u,size:d},values:null,scratch:[]}}const f=Ee(e,r,t);return{resource:f,values:null,scratch:[f]}}function y(e,r){return e.resource?[e.resource,...r]:r}function De(e,r,t){const c=r.map(n=>{const s=z(n);return`${j(s.buffer)}@${s.offset??0}:${s.size??"*"}`});return`${t?"i":"u"}|p${j(e)}|${c.join("|")}`}function S(e,r,t,c,n,s=1,f=1,d=null){const u=d?1:0,a=F(t),p=!d&&c.length>0&&H.has(z(c[0]).buffer);let l=null,i=null;if((d||p)&&a.bindGroupLimit>0){i=De(r,c,!!d),l=a.bindGroups.get(i)??null;const m=p&&a.uniformFrame?.warm;m&&a.stats.uniformPoolWarmBindGroupLookups++,l?(a.bindGroups.delete(i),a.bindGroups.set(i,l),a.stats.bindGroupCacheHits++,p&&(a.stats.uniformPoolBindGroupCacheHits++,a.uniformFrame?.warm&&a.stats.uniformPoolWarmBindGroupCacheHits++)):m&&(a.uniformFrame.warm=!1,a.stats.uniformPoolWarmBindGroupResets++)}if(l||(l=t.createBindGroup({layout:r.getBindGroupLayout(0),entries:c.map((m,h)=>({binding:h+u,resource:z(m)}))}),a.stats.bindGroupsCreated++,i&&(a.bindGroups.set(i,l),a.bindGroups.size>a.bindGroupLimit&&(a.bindGroups.delete(a.bindGroups.keys().next().value),a.stats.bindGroupEvictions++))),e.setPipeline(r),d){if(typeof e.setImmediates!="function")throw new Error("WebGPU immediate shader selected but pass.setImmediates is unavailable");e.setImmediates(0,d),a.stats.immediateSets++}e.setBindGroup(0,l),e.dispatchWorkgroups(n,s,f)}function Ie(e,r,{x:t,w:c,b:n=null,y:s,M:f,K:d,N:u,storeKV:a=null,scales:p=null,fusedArgmax:l=null,splitK:i=null,flags:m={}}){if(m.tiled&&m.gemv)throw new Error("flags.tiled and flags.gemv are exclusive");if(m.tiled){if(a)throw new Error("storeKV requires flags.gemv");return Ae(e,r,{x:t,w:c,b:n,y:s,M:f,K:d,N:u,scales:p,fusedArgmax:l,splitK:i,flags:m})}if(l)throw new Error("fusedArgmax requires flags.tiled");if(i)throw new Error("splitK requires flags.tiled");if(m.gemv)return Fe(e,r,{x:t,w:c,b:n,y:s,M:f,K:d,N:u,storeKV:a,scales:p,flags:m});if(p)throw new Error("scales (wq8) requires flags.tiled or flags.gemv");if(a)throw new Error("storeKV requires flags.gemv");const h=m.wg??64,o=B(e,"gemm",oe,{...m,bias:!!n}),v=T(e,"gemm dims",[f,d,u,0],!!m.immediate),w=[...v.scratch];let g=n;return g||(g=q(e)),S(r,o,e,y(v,[t,c,g,s]),Math.ceil(u/h),f,1,v.values),{pipeline:o,scratch:w}}function Fe(e,r,{x:t,w:c,b:n,y:s,M:f,K:d,N:u,storeKV:a=null,scales:p=null,flags:l}){const i=!!l.wt,m=!!l.wq8;if(m&&(!i||!p||a))throw new Error("gemv wq8: needs wt layout and scales, excludes storeKV");if(i&&d%4!==0)throw new Error(`gemv wt requires K%4==0, got K=${d}`);if(!i&&u%4!==0)throw new Error(`gemv requires N%4==0, got N=${u}`);if(a&&u%3!==0)throw new Error("storeKV requires fused QKV (N=3·H·D)");const h=l.tk??16,o=l.tn??(i?8:4),v=i?l.mt??8:1,w=B(e,"gemm_gemv",ie,{t:l.t,outT:l.outT,wg:h*o,bias:!!n,silu:l.silu,wt:i,immediate:!!l.immediate,sg:!!l.sg&&i,defines:{TK:h,TN:o,NWT:!i,STORE_KV:!!a,WQ8:m,WQF:!m,...i?{MT:v}:{}}}),g=T(e,"gemv dims",a?[f,d,u,0,a.t,a.Lmax,0,0]:[f,d,u,0],!!l.immediate),I=[...g.scratch];let k=n;k||(k=q(e));const b=Math.ceil(i?u/o:u/(4*o)),E=i?Math.ceil(f/v):f,_=[t,c,k,s];return a&&_.push(a.kCache,a.vCache),m&&_.push(p),S(r,w,e,y(g,_),b,E,1,g.values),{pipeline:w,scratch:I}}function Ne(e,r,t=16){const c=Math.ceil(e/r/t)*t;return{KSL:c,nz:Math.ceil(e/c)}}function Ae(e,r,{x:t,w:c,b:n,y:s,M:f,K:d,N:u,scales:a=null,fusedArgmax:p=null,splitK:l=null,flags:i}){const m=i.bm??64,h=i.bn??64,o=i.bkk??16;if(m%4!==0||h%4!==0)throw new Error(`gemm_tiled: BM/BN must be multiples of 4 (${m}, ${h})`);const v=!!i.sh16,w=!!i.dbuf;if(p&&(v||w))throw new Error("gemm_tiled2 fused argmax: sh16/dbuf unsupported (pVal aliases f32 Xs)");if(l){if(p)throw new Error("gemm_tiled2 splitK: exclusive with fusedArgmax");if(!(l.sk>=2))throw new Error(`gemm_tiled2 splitK: sk must be >= 2, got ${l.sk}`);if(n||i.silu)throw new Error("gemm_tiled2 splitK: pass bias/silu to dispatchGemmReduce, not the GEMM")}const g=p?m*(h/4)*4:0,I=v&&i.t==="f16"?2:4,k=(m+h)*o*I*(w?2:1)+g;if(k>16384)throw new Error(`gemm_tiled: shared memory ${k} bytes > 16384 limit`);if(p&&o<h/4)throw new Error(`gemm_tiled2 fused argmax: BK=${o} < BN/4=${h/4} — pVal cannot alias Xs`);const b=!!i.wq8;if(b&&(!a||!i.wt))throw new Error("gemm_tiled2 wq8: needs scales and wt layout");const E=d%4===0&&o%4===0&&(i.wt||u%4===0),_=(i.tiledV??(E?2:1))===2;if(_&&!E)throw new Error(`gemm_tiled2: shape M=${f} K=${d} N=${u} wt=${!!i.wt} BK=${o} not vec4-eligible`);if(b&&!_)throw new Error("gemm_tiled2 wq8: v1 fallback has no int8 path");if(p&&!_)throw new Error("gemm_tiled2 fused argmax: v2 only");if((v||w)&&!_)throw new Error("gemm_tiled2 sh16/dbuf: v2 only");if(l&&(!_||b))throw new Error("gemm_tiled2 splitK: v2 only, no wq8");const D=l?Ne(d,l.sk,o):null,x=_&&i.tm8?8:4;if(m%x!==0)throw new Error(`gemm_tiled: BM=${m} not a multiple of TM=${x}`);const W=m/x*(h/4);if(W>256)throw new Error(`gemm_tiled: ${W} threads > 256 workgroup limit`);const O=_?B(e,"gemm_tiled2",de,{t:i.t,outT:i.outT,wg:W,bias:!!n,silu:i.silu,immediate:!!i.immediate,wt:i.wt&&!b,defines:{BM:m,BN:h,BK:o,TM8:x===8,WNT:!i.wt&&!b,WQ8:b,WQF:!b,STORE_Y:!p&&!l,ARGMAX:!!p,SPLITK:!!l,NOSPLITK:!l,SH16:v,SH32:!v,DBUF:w,SBUF:!w,...l?{KSL:D.KSL}:{},...p?{PENALTY:J,MASK_WORDS:p.maskWords??K}:{}}}):B(e,"gemm_tiled",ue,{t:i.t,outT:i.outT,wg:W,bias:!!n,silu:i.silu,wt:i.wt,immediate:!!i.immediate,defines:{BM:m,BN:h,BK:o}}),M=T(e,"gemm_tiled dims",[f,d,u,0],!!i.immediate),R=[...M.scratch];let G=n;G||(G=q(e));const L=[t,c,G,p?.partials??l?.parts??s];return b&&L.push(a),p&&L.push(p.lbias,p.seen),S(r,O,e,y(M,L),Math.ceil(u/h),Math.ceil(f/m),D?.nz??1,M.values),{pipeline:O,scratch:R}}function Le(e,r,{parts:t,b:c=null,y:n,M:s,N:f,nz:d,storeKV:u=null,flags:a={}}){if(u&&f%3!==0)throw new Error("gemm_reduce storeKV requires fused QKV (N=3·H·D)");const p=a.wg??128,l=B(e,"gemm_reduce",me,{t:a.t,outT:a.outT,wg:p,bias:!!c,silu:!!a.silu,immediate:!!a.immediate,defines:{STORE_KV:!!u}}),i=T(e,"gemm_reduce dims",[s,f,d,u?.t??0,u?.Lmax??0,0,0,0],!!a.immediate),m=[...i.scratch];let h=c;h||(h=q(e));const o=[t,h,n];return u&&o.push(u.kCache,u.vCache),S(r,l,e,y(i,o),Math.ceil(s*f/p),1,1,i.values),{pipeline:l,scratch:m}}function xe({D4:e,t:r,qb:t=null,jb:c=null,qbAlign8:n=!1}){let s=t??Math.max(1,Math.min(16,Math.floor(256/e)));if(n&&t==null&&s*e%16!==0){for(;s>1&&s*e%16!==0;)s--;if(s*e%16!==0)throw new Error(`attention block: no QB <= 256 threads aligns D4=${e} to 16 threads`)}const f=r==="f16"?8:16,d=a=>s*e*16+2*a*e*f+s*a*4+3*s*4;let u=c??32;if(c==null)for(;u>8&&d(u)>16384;)u>>=1;return{QB:s,JB:u,shared:d(u)}}function Pe(e,r,{q:t,k:c,v:n,lens:s=null,y:f,B:d,M:u,L:a,lenMode:p,step:l=0,starts:i=null,qStride:m=A*N,qOff:h=0,kvStride:o=A*N,kOff:v=0,vOff:w=0,flags:g={}}){for(const[_,D]of[["qStride",m],["qOff",h],["kvStride",o],["kOff",v],["vOff",w],["HEAD_DIM",N]])if(D%4!==0)throw new Error(`attention: ${_}=${D} not vec4-aligned`);if(g.block){if(p!==1)throw new Error("attention block: lenMode must be 1");if(!s)throw new Error("attention block: lens buffer required");if(g.packed&&!i)throw new Error("attention block: packed needs a starts buffer");const _=N/4,{QB:D,JB:x,shared:W}=xe({D4:_,t:g.t,qb:g.qb??null,jb:g.jb??null,qbAlign8:!!g.qbAlign8}),O=D*_;if(O>256)throw new Error(`attention block: QB=${D} needs ${O} > 256 threads`);if(W>16384)throw new Error(`attention block: QB=${D} JB=${x} needs ${W}B shared > 16384`);const M=B(e,"attention_block",le,{t:g.t,immediate:!!g.immediate,defines:{H:A,D:N,QB:D,JB:x,ATTN_SCALE:P,Q_STRIDE:m,Q_OFF:h,KV_STRIDE:o,K_OFF:v,V_OFF:w,PACKED:!!g.packed,NOPACKED:!g.packed}}),R=T(e,"attn params",[d,u,a,p,l,0,0,0],!!g.immediate),G=[t,c,n,s,f];return g.packed&&G.push(i),S(r,M,e,y(R,G),Math.ceil(u/D),A,d,R.values),{pipeline:M,scratch:R.scratch}}const I=B(e,"attention",ce,{t:g.t,wg:g.wg??128,sg:!!g.sg,immediate:!!g.immediate,defines:{H:A,D:N,SCORES_CAP:U,ATTN_SCALE:P,Q_STRIDE:m,Q_OFF:h,KV_STRIDE:o,K_OFF:v,V_OFF:w}}),k=T(e,"attn params",[d,u,a,p,l,0,0,0],!!g.immediate),b=[...k.scratch];let E=s;return E||(E=q(e)),S(r,I,e,y(k,[t,c,n,E,f]),d*u,A,1,k.values),{pipeline:I,scratch:b}}function Ce(e,r,{x:t,r:c,gamma:n,beta:s,y:f,rows:d,flags:u={}}){const a=B(e,"add_ln",fe,{t:u.t,wg:u.wg??256,sg:!!u.sg,immediate:!!u.immediate,defines:{D:ee,EPS:Q}}),p=T(e,"add_ln params",[d,0,0,0],!!u.immediate);return S(r,a,e,y(p,[t,c,n,s,f]),d,1,1,p.values),{pipeline:a,scratch:p.scratch}}function He(e,r,{x:t,w:c,b:n,r:s,gamma:f,beta:d,y:u,M:a,K:p,N:l,flags:i={}}){if(p%4!==0||l%4!==0)throw new Error(`gemm_row_ln: K=${p}/N=${l} must be vec4-aligned`);if((p+l)*4+(i.wg??128)*4>16384)throw new Error(`gemm_row_ln: shared memory over budget at K=${p}, N=${l}`);const m=B(e,"gemm_row_ln",pe,{t:i.t,wg:i.wg??128,sg:!!i.sg,immediate:!!i.immediate,defines:{KDIM:p,D:l,EPS:Q}}),h=T(e,"gemm_row_ln params",[a,0,0,0],!!i.immediate);return S(r,m,e,y(h,[t,c,n,s,f,d,u]),a,1,1,h.values),{pipeline:m,scratch:h.scratch}}function je(e,r,{ids:t,table:c,posEmbed:n,y:s,mode:f,nRows:d,step:u=0,batch:a,s:p=0,packed:l=!1,flags:i={}}){if(l&&(f!=="src"||p>65535))throw new Error(`embed: packed needs mode 'src' and s < 65536 (got ${f}, s=${p})`);const m=B(e,"embed",ve,{t:i.t,wg:i.wg??224,immediate:!!i.immediate,defines:{D:ee,EMBED_SCALE:$,SRC_IDS:f==="src",DECODE:f==="decode",DECODER_START:Y,PACKED:!!l,NOPACKED:!l}}),h=T(e,"embed params",[d,u,a,p],!!i.immediate);return S(r,m,e,y(h,[t,c,n,s]),d,1,1,h.values),{pipeline:m,scratch:h.scratch}}function We(e=256){const r=A*N/4,t=Math.max(Z/4,r+e);return(2*r+t)*16+U*4+e*4}function ze(e,r,{weights:t,layer:c,embed:n=!1,ring:s,kCache:f,vCache:d,crossKV:u,lens:a,x:p,B:l,t:i,S:m,kvCapacity:h=te,flags:o={}}){if(t.dtype!=="f16")throw new Error("decoder mega: needs f16 weights");const v=We(o.wg??256);if(v>16384)throw new Error(`decoder mega: shared memory ${v} bytes > 16384 limit at these dims`);const w=b=>{const E=t.tensors.get(b);if(!E)throw new Error(`decoder mega: missing tensor ${b}`);if(E.byteOffset%8!==0)throw new Error(`decoder mega: ${b} offset not vec4-aligned`);return E.byteOffset/8},g=b=>`dec.${c}.${b}`,I=B(e,"decoder_mega",we,{t:"f16",wg:o.wg??256,sg:!!o.sg,immediate:!!o.immediate,defines:{EMBED:!!n,NOEMBED:!n,...n?{TABLE4:w("shared.weight"),POS4:w("pos_embed"),EMBED_SCALE:$,DECODER_START:Y}:{},H:A,D:N,FFN4:Z/4,LMAX:h,SCORES_CAP:U,ATTN_SCALE:P,EPS:Q,QKVW4:w(g("self_qkv.weight")),QKVB4:w(g("self_qkv.bias")),OUTW4:w(g("self_out.weight")),OUTB4:w(g("self_out.bias")),LN1G4:w(g("ln1.weight")),LN1B4:w(g("ln1.bias")),CQW4:w(g("cross_q.weight")),CQB4:w(g("cross_q.bias")),COW4:w(g("cross_out.weight")),COB4:w(g("cross_out.bias")),LN2G4:w(g("ln2.weight")),LN2B4:w(g("ln2.bias")),FC1W4:w(g("fc1.weight")),FC1B4:w(g("fc1.bias")),FC2W4:w(g("fc2.weight")),FC2B4:w(g("fc2.bias")),LN3G4:w(g("ln3.weight")),LN3B4:w(g("ln3.bias"))}}),k=T(e,"mega params",[l,i,m,0],!!o.immediate);return S(r,I,e,y(k,[t.buffer,s,f,d,u,a,p]),l,1,1,k.values),{pipeline:I,scratch:k.scratch}}function Ue(e,r,{x:t,y:c,starts:n,lens:s,B:f,S:d,N:u,flags:a={}}){if(u%4!==0)throw new Error(`scatter_rows: N=${u} not vec4-aligned`);const p=B(e,"scatter_rows",ge,{t:a.t,wg:a.wg??128,immediate:!!a.immediate}),l=T(e,"scatter_rows params",[f,d,u/4,0],!!a.immediate);return S(r,p,e,y(l,[n,s,t,c]),f*d,1,1,l.values),{pipeline:p,scratch:l.scratch}}function Qe(e,r){if(!Array.isArray(r))throw new Error("bind-group targeted purge needs an array of buffers");const t=F(e);t.stats.bindGroupTargetedPurgeCalls++;const c=new Set;for(const f of r){if(!f)continue;const d=f.buffer??f,u=C.get(d);u&&c.add(u)}if(c.size===0)return 0;const n=[...c].map(f=>`|${f}@`);let s=0;for(const f of[...t.bindGroups.keys()])n.some(d=>f.includes(d))&&(t.bindGroups.delete(f),s++);return t.stats.bindGroupTargetedPurges+=s,s}function Ve(e,r,{data:t,map:c,params:n,rowStrideU32:s,copyLenU32:f,flags:d={}}){if(!Number.isInteger(s)||s<1||!Number.isInteger(f)||f<1||f>s)throw new Error(`compact_gather: bad row shape stride=${s} copy=${f}`);const u=B(e,"compact_gather",he,{wg:d.wg??256});return S(r,u,e,[n,c,t],1,1,1,null),{pipeline:u,scratch:[]}}function Xe(e,r,{logits:t,bias:c,bitmask:n,done:s,tokens:f,B:d,t:u,short:a=null,flags:p={}}){const l=B(e,"argmax_penalty",be,{wg:p.wg??256,immediate:!!p.immediate,defines:{V:a?.n??se,EOS:re,PAD:ne,PENALTY:J,MASK_WORDS:a?.maskWords??K,SHORT:!!a,NOSHORT:!a,...a?{GMASK_WORDS:K}:{}}}),i=T(e,"argmax params",[d,u,0,0],!!p.immediate),m=[t,c,n,s,f];return a&&m.push(a.idmap,a.gmask),S(r,l,e,y(i,m),d,1,1,i.values),{pipeline:l,scratch:i.scratch}}function Ye(e,r,{partials:t,bitmask:c,done:n,tokens:s,B:f,t:d,NT:u,short:a=null,flags:p={}}){const l=B(e,"argmax_reduce",_e,{wg:p.wg??256,immediate:!!p.immediate,defines:{EOS:re,PAD:ne,MASK_WORDS:a?.maskWords??K,SHORT:!!a,NOSHORT:!a,...a?{GMASK_WORDS:K}:{}}}),i=T(e,"argmax_reduce params",[f,d,u,0],!!p.immediate),m=[t,c,n,s];return a&&m.push(a.idmap,a.gmask),S(r,l,e,y(i,m),f,1,1,i.values),{pipeline:l,scratch:i.scratch}}function $e(e,r){const t=e.createCommandEncoder(),c=t.beginComputePass(),{scratch:n}=Ie(e,c,r);c.end(),e.queue.submit([t.finish()]);for(const s of n)s.destroy()}export{Le as A,Be as B,we as C,ie as D,fe as a,pe as b,ce as c,le as d,ve as e,be as f,B as g,Oe as h,Ie as i,Ye as j,Xe as k,He as l,Ge as m,Ke as n,qe as o,je as p,Pe as q,$e as r,Re as s,Ce as t,Ue as u,Ne as v,We as w,Qe as x,ze as y,Ve as z};
//# sourceMappingURL=pipelines-CGkcYAmi.js.map
