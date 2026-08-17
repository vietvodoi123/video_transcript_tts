function qe(e=globalThis.navigator?.gpu){return!!e?.wgslLanguageFeatures?.has?.("immediate_address_space")}const Oe=new WeakSet;async function Pe({requireF16:e=!1}={}){if(!navigator.gpu)throw new Error("WebGPU unavailable");const n=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!n)throw new Error("no adapter");const r=["shader-f16","timestamp-query","subgroups"].filter(o=>n.features.has(o));if(e&&!r.includes("shader-f16"))throw new Error("shader-f16 unsupported");const s=await n.requestDevice({requiredFeatures:r,requiredLimits:{maxStorageBufferBindingSize:Math.min(536870912,n.limits.maxStorageBufferBindingSize),maxBufferSize:Math.min(1073741824,n.limits.maxBufferSize)}}),a=qe(navigator.gpu);return a&&Oe.add(s),{device:s,adapterInfo:{vendor:n.info?.vendor??"",architecture:n.info?.architecture??"",device:n.info?.device??"",description:n.info?.description??""},hasF16:r.includes("shader-f16"),hasTimestamps:r.includes("timestamp-query"),hasSubgroups:r.includes("subgroups"),hasImmediates:a,subgroupMinSize:n.info?.subgroupMinSize??0,subgroupMaxSize:n.info?.subgroupMaxSize??0,limits:{maxStorageBufferBindingSize:s.limits.maxStorageBufferBindingSize,maxBufferSize:s.limits.maxBufferSize,maxComputeWorkgroupStorageSize:s.limits.maxComputeWorkgroupStorageSize,maxComputeInvocationsPerWorkgroup:s.limits.maxComputeInvocationsPerWorkgroup},adapterLimits:(()=>{const o={};try{for(const i of Object.getOwnPropertyNames(Object.getPrototypeOf(n.limits))){const u=n.limits[i];typeof u=="number"&&Number.isFinite(u)&&(o[i]=u)}}catch{}return o})(),features:(()=>{try{return[...n.features]}catch{return[]}})()}}const Ce=e=>e>=33&&e<=47||e>=58&&e<=64||e>=91&&e<=96||e>=123&&e<=126,Ue=/\p{P}/u,ze=/\p{Mn}/u,je=/[\p{Cc}\p{Cf}\p{Cn}\p{Co}\p{Cs}]/u,$e=/\s/u;function He(e){return e>=19968&&e<=40959||e>=13312&&e<=19903||e>=131072&&e<=173791||e>=173824&&e<=177983||e>=177984&&e<=178207||e>=178208&&e<=183983||e>=63744&&e<=64255||e>=194560&&e<=195103}function Qe(e,n){return Ce(n)||Ue.test(e)}function Xe(e,{unk:n="[UNK]",cls:t="[CLS]",sep:r="[SEP]",maxChars:s=100}={}){const a=new Map,o=e.split(`
`);for(let c=0;c<o.length;c++){const d=o[c].replace(/\r$/,"");(d.length||c<o.length-1)&&a.set(d,c)}const i=a.get(n),u=a.get(t),l=a.get(r);if(i===void 0||u===void 0||l===void 0)throw new Error("vocab is missing [UNK]/[CLS]/[SEP]");function f(c,d){if(d===0||d===65533)return null;if(c==="	"||c===`
`||c==="\r")return" ";if(je.test(c))return null;if($e.test(c))return" ";let p=c.normalize("NFD"),h="";for(const g of p)ze.test(g)||(h+=g);return h.toLowerCase()}function m(c){const d=Array.from(c),p=[];let h=null;const g=()=>{h&&h.length&&p.push(h),h=null};for(let v=0;v<d.length;v++){const T=d[v],B=T.codePointAt(0),y=f(T,B);if(y!==null){if(y===" "){g();continue}if(He(B)){g(),p.push([{ch:y,orig:v}]);continue}for(const _ of y){const S=_.codePointAt(0);Qe(_,S)?(g(),p.push([{ch:_,orig:v}])):(h||(h=[]),h.push({ch:_,orig:v}))}}}g();const b=[u],w=[[0,0]];for(const v of p){const T=v[0].orig,B=v[v.length-1].orig+1;if(v.length>s){b.push(i),w.push([T,B]);continue}const y=[];let _=0,S=!1;for(;_<v.length;){let E=v.length,N=-1;for(;E>_;){let x="";for(let M=_;M<E;M++)x+=v[M].ch;_>0&&(x=`##${x}`);const I=a.get(x);if(I!==void 0){N=I;break}E--}if(N<0){S=!0;break}y.push({id:N,start:_,end:E}),_=E}if(S)b.push(i),w.push([T,B]);else for(const E of y)b.push(E.id),w.push([v[E.start].orig,v[E.end-1].orig+1])}return b.push(l),w.push([0,0]),{ids:b,offsets:w}}return{tokenize:m,vocab:a,unkId:i,clsId:u,sepId:l}}const Ye=`// GEMM: Y[m,n] = Σ_k X[m,k] · W[k,n] (+ B[n]), optional SiLU.
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

{{IF_GELU}}
// GELU (erf form, matching torch.nn.functional.gelu default): erf via
// Abramowitz–Stegun 7.1.26 (|err| <= 1.5e-7). The JS reference uses the same
// polynomial, so CPU/GPU agree to storage-type rounding.
fn erfApprox(x: f32) -> f32 {
  let s = sign(x);
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
  return s * y;
}
fn gelu1(x: f32) -> f32 {
  return 0.5 * x * (1.0 + erfApprox(x * 0.7071067811865476));
}
{{/IF_GELU}}

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
  {{IF_GELU}}acc = gelu1(acc);{{/IF_GELU}}
  {{IF_LRELU}}acc = max(acc, acc * 0.01);{{/IF_LRELU}}
  {{IF_RELU}}acc = max(acc, 0.0);{{/IF_RELU}}
  Y[m * dims.N + n] = {{OUT_T}}(acc);
}
`,Ve=`// GEMV-style GEMM for SMALL M (decode-step projections, M = batch ≤ ~16).
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

{{IF_GELU}}
// GELU (erf form) via Abramowitz–Stegun 7.1.26 — see gemm.wgsl.
fn erfApprox(x: f32) -> f32 {
  let s = sign(x);
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
  return s * y;
}
fn gelu1(x: f32) -> f32 {
  return 0.5 * x * (1.0 + erfApprox(x * 0.7071067811865476));
}
{{/IF_GELU}}

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
      {{IF_GELU}}v = gelu1(v);{{/IF_GELU}}
      {{IF_LRELU}}v = max(v, v * 0.01);{{/IF_LRELU}}
      {{IF_RELU}}v = max(v, 0.0);{{/IF_RELU}}
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
    {{IF_GELU}}v = vec4<f32>(gelu1(v.x), gelu1(v.y), gelu1(v.z), gelu1(v.w));{{/IF_GELU}}
    {{IF_LRELU}}v = max(v, v * 0.01);{{/IF_LRELU}}
    {{IF_RELU}}v = max(v, vec4<f32>(0.0));{{/IF_RELU}}
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
`,Je=`// Tiled GEMM: Y[m,n] = Σ_k X[m,k] · W[k,n] (+ B[n]), optional SiLU — the
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

{{IF_GELU}}
// GELU (erf form) via Abramowitz–Stegun 7.1.26 — see gemm.wgsl.
fn erfApprox(x: f32) -> f32 {
  let s = sign(x);
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
  return s * y;
}
fn gelu1(x: f32) -> f32 {
  return 0.5 * x * (1.0 + erfApprox(x * 0.7071067811865476));
}
fn gelu4(v: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(gelu1(v.x), gelu1(v.y), gelu1(v.z), gelu1(v.w));
}
{{/IF_GELU}}

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
  {{IF_GELU}}
  acc0 = gelu4(acc0);
  acc1 = gelu4(acc1);
  acc2 = gelu4(acc2);
  acc3 = gelu4(acc3);
  {{/IF_GELU}}
  {{IF_LRELU}}
  acc0 = max(acc0, acc0 * 0.01);
  acc1 = max(acc1, acc1 * 0.01);
  acc2 = max(acc2, acc2 * 0.01);
  acc3 = max(acc3, acc3 * 0.01);
  {{/IF_LRELU}}
  {{IF_RELU}}
  acc0 = max(acc0, vec4<f32>(0.0));
  acc1 = max(acc1, vec4<f32>(0.0));
  acc2 = max(acc2, vec4<f32>(0.0));
  acc3 = max(acc3, vec4<f32>(0.0));
  {{/IF_RELU}}

  storeRow(rowBase + tRow, colBase + tCol, acc0);
  storeRow(rowBase + tRow + 1u, colBase + tCol, acc1);
  storeRow(rowBase + tRow + 2u, colBase + tCol, acc2);
  storeRow(rowBase + tRow + 3u, colBase + tCol, acc3);
}
`,Ze=`// Tiled GEMM v2 — gemm_tiled.wgsl with the memory system fixed:
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

{{IF_GELU}}
// GELU (erf form) via Abramowitz–Stegun 7.1.26 — see gemm.wgsl.
fn erfApprox(x: f32) -> f32 {
  let s = sign(x);
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
  return s * y;
}
fn gelu1(x: f32) -> f32 {
  return 0.5 * x * (1.0 + erfApprox(x * 0.7071067811865476));
}
fn gelu4(v: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(gelu1(v.x), gelu1(v.y), gelu1(v.z), gelu1(v.w));
}
{{/IF_GELU}}

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
  {{IF_GELU}}
  acc0 = gelu4(acc0);
  acc1 = gelu4(acc1);
  acc2 = gelu4(acc2);
  acc3 = gelu4(acc3);
  {{IF_TM8}}
  acc4 = gelu4(acc4);
  acc5 = gelu4(acc5);
  acc6 = gelu4(acc6);
  acc7 = gelu4(acc7);
  {{/IF_TM8}}
  {{/IF_GELU}}
  {{IF_LRELU}}
  acc0 = max(acc0, acc0 * 0.01);
  acc1 = max(acc1, acc1 * 0.01);
  acc2 = max(acc2, acc2 * 0.01);
  acc3 = max(acc3, acc3 * 0.01);
  {{IF_TM8}}
  acc4 = max(acc4, acc4 * 0.01);
  acc5 = max(acc5, acc5 * 0.01);
  acc6 = max(acc6, acc6 * 0.01);
  acc7 = max(acc7, acc7 * 0.01);
  {{/IF_TM8}}
  {{/IF_LRELU}}
  {{IF_RELU}}
  let z4 = vec4<f32>(0.0);
  acc0 = max(acc0, z4);
  acc1 = max(acc1, z4);
  acc2 = max(acc2, z4);
  acc3 = max(acc3, z4);
  {{IF_TM8}}
  acc4 = max(acc4, z4);
  acc5 = max(acc5, z4);
  acc6 = max(acc6, z4);
  acc7 = max(acc7, z4);
  {{/IF_TM8}}
  {{/IF_RELU}}

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
`,en=`// Unified scaled-dot-product attention — the ONE kernel for encoder self,
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
`,nn=`// Blocked (flash-style) scaled-dot-product attention — ENCODER SELF/FUSED
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
`,tn=`// Add + LayerNorm (post-LN): Y[r,i] = gamma[i]·(v-μ)/√(σ²+EPS) + beta[i],
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
`,rn=`// BERT/ELECTRA embedding: Y[r,:] = LayerNorm(word[id] + pos[p]) at width
// EMB (embedding_size, 128 for LTP/small). token_type[0] is folded into the
// pos table at export time (single-sentence inference), and the projection to
// d_model is a separate GEMM. One workgroup per row: dispatchWorkgroups(nRows).
//
// Modes (exactly one of PACKED / NOPACKED):
//   NOPACKED: ids is [B*S]; id = ids[r], pos = r % s (batch rows consecutive).
//   PACKED:   row-packed (pad rows dropped, T = Σ lens rows); each ids word
//             carries its own position: ids[r] = (pos << 16) | id — both
//             halves fit u16 (vocab 21128, srcCap 512; enforced by
//             parseModelConfig).
//
// LayerNorm math is f32 regardless of storage type (population variance via
// a second pass over shared memory — same recipe as add_layernorm.wgsl).
//
// Template placeholders (buildShader in pipelines.js):
//   ENABLE_F16   the f16 enable directive when T is f16, else empty
//   T            storage type of tables/gamma/beta/Y (f16|f32)
//   WG           workgroup size (128 → 1 element per thread at EMB=128;
//                must be a power of two)
//   EMB          row width (embedding_size)
//   EPS          layernorm epsilon literal
{{ENABLE_IMMEDIATE}}
{{ENABLE_F16}}

struct Params { nRows: u32, s: u32, _pad1: u32, _pad2: u32 }

{{PARAM_BINDING}}var<{{PARAM_ADDRESS}}> params: Params;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> wordTable: array<{{T}}>;
@group(0) @binding(3) var<storage, read> posTable: array<{{T}}>;
@group(0) @binding(4) var<storage, read> gamma: array<{{T}}>;
@group(0) @binding(5) var<storage, read> beta: array<{{T}}>;
@group(0) @binding(6) var<storage, read_write> Y: array<{{T}}>;

const EMB: u32 = {{EMB}}u;
const WG: u32 = {{WG}}u;

var<workgroup> vbuf: array<f32, EMB>;
var<workgroup> scratch: array<f32, WG>;

@compute @workgroup_size({{WG}})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  // Uniform per workgroup (one workgroup per row) — safe early return.
  if (wid.x >= params.nRows) { return; }
  let r = wid.x;
{{IF_PACKED}}
  let id = ids[r] & 0xffffu;
  let pos = ids[r] >> 16u;
{{/IF_PACKED}}
{{IF_NOPACKED}}
  let id = ids[r];
  let pos = r % params.s;
{{/IF_NOPACKED}}
  let toff = id * EMB;
  let poff = pos * EMB;
  let base = r * EMB;
  let tid = lid.x;

  var sum: f32 = 0.0;
  for (var i = tid; i < EMB; i = i + WG) {
    let v = f32(wordTable[toff + i]) + f32(posTable[poff + i]);
    vbuf[i] = v;
    sum = sum + v;
  }
  scratch[tid] = sum;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { scratch[tid] = scratch[tid] + scratch[tid + s]; }
    workgroupBarrier();
  }
  let mu = scratch[0] / f32(EMB);
  workgroupBarrier();

  var sq: f32 = 0.0;
  for (var i = tid; i < EMB; i = i + WG) {
    let dev = vbuf[i] - mu;
    sq = sq + dev * dev;
  }
  scratch[tid] = sq;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { scratch[tid] = scratch[tid] + scratch[tid + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(scratch[0] / f32(EMB) + {{EPS}});

  for (var i = tid; i < EMB; i = i + WG) {
    Y[base + i] = {{T}}(f32(gamma[i]) * (vbuf[i] - mu) * inv + f32(beta[i]));
  }
}
`;let Fe,O,P,Ne,Me,de,oe;const me=["dModel","heads","headDim","ffn","encLayers","embSize","vocab","maxPos","srcCap","pad","cls","sep","unk"];function an(e){if(!e||typeof e!="object")throw new Error("model config: not an object");const n={};for(const t of me){const r=e[t];if(!Number.isInteger(r)||r<0)throw new Error(`model config: ${t} must be a non-negative integer, got ${r}`);n[t]=r}for(const t of me.slice(0,9))if(n[t]===0)throw new Error(`model config: ${t} must be positive`);if(!(typeof e.lnEps=="number"&&e.lnEps>0&&e.lnEps<.1))throw new Error(`model config: lnEps must be in (0, 0.1), got ${e.lnEps}`);if(n.lnEps=e.lnEps,n.heads*n.headDim!==n.dModel)throw new Error(`model config: heads·headDim = ${n.heads*n.headDim} != dModel ${n.dModel}`);for(const t of["dModel","headDim","ffn","embSize"])if(n[t]%4!==0)throw new Error(`model config: ${t} = ${n[t]} must be a multiple of 4 (vec4 kernels)`);if(n.srcCap>n.maxPos)throw new Error(`model config: srcCap ${n.srcCap} must not exceed maxPos ${n.maxPos}`);if(n.vocab>65535)throw new Error(`model config: vocab ${n.vocab} >= 65536 breaks packed ids`);if(n.srcCap>65535)throw new Error(`model config: srcCap ${n.srcCap} >= 65536 breaks packed ids`);for(const t of["pad","cls","sep","unk"])if(n[t]>=n.vocab)throw new Error(`model config: ${t} = ${n[t]} out of vocab ${n.vocab}`);return n}function Ae(e){const n=an(e);return Fe=n.dModel,O=n.heads,P=n.headDim,n.ffn,n.vocab,Ne=n.embSize,n.encLayers,n.maxPos,n.srcCap,Me=Math.ceil(n.srcCap/32)*32+32,n.pad,n.cls,n.sep,n.unk,de=n.lnEps,oe=1/Math.sqrt(n.headDim),n}Ae({dModel:256,heads:4,headDim:64,ffn:1024,encLayers:12,embSize:128,vocab:21128,maxPos:512,srcCap:512,pad:0,cls:101,sep:102,unk:100,lnEps:1e-12});const sn=1.2;let on=Math.ceil(21128/32);function De(e={}){const n=e.defines??{};return{t:e.t??"f32",outT:e.outT??e.t??"f32",wg:e.wg??64,bias:!!e.bias,silu:!!e.silu,gelu:!!e.gelu,lrelu:!!e.lrelu,relu:!!e.relu,wt:!!e.wt,sg:!!e.sg,immediate:!!e.immediate,defines:Object.fromEntries(Object.keys(n).sort().map(t=>[t,n[t]]))}}function cn(e,n={}){const{t,outT:r,wg:s,bias:a,silu:o,gelu:i,lrelu:u,relu:l,wt:f,sg:m,immediate:c,defines:d}=De(n),p={T:t,OUT_T:r,WG:String(s),ENABLE_F16:t==="f16"||r==="f16"?"enable f16;":"",ENABLE_SG:m?"enable subgroups;":"",ENABLE_IMMEDIATE:c?"requires immediate_address_space;":"",PARAM_BINDING:c?"":"@group(0) @binding(0) ",PARAM_ADDRESS:c?"immediate":"uniform"},h={BIAS:!!a,SILU:!!o,GELU:!!i,LRELU:!!u,RELU:!!l,WT:!!f,SG:!!m,NOSG:!m};for(const[b,w]of Object.entries(d))typeof w=="boolean"?h[b.toUpperCase()]=w:p[b.toUpperCase()]=String(w);let g=e;for(let b=null;b!==g;)b=g,g=g.replace(/\{\{IF_([A-Z0-9_]+)\}\}([\s\S]*?)\{\{\/IF_\1\}\}/g,(w,v,T)=>{if(!(v in h))throw new Error(`buildShader: unknown conditional {{IF_${v}}}`);return h[v]?T:""});return g=g.replace(/\{\{([A-Z0-9_/]+)\}\}/g,(b,w)=>{if(!(w in p))throw new Error(`buildShader: unresolved placeholder {{${w}}}`);return p[w]}),g}const he=new WeakMap,un=256,ge=new WeakMap,we=new WeakMap,ln=new WeakSet;let dn=1;function ve(e){let n=we.get(e);return n||(n=dn++,we.set(e,n)),n}function Y(e){let n=ge.get(e);return n||(n={bindGroups:new Map,bindGroupLimit:un,activeUniformPools:0,uniformPoolOriginalBindGroupLimit:null,dummyStorage:null,uniformFrame:null,stats:{uniformBuffersCreated:0,uniformPoolBuffersCreated:0,uniformPoolBuffersDestroyed:0,uniformPoolFramesBegun:0,uniformPoolFramesFlushed:0,uniformPoolBlocks:0,uniformPoolBytes:0,uniformPoolBindGroupCacheHits:0,uniformPoolWarmBindGroupLookups:0,uniformPoolWarmBindGroupCacheHits:0,uniformPoolWarmBindGroupResets:0,uniformPoolGenerationInvalidations:0,uniformPoolCachePurges:0,dummyBuffersCreated:0,bindGroupsCreated:0,bindGroupCacheHits:0,bindGroupEvictions:0,bindGroupTargetedPurgeCalls:0,bindGroupTargetedPurges:0,immediateSets:0}},ge.set(e,n)),n}function fn(e){const n=Y(e);return{...n.stats,bindGroupCacheSize:n.bindGroups.size,bindGroupCacheLimit:n.bindGroupLimit}}function q(e,n,t,r={}){let s=he.get(e);s||(s=new Map,he.set(e,s));const a=`${n}:${JSON.stringify(De(r))}`;let o=s.get(a);if(o){if(o.source!==t)throw new Error(`pipeline cache key collision: ${a}`);return o.pipeline}const i=e.createShaderModule({label:a,code:cn(t,r)}),u=e.createComputePipeline({label:a,layout:"auto",compute:{module:i,entryPoint:"main"}});return s.set(a,{pipeline:u,source:t}),u}function ie(e){return e.buffer?e:{buffer:e}}function pn(e,n,t){Y(e).stats.uniformBuffersCreated++;const r=Math.max(16,Math.ceil(t.length*4/16)*16),s=e.createBuffer({label:n,size:r,usage:GPUBufferUsage.UNIFORM,mappedAtCreation:!0});return new Uint32Array(s.getMappedRange()).set(t),s.unmap(),s}function ne(e){const n=Y(e);return n.dummyStorage||(n.dummyStorage=e.createBuffer({label:"shared dummy storage",size:4,usage:GPUBufferUsage.STORAGE}),n.stats.dummyBuffersCreated++),n.dummyStorage}function U(e,n,t,r){if(r)return{resource:null,values:Uint32Array.from(t),scratch:[]};const s=Y(e),a=s.uniformFrame;if(a){const i=Math.max(16,Math.ceil(t.length*4/16)*16),u=Math.ceil(a.bank.cursor/a.alignment)*a.alignment,l=u+i;if(l>a.bankBytes)throw new Error(`uniform pool bank ${a.index} overflow: need ${l} bytes, cap ${a.bankBytes}`);return a.bank.cpu.fill(0,u/4,l/4),a.bank.cpu.set(t,u/4),a.bank.cursor=l,a.bank.blocks++,a.cacheEnabled&&(s.bindGroupLimit=Math.max(s.bindGroupLimit,a.bank.blocks*a.cacheBanks),a.cacheLimitStats.highWater=Math.max(a.cacheLimitStats.highWater,s.bindGroupLimit)),{resource:{buffer:a.bank.buffer,offset:u,size:i},values:null,scratch:[]}}const o=pn(e,n,t);return{resource:o,values:null,scratch:[o]}}function z(e,n){return e.resource?[e.resource,...n]:n}function mn(e,n,t){const r=n.map(s=>{const a=ie(s);return`${ve(a.buffer)}@${a.offset??0}:${a.size??"*"}`});return`${t?"i":"u"}|p${ve(e)}|${r.join("|")}`}function j(e,n,t,r,s,a=1,o=1,i=null){const u=i?1:0,l=Y(t),f=!i&&r.length>0&&ln.has(ie(r[0]).buffer);let m=null,c=null;if((i||f)&&l.bindGroupLimit>0){c=mn(n,r,!!i),m=l.bindGroups.get(c)??null;const d=f&&l.uniformFrame?.warm;d&&l.stats.uniformPoolWarmBindGroupLookups++,m?(l.bindGroups.delete(c),l.bindGroups.set(c,m),l.stats.bindGroupCacheHits++,f&&(l.stats.uniformPoolBindGroupCacheHits++,l.uniformFrame?.warm&&l.stats.uniformPoolWarmBindGroupCacheHits++)):d&&(l.uniformFrame.warm=!1,l.stats.uniformPoolWarmBindGroupResets++)}if(m||(m=t.createBindGroup({layout:n.getBindGroupLayout(0),entries:r.map((d,p)=>({binding:p+u,resource:ie(d)}))}),l.stats.bindGroupsCreated++,c&&(l.bindGroups.set(c,m),l.bindGroups.size>l.bindGroupLimit&&(l.bindGroups.delete(l.bindGroups.keys().next().value),l.stats.bindGroupEvictions++))),e.setPipeline(n),i){if(typeof e.setImmediates!="function")throw new Error("WebGPU immediate shader selected but pass.setImmediates is unavailable");e.setImmediates(0,i),l.stats.immediateSets++}e.setBindGroup(0,m),e.dispatchWorkgroups(s,a,o)}function H(e,n,{x:t,w:r,b:s=null,y:a,M:o,K:i,N:u,storeKV:l=null,scales:f=null,fusedArgmax:m=null,splitK:c=null,flags:d={}}){if(d.tiled&&d.gemv)throw new Error("flags.tiled and flags.gemv are exclusive");if(d.tiled){if(l)throw new Error("storeKV requires flags.gemv");return wn(e,n,{x:t,w:r,b:s,y:a,M:o,K:i,N:u,scales:f,fusedArgmax:m,splitK:c,flags:d})}if(m)throw new Error("fusedArgmax requires flags.tiled");if(c)throw new Error("splitK requires flags.tiled");if(d.gemv)return hn(e,n,{x:t,w:r,b:s,y:a,M:o,K:i,N:u,storeKV:l,scales:f,flags:d});if(f)throw new Error("scales (wq8) requires flags.tiled or flags.gemv");if(l)throw new Error("storeKV requires flags.gemv");const p=d.wg??64,h=q(e,"gemm",Ye,{...d,bias:!!s}),g=U(e,"gemm dims",[o,i,u,0],!!d.immediate),b=[...g.scratch];let w=s;return w||(w=ne(e)),j(n,h,e,z(g,[t,r,w,a]),Math.ceil(u/p),o,1,g.values),{pipeline:h,scratch:b}}function hn(e,n,{x:t,w:r,b:s,y:a,M:o,K:i,N:u,storeKV:l=null,scales:f=null,flags:m}){const c=!!m.wt,d=!!m.wq8;if(d&&(!c||!f||l))throw new Error("gemv wq8: needs wt layout and scales, excludes storeKV");if(c&&i%4!==0)throw new Error(`gemv wt requires K%4==0, got K=${i}`);if(!c&&u%4!==0)throw new Error(`gemv requires N%4==0, got N=${u}`);if(l&&u%3!==0)throw new Error("storeKV requires fused QKV (N=3·H·D)");const p=m.tk??16,h=m.tn??(c?8:4),g=c?m.mt??8:1,b=q(e,"gemm_gemv",Ve,{t:m.t,outT:m.outT,wg:p*h,bias:!!s,silu:m.silu,gelu:m.gelu,lrelu:m.lrelu,relu:m.relu,wt:c,immediate:!!m.immediate,sg:!!m.sg&&c,defines:{TK:p,TN:h,NWT:!c,STORE_KV:!!l,WQ8:d,WQF:!d,...c?{MT:g}:{}}}),w=U(e,"gemv dims",l?[o,i,u,0,l.t,l.Lmax,0,0]:[o,i,u,0],!!m.immediate),v=[...w.scratch];let T=s;T||(T=ne(e));const B=Math.ceil(c?u/h:u/(4*h)),y=c?Math.ceil(o/g):o,_=[t,r,T,a];return l&&_.push(l.kCache,l.vCache),d&&_.push(f),j(n,b,e,z(w,_),B,y,1,w.values),{pipeline:b,scratch:v}}function gn(e,n,t=16){const r=Math.ceil(e/n/t)*t;return{KSL:r,nz:Math.ceil(e/r)}}function wn(e,n,{x:t,w:r,b:s,y:a,M:o,K:i,N:u,scales:l=null,fusedArgmax:f=null,splitK:m=null,flags:c}){const d=c.bm??64,p=c.bn??64,h=c.bkk??16;if(d%4!==0||p%4!==0)throw new Error(`gemm_tiled: BM/BN must be multiples of 4 (${d}, ${p})`);const g=!!c.sh16,b=!!c.dbuf;if(f&&(g||b))throw new Error("gemm_tiled2 fused argmax: sh16/dbuf unsupported (pVal aliases f32 Xs)");if(m){if(f)throw new Error("gemm_tiled2 splitK: exclusive with fusedArgmax");if(!(m.sk>=2))throw new Error(`gemm_tiled2 splitK: sk must be >= 2, got ${m.sk}`);if(s||c.silu||c.gelu||c.lrelu)throw new Error("gemm_tiled2 splitK: pass bias/silu/gelu/lrelu to dispatchGemmReduce, not the GEMM")}const w=f?d*(p/4)*4:0,v=g&&c.t==="f16"?2:4,T=(d+p)*h*v*(b?2:1)+w;if(T>16384)throw new Error(`gemm_tiled: shared memory ${T} bytes > 16384 limit`);if(f&&h<p/4)throw new Error(`gemm_tiled2 fused argmax: BK=${h} < BN/4=${p/4} — pVal cannot alias Xs`);const B=!!c.wq8;if(B&&(!l||!c.wt))throw new Error("gemm_tiled2 wq8: needs scales and wt layout");const y=i%4===0&&h%4===0&&(c.wt||u%4===0),_=(c.tiledV??(y?2:1))===2;if(_&&!y)throw new Error(`gemm_tiled2: shape M=${o} K=${i} N=${u} wt=${!!c.wt} BK=${h} not vec4-eligible`);if(B&&!_)throw new Error("gemm_tiled2 wq8: v1 fallback has no int8 path");if(f&&!_)throw new Error("gemm_tiled2 fused argmax: v2 only");if((g||b)&&!_)throw new Error("gemm_tiled2 sh16/dbuf: v2 only");if(m&&(!_||B))throw new Error("gemm_tiled2 splitK: v2 only, no wq8");const S=m?gn(i,m.sk,h):null,E=_&&c.tm8?8:4;if(d%E!==0)throw new Error(`gemm_tiled: BM=${d} not a multiple of TM=${E}`);const N=d/E*(p/4);if(N>256)throw new Error(`gemm_tiled: ${N} threads > 256 workgroup limit`);const x=_?q(e,"gemm_tiled2",Ze,{t:c.t,outT:c.outT,wg:N,bias:!!s,silu:c.silu,gelu:c.gelu,lrelu:c.lrelu,relu:c.relu,immediate:!!c.immediate,wt:c.wt&&!B,defines:{BM:d,BN:p,BK:h,TM8:E===8,WNT:!c.wt&&!B,WQ8:B,WQF:!B,STORE_Y:!f&&!m,ARGMAX:!!f,SPLITK:!!m,NOSPLITK:!m,SH16:g,SH32:!g,DBUF:b,SBUF:!b,...m?{KSL:S.KSL}:{},...f?{PENALTY:sn,MASK_WORDS:f.maskWords??on}:{}}}):q(e,"gemm_tiled",Je,{t:c.t,outT:c.outT,wg:N,bias:!!s,silu:c.silu,gelu:c.gelu,lrelu:c.lrelu,relu:c.relu,wt:c.wt,immediate:!!c.immediate,defines:{BM:d,BN:p,BK:h}}),I=U(e,"gemm_tiled dims",[o,i,u,0],!!c.immediate),M=[...I.scratch];let A=s;A||(A=ne(e));const R=[t,r,A,f?.partials??m?.parts??a];return B&&R.push(l),f&&R.push(f.lbias,f.seen),j(n,x,e,z(I,R),Math.ceil(u/p),Math.ceil(o/d),S?.nz??1,I.values),{pipeline:x,scratch:M}}function vn({D4:e,t:n,qb:t=null,jb:r=null,qbAlign8:s=!1}){let a=t??Math.max(1,Math.min(16,Math.floor(256/e)));if(s&&t==null&&a*e%16!==0){for(;a>1&&a*e%16!==0;)a--;if(a*e%16!==0)throw new Error(`attention block: no QB <= 256 threads aligns D4=${e} to 16 threads`)}const o=n==="f16"?8:16,i=l=>a*e*16+2*l*e*o+a*l*4+3*a*4;let u=r??32;if(r==null)for(;u>8&&i(u)>16384;)u>>=1;return{QB:a,JB:u,shared:i(u)}}function bn(e,n,{q:t,k:r,v:s,lens:a=null,y:o,B:i,M:u,L:l,lenMode:f,step:m=0,starts:c=null,qStride:d=O*P,qOff:p=0,kvStride:h=O*P,kOff:g=0,vOff:b=0,flags:w={}}){for(const[_,S]of[["qStride",d],["qOff",p],["kvStride",h],["kOff",g],["vOff",b],["HEAD_DIM",P]])if(S%4!==0)throw new Error(`attention: ${_}=${S} not vec4-aligned`);if(w.block){if(!a)throw new Error("attention block: lens buffer required");if(w.packed&&!c)throw new Error("attention block: packed needs a starts buffer");const _=P/4,{QB:S,JB:E,shared:N}=vn({D4:_,t:w.t,qb:w.qb??null,jb:w.jb??null,qbAlign8:!!w.qbAlign8}),x=S*_;if(x>256)throw new Error(`attention block: QB=${S} needs ${x} > 256 threads`);if(N>16384)throw new Error(`attention block: QB=${S} JB=${E} needs ${N}B shared > 16384`);const I=q(e,"attention_block",nn,{t:w.t,immediate:!!w.immediate,defines:{H:O,D:P,QB:S,JB:E,ATTN_SCALE:oe,Q_STRIDE:d,Q_OFF:p,KV_STRIDE:h,K_OFF:g,V_OFF:b,PACKED:!!w.packed,NOPACKED:!w.packed}}),M=U(e,"attn params",[i,u,l,f,m,0,0,0],!!w.immediate),A=[t,r,s,a,o];return w.packed&&A.push(c),j(n,I,e,z(M,A),Math.ceil(u/S),O,i,M.values),{pipeline:I,scratch:M.scratch}}const v=q(e,"attention",en,{t:w.t,wg:w.wg??128,sg:!!w.sg,immediate:!!w.immediate,defines:{H:O,D:P,SCORES_CAP:Me,ATTN_SCALE:oe,Q_STRIDE:d,Q_OFF:p,KV_STRIDE:h,K_OFF:g,V_OFF:b}}),T=U(e,"attn params",[i,u,l,f,m,0,0,0],!!w.immediate),B=[...T.scratch];let y=a;return y||(y=ne(e)),j(n,v,e,z(T,[t,r,s,y,o]),i*u,O,1,T.values),{pipeline:v,scratch:B}}function be(e,n,{x:t,r,gamma:s,beta:a,y:o,rows:i,eps:u=null,flags:l={}}){const f=q(e,"add_ln",tn,{t:l.t,wg:l.wg??256,sg:!!l.sg,immediate:!!l.immediate,defines:{D:Fe,EPS:u??de}}),m=U(e,"add_ln params",[i,0,0,0],!!l.immediate);return j(n,f,e,z(m,[t,r,s,a,o]),i,1,1,m.values),{pipeline:f,scratch:m.scratch}}function _n(e,n,{ids:t,wordTable:r,posTable:s,gamma:a,beta:o,y:i,nRows:u,s:l=0,packed:f=!1,flags:m={}}){if(f&&l>65535)throw new Error(`embed_bert: packed needs s < 65536 (got s=${l})`);const c=q(e,"embed_bert",rn,{t:m.t,wg:m.wg??128,immediate:!!m.immediate,defines:{EMB:Ne,EPS:de,PACKED:!!f,NOPACKED:!f}}),d=U(e,"embed_bert params",[u,l,0,0],!!m.immediate);return j(n,c,e,z(d,[t,r,s,a,o,i]),u,1,1,d.values),{pipeline:c,scratch:d.scratch}}const Bn=`// D8 Stage-G compact token head.
//
// The classifier GEMM writes seven f32 logits per real token. This kernel
// selects the first-index argmax and computes its f32 softmax probability.
// Output is one vec2<u32> per token: (label, bitcast<f32>(probability)).

struct Params {
  rows: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> compact: array<vec2<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let row = invocation.x;
  if (row >= params.rows) {
    return;
  }

  let base = row * 7u;
  var selected: u32 = 0u;
  var maximum: f32 = logits[base];
  for (var label: u32 = 1u; label < 7u; label = label + 1u) {
    let value = logits[base + label];
    if (value > maximum) {
      maximum = value;
      selected = label;
    }
  }

  var denominator: f32 = 0.0;
  for (var label: u32 = 0u; label < 7u; label = label + 1u) {
    denominator = denominator + exp(logits[base + label] - maximum);
  }
  let probability = 1.0 / denominator;
  compact[row] = vec2<u32>(selected, bitcast<u32>(probability));
}
`,F=256,yn=4,Tn=64,Q=3*F,X=1024,G=7,_e=12,ce=512,K=2;function We(e){return Math.ceil(e/4)*4}function En(e,n){return[e*4,n*4,n*4,e*F*K,e*F*K,e*F*K,e*Q*K,e*X*K,e*G*4,e*8,e*8,e*G*4,16].reduce((t,r)=>t+We(Math.max(4,r)),0)}function W(e,n,t,r){return e.createBuffer({label:r,size:We(Math.max(4,n)),usage:t})}function L(e,n){if(!e)throw new Error(n)}function kn(e,n,t,r){const s=GPUBufferUsage.STORAGE,a=s|GPUBufferUsage.COPY_DST,o=s|GPUBufferUsage.COPY_SRC,i={ids:W(e,t*4,a,`${n}: packed ids`),lens:W(e,r*4,a,`${n}: lens`),starts:W(e,r*4,a,`${n}: starts`),a:W(e,t*F*K,s,`${n}: hidden a`),b:W(e,t*F*K,s,`${n}: hidden b`),y:W(e,t*F*K,s,`${n}: sublayer y`),qkv:W(e,t*Q*K,s,`${n}: qkv`),ffn:W(e,t*X*K,s,`${n}: ffn`),logits:W(e,t*G*4,o,`${n}: logits`),compact:W(e,t*8,o,`${n}: compact`),compactStaging:W(e,t*8,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,`${n}: compact staging`),logitsStaging:W(e,t*G*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,`${n}: logits staging`),headParams:W(e,16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,`${n}: head params`)};return{key:n,rowCapacity:t,batchCapacity:r,estimatedBytes:En(t,r),buffers:i,headBindGroup:null,uses:0,destroy(){for(const u of Object.values(i))u.destroy()}}}function Sn(e){L(Array.isArray(e)&&e.length>0,"run requires at least one row");const n=new Uint32Array(e.length),t=new Uint32Array(e.length);let r=0,s=0;for(let i=0;i<e.length;i+=1){const u=e[i].input_ids;L(Array.isArray(u)||ArrayBuffer.isView(u),`row ${i} input_ids missing`),L(u.length>=2&&u.length<=ce,`row ${i} token length ${u.length} outside 2..${ce}`),t[i]=r,n[i]=u.length,r+=u.length,s=Math.max(s,u.length)}const a=new Uint32Array(r);let o=0;for(const i of e)for(let u=0;u<i.input_ids.length;u+=1){const l=Number(i.input_ids[u]);L(Number.isInteger(l)&&l>=0&&l<65536,`invalid token id ${l}`),a[o]=u<<16|l,o+=1}return{words:a,lens:n,starts:t,total:r,sequence:s}}function Be(e,n){const t=new Uint8Array(e.getMappedRange(0,n)),r=new ArrayBuffer(n);return new Uint8Array(r).set(t),e.unmap(),r}class xn{constructor(n,t){this.ctx=n,this.device=n.device,this.weights=t,this.workspaces=new Map,this.submissions=0,this.workspaceAllocations=0,this.uncapturedErrors=[],this.deviceLoss=null,this.destroyed=!1;const r=t.model,s={dModel:F,heads:yn,headDim:Tn,ffn:X,encLayers:_e,embSize:F,maxPos:ce};for(const[o,i]of Object.entries(s))L(r[o]===i,`active model ${o} ${r[o]} != ${i}`);L(t.dtype==="f16",`D8 requires f16 weights, got ${t.dtype}`);const a=this.device.createShaderModule({label:"D8 Stage-G softmax argmax7",code:Bn});this.headPipeline=this.device.createComputePipeline({label:"D8 Stage-G softmax argmax7",layout:"auto",compute:{module:a,entryPoint:"main"}}),this.errorListener=o=>{this.uncapturedErrors.push(`${o.error?.message??o.error??o}`)},this.device.addEventListener?.("uncapturederror",this.errorListener),this.device.lost.then(o=>{this.deviceLoss={reason:o.reason??"unknown",message:o.message??""}})}workspace(n,t,r){L(!this.destroyed,"engine is destroyed");let s=this.workspaces.get(n);return s&&(s.rowCapacity<t||s.batchCapacity<r)&&(s.destroy(),this.workspaces.delete(n),s=null),s||(s=kn(this.device,n,t,r),s.headBindGroup=this.device.createBindGroup({label:`${n}: head bind group`,layout:this.headPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:s.buffers.headParams}},{binding:1,resource:{buffer:s.buffers.logits}},{binding:2,resource:{buffer:s.buffers.compact}}]}),this.workspaces.set(n,s),this.workspaceAllocations+=1),s}async runTokenized(n,{workspaceKey:t="default",rowCapacity:r=null,batchCapacity:s=null,debugLogits:a=!1}={}){if(L(!this.destroyed,"engine is destroyed"),this.deviceLoss)throw new Error(`WebGPU device lost: ${JSON.stringify(this.deviceLoss)}`);const o=Sn(n),i=r??o.total,u=s??n.length;L(i>=o.total,`row capacity ${i} < ${o.total}`),L(u>=n.length,`batch capacity ${u} < ${n.length}`);const l=this.workspace(t,i,u),f=n.length,m=o.sequence,c=o.total,{device:d}=this,p=l.buffers;d.queue.writeBuffer(p.ids,0,o.words),d.queue.writeBuffer(p.lens,0,o.lens),d.queue.writeBuffer(p.starts,0,o.starts),d.queue.writeBuffer(p.headParams,0,new Uint32Array([c,0,0,0]));const h=d.createCommandEncoder({label:`D8 Stage-G ${t}`}),g=h.beginComputePass({label:`D8 Stage-G ${t}`}),b=[],w=({scratch:k})=>b.push(...k),v=k=>this.weights.bindingFor(k),T=this.ctx.hasImmediates===!0,B={t:"f16",immediate:T},y={...B,tiled:!0,tm8:!0},_={...B,block:!0,packed:!0};w(_n(d,g,{ids:p.ids,wordTable:v("emb.word"),posTable:v("emb.pos"),gamma:v("emb.ln.weight"),beta:v("emb.ln.bias"),y:p.a,nRows:c,s:m,packed:!0,flags:B}));for(let k=0;k<_e;k+=1){const D=Ge=>`enc.${k}.${Ge}`;w(H(d,g,{x:p.a,w:v(D("qkv.weight")),b:v(D("qkv.bias")),y:p.qkv,M:c,K:F,N:Q,flags:y})),w(bn(d,g,{q:p.qkv,k:p.qkv,v:p.qkv,lens:p.lens,starts:p.starts,y:p.b,B:f,M:m,L:m,lenMode:1,qStride:Q,qOff:0,kvStride:Q,kOff:F,vOff:2*F,flags:_})),w(H(d,g,{x:p.b,w:v(D("out.weight")),b:v(D("out.bias")),y:p.y,M:c,K:F,N:F,flags:y})),w(be(d,g,{x:p.y,r:p.a,gamma:v(D("ln1.weight")),beta:v(D("ln1.bias")),y:p.b,rows:c,flags:B})),w(H(d,g,{x:p.b,w:v(D("fc1.weight")),b:v(D("fc1.bias")),y:p.ffn,M:c,K:F,N:X,flags:{...y,gelu:!0}})),w(H(d,g,{x:p.ffn,w:v(D("fc2.weight")),b:v(D("fc2.bias")),y:p.y,M:c,K:X,N:F,flags:y})),w(be(d,g,{x:p.y,r:p.b,gamma:v(D("ln2.weight")),beta:v(D("ln2.bias")),y:p.a,rows:c,flags:B}))}w(H(d,g,{x:p.a,w:v("head.ner.weight"),b:v("head.ner.bias"),y:p.logits,M:c,K:F,N:G,flags:{t:"f16",outT:"f32",wt:!0,immediate:T}})),g.setPipeline(this.headPipeline),g.setBindGroup(0,l.headBindGroup),g.dispatchWorkgroups(Math.ceil(c/64)),g.end(),h.copyBufferToBuffer(p.compact,0,p.compactStaging,0,c*8),a&&h.copyBufferToBuffer(p.logits,0,p.logitsStaging,0,c*G*4);const S=performance.now();d.queue.submit([h.finish()]),this.submissions+=1;for(const k of b)k.destroy();const E=[p.compactStaging.mapAsync(GPUMapMode.READ,0,c*8)];a&&E.push(p.logitsStaging.mapAsync(GPUMapMode.READ,0,c*G*4)),await Promise.all(E);const N=performance.now(),x=Be(p.compactStaging,c*8),I=new Uint32Array(x),M=new Float32Array(x),A=new Uint32Array(c),R=new Float32Array(c);let V=!0;for(let k=0;k<c;k+=1)A[k]=I[k*2],R[k]=M[k*2+1],(A[k]>=G||!Number.isFinite(R[k])||R[k]<0||R[k]>1)&&(V=!1);let re=null;if(a){const k=Be(p.logitsStaging,c*G*4);re=new Float32Array(k);for(const D of re)if(!Number.isFinite(D)){V=!1;break}}if(l.uses+=1,!V)throw new Error("D8 Stage-G produced invalid labels, probabilities, or logits");if(this.deviceLoss)throw new Error(`WebGPU device lost: ${JSON.stringify(this.deviceLoss)}`);if(this.uncapturedErrors.length>0)throw new Error(`WebGPU uncaptured errors: ${this.uncapturedErrors.join(" | ")}`);return{rows:c,batch:f,sequence:m,starts:o.starts,lens:o.lens,labels:A,probabilities:R,logits:re,finite:V,gpuWallMs:N-S,submissions:1,workspace:{key:l.key,rowCapacity:l.rowCapacity,batchCapacity:l.batchCapacity,uses:l.uses}}}diagnostics(){return{submissions:this.submissions,workspaceAllocations:this.workspaceAllocations,workspaceCount:this.workspaces.size,workspaceEstimatedBytes:[...this.workspaces.values()].reduce((n,t)=>n+t.estimatedBytes,0),workspaces:[...this.workspaces.values()].map(n=>({key:n.key,rowCapacity:n.rowCapacity,batchCapacity:n.batchCapacity,estimatedBytes:n.estimatedBytes,uses:n.uses})),dispatch:fn(this.device),uncapturedErrors:[...this.uncapturedErrors],deviceLoss:this.deviceLoss,hasImmediates:this.ctx.hasImmediates,hasF16:this.ctx.hasF16,hasSubgroups:this.ctx.hasSubgroups}}releaseWorkspaces(){let n=0;const t=this.workspaces.size;for(const r of this.workspaces.values())n+=r.estimatedBytes,r.destroy();return this.workspaces.clear(),{count:t,estimatedBytes:n}}destroy({destroyWeights:n=!0}={}){this.destroyed||(this.destroyed=!0,this.device.removeEventListener?.("uncapturederror",this.errorListener),this.releaseWorkspaces(),n&&this.weights.buffer.destroy())}}const Le=.703,ue=Object.freeze(["O","B-Nh","I-Nh","B-Ns","I-Ns","B-Ni","I-Ni"]),In=Object.freeze({Nh:"Người",Ns:"Địa danh",Ni:"Tổ chức"});function Ke(e){const n=Array.from(e),t=new Uint32Array(n.length+1);let r=0;for(let s=0;s<n.length;s+=1)t[s]=r,r+=n[s].length;return t[n.length]=r,{points:n,utf16:t}}function J(e,n){if(!e)throw new Error(n)}function Fn(e,n,t,r,{threshold:s=Le}={}){J(e.length===n.length,"BIO labels/probabilities length mismatch"),J(e.length===t.length,"BIO labels/offsets length mismatch");const{points:a,utf16:o}=Ke(r),i=[];let u=0;for(;u<e.length;){const l=t[u];J(Array.isArray(l)&&l.length===2,`invalid token offset at ${u}`);const[f,m]=l,c=ue[e[u]]??"O";if(f===m||c==="O"){u+=1;continue}const d=c.split("-",2)[1],p=[u];let h=m,g=u+1;for(;g<e.length;){const[w,v]=t[g],T=ue[e[g]]??"O";if(w===v||T!==`I-${d}`||w>h)break;p.push(g),h=v,g+=1}const b=Math.min(...p.map(w=>n[w]));b>=s&&(J(Number.isInteger(f)&&Number.isInteger(h)&&f>=0&&h>=f&&h<=a.length,`entity offset ${f}:${h} outside text length ${a.length}`),i.push({type:d,typeLabel:In[d]??d,startCodepoint:f,endCodepoint:h,startUtf16:o[f],endUtf16:o[h],text:a.slice(f,h).join(""),confidence:b,tokenCount:p.length})),u=g>u+1?g:u+1}return i}function Nn(e){return`${e.type}\0${e.startCodepoint}\0${e.endCodepoint}`}const Mn=/[\s,，。！？；：、…．.!?;:）)】》”"'』」]/u,An=e=>e>=33&&e<=47||e>=58&&e<=64||e>=91&&e<=96||e>=123&&e<=126,Dn=/\p{P}/u,fe=/\s/u,ye=new Set(["。","！","？","…","!","?"]),Wn=new Set(["”","』","」","）"]);function C(e,n){if(!e)throw new Error(n)}function le(e,n,t,r){const s=n.slice(t,r).join(""),a=e.tokenize(s);return{text:s,...a}}function Ln(e,n,t){for(let r=n;r<t;r+=1)if(!fe.test(e[r]))return!0;return!1}function Kn(e,n){let t=n;for(;t<e.length&&fe.test(e[t]);)t+=1;return t}function Rn(e){const n=[];let t=0,r=0;for(;r<e.length;){let s=null;if(ye.has(e[r])){for(r+=1;r<e.length&&ye.has(e[r]);)r+=1;for(;r<e.length&&Wn.has(e[r]);)r+=1;s="sentence"}else e[r]==="\r"||e[r]===`
`?(e[r]==="\r"&&e[r+1]===`
`?r+=2:r+=1,s="paragraph"):r+=1;s!==null&&(r=Kn(e,r),Ln(e,t,r)&&(n.push({start:t,end:r,boundary:r===e.length?"end":s}),t=r))}return t<e.length&&n.push({start:t,end:e.length,boundary:"end"}),n}function Gn(e){return e>=19968&&e<=40959||e>=13312&&e<=19903||e>=131072&&e<=173791||e>=173824&&e<=177983||e>=177984&&e<=178207||e>=178208&&e<=183983||e>=63744&&e<=64255||e>=194560&&e<=195103}function Te(e){const n=e.codePointAt(0);return fe.test(e)||An(n)||Dn.test(e)||Gn(n)}function qn(e,n,t){let r=Math.min(e.length,Math.max(n+1,t));for(;r<e.length&&!Te(e[r-1])&&!Te(e[r]);)r+=1;return r}function On(e,n,t,r){let s=r*2;for(;;){const a=qn(n,t,t+s),o=le(e,n,t,a);if(o.ids.length>r||a===n.length)return{end:a,encoded:o};const i=a-t;s=Math.max(s*2,i*2)}}function Pn(e,n,t,r,s){let a=r,o=le(e,n,t,a);for(;o.ids.length>s;){const i=o.offsets[s-2]?.[1];C(Number.isInteger(i)&&i>0,"unable to find a safe token cut"),a=Math.min(a-1,t+i),C(a>t,"segmentation made no progress"),o=le(e,n,t,a)}return{end:a,encoded:o}}function Cn(e,n,{maxTokens:t,overlapCodepoints:r,breakWindowCodepoints:s}){const a=Array.from(n),o=[];let i=0,u=0;for(;i<a.length;){const l=On(e,a,i,t),f=l.encoded;if(l.end===a.length&&f.ids.length<=t){o.push({text:f.text,input_ids:f.ids,offsets:f.offsets,startCodepoint:i,endCodepoint:a.length,ownedStartCodepoint:u,ownedEndCodepoint:a.length,artificialLeftEdge:i<u,artificialRightEdge:!1,boundary:"end"});break}const m=f.offsets[t-2]?.[1];C(Number.isInteger(m)&&m>0,"tokenizer did not expose a cut offset");let c=Math.min(a.length,i+m),d=!1;const p=Math.max(i+1,c-s);for(let b=c;b>p;b-=1)if(Mn.test(a[b-1])){c=b,d=!0;break}const h=Pn(e,a,i,c,t),g=h.end;g!==c&&(d=!1),C(g>u,"overlap is too large for the active token window"),o.push({text:h.encoded.text,input_ids:h.encoded.ids,offsets:h.encoded.offsets,startCodepoint:i,endCodepoint:g,ownedStartCodepoint:u,ownedEndCodepoint:g,artificialLeftEdge:i<u,artificialRightEdge:!d,boundary:d?"safe":"hard"}),u=g,i=d?g:Math.max(i+1,g-r)}for(const l of o)C(l.input_ids.length<=t,"segmentation emitted an oversized row");return o}function Un(e,n,{maxTokens:t=512,overlapCodepoints:r=64,breakWindowCodepoints:s=96}={}){C(Number.isInteger(t)&&t>=4,"maxTokens must be at least 4"),C(Number.isInteger(r)&&r>=0,"overlapCodepoints must be non-negative");const a=Array.from(n);if(a.length===0)return[];const o={maxTokens:t,overlapCodepoints:r,breakWindowCodepoints:s},i=[];for(const u of Rn(a)){const l=a.slice(u.start,u.end).join(""),f=Cn(e,l,o);for(let m=0;m<f.length;m+=1){const c=f[m],d=m+1===f.length;i.push({...c,startCodepoint:u.start+c.startCodepoint,endCodepoint:u.start+c.endCodepoint,ownedStartCodepoint:u.start+c.ownedStartCodepoint,ownedEndCodepoint:u.start+c.ownedEndCodepoint,boundary:d?u.boundary:c.boundary})}}return i}function zn(e,n){const t=Array.from(n.text).length;return!(n.artificialLeftEdge&&e.startCodepoint===0||n.artificialRightEdge&&e.endCodepoint===t||n.startCodepoint+e.endCodepoint<=n.ownedStartCodepoint)}function $(e,n){if(!e)throw new Error(n)}function jn(){try{return new DOMException("The D8 analysis was aborted","AbortError")}catch{const e=new Error("The D8 analysis was aborted");return e.name="AbortError",e}}function ae(e){if(e?.aborted)throw jn()}function Ee(e){let n=1;for(;n<e;)n*=2;return n}function $n(e,n,t){const r=[...e].sort((i,u)=>i.input_ids.length-u.input_ids.length||i.flatIndex-u.flatIndex),s=[];let a=[],o=0;for(const i of r)a.length>0&&(a.length>=n||o+i.input_ids.length>t)&&(s.push(a),a=[],o=0),a.push(i),o+=i.input_ids.length;return a.length&&s.push(a),s}class Hn{constructor({ctx:n,classifier:t,tokenizer:r,manifest:s,provenance:a,assetBase:o,ownsContext:i}){this.ctx=n,this.classifier=t,this.tokenizer=r,this.manifest=s,this.provenance=a,this.assetBase=o,this.ownsContext=i,this.destroyed=!1,this.analysisCount=0}async analyze(n,{batchSize:t=32,maxPackedTokens:r=32768,signal:s=null,releaseWorkspaces:a=!1,onProgress:o=null}={}){$(!this.destroyed,"D8 engine is destroyed"),$(Number.isInteger(t)&&t>=1,"batchSize must be a positive integer"),$(Number.isInteger(r)&&r>=512,"maxPackedTokens must be an integer >= 512");const i=typeof n=="string",u=i?[n]:n;$(Array.isArray(u),"analyze expects a string or an array of strings"),$(u.every(h=>typeof h=="string"),"analyze input contains a non-string"),ae(s);const l=performance.now(),f=u.map((h,g)=>({index:g,text:h,entities:[],segmentCount:0,tokenCount:0,gpuWallMs:0,elapsedMs:0})),m=u.map(h=>Ke(h)),c=[];for(let h=0;h<u.length;h+=1){const g=u[h],b=Un(this.tokenizer,g);f[h].segmentCount=b.length;for(let w=0;w<b.length;w+=1){const v=b[w];f[h].tokenCount+=v.input_ids.length,c.push({...v,textIndex:h,segmentIndex:w,flatIndex:c.length})}}if(c.length===0){const h=performance.now()-l;for(const g of f)g.elapsedMs=h;return i?f[0]:f}const d=$n(c,t,r);let p=0;try{for(let h=0;h<d.length;h+=1){ae(s);const g=d[h],b=g.reduce((B,y)=>B+y.input_ids.length,0),w=Ee(b),v=Ee(g.length),T=await this.classifier.runTokenized(g,{workspaceKey:`analysis:${w}:${v}`,rowCapacity:w,batchCapacity:v});ae(s);for(let B=0;B<g.length;B+=1){const y=g[B],_=T.starts[B],S=_+T.lens[B],E=Fn(T.labels.subarray(_,S),T.probabilities.subarray(_,S),y.offsets,y.text).filter(I=>zn(I,y)),N=f[y.textIndex],x=m[y.textIndex];for(const I of E){const M=y.startCodepoint+I.startCodepoint,A=y.startCodepoint+I.endCodepoint;N.entities.push({...I,startCodepoint:M,endCodepoint:A,startUtf16:x.utf16[M],endUtf16:x.utf16[A],text:x.points.slice(M,A).join("")})}N.gpuWallMs+=T.gpuWallMs*(y.input_ids.length/T.rows)}p+=g.length,o?.({stage:"analysis",completedRows:p,totalRows:c.length,batch:h+1,batches:d.length})}for(const h of f){const g=new Map;for(const b of h.entities){const w=Nn(b),v=g.get(w);(!v||b.confidence>v.confidence)&&g.set(w,b)}h.entities=[...g.values()].sort((b,w)=>b.startCodepoint-w.startCodepoint||b.endCodepoint-w.endCodepoint||b.type.localeCompare(w.type)),h.elapsedMs=performance.now()-l}return this.analysisCount+=1,i?f[0]:f}finally{a&&this.releaseWorkspaces()}}releaseWorkspaces(){return $(!this.destroyed,"D8 engine is destroyed"),this.classifier.releaseWorkspaces()}diagnostics(){return{model:{name:"lert_small_d8",threshold:Le,labels:[...ue],imageBytes:this.provenance.imageBytes,imageSha256:this.provenance.imageSha256,tensorCount:this.provenance.tensorCount},adapter:{...this.ctx.adapterInfo},capabilities:{hasF16:this.ctx.hasF16,hasImmediates:this.ctx.hasImmediates,hasSubgroups:this.ctx.hasSubgroups},assetBase:this.assetBase,ownsContext:this.ownsContext,analysisCount:this.analysisCount,weights:{...this.provenance},classifier:this.classifier.diagnostics(),destroyed:this.destroyed}}destroy({destroyDevice:n=this.ownsContext}={}){this.destroyed||(this.classifier.destroy(),n&&this.ctx.device.destroy?.(),this.destroyed=!0)}}function ee(e,n){return Math.ceil(e/n)*n}const Qn=new Float32Array(1);new Uint32Array(Qn.buffer);function Xn(e){const n=e.length,t=new Float32Array(n);for(let r=0;r<n;r++){const s=e[r],a=s>>10&31,o=s&1023,i=s&32768?-1:1;a===0?t[r]=i*o*2**-24:a===31?t[r]=o?NaN:i*(1/0):t[r]=i*(1024+o)*2**(a-25)}return t}const Yn={f16:2,f32:4};function Re(e){if(!e||typeof e!="object")throw new Error("manifest: not an object");if(e.version!==1)throw new Error(`manifest: expected version 1, got ${e.version}`);if(!e.model||typeof e.model!="object")throw new Error("manifest: missing model");if(!Array.isArray(e.tensors)||e.tensors.length===0)throw new Error("manifest: tensors must be a non-empty array");const n=new Map;let t=0,r=null;for(const s of e.tensors){const{name:a,dtype:o,shape:i,byteOffset:u,byteLength:l}=s??{};if(typeof a!="string"||!a)throw new Error("manifest: tensor without a name");if(o!=="f16"&&o!=="f32")throw new Error(`manifest: tensor "${a}" has unsupported dtype "${o}"`);if(!Array.isArray(i)||i.length===0||!i.every(c=>Number.isInteger(c)&&c>0))throw new Error(`manifest: tensor "${a}" has invalid shape ${JSON.stringify(i)}`);if(!Number.isInteger(u)||u<0||!Number.isInteger(l)||l<=0)throw new Error(`manifest: tensor "${a}" has invalid byteOffset/byteLength`);const f=i.reduce((c,d)=>c*d,1),m=f*Yn[o];if(l!==m)throw new Error(`manifest: tensor "${a}" byteLength ${l} != shape·dtype ${m}`);if(u%256!==0)throw new Error(`manifest: tensor "${a}" byteOffset ${u} is not 256-aligned`);if(n.has(a))throw new Error(`manifest: duplicate tensor name "${a}"`);if(u<t)throw new Error(`manifest: tensor "${a}" (offset ${u}) overlaps or is not ascending after "${r}" (ends at ${t})`);n.set(a,{dtype:o,shape:i,elems:f,byteOffset:u,byteLength:l}),t=u+l,r=a}if(e.bins!==void 0){if(!Array.isArray(e.bins)||e.bins.length===0)throw new Error("manifest: bins must be a non-empty array");for(const s of e.bins){if(typeof s?.file!="string"||!s.file)throw new Error("manifest: bins[].file must be a non-empty string");if(!Number.isInteger(s.byteLength)||s.byteLength<=0)throw new Error(`manifest: bins "${s.file}" byteLength must be a positive integer`)}}return{model:e.model,labels:e.labels??null,tensors:n}}function se(e){let n=0;for(const t of e.values())n=Math.max(n,t.byteOffset+t.byteLength);return n}function Vn(e,n){let t=0;const r=new Map;for(const[a,o]of e){const i=ee(t,256),u=o.elems*4;r.set(a,{dtype:"f32",shape:o.shape,elems:o.elems,byteOffset:i,byteLength:u}),t=i+u}const s=new Uint8Array(ee(t,4));for(const[a,o]of e){const i=r.get(a);if(o.dtype==="f16"){const u=new Uint16Array(n.buffer,n.byteOffset+o.byteOffset,o.elems);new Float32Array(s.buffer,i.byteOffset,o.elems).set(Xn(u))}else s.set(n.subarray(o.byteOffset,o.byteOffset+o.byteLength),i.byteOffset)}return{tensors:r,bytes:s}}function Jn(e,n,t,{targetDtype:r="f16"}={}){if(r!=="f16"&&r!=="f32")throw new Error(`unsupported targetDtype "${r}"`);const{model:s,labels:a,tensors:o}=Re(n);Ae(s);const i=se(o);if(t.byteLength<i)throw new Error(`weights.bin too short: ${t.byteLength} < ${i}`);let u=o,l=t;r==="f32"&&({tensors:u,bytes:l}=Vn(o,t));const f=ee(se(u),4);if(l.byteLength<f){const c=new Uint8Array(f);c.set(l.subarray(0,se(u))),l=c}const m=e.createBuffer({size:f,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});return e.queue.writeBuffer(m,0,l,0,f),{model:s,labels:a,dtype:r,buffer:m,tensors:u,bindingFor(c){const d=u.get(c);if(!d)throw new Error(`unknown tensor "${c}"`);return{buffer:m,offset:d.byteOffset,size:ee(d.byteLength,4)}}}}const pe="ltp-d8-weights-v1",ke="b48b81d283ff0d20b45badb259dcbf5b7491371510f45e6a609c2c8b4f3d2ead",Se="9cd23b6c8488a2dde401b8d80b011e88f54a2f5bd4c3838a17b0149498780468",xe=150,Zn=33554432,et=2e7;function nt(e){return[...new Uint8Array(e)].map(n=>n.toString(16).padStart(2,"0")).join("")}async function te(e){const n=e instanceof ArrayBuffer?e:e.buffer.slice(e.byteOffset,e.byteOffset+e.byteLength);return nt(await crypto.subtle.digest("SHA-256",n))}async function tt(e){const n=await fetch(e,{cache:"no-store"});if(!n.ok)throw new Error(`${e}: HTTP ${n.status}`);const t=new Uint8Array(await n.arrayBuffer()),r=await te(t);if(r!==ke)throw new Error(`manifest SHA-256 ${r} != ${ke}`);return JSON.parse(new TextDecoder().decode(t))}function rt(e){const n=Re(e);if(e.stage!=="D8-G")throw new Error("weight manifest stage drift");if(n.tensors.size!==xe)throw new Error(`weight tensor count ${n.tensors.size} != ${xe}`);if(!e.storage||e.storage.dtype!=="f16")throw new Error("weight manifest storage drift");if(e.storage.imageBytes>Zn)throw new Error(`weight image ${e.storage.imageBytes} exceeds frozen cap`);if(!/^[0-9a-f]{64}$/.test(e.storage.imageSha256))throw new Error("weight image lacks a full SHA-256");if(e.storage.imageSha256!==Se)throw new Error(`weight image drift: ${e.storage.imageSha256} != ${Se}`);if(e.storage.tokenTypeRow0FoldedIntoPosition!==!0||e.storage.qkvLayout!=="q|k|v"||e.storage.classifierWeightLayout!=="NK")throw new Error("weight layout contract drift");if(!Array.isArray(e.bins)||e.bins.length===0)throw new Error("weight shards are missing");for(const o of e.bins){if(o.byteLength>et)throw new Error(`${o.file} exceeds frozen shard cap`);if(!/^[0-9a-f]{64}$/.test(o.sha256))throw new Error(`${o.file} lacks a full SHA-256`)}for(const o of["emb.proj.weight","emb.proj.bias"])if(n.tensors.has(o))throw new Error(`forbidden tensor ${o}`);if([...n.tensors.keys()].some(o=>o.includes("pooler")))throw new Error("forbidden pooler tensor");const t=n.model,r={dModel:256,heads:4,headDim:64,ffn:1024,encLayers:12,embSize:256,vocab:21128,maxPos:512,srcCap:512,lnEps:1e-12};for(const[o,i]of Object.entries(r))if(t[o]!==i)throw new Error(`model ${o} drift: ${t[o]} != ${i}`);const s=e.labels?.ner,a=["O","B-Nh","I-Nh","B-Ns","I-Ns","B-Ni","I-Ni"];if(JSON.stringify(s)!==JSON.stringify(a))throw new Error("NER labels drift");return n}async function at(e){if(!e||typeof caches>"u")return null;try{return await caches.open(e)}catch{return null}}async function st(e,n,t){const r=async(a,o)=>{if(!a?.ok)return null;const i=new Uint8Array(await a.arrayBuffer());return i.byteLength!==n.byteLength||await te(i)!==n.sha256?null:{bytes:i,source:o}};if(t)try{const a=await r(await t.match(e),"cache");if(a)return a;await t.delete(e)}catch{}const s=await r(await fetch(e),"network");if(!s)throw new Error(`${n.file}: network bytes failed length or SHA-256 validation`);if(t)try{await t.put(e,new Response(s.bytes,{headers:{"Content-Type":"application/octet-stream"}}))}catch{}return s}async function ut(e=pe){if(!e||typeof caches>"u")return!1;try{return await caches.delete(e)}catch{return!1}}async function ot(e,n,{onProgress:t=null,cacheName:r=pe}={}){const s=await tt(`${n}/manifest.json`);rt(s);const a=s.bins.reduce((p,h)=>p+h.byteLength,0);if(a!==s.storage.imageBytes)throw new Error(`shard bytes ${a} != image bytes ${s.storage.imageBytes}`);let o=0;const i=await at(r);let u=0;const l=await Promise.all(s.bins.map(async p=>{const h=await st(`${n}/${p.file}`,p,i);return h.source==="cache"&&(u+=1),o+=h.bytes.byteLength,t?.({stage:"weights",loaded:o,total:a,part:p.file,source:h.source}),h.bytes})),f=new Uint8Array(a);let m=0;for(const p of l)f.set(p,m),m+=p.byteLength;const c=await te(f);if(c!==s.storage.imageSha256)throw new Error(`weight image SHA-256 ${c} != ${s.storage.imageSha256}`);const d=Jn(e,s,f,{targetDtype:"f16"});return{weights:d,manifest:s,provenance:{imageBytes:f.byteLength,imageSha256:c,shardCount:s.bins.length,bins:s.bins,tensorCount:d.tensors.size,cacheName:i?r:null,cacheHits:u}}}const Ie="45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c";function Z(e,n){if(!e)throw new Error(n)}async function it(e,n){const t=`${e}/vocab.txt`,r=await fetch(t);if(!r.ok)throw new Error(`${t}: HTTP ${r.status}`);const s=new Uint8Array(await r.arrayBuffer()),a=await te(s);if(a!==Ie)throw new Error(`vocab SHA-256 ${a} != ${Ie}`);return n?.({stage:"vocab",loaded:s.byteLength,total:s.byteLength,source:"network-or-http-cache"}),new TextDecoder().decode(s)}function ct(e){return Z(typeof e=="string"&&e.length>0,"assetBase must be a non-empty URL"),e.replace(/\/+$/,"")}async function lt({assetBase:e="/model/d8",ctx:n=null,onProgress:t=null,cacheName:r=pe}={}){const s=ct(e),a=n===null,o=n??await Pe({requireF16:!0});Z(o?.device,"a WebGPU context with device is required"),Z(o.hasF16===!0||o.device.features?.has?.("shader-f16"),"D8 requires shader-f16");let i=null;try{const[u,l]=await Promise.all([it(s,t),ot(o.device,`${s}/weights`,{cacheName:r,onProgress:t})]);i=l;const f=Xe(u);Z(f.vocab.size===21128,`vocab size ${f.vocab.size} != 21128`);const m=new xn(o,i.weights);return new Hn({ctx:o,classifier:m,tokenizer:f,manifest:i.manifest,provenance:i.provenance,assetBase:s,ownsContext:a})}catch(u){throw i?.weights?.buffer?.destroy?.(),a&&o.device.destroy?.(),u}}export{Hn as D8Ner,Se as D8_IMAGE_SHA256,ue as D8_LABELS,ke as D8_MANIFEST_SHA256,Le as D8_THRESHOLD,In as D8_TYPES,Ie as D8_VOCAB_SHA256,pe as D8_WEIGHT_CACHE,ut as clearD8WeightCache,Ke as codePointIndex,lt as createD8Ner,Fn as decodeBio,Un as segmentText,rt as validateFrozenManifest};
//# sourceMappingURL=index-DOgWbOaH.js.map
