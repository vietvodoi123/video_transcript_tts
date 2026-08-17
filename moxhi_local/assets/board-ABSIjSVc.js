import"./modulepreload-polyfill-B5Qt9EMX.js";import{p as H,a as O,b as k,P as N,d as G}from"./pseudonym-DeqB8BRK.js";import{M}from"./models-CvHl_Lx7.js";const i=e=>String(e??"").replace(/[&<>"]/g,s=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"})[s]),m=e=>{if(!(e>=1e3))return String(Math.round(e??0));const s=(e/1e3).toFixed(1);return`${(s.endsWith(".0")?s.slice(0,-2):s).replace(".",",")}k`},d=e=>Number(e??0).toLocaleString("vi-VN"),j=(e,s=Date.now())=>{const r=s-e;return r<9e4?"vừa xong":r<36e5?`${Math.round(r/6e4)} phút trước`:r<864e5?`${Math.round(r/36e5)} giờ trước`:`${Math.round(r/864e5)} ngày trước`},F=e=>{const s=/^vulkan[\s\d.]*\((.+?)\s*\)\s*$/i.exec(e);return(s?s[1].trim():e).replace(/^(\S+)( \1)+(?= |$)/i,"$1")},S=e=>F(e.gpuName||e.description||[e.vendor,e.architecture].filter(Boolean).join(" ")||"GPU");function I(e){const s=String(e.uaModel??"").trim();return!s||S(e).toLowerCase().includes(s.toLowerCase())?"":s}const P=new Map(M.map(e=>[e.id,e.name])),R=e=>P.get(e)??String(e??"?"),y=new Map(M.map((e,s)=>[e.id,s])),U=new Map([["not-checked","chưa xét"],["bypass","bỏ qua"],["cache-hit","cache"],["needs-baseline-observation","cần baseline"],["unamortized","chưa hoàn vốn"],["admitted","đạt"]]);function E(e){const s=new Map;for(const r of e){let a=s.get(r.model);a||s.set(r.model,a={model:r.model,byDevice:new Map});for(const t of r.entries){const n=[t.pseudo,t.vendor,t.architecture,t.device].join("|"),c=a.byDevice.get(n);if(!c){a.byDevice.set(n,{...t,dtype:r.dtype});continue}c.runs=(c.runs??0)+(t.runs??0),c.sentences=Math.max(c.sentences??0,t.sentences??0),c.lastAt=Math.max(c.lastAt??0,t.lastAt??0),(t.best??0)>(c.best??0)&&Object.assign(c,{best:t.best,dtype:r.dtype,description:t.description,gpuName:t.gpuName,platform:t.platform,browser:t.browser,uaModel:t.uaModel,deviceMemory:t.deviceMemory,cpuThreads:t.cpuThreads,scheduleMode:t.scheduleMode,scheduleSource:t.scheduleSource,scheduleConfiguredInFlight:t.scheduleConfiguredInFlight,scheduleConfiguredGroupSteps:t.scheduleConfiguredGroupSteps,scheduleEffectiveInFlight:t.scheduleEffectiveInFlight,scheduleEffectiveGroupSteps:t.scheduleEffectiveGroupSteps,scheduleCacheState:t.scheduleCacheState,scheduleTuneMs:t.scheduleTuneMs,scheduleAmortizationStatus:t.scheduleAmortizationStatus})}}return[...s.values()].map(r=>({model:r.model,entries:[...r.byDevice.values()].sort((a,t)=>(t.best??0)-(a.best??0))})).sort((r,a)=>(y.get(r.model)??99)-(y.get(a.model)??99))}const z=e=>/android|ios|iphone|ipad/i.test(String(e??""));function _(e,{filter:s="",mobileOnly:r=!1}={}){const a=s.trim().toLowerCase();return e.filter(t=>r&&!z(t.platform)?!1:a?[t.vendor,t.architecture,t.device,t.description,t.gpuName,t.uaModel,t.platform].filter(Boolean).join(" ").toLowerCase().includes(a):!0)}function K(e,s=Date.now()){const r=s-6048e5,a=new Map;for(const t of e){const n=t.vendor||"không rõ";let c=a.get(n);c||a.set(n,c={vendor:n,total:0,archs:[]}),c.total+=t.count,c.archs.push({architecture:t.architecture||"—",count:t.count,isNew:t.firstAt>r})}return[...a.values()].sort((t,n)=>n.total-t.total)}function V(e){const s=!!e.weekBest;return`<section class="hero-stat" aria-label="Tổng quan">
    <div class="hero-lead">
      <p class="figure"><b class="fig-num"${s?` data-tick="${Number(e.weekBest)}"`:""}>${s?m(e.weekBest):"—"}</b><span class="fig-unit">tok/giây</span></p>
      <p class="qualifier">${s?"Tốc độ nhanh nhất cộng đồng đo được bảy ngày qua — ẩn danh, ngay trong trình duyệt.":"Bảy ngày qua chưa có lượt gửi nào — bảng chờ số đo đầu tiên."}</p>
      <p class="hero-cta"><a class="cta-chip" href="/">Dịch &amp; gửi benchmark</a></p>
    </div>
    <div class="substats">
      <div class="substat"><b>${d(e.submissions)}</b><span>lượt gửi</span></div>
      <div class="substat"><b>${d(e.families)}</b><span>dòng GPU</span></div>
    </div>
  </section>`}function X(e,s=0){return e.map((r,a)=>`<button type="button" class="tab${a===s?" active":""}"
    data-tab="${a}" aria-pressed="${a===s}">${i(R(r.model))}</button>`).join("")}const W=e=>`<span class="rankmark${e<3?` r${e+1}`:""}">${e+1}</span>`,Y=(e,s,r=!1)=>{const{emoji:a,tone:t}=H(e.pseudo);return`<span class="ava t${t}" aria-hidden="true">${a}</span>
        <span class="who-id">
          <span class="pseudo t${t}">${i(O(e.pseudo))}</span>${r?'<span class="minetag">máy của bạn</span>':""}
          <time class="when">${j(e.lastAt,s)}</time>
        </span>`};function Z(e,s,r=Date.now(),a=""){return e.length?`<div class="tablewrap"><table role="table">
    <thead role="rowgroup"><tr role="row">
      <th role="columnheader" scope="col" class="rank">Hạng</th>
      <th role="columnheader" scope="col">Thiết bị</th>
      <th role="columnheader" scope="col" class="perf">Tốc độ</th>
      <th role="columnheader" scope="col">Người gửi</th>
    </tr></thead>
    <tbody role="rowgroup">${e.map((n,c)=>{const p=!!a&&n.pseudo===a,u=[],g=I(n);if(g&&u.push(i(g)),u.push(`${i(n.platform)} · ${i(n.browser)}`),n.dtype&&n.dtype!=="f16"&&u.push(i(n.dtype)),u.push(`${d(n.sentences)} câu`),n.runs>1&&u.push(`${d(n.runs)} lần gửi`),n.scheduleMode){const f=`${d(n.scheduleConfiguredInFlight)}×${d(n.scheduleConfiguredGroupSteps)}`,$=`${d(n.scheduleEffectiveInFlight)}×${d(n.scheduleEffectiveGroupSteps)}`,B=f===$?f:`${f}→${$}`,x=n.scheduleSource==="measured"?"cold":n.scheduleSource==="cache"?"warm":"fallback";if(u.push(`lịch ${i(n.scheduleMode)} ${B}`),u.push(`${x} · cache ${i(n.scheduleCacheState??"không rõ")}`),u.push(`tune ${d(n.scheduleTuneMs)} ms`),n.scheduleAmortizationStatus){const C=U.get(n.scheduleAmortizationStatus)??n.scheduleAmortizationStatus;u.push(`kinh tế ${i(C)}`)}}const v=[n.deviceMemory>0?`RAM ${n.deviceMemory>=32?"≥":"~"}${n.deviceMemory} GB`:"",n.cpuThreads>0?`~${n.cpuThreads} luồng CPU`:""].filter(Boolean).join(" · "),D=s>0?Math.max(2,Math.round(n.best/s*1e3)/10):0;return`<tr role="row"${p?' class="mine"':""}>
      <td role="cell" class="rank">${W(c)}</td>
      <td role="cell" class="main"${v?` title="${i(v)}"`:""}>
        <span class="devname">${i(S(n))}</span>
        <span class="devmeta">${u.join(" · ")}</span>
      </td>
      <td role="cell" class="perf">
        <span class="tok">${m(n.best)}<small> tok/s</small></span>
        <span class="bar" aria-hidden="true"><i class="fill" data-w="${D}"></i></span>
      </td>
      <td role="cell" class="who">
        ${Y(n,r,p)}
      </td>
    </tr>`}).join("")}</tbody>
  </table></div>`:`<div class="state miss">
      <p>Không thiết bị nào khớp bộ lọc.</p>
      <button type="button" class="ghost" data-act="clear">Xóa bộ lọc</button>
    </div>`}const w=e=>`<button type="button" class="covbtn" data-act="filter"
    data-q="${i(e)}" title="Lọc bảng theo ${i(e)}">${i(e)}</button>`;function J(e,s=Date.now()){return e.length?`<section class="coverage">
    <h2>Bản đồ thiết bị</h2>
    <div class="vendors">${K(e,s).map(a=>`<div class="vendor">
    <h3>${a.vendor==="không rõ"?i(a.vendor):w(a.vendor)}</h3>
    <ul>${a.archs.map(t=>`<li>
      <span class="arch">${w(t.architecture)}</span><span class="dots"></span>
      <span class="n">${d(t.count)}</span>${t.isNew?'<span class="newmark">mới tuần này</span>':""}
    </li>`).join("")}</ul>
  </div>`).join("")}</div>
  </section>`:""}function Q(){return`<div class="state empty">
    <p>Chưa có lượt gửi nào.</p>
    <p class="sub">Dịch xong một văn bản rồi bấm <b>Gửi benchmark</b> —
      thiết bị của bạn sẽ xuất hiện ở đây.</p>
    <a class="cta-chip" href="/">Dịch &amp; gửi benchmark</a>
  </div>`}function ee(e){return`<div class="state error">
    <p>Không tải được bảng xếp hạng <span class="detail">(${i(e)})</span>.</p>
    <button type="button" class="ghost" data-act="retry">Thử lại</button>
  </div>`}function te(){return`<section class="hero-stat skel" aria-hidden="true">
    <div class="hero-lead">
      <p class="figure"><b class="fig-num">—</b><span class="fig-unit">tok/giây</span></p>
      <p class="qualifier">Đang tải số liệu…</p>
    </div>
    <div class="substats">
      <div class="substat"><b>—</b><span>lượt gửi</span></div>
      <div class="substat"><b>—</b><span>dòng GPU</span></div>
    </div>
  </section>
  <section class="ranking" aria-label="Đang tải bảng xếp hạng">
    <h2>Bảng xếp hạng</h2>
    <div class="skeleton">${'<div class="skel-row"></div>'.repeat(5)}</div>
  </section>`}function se(e,s){return e.length?`<section class="ranking">
    <h2>Bảng xếp hạng</h2>
    <div class="controls">
      <div class="tabs">${X(e,s)}</div>
      <label class="search">Tìm thiết bị
        <input type="search" id="board-filter" name="board-filter"
          placeholder="Vendor, kiến trúc, GPU…" autocomplete="off">
      </label>
      <label class="switch">
        <input type="checkbox" id="board-mobile" name="board-mobile">
        chỉ thiết bị di động
      </label>
      <p class="count" aria-live="polite"></p>
    </div>
    <div class="bucket"></div>
  </section>`:`<section class="ranking">
      <h2>Bảng xếp hạng</h2>
      ${Q()}
    </section>`}const l=document.querySelector("#board-main"),o={data:null,merged:[],tab:0,filter:"",mobileOnly:!1,mine:null};async function ae(){try{const e=localStorage.getItem(N);return e?await G(e):""}catch{return""}}function ne(){const e=l.querySelectorAll(".fill[data-w]");requestAnimationFrame(()=>{for(const s of e)s.style.transform=`scaleX(${s.dataset.w/100})`})}function re(){if(matchMedia("(prefers-reduced-motion: reduce)").matches)return;const e=l.querySelector(".fig-num[data-tick]"),s=Number(e?.dataset.tick);if(!(s>0))return;const r=performance.now(),a=600,t=n=>{const c=Math.min(1,(n-r)/a);e.textContent=m(s*(1-(1-c)**3)),c<1&&requestAnimationFrame(t)};requestAnimationFrame(t)}function h(e=!1){const s=o.merged[o.tab];if(!s)return;const r=_(s.entries,{filter:o.filter,mobileOnly:o.mobileOnly}),a=s.entries.reduce((c,p)=>Math.max(c,p.best),0),t=l.querySelector(".bucket");t.classList.toggle("instant",e),t.innerHTML=Z(r,a,Date.now(),o.mine??"");const n=l.querySelector(".count");n&&(n.textContent=`${r.length} thiết bị`),ne()}function L(e=!1){const{totals:s,families:r}=o.data;l.innerHTML=V(s)+se(o.merged,o.tab)+J(r),e||re();const a=l.querySelector("#board-filter");a&&(a.value=o.filter,l.querySelector("#board-mobile").checked=o.mobileOnly,h(e))}let b=0;async function q(){try{const[e]=await Promise.all([fetch(`${k()}/api/board`),o.mine===null?ae().then(s=>{o.mine=s}):null]);if(!e.ok)throw new Error(`HTTP ${e.status}`);o.data=await e.json(),o.merged=E(o.data.buckets),o.tab=0,b=Date.now(),L()}catch(e){l.innerHTML=ee(e?.message??e)}}const A=5*6e4;async function T(){if(document.activeElement?.id!=="board-filter")try{const e=await fetch(`${k()}/api/board`);if(!e.ok)return;o.data=await e.json(),o.merged=E(o.data.buckets),o.tab>=o.merged.length&&(o.tab=0),b=Date.now(),L(!0)}catch{}}setInterval(()=>{document.visibilityState==="visible"&&o.data&&T()},A);document.addEventListener("visibilitychange",()=>{document.visibilityState==="visible"&&o.data&&Date.now()-b>A&&T()});l.addEventListener("click",e=>{const s=e.target.closest(".tab");if(s){o.tab=Number(s.dataset.tab),l.querySelectorAll(".tab").forEach(a=>{const t=a===s;a.classList.toggle("active",t),a.setAttribute("aria-pressed",String(t))}),h();return}const r=e.target.closest("[data-act]")?.dataset.act;if(r==="retry")l.innerHTML=te(),q();else if(r==="clear"){o.filter="",o.mobileOnly=!1;const a=l.querySelector("#board-filter");a&&(a.value="");const t=l.querySelector("#board-mobile");t&&(t.checked=!1),h(!0)}else if(r==="filter"){o.filter=e.target.closest('[data-act="filter"]').dataset.q??"";const a=l.querySelector("#board-filter");a&&(a.value=o.filter,a.focus({preventScroll:!0})),h(!0),a?.closest("section")?.scrollIntoView({block:"start",behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"auto":"smooth"})}});l.addEventListener("input",e=>{e.target.id==="board-filter"&&(o.filter=e.target.value,h(!0))});l.addEventListener("change",e=>{e.target.id==="board-mobile"&&(o.mobileOnly=e.target.checked,h(!0))});q();
//# sourceMappingURL=board-ABSIjSVc.js.map
