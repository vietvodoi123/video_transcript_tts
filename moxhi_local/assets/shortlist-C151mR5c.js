const n=new RegExp("[\\u2E80-\\u2EFF\\u3100-\\u312F\\u31C0-\\u31EF\\u3400-\\u9FFF\\uF900-\\uFAFF]|[\\u{20000}-\\u{3FFFF}]","u");function s(t){const F=[];for(let u=0;u<t.length;u++){const e=t[u];typeof e=="string"&&!n.test(e)&&F.push(u)}return Uint32Array.from(F)}export{s as e};
//# sourceMappingURL=shortlist-C151mR5c.js.map
