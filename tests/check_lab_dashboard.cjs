// node tests/check_lab_dashboard.cjs — DOM kimlikleri, rapor grafikleri ve HTML kaçışı.
// Tarayıcı yerleşim testi değildir; ek bağımlılık gerektirmeyen JS sözleşme kontrolü.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '../nihai/web');
const fragment = fs.readFileSync(path.join(root, 'lab_dashboard.html'), 'utf8');
const html = fs.readFileSync(path.join(root, 'web_dashboard.html'), 'utf8') + fragment;
const script = fs.readFileSync(path.join(root, 'lab_dashboard.js'), 'utf8');
const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]));
const fragmentIds = [...fragment.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]);
assert.equal(new Set(fragmentIds).size, fragmentIds.length, 'Yeni kartlarda yinelenen kimlik');
const elements = new Map();
function element(id) {
    assert(ids.has(id), `HTML kimliği eksik: ${id}`);
    if (!elements.has(id)) elements.set(id, {
        id, value: '', _html: '', textContent: '', appendChild() {}, addEventListener() {},
        classList: {contains: () => false},
        elements: Object.fromEntries(['strategy', 'min_votes', 'stop_atr', 'target_atr'].map(k => [k, {}])),
        set innerHTML(value) {
            this._html = value;
            for (const match of value.matchAll(/\bid="([^"]+)"/g)) ids.add(match[1]);
        },
        get innerHTML() { return this._html; },
    });
    return elements.get(id);
}
const charts = [], errors = [];
const ctx = vm.createContext({
    document: {
        getElementById: element,
        querySelector: selector => { assert(html.includes(selector.slice(1, -1))); return {addEventListener() {}}; },
        querySelectorAll: selector => selector === '[data-lab-tab]' ?
            [...fragment.matchAll(/data-lab-tab="([^"]+)"/g)].map(m => ({dataset: {labTab: m[1]}})) : [],
    },
    Chart: function(canvas, options) { charts.push({id: canvas.id, ...options});this.data=options.data;this.update=()=>{};this.destroy = () => {}; },
    apiCall: async endpoint => ({
        'lab/catalog': {tools: {}, datasets: [], strategies: ['macd']},
        jobs: {jobs: []},
        'strategy/status': {strategies: ['macd'], strategy: 'macd', min_votes: 2, stop_atr: 1.5, target_atr: 3},
        'system/deadman': {verdict:'SEANS_KAPALI',market_open:false,files:{}},
    })[endpoint],
    showToast: message => errors.push(message), setTimeout, clearTimeout, setInterval: () => 1,
});
vm.runInContext(script, ctx);
for (const name of [...fragment.matchAll(/\b(lab[A-Z]\w*)\(/g)].map(m => m[1])) {
    assert.equal(typeof ctx[name], 'function', `HTML olay işleyicisi eksik: ${name}`);
}
assert(!ctx.labTable([{value: '<script>bad()</script>'}]).includes('<script>'));
ctx.labRenderReport({result: {results: [{symbol: 'TEST', equity_curve: [100, 105],
    monte_carlo: {bands: {p05: [100, 103], p50: [100, 105], p95: [100, 109]}}}]}});
assert.equal(charts.length, 2);
assert.equal(charts[1].data.datasets.length, 3);
assert.equal(charts[0].data.datasets[0].data[1], 105);
ctx.labRenderReport({result:{results:[{symbol:'TEST',execution_costs:{records:2,
    delay_cost:null,market_impact:12,fees:3,opportunity_cost:0,total_shortfall:null,
    measured:{delay_cost:0,market_impact:1,fees:2,opportunity_cost:2,total_shortfall:0}}}]}});
assert(element('lab-job-result').innerHTML.includes('Karar–dolum farkı'));
assert(element('lab-job-result').innerHTML.includes('1 / 2'));
assert(element('lab-job-result').innerHTML.includes('PnL maliyetleri zaten içerir'));
ctx.labRenderReport({result: {ablations: [{configuration: 'base', weighted_E: 4}]}});
assert.equal(charts.at(-1).data.datasets[0].data[0], 4);
ctx.labRenderReport({result:{report_type:'wiki_trends',article:'Test',observed_days:2,requested_days:2,
    cache_state:'fresh',rows:[{date:'2025-01-01',views:10,baseline_mean:null},{date:'2025-01-02',views:20,baseline_mean:10}]}});
assert.equal(charts.at(-1).data.datasets[0].data[1],20);
assert.equal(charts.at(-1).data.datasets[1].data[0],null);
ctx.labRenderReport({result:{report_type:'fda_calendar',sponsor:'<script>bad()</script>',cache_state:'fresh',events:[]}});
assert(!element('lab-job-result').innerHTML.includes('<script>'));
ctx.labRenderReport({result:{report_type:'pdufa_calendar',n_upcoming:1,n_filings_read:3,n_filings_total:4,cache_state:'fresh',
    truncated:true,upcoming:[{pdufa_date:'2026-11-30',ticker:'VRTX',company:'<script>bad()</script>',n_mentions:2}]}});
assert(element('lab-job-result').innerHTML.includes('2026-11-30') && element('lab-job-result').innerHTML.includes('kısmi'));
assert(!element('lab-job-result').innerHTML.includes('<script>'));
const deadman={market_open:true,project_stopped:false,verdict:'SAGLIKLI',grace_min:0,files:{predictions:{age_min:4.8,limit_min:5}}};
assert.equal(ctx.labDeadmanView(deadman,0).alarm,false);
assert.equal(ctx.labDeadmanView(deadman,20).alarm,true);
assert.equal(ctx.labDeadmanView({...deadman,market_open:false,verdict:'SEANS_KAPALI'},20).alarm,false);
assert.equal(ctx.labDeadmanView({...deadman,project_stopped:true,verdict:'PROJE_KAPALI'},20).alarm,false);
assert.equal(ctx.labDeadmanView({...deadman,files:{predictions:{age_min:null,limit_min:5}}},20).alarm,false);
assert.equal(ctx.labDeadmanView(deadman,46).alarm,true);
assert.equal(ctx.labDeadmanView(null,46).alarm,true);
const position={symbol:'TEST',side:'LONG',entry_time:'2026-09-08',entry_price:100,asset_type:'stock',last_mark:103,stop:99,trailing_enabled:true,state_updated_at:1000};
ctx.labCapturePositions([position]);
element('lab-trail-position').value=JSON.stringify(['TEST','LONG','2026-09-08']);
ctx.labCapturePositions([{...position,last_mark:105,stop:102,state_updated_at:1005}]);
ctx.labDrawTrail();
const stopChart=vm.runInContext("labLiveCharts.get('lab-trail-chart')",ctx);
assert.deepEqual(Array.from(stopChart.data.datasets[1].data),[99,102]);
ctx.labCapturePositions([{...position,last_mark:105,stop:102,state_updated_at:1005}]);
assert.equal(stopChart.data.datasets[1].data.length,2,'Bayat durum yeni gözlem sayılmamalı');
ctx.labCapturePositions([]);
assert.equal(stopChart.data.datasets.length,0,'Kapanan pozisyonun grafiği temizlenmeli');
setImmediate(() => {
    assert.deepEqual(errors, []);
    console.log('Laboratuvar JS sözleşmeleri geçti.');
});
