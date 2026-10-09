// node tests/check_quant_dashboard.cjs — sözdizimi, QP formu ve hata/kaçış sözleşmesi.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../nihai/web/web_dashboard.html'), 'utf8');
for (const match of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)) new vm.Script(match[1]);
const fields = {name:'<script>bad()</script>',delta:'1',gamma:'0',vega:'0',lower:'-500',upper:'500'};
const row = {querySelectorAll:()=>Object.entries(fields).map(([field,value])=>({dataset:{field},value}))};
const button = {disabled:false};
const elements = {stress:{},'qm-ins-result':{},'qm-ins-floor':{value:'90'},'qm-ins-dd':{value:'10'},'qm-ins-m':{value:'3'},
    'qm-gh-legs':{rows:[row]},'qm-gh-result':{},'qm-gh-target':{value:'delta'},
    'qm-gh-delta':{value:'200'},'qm-gh-gamma':{value:'8'},'qm-gh-vega':{value:'30'}};
let captured, fail = false;
const ctx = vm.createContext({qmEl:id=>elements[id]||(elements[id]={}), qmFmt:v=>String(v),
    qmChart(){}, QM_PALETTE:['blue'],
    qmPct:v=>String(v*100)+'%', qmCells:(id,cells)=>{(elements[id]||(elements[id]={})).cells=cells;},
    escapeHtml:s=>String(s).replaceAll('<','&lt;').replaceAll('>','&gt;'),
    apiCall:async(...args)=>{
        captured=args;
        if (args[0].startsWith('quant/insurance')) return {ok:true,equity:100,asof:'2025-01-01',
            cppi:{taban:90,riskli_tutar:30},tipp:{taban:108,riskli_tutar:0,floor_shortfall:8},note:'Kayıtlı NLV'};
        return fail ? {ok:false,reason:'Olanaksız'} : {ok:true, instruments:[{name:fields.name,quantity:-200}],
            after:{delta:0,gamma:8,vega:30},max_residual:0,note:'Sürekli adet'};
    }});
vm.runInContext(html.slice(html.indexOf('function qmRenderStress('), html.indexOf('async function loadQuantCatalog()')), ctx);
vm.runInContext(html.slice(html.indexOf('function qmRenderPortfolio('), html.indexOf('function qmRenderMicro(')), ctx);
(async()=>{
    const event = {preventDefault(){},target:{querySelector:()=>button}};
    await ctx.qmGreekHedge(event);
    assert.equal(captured[0],'quant/greek_hedge');
    assert.equal(captured[1],'POST');
    assert.equal(captured[2].instruments[0].lower,-500);
    assert.equal(captured[2].current.delta,200);
    assert.equal(captured[2].neutralize[0],'delta');
    assert(!elements['qm-gh-result'].innerHTML.includes('<script>'));
    assert(elements['qm-gh-result'].innerHTML.includes('-200'));
    assert.equal(button.disabled,false);
    const portfolio={symbols:['AAA','BBB'],n_obs:100,interval:'1d',covariance_model:'rmt',
        frontier:{vol:[],ret:[],sharpe:[]},correlation:[[1,0],[0,1]],methods:{cvar_rp:{weights:[.5,.5],risk_contrib:[.5,.5],
            tail_risk:{alpha:.05,es:.02,shares:[.5,.5],max_error:0}}}};
    ctx.qmRenderPortfolio(portfolio);
    assert(elements['qm-pf-tail'].innerHTML.includes('50%'));
    assert(elements['qm-pf-tail'].innerHTML.includes('AAA'));
    assert(!elements['qm-pf-tail'].innerHTML.includes('kink'));
    portfolio.methods.cvar_rp.tail_risk.kink=true;  // Tur-78 P3: VaR bağında katkı tek değerli değildir
    ctx.qmRenderPortfolio(portfolio);
    assert(elements['qm-pf-tail'].innerHTML.includes('kink'));
    delete portfolio.methods.cvar_rp.tail_risk;
    ctx.qmRenderPortfolio(portfolio);
    assert(elements['qm-pf-tail'].innerHTML.includes('ölçülemedi'));
    fail=true;
    await ctx.qmGreekHedge(event);
    assert.equal(elements['qm-gh-result'].textContent,'Olanaksız');
    elements['qm-gh-legs'].rows=[];
    await ctx.qmGreekHedge(event);
    assert.equal(elements['qm-gh-result'].textContent,'En az bir hedge adayı ekleyin.');
    ctx.qmRenderStress('stress',{note:'<script>bad()</script>',stress_requirement:600,margin_shortfall:null,
        scenarios:[{spot_shock:-.12,iv_shock:.2,pnl:-600}]});
    assert(!elements.stress.innerHTML.includes('<script>'));
    assert(elements.stress.innerHTML.includes('Teminat açığı ölçülmedi.'));
    assert(elements.stress.innerHTML.includes('-12%'));
    ctx.qmRenderStress('stress',null);
    assert.equal(elements.stress.textContent,'Stres ölçümü yok.');
    await ctx.qmInsurance(event);
    assert(captured[0].includes('initial_floor=90&max_drawdown=0.1&multiplier=3'));
    assert.equal(elements['qm-ins-result'].cells[3].v,'8');
    assert.equal(button.disabled,false);
    console.log('Quant JS sözdizimi ve Greek formu kontrolleri geçti.');
})().catch(err=>{console.error(err);process.exitCode=1;});
