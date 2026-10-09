'use strict';
let labCatalog = null, labJob = '', labTimer = null, labPreview = null;
let labCharts = [];
const labEl = id => document.getElementById(id);
const labEsc = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labLabels = {running:'Çalışıyor',completed:'Tamamlandı',failed:'Başarısız',cancelled:'İptal edildi',unobserved:'Süreç tutamacı bu panelde yok; bitiş raporu bekleniyor',
    sharpe_annualized:'Sharpe',sortino_annualized:'Sortino',max_drawdown_pct:'Azami düşüş (%)',total_return_pct:'Getiri (%)',final_equity:'Son bakiye',trades_count:'İşlem sayısı',
    short_term_total:'Kısa vade toplamı',long_term_total:'Uzun vade toplamı',total_1256:'§1256 toplamı',n_wash_candidates:'Wash-sale adayı',
    measured:'Ölçüldü',reason:'Açıklama',ok:'Geçerli',n:'Kayıt sayısı',head:'Son hash',merkle_root:'Merkle kökü',broken_at:'İlk bozuk satır',
    offset_ms:'Saat kayması (ms)',roundtrip_ms:'Gidiş dönüş (ms)',warn:'Eşik aşıldı',server:'NTP sunucusu',threshold_ms:'Uyarı eşiği (ms)',
    epoch:'Epoch',max_epochs:'Azami epoch',fold:'Bölme',train_loss_last_batch:'Son minibatch kaybı',val_loss:'Doğrulama kaybı'};
const labValue = v => v === null || v === undefined ? 'Ölçülmedi' : typeof v === 'boolean' ? (v ? 'Evet' : 'Hayır') : typeof v === 'object' ? JSON.stringify(v) : String(v);

function labTable(rows, columns) {
    if (!rows.length) return '<p class="lab-note">Kayıt yok.</p>';
    columns = columns || [...new Set(rows.flatMap(r => Object.keys(r)))];
    return '<table><thead><tr>'+columns.map(k=>'<th>'+labEsc(labLabels[k]||k)+'</th>').join('')+'</tr></thead><tbody>'+rows.slice(0,200).map(row=>
        '<tr>'+columns.map(k=>'<td>'+labEsc(labValue(row[k]))+'</td>').join('')+'</tr>').join('')+'</tbody></table>'+
        (rows.length>200?'<p class="lab-note">'+rows.length+' kaydın ilk 200 satırı gösteriliyor; tamamı JSON raporunda.</p>':'');
}
function labObject(value) {
    if (Array.isArray(value)) return value.every(v=>v && typeof v==='object' && !Array.isArray(v)) ? labTable(value) : '<pre>'+labEsc(value.map(labValue).join('\n'))+'</pre>';
    if (value && typeof value==='object') {
        let scalars=[], nested='';
        for (const [key,v] of Object.entries(value)) {
            if (v && typeof v==='object') nested += '<details><summary>'+labEsc(labLabels[key]||key)+(Array.isArray(v)?' ('+v.length+')':'')+'</summary>'+labObject(v)+'</details>';
            else scalars.push({alan:labLabels[key]||key,değer:labValue(v)});
        }
        return (scalars.length?labTable(scalars,['alan','değer']):'')+nested;
    }
    return '<p>'+labEsc(labValue(value))+'</p>';
}
async function labFetch(endpoint, body) {
    const data=await apiCall(endpoint,body===undefined?'GET':'POST',body);
    if (!data || data.success===false) throw new Error(data?.message||'İşlem başarısız.');
    return data;
}
function labError(error, target) {
    if(target) labEl(target).innerHTML='<p class="lab-error" role="alert">'+labEsc(error.message)+'</p>';
    else showToast(error.message,'error');
}
async function labRead(endpoint,target) {
    labEl(target).textContent='Okunuyor…';
    try { const data=await labFetch(endpoint);labEl(target).innerHTML=labObject(data);return data; }
    catch(error){labError(error,target);return null;}
}
function labOpen() {
    const button=document.querySelector('[aria-controls="tab-lab"]');
    switchTab('tab-lab',button);
}
async function labStart(endpoint, body) {
    try {
        const result=await labFetch(endpoint,body);
        if(!result.job_id) throw new Error('Sunucu iş kimliği döndürmedi.');
        labOpen();await labRefresh();await labSelectJob(result.job_id);
        labEl('lab-job-panel').scrollIntoView({behavior:'smooth',block:'start'});
    }catch(error){labError(error);}
}
function labForm(id, numeric=[]) {
    const form=labEl(id);if(!form.reportValidity()) return null;
    const data=Object.fromEntries(new FormData(form));
    for(const key of numeric) data[key]=Number(data[key]);
    return data;
}
async function labBacktest(event) {
    event.preventDefault();
    const data=labForm('lab-backtest',['equity','mc_runs','impact_coef','stop_atr','target_atr','is_bars','oos_bars']);if(!data)return;
    data.symbols=data.symbols.split(/[\s,]+/).filter(Boolean);data.from_date=data.from_date||null;data.to_date=data.to_date||null;
    const button=event.submitter;button.disabled=true;
    try{await labStart('backtest/run',data);}finally{button.disabled=false;}
}
function labToolFields() {
    const spec=labCatalog?.tools[labEl('lab-tool').value];if(!spec)return;
    labEl('lab-tool-note').textContent=spec.note||'Mevcut araştırma aracı seçilen parametrelerle çalıştırılır. Veri önkoşulları ve ölçümler iş günlüğünde görünür.';
    labEl('lab-tool-fields').innerHTML=Object.entries(spec.fields).map(([key,s])=>{
        const attrs=' name="'+labEsc(key)+'" class="mini-input"';let input;
        if(s.type==='choice') input='<select'+attrs+'>'+s.choices.map(v=>'<option'+(v===s.default?' selected':'')+'>'+labEsc(v)+'</option>').join('')+'</select>';
        else if(s.type==='dataset') input='<select'+attrs+' required>'+labDatasetOptions()+'</select>';
        else if(s.type==='bool') input='<input type="checkbox" name="'+labEsc(key)+'"'+(s.default?' checked':'')+'>';
        else input='<input'+attrs+' type="'+(s.type==='number'?'number':s.type==='date'?'date':'text')+'" value="'+labEsc(s.default)+'" required'+(s.type==='number'?' min="'+s.min+'" max="'+s.max+'" step="1"':' maxlength="400"')+'>';
        return '<label>'+labEsc(s.label)+input+'</label>';
    }).join('');
}
function labDatasetOptions(){return '<option value="">CSV seçin</option>'+(labCatalog?.datasets||[]).map(p=>'<option value="'+labEsc(p.name)+'">'+labEsc(p.name)+' ('+(p.bytes/1e6).toFixed(1)+' MB)</option>').join('');}
async function labRunTool(event){
    event.preventDefault();const tool=labEl('lab-tool').value,spec=labCatalog.tools[tool];
    const params=labForm('lab-tool-form');if(!params)return;
    for(const [key,field] of Object.entries(spec.fields)){
        if(field.type==='number')params[key]=Number(params[key]);
        if(field.type==='bool')params[key]=Boolean(params[key]);
    }
    if(tool==='dolt'&&params.table==='option_chain'&&!confirm('Yaklaşık 10 GB opsiyon zinciri OneDrive dışına indirilecek. Başlatılsın mı?'))return;
    const button=event.submitter;button.disabled=true;
    try{await labStart('research/run',{tool,parameters:params});}finally{button.disabled=false;}
}
async function labRetrain(event){
    event.preventDefault();const data=labForm('lab-retrain',['min_samples','folds']);if(!data)return;
    data.promote_flag=Boolean(data.promote_flag);data.confirmed=false;
    if(data.promote_flag){data.confirmed=confirm('Yeni model tüm mevcut kanıt kapılarını geçerse canlı meta model dosyası kaydedilecek; önceki sürüm arşivlenecek. Onaylıyor musunuz?');if(!data.confirmed)return;}
    await labStart('ml/retrain',data);
}
async function labDsr(event){
    event.preventDefault();const data=labForm('lab-dsr',['n_trials','blocks']);if(!data)return;
    data.columns=data.columns.split(',').map(s=>s.trim()).filter(Boolean);await labStart('research/run-dsr',data);
}
async function labEdgeRun(event){
    event.preventDefault();const data=labForm('lab-edge-form',['n_folds','thr','warmup_days']);if(!data)return;
    data.feature_cols=data.feature_cols.split(',').map(s=>s.trim()).filter(Boolean);data.pnl_col=data.pnl_col.trim()||null;
    const button=event.submitter;button.disabled=true;
    try{await labStart('research/edge-lab/run',data);}finally{button.disabled=false;}
}
async function labRefresh(){
    try{
        if(!labCatalog){
            labCatalog=await labFetch('lab/catalog');
            labEl('lab-tool').innerHTML=['research','ml','data'].map(group=>'<optgroup label="'+({research:'Araştırma',ml:'Model eğitimi',data:'Veri kaynakları'}[group])+'">'+
                Object.entries(labCatalog.tools).filter(([,s])=>s.group===group).map(([key,s])=>'<option value="'+labEsc(key)+'">'+labEsc(s.label)+'</option>').join('')+'</optgroup>').join('');
            labEl('lab-strategies').innerHTML=labCatalog.strategies.map(s=>'<option>'+labEsc(s)+'</option>').join('');
            document.querySelectorAll('.lab-dataset').forEach(select=>select.innerHTML=labDatasetOptions());labToolFields();
        }
        const list=await labFetch('jobs');
        labEl('lab-job-select').innerHTML='<option value="">İş seçin</option>'+list.jobs.map(j=>'<option value="'+j.job_id+'">'+
            labEsc(new Date(j.created_at*1000).toLocaleString('tr-TR')+' · '+(j.parameters.tool||j.kind)+' · '+(labLabels[j.status]||j.status))+'</option>').join('');
        labEl('lab-job-select').value=labJob;
    }catch(error){labError(error,'lab-job-state');}
}
async function labSelectJob(jobId){
    clearTimeout(labTimer);labJob=jobId;labEl('lab-job-select').value=jobId;if(!jobId)return;
    labEl('lab-job-result').textContent='';labEl('lab-download').disabled=true;
    await labPollJob(jobId);
}
async function labPollJob(jobId){
    try{
        const data=await labFetch('jobs/'+jobId);if(labJob!==jobId)return;
        labEl('lab-job-state').textContent=(labLabels[data.status]||data.status)+' · '+data.elapsed_seconds+' sn'+(data.error?' · '+data.error:'');
        labEl('lab-job-log').textContent=data.log||'Henüz çıktı yok.';
        const epochs=(data.log||'').split('\n').filter(s=>s.startsWith('EPOCH ')).flatMap(s=>{try{return [JSON.parse(s.slice(6))];}catch{return [];}});
        labEl('lab-epochs').innerHTML=epochs.length?'<p class="lab-note">Son epoch ölçümleri; eğitim kaybı son minibatch değeridir.</p>'+labTable(epochs.slice(-10)):'';
        labEl('lab-cancel').disabled=data.status!=='running';
        if(['running','unobserved'].includes(data.status)){labTimer=setTimeout(()=>labPollJob(jobId),2000);return;}
        labEl('lab-download').disabled=false;
        const report=await labFetch('jobs/'+jobId+'/report');if(labJob!==jobId)return;
        labRenderReport(report);
        if(data.kind==='screener'&&data.status==='completed'){labReadScreen();if(typeof loadAllData==='function')loadAllData();}
        if(data.kind==='edge_lab'&&data.status==='completed')labReadEdge();
    }catch(error){labError(error,'lab-job-state');}
}
function labChart(target, labels, datasets, type='line'){
    if(typeof Chart==='undefined'){labEl(target).parentElement.innerHTML='<p class="lab-note">Grafik kütüphanesi yüklenemedi; sayısal sonuçlar aşağıda.</p>';return;}
    labCharts.push(new Chart(labEl(target),{type,data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,animation:false,
        plugins:{legend:{labels:{color:'#94a3b8'}}},scales:{x:{ticks:{color:'#94a3b8',maxTicksLimit:10}},y:{ticks:{color:'#94a3b8'}}}}}));
}
function labRenderReport(report){
    labCharts.forEach(c=>c.destroy());labCharts=[];
    const result=report.result;if(!result){labEl('lab-job-result').innerHTML=labObject(report);return;}
    let html='';
    if(result.chart_job&&/^[0-9a-f]{32}$/.test(result.chart_job))html+='<p>'+labEsc(result.symbol+' · '+result.interval)+'</p><img style="max-width:100%;height:auto" alt="'+labEsc(result.symbol)+' fiyat grafiği" src="/api/telegram/chart/'+result.chart_job+'.png"><div class="lab-actions"><a class="btn-mini" href="/api/telegram/chart/'+result.chart_job+'.png" download="chart.png">PNG indir</a><button class="btn-mini" onclick="labSendChart(\''+result.chart_job+'\')">Bu grafiği Telegram’a gönder</button></div>';
    if(result.note)html+='<p class="lab-note">'+labEsc(result.note)+'</p>';
    if(result.report_type==='fda_calendar')html+='<p>'+labEsc(result.sponsor)+' · '+labEsc(result.cache_state)+' · '+labEsc(result.events.length)+' onay kaydı'+(result.truncated?' · Kaynak sonucu kısmi; tüm başvurular kapsanmıyor.':'')+'</p><div class="lab-out">'+labTable(result.events.slice(0,100),['date','application','sponsor','submission_type','products'])+'</div>';
    if(result.report_type==='pdufa_calendar')html+='<p>'+labEsc(result.n_upcoming)+' yaklaşan PDUFA · '+labEsc(result.n_filings_read)+'/'+labEsc(result.n_filings_total)+' dosyalama · '+labEsc(result.cache_state)+(result.truncated?' · Kaynak sonucu kısmi; tüm dosyalamalar okunmadı.':'')+'</p><div class="lab-out">'+labTable(result.upcoming.slice(0,100),['pdufa_date','ticker','company','n_mentions','revised','possibly_superseded','last_seen'])+'</div>';
    if(result.report_type==='wiki_trends')html+='<p>'+labEsc(result.article)+' · '+labEsc(result.observed_days)+'/'+labEsc(result.requested_days)+' gün · '+labEsc(result.cache_state)+'</p><div class="lab-chart"><canvas id="lab-wiki-views" role="img" aria-label="Günlük Wikipedia görüntülenmeleri ve geçmiş ortalaması"></canvas></div><div class="lab-out">'+labTable(result.rows.slice(-14),['date','views','baseline_mean','ratio','zscore'])+'</div>';
    if(result.verdict)html+='<p class="lab-state">'+labEsc(result.verdict)+'</p>';
    if(result.promoted!==undefined)html+='<p class="lab-note">'+(result.promoted?'Model dosyası kaydedildi; güncel canlı süreçler sonraki turda yeniden yükler.':'Canlı model dosyası değiştirilmedi.')+'</p>';
    if(result.eligible!==undefined){
        html+='<p class="lab-state">'+(result.eligible?'Terfi kapıları geçti':'Terfi uygun değil; kanıt kapıları aşağıda')+'</p>'+labObject(result.checks);
        if(result.eligible&&/^[0-9a-f]{32}$/.test(labJob))html+='<button class="btn-mini" onclick="labActivate(\''+labJob+'\')">İncelenen gölge meta modelini canlıya terfi et</button>';
    }
    if(result.auc_mean!==undefined)html+='<div class="lab-out">'+labTable([result],['tier','n_folds_done','auc_mean','auc_std','brier_mean','ece_mean'])+'</div>';
    if(result.gates)html+='<div class="lab-out">'+labTable(result.gates)+'</div>';
    if(result.columns)html+='<h4>Deflated Sharpe</h4><div class="lab-out">'+labTable(result.columns)+'</div>';
    if(result.pbo)html+='<h4>Backtest aşırı uyum olasılığı (PBO)</h4><div class="lab-out">'+labObject(result.pbo)+'</div>';
    if(result.ablations)html+='<div class="lab-chart"><canvas id="lab-ablation" role="img" aria-label="Konfigürasyonlara göre işlem başına beklenti"></canvas></div><div class="lab-out">'+labTable(result.ablations)+'</div>';
    if(result.results&&Array.isArray(result.results)){
        result.results.forEach((r,index)=>{
            if(r.option_data){const d=r.option_data;html+='<p class="lab-note">Gerçek Dolt bid/ask · günlük kapanış varsayımı · 1 bar gecikme · kotasyon '+labEsc(d.quoted_bars)+'/'+labEsc(d.evaluated_bars)+' gün. '+(d.complete?'Değerleme tamamlandı.':'Eksik değerleme veya kapanmamış pozisyon var; ölçülemeyen metrikler boş bırakıldı.')+' Makas maliyeti (orta fiyata göre): $'+labEsc(d.spread_cost?.toFixed(2))+' · ücret: $'+labEsc(d.fees?.toFixed(2))+'. Kaynak CSV geçmiş veri revizyonlarının yayımlanma zamanını kanıtlamaz.</p>';}
            html+='<h4>'+labEsc(r.symbol)+'</h4><div class="lab-out">'+labTable([r],r.folds?['oos_trades_count','oos_total_return_pct','oos_max_drawdown_pct','oos_sharpe_annualized','wf_efficiency']:['trades_count','total_return_pct','final_equity','sharpe_annualized','sortino_annualized','max_drawdown_pct'])+'</div>';
            if(r.execution_costs){const c=r.execution_costs;html+='<h4>İcra maliyeti ($)</h4>'+labTable(Object.entries({delay_cost:'Gecikme',market_impact:'Karar–dolum farkı',fees:'Ücret',opportunity_cost:'Fırsat maliyeti',total_shortfall:'Toplam shortfall'}).map(([key,label])=>({bileşen:label,tutar:c[key],ölçülen:(c.measured?.[key]??0)+' / '+c.records})))+'<p class="lab-note">Kapanan sanal işlemlerin toplamı; PnL maliyetleri zaten içerir. Tam dolum varsayımında fırsat maliyeti sıfırdır. Filtrelenen adayların miktarı ve değerlendirme fiyatı kaydedilmediğinden bu işlemler toplama dahil değildir.</p>';}
            if(r.equity_curve?.length)html+='<div class="lab-chart"><canvas id="lab-equity-'+index+'" role="img" aria-label="Backtest bakiye eğrisi"></canvas></div>';
            if(r.monte_carlo?.bands)html+='<div class="lab-chart"><canvas id="lab-mc-'+index+'" role="img" aria-label="Monte Carlo yüzde 5, 50 ve 95 bakiye bantları"></canvas></div>';
            if(r.folds?.length)html+='<div class="lab-chart"><canvas id="lab-wf-'+index+'" role="img" aria-label="IS ve OOS ölçütleri"></canvas></div><div class="lab-out">'+labTable(r.folds)+'</div>';
        });
    }
    labEl('lab-job-result').innerHTML=html+'<details><summary>Tam sonuç ve kanıt metrikleri</summary><div class="lab-out">'+labObject(result)+'</div></details>';
    if(result.report_type==='wiki_trends')labChart('lab-wiki-views',result.rows.map(r=>r.date),[{label:'Görüntülenme',data:result.rows.map(r=>r.views),borderColor:'#38bdf8',pointRadius:0},{label:'Önceki 28 gün ortalaması',data:result.rows.map(r=>r.baseline_mean),borderColor:'#a78bfa',pointRadius:0,borderDash:[4,3]}]);
    if(result.ablations)labChart('lab-ablation',result.ablations.map(r=>r.configuration),[{label:'İşlem başına beklenti ($)',data:result.ablations.map(r=>r.weighted_E),backgroundColor:'#38bdf8'}],'bar');
    if(Array.isArray(result.results))result.results.forEach((r,index)=>{
        if(r.equity_curve?.length)labChart('lab-equity-'+index,r.equity_curve.map((_,i)=>i),[{label:'Bakiye ($)',data:r.equity_curve,borderColor:'#38bdf8',pointRadius:0}]);
        if(r.monte_carlo?.bands)labChart('lab-mc-'+index,r.monte_carlo.bands.p50.map((_,i)=>i),Object.entries(r.monte_carlo.bands).map(([name,data],i)=>({label:name,data,borderColor:['#64748b','#22c55e','#a78bfa'][i],pointRadius:0})));
        if(r.folds?.length)labChart('lab-wf-'+index,r.folds.map(f=>f.fold),[{label:'IS',data:r.folds.map(f=>f.is_metric),backgroundColor:'#38bdf8'},{label:'OOS',data:r.folds.map(f=>f.oos_metric),backgroundColor:'#22c55e'}],'bar');
    });
    if(result.models){
        for(const [name,model] of Object.entries(result.models))if(model.feature_importance){
            const id='lab-importance-'+name;labEl('lab-job-result').insertAdjacentHTML('afterbegin','<h4>'+labEsc(name)+' · özellik önemi (OOS AUC düşüşü)</h4><div class="lab-chart"><canvas id="'+labEsc(id)+'"></canvas></div>');
            const rows=model.feature_importance.slice(0,12);labChart(id,rows.map(r=>r.feature),[{label:'OOS AUC düşüşü',data:rows.map(r=>r.auc_drop),backgroundColor:'#38bdf8'}],'bar');
        }
    }
}
async function labCancel(){if(!labJob||!confirm('Bu işin çalışan süreci ve alt işleri durdurulsun mu? Tamamlanmamış hesap kaybolabilir.'))return;try{await labFetch('jobs/'+labJob+'/cancel',{});await labPollJob(labJob);}catch(error){labError(error);}}
async function labDownload(){try{const data=await labFetch('jobs/'+labJob+'/report');const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='nihai-'+labJob+'.json';a.click();URL.revokeObjectURL(url);}catch(error){labError(error);}}
async function labReadEdge(){const data=await labRead('research/edge-lab/summary','lab-edge');if(data)labEl('lab-edge').innerHTML='<p class="lab-note">'+data.families+' aile · '+data.runs+' koşu</p>'+labTable(data.entries,['candidate','all_passed','failed_gates','dsr','n_trials','n_rows','ts']);}
async function labReadScreen(){const data=await labRead('screener/last-report','lab-screen');if(data)labEl('lab-screen').innerHTML='<p class="lab-note">'+labEsc(data.file||data.reason)+'</p>'+labTable(data.rows);}
async function labScreener(event){event.preventDefault();const data=labForm('lab-screener',['top_n']);if(data&&confirm('Tarama sonunda izleme listesi sıralanmış sonuçlarla güncellenecek. Başlatılsın mı?'))await labStart('screener/run-now',data);}
const labTaxQuery=()=>labEl('lab-tax-year').value?'year='+encodeURIComponent(labEl('lab-tax-year').value):'';
async function labReadTax(){await labRead('reports/tax-8949?'+labTaxQuery(),'lab-tax');}
function labTaxDownload(section){window.location.href='/api/reports/tax-8949/download?section='+section+'&'+labTaxQuery();}
async function labStrategyLoad(){try{const data=await labFetch('strategy/status');labEl('lab-active-strategy').innerHTML=data.strategies.map(s=>'<option>'+labEsc(s)+'</option>').join('');const form=labEl('lab-strategy');for(const key of ['strategy','min_votes','stop_atr','target_atr'])form.elements[key].value=data[key];}catch(error){labError(error);}}
async function labStrategySave(event){event.preventDefault();const data=labForm('lab-strategy',['min_votes','stop_atr','target_atr']);if(!data||!confirm('Yeni sinyaller bu strateji ve parametrelerle üretilecek. Kaydedilsin mi?'))return;try{const r=await labFetch('strategy/set-active',data);showToast(r.message,'success');}catch(error){labError(error);}}
async function labReadGuards(){const r=await labRead('guards/blocked-signals','lab-guards');if(r)labEl('lab-guards').innerHTML='<p class="lab-note">'+labEsc(r.note+' '+r.scope)+'</p>'+labTable(r.rows);}
async function labReadTiers(){
    const r=await labRead('system/tiers','lab-tiers');if(!r)return;
    const rows=r.tiers.filter(t=>t.measured);
    labLiveChart('lab-tier-chart',rows.map(t=>t.tier),[
        {label:'Tarama süresi (sn)',data:rows.map(t=>(t.symbols||[]).reduce((s,v)=>s+v.seconds,0)),backgroundColor:'#38bdf8'},
        {label:'Tur içi bekleme (sn)',data:rows.map(t=>Math.max(0,t.last_cycle_seconds-(t.symbols||[]).reduce((s,v)=>s+v.seconds,0))),backgroundColor:'#a78bfa'}],'bar');
    labEl('lab-tiers').innerHTML='<p class="lab-note">'+(r.enabled?'İki katman açık.':'İki katman kapalı; varsa son kayıtlar gösteriliyor.')+' CPU kullanımı değil, son tamamlanan turun süreleri.</p>'+labTable(rows.map(t=>({katman:t.tier,sembol:t.symbols.length,tur:t.cycles,saniye:t.last_cycle_seconds,gözlem:new Date(t.updated_at*1000).toLocaleString('tr-TR')})))+(!rows.length?'<p>Henüz ölçüm yok.</p>':'');
}
async function labReadHub(){
    const r=await labRead('hub/streams','lab-hub');if(!r)return;
    const s=r.streams;if(!r.ok||!s){labEl('lab-hub').textContent=r.reason||'Akış ölçülmedi.';return;}
    const rows=Object.entries(s.tick_latency||{}).map(([kind,m])=>({akış:kind,örnek:m.samples,son_ms:m.last?.delay_ms??null,p50_ms:m.p50_ms,p95_ms:m.p95_ms,azami_ms:m.max_ms,negatif:m.negative_samples,damga_yok:m.missing_timestamps,son_tick:m.last?new Date(m.last.received_at*1000).toLocaleTimeString('tr-TR'):null}));
    labEl('lab-hub').innerHTML='<p class="lab-note">'+labEsc(s.latency_note||'Bu hub sürümü ham tick gecikmesi ölçmüyor; güncel sürümle yeniden başlatın.')+'</p>'+labTable(rows)+'<details><summary>Abonelik ve bağlantı ayrıntıları</summary>'+labObject(s)+'</details>';
}
async function labHub(action){if(confirm('Hub (nihai.data.hub_server) işlemi: '+({start:'başlat',stop:'durdur',restart:'yeniden başlat',sync:'ortak dosyaları eşitle'}[action])+'. Onaylıyor musunuz?'))await labStart('hub/'+action,{confirmed:true});}
async function labReadWatchdogs(){await labRead('system/watchdogs','lab-watchdogs');}
async function labPing(){if(!confirm('Mevcut kanıt sağlığı kontrol edilsin ve sağlıklıysa yapılandırılmış dış deadman adresine ping gönderilsin mi?'))return;try{const r=await labFetch('system/deadman/ping',{confirmed:true});labEl('lab-watchdogs').innerHTML=labObject(r);}catch(error){labError(error,'lab-watchdogs');}}
async function labTelegramTest(){if(!confirm('Yapılandırılmış Telegram sohbetine bir test bildirimi gönderilsin mi?'))return;try{const r=await labFetch('telegram/send-test',{confirmed:true});showToast(r.message,'success');await labRead('telegram/status','lab-telegram');}catch(error){labError(error,'lab-telegram');}}
async function labMaintenance(event,apply){
    event.preventDefault();const data=labForm('lab-maintenance',['days','compress_days']);if(!data)return;
    if(apply){if(!labPreview||JSON.stringify(data)!==JSON.stringify(labPreview.parameters)){labEl('lab-maintenance-apply').disabled=true;labError(new Error('Parametreler değişti; tekrar önizleyin.'));return;}
        if(!confirm(labPreview.delete.length+' dosya silinecek, '+labPreview.compress.length+' dosya sıkıştırılacak. Önizlenen bakımı onaylıyor musunuz?'))return;
        await labStart('system/log-maintenance',{...data,apply:true,confirmed:true,plan_token:labPreview.plan_token});labEl('lab-maintenance-apply').disabled=true;return;}
    try{const r=await labFetch('system/log-maintenance',{...data,apply:false});labPreview={...r,parameters:data};labEl('lab-maintenance-result').innerHTML=labObject(r);labEl('lab-maintenance-apply').disabled=false;}catch(error){labError(error,'lab-maintenance-result');}
}
async function labOpenData(){labOpen();await labRefresh();labEl('lab-tool').value='dolt';labToolFields();labEl('lab-tool').focus();}

document.querySelectorAll('[data-lab-tab]').forEach(section=>labEl(section.dataset.labTab).appendChild(section));
labRefresh();labStrategyLoad();
document.querySelector('[aria-controls="tab-evidence"]').addEventListener('click',labReadEdge);
document.querySelector('[aria-controls="tab-watchlist"]').addEventListener('click',labReadScreen);

const labLiveCharts = new Map(), labTrailHistory = new Map();
let labDeadman = null, labDeadmanAt = 0, labLiveBusy = false, labLiveTick = 0;
const labActive = id => labEl(id).classList.contains('active');
function labLiveChart(id, labels, datasets, type='line'){
    const previous=labLiveCharts.get(id);
    if(typeof Chart==='undefined')return;
    if(previous){previous.data={labels,datasets};previous.update('none');return;}
    labLiveCharts.set(id,new Chart(labEl(id),{type,data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,animation:false,
        plugins:{legend:{labels:{color:'#94a3b8'}}},scales:{x:{ticks:{color:'#94a3b8',maxTicksLimit:8}},y:{ticks:{color:'#94a3b8'}}}}}));
}
async function labTelegramChart(event){event.preventDefault();const data=labForm('lab-telegram-chart',['lookback']);if(data)await labStart('telegram/chart',data);}
async function labSendChart(id){
    if(!confirm('Önizlediğiniz grafik yapılandırılmış Telegram sohbetine gönderilsin mi?'))return;
    try{const r=await labFetch('telegram/chart/'+id+'/send',{confirmed:true});showToast(r.message,'success');}catch(error){labError(error);}
}
async function labReviewPromotion(){await labStart('ml/promote-shadow',{interval:labEl('lab-promotion-interval').value});}
async function labActivate(id){if(confirm('İncelenen model, strateji ve bar aralığında meta kapısı canlı sinyalleri filtreleyecek. Kanıt yeniden denetlensin ve uygunsa terfi edilsin mi?'))await labStart('ml/activate-shadow',{review_job:id,confirmed:true});}
async function labDemote(){if(!confirm('Web üzerinden terfi edilen meta kapısı gölge kipe alınsın mı?'))return;try{const r=await labFetch('ml/demote-shadow',{confirmed:true});showToast(r.message,'success');await labReadPromotion();}catch(error){labError(error);}}
async function labReadPromotion(){
    const r=await labRead('ml/promotion-status','lab-promotion-status');if(!r)return;
    labEl('lab-promotion-status').innerHTML='<p class="lab-note">Model: '+labEsc(r.kind||'yok')+' · '+labEsc(r.model_id||'kimlik yok')+'</p><p>İstenen kip: '+labEsc(r.policy?.mode||'Mevcut yapılandırma')+'</p>'+labTable((r.applied||[]).map(a=>({süreç:a.pid,kip:a.mode,aralık:a.interval,strateji:a.strategy,son_uygulama:new Date(a.observed_at*1000).toLocaleString('tr-TR'),sürüm_eşleşti:a.model_id===r.policy?.model_id})))+'<p class="lab-note">Uygulama kaydı yalnız canlı süreç uygun bir sinyali işlerken yazılır. Boş veya eski kayıt, çalışan sürecin terfiyi aldığını doğrulamaz.</p>'+(r.error?'<p class="lab-error">Terfi kaydı okunamadı: '+labEsc(r.error)+'</p>':'');
}
async function labReadTimeline(){
    const r=await labRead('portfolio/timeline','lab-timeline');if(!r)return;
    if(!r.measured){labEl('lab-timeline').textContent=r.reason;return;}
    labEl('lab-timeline').innerHTML='<p class="lab-note">'+labEsc(r.as_of+' · '+r.timezone+' · '+r.note)+'</p><h4>Nakit çözülme günleri</h4>'+labTable(r.settlements.map(s=>({tarih:s.settle_date,tutar:s.amount,kalan_gün:s.days_left,durum:s.matured?'Vadesi geldi':'Bekliyor'})))+'<h4>Wash-sale izleme / karantina</h4>'+labTable(r.quarantine.map(q=>({sembol:q.symbol,son_zarar:q.last_loss_date,zarar:q.loss,işlem:q.trades,serbest_gün:q.release_date,kalan_gün:q.days_left})))+'<p class="lab-note">Kayıt zamanı: '+labEsc(r.saved_at)+'</p>';
}
function labCapturePositions(positions){
    if(!Array.isArray(positions)){labEl('lab-trail-state').textContent='Pozisyon verisi alınamadı; son grafik güncel olmayabilir.';return;}
    const select=labEl('lab-trail-position'),selected=select.value,keys=new Set();
    // ponytail: son 120 tarayıcı gözlemi; kalıcı geçmiş gerekirse motorun durum defterine eklenir.
    for(const p of positions.filter(p=>p.asset_type==='stock')){
        const key=JSON.stringify([p.symbol,p.side,p.entry_time]);keys.add(key);
        let item=labTrailHistory.get(key);
        if(!item){item={position:p,points:[]};labTrailHistory.set(key,item);}
        item.position=p;
        const stamp=p.state_updated_at??Date.now()/1000;
        if(item.points.at(-1)?.time!==stamp){
            item.points.push({time:stamp,price:p.last_mark>0?p.last_mark:null,stop:p.stop>0?p.stop:null});
            if(item.points.length>120)item.points.shift();
        }
    }
    for(const key of labTrailHistory.keys())if(!keys.has(key))labTrailHistory.delete(key);
    select.innerHTML=[...labTrailHistory].map(([key,v])=>'<option value="'+labEsc(key)+'">'+labEsc(v.position.symbol+' '+v.position.side+' · '+v.position.entry_time)+'</option>').join('');
    if(keys.has(selected))select.value=selected;
    labDrawTrail();
}
function labDrawTrail(){
    const item=labTrailHistory.get(labEl('lab-trail-position').value);
    if(!item){labEl('lab-trail-state').textContent='Açık hisse pozisyonu yok.';labEl('lab-trail-values').innerHTML='';labLiveChart('lab-trail-chart',[],[]);return;}
    const p=item.position,points=item.points;
    labEl('lab-trail-state').textContent=(p.trailing_enabled?'Trailing açık':'Trailing kapalı; mevcut stop gösteriliyor')+' · son durum: '+new Date(points.at(-1).time*1000).toLocaleString('tr-TR');
    labEl('lab-trail-values').innerHTML=labTable([{sembol:p.symbol,son_fiyat:p.last_mark||null,stop:p.stop||null,giriş:p.entry_price}]);
    labLiveChart('lab-trail-chart',points.map(v=>new Date(v.time*1000).toLocaleTimeString('tr-TR')),[
        {label:'Son işaretlenen fiyat ($)',data:points.map(v=>v.price),borderColor:'#38bdf8',pointRadius:2},
        {label:'Aktif stop ($)',data:points.map(v=>v.stop),borderColor:'#f87171',stepped:true,pointRadius:2}]);
}
async function labReadDeadman(){
    try{labDeadman=await labFetch('system/deadman');labDeadmanAt=Date.now();labRenderDeadman();}
    catch(error){labEl('lab-deadman-state').textContent='Sağlık sorgusu başarısız; yeniden denenecek.';}
}
function labDeadmanView(data,elapsed){
    if(!data)return {alarm:elapsed>45,text:elapsed>45?'Deadman sağlık bağlantısı kurulamadı.':'Sağlık ölçümü bekleniyor.',rows:[]};
    const open=data.market_open===true&&!data.project_stopped;
    const rows=Object.entries(data.files).map(([name,f])=>{
        const left=f.age_min==null?null:(f.limit_min+(data.grace_min||0)-f.age_min)*60-elapsed;
        return {name,left};
    });
    if(elapsed>45)return {alarm:true,text:'Deadman sağlık bağlantısı kesildi; durum doğrulanamıyor.',rows};
    const expired=rows.filter(r=>r.left!==null&&r.left<0);
    const alarm=open&&expired.length>0;
    return {alarm,text:alarm?'Kanıt akışı gecikti: '+expired.map(r=>r.name).join(', '):!open?
        ({SEANS_KAPALI:'Seans kapalı; alarm beklemede.',PROJE_KAPALI:'Proje kapalı; alarm beklemede.',BILINMIYOR:'Seans durumu bilinmiyor.'}[data.verdict]||data.verdict):
        rows.some(r=>r.left===null)?'Bazı kaynaklarda henüz ölçüm yok.':'Kanıt akışı eşik içinde.',rows};
}
function labRenderDeadman(){
    const v=labDeadmanView(labDeadman,labDeadmanAt?(Date.now()-labDeadmanAt)/1000:0),banner=labEl('lab-deadman-alarm');
    const fresh=labEl('lab-deadman-state');if(fresh.textContent!==v.text)fresh.textContent=v.text;
    if(banner.textContent!==v.text)banner.textContent=v.text;
    banner.hidden=!v.alarm;
    labEl('lab-deadman-countdown').innerHTML=labTable(v.rows.map(r=>({kaynak:r.name,kalan:r.left===null?'Ölçülmedi':r.left<0?Math.ceil(-r.left)+' sn gecikti':Math.floor(r.left/60)+' dk '+Math.floor(r.left%60)+' sn'})));
}
async function labPollLive(){
    if(labLiveBusy||document.hidden)return;
    labLiveBusy=true;
    try{
        const tasks=[];
        if(labLiveTick%3===0)tasks.push(labReadDeadman());
        if(labActive('tab-system')){tasks.push(labReadHub());if(typeof loadMetrics==='function')tasks.push(loadMetrics());}
        if(labActive('tab-operator'))tasks.push(labReadTiers());
        if(labActive('tab-risk')&&labLiveTick%3===0)tasks.push(labReadTimeline());
        if(labActive('tab-lab')&&labLiveTick%3===0)tasks.push(labReadPromotion());
        labLiveTick++;await Promise.allSettled(tasks);
    }finally{labLiveBusy=false;}
}
document.querySelector('[aria-controls="tab-risk"]').addEventListener('click',labReadTimeline);
document.querySelector('[aria-controls="tab-operator"]').addEventListener('click',labReadTiers);
document.querySelector('[aria-controls="tab-system"]').addEventListener('click',labReadHub);
labDeadmanAt=Date.now();setInterval(labPollLive,5000);setInterval(labRenderDeadman,1000);labReadDeadman();
