"""Local research dashboard for Digital Twin cloud removal."""
import argparse
import base64
import csv
import io
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from evaluation.baselines import spatial_inpainting, temporal_replacement
from evaluation.metrics import calculate_metrics
from preprocessing.cloud_mask import cloud_coverage, estimate_mask
from testing.inference import load_model, reconstruct
from digital_twin.twin_builder import build_twin

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MAX_UPLOAD_BYTES = 400 * 1024 * 1024
PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TerraTwin | Cloud Reconstruction</title>
<style>
:root{--ink:#142b3b;--muted:#688090;--line:#e4ebef;--paper:#f5f8fa;--white:#fff;--blue:#176b87;--mint:#dff5ee;--green:#18795c;--amber:#a66a10;--shadow:0 8px 24px #24465d0c}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}button,input,select{font:inherit}button{cursor:pointer}.shell{display:grid;grid-template-columns:248px 1fr;min-height:100vh}.side{background:#102b3b;color:#dceaf0;padding:24px 16px;display:flex;flex-direction:column;gap:26px}.brand{display:flex;align-items:center;gap:12px;padding:0 8px}.logo{width:38px;height:38px;background:#6de0bf;color:#102b3b;border-radius:12px;display:grid;place-items:center;font-weight:900;font-size:18px}.brand strong{font-size:15px;letter-spacing:.02em}.brand small{display:block;color:#98b2bf;font-size:11px}.nav{display:grid;gap:6px}.nav button{border:0;background:transparent;color:#a9c0cb;text-align:left;padding:12px;border-radius:9px}.nav button:hover,.nav button.active{background:#1d4053;color:#fff}.nav button.active{box-shadow:inset 3px 0 #6de0bf}.side-foot{margin-top:auto;color:#93acb9;font-size:11px;padding:8px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#efb85e;margin-right:7px}.dot.ok{background:#68d1aa}.main{min-width:0}.top{height:70px;background:#fff;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 34px;position:sticky;top:0;z-index:2}.crumb{color:var(--muted);font-size:12px}.top-right{display:flex;gap:12px;align-items:center;color:var(--muted);font-size:12px}.pill{border:1px solid var(--line);border-radius:30px;padding:7px 11px;background:#fff}.content{max-width:1440px;padding:30px 34px 60px;margin:auto}.view{display:none}.view.active{display:block}.heading{display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:24px;gap:16px}.heading h1{margin:0;font-size:26px;letter-spacing:-.04em}.heading p{margin:6px 0 0;color:var(--muted)}.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-weight:700;color:var(--blue);font-size:10px}.button{border:1px solid #d4e0e6;background:#fff;color:var(--ink);padding:9px 13px;border-radius:8px;font-weight:650}.button:hover{border-color:#8eb5c5}.button.primary{background:var(--blue);color:white;border-color:var(--blue)}.button:disabled{opacity:.45;cursor:not-allowed}.grid{display:grid;gap:16px}.stats{grid-template-columns:repeat(4,minmax(0,1fr));margin-bottom:18px}.card{background:#fff;border:1px solid var(--line);border-radius:13px;padding:18px;box-shadow:var(--shadow);min-width:0}.stat-label{color:var(--muted);font-size:12px}.stat-value{font-size:27px;font-weight:720;letter-spacing:-.04em;margin:8px 0 2px}.stat-note{font-size:11px;color:var(--muted)}.cols{grid-template-columns:1.35fr 1fr}.card-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}.card h2{font-size:15px;margin:0}.card h3{font-size:13px;margin:0}.subtle{color:var(--muted);font-size:12px}.status-chip{font-size:11px;padding:4px 9px;border-radius:20px;background:#f0f4f6;color:#687f8b}.status-chip.good{background:#e5f6ef;color:var(--green)}.status-chip.warn{background:#fff4df;color:var(--amber)}.empty{border:1px dashed #ccdae1;border-radius:10px;padding:28px 18px;text-align:center;color:var(--muted)}.empty b{display:block;color:var(--ink);margin-bottom:5px}.actions{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px}.actions .button{padding:13px 8px}.quick{display:grid;gap:10px}.quick-row{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line);padding:9px 0}.quick-row:last-child{border:0}.quick-row span:first-child{color:var(--muted)}.drop{border:1px dashed #a9c2ce;background:#f8fbfc;border-radius:13px;padding:27px;text-align:center;transition:.15s}.drop.over{background:#e7f6f1;border-color:#55a991}.drop input{max-width:100%;margin:12px auto}.drop p{margin:5px;color:var(--muted)}.upload-layout{grid-template-columns:minmax(0,1.1fr) minmax(280px,.9fr)}.hint{font-size:12px;color:var(--muted);background:#f5f8fa;border-radius:8px;padding:11px;margin-top:12px}.outputs{grid-template-columns:repeat(3,minmax(0,1fr));margin-top:16px}.preview{background:#eef3f5;border-radius:9px;min-height:180px;display:grid;place-items:center;overflow:hidden}.preview img{width:100%;height:220px;object-fit:contain}.preview .blank{color:#8aa0ac;font-size:12px}.preview-label{font-size:12px;font-weight:650;margin:0 0 8px}.metric-grid{grid-template-columns:repeat(5,minmax(0,1fr));margin-top:14px}.metric{background:#f5f8fa;border-radius:9px;padding:12px}.metric b{display:block;font-size:17px}.metric span{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em}.formrow{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.formrow label{font-size:12px;color:var(--muted)}.formrow select,.formrow input{padding:8px 10px;border:1px solid var(--line);border-radius:7px;background:white}.job-log{background:#102b3b;color:#d4e8ef;padding:14px;border-radius:9px;max-height:300px;overflow:auto;white-space:pre-wrap;font:11px/1.5 ui-monospace,monospace}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em}.chart{width:100%;height:210px}.filelist{display:grid;gap:8px}.fileitem{display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid var(--line)}.notice{padding:13px 15px;border-radius:9px;background:#fff5df;color:#775117;margin-bottom:16px}.notice.good{background:#e8f7f0;color:#176548}.toast{position:fixed;right:20px;bottom:20px;background:#102b3b;color:white;padding:12px 16px;border-radius:9px;box-shadow:var(--shadow);z-index:9;display:none}.footnote{font-size:11px;color:var(--muted);margin-top:14px}@media(max-width:1000px){.shell{grid-template-columns:200px 1fr}.stats{grid-template-columns:repeat(2,1fr)}.cols,.upload-layout{grid-template-columns:1fr}}@media(max-width:700px){.shell{grid-template-columns:1fr}.side{padding:12px;gap:10px}.brand{padding:0 4px}.nav{grid-template-columns:repeat(4,1fr)}.nav button{font-size:11px;text-align:center;padding:8px}.side-foot{display:none}.top{height:54px;padding:0 14px}.content{padding:20px 14px}.heading{align-items:flex-start;flex-direction:column}.stats{gap:9px}.stat-value{font-size:22px}.outputs{grid-template-columns:repeat(2,minmax(0,1fr))}.metric-grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="shell"><aside class="side"><div class="brand"><div class="logo">T</div><div><strong>TERRATWIN</strong><small>Earth observation lab</small></div></div><nav class="nav"><button class="active" data-view="overview">◫ &nbsp; Overview</button><button data-view="reconstruct">◉ &nbsp; Reconstruct</button><button data-view="experiments">▤ &nbsp; Experiments</button><button data-view="data">▧ &nbsp; Data & model</button></nav><div class="side-foot"><span id="sideDot" class="dot"></span><span id="sideStatus">Connecting to local service</span><p>Digital Twin assisted cloud removal</p></div></aside><main class="main"><header class="top"><div class="crumb">Research workspace <span style="color:#b4c3ca"> / </span><b id="crumbName">Overview</b></div><div class="top-right"><span class="pill">LOCAL INFERENCE</span><span id="clock"></span></div></header><div class="content">
<section class="view active" id="view-overview"><div class="heading"><div><div class="eyebrow">Workspace overview</div><h1>Cloud removal, with temporal context.</h1><p>Inspect the model, prepare scenes and track reproducible experiments.</p></div><button class="button primary" onclick="showView('reconstruct')">＋ &nbsp; New reconstruction</button></div><div id="datasetNotice"></div><div class="grid stats"><div class="card"><div class="stat-label">Model status</div><div class="stat-value" id="modelStat">—</div><div class="stat-note" id="modelNote">Checking checkpoint</div></div><div class="card"><div class="stat-label">Manifest scenes</div><div class="stat-value" id="sceneStat">—</div><div class="stat-note" id="splitNote">Train / validation / test</div></div><div class="card"><div class="stat-label">Test scenes scored</div><div class="stat-value" id="testedStat">—</div><div class="stat-note">Held-out scenes with computed metrics</div></div><div class="card"><div class="stat-label">Latest inference</div><div class="stat-value" id="inferStat">—</div><div class="stat-note" id="inferNote">No scene processed in this session</div></div></div><div class="grid cols"><div class="card"><div class="card-head"><div><h2>Research pipeline</h2><div class="subtle">Status from files in this workspace</div></div><span class="status-chip" id="pipeChip">Checking</span></div><div class="quick" id="pipelineRows"></div></div><div class="card"><div class="card-head"><h2>Quick actions</h2></div><div class="actions"><button class="button" onclick="launchJob('train')">Train model</button><button class="button" onclick="showView('reconstruct')">Upload scene</button><button class="button" onclick="launchJob('test')">Run evaluation</button><button class="button" id="publicDemoBtn" onclick="runPublicDemo()">Public test demo</button></div><div class="hint">Training and evaluation run as background jobs. Outputs are saved beneath <code>results/</code>.</div><div style="height:20px"></div><div class="card-head"><h2>Latest held-out metrics</h2><span class="subtle" id="metricsSource">No test report</span></div><div id="summaryMetrics" class="metric-grid"><div class="empty" style="grid-column:1/-1">Run the test command to populate held-out metrics.</div></div></div></div><div class="card" style="margin-top:16px"><div class="card-head"><h2>Recent outputs</h2><button class="button" onclick="refreshStatus()">Refresh</button></div><div id="recentFiles" class="filelist"><div class="empty">No output figures found yet.</div></div></div></section>
<section class="view" id="view-reconstruct"><div class="heading"><div><div class="eyebrow">Model inference</div><h1>Reconstruct a cloudy scene</h1><p>Upload a normalized multispectral NPZ or a GeoTIFF with the model’s expected band count.</p></div><span class="status-chip" id="inferReady">Checking model</span></div><div id="noModelNotice"></div><div class="grid upload-layout"><div class="card"><div class="drop" id="drop"><div style="font-size:30px;color:#176b87">⇧</div><b>Drop a scene file here</b><p>or choose a file from your computer</p><input id="sceneFile" type="file" accept=".npz,.tif,.tiff"><div class="subtle">NPZ contract: cloudy (C,H,W), optional mask (H,W), history (T,C,H,W), history_masks, and clear for scoring.</div></div><div class="formrow" style="margin-top:14px"><label><input type="checkbox" id="autoMask" checked> Estimate mask if absent</label><button class="button primary" id="runInference" onclick="runInference()">Run reconstruction</button><span id="inferMessage" class="subtle"></span></div><div class="hint">Use reflectance normalized to [0,1], in the same band order used for training. For geospatial TIFF preparation and alignment, see the README data-preparation steps.</div></div><div class="card"><div class="card-head"><h2>Scene details</h2></div><div class="quick"><div class="quick-row"><span>Loaded checkpoint</span><b id="detailCheckpoint">—</b></div><div class="quick-row"><span>Expected optical bands</span><b id="detailBands">—</b></div><div class="quick-row"><span>Temporal reference</span><b id="detailHistory">Optional</b></div><div class="quick-row"><span>Output scale</span><b>Reflectance 0–1</b></div></div><div class="hint">Without history arrays, Digital Twin channels are zero-support. That result is an image-only fallback, not a temporal Twin reconstruction.</div></div></div><div id="inferenceResults" style="display:none"><div class="card" style="margin-top:16px"><div class="card-head"><div><h2>Reconstruction results</h2><span class="subtle" id="resultScene"></span></div><div class="formrow"><a id="downloadNpz" class="button" download="reconstruction.npz">Download NPZ</a><a id="downloadGeo" class="button" style="display:none" download="reconstruction.tif">Download GeoTIFF</a></div></div><div id="metricGrid" class="grid metric-grid"></div></div><div id="demoNotice"></div><div id="previews" class="grid outputs"></div></div></section>
<section class="view" id="view-experiments"><div class="heading"><div><div class="eyebrow">Training & evaluation</div><h1>Experiment control</h1><p>Launch reproducible jobs using the manifest and configuration file.</p></div></div><div class="grid cols"><div class="card"><div class="card-head"><h2>Run a job</h2></div><div class="formrow"><label>Experiment</label><select id="jobMode"><option value="train">Train proposed model</option><option value="test">Evaluate held-out test split</option><option value="experiments">Train seven ablations</option></select><button class="button primary" onclick="launchJob()">Start job</button></div><p class="subtle">Ablations independently train seven feature variants; runtime depends on dataset and hardware. Jobs can be monitored here and continue if you navigate to another tab.</p><div id="jobState" class="hint">No job has been started.</div></div><div class="card"><div class="card-head"><h2>Experiment outputs</h2></div><div id="resultFiles" class="filelist"><div class="empty">No generated reports.</div></div><div class="hint">Metrics are calculated from the manifest test split. Synthetic smoke checks are never written here.</div></div></div><div class="card" style="margin-top:16px"><div class="card-head"><h2>Job log</h2><span id="jobChip" class="status-chip">Idle</span></div><pre id="jobLog" class="job-log">Waiting for a job.</pre></div><div class="card" style="margin-top:16px"><div class="card-head"><h2>Test set summary</h2><span class="subtle">Aggregates the real test_metrics.csv</span></div><div class="table-wrap" id="metricsTable"><div class="empty">No test metrics available yet.</div></div></div></section>
<section class="view" id="view-data"><div class="heading"><div><div class="eyebrow">Dataset & checkpoint</div><h1>Data readiness</h1><p>Check split completeness and input requirements before running experiments.</p></div><button class="button" onclick="refreshStatus()">Refresh status</button></div><div class="grid cols"><div class="card"><div class="card-head"><h2>Dataset manifest</h2><span id="manifestChip" class="status-chip">Not found</span></div><div class="quick" id="splitRows"></div><div class="hint">Keep geography disjoint across train, validation and test. Add historical archive observations as <code>reference</code> rows. NPZ array keys: cloudy, clear and mask.</div><div class="actions" style="margin-top:12px"><button class="button" onclick="launchJob('prepare')">Validate manifest</button><a class="button" href="/api/docs?name=dataset/README.md" target="_blank">Data format guide</a></div></div><div class="card"><div class="card-head"><h2>Model checkpoint</h2><span id="checkpointChip" class="status-chip">Not found</span></div><div class="quick"><div class="quick-row"><span>Path</span><b id="checkpointPath">checkpoints/best.pt</b></div><div class="quick-row"><span>Parameters</span><b id="parameterCount">—</b></div><div class="quick-row"><span>Device</span><b id="deviceName">—</b></div><div class="quick-row"><span>Input band count</span><b id="bandCount">—</b></div></div><div class="hint">To train: add calibrated, aligned samples and a region-disjoint manifest, then start “Train proposed model”.</div></div></div><div class="card" style="margin-top:16px"><div class="card-head"><h2>Recommended preparation</h2></div><ol class="subtle"><li>Download same-area LISS-IV scenes from Bhoonidhi or use the SEN12MS-CR-TS benchmark.</li><li>Calibrate and align observations to one CRS/grid; create validated 0/1/2 cloud/shadow masks.</li><li>Package each acquisition as NPZ and record source, region and date in <code>dataset/manifest.csv</code>.</li><li>Split by region before training. Mark legitimate earlier archive scenes as <code>reference</code>.</li><li>Train, evaluate, then run ablations on the identical held-out test scenes.</li></ol><p class="footnote">A satellite weather image or RGB screenshot is not a substitute for calibrated multispectral target data.</p></div></section>
</div></main></div><div id="toast" class="toast"></div>
<script>
const $=s=>document.querySelector(s);let pollTimer=null,latestStatus=null;
function toast(t){let x=$('#toast');x.textContent=t;x.style.display='block';setTimeout(()=>x.style.display='none',3300)}
function showView(v){document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x.id==='view-'+v));document.querySelectorAll('.nav button').forEach(x=>x.classList.toggle('active',x.dataset.view===v));$('#crumbName').textContent={overview:'Overview',reconstruct:'Reconstruct',experiments:'Experiments',data:'Data & model'}[v]||v;location.hash=v;if(v==='experiments')refreshJob()}
document.querySelectorAll('.nav button').forEach(b=>b.onclick=()=>showView(b.dataset.view));if(location.hash)showView(location.hash.slice(1));
$('#clock').textContent=new Date().toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric'});
function n(v){return v==null?'—':Number(v).toLocaleString()}
function setRows(el,arr){$(el).innerHTML=arr.map(([k,v])=>`<div class="quick-row"><span>${k}</span><b>${v}</b></div>`).join('')}
function metricCards(m){if(!m)return '<div class="empty" style="grid-column:1/-1">No measured test metrics yet. Run held-out evaluation after training.</div>';let items=[['PSNR',m.full_psnr,'dB'],['SSIM',m.full_ssim,''],['Cloud PSNR',m.cloud_psnr,'dB'],['Cloud MAE',m.cloud_mae,''],['Cloud SAM',m.cloud_sam,'°']];return items.map(([k,v,u])=>`<div class="metric"><span>${k}</span><b>${Number.isFinite(+v)?(+v).toFixed(3)+u:'—'}</b></div>`).join('')}
function safeName(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
async function refreshStatus(){try{let r=await fetch('/api/status');latestStatus=await r.json();renderStatus(latestStatus);let j=await fetch('/api/results');renderFiles(await j.json());}catch(e){$('#sideStatus').textContent='Local service unavailable';$('#sideDot').className='dot';}}
function renderStatus(s){$('#sideStatus').textContent=s.model.loaded?'Model ready':'Model checkpoint missing';$('#sideDot').className='dot '+(s.model.loaded?'ok':'');$('#modelStat').textContent=s.model.loaded?'Ready':'Not trained';$('#modelNote').textContent=s.model.name||'Train a checkpoint to enable inference';$('#sceneStat').textContent=n(s.data.total);$('#splitNote').textContent=`${s.data.train} train · ${s.data.validation} validation · ${s.data.test} test · ${s.data.reference} reference`;
$('#testedStat').textContent=n(s.results.scenes);$('#publicDemoBtn').disabled=!s.demo_available;$('#publicDemoBtn').title=s.demo_available?'Run the local bundled SEN12MS-CR demo':'No local pretrained demo is bundled in this repository';$('#metricsSource').textContent=s.results.available?'test_metrics.csv':'No test report';$('#summaryMetrics').innerHTML=metricCards(s.results.mean);$('#detailCheckpoint').textContent=s.model.name||'Not loaded';$('#detailBands').textContent=s.model.bands??'—';$('#detailHistory').textContent='Optional NPZ arrays';$('#inferReady').textContent=s.model.loaded?'Model ready':'Checkpoint required';$('#inferReady').className='status-chip '+(s.model.loaded?'good':'warn');$('#runInference').disabled=!s.model.loaded;$('#noModelNotice').innerHTML=s.model.loaded?'':'<div class="notice">No research checkpoint is loaded. This dashboard is ready, but inference becomes available after you add scenes and train <code>checkpoints/best.pt</code>.</div>';
$('#manifestChip').textContent=s.data.manifest?'Manifest found':'Manifest missing';$('#manifestChip').className='status-chip '+(s.data.manifest?'good':'warn');$('#checkpointChip').textContent=s.model.exists?'Checkpoint found':'Checkpoint missing';$('#checkpointChip').className='status-chip '+(s.model.exists?'good':'warn');$('#checkpointPath').textContent=s.model.path;$('#parameterCount').textContent=s.model.parameters?n(s.model.parameters):'—';$('#deviceName').textContent=s.model.device||'—';$('#bandCount').textContent=s.model.bands??'—';
setRows('#splitRows',[['Manifest path',safeName(s.data.manifest_path)],['Training scenes',n(s.data.train)],['Validation scenes',n(s.data.validation)],['Test scenes',n(s.data.test)],['Reference scenes',n(s.data.reference)],['Region split','Disjoint check on launch']]);
let steps=[['Dataset manifest',s.data.manifest?'Ready':'Needs data'],['Trained checkpoint',s.model.exists?'Available':'Not trained'],['Held-out evaluation',s.results.available?'Metrics available':'Pending'],['Temporal reference archive',s.data.reference?'Available':'None registered']];setRows('#pipelineRows',steps);$('#pipeChip').textContent=s.model.loaded?'Inference ready':'Data setup required';$('#pipeChip').className='status-chip '+(s.model.loaded?'good':'warn');$('#datasetNotice').innerHTML=s.data.total?'<div class="notice good">Manifest detected. Confirm its train/validation/test regions are disjoint before training.</div>':'<div class="notice">No research manifest detected at <code>dataset/manifest.csv</code>. Add real scenes to enable training and paper evaluation. The bundled single-chip demo is not a temporal benchmark.</div>';if(s.results.available)renderMetricsTable(s.results.rows)}
function renderMetricsTable(rows){if(!rows?.length)return;let cols=['variant','scene','model_full_psnr','model_full_ssim','model_cloud_psnr','model_cloud_mae'];let keys=Object.keys(rows[0]);cols=cols.filter(c=>keys.includes(c));if(!cols.length)cols=keys.slice(0,7);$('#metricsTable').innerHTML='<div class="table-wrap"><table><thead><tr>'+cols.map(c=>`<th>${safeName(c)}</th>`).join('')+'</tr></thead><tbody>'+rows.slice(0,30).map(r=>'<tr>'+cols.map(c=>`<td>${safeName(typeof r[c]==='number'?r[c].toFixed(4):r[c])}</td>`).join('')+'</tr>').join('')+'</tbody></table></div>'}
function renderFiles(d){let render=(arr,el)=>{if(!arr?.length){$(el).innerHTML='<div class="empty">No generated reports or figures yet.</div>';return}$(el).innerHTML=arr.map(f=>`<div class="fileitem"><span>${safeName(f.name)}</span><a href="/api/file?name=${encodeURIComponent(f.name)}" target="_blank">Open</a></div>`).join('')};render(d.recent,'#recentFiles');render(d.reports,'#resultFiles')}
async function runPublicDemo(){try{showView('reconstruct');$('#inferMessage').textContent='Loading held-out Sentinel demo…';let r=await fetch('/api/demo',{method:'POST'});let d=await r.json();if(!r.ok)throw Error(d.error);renderInference(d,d.scene||'SEN12MS-CR held-out demo');$('#inferMessage').textContent='Demo complete';toast('Public benchmark demo complete')}catch(e){toast(e.message);$('#inferMessage').textContent=''}}
function renderInference(d,label){$('#inferenceResults').style.display='block';$('#resultScene').textContent=`${label} · ${d.width}×${d.height} · ${d.bands} bands · ${d.cloud_percent?.toFixed(1)??'—'}% cloud estimate` ;$('#previews').innerHTML=d.previews.map(x=>`<div class="card"><div class="preview-label">${safeName(x.label)}</div><div class="preview">${x.image?`<img alt="${safeName(x.label)}" src="data:image/png;base64,${x.image}">`:'<span class="blank">Not available in benchmark sample</span>'}</div></div>`).join('');$('#metricGrid').innerHTML=metricCards(d.metrics);$('#demoNotice').innerHTML=d.demo_info?`<div class="notice">${safeName(d.demo_info.scope)} Source: ${safeName(d.demo_info.source)} · ${safeName(d.demo_info.method)}. This chip has no Twin history or supplied cloud mask.</div>`:'';if(d.output){$('#downloadNpz').href='data:application/octet-stream;base64,'+d.output;$('#downloadNpz').style.display='inline-block'}else $('#downloadNpz').style.display='none';$('#downloadGeo').style.display='none';$('#inferStat').textContent='Complete';$('#inferNote').textContent=new Date().toLocaleTimeString()}
async function runInference(){let f=$('#sceneFile').files[0];if(!f)return toast('Choose an NPZ or GeoTIFF scene first');let fd=new FormData();fd.append('scene',f);fd.append('auto_mask',$('#autoMask').checked?'1':'0');$('#runInference').disabled=true;$('#inferMessage').textContent='Running model…';try{let r=await fetch('/api/infer',{method:'POST',body:fd});let d=await r.json();if(!r.ok)throw Error(d.error||'Inference failed');renderInference(d,f.name);if(d.geotiff){$('#downloadGeo').href='data:image/tiff;base64,'+d.geotiff;$('#downloadGeo').style.display='inline-block'}$('#inferMessage').textContent='Finished';$('#inferStat').textContent='Complete';$('#inferNote').textContent=new Date().toLocaleTimeString();toast('Reconstruction complete');refreshStatus()}catch(e){$('#inferMessage').textContent='';toast(e.message)}finally{$('#runInference').disabled=!latestStatus?.model?.loaded}}
$('#drop').addEventListener('dragover',e=>{e.preventDefault();$('#drop').classList.add('over')});$('#drop').addEventListener('dragleave',()=>$('#drop').classList.remove('over'));$('#drop').addEventListener('drop',e=>{e.preventDefault();$('#drop').classList.remove('over');if(e.dataTransfer.files.length){$('#sceneFile').files=e.dataTransfer.files;toast('Scene selected')}});
let activeJob=null;async function launchJob(mode){mode=mode||$('#jobMode').value;if(!['train','test','experiments','prepare'].includes(mode))return;try{let r=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode})});let d=await r.json();if(!r.ok)throw Error(d.error);activeJob=d.id;showView('experiments');$('#jobState').textContent=`${mode} started · job ${d.id}`;toast(`${mode} job started`);pollJob()}catch(e){toast(e.message)}}
async function refreshJob(){try{let r=await fetch('/api/jobs');let d=await r.json();if(d.job){activeJob=d.job.id;updateJob(d.job);if(d.job.status==='running')pollJob()}}catch(e){}}
function updateJob(j){$('#jobState').textContent=`${j.mode} · ${j.status}${j.exit_code==null?'':` · exit ${j.exit_code}`}`;$('#jobChip').textContent=j.status;$('#jobChip').className='status-chip '+(j.status==='completed'?'good':j.status==='running'?'warn':'');$('#jobLog').textContent=j.log||'No log output yet';$('#jobLog').scrollTop=$('#jobLog').scrollHeight}
async function pollJob(){if(pollTimer)clearTimeout(pollTimer);try{let r=await fetch('/api/jobs');let d=await r.json();if(d.job){updateJob(d.job);if(d.job.status==='running')pollTimer=setTimeout(pollJob,1800);else{refreshStatus();toast(`${d.job.mode} ${d.job.status}`)}}}catch(e){}}
refreshStatus();refreshJob();setInterval(refreshStatus,12000);
</script></body></html>'''


def _b64_image(array, mode="rgb"):
    x = np.asarray(array)
    if mode == "mask":
        rgb = np.zeros((*x.shape, 3), np.uint8)
        rgb[x == 1] = [240, 94, 77]
        rgb[x == 2] = [64, 137, 225]
        rgb[x == 0] = [223, 233, 236]
        image = Image.fromarray(rgb)
    elif mode == "gray":
        q = x.squeeze()
        if q.max() > q.min(): q = (q - q.min()) / (q.max() - q.min())
        image = Image.fromarray((np.clip(q, 0, 1) * 255).astype(np.uint8), "L")
    else:
        if x.shape[0] >= 4: c = x[[3, 2, 1]]  # Sentinel-2 natural-color bands.
        elif x.shape[0] == 3: c = x[[2, 1, 0]]  # LISS-IV false-color composite.
        else: c = np.repeat(x[:1], 3, axis=0)
        image = Image.fromarray((np.moveaxis(np.clip(c, 0, 1), 0, -1) * 255).astype(np.uint8), "RGB")
    buf = io.BytesIO(); image.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _read_scene(filename, raw, auto_mask=True):
    suffix = Path(filename).suffix.lower()
    if suffix == ".npz":
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        if "cloudy" not in z: raise ValueError("NPZ must contain an array named 'cloudy'")
        cloudy = np.asarray(z["cloudy"], np.float32)
        if cloudy.ndim != 3: raise ValueError("cloudy must use channels-first shape (bands,height,width)")
        mask = np.asarray(z["mask"], np.uint8) if "mask" in z else None
        hist = np.asarray(z["history"], np.float32) if "history" in z else None
        hm = np.asarray(z["history_masks"], np.uint8) if "history_masks" in z else None
        clear = np.asarray(z["clear"], np.float32) if "clear" in z else None
        return cloudy, mask, hist, hm, clear, None
    if suffix in (".tif", ".tiff"):
        try:
            import rasterio
        except ImportError as exc:
            raise ValueError("Install rasterio to upload GeoTIFF files; NPZ works without it") from exc
        with rasterio.MemoryFile(raw) as mem:
            with mem.open() as src:
                cloudy = src.read().astype(np.float32)
                nodata = src.nodata
                profile = src.profile.copy()
        if nodata is not None and np.any(cloudy == nodata):
            raise ValueError("GeoTIFF contains nodata; crop/repair to a valid footprint before inference")
        scale = 10000.0 if np.nanmax(cloudy) > 1.5 else 1.0
        cloudy = np.clip(np.nan_to_num(cloudy / scale), 0, 1)
        return cloudy, None, None, None, None, profile
    raise ValueError("Upload a .npz or GeoTIFF (.tif/.tiff) file")


def _metrics_summary(path):
    if not path.is_file(): return {"available": False, "scenes": 0, "mean": None, "rows": []}
    with path.open(newline="", encoding="utf8") as f: rows = list(csv.DictReader(f))
    numeric = {}
    for key in rows[0] if rows else []:
        values = []
        for row in rows:
            try: values.append(float(row[key]))
            except (ValueError, TypeError): pass
        if values: numeric[key] = sum(values) / len(values)
    keys = ("model_full_psnr", "model_full_ssim", "model_cloud_psnr", "model_cloud_mae", "model_cloud_sam")
    mean = {k.removeprefix("model_"): numeric[k] for k in keys if k in numeric}
    return {"available": bool(rows), "scenes": len(rows), "mean": mean or None, "rows": rows[:100]}


def _manifest_summary(path):
    split = {k: 0 for k in ("train", "validation", "test", "reference")}
    if not path.is_file(): return {"manifest": False, "manifest_path": "dataset/manifest.csv", "total": 0, **split}
    try:
        with path.open(newline="", encoding="utf8") as f:
            for row in csv.DictReader(f):
                if row.get("split") in split: split[row["split"]] += 1
        return {"manifest": True, "manifest_path": "dataset/manifest.csv", "total": sum(split.values()), **split}
    except Exception as exc:
        return {"manifest": False, "manifest_path": f"manifest parse error: {exc}", "total": 0, **split}


def _safe_results_file(name):
    name = Path(name).name
    for root in (RESULTS, ROOT / "checkpoints"):
        p = (root / name).resolve()
        if p.parent == root.resolve() and p.is_file(): return p
    raise FileNotFoundError(name)


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TerraTwin/1.0"

        def send_bytes(self, body, content_type, status=200):
            self.send_response(status); self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body))); self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

        def send_json(self, data, status=200):
            self.send_bytes(json.dumps(data, allow_nan=False, default=str).encode(), "application/json; charset=utf-8", status)

        def read_body(self, limit=MAX_UPLOAD_BYTES):
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1: raise ValueError("Request body is empty")
            if length > limit: raise ValueError(f"Upload exceeds the {limit // (1024 * 1024)} MiB limit")
            return self.rfile.read(length)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/": return self.send_bytes(PAGE.encode(), "text/html; charset=utf-8")
            if parsed.path == "/api/status":
                cp = state["checkpoint_path"]
                model = state["model"]
                ck_exists = cp.is_file()
                training = state.get("job")
                data = _manifest_summary(ROOT / "dataset" / "manifest.csv")
                results = _metrics_summary(RESULTS / "test_metrics.csv")
                if ck_exists and model is None:
                    try:
                        model, cfg = load_model(cp, state["device"])
                        state.update(model=model, config=cfg, features=cfg.get("features"), parameters=sum(p.numel() for p in model.parameters()))
                    except Exception as exc: state["load_error"] = str(exc)
                model = state["model"]
                result = {"model": {"exists": ck_exists, "loaded": model is not None, "load_error": state.get("load_error"), "name": cp.name if ck_exists else None,
                                    "path": str(cp.relative_to(ROOT)) if cp.is_relative_to(ROOT) else str(cp),
                                    "bands": state.get("config", {}).get("bands") if model is not None else None,
                                    "parameters": state.get("parameters"), "device": state["device"]},
                          "data": data, "results": results,
                          "demo_available": (ROOT.parent / "cp1" / "dataset" / "demo_sample" / "sen12mscr_heldout_chip.npz").is_file() and (ROOT.parent / "cp1" / "checkpoints" / "clear-net-best.pt").is_file(),
                          "job": {"mode": training["mode"], "status": _job_status(training)} if training else None}
                return self.send_json(result)
            if parsed.path == "/api/results":
                reports = sorted([p.name for p in RESULTS.glob("*.csv") if p.is_file()], key=lambda s: (RESULTS / s).stat().st_mtime, reverse=True)
                recent = sorted([p.name for p in RESULTS.glob("*.png") if p.is_file()], key=lambda s: (RESULTS / s).stat().st_mtime, reverse=True)
                return self.send_json({"reports": reports[:25], "recent": recent[:12]})
            if parsed.path == "/api/docs":
                name = parse_qs(parsed.query).get("name", [""])[0]
                p = (ROOT / name).resolve()
                if p != ROOT and ROOT in p.parents and p.is_file() and p.suffix.lower() in (".md", ".csv"):
                    return self.send_bytes(p.read_bytes(), "text/markdown; charset=utf-8" if p.suffix.lower()==".md" else "text/csv; charset=utf-8")
                return self.send_json({"error": "Document not found"}, 404)
            if parsed.path == "/api/file":
                name = parse_qs(parsed.query).get("name", [""])[0]
                try:
                    p = _safe_results_file(name)
                    if p.suffix.lower() == ".csv":
                        with p.open(newline="", encoding="utf8") as f: return self.send_json(list(csv.DictReader(f)))
                    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                    return self.send_bytes(p.read_bytes(), ctype)
                except FileNotFoundError: return self.send_json({"error": "File not found"}, 404)
            if parsed.path == "/api/jobs":
                job = state.get("job")
                return self.send_json({"job": _public_job(job) if job else None})
            if parsed.path == "/api/job-log":
                job = state.get("job")
                if not job: return self.send_json({"log": ""})
                return self.send_json({"log": _tail(job["log_path"])})
            return self.send_json({"error": "Not found"}, 404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/infer": return self.handle_infer()
            if parsed.path == "/api/demo": return self.handle_demo()
            if parsed.path == "/api/jobs": return self.handle_job()
            return self.send_json({"error": "Not found"}, 404)

        def handle_infer(self):
            if state["model"] is None: return self.send_json({"error": "No research checkpoint is loaded. Train a model, then restart the dashboard with --checkpoint checkpoints/best.pt."}, 503)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_UPLOAD_BYTES: raise ValueError(f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB")
                raw = self.rfile.read(length)
                msg = BytesParser(policy=default).parsebytes((f'Content-Type: {self.headers["Content-Type"]}\r\nMIME-Version: 1.0\r\n\r\n').encode() + raw)
                file_name = "scene.npz"; file_data = None; auto_mask = True
                for part in msg.iter_parts():
                    if part.get_param("name", header="content-disposition") == "scene":
                        file_name = part.get_filename() or file_name; file_data = part.get_payload(decode=True)
                    if part.get_param("name", header="content-disposition") == "auto_mask":
                        auto_mask = part.get_payload(decode=True).decode(errors="ignore") == "1"
                if file_data is None: raise ValueError("Choose a scene file")
                cloudy, mask, hist, hm, clear, geo_profile = _read_scene(file_name, file_data, auto_mask)
                if cloudy.shape[0] != int(state["config"]["bands"]):
                    raise ValueError(f"This checkpoint expects {state['config']['bands']} bands; uploaded scene has {cloudy.shape[0]}")
                if np.nanmax(cloudy) > 1.5 or np.nanmin(cloudy) < -0.01:
                    raise ValueError("Input reflectance must be normalized to [0,1]")
                if mask is None:
                    if not auto_mask: raise ValueError("Scene has no mask. Enable mask estimation or upload an NPZ containing mask")
                    mask = estimate_mask(cloudy)
                if mask.shape != cloudy.shape[-2:] or not np.isin(mask, [0, 1, 2]).all(): raise ValueError("Mask must match image size and use labels 0/1/2")
                if hist is not None:
                    if hist.ndim != 4 or hist.shape[1:] != cloudy.shape: raise ValueError("history must have shape (T,bands,height,width), aligned to cloudy")
                    if hm is not None and hm.shape != (hist.shape[0], *cloudy.shape[-2:]): raise ValueError("history_masks dimensions do not match history")
                model = state["model"]; device = state["device"]
                t0 = time.perf_counter()
                prediction, confidence = reconstruct(model, cloudy, mask, hist, hm, device, state["features"])
                if str(device).startswith("cuda"):
                    import torch; torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                temporal = temporal_replacement(cloudy, mask, hist, hm)
                spatial = spatial_inpainting(cloudy, mask)
                reference = build_twin(hist, hm)["median"] if hist is not None and len(hist) else None
                metrics = calculate_metrics(prediction, clear, mask) if clear is not None else None
                if clear is not None:
                    temporal_metrics = calculate_metrics(temporal, clear, mask)
                    spatial_metrics = calculate_metrics(spatial, clear, mask)
                    metrics["temporal_full_psnr"] = temporal_metrics["full_psnr"]
                    metrics["temporal_cloud_psnr"] = temporal_metrics["cloud_psnr"]
                    metrics["inpaint_full_psnr"] = spatial_metrics["full_psnr"]
                    metrics["inpaint_cloud_psnr"] = spatial_metrics["cloud_psnr"]
                previews = [
                    {"label": "Cloudy input", "image": _b64_image(cloudy)},
                    {"label": "Cloud / shadow mask", "image": _b64_image(mask, "mask")},
                    {"label": "Historical Twin median", "image": _b64_image(reference) if reference is not None else None},
                    {"label": "Temporal replacement baseline", "image": _b64_image(temporal)},
                    {"label": "Spatial inpainting baseline", "image": _b64_image(spatial)},
                    {"label": "AI reconstruction", "image": _b64_image(prediction)},
                    {"label": "Confidence / clear-history support", "image": _b64_image(confidence, "gray")},
                ]
                if clear is not None:
                    previews.insert(0, {"label": "Clear reference (scoring only)", "image": _b64_image(clear)})
                    previews.append({"label": "Absolute error (mean bands)", "image": _b64_image(np.mean(np.abs(prediction-clear), axis=0)[None], "gray")})
                out = io.BytesIO(); np.savez_compressed(out, reconstruction=prediction, mask=mask, confidence=confidence)
                run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
                RESULTS.mkdir(parents=True, exist_ok=True)
                (RESULTS / f"dashboard_{run_id}.npz").write_bytes(out.getvalue())
                geotiff_b64 = None
                if geo_profile is not None:
                    try:
                        import rasterio
                        from rasterio.io import MemoryFile
                        geo_profile.update(count=prediction.shape[0], height=prediction.shape[1], width=prediction.shape[2], dtype="float32", nodata=None)
                        with MemoryFile() as mem:
                            with mem.open(**geo_profile) as dst: dst.write(prediction.astype(np.float32))
                            geo_bytes = mem.read()
                        geotiff_b64 = base64.b64encode(geo_bytes).decode("ascii")
                        (RESULTS / f"dashboard_{run_id}.tif").write_bytes(geo_bytes)
                    except ImportError: pass
                preview_bytes = base64.b64decode(next(p["image"] for p in previews if p["label"] == "AI reconstruction"))
                (RESULTS / f"dashboard_{run_id}.png").write_bytes(preview_bytes)
                report = {"run_id": run_id, "scene": Path(file_name).name, "width": int(cloudy.shape[2]), "height": int(cloudy.shape[1]),
                          "bands": int(cloudy.shape[0]), "cloud_percent": cloud_coverage(mask), "inference_seconds": elapsed, "metrics": metrics}
                (RESULTS / f"dashboard_{run_id}.json").write_text(json.dumps(report, indent=2, allow_nan=False))
                return self.send_json({"previews": previews, "metrics": metrics, "output": base64.b64encode(out.getvalue()).decode(), "run_id": run_id,
                                       "width": cloudy.shape[2], "height": cloudy.shape[1], "bands": cloudy.shape[0],
                                       "cloud_percent": cloud_coverage(mask), "seconds": elapsed, "scene": Path(file_name).name, "geotiff": geotiff_b64})
            except Exception as exc: return self.send_json({"error": str(exc)}, 400)

        def handle_demo(self):
            try:
                import torch
                cp_root = ROOT.parent / "cp1"
                sample_path = cp_root / "dataset" / "demo_sample" / "sen12mscr_heldout_chip.npz"
                weights_path = cp_root / "checkpoints" / "clear-net-best.pt"
                if not sample_path.is_file() or not weights_path.is_file():
                    raise FileNotFoundError("Bundled SEN12MS-CR demo sample or pretrained checkpoint is missing")
                demo = state.get("demo_model")
                if demo is None:
                    sys.path.insert(0, str(cp_root))
                    from vendor.clear_net.CLEAR_Net import CLEAR_Net
                    demo = CLEAR_Net(return_decomposition=True)
                    ck = torch.load(weights_path, map_location="cpu", weights_only=False)
                    demo.load_state_dict(ck["model"], strict=True)
                    demo.to("cpu").eval()
                    state["demo_model"] = demo
                from clear_net_infer import predict_scene
                with np.load(sample_path, allow_pickle=False) as z:
                    sar = np.asarray(z["sar"], np.float32)
                    cloudy = np.asarray(z["cloudy"], np.float32)
                    target = np.asarray(z["target"], np.float32)
                    scene, patch = str(z["scene"].item()), str(z["patch"].item())
                sar_t = torch.from_numpy(sar).unsqueeze(0)
                cloudy_t = torch.from_numpy(cloudy).unsqueeze(0)
                t0 = time.perf_counter()
                prediction = predict_scene(demo, sar_t, cloudy_t, torch.device("cpu")).squeeze(0).cpu().numpy()
                elapsed = time.perf_counter() - t0
                # This benchmark stores optical bands as reflectance/2000; scale to
                # reflectance/10000 for metric calculation and previews.
                pred_norm = np.clip(prediction / 5.0, 0, 1)
                cloudy_norm = np.clip(cloudy / 5.0, 0, 1)
                target_norm = np.clip(target / 5.0, 0, 1)
                no_mask = np.zeros(cloudy.shape[-2:], np.uint8)
                metrics_all = calculate_metrics(pred_norm, target_norm, no_mask)
                metrics = {k: v for k, v in metrics_all.items() if k.startswith("full_")}
                previews = [
                    {"label": "Cloudy Sentinel-2 input", "image": _b64_image(cloudy_norm)},
                    {"label": "Cloud / shadow mask", "image": None},
                    {"label": "Historical reference / Twin", "image": None},
                    {"label": "Temporal replacement baseline", "image": None},
                    {"label": "Spatial inpainting baseline", "image": None},
                    {"label": "CLEAR-Net SAR-guided reconstruction", "image": _b64_image(pred_norm)},
                    {"label": "Clear reference (held-out scoring target)", "image": _b64_image(target_norm)},
                    {"label": "Absolute error (mean bands)", "image": _b64_image(np.mean(np.abs(pred_norm-target_norm),axis=0)[None], "gray")},
                ]
                info = {"source": "SEN12MS-CR v2 public test split", "scene": scene, "patch": patch,
                        "chip_size_px": list(cloudy.shape[-2:]), "method": "Pretrained CLEAR-Net, Sentinel-1 SAR + 13 Sentinel-2 bands",
                        "scope": "One selected held-out chip; exploratory demo, not a benchmark result or Digital Twin experiment.",
                        "inference_seconds": elapsed, "metric_scale": "reflectance normalized by 10000"}
                info_path = cp_root / "outputs" / "demo_sen12mscr_metrics.json"
                if info_path.is_file():
                    try: info["existing_demo_record"] = json.loads(info_path.read_text())
                    except Exception: pass
                output = io.BytesIO(); np.savez_compressed(output, prediction_reflectance=prediction, target=target, cloudy=cloudy)
                return self.send_json({"previews": previews, "metrics": metrics, "output": base64.b64encode(output.getvalue()).decode(),
                                       "width": cloudy.shape[2], "height": cloudy.shape[1], "bands": cloudy.shape[0],
                                       "cloud_percent": None, "seconds": elapsed, "scene": f"SEN12MS-CR test scene {scene}, patch {patch}", "demo_info": info})
            except Exception as exc: return self.send_json({"error": str(exc)}, 500)

        def handle_job(self):
            try:
                body = json.loads(self.read_body(1024 * 16)); mode = body.get("mode")
                if mode not in ("prepare", "train", "test", "experiments"):
                    return self.send_json({"error": "Allowed jobs: prepare, train, test, experiments"}, 400)
                old = state.get("job")
                if old and _job_status(old) == "running": return self.send_json({"error": "A research job is already running", "job": _public_job(old)}, 409)
                manifest = ROOT / "dataset" / "manifest.csv"
                if not manifest.is_file(): return self.send_json({"error": "Create dataset/manifest.csv before launching research jobs"}, 400)
                job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:5]
                job_dir = RESULTS / "jobs"; job_dir.mkdir(parents=True, exist_ok=True)
                log_path = job_dir / f"{job_id}.log"
                cmd = [sys.executable, "main.py", "--mode", mode, "--manifest", "dataset/manifest.csv"]
                if mode in ("train", "test", "experiments"): cmd += ["--config", "configs/config.yaml"]
                if mode == "test":
                    ck = state["checkpoint_path"]
                    if not ck.is_file(): return self.send_json({"error": "Train the model or provide a checkpoint before evaluation"}, 400)
                    cmd += ["--checkpoint", str(ck)]
                if mode == "prepare": cmd = [sys.executable, "main.py", "--mode", "prepare", "--manifest", "dataset/manifest.csv"]
                with log_path.open("wb") as log:
                    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state["job"] = {"id": job_id, "mode": mode, "process": proc, "log_path": log_path,
                                "started": datetime.now(timezone.utc).isoformat(), "command": cmd}
                return self.send_json(_public_job(state["job"]), 202)
            except Exception as exc: return self.send_json({"error": str(exc)}, 400)

        def log_message(self, fmt, *args): pass
    return Handler


def _job_status(job):
    rc = job["process"].poll()
    return "running" if rc is None else ("completed" if rc == 0 else "failed")


def _tail(path, limit=12000):
    try: return path.read_text(errors="replace")[-limit:]
    except OSError: return ""


def _public_job(job):
    return {"id": job["id"], "mode": job["mode"], "status": _job_status(job),
            "exit_code": job["process"].poll(), "started": job["started"], "log": _tail(job["log_path"])}


def main():
    ap = argparse.ArgumentParser(description="TerraTwin local cloud-removal research dashboard")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt", help="Research checkpoint; dashboard still opens if missing")
    ap.add_argument("--host", default="127.0.0.1", help="Bind to loopback by default")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-upload-mb", type=int, default=400)
    args = ap.parse_args()
    global MAX_UPLOAD_BYTES
    MAX_UPLOAD_BYTES = args.max_upload_mb * 1024 * 1024
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute(): checkpoint = (ROOT / checkpoint).resolve()
    state = {"checkpoint_path": checkpoint, "model": None, "config": {}, "features": None,
             "parameters": None, "device": args.device, "job": None, "demo_model": None}
    if checkpoint.is_file():
        try:
            model, cfg = load_model(checkpoint, args.device)
            state.update(model=model, config=cfg, features=cfg.get("features"),
                         parameters=sum(p.numel() for p in model.parameters()))
            print(f"Loaded checkpoint {checkpoint} on {args.device}")
        except Exception as exc: print(f"Checkpoint exists but could not load: {exc}")
    else: print(f"No checkpoint at {checkpoint}; dashboard will open in data/setup mode")
    server = DashboardServer((args.host, args.port), make_handler(state))
    print(f"TerraTwin dashboard: http://{args.host}:{args.port}")
    try: server.serve_forever()
    except KeyboardInterrupt: print("\nDashboard stopped")
    finally: server.server_close()


if __name__ == "__main__": main()
