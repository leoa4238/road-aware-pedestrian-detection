"""Build a local, offline labeling page for risky-pedestrian labels (docs/RISK_LABEL_CRITERIA_DRAFT.md).

Writes <output>/index.html plus context, crop and mask-overlay images. Open index.html in a
browser on this PC; nothing is uploaded (AI-Hub images must not be redistributed).
Labels autosave in the browser and are exported as CSV with the "CSV 저장" button.

Example:
    python aihub/make_label_set.py --view data/views/session_v1 --output data/labeling/pilot_v0 --n 400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from error_analysis_core import read_image, write_image

RISK_SCENARIOS = {"A04", "A13", "B02", "B08", "B09", "B11", "B12"}
MASK_COLORS = {1: (255, 140, 0), 2: (60, 200, 60)}       # BGR: road = blue, sidewalk = green


def sample(view, n, min_height, per_clip, risk_share, seed):
    images = pd.read_csv(view / "images.csv")
    road = pd.read_csv(view / "objects_road.csv")
    peds = road[(road["cls"] == "pedestrian") & (road["h"] >= min_height) & (road["split"] != "test")]
    peds = peds.merge(images[["image", "clip", "scenario", "road", "time", "weather"]], on="image")
    rng = np.random.default_rng(seed)
    peds = peds.iloc[rng.permutation(len(peds))]
    peds = peds.groupby("clip", sort=False).head(per_clip)
    risky = peds[peds["scenario"].isin(RISK_SCENARIOS)]
    other = peds[~peds["scenario"].isin(RISK_SCENARIOS)]
    take = min(len(risky), round(n * risk_share))
    chosen = pd.concat([risky.head(take), other.head(n - take)])
    return chosen.iloc[rng.permutation(len(chosen))].reset_index(drop=True)


def render(view, row, out, item_id):
    frame = read_image(view / "images" / row.split / (row.image + ".jpg"))
    if frame is None:
        raise FileNotFoundError(row.image)
    s = frame.shape[1] / row.img_w
    x0, y0, x1, y1 = (int(round(v * s)) for v in (row.x, row.y, row.x + row.w, row.y + row.h))
    ctx = frame.copy()
    cv2.rectangle(ctx, (x0, y0), (x1, y1), (0, 255, 255), 3)
    write_image(out / "ctx" / f"{item_id}.jpg", ctx)
    mask = read_image(view / "road_masks" / row.split / (row.image + ".png"), cv2.IMREAD_GRAYSCALE)
    over = frame.copy()
    if mask is not None:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        for value, color in MASK_COLORS.items():
            sel = mask == value
            over[sel] = (0.55 * over[sel] + 0.45 * np.array(color)).astype(np.uint8)
    cv2.rectangle(over, (x0, y0), (x1, y1), (0, 255, 255), 3)
    write_image(out / "mask" / f"{item_id}.jpg", over)
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, 0.8 * max(x1 - x0, y1 - y0)
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(frame.shape[1], cx + half)), int(min(frame.shape[0], cy + half))
    crop = frame[b:d, a:c]
    crop = cv2.resize(crop, (round(crop.shape[1] * 360 / crop.shape[0]), 360), interpolation=cv2.INTER_CUBIC)
    write_image(out / "crop" / f"{item_id}.jpg", crop)


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>위험 보행자 라벨링 __NAME__</title>
<style>
body{margin:0;font-family:system-ui,'Malgun Gothic',sans-serif;background:#1e1f22;color:#e8e8e8}
header{display:flex;gap:16px;align-items:center;padding:8px 16px;background:#2b2d31;flex-wrap:wrap}
main{display:flex;gap:16px;padding:12px 16px;align-items:flex-start}
#ctxwrap{position:relative;flex:3}#ctx{width:100%;display:block;border-radius:6px}
#guide{position:absolute;left:50%;bottom:0;height:45%;border-left:2px dashed rgba(255,255,255,.55);display:none}
#side{flex:1;min-width:260px}#crop{max-width:100%;max-height:360px;display:block;margin-bottom:12px;border-radius:6px}
.lab{display:inline-block;padding:6px 12px;margin:3px;border-radius:6px;background:#3a3c42;cursor:pointer}
.on{outline:3px solid #ffd400}.k3{background:#8b1f1f}.k2{background:#8a6a12}.k1{background:#1f6b3a}.k0{background:#555}
#note{width:100%;box-sizing:border-box;margin-top:8px;padding:6px;background:#2b2d31;color:#eee;border:1px solid #555;border-radius:4px}
small{color:#aaa}button{padding:6px 10px}
</style></head><body>
<header><b id="pos"></b><span id="done"></span>
<button id="exp">CSV 저장</button><label>CSV 불러오기 <input type="file" id="imp" accept=".csv"></label>
<small>3 위험 · 2 주의 · 1 안전 · 0 제외 · ←/→ 이동 · V 마스크 보기(라벨 후) · M 마스크 오류 · G 중앙선 · N 메모</small></header>
<main><div id="ctxwrap"><img id="ctx"><div id="guide"></div></div>
<div id="side"><img id="crop">
<div><span class="lab k3" data-v="3">3 위험</span><span class="lab k2" data-v="2">2 주의</span><span class="lab k1" data-v="1">1 안전</span><span class="lab k0" data-v="0">0 제외</span></div>
<p id="mflag"></p><input id="note" placeholder="메모 (N으로 입력, Enter로 저장)"><p><small id="meta"></small></p>
<p><small>노란 박스가 판단할 보행자입니다. 주행 경로는 이미지 하단 중앙에서 앞으로 뻗는 차 한 대 폭입니다.
시선은 보지 않고, 도로 마스크는 라벨을 고른 뒤에만 볼 수 있습니다.</small></p></div></main>
<script>
const ITEMS=__ITEMS__, KEY='risklabels___NAME__';
let S={};try{S=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
let i=0,showMask=false;
const $=id=>document.getElementById(id);
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}}
function firstOpen(){const k=ITEMS.findIndex(t=>!(S[t.id]&&S[t.id].label!=null));return k<0?0:k}
function draw(){const t=ITEMS[i],s=S[t.id]||{};
 if(s.label==null)showMask=false;
 $('ctx').src=(showMask?'mask/':'ctx/')+t.id+'.jpg';$('crop').src='crop/'+t.id+'.jpg';
 $('pos').textContent=(i+1)+' / '+ITEMS.length;
 $('done').textContent='완료 '+ITEMS.filter(x=>S[x.id]&&S[x.id].label!=null).length;
 document.querySelectorAll('.lab').forEach(e=>e.classList.toggle('on',String(s.label)===e.dataset.v));
 $('mflag').textContent=(s.mask_error?'마스크 오류 표시됨':'')+(showMask?'  (마스크 보는 중: 파랑=도로, 초록=보도)':'');
 $('note').value=s.note||'';$('meta').textContent=t.id;}
function setLabel(v){const t=ITEMS[i];S[t.id]=Object.assign(S[t.id]||{},{label:v});save();
 if(i<ITEMS.length-1)i++;draw();}
document.querySelectorAll('.lab').forEach(e=>e.onclick=()=>setLabel(+e.dataset.v));
$('note').addEventListener('keydown',ev=>{if(ev.key==='Enter'){const t=ITEMS[i];
 S[t.id]=Object.assign(S[t.id]||{},{note:$('note').value});save();$('note').blur();}});
document.addEventListener('keydown',ev=>{if(document.activeElement===$('note'))return;
 const t=ITEMS[i],s=S[t.id]||{};
 if('0123'.includes(ev.key))setLabel(+ev.key);
 else if(ev.key==='ArrowRight'){if(i<ITEMS.length-1)i++;draw();}
 else if(ev.key==='ArrowLeft'){if(i>0)i--;draw();}
 else if(ev.key==='v'||ev.key==='V'){if(s.label!=null){showMask=!showMask;draw();}}
 else if(ev.key==='m'||ev.key==='M'){if(s.label!=null){S[t.id]=Object.assign(s,{mask_error:!s.mask_error});save();draw();}}
 else if(ev.key==='g'||ev.key==='G'){const g=$('guide');g.style.display=g.style.display==='block'?'none':'block';}
 else if(ev.key==='n'||ev.key==='N'){ev.preventDefault();$('note').focus();}});
$('exp').onclick=()=>{const rows=['id,label,mask_error,note'];ITEMS.forEach(t=>{const s=S[t.id]||{};
 rows.push([t.id,s.label??'',s.mask_error?1:0,JSON.stringify(s.note||'')].join(','));});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['﻿'+rows.join('\n')],{type:'text/csv'}));
 a.download='labels___NAME__.csv';a.click();};
$('imp').onchange=async ev=>{const f=ev.target.files[0];if(!f)return;
 const lines=(await f.text()).replace(/^﻿/,'').split(/\r?\n/).slice(1);
 lines.forEach(l=>{if(!l.trim())return;const m=l.match(/^([^,]+),([^,]*),([^,]*),(.*)$/);if(!m)return;
  S[m[1]]={label:m[2]===''?null:+m[2],mask_error:m[3]==='1',note:m[4]?JSON.parse(m[4]):''};});
 save();i=firstOpen();draw();};
i=firstOpen();draw();
</script></body></html>"""


ATTR_PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>보행자 조건 라벨링 __NAME__</title>
<style>
body{margin:0;font-family:system-ui,'Malgun Gothic',sans-serif;background:#1e1f22;color:#e8e8e8}
header{display:flex;gap:16px;align-items:center;padding:8px 16px;background:#2b2d31;flex-wrap:wrap}
main{display:flex;gap:16px;padding:12px 16px;align-items:flex-start}
#ctxwrap{position:relative;flex:3}#ctx{width:100%;display:block;border-radius:6px}
#guide{position:absolute;left:0;top:0;width:100%;height:100%;display:none;pointer-events:none}
#side{flex:1;min-width:300px}#crop{max-width:100%;max-height:320px;display:block;margin-bottom:8px;border-radius:6px}
h4{margin:10px 0 4px}.opt{display:inline-block;padding:5px 9px;margin:2px;border-radius:6px;background:#3a3c42;cursor:pointer;font-size:14px}
.on{background:#2f6fd6;outline:2px solid #ffd400}.tog.on{background:#a0521d}
#note{width:100%;box-sizing:border-box;margin-top:8px;padding:6px;background:#2b2d31;color:#eee;border:1px solid #555;border-radius:4px}
small{color:#aaa}button{padding:6px 10px}#warn{color:#ff8a65;min-height:1.2em}
</style></head><body>
<header><b id="pos"></b><span id="done"></span>
<button id="exp">CSV 저장</button><label>CSV 불러오기 <input type="file" id="imp" accept=".csv"></label>
<label>AI 초안 불러오기(검수용) <input type="file" id="impd" accept=".csv"></label>
<small>A 위치 제안 켜기/끄기 ·
위치 1-4 · 방향 Q W E R · C 횡단보도 · O 가림 · B 보도·차도 경계 · X 제외 · Space 다음 · ←/→ 이동 · V 마스크(위치 선택 후) · M 마스크 오류 · G 경로 띠 · N 메모</small></header>
<main><div id="ctxwrap"><img id="ctx"><svg id="guide" viewBox="0 0 100 100" preserveAspectRatio="none"><polygon points="32,100 68,100 50.6,38 49.4,38" fill="rgba(255,212,0,.13)" stroke="rgba(255,212,0,.85)" stroke-width=".25" stroke-dasharray="1 .8"/></svg></div>
<div id="side"><img id="crop">
<h4>① 주행 경로와의 위치</h4>
<span class="opt" data-f="position" data-v="1">1 경로 안</span><span class="opt" data-f="position" data-v="2">2 경로 가장자리</span>
<span class="opt" data-f="position" data-v="3">3 차도, 경로 밖</span><span class="opt" data-f="position" data-v="4">4 보행 공간</span>
<h4>② 움직이는 방향</h4>
<span class="opt" data-f="direction" data-v="q">Q 경로로 들어옴</span><span class="opt" data-f="direction" data-v="w">W 나란히</span>
<span class="opt" data-f="direction" data-v="e">E 멀어짐</span><span class="opt" data-f="direction" data-v="r">R 서 있음/모름</span>
<h4>③ ④ ⑤ 해당하면 켜기</h4>
<span class="opt tog" data-t="crosswalk">C 횡단보도 위</span><span class="opt tog" data-t="occluded">O 가림, 차 사이</span><span class="opt tog" data-t="boundary">B 보도·차도 경계에 걸침</span>
<span class="opt tog" data-t="excluded">X 제외(판단 불가)</span>
<p id="sugline"></p><p id="warn"></p><p id="mflag"></p><input id="note" placeholder="메모 (N으로 입력, Enter로 저장)"><p><small id="meta"></small></p>
<p><small>노란 박스가 대상입니다. 주행 경로 = 이미지 하단 중앙에서 앞으로 뻗는 차 한 대 폭. G로 대략적인 경로 띠를 켤 수 있습니다(곡선 도로에서는 눈으로 보정).
위험 여부는 고르지 않습니다. 보이는 사실만 고르세요. 거리는 자동으로 계산합니다.</small></p></div></main>
<script>
const ITEMS=__ITEMS__, KEY='attrlabels___NAME__';
const DIR={q:'q',w:'w',e:'e',r:'r'};
let S={};try{S=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
let SUG=localStorage.getItem(KEY+'_sug')!=='off';
// Labels saved before this version have no "checked" flag: a filled, non-draft entry counts as checked.
Object.values(S).forEach(s=>{if(s.checked===undefined&&!s.draft&&(s.excluded||(s.position&&s.direction)))s.checked=true;});
let i=0,showMask=false;const $=id=>document.getElementById(id);
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}}
function cur(){const t=ITEMS[i];return S[t.id]||(S[t.id]={});}
function filled(s){return !!(s.excluded||(s.position&&s.direction));}
function complete(s){return !!s.checked;}
function applySuggestion(t,s){if(SUG&&!s.checked&&!s.position&&t.sug){s.position=t.sug;s.auto=true;}}
function firstOpen(){const k=ITEMS.findIndex(t=>!complete(S[t.id]||{}));return k<0?0:k}
function draw(){const t=ITEMS[i],s=cur();applySuggestion(t,s);
 if(!s.position)showMask=false;
 $('ctx').src=(showMask?'mask/':'ctx/')+t.id+'.jpg';$('crop').src='crop/'+t.id+'.jpg';
 $('pos').textContent=(i+1)+' / '+ITEMS.length;
 $('done').textContent='완료 '+ITEMS.filter(x=>complete(S[x.id]||{})).length;
 document.querySelectorAll('.opt[data-f]').forEach(e=>e.classList.toggle('on',String(s[e.dataset.f])===e.dataset.v));
 document.querySelectorAll('.tog').forEach(e=>e.classList.toggle('on',!!s[e.dataset.t]));
 $('mflag').textContent=(s.mask_error?'마스크 오류 표시됨':'')+(showMask?'  (마스크: 파랑=도로, 초록=보도)':'');
 $('note').value=s.note||'';$('meta').textContent=t.id;$('warn').textContent='';
 const tags=[];if(s.draft&&!s.checked)tags.push('AI 초안: 검수 필요');
 if(s.auto&&!s.checked)tags.push('위치 자동 제안 '+t.sug+' (신뢰도 '+Math.round(100*t.conf)+'%) · '+t.why+' · 맞으면 방향만 고르고 Space');
 if(!SUG)tags.push('위치 제안 꺼짐 (A)');if(s.checked)tags.push('확인됨');
 $('sugline').textContent=tags.join(' | ');}
function setF(f,v){const s=cur();s[f]=v;if(f==='position')s.auto=false;s.checked=false;save();draw();}
function tog(f){const s=cur();s[f]=!s[f];s.checked=false;save();draw();}
function next(){const t=ITEMS[i],s=cur();
 if(!filled(s)){$('warn').textContent='위치와 방향을 고르거나, 제외(X)를 켜세요.';return;}
 if(!s.checked){s.checked=true;s.pos_suggested=t.sug||'';s.sug_accepted=!!(s.auto&&t.sug&&s.position===t.sug);save();}
 const k=ITEMS.findIndex((x,j)=>j>i&&!complete(S[x.id]||{}));i=k<0?Math.min(i+1,ITEMS.length-1):k;draw();}
document.querySelectorAll('.opt[data-f]').forEach(e=>e.onclick=()=>setF(e.dataset.f,e.dataset.v));
document.querySelectorAll('.tog').forEach(e=>e.onclick=()=>tog(e.dataset.t));
$('note').addEventListener('keydown',ev=>{if(ev.key==='Enter'){cur().note=$('note').value;save();$('note').blur();}});
document.addEventListener('keydown',ev=>{if(document.activeElement===$('note'))return;
 const k=ev.key.toLowerCase(),s=cur();
 if('1234'.includes(k)&&k.length===1)setF('position',k);
 else if(DIR[k])setF('direction',k);
 else if(k==='c')tog('crosswalk');else if(k==='o')tog('occluded');else if(k==='b')tog('boundary');else if(k==='x')tog('excluded');
 else if(k===' '){ev.preventDefault();next();}
 else if(ev.key==='ArrowRight'){if(i<ITEMS.length-1)i++;draw();}
 else if(ev.key==='ArrowLeft'){if(i>0)i--;draw();}
 else if(k==='v'){if(s.position){showMask=!showMask;draw();}}
 else if(k==='m'){if(s.position){s.mask_error=!s.mask_error;save();draw();}}
 else if(k==='a'){SUG=!SUG;localStorage.setItem(KEY+'_sug',SUG?'on':'off');
  if(!SUG&&s.auto&&!s.checked){delete s.position;s.auto=false;save();}draw();}
 else if(k==='g'){const g=$('guide');g.style.display=g.style.display==='block'?'none':'block';}
 else if(k==='n'){ev.preventDefault();$('note').focus();}});
const COLS=['position','direction','crosswalk','occluded','boundary','excluded','mask_error','checked','from_draft','sug_accepted','pos_suggested'],
 FLAGS=['crosswalk','occluded','boundary','excluded','mask_error','checked'];
function csvRows(){const rows=['id,'+COLS.join(',')+',note'];ITEMS.forEach(t=>{const s=S[t.id]||{};
 // unchecked entries are exported empty, so an unreviewed suggestion or draft never becomes a label
 const ok=!!s.checked;
 rows.push([t.id,ok?(s.position||''):'',ok?(s.direction||''):'',...FLAGS.map(f=>ok&&s[f]?1:0),s.draft?1:0,ok&&s.sug_accepted?1:0,
  ok?(s.pos_suggested||''):'',JSON.stringify(s.note||'')].join(','));});return rows;}
$('exp').onclick=()=>{const rows=csvRows();
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['﻿'+rows.join('\n')],{type:'text/csv'}));
 a.download='attributes___NAME__.csv';a.click();};
function parseCsv(text){const lines=text.replace(/^\ufeff/,'').split(/\r?\n/);const hdr=lines[0].split(',');const out=[];
 lines.slice(1).forEach(l=>{if(!l.trim())return;const parts=l.split(',');const n=hdr.length-1;
  const v=parts.slice(0,n).concat([parts.slice(n).join(',')]);const s={};
  hdr.forEach((h,j)=>{const x=v[j];if(h==='id')return;
   if(h==='note')s.note=x?JSON.parse(x):'';else if(h==='position'||h==='direction'||h==='pos_suggested')s[h]=x||undefined;
   else s[h]=x==='1';});
  out.push([v[0],s,hdr.includes('checked')]);});return out;}
// A labels CSV: rows count as checked (older files have no "checked" column and were all reviewed).
function importLabels(text){parseCsv(text).forEach(([id,s,hasChecked])=>{if(!hasChecked)s.checked=filled(s);
 s.draft=!!s.from_draft;delete s.from_draft;S[id]=s;});save();i=firstOpen();draw();}
// The AI draft: values are loaded for review, never as checked labels. An entry already checked keeps
// the reviewer's values, unless it equals the draft exactly (likely the draft loaded with the old button).
function importDraft(text){let n=0;parseCsv(text).forEach(([id,d])=>{const s=S[id];
  const same=s&&['position','direction','crosswalk','occluded','boundary','excluded'].every(k=>(s[k]||false)==(d[k]||false));
  if(s&&s.checked&&!same)return;
  S[id]=Object.assign(d,{draft:true,checked:false,auto:false});n++;});
 save();i=firstOpen();draw();return n;}
$('imp').onchange=async ev=>{const f=ev.target.files[0];if(f)importLabels(await f.text());};
$('impd').onchange=async ev=>{const f=ev.target.files[0];if(f)alert('AI 초안 '+importDraft(await f.text())+'명을 검수 대상으로 불러왔습니다.');};
i=firstOpen();draw();
</script></body></html>"""


# Approximate ego-path band, the same one the labeling page draws with G (fractions of width/height).
BAND_BOTTOM, BAND_TOP, BAND_Y_TOP = (.32, .68), (.494, .506), .38
IN_PATH, EDGE = 0.85, 1.35          # |lateral| in half-band-widths: <= IN_PATH -> 1, <= EDGE -> 2, else 3


def lateral_in_band(fx, fy):
    """Signed foot offset from the band centre in half-band-widths (|x| <= 1 = inside the band)."""
    t = np.clip((1 - fy) / (1 - BAND_Y_TOP), 0, 1)
    left = BAND_BOTTOM[0] + t * (BAND_TOP[0] - BAND_BOTTOM[0])
    right = BAND_BOTTOM[1] + t * (BAND_TOP[1] - BAND_BOTTOM[1])
    return (fx - (left + right) / 2) / ((right - left) / 2)


def suggest_position(fx, fy, h, mask):
    """Position suggestion ("1"-"4"), confidence in [0, 1] and a short reason, from numbers only.

    fx, fy, h: foot point and box height as fractions of the image. mask: road mask (0/1/2) or None.
    Uses the floor window around the feet (SegFormer paints the feet themselves as person).
    """
    if mask is None:
        return "", 0.0, "마스크 없음"
    mh, mw = mask.shape
    x, y, hh = int(np.clip(fx, 0, .999) * mw), int(np.clip(fy, 0, .999) * mh), max(int(h * mh), 4)
    win = mask[max(0, y - hh // 20): min(mh, y + hh // 10 + 1), max(0, x - hh // 4): min(mw, x + hh // 4 + 1)]
    road, side = float((win == 1).mean()), float((win == 2).mean())
    if road < 0.5:
        conf = 0.8 if road < 0.2 else 0.4
        return "4", conf, "발 주변이 도로 아님 (도로 %.0f%%, 보도 %.0f%%)" % (100 * road, 100 * side)
    lat = abs(float(lateral_in_band(fx, fy)))
    pos = "1" if lat <= IN_PATH else "2" if lat <= EDGE else "3"
    # Far from the band thresholds and a clean road window -> high confidence. The band is fixed, so
    # curved roads and narrow alleys are where suggestions 1-3 go wrong.
    margin = min(abs(lat - IN_PATH), abs(lat - EDGE)) / 0.5
    conf = min(1.0, margin) * 0.6 + min(1.0, (road - 0.5) / 0.4) * 0.4
    return pos, round(conf, 3), "도로 위, 경로 띠 기준 좌우 %.2f" % lat


def position_suggestions(items, view):
    rows = []
    for (split, image), group in items.groupby(["split", "image"], sort=False):
        mask = read_image(view / "road_masks" / split / (image + ".png"), cv2.IMREAD_GRAYSCALE)
        for r in group.itertuples():
            pos, conf, why = suggest_position(r.foot_x / r.img_w, r.foot_y / r.img_h, r.h / r.img_h, mask)
            rows.append({"id": r.id, "sug": pos, "conf": conf, "why": why})
    return pd.DataFrame(rows)


def write_attribute_page(output, view=None, suggest=True):
    """Condition-by-condition labeling page over an existing set (docs/RISK_LABEL_CRITERIA_DRAFT.md v1).

    With suggestions, each pedestrian gets a pre-filled position computed from numbers only (foot point,
    box size, road mask values; no image pixels), and items are ordered from the clearest to the most
    ambiguous suggestion. Whether the labeler kept the suggestion is exported (sug_accepted) so that
    anchoring on the suggestion can be measured.
    """
    items = pd.read_csv(output / "items.csv")
    if suggest and view is not None:
        sug = position_suggestions(items, view).sort_values("conf", ascending=False, kind="stable")
        payload = sug.to_dict("records")
        counts = sug["sug"].replace("", "none").value_counts().to_dict()
        print("position suggestions:", counts, "| confidence >= 0.6: %d / %d" % ((sug["conf"] >= .6).sum(), len(sug)))
        draft = output / "attributes_pilot_v0_claude_draft.csv"
        if draft.exists():
            d = pd.read_csv(draft, dtype={"position": "string"}).merge(sug, on="id")
            d = d[(d["excluded"] == 0) & d["position"].notna() & (d["sug"] != "")]
            if len(d):
                print("suggestion vs AI draft position agreement: %.2f (n=%d)" % ((d["position"] == d["sug"]).mean(), len(d)))
    else:
        payload = [{"id": i, "sug": "", "conf": 0, "why": ""} for i in items["id"]]
    page = ATTR_PAGE.replace("__NAME__", output.name).replace("__ITEMS__", json.dumps(payload, ensure_ascii=False))
    (output / "index_attr.html").write_text(page, encoding="utf-8")
    print("open:", (output / "index_attr.html").resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attributes-page", action="store_true",
                        help="only write index_attr.html for an existing set in --output")
    parser.add_argument("--view", type=Path, required=True, help="data/views/session_v1 (make_split_view.py)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--min-height", type=float, default=64, help="original-pixel bbox height")
    parser.add_argument("--per-clip", type=int, default=3)
    parser.add_argument("--risk-share", type=float, default=.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-suggest", action="store_true", help="attributes page without position suggestions")
    args = parser.parse_args()
    if args.attributes_page:
        write_attribute_page(args.output, args.view, suggest=not args.no_suggest)
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Output folder is not empty; labels may already exist there: %s" % args.output)
    chosen = sample(args.view, args.n, args.min_height, args.per_clip, args.risk_share, args.seed)
    chosen["id"] = [f"{r.image}__{int(r.obj_idx)}" for r in chosen.itertuples()]
    for row in chosen.itertuples():
        render(args.view, row, args.output, row.id)
    keep = ["id", "image", "obj_idx", "split", "clip", "scenario", "road", "time", "weather",
            "x", "y", "w", "h", "foot_x", "foot_y", "img_w", "img_h"]
    chosen[keep].to_csv(args.output / "items.csv", index=False, encoding="utf-8-sig")
    name = args.output.name
    page = PAGE.replace("__NAME__", name).replace("__ITEMS__", json.dumps([{"id": i} for i in chosen["id"]]))
    (args.output / "index.html").write_text(page, encoding="utf-8")
    print("items:", len(chosen), "clips:", chosen["clip"].nunique(),
          "risk-scenario share: %.2f" % chosen["scenario"].isin(RISK_SCENARIOS).mean())
    print("open:", (args.output / "index.html").resolve())


if __name__ == "__main__":
    main()
