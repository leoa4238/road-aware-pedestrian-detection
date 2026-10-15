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
<small>위치 1-4 · 방향 Q W E R · C 횡단보도 · O 가림 · B 보도·차도 경계 · X 제외 · Space 다음 · ←/→ 이동 · V 마스크(위치 선택 후) · M 마스크 오류 · G 경로 띠 · N 메모</small></header>
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
<p id="warn"></p><p id="mflag"></p><input id="note" placeholder="메모 (N으로 입력, Enter로 저장)"><p><small id="meta"></small></p>
<p><small>노란 박스가 대상입니다. 주행 경로 = 이미지 하단 중앙에서 앞으로 뻗는 차 한 대 폭. G로 대략적인 경로 띠를 켤 수 있습니다(곡선 도로에서는 눈으로 보정).
위험 여부는 고르지 않습니다. 보이는 사실만 고르세요. 거리는 자동으로 계산합니다.</small></p></div></main>
<script>
const ITEMS=__ITEMS__, KEY='attrlabels___NAME__';
const DIR={q:'q',w:'w',e:'e',r:'r'};
let S={};try{S=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
let i=0,showMask=false;const $=id=>document.getElementById(id);
function save(){try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){}}
function cur(){const t=ITEMS[i];return S[t.id]||(S[t.id]={});}
function complete(s){return s.excluded||(s.position&&s.direction);}
function firstOpen(){const k=ITEMS.findIndex(t=>!complete(S[t.id]||{}));return k<0?0:k}
function draw(){const t=ITEMS[i],s=S[t.id]||{};
 if(!s.position)showMask=false;
 $('ctx').src=(showMask?'mask/':'ctx/')+t.id+'.jpg';$('crop').src='crop/'+t.id+'.jpg';
 $('pos').textContent=(i+1)+' / '+ITEMS.length;
 $('done').textContent='완료 '+ITEMS.filter(x=>complete(S[x.id]||{})).length;
 document.querySelectorAll('.opt[data-f]').forEach(e=>e.classList.toggle('on',String(s[e.dataset.f])===e.dataset.v));
 document.querySelectorAll('.tog').forEach(e=>e.classList.toggle('on',!!s[e.dataset.t]));
 $('mflag').textContent=(s.mask_error?'마스크 오류 표시됨':'')+(showMask?'  (마스크: 파랑=도로, 초록=보도)':'');
 $('note').value=s.note||'';$('meta').textContent=t.id;$('warn').textContent='';}
function setF(f,v){cur()[f]=v;save();draw();}
function tog(f){const s=cur();s[f]=!s[f];save();draw();}
function next(){if(!complete(cur())){$('warn').textContent='위치와 방향을 고르거나, 제외(X)를 켜세요.';return;}
 if(i<ITEMS.length-1)i++;draw();}
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
 else if(k==='g'){const g=$('guide');g.style.display=g.style.display==='block'?'none':'block';}
 else if(k==='n'){ev.preventDefault();$('note').focus();}});
const COLS=['position','direction','crosswalk','occluded','boundary','excluded','mask_error'],FLAGS=COLS.slice(2);
$('exp').onclick=()=>{const rows=['id,'+COLS.join(',')+',note'];ITEMS.forEach(t=>{const s=S[t.id]||{};
 rows.push([t.id,s.position||'',s.direction||'',...FLAGS.map(f=>s[f]?1:0),JSON.stringify(s.note||'')].join(','));});
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob(['﻿'+rows.join('\n')],{type:'text/csv'}));
 a.download='attributes___NAME__.csv';a.click();};
$('imp').onchange=async ev=>{const f=ev.target.files[0];if(!f)return;
 const lines=(await f.text()).replace(/^\ufeff/,'').split(/\r?\n/);const hdr=lines[0].split(',');
 lines.slice(1).forEach(l=>{if(!l.trim())return;const parts=l.split(',');const n=hdr.length-1;
  const v=parts.slice(0,n).concat([parts.slice(n).join(',')]);const s={};
  hdr.forEach((h,j)=>{const x=v[j];if(h==='id')return;
   if(h==='note')s.note=x?JSON.parse(x):'';else if(h==='position'||h==='direction')s[h]=x||undefined;else s[h]=x==='1';});
  S[v[0]]=s;});
 save();i=firstOpen();draw();};
i=firstOpen();draw();
</script></body></html>"""


def write_attribute_page(output):
    """Condition-by-condition labeling page over an existing set (docs/RISK_LABEL_CRITERIA_DRAFT.md v1)."""
    items = pd.read_csv(output / "items.csv")
    page = ATTR_PAGE.replace("__NAME__", output.name).replace(
        "__ITEMS__", json.dumps([{"id": i} for i in items["id"]]))
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
    args = parser.parse_args()
    if args.attributes_page:
        write_attribute_page(args.output)
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
