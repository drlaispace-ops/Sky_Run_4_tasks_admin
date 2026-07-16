"""Build the interactive training lab from tier_d/admin/reports/runs.json.

    python -m tier_d.admin.viz.export_runs && python -m tier_d.admin.viz.build_lab

Self-contained: the run data is embedded gzip+base64 and inflated in the browser
with DecompressionStream, because the artifact CSP blocks every external host.
"""

from __future__ import annotations

import json
from pathlib import Path
from string import Template

RUNS = Path("tier_d/admin/reports/runs.json")
OUT = Path("tier_d/admin/reports/lab.html")


def build() -> None:
    payload = RUNS.read_text()
    data = json.loads(payload)
    n_maps = len(data["maps"])
    n_frames = data["meta"]["n_frames"]
    episodes = data["meta"]["episodes"]
    stride = data["meta"]["stride"]

    html = Template(BODY).safe_substitute(
        payload=payload,
        n_maps=n_maps,
        n_frames=n_frames,
        episodes=episodes,
        stride=stride,
    )
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB) · "
          f"{n_maps} maps × {n_frames} frames")


BODY = r"""<title>AI SkyRun — training lab</title>
<style>
  :root{
    color-scheme: light dark;
    --ground:#f5f7f4; --surface:#ffffff; --sunk:#eef1ec;
    --ink:#121a16; --body:#33403a; --muted:#67766e; --line:#dbe1db;
    --accent:#15756c; --pass:#2f6d4f; --stop:#a8322e; --oracle:#7048c4;
  }
  @media (prefers-color-scheme: dark){
    :root{ --ground:#0d1210; --surface:#141b18; --sunk:#101715;
      --ink:#e7ede9; --body:#b3c0b9; --muted:#7f8f86; --line:#26302b;
      --accent:#4cb9aa; --pass:#57bf85; --stop:#ef6a60; --oracle:#a78bfa; }
  }
  :root[data-theme="dark"]{ --ground:#0d1210; --surface:#141b18; --sunk:#101715;
    --ink:#e7ede9; --body:#b3c0b9; --muted:#7f8f86; --line:#26302b;
    --accent:#4cb9aa; --pass:#57bf85; --stop:#ef6a60; --oracle:#a78bfa; }
  :root[data-theme="light"]{ --ground:#f5f7f4; --surface:#ffffff; --sunk:#eef1ec;
    --ink:#121a16; --body:#33403a; --muted:#67766e; --line:#dbe1db;
    --accent:#15756c; --pass:#2f6d4f; --stop:#a8322e; --oracle:#7048c4; }

  *{box-sizing:border-box}
  body{margin:0;background:var(--ground);color:var(--body);
    font:15px/1.6 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    -webkit-font-smoothing:antialiased}
  .wrap{max-width:1120px;margin:0 auto;padding:44px 22px 80px}
  h1,h2{font-family:ui-serif,"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
    color:var(--ink);font-weight:600;letter-spacing:-.01em;text-wrap:balance}
  h1{font-size:clamp(26px,3.6vw,36px);line-height:1.15;margin:0 0 10px}
  h2{font-size:21px;margin:0 0 14px}
  .eyebrow{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:11px;
    letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 12px}
  .lede{font-size:17px;max-width:68ch;margin:0 0 28px}
  .lede b{color:var(--ink);font-weight:600}
  p{max-width:70ch}

  .board{display:grid;grid-template-columns:minmax(0,1fr) 292px;gap:22px;align-items:start}
  @media (max-width:900px){ .board{grid-template-columns:1fr} }

  .stage{background:var(--surface);border:1px solid var(--line);border-radius:12px;
    padding:16px;box-shadow:0 1px 2px rgba(0,0,0,.05),0 10px 30px -18px rgba(0,0,0,.3)}
  canvas{width:100%;height:auto;display:block;border-radius:8px;background:var(--sunk);
    image-rendering:auto}
  #cv{aspect-ratio:1/1}
  #curve{height:96px;margin-top:12px;aspect-ratio:auto}

  .panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;
    padding:16px;box-shadow:0 1px 2px rgba(0,0,0,.05),0 10px 30px -18px rgba(0,0,0,.3)}
  .panel + .panel{margin-top:16px}
  .lbl{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:10.5px;
    letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}

  .maps{display:grid;grid-template-columns:repeat(5,1fr);gap:6px}
  .maps button{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:12px;
    padding:7px 0;border:1px solid var(--line);border-radius:6px;background:var(--sunk);
    color:var(--body);cursor:pointer;transition:none}
  .maps button:hover{border-color:var(--accent)}
  .maps button[aria-pressed="true"]{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600}

  .transport{display:flex;gap:8px;align-items:center;margin-top:14px}
  .transport button{font-size:13px;padding:8px 14px;border:1px solid var(--line);
    border-radius:7px;background:var(--sunk);color:var(--ink);cursor:pointer;min-width:86px}
  .transport button:hover{border-color:var(--accent)}
  input[type=range]{width:100%;accent-color:var(--accent);margin:14px 0 2px}
  .scrub{display:flex;justify-content:space-between;font-family:ui-monospace,"SF Mono",Menlo,monospace;
    font-size:11px;color:var(--muted);font-variant-numeric:tabular-nums}

  .stats{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .stat{background:var(--sunk);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
  .stat .k{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:9.5px;
    letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
  .stat .v{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:19px;font-weight:600;
    color:var(--ink);font-variant-numeric:tabular-nums;margin-top:2px}
  .stat .v.ok{color:var(--pass)} .stat .v.bad{color:var(--stop)}

  .toggles{display:flex;flex-direction:column;gap:8px}
  .toggles label{display:flex;align-items:center;gap:9px;font-size:13px;cursor:pointer}
  .toggles input{accent-color:var(--accent);width:15px;height:15px}
  .swatch{width:16px;height:3px;border-radius:2px;flex:none}

  .legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:12px;font-size:11.5px;color:var(--muted);
    font-family:ui-monospace,"SF Mono",Menlo,monospace}
  .ramp{height:8px;border-radius:4px;margin-top:8px;
    background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)}
  .ramp-lbl{display:flex;justify-content:space-between;font-size:10.5px;color:var(--muted);
    font-family:ui-monospace,"SF Mono",Menlo,monospace;margin-top:3px}

  code{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:12.5px;background:var(--sunk);
    border:1px solid var(--line);border-radius:4px;padding:1px 5px;color:var(--ink)}
  .note{border:1px solid var(--line);border-left:3px solid var(--accent);background:var(--surface);
    border-radius:0 10px 10px 0;padding:16px 18px;margin:34px 0 0;max-width:80ch}
  .note h2{font-size:17px;margin-bottom:6px}
  .note p:last-child{margin-bottom:0}
  footer{margin-top:54px;padding-top:18px;border-top:1px solid var(--line);
    font-size:12.5px;color:var(--muted)}
  :focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:5px}
  @media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
  #boot{padding:40px;text-align:center;color:var(--muted);font-family:ui-monospace,Menlo,monospace;font-size:13px}
</style>

<div class="wrap">
  <p class="eyebrow">AI SkyRun · інтерактивна лабораторія</p>
  <h1>Як дрон вчиться — кадр за кадром</h1>
  <p class="lede">
    $n_maps карт, $episodes епізодів на кожну, знімок кожні $stride епізодів —
    <b>$n_frames кадрів на карту</b>. Перетягніть повзунок і дивіться, як росте
    поверхня цінності. Піки зʼявляються <b>лише там, де агент справді врізався</b>:
    дерево, якого він не торкався, лишається пласким.
  </p>

  <div id="boot">розпаковую дані навчання…</div>

  <div class="board" id="board" hidden>
    <div class="stage">
      <canvas id="cv" width="720" height="720"></canvas>
      <input type="range" id="scrub" min="0" max="$n_frames" value="0" step="1" aria-label="епізод">
      <div class="scrub"><span>епізод 0</span><span id="epnow">0</span><span>$episodes</span></div>
      <div class="transport">
        <button id="play">▶ Пуск</button>
        <button id="rew">⟲ На початок</button>
        <button id="end">До кінця ⇥</button>
      </div>
      <canvas id="curve" width="1200" height="190"></canvas>
      <div class="legend">
        <span style="color:var(--oracle)">— — A* оптимум</span>
        <span style="color:var(--accent)">—— політ політики</span>
        <span style="color:var(--stop)">—— політ обірвано</span>
      </div>
    </div>

    <div>
      <div class="panel">
        <p class="lbl">Карта (seed)</p>
        <div class="maps" id="maps"></div>
      </div>

      <div class="panel">
        <p class="lbl">Стан на цьому кадрі</p>
        <div class="stats">
          <div class="stat"><div class="k">епізод</div><div class="v" id="s_ep">0</div></div>
          <div class="stat"><div class="k">результат</div><div class="v" id="s_ok">—</div></div>
          <div class="stat"><div class="k">L / L*</div><div class="v" id="s_eff">—</div></div>
          <div class="stat"><div class="k">довжина L</div><div class="v" id="s_len">—</div></div>
          <div class="stat"><div class="k">колізій усього</div><div class="v" id="s_coll">0</div></div>
          <div class="stat"><div class="k">дерев вивчено</div><div class="v" id="s_learn">0</div></div>
        </div>
      </div>

      <div class="panel">
        <p class="lbl">Шари</p>
        <div class="toggles">
          <label><input type="checkbox" id="t_v" checked><span>поверхня −V(s)</span></label>
          <label><input type="checkbox" id="t_pol"><span>стрілки політики</span></label>
          <label><input type="checkbox" id="t_tree" checked><span>дерева</span></label>
          <label><input type="checkbox" id="t_astar" checked>
            <span class="swatch" style="background:var(--oracle)"></span><span>A* оптимум</span></label>
          <label><input type="checkbox" id="t_path" checked>
            <span class="swatch" style="background:var(--accent)"></span><span>шлях політики</span></label>
        </div>
        <div class="ramp"></div>
        <div class="ramp-lbl"><span>−V = −100 (ціль)</span><span>+100 (біль)</span></div>
      </div>
    </div>
  </div>

  <div class="note">
    <h2>Що саме ви бачите</h2>
    <p>
      Висота клітинки-дерева — це <b>не</b> його координата. Це найгірше значення
      <code>Q(сусід, дія-в-дерево)</code>, яке агент вивчив, врізавшись. Тому лічильник
      «дерев вивчено» росте поступово, а не стрибає одразу до максимуму. Це і є
      інваріант 2: <em>стан — це досвід, а не мапа</em>.
    </p>
    <p>
      Зверніть увагу на розрив: поверхня виглядає «готовою» задовго до того, як жадібна
      політика вперше долітає до B. Саме тому <b>sample efficiency</b> міряють по жадібній
      політиці, а не по епізодах дослідження.
    </p>
  </div>

  <footer>
    Дані згенеровано <code>python -m tier_d.admin.viz.export_runs</code> · сторінку зібрано
    <code>python -m tier_d.admin.viz.build_lab</code>. Seeds 0–9 — це демо-набір, не прихований
    (той — 101–110), тож інваріант 5 не порушено.
  </footer>
</div>

<script id="runs" type="application/json">$payload</script>
<script>
(async function(){
  const D = JSON.parse(document.getElementById('runs').textContent);
  const M = D.meta, N = M.grid;

  // --- inflate the gzip+base64 blobs ------------------------------------
  const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  async function gunzip(s){
    const st = new DecompressionStream('gzip');
    const buf = await new Response(new Blob([b64(s)]).stream().pipeThrough(st)).arrayBuffer();
    return new Uint8Array(buf);
  }
  let V, POL, PATHS;
  try{
    V   = await gunzip(D.v);
    POL = await gunzip(D.pol);
    const p = await gunzip(D.paths);
    PATHS = new Uint16Array(p.buffer, p.byteOffset, p.byteLength/2);
  }catch(e){
    document.getElementById('boot').textContent =
      'Ваш браузер не підтримує DecompressionStream. Спробуйте Chrome/Safari/Firefox новішої версії.';
    return;
  }
  document.getElementById('boot').hidden = true;
  document.getElementById('board').hidden = false;

  const NF = M.n_frames, CELLS = N*N;
  const OFF = D.path_offsets;

  // blocked bitmask per map
  D.maps.forEach(m => {
    const bytes = b64(m.blocked); const bits = new Uint8Array(CELLS);
    for(let i=0;i<CELLS;i++) bits[i] = (bytes[i>>3] >> (7-(i&7))) & 1;
    m.bits = bits;
  });

  // --- viridis ----------------------------------------------------------
  const STOPS = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  function viridis(t){
    t = Math.min(1, Math.max(0, t)); const x = t*(STOPS.length-1);
    const i = Math.min(STOPS.length-2, Math.floor(x)), f = x-i;
    const a = STOPS[i], b = STOPS[i+1];
    return [a[0]+(b[0]-a[0])*f, a[1]+(b[1]-a[1])*f, a[2]+(b[2]-a[2])*f];
  }

  // --- state ------------------------------------------------------------
  let map = 0, frame = 0, playing = false, raf = null;
  const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
  const cu = document.getElementById('curve'), cx = cu.getContext('2d');
  const T = id => document.getElementById(id);
  const S = cv.width / M.domain;
  const px = (x,y) => [x*S, cv.height - y*S];
  const cellXY = c => [ (c%N + 0.5)*M.cell, (Math.floor(c/N) + 0.5)*M.cell ];

  const vAt = (m,f,c) => V[(m*NF + f)*CELLS + c];
  const pAt = (m,f,c) => POL[(m*NF + f)*CELLS + c];
  const pathOf = (m,f) => { const k = m*NF+f; return PATHS.subarray(OFF[k], OFF[k+1]); };

  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

  function drawBoard(){
    const mp = D.maps[map], fr = mp.frames[frame];
    ctx.clearRect(0,0,cv.width,cv.height);

    // surface
    if(T('t_v').checked){
      const img = ctx.createImageData(N,N);
      for(let c=0;c<CELLS;c++){
        // stored q = (V+100)/200*255  ->  -V normalised = 1 - q/255
        const t = 1 - vAt(map,frame,c)/255;
        const [r,g,b] = viridis(t);
        const row = N-1-Math.floor(c/N), col = c%N, k = 4*(row*N+col);
        img.data[k]=r; img.data[k+1]=g; img.data[k+2]=b; img.data[k+3]=255;
      }
      const off = document.createElement('canvas'); off.width=off.height=N;
      off.getContext('2d').putImageData(img,0,0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off,0,0,cv.width,cv.height);
    } else {
      ctx.fillStyle = css('--sunk'); ctx.fillRect(0,0,cv.width,cv.height);
    }

    // policy arrows
    if(T('t_pol').checked){
      ctx.strokeStyle = 'rgba(255,255,255,.62)'; ctx.lineWidth = 1.3;
      const L = S*M.cell*0.34;
      for(let c=0;c<CELLS;c++){
        if(mp.bits[c]) continue;
        const a = M.actions[pAt(map,frame,c)];
        const [wx,wy] = cellXY(c), [sx,sy] = px(wx,wy);
        const n = Math.hypot(a[0],a[1]);
        const ex = sx + a[1]/n*L, ey = sy - a[0]/n*L;
        ctx.beginPath(); ctx.moveTo(sx-(ex-sx)*.4, sy-(ey-sy)*.4); ctx.lineTo(ex,ey); ctx.stroke();
      }
    }

    // trees
    if(T('t_tree').checked){
      for(const [tx,ty] of mp.trees){
        const [x,y] = px(tx,ty);
        ctx.beginPath(); ctx.arc(x,y,M.tree_r*S,0,6.2832);
        ctx.fillStyle='rgba(47,109,79,.85)'; ctx.fill();
        ctx.strokeStyle='rgba(255,255,255,.55)'; ctx.lineWidth=1; ctx.stroke();
      }
    }

    const line = (cells, color, w, dash) => {
      if(!cells.length) return;
      ctx.setLineDash(dash||[]); ctx.strokeStyle=color; ctx.lineWidth=w;
      ctx.lineJoin=ctx.lineCap='round'; ctx.beginPath();
      cells.forEach((c,i)=>{ const [wx,wy]=cellXY(c); const [x,y]=px(wx,wy);
        i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
      ctx.stroke(); ctx.setLineDash([]);
    };
    if(T('t_astar').checked) line(mp.astar, css('--oracle'), 3, [7,5]);
    if(T('t_path').checked)  line(Array.from(pathOf(map,frame)),
                                  fr.ok ? css('--accent') : css('--stop'), 3.4);

    // A and B
    for(const [c,col,r] of [[M.start_cell,'#1f6feb',9],[M.goal_cell,'#d1242f',11]]){
      const [wx,wy]=cellXY(c); const [x,y]=px(wx,wy);
      ctx.beginPath(); ctx.arc(x,y,r,0,6.2832); ctx.fillStyle=col; ctx.fill();
      ctx.strokeStyle='#fff'; ctx.lineWidth=2.2; ctx.stroke();
    }
  }

  function drawCurve(){
    const mp = D.maps[map];
    const W=cu.width, H=cu.height, pad=26;
    cx.clearRect(0,0,W,H);
    const Ls = mp.Lstar, top = Ls*2.1;
    const X = i => pad + (W-2*pad) * (i/(NF-1));
    const Y = v => H-pad - (H-2*pad) * ((v-Ls)/(top-Ls));

    cx.strokeStyle=css('--line'); cx.lineWidth=1;
    cx.beginPath(); cx.moveTo(pad,H-pad); cx.lineTo(W-pad,H-pad); cx.stroke();

    cx.setLineDash([6,5]); cx.strokeStyle=css('--oracle'); cx.lineWidth=1.6;
    cx.beginPath(); cx.moveTo(pad,Y(Ls)); cx.lineTo(W-pad,Y(Ls)); cx.stroke(); cx.setLineDash([]);

    cx.strokeStyle=css('--accent'); cx.lineWidth=2; cx.beginPath();
    let started=false;
    for(let i=0;i<=frame;i++){
      const f = mp.frames[i]; if(!f.ok){ started=false; continue; }
      const x=X(i), y=Y(Math.min(f.len, top));
      if(started) cx.lineTo(x,y); else { cx.moveTo(x,y); started=true; }
    }
    cx.stroke();

    cx.strokeStyle=css('--muted'); cx.lineWidth=1;
    cx.beginPath(); cx.moveTo(X(frame),pad*0.4); cx.lineTo(X(frame),H-pad); cx.stroke();

    cx.fillStyle=css('--muted'); cx.font='11px ui-monospace,Menlo,monospace';
    cx.fillText('L* = '+Ls.toFixed(2), pad+4, Y(Ls)-6);
    cx.fillText('довжина жадібного шляху', pad+4, 14);
  }

  function stats(){
    const mp = D.maps[map], f = mp.frames[frame];
    T('s_ep').textContent = f.ep;
    T('s_coll').textContent = f.coll;
    T('s_learn').textContent = f.learned + ' / ' + mp.n_blocked;
    const ok = T('s_ok');
    ok.textContent = f.ok ? 'долетів' : (f.hit ? 'колізія' : 'блукає');
    ok.className = 'v ' + (f.ok ? 'ok' : 'bad');
    T('s_len').textContent = f.ok ? f.len.toFixed(2) : '—';
    T('s_eff').textContent = f.ok ? (mp.Lstar/f.len).toFixed(3) : '—';
    T('epnow').textContent = 'епізод ' + f.ep;
  }

  function render(){ drawBoard(); drawCurve(); stats(); }

  // --- controls ---------------------------------------------------------
  const mapsEl = T('maps');
  D.maps.forEach((m,i)=>{
    const b = document.createElement('button');
    b.textContent = m.seed; b.setAttribute('aria-pressed', i===0);
    b.onclick = () => { map=i; [...mapsEl.children].forEach((c,j)=>c.setAttribute('aria-pressed', j===i)); render(); };
    mapsEl.appendChild(b);
  });

  const scrub = T('scrub');
  scrub.max = NF-1;
  scrub.oninput = () => { frame = +scrub.value; render(); };
  ['t_v','t_pol','t_tree','t_astar','t_path'].forEach(id => T(id).onchange = render);

  const play = T('play');
  function tick(){
    if(!playing) return;
    frame = (frame+1) % NF; scrub.value = frame; render();
    raf = setTimeout(tick, 55);
  }
  play.onclick = () => {
    playing = !playing;
    play.textContent = playing ? '❚❚ Пауза' : '▶ Пуск';
    if(playing) tick(); else clearTimeout(raf);
  };
  T('rew').onclick = () => { frame=0; scrub.value=0; render(); };
  T('end').onclick = () => { frame=NF-1; scrub.value=frame; render(); };

  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);
  new MutationObserver(render).observe(document.documentElement, {attributes:true, attributeFilter:['data-theme']});

  render();
})();
</script>
"""

if __name__ == "__main__":
    build()
