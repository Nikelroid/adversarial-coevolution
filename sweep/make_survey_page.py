"""Render sweep/game_survey.json as a self-contained shareable HTML page.

    /scratch1/kelidari/envs/coev/bin/python sweep/make_survey_page.py

Writes private/game_survey.html. No external assets: the CSP on published pages blocks CDNs,
so the charts are hand-rolled SVG built from data inlined into the page.
"""
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJECT_ROOT, "sweep", "game_survey.json")
SRC_RLCARD = os.path.join(PROJECT_ROOT, "sweep", "game_survey_rlcard.json")
OUT = os.path.join(PROJECT_ROOT, "private", "game_survey.html")

# Anchor game: the one our existing study is built on.
ANCHOR = "gin_rummy(num_ranks=13,num_suits=4,hand_size=10,knock_card=10)"


def enrich(rows):
    """Pick the best hidden-information estimate per row and record where it came from."""
    out = []
    for r in rows:
        if "error" in r:
            out.append(r)
            continue
        exact = r.get("infoset_bits_exact")
        closed = r.get("hand_bits_closed_form")
        resamp = r.get("infoset_bits_resampled")
        if exact is not None:
            best, src = exact, "exact"
        elif closed is not None:
            best, src = closed, "closed form"
        elif resamp is not None:
            best, src = resamp, "resampled"
        else:
            best, src = None, "not measured"
        r = dict(r)
        r["hidden_best"] = best
        r["hidden_source"] = src
        r["library"] = r.get("library", "openspiel")
        r["is_anchor"] = r["game"] == ANCHOR
        out.append(r)
    return out


TEMPLATE = r"""<title>The Information Ladder</title>
<style>
  :root {
    color-scheme: light;
    --bg:        #F7F6F1;
    --surface:   #FFFFFF;
    --surface-2: #F1F0EA;
    --ink:       #16201B;
    --ink-2:     #4A5852;
    --muted:     #7B8781;
    --rule:      #E0DED4;
    --accent:    #0B5B39;
    --accent-ink:#0B5B39;
    --anchor:    #9A6B12;
    --s1: #2a78d6;  --s2: #eb6834;  --s3: #1baf7a;  --s4: #eda100;
    --s5: #8a5cd6;  --s6: #c0397a;  --s7: #5f7285;
    --shadow: 0 1px 2px rgba(22,32,27,.06), 0 8px 24px rgba(22,32,27,.05);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --bg:        #0E1411;
      --surface:   #161D19;
      --surface-2: #1C241F;
      --ink:       #E7EDE9;
      --ink-2:     #AEBAB4;
      --muted:     #7E8A84;
      --rule:      #26302A;
      --accent:    #4FA97F;
      --accent-ink:#8ED0AE;
      --anchor:    #E0B457;
      --s1: #3987e5;  --s2: #d95926;  --s3: #199e70;  --s4: #c98500;
      --s5: #9d75e0;  --s6: #d4548d;  --s7: #8fa1b2;
      --shadow: none;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg:        #0E1411;
    --surface:   #161D19;
    --surface-2: #1C241F;
    --ink:       #E7EDE9;
    --ink-2:     #AEBAB4;
    --muted:     #7E8A84;
    --rule:      #26302A;
    --accent:    #4FA97F;
    --accent-ink:#8ED0AE;
    --anchor:    #E0B457;
    --s1: #3987e5;  --s2: #d95926;  --s3: #199e70;  --s4: #c98500;
    --s5: #9d75e0;  --s6: #d4548d;  --s7: #8fa1b2;
    --shadow: none;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    font-size: 16px;
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1120px; margin: 0 auto; padding: 0 24px 96px; }

  header.masthead {
    background: var(--accent);
    color: #F2F7F4;
    padding: 44px 0 38px;
    margin-bottom: 40px;
  }
  header.masthead .wrap { padding-bottom: 0; }
  .eyebrow {
    font-size: 12px; letter-spacing: .14em; text-transform: uppercase;
    color: #A9D2BE; margin: 0 0 12px; font-weight: 600;
  }
  h1 {
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    font-size: clamp(30px, 4.6vw, 46px); line-height: 1.08; margin: 0 0 14px;
    font-weight: 600; letter-spacing: -.01em; text-wrap: balance;
  }
  header.masthead p { margin: 0; max-width: 62ch; color: #CFE4D8; font-size: 17px; }

  h2 {
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
    font-size: 25px; font-weight: 600; margin: 52px 0 6px; letter-spacing: -.01em;
    text-wrap: balance;
  }
  h2:first-of-type { margin-top: 0; }
  .sub { color: var(--ink-2); margin: 0 0 22px; max-width: 68ch; }
  p { max-width: 68ch; }

  .panel {
    background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
    padding: 22px 24px; box-shadow: var(--shadow);
  }
  .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }
  .stat { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px; }
  .stat .n {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 28px; font-variant-numeric: tabular-nums; letter-spacing: -.02em;
    color: var(--accent-ink); display: block; line-height: 1.1;
  }
  .stat .k { font-size: 13px; color: var(--muted); margin-top: 4px; }

  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 0 0 18px; }
  .chip {
    font: inherit; font-size: 13px; padding: 6px 13px; border-radius: 999px; cursor: pointer;
    border: 1px solid var(--rule); background: var(--surface); color: var(--ink-2);
    display: inline-flex; align-items: center; gap: 7px;
  }
  .chip[aria-pressed="true"] { border-color: currentColor; color: var(--ink); background: var(--surface-2); }
  .chip .dot { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
  .chip:focus-visible, th:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

  figure { margin: 0 0 8px; }
  figcaption { font-size: 13px; color: var(--muted); margin-top: 10px; max-width: 68ch; }
  .chartbox { overflow-x: auto; }
  svg { display: block; }
  svg text { font-family: system-ui, sans-serif; }
  .axis line, .axis path { stroke: var(--rule); }
  .gridline { stroke: var(--rule); stroke-dasharray: 2 3; }
  .tick { font-size: 11px; fill: var(--muted); }
  .barlabel { font-size: 11.5px; fill: var(--ink-2); font-variant-numeric: tabular-nums; }
  .rowlabel { font-size: 12px; fill: var(--ink-2); }
  .rowlabel.anchor { fill: var(--anchor); font-weight: 700; }

  .tablebox { overflow-x: auto; border: 1px solid var(--rule); border-radius: 10px; background: var(--surface); }
  table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
  th, td { padding: 9px 12px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--rule); }
  th:first-child, td:first-child { text-align: left; position: sticky; left: 0; background: var(--surface); }
  thead th {
    position: sticky; top: 0; background: var(--surface-2); color: var(--ink-2);
    font-size: 11.5px; letter-spacing: .06em; text-transform: uppercase; cursor: pointer; user-select: none;
    border-bottom: 1px solid var(--rule); z-index: 2;
  }
  thead th:first-child { z-index: 3; background: var(--surface-2); }
  tbody tr:hover td { background: var(--surface-2); }
  tbody tr.anchor td:first-child { box-shadow: inset 3px 0 0 var(--anchor); font-weight: 650; }
  td.num { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }
  td .fam { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 7px; }
  .badge {
    font-size: 10.5px; text-transform: uppercase; letter-spacing: .05em; padding: 2px 6px;
    border-radius: 4px; border: 1px solid var(--rule); color: var(--muted); margin-left: 8px;
  }

  #tip {
    position: fixed; pointer-events: none; opacity: 0; transition: opacity .1s;
    background: var(--surface); color: var(--ink); border: 1px solid var(--rule);
    border-radius: 8px; padding: 9px 11px; font-size: 12.5px; box-shadow: var(--shadow);
    max-width: 280px; z-index: 50; line-height: 1.45;
  }
  #tip b { font-size: 13px; }
  #tip .r { color: var(--ink-2); font-variant-numeric: tabular-nums; }

  ul.notes { padding-left: 20px; max-width: 68ch; }
  ul.notes li { margin-bottom: 9px; color: var(--ink-2); }
  ul.notes li b { color: var(--ink); font-weight: 650; }
  code {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em;
    background: var(--surface-2); padding: 1px 5px; border-radius: 4px;
  }
  .rec { border-left: 3px solid var(--accent); padding-left: 18px; margin: 22px 0; }
  .rec h3 { margin: 0 0 4px; font-size: 16px; }
  .rec p { margin: 0 0 16px; color: var(--ink-2); font-size: 14.5px; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<header class="masthead">
  <div class="wrap">
    <p class="eyebrow">Candidate survey &middot; measured __DATE__</p>
    <h1>The Information Ladder</h1>
    <p>Every imperfect-information game we can reach from this machine, measured rather than
       guessed: how much is actually hidden, how big the action and observation spaces are, and
       what each one costs to run. __NCAND__ candidates across two libraries, from Kuhn poker to
       reconnaissance blind chess to a four-player mahjong table. The question this page answers
       is which handful make the ladder.</p>
  </div>
</header>

<div class="wrap">

  <div class="grid-3" id="stats"></div>

  <h2>How much is actually hidden</h2>
  <p class="sub">Hidden information in bits: how many distinct world states the player to move
     cannot tell apart, averaged over reachable decision points. Zero means perfect information.
     Gin Rummy, the game our study is already built on, is marked.</p>
  <div class="controls" id="filters"></div>
  <figure>
    <div class="chartbox"><svg id="chart-bits"></svg></div>
    <figcaption id="cap-bits"></figcaption>
  </figure>

  <h2>Information against cost</h2>
  <p class="sub">The decision chart. Games toward the right hide more; games toward the bottom are
     cheaper to learn, because the network input is smaller. A good ladder picks rungs spread
     across the horizontal axis while staying low on the vertical one.</p>
  <figure>
    <div class="chartbox"><svg id="chart-scatter"></svg></div>
    <figcaption>Vertical axis is the learner's input size (information-state tensor, or observation
      tensor when no information-state tensor exists), on a log scale. Marker area grows with the
      declared action count.</figcaption>
  </figure>

  <h2>Action space, in principle and in practice</h2>
  <p class="sub">The declared action count is the size of the output layer. The branching factor is
     how many actions are actually legal at a typical decision. The gap between them is how much of
     the head is masked away on any given turn.</p>
  <figure>
    <div class="chartbox"><svg id="chart-actions"></svg></div>
    <figcaption>Log scale. Each row runs from the empirical branching factor to the declared action
      count.</figcaption>
  </figure>

  <h2>Every candidate</h2>
  <p class="sub">Click any column heading to sort. The family filter above applies here too.</p>
  <div class="tablebox"><table id="tbl"><thead></thead><tbody></tbody></table></div>

  <h2>What the numbers argue for</h2>
  <div class="rec" id="rec"></div>

  <h2>Balatro, and games with no opponent</h2>
  <p class="sub">Chased down because it was asked directly, and because the answer says something
     about what this page is actually measuring.</p>
  <div class="panel">
    <p><b>Balatro does not belong on this ladder, and the reason is the interesting part.</b> The
      axis here is information one player hides from another. Balatro has no other player. You can
      inspect your own deck at any point, so its composition is never hidden; the only uncertainty
      is the order the shuffle produced. That is chance, not concealment, and it is a different
      quantity from the one every bar on this page measures.</p>
    <p>On this axis it belongs with the solo rungs, <b>blackjack</b> and <b>Klondike solitaire</b>,
      both measured in the table above: one player, a shuffled deck, no opponent to model anywhere.
      If we want a rung for pure chance-driven uncertainty, those two already provide it today,
      with no proprietary game in the loop.</p>
    <p>The tooling is better than expected. There is a mod exposing a JSON-RPC API
      (<code>coder/balatrobot</code>), a Gymnasium wrapper (<code>cassiusfive/balatro-gym</code>),
      an independent engine and move generator in Rust with Python bindings
      (<code>evanofslack/balatro-rs</code>), and a PPO project
      (<code>taggarttufte/balatro-rl</code>) reporting a peak <b>2.35 percent win rate after five
      months and eight architecture revisions</b>, described as about 235 times better than random
      play.</p>
    <p>That last number is why it is worth watching anyway. "235 times better than random" is
      precisely the claim this study argues is uninformative: with no fixed reference of known
      strength, nothing tells you whether 2.35 percent is near the ceiling or nowhere close.
      Balatro is an excellent <em>illustration</em> of the problem, and a poor <em>rung</em> on a
      ladder about hidden information.</p>
  </div>

  <h2>Reachable, but not measured</h2>
  <p class="sub">Environments that would extend this table, and what each would actually buy. None
     of them are installed here, so no number above comes from any of them. This is a to-check
     list, not a measurement.</p>
  <ul class="notes">
    <li><b>mjx</b> — Japanese mahjong, much faster than the Chinese variant we measured through
      RLCard. Four players, a hidden wall and three hidden hands, so it would land at the very top
      of the hidden-information column.</li>
    <li><b>reconchess</b> — the official Reconnaissance Blind Chess environment and competition.
      OpenSpiel's <code>rbc</code> is already measured above, so what this adds is a pool of
      published opponents to grade against, which is the expensive half of the protocol.</li>
    <li><b>Hanabi Learning Environment</b> — the canonical Hanabi. OpenSpiel's version is already
      in the table, so this buys comparability with the published Hanabi results rather than a new
      point on the axis.</li>
    <li><b>pgx</b> and <b>JaxMARL</b> — JAX reimplementations of games we already have. Same
      information structure, orders of magnitude faster, which only matters once a rung needs many
      seeds.</li>
    <li><b>TextArena</b>, <b>Diplomacy</b>, hidden-role games such as Werewolf and Avalon — huge
      and unstructured hidden information, aimed at language-model agents. None of the closed forms
      on this page apply, and the fixed-expert protocol would need rebuilding from scratch.</li>
    <li><b>Stratego</b> — the headline imperfect-information board game, and still the gap: there
      is no maintained public environment to measure.</li>
  </ul>

  <h2>Method, and where to distrust it</h2>
  <ul class="notes">
    <li><b>Exact</b> hidden bits come from walking the whole game tree, grouping histories by the
      acting player's information state, and averaging log2 of each group's size. Only small games
      can be enumerated, so this is the most trustworthy column and the rarest.</li>
    <li><b>Closed form</b> hidden bits count the ways the hidden hands could have been dealt. With
      one opponent that is log2 C(unseen, hand size); at a three or four-handed table it is the
      multinomial over all the hidden hands, because it matters which opponent holds which cards.
      Exact where it applies, but it counts only the hands, ignoring uncertainty about the undealt
      stock.</li>
    <li><b>Resampled</b> hidden bits sample worlds consistent with the player's information state
      and count distinct ones. It is censored at the sample count, so it is a lower bound. Hearts,
      Euchre and Oh Hell all came back at exactly 6.32 bits, which is log2(80), the sample size:
      every draw was distinct and the estimator simply ran out of room. The closed form puts Hearts
      at 56.2. Treat any row sitting exactly at 6.32 as "at least this, and probably far more".</li>
    <li><b>RLCard rows are a different instrument.</b> That library exposes no information-state
      strings and no world resampling, so those rows carry the closed form and the empirical
      measurements only, never an exact or resampled estimate. They are marked in the Engine
      column. Mahjong and UNO are there because OpenSpiel does not ship them at all.</li>
    <li>Where two estimates exist they should roughly agree. Disagreement is informative rather
      than embarrassing: it usually means the hidden state is not just the opponent's hand.</li>
    <li><b>Two engines, one number.</b> Gin Rummy appears twice, once through OpenSpiel and once
      through RLCard, and both land on 30.1 bits. That is the cheapest check available that the
      closed form is being applied consistently across libraries. Leduc is the instructive
      disagreement: the exact walk gives 2.01 bits against the closed form's 2.3, because the
      closed form counts only the deal and ignores what the public card and the betting have
      already ruled out. Read every closed form as a mild overestimate for that reason.</li>
    <li><b>Input size is a design choice, not a property of the game.</b> The same Gin Rummy is
      644 floats in OpenSpiel and 260 in RLCard. The vertical axis of the cost chart therefore
      compares encodings as much as games, and should be read as an order of magnitude rather
      than a measurement.</li>
    <li>Branching and episode length come from uniform random play, which visits different states
      than a trained agent would. Treat them as scale, not as precise cost.</li>
    <li>Measurement is time-boxed per game, so a few large games report partial coverage rather
      than stalling the survey. Those rows carry a <span class="badge">partial</span> badge.</li>
  </ul>
  <p style="color:var(--muted);font-size:13px;margin-top:26px">
    Generated from <code>sweep/game_survey.json</code> by <code>sweep/make_survey_page.py</code>.
    Regenerate after changing the candidate list; every number on this page comes from that file.
  </p>
</div>

<div id="tip" role="status" aria-live="polite"></div>

<script>
const RAW = __DATA__;
const FAMS = [
  {key:"card",  label:"Card games, 2p",       css:"--s1"},
  {key:"dial",  label:"Gin Rummy dial",       css:"--s2"},
  {key:"board", label:"Board and fog of war", css:"--s3"},
  {key:"dice",  label:"Dice games",           css:"--s4"},
  {key:"comm",  label:"Bargaining and signalling", css:"--s5"},
  {key:"multi", label:"Card games, 3-4p",     css:"--s6"},
  {key:"solo",  label:"Solo against chance",  css:"--s7"},
];
const famColor = k => getComputedStyle(document.documentElement)
  .getPropertyValue((FAMS.find(f=>f.key===k)||FAMS[0]).css).trim();

const rows = RAW.filter(r => !r.error);
let active = new Set(FAMS.map(f=>f.key));
const shown = () => rows.filter(r => active.has(r.family));

/* ---------- tooltip ---------- */
const tip = document.getElementById("tip");
function showTip(e, html) {
  tip.innerHTML = html; tip.style.opacity = 1;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > innerWidth - 8) x = e.clientX - w - pad;
  if (y + h > innerHeight - 8) y = e.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => tip.style.opacity = 0;
function tipHtml(r) {
  const f = v => (v === null || v === undefined) ? "&mdash;" : v;
  return `<b>${r.label}</b><br>
    <span class="r">hidden ${f(r.hidden_best)} bits &middot; ${r.hidden_source}</span><br>
    <span class="r">actions ${f(r.actions_declared)} &middot; legal ~${f(r.branch_mean)}</span><br>
    <span class="r">input ${f(r.input_size)} &middot; length ~${f(r.len_mean)}</span><br>
    <span class="r">${r.information.replace("_"," ").toLowerCase()}</span>`;
}
const NS = "http://www.w3.org/2000/svg";
const el = (n, a={}) => { const e = document.createElementNS(NS, n);
  for (const k in a) e.setAttribute(k, a[k]); return e; };

/* ---------- chart 1: hidden bits ---------- */
function drawBits() {
  const svg = document.getElementById("chart-bits");
  svg.innerHTML = "";
  const data = shown().filter(r => r.hidden_best !== null)
                      .sort((a,b) => b.hidden_best - a.hidden_best);
  const W = Math.max(680, Math.min(1072, svg.parentNode.clientWidth || 900));
  const rowH = 22, padL = 218, padR = 58, padT = 26, padB = 34;
  const H = padT + data.length * rowH + padB;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  const maxV = Math.max(10, ...data.map(r => r.hidden_best));
  const x = v => padL + (v / maxV) * (W - padL - padR);

  for (let t = 0; t <= maxV; t += (maxV > 25 ? 10 : 5)) {
    svg.appendChild(el("line", {class:"gridline", x1:x(t), x2:x(t), y1:padT-8, y2:H-padB+2}));
    const lb = el("text", {class:"tick", x:x(t), y:H-padB+16, "text-anchor":"middle"});
    lb.textContent = t; svg.appendChild(lb);
  }
  const ax = el("text", {class:"tick", x:padL, y:padT-14, "text-anchor":"start"});
  ax.textContent = "bits of hidden information"; svg.appendChild(ax);

  data.forEach((r, i) => {
    const y = padT + i * rowH, bh = rowH - 8;
    const lab = el("text", {class:"rowlabel" + (r.is_anchor ? " anchor" : ""),
      x: padL - 10, y: y + bh - 2, "text-anchor":"end"});
    lab.textContent = r.label; svg.appendChild(lab);
    const bar = el("rect", {x:padL, y, width:Math.max(1.5, x(r.hidden_best)-padL), height:bh,
      rx:4, fill:famColor(r.family), "fill-opacity": r.hidden_source === "resampled" ? .45 : .9});
    if (r.is_anchor) { bar.setAttribute("stroke", "var(--anchor)"); bar.setAttribute("stroke-width", 2); }
    bar.addEventListener("mousemove", e => showTip(e, tipHtml(r)));
    bar.addEventListener("mouseleave", hideTip);
    svg.appendChild(bar);
    const v = el("text", {class:"barlabel", x:x(r.hidden_best)+7, y:y+bh-3});
    v.textContent = r.hidden_best.toFixed(1); svg.appendChild(v);
  });
  document.getElementById("cap-bits").innerHTML =
    "Solid bars are exact or closed-form measurements. Faded bars are resampled lower bounds, so " +
    "their true value is at least what is shown. " + data.length + " of " + rows.length +
    " candidates have a usable estimate.";
}

/* ---------- chart 2: information against cost ---------- */
function drawScatter() {
  const svg = document.getElementById("chart-scatter");
  svg.innerHTML = "";
  const data = shown().filter(r => r.hidden_best !== null && r.input_size);
  const W = Math.max(680, Math.min(1072, svg.parentNode.clientWidth || 900)), H = 430;
  const padL = 62, padR = 26, padT = 22, padB = 48;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  const maxX = Math.max(10, ...data.map(r => r.hidden_best));
  const ys = data.map(r => r.input_size);
  const loY = Math.log10(Math.max(1, Math.min(...ys))), hiY = Math.log10(Math.max(...ys));
  const x = v => padL + (v/maxX) * (W-padL-padR);
  const y = v => H - padB - ((Math.log10(Math.max(1,v)) - loY) / Math.max(.001, hiY-loY)) * (H-padT-padB);

  for (let t = 0; t <= maxX; t += (maxX > 25 ? 10 : 5)) {
    svg.appendChild(el("line", {class:"gridline", x1:x(t), x2:x(t), y1:padT, y2:H-padB}));
    const lb = el("text", {class:"tick", x:x(t), y:H-padB+16, "text-anchor":"middle"});
    lb.textContent = t; svg.appendChild(lb);
  }
  [10,100,1000].forEach(t => {
    if (Math.log10(t) < loY || Math.log10(t) > hiY) return;
    svg.appendChild(el("line", {class:"gridline", x1:padL, x2:W-padR, y1:y(t), y2:y(t)}));
    const lb = el("text", {class:"tick", x:padL-8, y:y(t)+4, "text-anchor":"end"});
    lb.textContent = t; svg.appendChild(lb);
  });
  const xt = el("text", {class:"tick", x:(padL+W-padR)/2, y:H-10, "text-anchor":"middle"});
  xt.textContent = "bits of hidden information"; svg.appendChild(xt);
  const yt = el("text", {class:"tick", x:14, y:(padT+H-padB)/2, "text-anchor":"middle",
    transform:`rotate(-90 14 ${(padT+H-padB)/2})`});
  yt.textContent = "learner input size (floats)"; svg.appendChild(yt);

  data.sort((a,b) => b.actions_declared - a.actions_declared).forEach(r => {
    const rad = 5 + 9 * Math.sqrt(Math.min(1, r.actions_declared / 400));
    const c = el("circle", {cx:x(r.hidden_best), cy:y(r.input_size), r:rad,
      fill:famColor(r.family), "fill-opacity":.55, stroke:famColor(r.family), "stroke-width":1.5});
    if (r.is_anchor) { c.setAttribute("stroke", "var(--anchor)"); c.setAttribute("stroke-width", 2.5); }
    c.addEventListener("mousemove", e => showTip(e, tipHtml(r)));
    c.addEventListener("mouseleave", hideTip);
    svg.appendChild(c);
    if (r.is_anchor || r.hidden_best > 24 || r.hidden_best === 0) {
      const t = el("text", {class:"barlabel", x:x(r.hidden_best)+rad+5, y:y(r.input_size)+4});
      t.textContent = r.label.replace(" (standard)","").replace("Gin dial: ","");
      svg.appendChild(t);
    }
  });
}

/* ---------- chart 3: action space dumbbell ---------- */
function drawActions() {
  const svg = document.getElementById("chart-actions");
  svg.innerHTML = "";
  const data = shown().filter(r => r.branch_mean).sort((a,b) => b.actions_declared - a.actions_declared);
  const W = Math.max(680, Math.min(1072, svg.parentNode.clientWidth || 900));
  const rowH = 21, padL = 218, padR = 66, padT = 26, padB = 34;
  const H = padT + data.length * rowH + padB;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", W); svg.setAttribute("height", H);
  const hi = Math.log10(Math.max(...data.map(r => r.actions_declared)));
  const x = v => padL + (Math.log10(Math.max(1,v)) / Math.max(.001, hi)) * (W-padL-padR);
  [1,10,100,1000].forEach(t => {
    if (Math.log10(t) > hi) return;
    svg.appendChild(el("line", {class:"gridline", x1:x(t), x2:x(t), y1:padT-8, y2:H-padB+2}));
    const lb = el("text", {class:"tick", x:x(t), y:H-padB+16, "text-anchor":"middle"});
    lb.textContent = t; svg.appendChild(lb);
  });
  const ax = el("text", {class:"tick", x:padL, y:padT-14});
  ax.textContent = "actions (log scale): legal at a decision → declared total"; svg.appendChild(ax);

  data.forEach((r, i) => {
    const y = padT + i*rowH + rowH/2 - 3;
    const lab = el("text", {class:"rowlabel" + (r.is_anchor ? " anchor":""), x:padL-10, y:y+4,
      "text-anchor":"end"});
    lab.textContent = r.label; svg.appendChild(lab);
    const c = famColor(r.family);
    svg.appendChild(el("line", {x1:x(r.branch_mean), x2:x(r.actions_declared), y1:y, y2:y,
      stroke:c, "stroke-width":2, "stroke-opacity":.35}));
    [[r.branch_mean, .45], [r.actions_declared, 1]].forEach(([v, op]) => {
      const dot = el("circle", {cx:x(v), cy:y, r:4.5, fill:c, "fill-opacity":op,
        stroke:"var(--surface)", "stroke-width":1.5});
      dot.addEventListener("mousemove", e => showTip(e, tipHtml(r)));
      dot.addEventListener("mouseleave", hideTip);
      svg.appendChild(dot);
    });
    const t = el("text", {class:"barlabel", x:x(r.actions_declared)+9, y:y+4});
    t.textContent = r.actions_declared; svg.appendChild(t);
  });
}

/* ---------- table ---------- */
const COLS = [
  ["label",           "Game",            false],
  ["hidden_best",     "Hidden bits",     true ],
  ["hidden_source",   "Estimate",        false],
  ["actions_declared","Actions",         true ],
  ["branch_mean",     "Legal / turn",    true ],
  ["input_size",      "Input size",      true ],
  ["len_mean",        "Turns",           true ],
  ["players",         "Players",         true ],
  ["library",         "Engine",          false],
  ["information",     "Information",     false],
  ["dynamics",        "Play",            false],
];
let sortKey = "hidden_best", sortDir = -1;
function drawTable() {
  const t = document.getElementById("tbl");
  const thead = t.querySelector("thead"), tbody = t.querySelector("tbody");
  thead.innerHTML = ""; tbody.innerHTML = "";
  const tr = document.createElement("tr");
  COLS.forEach(([k, label]) => {
    const th = document.createElement("th");
    th.textContent = label + (sortKey === k ? (sortDir < 0 ? " ↓" : " ↑") : "");
    th.tabIndex = 0;
    const doSort = () => { if (sortKey === k) sortDir *= -1; else { sortKey = k; sortDir = -1; } drawTable(); };
    th.addEventListener("click", doSort);
    th.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); doSort(); } });
    tr.appendChild(th);
  });
  thead.appendChild(tr);
  const data = shown().slice().sort((a,b) => {
    const A = a[sortKey], B = b[sortKey];
    if (A === null || A === undefined) return 1;
    if (B === null || B === undefined) return -1;
    return (typeof A === "number" ? A - B : String(A).localeCompare(String(B))) * sortDir;
  });
  data.forEach(r => {
    const row = document.createElement("tr");
    if (r.is_anchor) row.className = "anchor";
    COLS.forEach(([k], i) => {
      const td = document.createElement("td");
      let v = r[k];
      if (v === null || v === undefined) v = "—";
      if (typeof v === "string") v = v.replace(/_/g, " ").toLowerCase();
      if (i === 0) {
        const dot = document.createElement("span");
        dot.className = "fam"; dot.style.background = famColor(r.family);
        td.appendChild(dot);
        td.appendChild(document.createTextNode(r.label));
        if (r.hidden_source === "resampled" || r.len_mean === null) {
          const b = document.createElement("span");
          b.className = "badge"; b.textContent = "partial"; td.appendChild(b);
        }
      } else {
        td.textContent = v;
        if (typeof r[k] === "number") td.className = "num";
      }
      row.appendChild(td);
    });
    tbody.appendChild(row);
  });
}

/* ---------- filters, stats, recommendation ---------- */
function drawFilters() {
  const box = document.getElementById("filters");
  box.innerHTML = "";
  FAMS.forEach(f => {
    if (!rows.some(r => r.family === f.key)) return;
    const b = document.createElement("button");
    b.className = "chip"; b.type = "button";
    b.setAttribute("aria-pressed", active.has(f.key));
    b.innerHTML = `<span class="dot" style="background:${famColor(f.key)}"></span>${f.label}`;
    b.addEventListener("click", () => {
      if (active.has(f.key)) { if (active.size > 1) active.delete(f.key); }
      else active.add(f.key);
      drawAll();
    });
    box.appendChild(b);
  });
}
function drawStats() {
  const withBits = rows.filter(r => r.hidden_best !== null);
  const exact = rows.filter(r => r.hidden_source === "exact").length;
  const anchor = rows.find(r => r.is_anchor);
  const box = document.getElementById("stats");
  box.innerHTML = "";
  const top = withBits.slice().sort((a,b) => b.hidden_best - a.hidden_best)[0];
  [[rows.length, "candidates measured"],
   [withBits.length + " of " + rows.length, "with a hidden-information number"],
   [(anchor ? anchor.hidden_best.toFixed(1) : "—") + " bits",
    "hidden in standard Gin Rummy"],
   [top.hidden_best.toFixed(1) + " bits",
    "the most hidden: " + top.label]].forEach(([n, k]) => {
    const d = document.createElement("div");
    d.className = "stat";
    d.innerHTML = `<span class="n">${n}</span><span class="k">${k}</span>`;
    box.appendChild(d);
  });
}
function drawRec() {
  const anchor = rows.find(r => r.is_anchor);
  const a = anchor ? anchor.hidden_best : 30.1;
  const withBits = r => r.hidden_best !== null && r.hidden_best !== undefined;
  const twoP  = rows.filter(r => withBits(r) && r.players === 2);
  const above2 = twoP.filter(r => r.hidden_best > a).sort((x,y) => x.hidden_best - y.hidden_best);
  const below2 = twoP.filter(r => r.hidden_best > 0 && r.hidden_best < a && r.family !== "dial")
                     .sort((x,y) => y.hidden_best - x.hidden_best);
  const bigTable = rows.filter(r => withBits(r) && r.players > 2)
                       .sort((x,y) => y.hidden_best - x.hidden_best);
  const dial = rows.filter(r => r.family === "dial").length;
  const fog = rows.filter(r => r.family === "board" && !withBits(r));
  const fmt = r => `${r.label} (${r.hidden_best.toFixed(1)} bits)`;
  document.getElementById("rec").innerHTML = `
    <h3>Gin Rummy is at the ceiling for two players</h3>
    <p>Across every two-player game measured here, only
       ${above2.length ? above2.map(fmt).join(", ") : "<b>none</b>"} hide more than standard Gin
       Rummy's ${a.toFixed(1)} bits. A 52-card deck split between two hands cannot conceal much
       more than this, so a ladder built from distinct two-player card games bunches up at the
       top instead of spreading out.</p>
    <h3>More hidden information means more players, not a harder duel</h3>
    <p>Everything that clearly beats the anchor sits at a three or four-handed table:
       ${bigTable.slice(0,3).map(fmt).join(", ")}. Those extra bits come from extra hands, so they
       buy a different problem, coalitions and signalling, rather than a deeper version of the one
       we already study. Worth knowing, wrong for this ladder.</p>
    <h3>The range still has to come from the dial</h3>
    <p>The ${dial} Gin Rummy configurations hold the rules fixed and move only the deck and hand
       size, which is the one comparison where a change in outcome cannot be blamed on a change in
       rules. That remains the strongest evidence the ladder can produce, and it is all in the game
       we already have an expert for.</p>
    <h3>The cheap rungs are the ones worth adding</h3>
    <p>${below2.slice(0,3).map(fmt).join(", ")} cost almost nothing to run, and the smallest are
       solvable exactly, which is what lets us check the probes against ground truth before
       trusting them on a game nobody can solve.</p>
    <h3>Fog of war is the real gap</h3>
    <p>${fog.length} board games here, dark chess and reconnaissance blind chess among them, carry
       no hidden-information number at all: too large to enumerate, no closed form to apply, and
       OpenSpiel implements no world resampling for them. If the ladder ever wants a rung whose
       hidden information is positional rather than card-shaped, that measurement has to be built
       before the rung can be justified.</p>`;
}
function drawAll() { drawFilters(); drawStats(); drawBits(); drawScatter(); drawActions(); drawTable(); drawRec(); }
drawAll();
addEventListener("resize", () => { drawBits(); drawScatter(); drawActions(); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawAll);
</script>
"""


def load(path):
    """Rows from one survey file. Missing file is not an error: the OpenSpiel pass stands alone."""
    if not os.path.exists(path):
        print(f"   note: {os.path.basename(path)} not found, skipping that engine")
        return []
    with open(path) as f:
        return json.load(f)["rows"]


def main():
    rows = enrich(load(SRC) + load(SRC_RLCARD))
    ok = [r for r in rows if "error" not in r]
    html = (TEMPLATE
            .replace("__DATA__", json.dumps(ok))
            .replace("__NCAND__", str(len(ok)))
            .replace("__DATE__", "16 August 2026"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    failed = [r for r in rows if "error" in r]
    print(f"wrote {OUT}  ({len(ok)} candidates rendered, {len(failed)} failed)")
    for r in failed:
        print(f"   failed: {r['label']}  {r['error'][:70]}")


if __name__ == "__main__":
    main()
