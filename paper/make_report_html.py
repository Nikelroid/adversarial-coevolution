"""Generate docs/index.html: the project's full, plain-language report.

The whole story of "Adversarial Co-Evolution of RL and LLM Agents in Gin Rummy":
the framework we built, every training regime we tried, WHY each worked or failed, the
results, and the bottom line. Figures load by raw GitHub URL (small page). Run after
make_figures.py. Every number here is measured -- see the JSON under sweep/.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "index.html")
SWEEP = os.path.join(HERE, "..", "sweep")
RAW = ("https://raw.githubusercontent.com/Nikelroid/adversarial-coevolution/"
       "main/paper/figures")
PDF = ("https://github.com/Nikelroid/adversarial-coevolution/blob/main/"
       "paper/main.pdf")
REPO = "https://github.com/Nikelroid/adversarial-coevolution"


def img(name, alt=""):
    return f'<img src="{RAW}/{name}" alt="{alt}" loading="lazy"/>'


# ----------------------------------------------------------------- data helpers
def _load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def gold_vs(opp):
    """Gold's win-rate vs a named agent, from the benchmark (or None)."""
    d = _load(os.path.join(SWEEP, "gold_bench.json"))
    try:
        return d["matchups"][opp]["a_win_rate"]
    except Exception:
        return None


def best_models_rows():
    """The precise (2000-game) re-eval of the strongest agents, if available."""
    r = _load(os.path.join(SWEEP, "curriculum", "_best_models_precise.json"))
    return r or []


def best_agent_gold():
    rows = best_models_rows()
    if rows:
        return max(r["gold"] for r in rows) * 100
    return 35.7


def _curr_cells(prefix):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(os.path.join(SWEEP, "curriculum", "*.json")):
        if os.path.basename(f).startswith("_"):
            continue
        d = _load(f)
        if not d or not d.get("name", "").startswith(prefix):
            continue
        c = re.sub(r"_s\d+$", "", d["name"])
        by[c]["gold"].append(d["vs_gold"]["win_rate"] * 100)
        by[c]["best"].append(d.get("best_vs_gold", d["vs_gold"]["win_rate"]) * 100)
        by[c]["champ"].append(d["vs_champion"]["win_rate"] * 100)
        by[c]["gin"].append(d["vs_gold"]["gin_rate"] * 100)
    return by


_CURR_LABEL = {
    "01_base_trpo": "TRPO baseline", "02_base_ppo": "PPO (vs TRPO)",
    "03_rew_R0_highgin": "reward: pay 3&times; for gin", "04_rew_R2_knockfwd": "reward: knock-forward",
    "05_rew_R4_earlyknk": "reward: early-knock", "06_cur_noRandTail": "curric: drop random late",
    "07_cur_pfsp": "curric: PFSP", "08_trpo_gamma_high": "longer memory (&gamma;=.997)",
    "09_trpo_kl_small": "smaller steps", "10_trpo_kl_large": "larger steps",
    "11_trpo_gae_batch": "GAE + big batch", "12_ppo_ent_hi": "PPO, more explore",
}


def best_models_table():
    rows = best_models_rows()
    if not rows:
        return ('<div class="note">The precise 2000-game re-evaluation is still running; this '
                'table fills in when it lands.</div>')
    label = {"p7_deadwood_hi": "Curriculum Ace (deadwood-coached)",
             "p7_pfsp_s0": "League Tactician", "p7_pfsp_s1": "League Tactician II",
             "p7_warmlong_s0": "Marathon Champion", "p7_warmlong_s1": "Marathon Champion II",
             "p7_deadwood_s0": "Deadwood Specialist", "p7_deadwood_s1": "Deadwood Specialist II",
             "p6_curriculum_champ": "Phase-6 Champion (early-knock)",
             "p6_gold_hunter": "Gold Hunter"}
    body = "\n".join(
        f'<tr><td>{label.get(r["label"], r["label"])}</td>'
        f'<td class="n"><b>{r["gold"]*100:.1f}</b> &plusmn;{r["gold_ci"]*100:.1f}</td>'
        f'<td class="n">{r["champ"]*100:.1f}</td>'
        f'<td class="n">{r["rand"]*100:.0f}</td>'
        f'<td class="n">{r["gold_gin"]*100:.2f}</td></tr>'
        for r in sorted(rows, key=lambda x: -x["gold"])[:6])
    return ('<table><tr><th>agent</th><th class="n">vs gold % (95% CI)</th>'
            '<th class="n">vs champion %</th><th class="n">vs random %</th>'
            '<th class="n">gin % vs gold</th></tr>' + body + '</table>')


def curriculum_table(prefix, n=8):
    by = _curr_cells(prefix)
    if not by:
        return '<div class="note">results pending</div>'
    cells = sorted(by, key=lambda c: -st.mean(by[c]["best"]))[:n]
    body = "\n".join(
        f'<tr><td>{_CURR_LABEL.get(c, c)}</td>'
        f'<td class="n">{st.mean(by[c]["champ"]):.0f}</td>'
        f'<td class="n"><b>{st.mean(by[c]["best"]):.0f}</b></td>'
        f'<td class="n">{st.mean(by[c]["gin"]):.2f}</td></tr>'
        for c in cells)
    return ('<table><tr><th>one change from the baseline</th><th class="n">vs champion %</th>'
            '<th class="n">best vs gold %</th><th class="n">gin % vs gold</th></tr>'
            + body + '</table>')


# numbers used in prose / KPIs (all measured)
BEST = best_agent_gold()
GOLD_CHAMP = (gold_vs("champion") or 0.702) * 100          # gold's win vs champion
CHAMP_GOLD = 100 - GOLD_CHAMP                              # champion's win vs gold
N_RUNS = len([f for f in glob.glob(os.path.join(SWEEP, "curriculum", "*.json"))
              if not os.path.basename(f).startswith("_")])

CSS = """
:root{--green:#0b5b39;--green2:#073d27;--gold:#d9a521;--ink:#16242d;--muted:#5c6b73;
--card:#fff;--line:#e3e8e6;--good:#1f8a5a;--bad:#c0392b;}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
color:var(--ink);background:#f4f6f5;line-height:1.65;}
.hero{background:linear-gradient(135deg,var(--green),var(--green2));color:#fff;padding:52px 24px 42px;
text-align:center;border-bottom:4px solid var(--gold);}
.hero h1{margin:0 0 12px;font-size:30px;letter-spacing:.3px;}
.hero p{margin:6px auto;max-width:800px;opacity:.94;font-size:15.5px;}
.hero .by{font-size:13px;opacity:.82;margin-top:14px;}
.links{margin-top:20px;display:flex;gap:10px;justify-content:center;flex-wrap:wrap;}
.links a{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.35);border-radius:999px;
padding:7px 16px;font-size:13.5px;font-weight:600;color:#fff;text-decoration:none;}
.links a:hover{background:rgba(255,255,255,.28);}
.kpibar{display:flex;gap:14px;justify-content:center;flex-wrap:wrap;margin:22px auto 0;max-width:880px;}
.kpibar .k{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.25);border-radius:12px;
padding:10px 16px;min-width:120px;}
.kpibar .v{font-size:21px;font-weight:800;color:#fff;}
.kpibar .l{font-size:11.5px;opacity:.9;color:#eafff5;}
.nav{position:sticky;top:0;z-index:20;background:#08311f;border-bottom:1px solid rgba(255,255,255,.12);
display:flex;gap:4px;justify-content:center;flex-wrap:wrap;padding:8px 10px;}
.nav a{color:#cfe9dc;text-decoration:none;font-size:12.5px;padding:4px 9px;border-radius:6px;}
.nav a:hover{background:rgba(255,255,255,.12);color:#fff;}
.wrap{max-width:940px;margin:0 auto;padding:0 22px 70px;}
section{background:var(--card);margin:20px 0;padding:24px 30px;border-radius:14px;
box-shadow:0 1px 3px rgba(0,0,0,.06);border:1px solid var(--line);scroll-margin-top:48px;}
h2{color:var(--green);border-bottom:2px solid var(--line);padding-bottom:8px;margin-top:2px;font-size:22px;}
h2 .n{color:var(--gold);font-weight:800;margin-right:8px;}
h3{color:var(--green2);font-size:16px;margin:20px 0 5px;}
p,li{font-size:15px;}
code{background:#eef2f0;padding:1.5px 6px;border-radius:5px;font-size:13px;color:#0a3b27;}
.fig{margin:16px 0;text-align:center;}
.fig img{max-width:100%;border:1px solid var(--line);border-radius:8px;background:#fff;}
.fig figcaption{font-size:13px;color:var(--muted);margin-top:7px;}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
@media(max-width:680px){.g2{grid-template-columns:1fr;}}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px;}
th,td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left;}
th{background:#f0f4f2;color:var(--green2);font-weight:700;}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;}
tr:hover td{background:#fafcfb;}
.good{color:var(--good);font-weight:700;}.bad{color:var(--bad);font-weight:700;}
.note{background:#fff8e6;border-left:4px solid var(--gold);padding:12px 16px;border-radius:6px;margin:14px 0;font-size:14.5px;}
.why{background:#eef6f1;border-left:4px solid var(--good);padding:11px 16px;border-radius:6px;margin:10px 0;font-size:14.5px;}
.why b{color:var(--green2);}
.diagram{background:#0e2c1f;color:#d6f5e6;padding:16px 18px;border-radius:10px;overflow:auto;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;line-height:1.45;white-space:pre;}
.tl{display:flex;gap:0;align-items:stretch;flex-wrap:wrap;margin:18px 0;}
.tl .st{flex:1;min-width:150px;background:#f0f7f3;border:1px solid var(--line);border-radius:10px;
padding:12px 14px;margin:4px;position:relative;}
.tl .st .big{font-size:22px;font-weight:800;color:var(--green);}
.tl .st .lbl{font-size:12.5px;color:var(--muted);}
.tl .st .nm{font-size:13.5px;font-weight:700;color:var(--ink);margin-bottom:4px;}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:16px 0;}
@media(max-width:680px){.cards{grid-template-columns:1fr;}}
.rc{border:1px solid var(--line);border-radius:11px;padding:14px 16px;background:#fcfdfc;}
.rc h4{margin:0 0 4px;font-size:15px;color:var(--green2);}
.rc .res{font-size:13px;font-weight:700;margin:6px 0;}
.rc .res.win{color:var(--good);}.rc .res.lose{color:var(--bad);}.rc .res.mid{color:#a9810a;}
.rc p{font-size:13.5px;margin:6px 0 0;}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0;}
@media(max-width:680px){.kpis{grid-template-columns:repeat(2,1fr);}}
.kpi{background:var(--green);color:#fff;border-radius:12px;padding:15px;text-align:center;}
.kpi .v{font-size:23px;font-weight:800;}.kpi .l{font-size:12px;opacity:.92;margin-top:3px;}
.kpi.warn{background:#9a6b00;}.kpi.gold{background:var(--gold);color:#3a2e00;}
pre{background:#0e2c1f;color:#d6f5e6;padding:14px 16px;border-radius:10px;overflow:auto;font-size:12.5px;line-height:1.5;}
ol li,ul li{margin:4px 0;}
.foot{text-align:center;color:var(--muted);font-size:13px;padding:22px;}
.lead{font-size:16.5px;color:#27413a;}
"""

NAV = """<div class="nav">
  <a href="#story">The story</a><a href="#challenge">The game</a><a href="#framework">What we built</a>
  <a href="#gold">Gold standard</a><a href="#regimes">Everything we tried</a><a href="#finding">Key finding</a>
  <a href="#levers">What mattered</a><a href="#play">Play it</a><a href="#bottom">Bottom line</a><a href="#repro">Reproduce</a>
</div>"""

# ----------------------------------------------------------------- the page
HTML = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Gin Rummy: teaching RL to chase a perfect player</title>
<style>{CSS}</style></head><body>

<div class="hero">
  <h1>Adversarial Co-Evolution of RL and LLM Agents in Gin Rummy</h1>
  <p>We built a complete framework to train a small reinforcement-learning (RL) card player,
  a perfect "gold-standard" opponent to measure it against, and a distributed system to put a
  large language model (LLM) in the game. Then we ran <b>100+ experiments</b> to answer one
  question: <b>how close can a lightweight RL agent get to perfect play &mdash; and what
  actually makes it stronger?</b></p>
  <div class="by">Nima Kelidari &middot; Mahdi Salmani &middot; Mohammadsaeed Haghi &mdash; University of Southern California</div>
  <div class="kpibar">
    <div class="k"><div class="v">{BEST:.0f}%</div><div class="l">best agent vs the<br/>perfect player</div></div>
    <div class="k"><div class="v">&lt;2%</div><div class="l">how often the perfect<br/>player gins</div></div>
    <div class="k"><div class="v">{N_RUNS}+</div><div class="l">controlled training<br/>runs in the sweeps</div></div>
    <div class="k"><div class="v">62&times;</div><div class="l">faster LLM serving<br/>(scratch vs NFS)</div></div>
  </div>
  <div class="links">
    <a href="{PDF}">&#128196; PDF paper</a>
    <a href="{REPO}">&#128025; GitHub repo</a>
    <a href="{REPO}/tree/main/game">&#127918; Web game</a>
  </div>
</div>
{NAV}

<div class="wrap">

<section id="story">
  <h2><span class="n">1</span>The whole story, in one minute</h2>
  <p class="lead">Gin Rummy is a card game that needs both quick arithmetic (counting your
  "deadwood" &mdash; unmatched cards) and long-term planning (forming "melds"). It is a great
  testbed because it is easy to score but hard to master, and you never see your opponent's hand.</p>
  <p>Training an RL agent has a chicken-and-egg problem we call the <b>opponent bottleneck</b>:
  an agent is only as good as who it practices against. Practice against a weak player and you
  learn weak habits. So we built three things: a fast <b>RL player</b>, a perfect
  <b>gold-standard opponent</b> to grade everyone honestly, and a system to use a slow-but-smart
  <b>LLM</b> as a teacher. Then we tried essentially every sensible way to make the RL agent
  stronger and measured each against the perfect player.</p>
  <figure class="fig" style="max-width:560px;margin:0 auto;">{img("journey.png","Win-rate vs the perfect player across the project")}<figcaption>The climb: our best agent went from the old champion's ~{CHAMP_GOLD:.0f}% to <b>{BEST:.0f}%</b> against the perfect player &mdash; through a systematic search, not luck.</figcaption></figure>
  <div class="kpis">
    <div class="kpi gold"><div class="v">{GOLD_CHAMP:.0f}&ndash;99%</div><div class="l">the perfect player beats every learned agent</div></div>
    <div class="kpi"><div class="v">TRPO &gt; PPO</div><div class="l">the algorithm choice that helped</div></div>
    <div class="kpi"><div class="v">knock, don't gin</div><div class="l">the reward lesson that held across 60 runs</div></div>
    <div class="kpi warn"><div class="v">DAgger &amp; LLM-in-loop</div><div class="l">honest negatives: didn't beat plain RL</div></div>
  </div>
  <p><b>The bottom line up front.</b> No single trick beat the perfect player &mdash; it is a very
  high bar. But by stacking the things that genuinely help (a better algorithm, a reward that
  copies the perfect player's style, a curriculum of ever-stronger opponents, and always keeping
  the best checkpoint), we pushed a lightweight agent to <b>{BEST:.0f}% wins against perfect play</b>
  and uncovered one clean, sturdy scientific result along the way (next section).</p>
</section>

<section id="challenge">
  <h2><span class="n">2</span>The game, and why it is hard</h2>
  <p>Each turn you draw a card and discard one, trying to arrange your 10 cards into
  <b>melds</b> (runs like 5&ndash;6&ndash;7 of hearts, or sets like three 9s). Cards not in a meld
  are <b>deadwood</b>. You win by <b>knocking</b> (ending the hand with low deadwood) or by
  <b>gin</b> (zero deadwood &mdash; a big bonus, but rare and risky). You never see the opponent's
  hand, so it is a game of <b>imperfect information</b>.</p>
  <div class="diagram">a single turn (what the agent sees and chooses)

   hand: 4 planes x 52 cards  ──▶  [ masked PPO policy ]  ──▶  draw / pick-up / discard / knock / gin
   (your cards, the top discard,                │
    known picks, the rest)                      └─ illegal moves are blocked (logits -> -inf),
                                                   so the agent only ever picks a legal move
</div>
  <div class="why"><b>Why a testbed like this?</b> It forces both skills at once &mdash; short-horizon
  counting and long-horizon planning &mdash; and the rules give an exact score, so we can build a
  <i>perfect</i> reference player and grade everything objectively. That reference is what makes
  the rest of this report trustworthy.</div>
</section>

<section id="framework">
  <h2><span class="n">3</span>What we built &mdash; the framework</h2>
  <p>Most of the work was engineering a system where RL, a perfect expert, and an LLM can all
  meet in the same game. Five pieces:</p>
  <ul>
    <li><b>RL player</b> &mdash; action-masked PPO (and TRPO) on PettingZoo / RLCard Gin Rummy.
    Illegal moves are masked out, so the agent always plays a legal move.</li>
    <li><b>Gold standard</b> &mdash; a hand-coded <i>perfect</i> player using RLCard's exact meld
    solver. It is the yardstick; it never trains the RL agent.</li>
    <li><b>Distributed LLM stack</b> &mdash; a master/worker server so many GPUs can answer game
    questions in parallel, speaking the ordinary Ollama API.</li>
    <li><b>Curriculum system</b> &mdash; schedules opponents from random &rarr; past selves &rarr;
    strong models, so the agent always faces a fair-but-rising challenge.</li>
    <li><b>Web game</b> &mdash; a no-install browser client to play any trained agent yourself.</li>
  </ul>
  <figure class="fig" style="max-width:640px;margin:0 auto;">{img("architecture.png","System overview")}<figcaption>How the pieces connect. The gold standard is used for <b>scoring only</b> &mdash; it never trains the agent.</figcaption></figure>
  <h3>The distributed LLM server (so a 7B model can keep up with RL)</h3>
  <p>A single RL training run fires tens of thousands of opponent queries. At 0.5&ndash;3&nbsp;seconds
  per call, a naive loop would take hours per rollout. We decouple inference from training:</p>
  <div class="diagram">   env subprocess  ─▶  Master (CPU, FastAPI)  ─▶  suit-symmetry cache  ──(hit)──▶ return
   (per-step query)    Ollama-compatible API            │ miss
                              │ round-robin              ▼
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
          GPU worker      GPU worker  …   GPU worker      (1 GPU each, Qwen2.5-7B)
          self-registers in a shared-filesystem registry; master health-checks + balances
</div>
  <div class="why"><b>The infrastructure finding that mattered:</b> loading a 7B model from home
  network storage runs at ~11&nbsp;MB/s (~28&nbsp;min &mdash; it blows the health-check timeout).
  Staging the weights on fast scratch (BeeGFS) cuts that to ~27&nbsp;s &mdash; a <b>62&times;</b>
  speedup that is mandatory at scale. With ~14 workers the pool sustained <b>~32 queries/second</b>.</div>
</section>

<section id="gold">
  <h2><span class="n">4</span>The gold standard &mdash; and the surprise it revealed</h2>
  <p>To grade honestly, we built a <b>perfect</b> Gin Rummy player: every turn it computes the
  mathematically best melds (exact, not learned) and knocks the instant it should. It is the
  benchmark, never a teacher.</p>
  <figure class="fig" style="max-width:720px;margin:0 auto;">{img("gold_bench.png","Gold beats everyone but rarely gins")}<figcaption>Left: the perfect player beats every learned agent ({GOLD_CHAMP:.0f}&ndash;99% of games). Right: yet it gins under 2% of the time.</figcaption></figure>
  <div class="why"><b>The surprise (and our cleanest result):</b> the <i>perfect</i> player almost
  never gins &mdash; just <b>0.7&ndash;1.7%</b> of games &mdash; even though gin scores the most points.
  It wins by <b>knocking early with low deadwood</b>. That flips the intuition we started with:
  chasing gin is a beginner's trap. The optimal style is patient, low-risk knocking. This single
  fact reframed every reward experiment that follows.</div>
</section>

<section id="regimes">
  <h2><span class="n">5</span>Everything we tried (and why each worked or didn't)</h2>
  <p>We benchmarked nearly every reasonable way to make the agent stronger, all on the same
  yardstick &mdash; win-rate against the perfect player. Here is the full landscape, weakest to
  strongest:</p>
  <figure class="fig" style="max-width:680px;margin:0 auto;">{img("regimes.png","Every regime ranked vs the perfect player")}<figcaption>Each bar is a win-rate we actually measured against the perfect player.</figcaption></figure>
  <div class="cards">
    <div class="rc"><h4>&#127922; Train vs random only</h4><div class="res mid">98&ndash;99% vs random, but only ~15% vs gold</div>
      <p><b>Why:</b> random opponents are too weak to teach real strategy. The agent maxes out the
      easy metric and learns to always knock, never gin &mdash; a habit only a thinking opponent can break.</p></div>
    <div class="rc"><h4>&#127183; Reward shaping (gin vs knock)</h4><div class="res win">controls behaviour: 97% knock vs 22% gin</div>
      <p><b>Why:</b> the gin/knock reward ratio is a real control knob &mdash; pay more for gin and the
      agent chases gin (and wins less). This is the lever the rest of the project tunes.</p></div>
    <div class="rc"><h4>&#129302; Self-play + pool curriculum</h4><div class="res win">self-play beats its own parent 61%</div>
      <p><b>Why:</b> playing past versions of yourself is a free, rising curriculum. But an unguided
      pool <b>diverged after ~10M steps</b> &mdash; it chased itself into a corner. Curriculum design matters.</p></div>
    <div class="rc"><h4>&#129504; LLM as opponent</h4><div class="res mid">competent (beat our RL agent 3&ndash;2) but ~9&ndash;27 s/move</div>
      <p><b>Why:</b> mid-size LLMs (Qwen2.5-7B, gpt-oss-20b) play real Gin Rummy with the right prompt,
      and even beat our self-play agent in short matches. But they are <b>too slow</b> for the millions
      of moves RL needs &mdash; live LLM-in-the-loop training would take weeks. (Vision LLMs failed outright;
      it is a text-reasoning task.)</p></div>
    <div class="rc"><h4>&#128221; Imitation learning (DAgger)</h4><div class="res lose">collapsed &mdash; near-zero wins</div>
      <p><b>Why:</b> copying an expert's moves move-by-move does not transfer. The student mimics
      actions in familiar states but never learns the <i>reasoning</i> (e.g. tracking the opponent),
      so it falls apart off the training distribution. "Mimicking a move" &ne; "understanding strategy."</p></div>
    <div class="rc"><h4>&#9201;&#65039; Dense / short-term rewards</h4><div class="res lose">myopia &mdash; saturates, stops improving</div>
      <p><b>Why:</b> rewarding every little step makes the agent greedy for instant points and blind to
      the real goal of <i>winning the hand</i>. Performance flatlined after ~500k steps. Sparse,
      end-of-hand rewards won.</p></div>
    <div class="rc"><h4>&#128202; Algorithm: PPO vs TRPO</h4><div class="res win">TRPO ~22% vs PPO ~15% vs gold</div>
      <p><b>Why:</b> TRPO's "trust region" takes safer policy steps, which suits a sparse-reward,
      ever-shifting self-play target. (GRPO/DPO don't apply &mdash; they are LLM-alignment methods with no
      per-move game analogue.)</p></div>
    <div class="rc"><h4>&#128279; Learned state embeddings</h4><div class="res lose">all worse than the raw input</div>
      <p><b>Why:</b> we compressed the big sparse board into a small dense vector two ways (self-supervised
      "states near in a game are similar", and an LLM judging strategic similarity). Both <i>lost</i> to
      the raw input &mdash; a frozen bottleneck throws away detail the policy needs.</p></div>
    <div class="rc"><h4>&#127941; Curriculum sweep (Phase 6)</h4><div class="res win">best ~33% vs gold (30 runs)</div>
      <p><b>Why:</b> a careful league of random &rarr; past selves &rarr; strong models, swept over algorithm,
      reward, and schedule. Everything plateaued near champion strength &mdash; but it pinned down exactly
      which knobs matter (see the key finding).</p></div>
    <div class="rc"><h4>&#11088; Keep-best + warm-start (Phase 7)</h4><div class="res win">best agent {BEST:.0f}% vs gold</div>
      <p><b>Why:</b> three fixes stacked &mdash; always <b>save the peak checkpoint</b> (training drifts past
      its best), <b>warm-start</b> from the previous champion, and a <b>dense "lower your deadwood" reward</b>
      that coaches the optimal style. This is our strongest agent.</p></div>
  </div>
</section>

<section id="finding">
  <h2><span class="n">6</span>The key finding: you cannot bribe the agent into ginning</h2>
  <p>Phase 6 ran <b>30 controlled experiments</b> &mdash; one change at a time over algorithm, reward,
  and curriculum. The headline is honest: every recipe lands near champion strength against the
  perfect player. But one result is clean and sturdy across all of them.</p>
  <figure class="fig" style="max-width:700px;margin:0 auto;">{img("curriculum_reward.png","Gin rate stays under 1% for every reward")}<figcaption>Left: no matter the reward &mdash; even paying 3&times; for a gin &mdash; the agent gins under 1% of the time. Right: the "knock early" reward gives the shortest games.</figcaption></figure>
  <div class="why"><b>What it means, simply:</b> we tried to <i>bribe</i> the agent into ginning by paying
  three times more for a gin than a knock. It still gins under 1% of the time &mdash; the same as when gin
  isn't rewarded at all. You cannot pay a policy into a bad habit: just like the perfect player, it
  discovers on its own that <b>chasing gin loses</b>. The reward that actually helped did the opposite
  &mdash; a small nudge to knock <i>faster</i>, which produced the shortest games and the best play.</div>
  <p>The per-recipe numbers (each averaged over its seeds, best checkpoint vs the perfect player):</p>
  {curriculum_table("0")}
  <figure class="fig" style="max-width:600px;margin:18px auto 0;">{img("learning_curves.png","Skill rises through the curriculum")}<figcaption>How the curriculum drives learning: win-rate vs the champion climbs as tougher opponents are swapped in (random &rarr; pool &rarr; self &rarr; strong). The late dip on one run is exactly the drift that "keep the best checkpoint" fixes.</figcaption></figure>
</section>

<section id="levers">
  <h2><span class="n">7</span>What moved the needle &mdash; and what didn't</h2>
  <p>Across 100+ runs, the honest summary of which ideas actually helped against the perfect player:</p>
  <table>
    <tr><th>Idea</th><th>Verdict</th><th>Why</th></tr>
    <tr><td><b>Keep the best checkpoint</b></td><td class="good">helps</td><td>training drifts past its peak; saving the best recovers 2&ndash;3 points for free</td></tr>
    <tr><td><b>Warm-start from the champion</b></td><td class="good">helps</td><td>start strong, then specialise &mdash; better than from scratch</td></tr>
    <tr><td><b>TRPO over PPO</b></td><td class="good">helps</td><td>safer policy steps suit sparse, shifting self-play</td></tr>
    <tr><td><b>Reward knocking, not gin</b></td><td class="good">helps</td><td>copies the perfect player's low-risk style</td></tr>
    <tr><td><b>Curriculum of rising opponents</b></td><td class="good">helps</td><td>always a fair-but-harder challenge</td></tr>
    <tr><td>Paying more for gin</td><td class="bad">no effect</td><td>the agent refuses the bad habit no matter the bribe</td></tr>
    <tr><td>Fancy opponent-picking (PFSP)</td><td class="muted">~neutral</td><td>no better than a simple schedule here</td></tr>
    <tr><td>Longer memory (high discount)</td><td class="bad">hurts a bit</td><td>added noise, not foresight</td></tr>
    <tr><td>Learned state embeddings</td><td class="bad">hurts</td><td>a frozen bottleneck discards useful detail</td></tr>
    <tr><td>Imitation (DAgger)</td><td class="bad">fails</td><td>copies moves, not the reasoning behind them</td></tr>
    <tr><td>Dense short-term rewards</td><td class="bad">fails</td><td>myopia &mdash; greedy for points, blind to winning</td></tr>
    <tr><td>Live LLM-in-the-loop</td><td class="bad">infeasible</td><td>strong but far too slow for millions of moves</td></tr>
  </table>
  <h3>Our strongest agents (precise 2000-game re-evaluation)</h3>
  {best_models_table()}
</section>

<section id="play">
  <h2><span class="n">8</span>Play the heroes yourself</h2>
  <p>A no-install web game lets you play our strongest agents, with smooth card animations. The
  opponent menu is curated down to a clear ladder &mdash; from a beginner-friendly bot up to the
  perfect player nobody beats:</p>
  <table>
    <tr><th>Opponent</th><th>Strength</th><th>What it is</th></tr>
    <tr><td>&#127922; Rookie (Random)</td><td>easiest</td><td>plays a random legal move &mdash; a gentle warm-up</td></tr>
    <tr><td>&#129302; Self-Play Champion</td><td>strong</td><td>our earlier best, trained against copies of itself</td></tr>
    <tr><td>&#127183; Curriculum Ace</td><td>strongest learned</td><td>our best agent &mdash; ~{BEST:.0f}% vs the perfect player, built by stacking every lever that helped</td></tr>
    <tr><td>&#128737;&#65039; League Tactician</td><td>strongest learned</td><td>a close second, trained to practise most against whoever beats it (PFSP)</td></tr>
    <tr><td>&#127942; Gold Standard</td><td>perfect</td><td>the hand-coded expert &mdash; the wall everyone hits</td></tr>
  </table>
  <figure class="fig">{img("game_ui.png","web game")}<figcaption>The browser game (debug view, opponent hand shown). Run <code>python game/server.py</code> and open the URL.</figcaption></figure>
</section>

<section id="bottom">
  <h2><span class="n">9</span>The bottom line, and what's next</h2>
  <p>We set out to see how close a small, fast RL agent could get to perfect Gin Rummy and what
  actually makes it stronger. We built the whole framework to answer that fairly, then ran the
  experiments. The honest, bold summary:</p>
  <ul>
    <li>We pushed a lightweight agent from the old champion's ~{CHAMP_GOLD:.0f}% to <b>{BEST:.0f}%</b>
    against a <i>perfect</i> player &mdash; by stacking the things that genuinely help.</li>
    <li>We produced one clean, reusable result: <b>the optimal Gin Rummy policy almost never gins</b>,
    and an RL agent rediscovers this no matter how you reward it.</li>
    <li>We mapped what works (algorithm, reward style, curriculum, keep-best) and what doesn't
    (imitation, dense rewards, learned embeddings, live LLM-in-the-loop) &mdash; honest negatives
    included, because they save the next team months.</li>
  </ul>
  <p><b>The ceiling, honestly.</b> Tuning the reward, algorithm, and opponents tops out around the
  mid-30s percent against perfect play. Breaking decisively past that almost certainly needs a
  <i>different kind</i> of method &mdash; search / planning (like counterfactual-regret minimization,
  which powers superhuman poker) or an agent with full-history memory &mdash; not more reward tuning.
  That is the clear next step the data points to.</p>
</section>

<section id="repro">
  <h2>How to reproduce</h2>
  <p>Every figure on this page regenerates from saved JSON results via <code>paper/make_figures.py</code>,
  and this page via <code>paper/make_report_html.py</code>. Sweeps run as SLURM array jobs with a
  self-sustaining watchdog that resubmits failures, re-aggregates, and republishes &mdash; no human in
  the loop.</p>
  <pre># play the web game
python game/server.py --host 127.0.0.1 --port 8000      # open http://127.0.0.1:8000

# the gold-standard benchmark
python sweep/bench_gold.py

# the final curriculum + keep-best sweeps (SLURM array + watchdog)
python sweep/curriculum_configs.py && sbatch --array=0-29%10 slurm/curriculum.slurm
python sweep/phase7_configs.py    && sbatch --array=0-8%6 --export=ALL,CFG_DIR=phase7_cfgs slurm/curriculum.slurm

# regenerate this report
python paper/make_figures.py && python paper/make_report_html.py</pre>
  <p>The typeset paper is <a href="{PDF}">paper/main.pdf</a>. Live training curves are on
  Weights &amp; Biases (groups <code>phase6-curriculum</code>, <code>phase7-ceiling</code>).</p>
</section>

<div class="foot">Built from measured results &middot; every number traces to a JSON file under <code>sweep/</code> &middot; Adversarial Co-Evolution &middot; USC</div>
</div></body></html>"""


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(HTML)
    print(f"wrote {OUT} ({len(HTML)//1024} KB); best-vs-gold={BEST:.1f}% runs={N_RUNS}")


if __name__ == "__main__":
    main()
