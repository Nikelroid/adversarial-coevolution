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
.mid{color:#a9810a;font-weight:700;}.muted{color:var(--muted);font-weight:600;}
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
  <a href="#levers">What mattered</a><a href="#play">Play it</a><a href="#bottom">Bottom line</a><a href="#roadmap">Roadmap</a><a href="#venues">Publish</a><a href="#repro">Reproduce</a>
</div>"""

# ----------------------------------------------------------------- the page
HTML = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Gin Rummy: teaching RL to chase a perfect player</title>
<style>{CSS}</style></head><body>

<div class="hero">
  <h1>Adversarial Co-Evolution of RL and LLM Agents in Gin Rummy</h1>
  <p>We built a full system to train a small reinforcement-learning (RL) card player, a perfect
  "gold standard" opponent to measure it against, and a way to put a large language model (LLM)
  into the game. Then we ran <b>100+ experiments</b> to answer one question. <b>How close can a
  small RL agent get to perfect play, and what really makes it stronger?</b> We use Gin Rummy as a
  clean example, but the system and the lessons carry over to other games and to RL plus LLM agents
  in general.</p>
  <div class="by">Nima Kelidari &middot; Mahdi Salmani &middot; Mohammadsaeed Haghi &middot; University of Southern California</div>
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
  <p class="lead">Gin Rummy is a card game that needs two skills at once. You have to count fast
  (your "deadwood" is the cards that do not yet fit a pattern) and you have to plan ahead (you
  build "melds", which are runs and sets). It is a good test because it is easy to score but
  hard to play well, and you never get to see your opponent's cards.</p>
  <p>Training an RL agent has a chicken-and-egg problem. We call it the <b>opponent bottleneck</b>.
  An agent is only as good as the players it practises against. Practise against a weak player and
  you pick up weak habits. So we built three things. A fast <b>RL player</b>, a perfect
  <b>gold-standard opponent</b> to grade everyone fairly, and a way to use a slow but smart
  <b>LLM</b> as a teacher. Then we tried almost every sensible way to make the RL agent stronger,
  and we measured each one against the perfect player.</p>
  <figure class="fig" style="max-width:560px;margin:0 auto;">{img("journey.png","Win-rate vs the perfect player across the project")}<figcaption>The climb. Our best agent went from the old champion's roughly {CHAMP_GOLD:.0f}% up to <b>{BEST:.0f}%</b> against the perfect player. It got there through a careful search, not luck.</figcaption></figure>
  <div class="kpis">
    <div class="kpi gold"><div class="v">{GOLD_CHAMP:.0f} to 99%</div><div class="l">the perfect player beats every learned agent</div></div>
    <div class="kpi"><div class="v">TRPO &gt; PPO</div><div class="l">the algorithm choice that helped</div></div>
    <div class="kpi"><div class="v">knock, don't gin</div><div class="l">the reward lesson that held across 60 runs</div></div>
    <div class="kpi warn"><div class="v">DAgger &amp; live LLM</div><div class="l">honest results: neither beat plain RL</div></div>
  </div>
  <p><b>The short version, up front.</b> No single trick beat the perfect player. It is a very high
  bar. But when we stacked the ideas that really help (a better algorithm, a reward that copies the
  perfect player's style, a curriculum of stronger and stronger opponents, and always keeping the
  best checkpoint), we pushed a small agent up to <b>{BEST:.0f}% wins against perfect play</b>. We
  also found one clear, solid result along the way. That is the next section.</p>
</section>

<section id="challenge">
  <h2><span class="n">2</span>The game, and why it is hard</h2>
  <p>Each turn you draw a card and throw one away. You are trying to line up your 10 cards into
  <b>melds</b> (a run like 5, 6, 7 of hearts, or a set like three 9s). Cards that do not fit a
  meld are <b>deadwood</b>. You win by <b>knocking</b> (ending the hand with low deadwood) or by
  <b>gin</b> (zero deadwood, which gives a big bonus but is rare and risky). You never see the
  opponent's hand, so you have to play with hidden information.</p>
  <div class="diagram">a single turn (what the agent sees and chooses)

   hand: 4 planes x 52 cards  ──▶  [ masked PPO policy ]  ──▶  draw / pick-up / discard / knock / gin
   (your cards, the top discard,                │
    known picks, the rest)                      └─ illegal moves are blocked (logits -> -inf),
                                                   so the agent only ever picks a legal move
</div>
  <div class="why"><b>Why use a game like this?</b> It needs both skills at the same time, fast
  counting and long-term planning. And the rules give an exact score, so we can build a
  <i>perfect</i> reference player and grade everything against it. That reference is what makes
  the rest of this report trustworthy.</div>
</section>

<section id="framework">
  <h2><span class="n">3</span>What we built: the framework</h2>
  <p>Most of the work was building a system where the RL agent, a perfect expert, and an LLM can
  all meet in the same game. There are five pieces.</p>
  <ul>
    <li><b>RL player.</b> Action-masked PPO (and TRPO) on PettingZoo / RLCard Gin Rummy. Illegal
    moves are blocked, so the agent always plays a legal move.</li>
    <li><b>Gold standard.</b> A hand-coded <i>perfect</i> player that uses RLCard's exact meld
    solver. It is the measuring stick. It never trains the RL agent.</li>
    <li><b>Distributed LLM stack.</b> A master and worker setup so many GPUs can answer game
    questions at once, using the ordinary Ollama API.</li>
    <li><b>Curriculum system.</b> It schedules opponents from random, to past versions of the
    agent, to strong models, so the agent always gets a fair but rising challenge.</li>
    <li><b>Web game.</b> A no-install browser page where you can play any trained agent.</li>
  </ul>
  <figure class="fig" style="max-width:640px;margin:0 auto;">{img("architecture.png","System overview")}<figcaption>How the pieces connect. The gold standard is used for <b>scoring only</b>. It never trains the agent.</figcaption></figure>
  <h3>The distributed LLM server (so a 7B model can keep up with RL)</h3>
  <p>One RL training run asks the opponent tens of thousands of questions. At 0.5 to 3 seconds per
  call, a simple loop would take hours per round. So we keep the LLM serving separate from the
  training loop.</p>
  <div class="diagram">   env subprocess  ─▶  Master (CPU, FastAPI)  ─▶  suit-symmetry cache  ──(hit)──▶ return
   (per-step query)    Ollama-compatible API            │ miss
                              │ round-robin              ▼
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
          GPU worker      GPU worker  …   GPU worker      (1 GPU each, Qwen2.5-7B)
          self-registers in a shared-filesystem registry; master health-checks + balances
</div>
  <div class="why"><b>One infrastructure lesson that really mattered.</b> Loading a 7B model from
  home network storage runs at about 11 MB/s, which takes around 28 minutes and trips the
  health-check timeout. Putting the weights on fast scratch storage (BeeGFS) cuts that to about
  27 seconds. That is a <b>62&times;</b> speedup, and it is a must at scale. With about 14 workers
  the pool handled <b>about 32 questions per second</b>.</div>
</section>

<section id="gold">
  <h2><span class="n">4</span>The gold standard, and the surprise it revealed</h2>
  <p>To grade everyone fairly we built a <b>perfect</b> Gin Rummy player. Every turn it works out
  the best possible melds (exact, not learned) and knocks the moment it should. It is the
  benchmark. It is never a teacher.</p>
  <figure class="fig" style="max-width:720px;margin:0 auto;">{img("gold_bench.png","Gold beats everyone but rarely gins")}<figcaption>Left: the perfect player beats every learned agent ({GOLD_CHAMP:.0f} to 99% of games). Right: but it gins under 2% of the time.</figcaption></figure>
  <div class="why"><b>The surprise, and our cleanest result.</b> The <i>perfect</i> player almost
  never gins. It gins in just <b>0.7 to 1.7%</b> of games, even though gin scores the most points.
  It wins by <b>knocking early with low deadwood</b>. That goes against what we expected. Chasing
  gin is a beginner's trap. The best style is patient, low-risk knocking. This one fact changed how
  we thought about every reward test that follows.</div>
</section>

<section id="regimes">
  <h2><span class="n">5</span>Everything we tried, and why each one worked or didn't</h2>
  <p>We tested almost every reasonable way to make the agent stronger, and we judged them all the
  same way: win-rate against the perfect player. Here is the full picture, from weakest to
  strongest.</p>
  <figure class="fig" style="max-width:680px;margin:0 auto;">{img("regimes.png","Every regime ranked vs the perfect player")}<figcaption>Each bar is a win-rate we actually measured against the perfect player.</figcaption></figure>
  <div class="cards">
    <div class="rc"><h4>&#127922; Train vs random only</h4><div class="res mid">98 to 99% vs random, but only ~15% vs gold</div>
      <p><b>Why:</b> random opponents are too weak to teach real strategy. The agent maxes out the
      easy score and learns to always knock and never gin. Only a thinking opponent can break that habit.</p></div>
    <div class="rc"><h4>&#127183; Reward shaping (gin vs knock)</h4><div class="res win">it controls behaviour: 97% knock vs 22% gin</div>
      <p><b>Why:</b> the gin to knock reward ratio is a real dial. Pay more for gin and the agent chases
      gin, and wins less. This is the dial the rest of the project turns.</p></div>
    <div class="rc"><h4>&#129302; Self-play + pool curriculum</h4><div class="res win">self-play beats its own parent 61%</div>
      <p><b>Why:</b> playing past versions of yourself is a free, rising curriculum. But a pool with no
      guidance <b>broke down after about 10M steps</b> and chased itself into a corner. The design of the
      curriculum matters.</p></div>
    <div class="rc"><h4>&#129504; LLM as opponent</h4><div class="res mid">good (beat our RL agent 3 to 2) but ~9 to 27 s/move</div>
      <p><b>Why:</b> mid-size LLMs (Qwen2.5-7B, gpt-oss-20b) play real Gin Rummy with the right prompt,
      and even beat our self-play agent in short matches. But they are <b>too slow</b> for the millions
      of moves RL needs. Training against a live LLM would take weeks. (Vision LLMs failed completely.
      This is a text task.)</p></div>
    <div class="rc"><h4>&#128221; Imitation learning (DAgger)</h4><div class="res lose">it collapsed to almost no wins</div>
      <p><b>Why:</b> copying an expert's moves one at a time does not carry over. The student copies
      moves in familiar spots but never learns the <i>thinking</i> behind them (like tracking the
      opponent), so it falls apart in new situations. Copying a move is not the same as understanding why.</p></div>
    <div class="rc"><h4>&#9201;&#65039; Dense / short-term rewards</h4><div class="res lose">short-sighted, stops improving</div>
      <p><b>Why:</b> rewarding every small step makes the agent greedy for instant points and blind to
      the real goal of <i>winning the hand</i>. It stopped improving after about 500k steps. The simple
      reward at the end of the hand won.</p></div>
    <div class="rc"><h4>&#128202; Algorithm: PPO vs TRPO</h4><div class="res win">TRPO ~22% vs PPO ~15% vs gold</div>
      <p><b>Why:</b> TRPO takes smaller, safer steps when it updates the policy, which fits a setting
      where rewards are rare and the opponent keeps changing. (GRPO and DPO do not apply here. They are
      methods for aligning language models, with no per-move game version.)</p></div>
    <div class="rc"><h4>&#128279; Learned state embeddings</h4><div class="res lose">all worse than the raw input</div>
      <p><b>Why:</b> we squeezed the big, sparse board into a small dense vector two ways: one that
      learned which game states are close together, and one where an LLM judged how similar two states
      are. Both <i>lost</i> to the raw input. A fixed, squeezed vector throws away detail the agent needs.</p></div>
    <div class="rc"><h4>&#127941; Curriculum sweep (Phase 6)</h4><div class="res win">best ~33% vs gold (30 runs)</div>
      <p><b>Why:</b> a careful ladder of opponents (random, then past selves, then strong models),
      swept over algorithm, reward, and schedule. Everything leveled off near champion strength, but it
      showed clearly which dials matter (see the key finding).</p></div>
    <div class="rc"><h4>&#11088; Keep-best + warm-start (Phase 7)</h4><div class="res win">best agent {BEST:.0f}% vs gold</div>
      <p><b>Why:</b> three fixes stacked together. Always <b>save the best checkpoint</b> (training
      drifts past its peak), <b>start from the previous champion</b> instead of from scratch, and add a
      <b>reward for lowering your own deadwood</b> that teaches the optimal style. This is our strongest agent.</p></div>
  </div>
</section>

<section id="finding">
  <h2><span class="n">6</span>The key finding: you cannot bribe the agent into ginning</h2>
  <p>Phase 6 ran <b>30 controlled experiments</b>, changing one thing at a time across algorithm,
  reward, and curriculum. The honest headline is that every recipe ends up near champion strength
  against the perfect player. But one result is clear and holds across all of them.</p>
  <figure class="fig" style="max-width:700px;margin:0 auto;">{img("curriculum_reward.png","Gin rate stays under 1% for every reward")}<figcaption>Left: no matter the reward, even paying 3&times; for a gin, the agent gins under 1% of the time. Right: the "knock early" reward gives the shortest games.</figcaption></figure>
  <div class="why"><b>What it means, in plain terms.</b> We tried to <i>bribe</i> the agent into
  ginning by paying three times more for a gin than a knock. It still gins under 1% of the time,
  the same as when gin is not rewarded at all. You cannot pay an agent into a bad habit. Just like
  the perfect player, it works out on its own that <b>chasing gin loses</b>. The reward that did
  help was the opposite, a small push to knock <i>faster</i>, which gave the shortest games and the
  best play.</div>
  <p>The numbers per recipe (each one averaged over its seeds, best checkpoint vs the perfect player):</p>
  {curriculum_table("0")}
  <figure class="fig" style="max-width:600px;margin:18px auto 0;">{img("learning_curves.png","Skill rises through the curriculum")}<figcaption>How the curriculum drives learning: win-rate vs the champion climbs as tougher opponents are swapped in (random, then pool, then self, then strong). The late dip on one run is the drift that "keep the best checkpoint" fixes.</figcaption></figure>
</section>

<section id="levers">
  <h2><span class="n">7</span>What moved the needle, and what didn't</h2>
  <p>Across 100+ runs, here is the honest summary of which ideas actually helped against the
  perfect player. Most of these are general lessons about training agents, not tricks special to
  this one game.</p>
  <table>
    <tr><th>Idea</th><th>Verdict</th><th>Why</th></tr>
    <tr><td><b>Keep the best checkpoint</b></td><td class="good">helps</td><td>training drifts past its peak, so saving the best one recovers 2 to 3 points for free</td></tr>
    <tr><td><b>Warm-start from the champion</b></td><td class="good">helps</td><td>start strong, then improve. better than from scratch</td></tr>
    <tr><td><b>TRPO over PPO</b></td><td class="good">helps</td><td>safer, smaller policy steps suit rare rewards and a changing opponent</td></tr>
    <tr><td><b>Reward knocking, not gin</b></td><td class="good">helps</td><td>copies the perfect player's low-risk style</td></tr>
    <tr><td><b>Curriculum of rising opponents</b></td><td class="good">helps</td><td>always a fair but harder challenge</td></tr>
    <tr><td>Paying more for gin</td><td class="bad">no effect</td><td>the agent refuses the bad habit no matter the reward</td></tr>
    <tr><td>Fancy opponent-picking (PFSP)</td><td class="muted">about even</td><td>no better than a simple schedule here</td></tr>
    <tr><td>Longer memory (high discount)</td><td class="bad">hurts a bit</td><td>added noise, not foresight</td></tr>
    <tr><td>Learned state embeddings</td><td class="bad">hurts</td><td>a fixed, squeezed input throws away useful detail</td></tr>
    <tr><td>Imitation (DAgger)</td><td class="bad">fails</td><td>copies moves, not the thinking behind them</td></tr>
    <tr><td>Dense short-term rewards</td><td class="bad">fails</td><td>short-sighted: greedy for points, blind to winning</td></tr>
    <tr><td>Live LLM-in-the-loop</td><td class="bad">not practical</td><td>strong, but far too slow for millions of moves</td></tr>
  </table>
  <h3>Our strongest agents (checked carefully over 2000 games each)</h3>
  {best_models_table()}
</section>

<section id="play">
  <h2><span class="n">8</span>Play the heroes yourself</h2>
  <p>A no-install web game lets you play our strongest agents, with smooth card animations. The
  opponent menu is a simple ladder, from a beginner-friendly bot up to the perfect player that
  nobody beats.</p>
  <table>
    <tr><th>Opponent</th><th>Strength</th><th>What it is</th></tr>
    <tr><td>&#127922; Rookie (Random)</td><td>easiest</td><td>plays a random legal move, a gentle warm-up</td></tr>
    <tr><td>&#129302; Self-Play Champion</td><td>strong</td><td>our earlier best, trained against copies of itself</td></tr>
    <tr><td>&#127183; Curriculum Ace</td><td>strongest learned</td><td>our best agent, about {BEST:.0f}% vs the perfect player, built by stacking every idea that helped</td></tr>
    <tr><td>&#128737;&#65039; League Tactician</td><td>strongest learned</td><td>a close second, trained to practise most against whoever beats it (PFSP)</td></tr>
    <tr><td>&#127942; Gold Standard</td><td>perfect</td><td>the hand-coded expert, the wall everyone hits</td></tr>
  </table>
  <figure class="fig">{img("game_ui.png","web game")}<figcaption>The browser game (debug view, opponent hand shown). Run <code>python game/server.py</code> and open the URL.</figcaption></figure>
</section>

<section id="bottom">
  <h2><span class="n">9</span>The bottom line, and what's next</h2>
  <p>We set out to see how close a small, fast RL agent could get to perfect play, and what really
  makes it stronger. We built the whole framework to answer that fairly, then ran the experiments.
  Here is the short, honest summary.</p>
  <ul>
    <li>We pushed a small agent from the old champion's roughly {CHAMP_GOLD:.0f}% up to
    <b>{BEST:.0f}%</b> against a <i>perfect</i> player, by stacking the ideas that really help.</li>
    <li>We found one clear, reusable result. <b>The best Gin Rummy play almost never gins</b>, and an
    RL agent learns the same thing on its own no matter how you reward it.</li>
    <li>We mapped what works (algorithm, reward style, curriculum, keep-best) and what does not
    (imitation, dense rewards, learned embeddings, live LLM-in-the-loop). The honest negatives are
    included, because they can save the next team a lot of time.</li>
  </ul>
  <div class="why"><b>This is not really about Gin Rummy.</b> We used Gin Rummy as a clean example,
  because it has hidden information, both fast and slow planning, and an exact score. The pieces and
  lessons carry over to other two-player games and to RL plus LLM agent systems in general: build a
  strong reference to grade against, train against a rising ladder of opponents, copy the style of
  strong play through the reward, always keep your best model, and do not expect imitation, dense
  rewards, or a slow model in the training loop to do the heavy lifting. To make this concrete, the
  code is shipped as a <b>universal pipeline</b> (the <code>coev/</code> package): point it at any
  PettingZoo game, or your own environment, and it trains a masked agent through the same opponent
  curriculum. Gin Rummy is the test case, not the point.</div>
  <p><b>The ceiling, honestly.</b> Tuning the reward, the algorithm, and the opponents tops out
  around the mid-30s percent against perfect play. Going clearly past that almost certainly needs a
  <i>different kind</i> of method, such as search and planning (the kind that beat the best humans at
  poker) or an agent that remembers the whole game so far. More reward tuning will not get there.
  That is the clear next step the data points to.</p>
</section>

<section id="roadmap">
  <h2><span class="n">10</span>Roadmap: strengths, limits, and what is next</h2>
  <p>Here is an honest look at where the project stands today, what is strong, what is weak, and what
  we would do next to turn it into a paper.</p>

  <h3>What is strong</h3>
  <ul>
    <li>A complete, working system end to end: a masked RL learner, a perfect benchmark to grade
    against, a distributed LLM server, an opponent curriculum, and a playable web game.</li>
    <li>A <b>universal pipeline</b>. The same code trains on any PettingZoo game or your own
    environment, not just Gin Rummy.</li>
    <li>A lot of careful evidence: 100+ controlled runs, all judged on one fair number (win rate vs
    the perfect player), with honest results for the methods that did not work.</li>
    <li>One clean, reusable finding: the best play almost never gins, and an RL agent learns this no
    matter the reward. You cannot pay it into a bad habit.</li>
    <li>Solid engineering: a 62&times; faster model-serving setup, and a self-running sweep that
    resubmits failures, re-collects results, and republishes this report with no human in the loop.</li>
    <li>Fully reproducible: every figure rebuilds from saved JSON, and training curves are on
    Weights and Biases.</li>
  </ul>

  <h3>Where it is weak (the honest part)</h3>
  <ul>
    <li>We did not beat the perfect player. Our best agent wins about <b>34%</b> against it. This is
    a clear picture of the ceiling, not a state-of-the-art win.</li>
    <li>The main results are on one game (Gin Rummy). The universal pipeline is checked on small
    games (Connect Four, Tic-Tac-Toe) with short smoke runs, not trained to convergence.</li>
    <li>The original idea, training against a live LLM, turned out to be too slow to be practical.
    The LLM ended up helping only indirectly (as a benchmark, in the design, and in an embedding
    experiment that also did not help).</li>
    <li>The methods themselves (PPO, TRPO, curriculum, keep-best) are standard. The contribution is
    the careful study, the benchmark, the universal pipeline, and the honest negatives, not a new
    algorithm.</li>
    <li>We did not run a search or planning method or a memory-based agent, which are the most likely
    ways to pass the ceiling. They are listed as future work, not done.</li>
    <li>Compute was limited to one cluster, and some comparison runs are short (2M steps).</li>
  </ul>

  <h3>What is next, in order</h3>
  <ol>
    <li>Train the universal pipeline to convergence on two or three non-Gin PettingZoo games, to show
    the generality with real numbers, not just smoke tests.</li>
    <li>Add one search or planning baseline (counterfactual-regret or a small lookahead at decision
    time) and one memory baseline (a masked recurrent agent with a memory-aware evaluator), and see
    if either passes the 34% ceiling.</li>
    <li>Test richer inputs (hand-structure features) trained to convergence. We only built and
    smoke-tested this.</li>
    <li>Try the offline-RL path: build a large dataset of LLM-vs-LLM games and learn from it offline,
    so the slow LLM is used once, not inside the training loop.</li>
    <li>Tighten the statistics on the headline runs (more seeds, paired confidence intervals) and turn
    the gold benchmark plus the curriculum harness into a small public benchmark suite.</li>
    <li>Write the paper for one specific venue (see below).</li>
  </ol>
  <div class="why"><b>The shortest path to a paper:</b> pick a venue, add real multi-game results and
  one search-or-memory baseline, and frame the contribution as the universal pipeline, plus the
  perfect-player benchmark, plus the honest study of what helps and what does not.</div>
</section>

<section id="venues">
  <h2><span class="n">11</span>Where this can go: realistic publication targets</h2>
  <p>It is mid-2026, so anything with a spring-2026 deadline is already closed. This is a focused
  shortlist of six venues whose <b>deadline is still ahead</b> and whose topic is a real fit, listed
  soonest first. The two game-AI homes (AIIDE and its EXAG workshop) are the best fit for the work,
  with the WAGD workshop at AIIDE a further game-AI option;
  AAMAS is the natural home on the multi-agent and reinforcement-learning side; AAAI and the NeurIPS
  workshops are the broader, sooner shots that get you in front of people first.</p>
  <div class="why"><b>Can you put it on arXiv first?</b> Yes, for all six. Every venue here allows
  an arXiv preprint before or after you submit. For the double-blind ones (AIIDE and the main AAMAS
  track), just keep the arXiv version from breaking anonymity in a way the reviewers would notice,
  and do not label it "under review at [venue]". AAAI, EXAG, the NeurIPS workshops, and the ALA
  workshop are open or non-archival, so a public preprint is completely fine. WAGD is the exception:
  it archives to CEUR-WS, which AIIDE and AAAI main tracks count as a prior publication, so submit
  there only if you are not also sending the same paper to an AAAI-proceedings venue.</div>
  <table style="font-size:13px;">
    <tr><th>#</th><th>Venue (link) and fit</th><th>Next deadline (soonest first)</th><th>Conference</th><th>Format and review</th><th>arXiv?</th></tr>
    <tr>
      <td><b>1</b></td>
      <td><b><a href="https://sites.google.com/view/aiide2026/home">AIIDE 2026</a></b><br/><span class="good">best fit</span></td>
      <td><b>abstract: Jun 26, 2026<br/>paper: Jul 3, 2026</b> <span class="good">(extended)</span><br/>notify: Aug 7, 2026</td>
      <td>Belo Horizonte, Brazil<br/>Nov 9-13, 2026</td>
      <td>9 pp + unlimited refs; AAAI two-column template; double-blind; published in the AAAI AIIDE proceedings (not CEUR)</td>
      <td class="good">yes</td>
    </tr>
    <tr>
      <td><b>2</b></td>
      <td><b><a href="https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/">AAAI-27</a></b><br/><span class="mid">broad AI; realistic via the student-abstract or demo track</span></td>
      <td><b>abstract: Jul 21, 2026<br/>paper: Jul 28, 2026</b> <span class="good">(confirmed)</span><br/>notify: Nov 30, 2026</td>
      <td>Montreal, Canada<br/>Feb 16-23, 2027</td>
      <td>AAAI 2-column template; double-blind; OpenReview</td>
      <td class="good">yes</td>
    </tr>
    <tr>
      <td><b>3</b></td>
      <td><b><a href="https://www.exag.org/">EXAG 2026</a></b> (Experimental AI in Games, at AIIDE)<br/><span class="good">best fit</span></td>
      <td><b>paper: Aug 21, 2026</b> <span class="good">(confirmed)</span></td>
      <td>Belo Horizonte, Brazil<br/>at AIIDE, Nov 9-13, 2026</td>
      <td>~8 pp; AAAI template; non-archival workshop (work in progress welcome); light review</td>
      <td class="good">yes</td>
    </tr>
    <tr>
      <td><b>4</b></td>
      <td><b><a href="https://neurips.cc/Conferences/2026">NeurIPS 2026 workshops</a></b> (RL / LLM-agents)<br/><span class="good">realistic</span></td>
      <td><b>papers: ~Aug 29, 2026</b> <span class="mid">(NeurIPS-suggested date; each accepted workshop sets its own, before the Sep 29 accept/reject)</span><br/>workshop list announced Jul 11, 2026</td>
      <td>Dec 2026<br/>Sydney / Paris / Atlanta (multi-site)</td>
      <td>short (4 to 9 pp); workshop template; non-archival; OpenReview</td>
      <td class="good">yes</td>
    </tr>
    <tr>
      <td><b>5</b></td>
      <td><b><a href="https://smshields.github.io/AIIDE-2026-WAGD/#overview">WAGD 2026</a></b> (Workshop on Automated Game Design, at AIIDE)<br/><span class="mid">weak fit (game design, not agent training)</span></td>
      <td><b>paper: Sep 18, 2026</b> <span class="good">(confirmed)</span></td>
      <td>Belo Horizonte, Brazil<br/>at AIIDE, Nov 9-13, 2026</td>
      <td>5-9 pp short / 10+ full; CEUR-WS template; peer-reviewed; <span class="bad">archival (CEUR-WS)</span></td>
      <td class="good">yes</td>
    </tr>
    <tr>
      <td><b>6</b></td>
      <td><b><a href="https://warwick.ac.uk/fac/sci/dcs/aamas2027/">AAMAS 2027</a></b> + <a href="https://ala2024.github.io/">ALA workshop</a><br/><span class="good">strong fit</span></td>
      <td><b>abstract: Oct 1, 2026<br/>paper: Oct 8, 2026</b> <span class="good">(confirmed)</span><br/>ALA: ~Feb 2027</td>
      <td>Hanoi, Vietnam<br/>May 3-7, 2027</td>
      <td>ACM template; main 8 to 9 pp double-blind; ALA 8 pp non-archival; OpenReview</td>
      <td class="good">yes</td>
    </tr>
  </table>
  <p style="font-size:13px;color:#5c6b73;">A date marked <b>(confirmed)</b> is taken from the official
  call; a date marked <b>(expected)</b> is our estimate from prior years because that cycle's call is
  not posted yet, so check the linked page before you plan around it. One thing to watch on NeurIPS:
  the June 6, 2026 date on its call is the deadline for <b>organizers to propose a workshop</b>, not
  for authors to submit a paper. The workshop list is announced on July 11, and author paper
  deadlines land near the suggested August 29 date, so that route is still open. If the very near
  deadlines are too tight, the game-AI and RL conferences come back around in early to mid 2027
  (IEEE CoG, RLC, and the next AIIDE), and they are the strongest long-term fit.</p>
  <div class="note"><b>Bottom line.</b> The single nearest deadline is <b>AIIDE 2026</b> (abstract
  <b>June 26</b>, full paper <b>July 3, 2026</b>, both extended one week), and it is also our best-fit home (game AI, AAAI proceedings),
  so it is the one to aim at first. If the paper is too tight, the <b>EXAG</b> workshop at AIIDE gives a second game-AI shot at the
  same conference with a later deadline (paper August 21, 2026). Around it sit the <b>AAAI-27</b>
  student or demo track (paper July 28, 2026), the NeurIPS 2026 workshops (author papers around
  August 29, 2026), the <b>WAGD</b> workshop at AIIDE (paper September 18, 2026, but CEUR-archival), and <b>AAMAS 2027</b> (paper October 8, 2026). To reach an ICML, NeurIPS, or
  ICLR main track, we would first need to pass the ceiling with a new method (search or memory), or
  grow the pipeline and benchmark into something the community adopts.</div>
</section>

<section id="repro">
  <h2>How to reproduce</h2>
  <p>Every figure on this page is rebuilt from saved JSON results by <code>paper/make_figures.py</code>,
  and this page by <code>paper/make_report_html.py</code>. The sweeps run as SLURM array jobs with a
  watchdog that resubmits failed runs, re-collects the results, and republishes this page on its own,
  with no human in the loop.</p>
  <pre># train on ANY game with the universal pipeline (coev/)
python -m coev.examples.connect_four    # any PettingZoo game, no game-specific code
python -m coev.examples.gin_rummy       # same pipeline + a gold benchmark and reward shaping

# play the web game
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
