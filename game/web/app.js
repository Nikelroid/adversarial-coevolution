"use strict";

// Action ids (PettingZoo gin_rummy_v4). The server re-validates everything.
const A_DRAW = 2;
const A_TAKE = 3;
const A_DEAD = 4;
const A_GIN = 5;
const discardAction = (idx) => 6 + idx;
const knockAction = (idx) => 58 + idx;
const CARD_BACK = "/deck_images/card_back.png";

// animation timings (ms)
const T_DEAL_STAGGER = 55;
const T_FLY = 560;          // plain fly (deck/discard <-> hand/pile)
const T_TAKE = 760;         // taking the face-up discard into my hand (a touch slower)
const T_REVEAL = 1000;      // slow reveal flip (the 3 reveal cases)
const T_OPP_GAP = 140;      // gap between the opponent's draw and discard

let view = null;
let knockArmed = false;
let busy = false;
let lockMs = 0;             // how long to keep the board non-interactive after a render

// animation bookkeeping
let prevCardsByIdx = {};
let prevTopIdx = null;
let pendingDraw = null;     // {source:'stock'|'discard'}
let pendingDiscard = null;  // {idx}
let dealNext = false;
let renderGen = 0;

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const center = (r) => ({ x: r.left + r.width / 2, y: r.top + r.height / 2 });

// ---------------------------------------------------------------- networking
async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

function setPending(action) {
  pendingDraw = null;
  pendingDiscard = null;
  if (action === A_DRAW) pendingDraw = { source: "stock" };
  else if (action === A_TAKE) pendingDraw = { source: "discard" };
  else if (action >= 6 && action <= 57) pendingDiscard = { idx: action - 6 };
  else if (action >= 58 && action <= 109) pendingDiscard = { idx: action - 58 };
}

async function newGame() {
  if (busy) return;
  knockArmed = false;
  pendingDraw = null;
  pendingDiscard = null;
  dealNext = true;
  const key = selectedOpponent();
  await run(() => api("POST", "/api/new_game", key ? { opponent: key } : {}));
}

async function send(action) {
  if (busy) return;
  knockArmed = false;
  setPending(action);
  await run(() => api("POST", "/api/action", { action }));
}

// Keep the board non-interactive until the move's animation finishes, so a
// draw/take can't happen mid-discard (for me or the opponent).
async function run(fn) {
  busy = true;
  document.body.classList.add("locked");
  try {
    lockMs = 0;
    render(await fn());
    if (lockMs > 0) await sleep(lockMs);
  } catch (e) {
    flash(e.message);
  } finally {
    busy = false;
    document.body.classList.remove("locked");
  }
}

function flash(msg) {
  const m = $("message");
  m.textContent = msg;
  m.classList.add("flash");
  setTimeout(() => m.classList.remove("flash"), 1200);
}

// ---------------------------------------------------------------- opponents
function selectedOpponent() {
  const sel = $("opponent-select");
  return sel && sel.value ? sel.value : undefined;
}

async function loadOpponents() {
  const list = await api("GET", "/api/opponents");
  const sel = $("opponent-select");
  sel.innerHTML = "";
  list.forEach((o) => {
    const opt = document.createElement("option");
    opt.value = o.key;
    opt.textContent = `${o.label} · ${o.stat}`;
    sel.appendChild(opt);
  });
  sel.onchange = () => newGame();
}

// ---------------------------------------------------------------- card DOM
function cardEl(card, small) {
  const d = document.createElement("div");
  d.className = "card" + (small ? " small" : "");
  d.style.backgroundImage = `url(/deck_images/${card.img})`;
  d.title = card.label;
  d.dataset.idx = card.idx;
  return d;
}

function backEl() {
  const d = document.createElement("div");
  d.className = "card small back";
  d.style.backgroundImage = `url(${CARD_BACK})`;
  return d;
}

function setDiscImg(card) {
  const d = $("discard");
  if (card) {
    d.style.backgroundImage = `url(/deck_images/${card.img})`;
    d.textContent = "";
    d.title = card.label;
    d.classList.remove("empty-label");
  } else {
    d.style.backgroundImage = "none";
    d.textContent = "empty";
    d.classList.add("empty-label");
  }
}

// ---------------------------------------------------------------- animation
// Reflow: slide a card from its previous rect to its current slot (no rotation).
function flip(el, fromRect) {
  try {
    const to = el.getBoundingClientRect();
    const dx = fromRect.left - to.left;
    const dy = fromRect.top - to.top;
    if (Math.abs(dx) < 1 && Math.abs(dy) < 1) return;
    el.style.transition = "none";
    el.style.transform = `translate(${dx}px, ${dy}px)`;
    el.getBoundingClientRect();
    requestAnimationFrame(() => {
      el.style.transition = "transform .42s cubic-bezier(.2,.85,.25,1)";
      el.style.transform = "";
      setTimeout(() => { el.style.transition = ""; }, 440);
    });
  } catch (e) { /* end-state already correct */ }
}

// Plain fly-in on the REAL card (no rotation): travels from a source pile.
function flyOnlyCard(el, srcRect, delay, dur) {
  try {
    const r = el.getBoundingClientRect();
    const dx = (srcRect.left + srcRect.width / 2) - (r.left + r.width / 2);
    const dy = (srcRect.top + srcRect.height / 2) - (r.top + r.height / 2);
    el.style.setProperty("--dx", dx + "px");
    el.style.setProperty("--dy", dy + "px");
    el.style.setProperty("--flydur", (dur || T_FLY) + "ms");
    el.style.animationDelay = (delay || 0) + "ms";
    el.classList.add("fly-only");
  } catch (e) { /* end-state already correct */ }
}

// A free-flying card travelling the FULL path between two points, morphing size
// from (fw,fh) to (tw,th). reveal=true -> two-faced back->front turn (slow).
function flyCard(o) {
  try {
    const fw = o.fw || 74, fh = o.fh || 104;
    const tw = o.tw || fw, th = o.th || fh;
    const dur = o.slow ? T_REVEAL : (o.dur || T_FLY);
    const fc = center(o.from), tc = center(o.to);
    const g = document.createElement("div");
    g.className = "fly-card";
    g.style.left = (fc.x - fw / 2) + "px";
    g.style.top = (fc.y - fh / 2) + "px";
    g.style.width = fw + "px";
    g.style.height = fh + "px";
    let inner = null;
    if (o.reveal) {
      g.style.perspective = "1000px";
      inner = document.createElement("div");
      inner.className = "fc-inner";
      inner.style.transition = `transform ${dur}ms cubic-bezier(.4,.05,.2,1)`;
      const back = document.createElement("div");
      back.className = "fc-face fc-back";
      back.style.backgroundImage = `url(${CARD_BACK})`;
      const front = document.createElement("div");
      front.className = "fc-face fc-front";
      front.style.backgroundImage = o.img ? `url(/deck_images/${o.img})` : `url(${CARD_BACK})`;
      inner.appendChild(back);
      inner.appendChild(front);
      g.appendChild(inner);
    } else {
      g.classList.add("plain");
      g.style.backgroundImage = o.img ? `url(/deck_images/${o.img})` : `url(${CARD_BACK})`;
    }
    document.body.appendChild(g);
    g.style.transition = `transform ${dur}ms cubic-bezier(.3,.8,.3,1)`;
    const dx = tc.x - fc.x, dy = tc.y - fc.y;
    const sx = tw / fw, sy = th / fh;
    const start = () => {
      g.style.transform = `translate(${dx}px, ${dy}px) scale(${sx.toFixed(3)}, ${sy.toFixed(3)})`;
      if (inner) inner.style.transform = "rotateY(180deg)";
    };
    if (o.delay) setTimeout(() => requestAnimationFrame(start), o.delay);
    else requestAnimationFrame(start);
    setTimeout(() => g.remove(), (o.delay || 0) + dur + 90);
  } catch (e) { /* cosmetic */ }
}

function discardGhost(card, fromRect, pileRect) {
  if (!card || !fromRect) return;
  flyCard({ img: card.img, from: fromRect, to: pileRect, reveal: false,
            fw: fromRect.width, fh: fromRect.height, tw: pileRect.width, th: pileRect.height });
}

function pop(el) {
  if (!el) return;
  el.classList.remove("pop");
  void el.offsetWidth;
  el.classList.add("pop");
}

// Keep a real card hidden until its fly-in ghost lands, then reveal it.
function holdAppear(el, dur, delay) {
  if (!el) return;
  el.style.setProperty("--appdur", dur + "ms");
  el.style.animationDelay = (delay || 0) + "ms";
  el.classList.add("hold-appear");
}

// Animate the opponent's turn (draw then discard) and return its total ms.
function animateOpponentEvents(events, gen, finalTop, pileWasMine, ctx) {
  if (!events || !events.length) {
    if (pileWasMine) setDiscImg(finalTop);
    return 0;
  }
  let endMs = 0;
  try {
    const { oppNewEls, oppOldRects, showOpp, oppRect, stockRect, discRect,
            PILE_W, PILE_H, OPP_W, OPP_H } = ctx;
    const slotOf = (idx) => {
      const el = oppNewEls[idx];
      return el ? el.getBoundingClientRect() : oppRect;
    };
    let t = T_FLY + 40;     // start once my own discard fly has landed
    let lastDiscardLand = null;
    events.forEach((ev) => {
      if (ev.type === "opp_draw") {
        const dest = (showOpp && ev.card) ? slotOf(ev.card.idx) : oppRect;
        if (ev.source === "discard") {
          // already face-up on the pile -> plain fly straight into his hand
          if (showOpp) holdAppear(oppNewEls[ev.card && ev.card.idx], T_FLY, t);
          flyCard({ from: discRect, to: dest, img: ev.card ? ev.card.img : null,
                    reveal: false, delay: t, fw: PILE_W, fh: PILE_H, tw: OPP_W, th: OPP_H });
          t += T_FLY + T_OPP_GAP;
        } else if (showOpp) {
          // CASE 3: from the stock, turns around and reveals to me, into his hand
          holdAppear(oppNewEls[ev.card && ev.card.idx], T_REVEAL, t);
          flyCard({ from: stockRect, to: dest, img: ev.card ? ev.card.img : null,
                    reveal: true, slow: true, delay: t,
                    fw: PILE_W, fh: PILE_H, tw: OPP_W, th: OPP_H });
          t += T_REVEAL + T_OPP_GAP;
        } else {
          // hidden: a face-down card rotates as it goes into his hidden hand
          flyCard({ from: stockRect, to: dest, img: null, reveal: true, slow: true,
                    delay: t, fw: PILE_W, fh: PILE_H, tw: OPP_W, th: OPP_H });
          t += T_REVEAL + T_OPP_GAP;
        }
      } else if (ev.type === "opp_discard" && ev.card) {
        const src = (showOpp && oppOldRects[ev.card.idx]) ? oppOldRects[ev.card.idx] : oppRect;
        if (!showOpp) {
          // CASE 2: hidden card is revealed as it flips onto the pile
          flyCard({ from: src, to: discRect, img: ev.card.img, reveal: true, slow: true,
                    delay: t, fw: OPP_W, fh: OPP_H, tw: PILE_W, th: PILE_H });
          lastDiscardLand = t + T_REVEAL;
          t += T_REVEAL + T_OPP_GAP;
        } else {
          // revealed hand: the visible card flies straight to the pile (no spin)
          flyCard({ from: src, to: discRect, img: ev.card.img, reveal: false,
                    delay: t, fw: OPP_W, fh: OPP_H, tw: PILE_W, th: PILE_H });
          lastDiscardLand = t + T_FLY;
          t += T_FLY + T_OPP_GAP;
        }
      }
    });
    const at = lastDiscardLand != null ? lastDiscardLand : t;
    endMs = at + 120;
    setTimeout(() => {
      if (gen !== renderGen) return;
      if (pileWasMine) setDiscImg(finalTop);
      pop($("discard"));
    }, at);
  } catch (e) {
    if (pileWasMine) setDiscImg(finalTop);
  }
  return endMs;
}

// ---------------------------------------------------------------- render
function render(v) {
  view = v;
  const gen = ++renderGen;

  if (v.opponent_key) {
    const sel = $("opponent-select");
    if (sel && sel.value !== v.opponent_key) sel.value = v.opponent_key;
  }
  $("message").textContent = v.message || "";
  $("deadwood-badge").textContent = `deadwood: ${v.deadwood}`;

  const stock = $("stock");
  const disc = $("discard");
  const pileRect = stock.getBoundingClientRect();
  const PILE_W = pileRect.width || 78, PILE_H = pileRect.height || 112;

  // ---- opponent zone (capture old rects -> reflow when revealed) ----
  const oppMelded = new Set();
  (v.opponent_melds || []).forEach((m) => m.forEach((i) => oppMelded.add(i)));
  const opp = $("opponent");
  const showOpp = $("debug-toggle") && $("debug-toggle").checked;
  const oppOldRects = {};
  for (const el of Array.from(opp.children)) {
    const i = Number(el.dataset.idx);
    if (!Number.isNaN(i)) oppOldRects[i] = el.getBoundingClientRect();
  }
  opp.innerHTML = "";
  const oppNewEls = {};
  if (showOpp && v.opponent_hand_live && v.opponent_hand_live.length) {
    v.opponent_hand_live.forEach((c) => {
      const el = cardEl(c, true);
      if (oppMelded.has(c.idx)) el.classList.add("melded");
      opp.appendChild(el);
      oppNewEls[c.idx] = el;
    });
  } else {
    const n = v.opponent_count || 10;
    for (let i = 0; i < n; i++) opp.appendChild(backEl());
  }
  const oppRect = opp.getBoundingClientRect();
  const firstOpp = opp.querySelector(".card");
  const oR = firstOpp ? firstOpp.getBoundingClientRect() : null;
  const OPP_W = oR ? oR.width : 56, OPP_H = oR ? oR.height : 80;
  // reflow persisting opponent cards (revealed)
  if (showOpp) {
    Object.keys(oppNewEls).forEach((idx) => {
      if (oppOldRects[idx]) flip(oppNewEls[idx], oppOldRects[idx]);
    });
  }

  // ---- stock ----
  stock.className = "card-slot stock";
  stock.style.backgroundImage = `url(${CARD_BACK})`;
  stock.onclick = null;
  if (!v.done && v.legal.draw_stock) {
    stock.classList.add("clickable");
    stock.onclick = () => send(A_DRAW);
  }

  // ---- discard (choreography: my card lands first, opponent's reveals later) ----
  disc.className = "card-slot discard";
  disc.onclick = null;
  const myDiscard = pendingDiscard && prevCardsByIdx[pendingDiscard.idx];
  const oppTurn = (v.events || []).length > 0;
  const topIdx = v.top_discard ? v.top_discard.idx : null;
  if (myDiscard && oppTurn) {
    setDiscImg(myDiscard);
    pop(disc);
  } else {
    setDiscImg(v.top_discard);
    if (topIdx !== prevTopIdx && topIdx !== null) pop(disc);
  }
  prevTopIdx = topIdx;
  if (!v.done && v.legal.take_discard) {
    disc.classList.add("clickable");
    disc.onclick = () => send(A_TAKE);
  }

  // ---- my hand ----
  const handDiv = $("hand");
  const oldRects = {};
  for (const el of Array.from(handDiv.children)) {
    const i = Number(el.dataset.idx);
    if (!Number.isNaN(i)) oldRects[i] = el.getBoundingClientRect();
  }
  if (pendingDiscard && prevCardsByIdx[pendingDiscard.idx]) {
    const fr = oldRects[pendingDiscard.idx];
    discardGhost(prevCardsByIdx[pendingDiscard.idx], fr, disc.getBoundingClientRect());
  }

  const melded = new Set();
  (v.melds || []).forEach((m) => m.forEach((idx) => melded.add(idx)));
  const discardable = new Set(v.legal.discard);
  const knockable = new Set(v.legal.knock);

  handDiv.innerHTML = "";
  const newEls = {};
  v.hand.forEach((card) => {
    const el = cardEl(card, false);
    if (melded.has(card.idx)) el.classList.add("melded");
    if (!v.done && v.phase === "discard") {
      if (knockArmed && knockable.has(card.idx)) {
        el.classList.add("knockable");
        el.onclick = () => send(knockAction(card.idx));
      } else if (!knockArmed && discardable.has(card.idx)) {
        el.classList.add("playable");
        el.onclick = () => send(discardAction(card.idx));
      } else {
        el.classList.add("disabled");
      }
    }
    handDiv.appendChild(el);
    newEls[card.idx] = el;
  });

  const isDeal = dealNext ||
    (Object.keys(oldRects).length === 0 && v.hand.length > 1);
  v.hand.forEach((card, i) => {
    const el = newEls[card.idx];
    if (isDeal) {
      flyOnlyCard(el, pileRect, i * T_DEAL_STAGGER);
    } else if (oldRects[card.idx]) {
      flip(el, oldRects[card.idx]);
    } else if (pendingDraw && pendingDraw.source === "stock") {
      const slot = el.getBoundingClientRect();
      el.style.setProperty("--appdur", T_REVEAL + "ms");
      el.classList.add("hold-appear");
      flyCard({ from: pileRect, to: slot, reveal: true, slow: true, img: card.img,
                fw: PILE_W, fh: PILE_H, tw: slot.width, th: slot.height });
    } else if (pendingDraw) {
      flyOnlyCard(el, disc.getBoundingClientRect(), 0, T_TAKE);
    }
  });
  dealNext = false;

  // ---- opponent animation + interaction lock ----
  const oppEnd = animateOpponentEvents(v.events || [], gen, v.top_discard,
    !!(myDiscard && oppTurn),
    { oppNewEls, oppOldRects, showOpp, oppRect, stockRect: pileRect, discRect: disc.getBoundingClientRect(),
      PILE_W, PILE_H, OPP_W, OPP_H });

  if (isDeal) lockMs = v.hand.length * T_DEAL_STAGGER + 560;
  else if (pendingDraw) lockMs = (pendingDraw.source === "stock") ? T_REVEAL + 120 : T_TAKE + 120;
  lockMs = Math.max(lockMs, oppEnd);

  // ---- buttons ----
  setBtn("btn-draw", !v.done && v.legal.draw_stock, () => send(A_DRAW));
  setBtn("btn-take", !v.done && v.legal.take_discard, () => send(A_TAKE));
  setBtn("btn-gin", !v.done && v.legal.gin, () => send(A_GIN));
  setBtn("btn-dead", !v.done && v.legal.declare_dead, () => send(A_DEAD));
  const canKnock = !v.done && v.legal.knock.length > 0;
  const knockBtn = $("btn-knock");
  setBtn("btn-knock", canKnock, () => { knockArmed = !knockArmed; render(view); });
  knockBtn.classList.toggle("armed", knockArmed && canKnock);
  if (knockArmed && canKnock) $("message").textContent = "Knock: click the card to discard.";

  // ---- end overlay ----
  const overlay = $("overlay");
  if (v.done) {
    const card = overlay.querySelector(".overlay-card");
    card.className = "overlay-card " + (v.result || "draw");
    $("overlay-title").textContent =
      v.result === "win" ? "You win! 🎉" : v.result === "loss" ? "You lost" : "Draw";
    let sub = v.message || "";
    if (typeof v.deadwood === "number") sub += `  ·  your deadwood: ${v.deadwood}`;
    if (typeof v.opponent_deadwood === "number")
      sub += `  ·  opponent deadwood: ${v.opponent_deadwood}`;
    $("overlay-sub").textContent = sub;
    const reveal = overlay.querySelector(".reveal");
    const oh = $("overlay-opphand");
    oh.innerHTML = "";
    if (v.opponent_reveal && v.opponent_reveal.length) {
      v.opponent_reveal.forEach((c, i) => {
        const el = cardEl(c, true);
        if (oppMelded.has(c.idx)) el.classList.add("melded");
        el.classList.add("flip-in");
        el.style.animationDelay = (i * 55) + "ms";
        oh.appendChild(el);
      });
      reveal.style.display = "";
    } else {
      reveal.style.display = "none";
    }
    overlay.classList.remove("hidden");
  } else {
    overlay.classList.add("hidden");
  }

  // ---- bookkeeping ----
  prevCardsByIdx = {};
  v.hand.forEach((c) => { prevCardsByIdx[c.idx] = c; });
  if (v.opponent_reveal) v.opponent_reveal.forEach((c) => { prevCardsByIdx[c.idx] = c; });
  pendingDraw = null;
  pendingDiscard = null;
}

function setBtn(id, enabled, onclick) {
  const b = $(id);
  b.disabled = !enabled;
  b.onclick = enabled ? onclick : null;
}

$("btn-new").onclick = newGame;
$("overlay-again").onclick = newGame;
$("debug-toggle").onchange = () => { if (view) render(view); };

(async function init() {
  try {
    await loadOpponents();
    render(await api("GET", "/api/state"));
  } catch (e) {
    flash(e.message);
  }
})();
