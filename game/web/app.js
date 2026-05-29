"use strict";

// Action ids (PettingZoo gin_rummy_v4). The server re-validates everything.
const A_DRAW = 2;
const A_TAKE = 3;
const A_DEAD = 4;
const A_GIN = 5;
const discardAction = (idx) => 6 + idx;
const knockAction = (idx) => 58 + idx;
const CARD_BACK = "/deck_images/card_back.png";

let view = null;
let knockArmed = false;
let busy = false;

// animation bookkeeping
let prevCardsByIdx = {};   // idx -> card obj (for ghosts)
let prevTopIdx = null;     // last discard-top idx
let pendingDraw = null;    // {source:'stock'|'discard'} when we draw/take
let pendingDiscard = null; // {idx} when we discard/knock
let dealNext = false;      // animate a fresh deal on the next render
let renderGen = 0;         // bumped each render; guards stale scheduled DOM writes

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- networking
async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

async function refresh() {
  try { render(await api("GET", "/api/state")); } catch (e) { flash(e.message); }
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

async function run(fn) {
  busy = true;
  try { render(await fn()); }
  catch (e) { flash(e.message); }
  finally { busy = false; }
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
// FLIP a hand card from a previous rect into its slot; `grow` adds a 3D flip.
function flip(el, fromRect, opts) {
  opts = opts || {};
  try {
    const to = el.getBoundingClientRect();
    const dx = fromRect.left - to.left;
    const dy = fromRect.top - to.top;
    if (!opts.grow && Math.abs(dx) < 1 && Math.abs(dy) < 1) return;
    const sx = opts.grow ? (fromRect.width / to.width) || 1 : 1;
    const sy = opts.grow ? (fromRect.height / to.height) || 1 : 1;
    const delay = opts.delay || 0;
    const rot = opts.grow ? " rotateY(90deg)" : "";
    el.style.transition = "none";
    el.style.transform =
      `translate(${dx}px, ${dy}px) scale(${sx.toFixed(3)}, ${sy.toFixed(3)})${rot}`;
    if (opts.grow) el.style.opacity = "0.25";
    el.getBoundingClientRect(); // reflow
    requestAnimationFrame(() => {
      el.style.transition =
        `transform .5s cubic-bezier(.2,.85,.25,1) ${delay}ms, opacity .4s ${delay}ms`;
      el.style.transform = "";
      el.style.opacity = "";
      setTimeout(() => { el.style.transition = ""; }, 580 + delay);
    });
  } catch (e) { /* end-state already correct */ }
}

// Plain fly-in on the REAL card (no rotation): travels from the source pile to
// its slot. Used for the deal and for taking the already-visible discard.
function flyOnlyCard(el, srcRect, delay, dur) {
  try {
    const r = el.getBoundingClientRect();
    const dx = (srcRect.left + srcRect.width / 2) - (r.left + r.width / 2);
    const dy = (srcRect.top + srcRect.height / 2) - (r.top + r.height / 2);
    el.style.setProperty("--dx", dx + "px");
    el.style.setProperty("--dy", dy + "px");
    el.style.setProperty("--flydur", (dur || 520) + "ms");
    el.style.animationDelay = (delay || 0) + "ms";
    el.classList.add("fly-only");
  } catch (e) { /* end-state already correct */ }
}

// A free-flying card that travels the FULL path between two screen rects.
// reveal=true -> two-faced (shows the back, then turns to the front), slow.
// reveal=false -> a plain single-face card (img, or the card back if img is null).
function flyCard(o) {
  try {
    const W = o.w || (o.small ? 56 : 74);
    const H = o.h || (o.small ? 80 : 104);
    const dur = o.slow ? 1000 : 520;
    const fcx = o.from.left + o.from.width / 2, fcy = o.from.top + o.from.height / 2;
    const tcx = o.to.left + o.to.width / 2, tcy = o.to.top + o.to.height / 2;
    const g = document.createElement("div");
    g.className = "fly-card";
    g.style.left = (fcx - W / 2) + "px";
    g.style.top = (fcy - H / 2) + "px";
    g.style.width = W + "px";
    g.style.height = H + "px";
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
    const dx = tcx - fcx, dy = tcy - fcy;
    const start = () => {
      g.style.transform = `translate(${dx}px, ${dy}px)`;
      if (inner) inner.style.transform = "rotateY(180deg)";   // back -> front
    };
    if (o.delay) setTimeout(() => requestAnimationFrame(start), o.delay);
    else requestAnimationFrame(start);
    setTimeout(() => g.remove(), (o.delay || 0) + dur + 90);
  } catch (e) { /* cosmetic */ }
}

// My discarded card flying (no rotation — it was already face-up) onto the pile.
function discardGhost(card, fromRect) {
  if (!card || !fromRect) return;
  flyCard({ img: card.img, from: fromRect, to: $("discard").getBoundingClientRect(),
            reveal: false, w: fromRect.width, h: fromRect.height });
}

function pop(el) {
  if (!el) return;
  el.classList.remove("pop");
  void el.offsetWidth;
  el.classList.add("pop");
}

// Animate the opponent's turn (draw + discard) from the server-reported events,
// honouring hidden (face-down) vs visible (debug) hands.
function animateOpponentEvents(events, gen, finalTop, pileWasMine) {
  if (!events || !events.length) {
    if (pileWasMine) setDiscImg(finalTop);
    return;
  }
  try {
    const oppRect = $("opponent").getBoundingClientRect();
    const stockRect = $("stock").getBoundingClientRect();
    const discRect = $("discard").getBoundingClientRect();
    const showOpp = $("debug-toggle") && $("debug-toggle").checked;
    let t = 560;            // begin after my own discard fly settles
    let lastDiscardLand = null;
    events.forEach((ev) => {
      if (ev.type === "opp_draw") {
        if (ev.source === "discard") {                  // known card -> plain fly
          flyCard({ from: discRect, to: oppRect, img: ev.card ? ev.card.img : null,
                    reveal: false, small: true, delay: t });
          t += 640;
        } else if (showOpp) {                           // CASE 3: reveal into visible hand
          flyCard({ from: stockRect, to: oppRect, img: ev.card ? ev.card.img : null,
                    reveal: true, slow: true, small: true, delay: t });
          t += 1120;
        } else {                                        // hidden: a face-down card flies up
          flyCard({ from: stockRect, to: oppRect, img: null, reveal: false,
                    small: true, delay: t });
          t += 640;
        }
      } else if (ev.type === "opp_discard" && ev.card) {
        if (!showOpp) {                                 // CASE 2: hidden -> revealed on pile
          flyCard({ from: oppRect, to: discRect, img: ev.card.img, reveal: true,
                    slow: true, small: true, delay: t });
          lastDiscardLand = t + 1000;
          t += 1120;
        } else {                                        // visible card -> plain fly
          flyCard({ from: oppRect, to: discRect, img: ev.card.img, reveal: false,
                    small: true, delay: t });
          lastDiscardLand = t + 520;
          t += 640;
        }
      }
    });
    const at = lastDiscardLand != null ? lastDiscardLand : t + 150;
    setTimeout(() => {
      if (gen !== renderGen) return;          // a newer render happened; skip
      if (pileWasMine) setDiscImg(finalTop);  // reveal the opponent's actual discard
      pop($("discard"));
    }, at);
  } catch (e) {
    if (pileWasMine) setDiscImg(finalTop);
  }
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

  // ---- opponent zone ----
  const oppMelded = new Set();
  (v.opponent_melds || []).forEach((m) => m.forEach((i) => oppMelded.add(i)));
  const opp = $("opponent");
  opp.innerHTML = "";
  const showOpp = $("debug-toggle") && $("debug-toggle").checked;
  if (showOpp && v.opponent_hand_live && v.opponent_hand_live.length) {
    v.opponent_hand_live.forEach((c) => {
      const el = cardEl(c, true);
      if (oppMelded.has(c.idx)) el.classList.add("melded");
      opp.appendChild(el);
    });
  } else {
    const n = v.opponent_count || 10;
    for (let i = 0; i < n; i++) opp.appendChild(backEl());
  }

  // ---- stock ----
  const stock = $("stock");
  stock.className = "card-slot stock";
  stock.style.backgroundImage = `url(${CARD_BACK})`;
  stock.onclick = null;
  if (!v.done && v.legal.draw_stock) {
    stock.classList.add("clickable");
    stock.onclick = () => send(A_DRAW);
  }

  // ---- discard (with choreography: my card lands first, opponent's reveals later) ----
  const disc = $("discard");
  disc.className = "card-slot discard";
  disc.onclick = null;
  const myDiscard = pendingDiscard && prevCardsByIdx[pendingDiscard.idx];
  const oppTurn = (v.events || []).length > 0;
  const topIdx = v.top_discard ? v.top_discard.idx : null;
  if (myDiscard && oppTurn) {
    setDiscImg(myDiscard);                  // show my discard; opp sequence reveals the final top
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

  // ---- hand (FLIP + 3D flip-in) ----
  const handDiv = $("hand");
  const oldRects = {};
  for (const el of Array.from(handDiv.children)) {
    const i = Number(el.dataset.idx);
    if (!Number.isNaN(i)) oldRects[i] = el.getBoundingClientRect();
  }
  if (pendingDiscard && prevCardsByIdx[pendingDiscard.idx]) {
    discardGhost(prevCardsByIdx[pendingDiscard.idx], oldRects[pendingDiscard.idx]);
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
      flyOnlyCard(el, stock.getBoundingClientRect(), i * 55);      // deal: plain fly-in
    } else if (oldRects[card.idx]) {
      flip(el, oldRects[card.idx]);                                // reflow slide (smooth)
    } else if (pendingDraw && pendingDraw.source === "stock") {
      // CASE 1: drawing a fresh card from the stock -> slow reveal flip
      const slot = el.getBoundingClientRect();
      el.style.setProperty("--appdur", "1000ms");
      el.classList.add("hold-appear");
      flyCard({ from: stock.getBoundingClientRect(), to: slot, reveal: true, slow: true,
                img: card.img, w: slot.width, h: slot.height });
    } else if (pendingDraw) {
      // taking the (already-visible) discard -> plain fly, no rotation
      flyOnlyCard(el, disc.getBoundingClientRect(), 0);
    }
  });
  dealNext = false;

  // ---- opponent move animations ----
  animateOpponentEvents(v.events || [], gen, v.top_discard, !!(myDiscard && oppTurn));

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
