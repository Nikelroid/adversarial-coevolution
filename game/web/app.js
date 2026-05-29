"use strict";

// Action ids must match the PettingZoo gin_rummy_v4 action space (the backend
// re-validates every action against the env's mask, so these are just the
// agreed encoding).
const A_DRAW = 2;
const A_TAKE = 3;
const A_DEAD = 4;
const A_GIN = 5;
const discardAction = (idx) => 6 + idx;   // discard card `idx`
const knockAction = (idx) => 58 + idx;    // knock, discarding card `idx`

const CARD_BACK = "/deck_images/card_back.png";

let view = null;
let knockArmed = false;
let busy = false;

const $ = (id) => document.getElementById(id);

async function api(method, path, body) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

async function refresh() {
  try { render(await api("GET", "/api/state")); }
  catch (e) { flash(e.message); }
}

function selectedOpponent() {
  const sel = document.getElementById("opponent-select");
  return sel && sel.value ? sel.value : undefined;
}

async function newGame() {
  if (busy) return;
  knockArmed = false;
  const key = selectedOpponent();
  await run(() => api("POST", "/api/new_game", key ? { opponent: key } : {}));
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
  sel.onchange = () => newGame();   // switching opponent starts a fresh game
}

async function send(action) {
  if (busy) return;
  await run(() => api("POST", "/api/action", { action }));
}

// Single in-flight request at a time -> the client can never desync the server.
async function run(fn) {
  busy = true;
  try { render(await fn()); }
  catch (e) { flash(e.message); }
  finally { busy = false; }
}

function flash(msg) {
  const m = $("message");
  m.textContent = msg;
  m.style.outline = "2px solid #ff8e8e";
  setTimeout(() => (m.style.outline = "none"), 1200);
}

function cardDiv(card, { small = false } = {}) {
  const d = document.createElement("div");
  d.className = "card";
  d.style.backgroundImage = `url(/deck_images/${card.img})`;
  d.title = card.label;
  if (small) { d.style.width = "54px"; d.style.height = "78px"; }
  return d;
}

function backDiv() {
  const d = document.createElement("div");
  d.className = "card";
  d.style.backgroundImage = `url(${CARD_BACK})`;
  d.style.width = "54px";
  d.style.height = "78px";
  return d;
}

function render(v) {
  view = v;
  knockArmed = knockArmed && v.legal && v.legal.knock.length > 0 && v.phase === "discard";

  if (v.opponent_key) {
    const sel = $("opponent-select");
    if (sel && sel.value !== v.opponent_key) sel.value = v.opponent_key;
  }

  $("message").textContent = v.message || "";
  $("deadwood-badge").textContent = `deadwood: ${v.deadwood}`;

  // Opponent — face-down unless the "see opponent" debug toggle is on. Melded
  // (matched) opponent cards get the same gold bar as the player's.
  const oppMelded = new Set();
  (v.opponent_melds || []).forEach((m) => m.forEach((i) => oppMelded.add(i)));
  const opp = $("opponent");
  opp.innerHTML = "";
  const showOpp = $("debug-toggle") && $("debug-toggle").checked;
  if (showOpp && v.opponent_hand_live && v.opponent_hand_live.length) {
    v.opponent_hand_live.forEach((c) => {
      const el = cardDiv(c, { small: true });
      if (oppMelded.has(c.idx)) el.classList.add("melded");
      opp.appendChild(el);
    });
  } else {
    const n = v.opponent_count || 10;
    for (let i = 0; i < n; i++) opp.appendChild(backDiv());
  }

  // Stock
  const stock = $("stock");
  stock.className = "card-slot stock";
  stock.style.backgroundImage = `url(${CARD_BACK})`;
  stock.onclick = null;
  if (!v.done && v.legal.draw_stock) {
    stock.classList.add("clickable");
    stock.onclick = () => send(A_DRAW);
  }

  // Discard
  const disc = $("discard");
  disc.className = "card-slot discard";
  disc.onclick = null;
  if (v.top_discard) {
    disc.style.backgroundImage = `url(/deck_images/${v.top_discard.img})`;
    disc.textContent = "";
    disc.title = v.top_discard.label;
  } else {
    disc.style.backgroundImage = "none";
    disc.textContent = "empty";
    disc.classList.add("empty-label");
  }
  if (!v.done && v.legal.take_discard) {
    disc.classList.add("clickable");
    disc.onclick = () => send(A_TAKE);
  }

  // Player hand
  const melded = new Set();
  (v.melds || []).forEach((m) => m.forEach((idx) => melded.add(idx)));
  const discardable = new Set(v.legal.discard);
  const knockable = new Set(v.legal.knock);

  const hand = $("hand");
  hand.innerHTML = "";
  v.hand.forEach((card) => {
    const el = cardDiv(card);
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
    hand.appendChild(el);
  });

  // Buttons
  setBtn("btn-draw", !v.done && v.legal.draw_stock, () => send(A_DRAW));
  setBtn("btn-take", !v.done && v.legal.take_discard, () => send(A_TAKE));
  setBtn("btn-gin", !v.done && v.legal.gin, () => send(A_GIN));
  setBtn("btn-dead", !v.done && v.legal.declare_dead, () => send(A_DEAD));

  const canKnock = !v.done && v.legal.knock.length > 0;
  const knockBtn = $("btn-knock");
  setBtn("btn-knock", canKnock, () => {
    knockArmed = !knockArmed;
    render(view); // re-render to reflect armed state
  });
  knockBtn.classList.toggle("armed", knockArmed && canKnock);
  if (knockArmed && canKnock) $("message").textContent = "Knock: click the card to discard.";

  // Overlay
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
      v.opponent_reveal.forEach((c) => {
        const el = cardDiv(c, { small: true });
        if (oppMelded.has(c.idx)) el.classList.add("melded");
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
