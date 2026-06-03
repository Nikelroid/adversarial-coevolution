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
let toggleFlip = false;    // set when "see opponent" is toggled -> flip the whole hand

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

let OPP_INFO = {};   // key -> {emoji, label, stat, desc}

function showOpponentDesc() {
  const sel = $("opponent-select");
  const d = $("opponent-desc");
  if (!sel || !d) return;
  const o = OPP_INFO[sel.value];
  d.textContent = o ? `${o.emoji} ${o.label}: ${o.desc}` : "";
}

async function loadOpponents() {
  const list = await api("GET", "/api/opponents");
  const sel = $("opponent-select");
  sel.innerHTML = "";
  OPP_INFO = {};
  list.forEach((o) => {
    OPP_INFO[o.key] = o;
    const opt = document.createElement("option");
    opt.value = o.key;
    opt.textContent = `${o.emoji ? o.emoji + " " : ""}${o.label} · ${o.stat}`;
    sel.appendChild(opt);
  });
  sel.onchange = () => { showOpponentDesc(); newGame(); };
  showOpponentDesc();
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

// An opponent card: its face when revealed, otherwise a back that still carries
// the card's idx so it occupies its TRUE sorted slot (we just don't show it).
function oppCardEl(card, showOpp) {
  if (showOpp) return cardEl(card, true);
  const d = backEl();
  if (card) d.dataset.idx = card.idx;
  return d;
}

// Add the gold meld highlight AFTER the card's entry animation finishes, so the
// rim never appears mid-flight (and never competes with the fly/reveal anim).
function meldOn(el, delay) {
  if (!el) return;
  if (delay > 0) setTimeout(() => el.classList.add("melded"), delay);
  else el.classList.add("melded");
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
      const back = document.createElement("div");
      back.className = "fc-face fc-back";
      back.style.backgroundImage = `url(${CARD_BACK})`;
      const front = document.createElement("div");
      front.className = "fc-face fc-front";
      front.style.backgroundImage = o.img ? `url(/deck_images/${o.img})` : `url(${CARD_BACK})`;
      inner.appendChild(back);
      inner.appendChild(front);
      // reverseFlip starts on the FACE and turns to the BACK as it lands (a face-up
      // discard taken into a hidden hand). Normal reveal is back -> face.
      if (o.reverseFlip) inner.style.transform = "rotateY(180deg)";
      inner.style.transition = `transform ${dur}ms cubic-bezier(.4,.05,.2,1)`;
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
      if (inner) inner.style.transform = o.reverseFlip ? "rotateY(0deg)" : "rotateY(180deg)";
    };
    if (o.delay) setTimeout(() => requestAnimationFrame(start), o.delay);
    else requestAnimationFrame(start);
    setTimeout(() => g.remove(), (o.delay || 0) + dur + 90);
  } catch (e) { /* cosmetic */ }
}

function discardGhost(card, fromRect, pileRect) {
  if (!card || !fromRect) return;
  // keep the card the same size while flying to the pile (no shrink)
  flyCard({ img: card.img, from: fromRect, to: pileRect, reveal: false,
            fw: fromRect.width, fh: fromRect.height, tw: fromRect.width, th: fromRect.height });
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

// Play the opponent's turn one step at a time on the PRE-turn fan, ending at the
// post-turn hand. Steps: 1) open a space  2) fly the new card in  3) choose the
// discard  4) take it out of the hand  5) send it to the pile. Hidden and revealed
// hands differ only by whether cards rotate to show a face. Returns total ms.
function runOppTurn(o) {
  const { oppEl, showOpp, drawnCard, discCard, drawSource, underCard,
          stockRect, discRect, oppRect, sizes, gen } = o;
  const { OPP_W, OPP_H, PILE_W, PILE_H } = sizes;
  // Rotation rules. Revealed hand: a stock draw turns around to reveal its face;
  // a discard draw is already face-up so it flies straight. Hidden hand: the
  // reverse — a face-up discard flips DOWN to a back as it enters the hidden hand,
  // while a face-down stock card just slides in without turning.
  const drawReveal = showOpp ? (drawSource === "stock") : (drawSource === "discard");
  const drawReverse = !showOpp && drawSource === "discard";  // face -> back
  const drawDur = drawReveal ? T_REVEAL : T_FLY;
  const discReveal = !showOpp;                  // hidden hand -> the discard reveals as it leaves
  const discDur = discReveal ? T_REVEAL : T_FLY;
  // show a face only when revealed, or when the card came from the (public)
  // discard pile; a hidden stock draw stays face-down.
  const flyImg = (showOpp || drawSource === "discard")
    ? (drawnCard ? drawnCard.img : null) : null;

  const snap = () => {
    const m = new Map();
    Array.from(oppEl.children).forEach((el) => m.set(el, el.getBoundingClientRect()));
    return m;
  };
  const reflow = (old) => Array.from(oppEl.children).forEach((el) => {
    const r = old.get(el); if (r) flip(el, r);
  });

  let t = T_FLY + 60;          // wait for my own discard to land first
  let placeholder = null;

  // STEP 1 — open a space for the incoming card
  setTimeout(() => {
    if (gen !== renderGen || !drawnCard) return;   // no draw event -> nothing to bring in
    const old = snap();
    placeholder = oppCardEl(drawnCard, showOpp);
    placeholder.style.visibility = "hidden";          // reserve the slot, invisible for now
    // insert at the card's TRUE sorted position -- hidden backs carry idx too, so
    // the new card never just lands on the right end.
    if (drawnCard) {
      const ref = Array.from(oppEl.children).find((k) => Number(k.dataset.idx) > drawnCard.idx);
      oppEl.insertBefore(placeholder, ref || null);
    } else {
      oppEl.appendChild(placeholder);
    }
    reflow(old);                                       // neighbours slide apart
  }, t);
  t += 400;

  // STEP 2 — fly the new card into the opened slot
  const flyAt = t;
  setTimeout(() => {
    if (gen !== renderGen || !placeholder) return;
    const slot = placeholder.getBoundingClientRect();
    const from = drawSource === "discard" ? discRect : stockRect;
    flyCard({ from, to: slot, img: flyImg, reveal: drawReveal, slow: drawReveal,
              reverseFlip: drawReverse,
              fw: PILE_W, fh: PILE_H, tw: OPP_W, th: OPP_H });
    // the top discard was just taken -> reveal the card that was underneath
    // (null => genuinely empty); STEP 5 sets the opponent's new discard on land.
    if (drawSource === "discard") setDiscImg(underCard);
  }, flyAt);
  setTimeout(() => { if (gen === renderGen && placeholder) placeholder.style.visibility = ""; },
             flyAt + drawDur);
  t = flyAt + drawDur + 340;          // settle, then "think"

  // STEP 3/4/5 — choose a card, take it out, send it to the pile
  const chooseAt = t;
  setTimeout(() => {
    if (gen !== renderGen || !discCard) return;        // no discard -> nothing leaves the hand
    // find the discard at its TRUE slot (hidden backs carry idx too)
    let discEl = Array.from(oppEl.children).find((k) => Number(k.dataset.idx) === discCard.idx);
    if (!discEl) {                                     // fallback: rightmost card
      const kids = Array.from(oppEl.children);
      discEl = kids[kids.length - 1];
    }
    if (discEl) discEl.classList.add("lift");          // STEP 3: chosen card lifts out
    setTimeout(() => {
      if (gen !== renderGen) return;
      const old = snap();
      const fromSlot = discEl ? discEl.getBoundingClientRect() : oppRect;
      if (discEl) discEl.remove();                     // STEP 4: leaves the hand
      reflow(old);                                     // gap closes
      if (discCard) {                                  // STEP 5: flies to the pile
        flyCard({ from: fromSlot, to: discRect, img: discCard.img,
                  reveal: discReveal, slow: discReveal,
                  fw: OPP_W, fh: OPP_H, tw: PILE_W, th: PILE_H });
        setTimeout(() => {
          if (gen !== renderGen) return;
          setDiscImg(discCard);
          pop($("discard"));
        }, discDur);
      }
    }, 360);                                            // brief "choosing" pause
  }, chooseAt);

  return chooseAt + 360 + discDur + 200;
}

// ---------------------------------------------------------------- render
function render(v) {
  // toggling "see opponent" and arming knock re-render the SAME view object; on
  // those we must NOT replay the last move's animation.
  const isReRender = (v === view);
  const isToggle = toggleFlip; toggleFlip = false;
  view = v;
  const gen = ++renderGen;

  if (v.opponent_key) {
    const sel = $("opponent-select");
    if (sel && sel.value !== v.opponent_key) { sel.value = v.opponent_key; showOpponentDesc(); }
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
  // the opponent's move this step (drives the step-by-step turn choreography)
  const drawnEv = (v.events || []).find((e) => e.type === "opp_draw");
  const discEv = (v.events || []).find((e) => e.type === "opp_discard");
  const drawnCard = drawnEv ? drawnEv.card : null;
  const discCard = discEv ? discEv.card : null;
  const drawSource = drawnEv ? drawnEv.source : "stock";
  const underCard = drawnEv ? (drawnEv.under || null) : null;  // shown when opp takes the top
  const oppTurn = (v.events || []).length > 0;

  // Never choreograph the game-ending move -- at done the server sends
  // opponent_hand_live=null, so reconstructing a pre-turn fan would collapse it.
  const playTurn = oppTurn && !isReRender && !v.done;

  // Build the opponent fan. On a fresh move we render the PRE-turn hand (the
  // discarded card still present, the drawn card absent) and runOppTurn() plays
  // it out; otherwise the current hand. Hidden cards use oppCardEl so they sit in
  // their TRUE sorted slots (shown as backs), not bunched on the right. At game
  // over the live hand is null -> fall back to the revealed final hand.
  let oppCards = (v.opponent_hand_live || v.opponent_reveal || []).slice();
  if (playTurn) {
    // pre-turn hand = live, minus the drawn card (the opp picked it up), plus the
    // discarded card (it was in hand before). If the opp drew AND discarded the
    // SAME card, the hand is unchanged -- do NOT double-count it (that produced
    // duplicate/missing cards in the visible hand).
    if (drawnCard) {
      const i = oppCards.findIndex((c) => c.idx === drawnCard.idx);
      if (i >= 0) oppCards.splice(i, 1);
    }
    if (discCard && (!drawnCard || discCard.idx !== drawnCard.idx)) oppCards.push(discCard);
  }
  oppCards.sort((a, b) => a.idx - b.idx);
  if (oppCards.length) {
    oppCards.forEach((c, i) => {
      const el = oppCardEl(c, showOpp);
      // gold rim only when revealed; after the flip on a toggle, immediately otherwise.
      if (showOpp && oppMelded.has(c.idx)) meldOn(el, isToggle ? (i * 45 + 470) : 0);
      opp.appendChild(el);
      oppNewEls[c.idx] = el;
    });
  } else {
    const n = v.opponent_count || 10;        // fallback if the hand wasn't sent
    for (let i = 0; i < n; i++) opp.appendChild(backEl());
  }
  const oppRect = opp.getBoundingClientRect();
  const firstOpp = opp.querySelector(".card");
  const oR = firstOpp ? firstOpp.getBoundingClientRect() : null;
  const OPP_W = oR ? oR.width : 56, OPP_H = oR ? oR.height : 80;
  if (isToggle) {
    // "see opponent" toggled: just turn the whole hand around in place.
    Array.from(opp.children).forEach((el, i) => {
      el.classList.add("flip-turn");
      el.style.animationDelay = (i * 45) + "ms";
    });
  } else if (!playTurn) {
    // nothing moved -> slide persisting cards from their old slots (no-op if same)
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
  const topIdx = v.top_discard ? v.top_discard.idx : null;
  if (myDiscard) {
    // keep the current pile image during my discard flight; reveal my card only
    // when it lands (the opponent sequence updates the top later).
    const landCard = myDiscard;
    setTimeout(() => { if (gen !== renderGen) return; setDiscImg(landCard); pop(disc); }, T_FLY);
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
    let meldDelay = 0;                       // when to show the gold rim (after the anim)
    if (isDeal) {
      flyOnlyCard(el, pileRect, i * T_DEAL_STAGGER);
      meldDelay = i * T_DEAL_STAGGER + T_FLY;
    } else if (oldRects[card.idx]) {
      flip(el, oldRects[card.idx]);
    } else if (pendingDraw && pendingDraw.source === "stock") {
      const slot = el.getBoundingClientRect();
      el.style.setProperty("--appdur", T_REVEAL + "ms");
      el.classList.add("hold-appear");
      flyCard({ from: pileRect, to: slot, reveal: true, slow: true, img: card.img,
                fw: PILE_W, fh: PILE_H, tw: slot.width, th: slot.height });
      meldDelay = T_REVEAL;                  // a freshly drawn card reveals before it glows
    } else if (pendingDraw) {
      flyOnlyCard(el, disc.getBoundingClientRect(), 0, T_TAKE);
      meldDelay = T_TAKE;
    }
    if (melded.has(card.idx)) meldOn(el, meldDelay);
  });
  dealNext = false;

  // ---- opponent's step-by-step turn + interaction lock ----
  let oppEnd = 0;
  if (playTurn) {
    oppEnd = runOppTurn({
      oppEl: opp, showOpp, drawnCard, discCard, drawSource, underCard,
      stockRect: pileRect, discRect: disc.getBoundingClientRect(), oppRect,
      sizes: { OPP_W, OPP_H, PILE_W, PILE_H }, gen,
    });
  }

  if (isDeal) lockMs = v.hand.length * T_DEAL_STAGGER + 560;
  else if (pendingDraw) lockMs = (pendingDraw.source === "stock") ? T_REVEAL + 120 : T_TAKE + 120;
  if (myDiscard) lockMs = Math.max(lockMs, T_FLY + 150);  // let my discard land (e.g. knock-to-win)
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
        if (oppMelded.has(c.idx)) meldOn(el, i * 55 + 560);   // gold rim after the flip-in
        el.classList.add("flip-in");
        el.style.animationDelay = (i * 55) + "ms";
        oh.appendChild(el);
      });
      reveal.style.display = "";
    } else {
      reveal.style.display = "none";
    }
    // show the win/lose screen only AFTER the final move's animation finishes
    overlay.classList.add("hidden");
    setTimeout(() => { if (gen === renderGen) overlay.classList.remove("hidden"); },
               Math.max(lockMs, 0));
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
$("debug-toggle").onchange = () => { if (view && !busy) { toggleFlip = true; render(view); } };

(async function init() {
  try {
    await loadOpponents();
    render(await api("GET", "/api/state"));
  } catch (e) {
    flash(e.message);
  }
})();
