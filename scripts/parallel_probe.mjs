// Concurrency regression probe for the streamable-http transport.
//
// WHY: on 2026-07-22, under FastMCP 3.3.1, concurrent tools/call requests on one
// session occasionally received EACH OTHER'S responses (cross-contamination) or
// timed out. Root-caused to upstream FastMCP session handling (fixed by 3.4.x);
// verified clean on 3.4.4 (42/42 concurrent calls correct on 2026-07-24).
// Re-run this probe after every FastMCP upgrade before trusting parallel calls.
//
// USAGE:
//   node scripts/parallel_probe.mjs all            # tests C, A, B (local tools)
//   node scripts/parallel_probe.mjs D 4            # concurrent upstream calls
//   node scripts/parallel_probe.mjs E              # uncached upstream rotation
//   MTG_MCP_URL=http://127.0.0.1:8001/mcp node scripts/parallel_probe.mjs all
//
// Each response is checked on TWO axes: JSON-RPC id matches the request, and the
// content contains the marker expected for THAT tool (foreign markers = crossed).
// Expected result on a healthy server: 0 anomalies everywhere.

const BASE = process.env.MTG_MCP_URL || "https://mtg.solucia.app/mcp";
const HDRS = {
  "content-type": "application/json",
  accept: "application/json, text/event-stream",
};

async function post(body, sessionId, timeoutMs = 45000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  const headers = { ...HDRS };
  if (sessionId) headers["mcp-session-id"] = sessionId;
  const t0 = Date.now();
  try {
    const res = await fetch(BASE, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    const sid = res.headers.get("mcp-session-id");
    const text = await res.text();
    const ms = Date.now() - t0;
    const events = [];
    for (const line of text.split("\n")) {
      const m = line.match(/^data:\s*(\{.*\})\s*$/);
      if (m) {
        try { events.push(JSON.parse(m[1])); } catch { /* fragment */ }
      }
    }
    if (events.length === 0 && text.trim().startsWith("{")) {
      try { events.push(JSON.parse(text)); } catch { /* not JSON */ }
    }
    return { status: res.status, sid, events, ms, raw: text.slice(0, 300) };
  } catch (e) {
    return { status: "ERR", sid: null, events: [], ms: Date.now() - t0, raw: String(e) };
  } finally {
    clearTimeout(t);
  }
}

async function openSession() {
  const init = await post({
    jsonrpc: "2.0", id: 1, method: "initialize",
    params: {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "parallel-probe", version: "1.0" },
    },
  });
  const sid = init.sid;
  if (!sid) throw new Error("no mcp-session-id: " + init.raw);
  await post({ jsonrpc: "2.0", method: "notifications/initialized" }, sid);
  return sid;
}

// Three LOCAL tools (no upstream API): isolates transport behavior from rate limits.
const CALLS = [
  { id: 101, tool: "ping", args: {}, marker: "pong" },
  { id: 102, tool: "bulk_card_lookup", args: { name: "Sol Ring" }, marker: "Sol Ring" },
  { id: 103, tool: "rules_lookup", args: { query: "508.4" }, marker: "508.4" },
];

function makeCheck(calls) {
  return (call, r) => {
    const ev = r.events.find((e) => "id" in e);
    if (r.status === "ERR" || !ev) {
      return { verdict: "TIMEOUT/EMPTY", detail: `${r.status} ${r.raw.slice(0, 160)}`, ms: r.ms };
    }
    const idOk = ev.id === call.id;
    const content = JSON.stringify(ev.result ?? ev.error ?? {});
    const contentOk = content.includes(call.marker);
    const foreign = calls.filter((c) => c !== call && content.includes(c.marker)).map((c) => c.tool);
    let verdict = "OK";
    if (!idOk) verdict = "CROSSED ID";
    else if (!contentOk && foreign.length) verdict = "CROSSED CONTENT";
    else if (!contentOk) verdict = "UNEXPECTED";
    return {
      verdict,
      detail: `got_id=${ev.id} want=${call.id} foreign=[${foreign}] ${contentOk ? "" : "excerpt=" + content.slice(0, 140)}`,
      ms: r.ms,
    };
  };
}

async function runBatch(sid, calls, label, counters) {
  const check = makeCheck(calls);
  const results = await Promise.all(
    calls.map((c) =>
      post({ jsonrpc: "2.0", id: c.id, method: "tools/call", params: { name: c.tool, arguments: c.args } }, sid)
    )
  );
  results.forEach((r, i) => {
    const chk = check(calls[i], r);
    if (chk.verdict !== "OK") counters.anomalies++;
    counters.total++;
    console.log(`  ${label} | ${calls[i].tool.padEnd(24)} | ${chk.verdict.padEnd(16)} | ${chk.ms}ms | ${chk.detail}`);
  });
}

async function testA(rounds) {
  console.log(`\n=== TEST A: 1 session, ${CALLS.length} CONCURRENT local calls x${rounds} rounds ===`);
  const sid = await openSession();
  const c = { anomalies: 0, total: 0 };
  for (let round = 1; round <= rounds; round++) await runBatch(sid, CALLS, `round ${round}`, c);
  console.log(`  => TEST A anomalies: ${c.anomalies}/${c.total}`);
}

async function testB() {
  console.log(`\n=== TEST B: 3 separate sessions, 1 call each, concurrent ===`);
  const sids = await Promise.all([openSession(), openSession(), openSession()]);
  const check = makeCheck(CALLS);
  const results = await Promise.all(
    CALLS.map((c, i) =>
      post({ jsonrpc: "2.0", id: c.id, method: "tools/call", params: { name: c.tool, arguments: c.args } }, sids[i])
    )
  );
  results.forEach((r, i) => {
    const chk = check(CALLS[i], r);
    console.log(`  ${CALLS[i].tool.padEnd(24)} | ${chk.verdict.padEnd(16)} | ${chk.ms}ms | ${chk.detail}`);
  });
}

async function testC() {
  console.log(`\n=== TEST C: SEQUENTIAL control, 1 session ===`);
  const sid = await openSession();
  const check = makeCheck(CALLS);
  for (const c of CALLS) {
    const r = await post({ jsonrpc: "2.0", id: c.id, method: "tools/call", params: { name: c.tool, arguments: c.args } }, sid);
    const chk = check(c, r);
    console.log(`  ${c.tool.padEnd(24)} | ${chk.verdict.padEnd(16)} | ${chk.ms}ms | ${chk.detail}`);
  }
}

// Upstream tools (Scryfall + Commander Spellbook): slow responses, long overlap.
const UPSTREAM_CALLS = [
  { id: 201, tool: "scryfall_card_details", args: { name: "Lightning Bolt" }, marker: "Lightning Bolt" },
  { id: 202, tool: "scryfall_card_rulings", args: { name: "Aurelia, the Warleader" }, marker: "Aurelia" },
  { id: 203, tool: "spellbook_find_combos", args: { card_name: "Basalt Monolith" }, marker: "Basalt" },
];

async function testD(rounds) {
  console.log(`\n=== TEST D: 1 session, 3 concurrent UPSTREAM calls x${rounds} rounds ===`);
  const sid = await openSession();
  const check = makeCheck(UPSTREAM_CALLS);
  console.log("  -- sequential control --");
  for (const c of UPSTREAM_CALLS) {
    const r = await post({ jsonrpc: "2.0", id: c.id, method: "tools/call", params: { name: c.tool, arguments: c.args } }, sid);
    const chk = check(c, r);
    console.log(`  seq | ${c.tool.padEnd(24)} | ${chk.verdict.padEnd(16)} | ${chk.ms}ms | ${chk.detail}`);
  }
  console.log("  -- concurrent --");
  const counters = { anomalies: 0, total: 0 };
  for (let round = 1; round <= rounds; round++) await runBatch(sid, UPSTREAM_CALLS, `round ${round}`, counters);
  console.log(`  => TEST D anomalies: ${counters.anomalies}/${counters.total}`);
}

// Fresh card names each round: server cache cannot serve them, guaranteeing real
// upstream latency and long overlapping in-flight requests.
async function testE() {
  const ROUNDS = [
    [
      { id: 301, tool: "scryfall_card_details", args: { name: "Smothering Tithe" }, marker: "Smothering Tithe" },
      { id: 302, tool: "scryfall_card_rulings", args: { name: "Teysa Karlov" }, marker: "Teysa" },
      { id: 303, tool: "spellbook_find_combos", args: { card_name: "Isochron Scepter" }, marker: "Isochron" },
    ],
    [
      { id: 311, tool: "scryfall_card_details", args: { name: "Rhystic Study" }, marker: "Rhystic Study" },
      { id: 312, tool: "scryfall_card_rulings", args: { name: "Edgar Markov" }, marker: "Edgar" },
      { id: 313, tool: "spellbook_find_combos", args: { card_name: "Godo, Bandit Warlord" }, marker: "Godo" },
    ],
    [
      { id: 321, tool: "scryfall_card_details", args: { name: "Necropotence" }, marker: "Necropotence" },
      { id: 322, tool: "scryfall_card_rulings", args: { name: "Atraxa, Praetors' Voice" }, marker: "Atraxa" },
      { id: 323, tool: "spellbook_find_combos", args: { card_name: "Kiki-Jiki, Mirror Breaker" }, marker: "Kiki" },
    ],
    [
      { id: 331, tool: "scryfall_card_details", args: { name: "Swords to Plowshares" }, marker: "Swords to Plowshares" },
      { id: 332, tool: "scryfall_card_rulings", args: { name: "Yuriko, the Tiger's Shadow" }, marker: "Yuriko" },
      { id: 333, tool: "spellbook_find_combos", args: { card_name: "Basalt Monolith" }, marker: "Basalt" },
    ],
  ];
  console.log(`\n=== TEST E: 1 session, 3 concurrent UPSTREAM calls, FRESH cards per round ===`);
  const sid = await openSession();
  const counters = { anomalies: 0, total: 0 };
  for (let i = 0; i < ROUNDS.length; i++) await runBatch(sid, ROUNDS[i], `round ${i + 1}`, counters);
  console.log(`  => TEST E anomalies: ${counters.anomalies}/${counters.total}`);
}

const mode = process.argv[2] || "all";
if (mode === "all") {
  await testC();
  await testA(5);
  await testB();
} else if (mode === "A") await testA(Number(process.argv[3] || 5));
else if (mode === "B") await testB();
else if (mode === "C") await testC();
else if (mode === "D") await testD(Number(process.argv[3] || 4));
else if (mode === "E") await testE();
else console.log("usage: node scripts/parallel_probe.mjs [all|A|B|C|D|E] [rounds]");
