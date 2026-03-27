import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import http from 'node:http';

const PORT = 14096;
const BASE = `http://127.0.0.1:${PORT}`;
let server;

async function waitForHealth(attempts = 60, delay = 1000) {
  for (let i = 0; i < attempts; i++) {
    try {
      const res = await fetch(`${BASE}/global/health`);
      if (res.ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, delay));
  }
  throw new Error(`opencode server did not become healthy on port ${PORT}`);
}

before(async () => {
  server = spawn('opencode', ['serve', '--port', String(PORT)], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  server.stderr.on('data', (d) => process.stderr.write(d));
  await waitForHealth();
});

after(() => {
  server?.kill();
});

test('server is healthy', async () => {
  const res = await fetch(`${BASE}/global/health`);
  assert.ok(res.ok);
  const body = await res.json();
  assert.strictEqual(body.healthy, true);
});

test('run_parallel is registered as a tool', async () => {
  const res = await fetch(`${BASE}/experimental/tool/ids`);
  assert.ok(res.ok, 'tool/ids endpoint should respond');
  const ids = await res.json();
  assert.ok(Array.isArray(ids), 'tool IDs should be an array');
  assert.ok(ids.includes('run_parallel'), 'run_parallel should be in tool IDs');
});

test('all ergon agents are available', async () => {
  const res = await fetch(`${BASE}/agent`);
  assert.ok(res.ok);
  const agents = await res.json();
  const names = agents.map((a) => a.name);
  for (const expected of ['orchestrator', 'coder', 'architect', 'reviewer', 'tester', 'critic', 'researcher']) {
    assert.ok(names.includes(expected), `agent "${expected}" should be available`);
  }
});

// Tests the API calls that run_parallel.execute() makes internally:
// session.create(parentID) + session.prompt() in parallel + session.delete().
// This verifies the real-server integration without depending on the orchestrator
// deciding to call the tool (which is model-dependent and covered by unit tests).
test('run_parallel: parallel sub-sessions work end-to-end', { timeout: 600_000 }, async () => {
  // Create a parent session (mirrors what run_parallel receives as context.sessionID)
  const parentRes = await fetch(`${BASE}/session`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: 'e2e-run_parallel-parent' }),
  });
  assert.ok(parentRes.ok, 'parent session should be created');
  const parent = await parentRes.json();

  // Create two child sessions in parallel — same as run_parallel does
  const [child1Res, child2Res] = await Promise.all([
    fetch(`${BASE}/session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'researcher (parallel)', parentID: parent.id }),
    }),
    fetch(`${BASE}/session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'critic (parallel)', parentID: parent.id }),
    }),
  ]);
  assert.ok(child1Res.ok, 'child session 1 should be created');
  assert.ok(child2Res.ok, 'child session 2 should be created');
  const [child1, child2] = await Promise.all([child1Res.json(), child2Res.json()]);
  assert.ok(child1.id, 'child session 1 should have an ID');
  assert.ok(child2.id, 'child session 2 should have an ID');

  // Prompt both child sessions in parallel — same as run_parallel does
  function promptSession(id, agent, text) {
    return new Promise((resolve, reject) => {
      const payload = JSON.stringify({
        agent,
        parts: [{ type: 'text', text }],
      });
      const req = http.request(`${BASE}/session/${id}/message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
      }, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          assert.strictEqual(res.statusCode, 200, `prompt for ${agent} returned ${res.statusCode}`);
          resolve(JSON.parse(data));
        });
      });
      req.setTimeout(590_000, () => req.destroy(new Error(`prompt timeout for ${agent}`)));
      req.on('error', reject);
      req.write(payload);
      req.end();
    });
  }

  const brief = 'Name one programming language. Reply with just the name, nothing else.';
  const [resp1, resp2] = await Promise.all([
    promptSession(child1.id, 'researcher', brief),
    promptSession(child2.id, 'critic', brief),
  ]);

  // Both sub-agents should return a text part with non-empty content
  const text1 = resp1.parts?.find((p) => p.type === 'text')?.text ?? '';
  const text2 = resp2.parts?.find((p) => p.type === 'text')?.text ?? '';
  assert.ok(text1.length > 0, 'researcher sub-agent should return non-empty text');
  assert.ok(text2.length > 0, 'critic sub-agent should return non-empty text');

  // Clean up all three sessions — same as run_parallel does for child sessions
  await Promise.all([
    fetch(`${BASE}/session/${child1.id}`, { method: 'DELETE' }),
    fetch(`${BASE}/session/${child2.id}`, { method: 'DELETE' }),
    fetch(`${BASE}/session/${parent.id}`, { method: 'DELETE' }),
  ]);
});
