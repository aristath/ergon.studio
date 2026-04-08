import assert from 'assert';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

(async () => {
  const memoryPath = path.resolve(__dirname, '..', 'dist', 'memory.js');
  const { createMemoryClient, DEFAULT_RECALL_LIMIT, DEFAULT_MEMORY_URL } = await import(memoryPath);

  // --- defaults ---

  assert.strictEqual(DEFAULT_RECALL_LIMIT, 5);
  assert.strictEqual(DEFAULT_MEMORY_URL, 'http://127.0.0.1:8082');
  console.log('✅ defaults exported');

  // --- recall: happy path returns mapped items ---

  const calls = [];
  const mockFetchOk = async (url, init) => {
    calls.push({ method: 'fetch', url, init });
    return {
      ok: true,
      status: 200,
      json: async () => ({
        // Shape verified live against openmemory-js v1.3.3
        query: 'whatever',
        matches: [
          { id: 'm1', content: 'New Rust projects default to edition 2024', score: 0.92 },
          { id: 'm2', content: 'Python uses uv not pip', score: 0.81 },
        ],
      }),
    };
  };

  const happy = createMemoryClient({ fetch: mockFetchOk });
  const results = await happy.recall('rust edition');
  assert.strictEqual(results.length, 2);
  assert.strictEqual(results[0].id, 'm1');
  assert.strictEqual(results[0].content, 'New Rust projects default to edition 2024');
  assert.strictEqual(results[0].score, 0.92);
  assert.strictEqual(results[1].id, 'm2');

  // verify request shape
  const c = calls[0];
  assert.strictEqual(c.url, 'http://127.0.0.1:8082/memory/query');
  assert.strictEqual(c.init.method, 'POST');
  const body = JSON.parse(c.init.body);
  assert.strictEqual(body.query, 'rust edition');
  assert.strictEqual(body.k, 5);
  assert.strictEqual(body.filters, undefined, 'no filters when no userID');

  console.log('✅ recall happy path');

  // --- recall: custom limit ---

  calls.length = 0;
  await happy.recall('python', 10);
  const body2 = JSON.parse(calls[0].init.body);
  assert.strictEqual(body2.k, 10, 'custom limit must be passed through');

  console.log('✅ recall custom limit');

  // --- recall: empty query → [] without calling fetch ---

  calls.length = 0;
  const empty = await happy.recall('');
  assert.deepStrictEqual(empty, []);
  assert.strictEqual(calls.length, 0, 'empty query must not hit fetch');

  console.log('✅ recall empty query → []');

  // --- recall: handles fetch throwing ---

  const mockFetchThrows = async () => { throw new Error('connection refused'); };
  const throws = createMemoryClient({ fetch: mockFetchThrows });
  const r = await throws.recall('anything');
  assert.deepStrictEqual(r, [], 'recall must return [] on fetch failure, not throw');

  console.log('✅ recall fetch failure → []');

  // --- recall: non-OK response → [] ---

  const mockFetch500 = async () => ({ ok: false, status: 500, json: async () => ({}) });
  const c500 = createMemoryClient({ fetch: mockFetch500 });
  assert.deepStrictEqual(await c500.recall('q'), []);

  console.log('✅ recall non-OK → []');

  // --- recall: filters out malformed entries ---

  const mockFetchMalformed = async () => ({
    ok: true,
    status: 200,
    json: async () => ({
      matches: [
        null,
        { id: 'm3', content: '' },               // empty content
        { id: 'm4', content: 'good one', score: 0.5 },
        { /* no id, no content */ },
        'not an object',
      ],
    }),
  });

  const mal = createMemoryClient({ fetch: mockFetchMalformed });
  const filtered = await mal.recall('q');
  assert.strictEqual(filtered.length, 1, 'only well-formed non-empty items survive');
  assert.strictEqual(filtered[0].id, 'm4');

  console.log('✅ recall filters malformed entries');

  // --- recall: response without matches/memories array → [] ---

  const mockFetchNoMemories = async () => ({ ok: true, status: 200, json: async () => ({}) });
  const noMem = createMemoryClient({ fetch: mockFetchNoMemories });
  assert.deepStrictEqual(await noMem.recall('q'), []);

  console.log('✅ recall missing matches field → []');

  // --- recall: also accepts the legacy `memories` key as fallback ---

  const mockFetchLegacy = async () => ({
    ok: true,
    status: 200,
    json: async () => ({ memories: [{ id: 'L1', content: 'legacy shape', score: 0.7 }] }),
  });
  const legacy = createMemoryClient({ fetch: mockFetchLegacy });
  const legacyResults = await legacy.recall('q');
  assert.strictEqual(legacyResults.length, 1);
  assert.strictEqual(legacyResults[0].id, 'L1');

  console.log('✅ recall accepts legacy `memories` key');

  // --- save: happy path posts content + trims ---

  calls.length = 0;
  const mockFetchSave = async (url, init) => { calls.push({ url, init }); return { ok: true, json: async () => ({}) }; };
  const saver = createMemoryClient({ fetch: mockFetchSave });
  await saver.save('  New Rust projects default to edition 2024  ');
  assert.strictEqual(calls[0].url, 'http://127.0.0.1:8082/memory/add');
  const saveBody = JSON.parse(calls[0].init.body);
  assert.strictEqual(saveBody.content, 'New Rust projects default to edition 2024');
  assert.strictEqual(saveBody.user_id, undefined, 'no user_id when no userID set');

  console.log('✅ save trims and posts content');

  // --- save: empty content is a no-op ---

  calls.length = 0;
  await saver.save('');
  await saver.save('   ');
  assert.strictEqual(calls.length, 0, 'empty/whitespace content must not call fetch');

  console.log('✅ save empty/whitespace → no-op');

  // --- save: failure is swallowed (never throws) ---

  const failingSaver = createMemoryClient({ fetch: async () => { throw new Error('disk full'); } });
  // Must not throw
  await failingSaver.save('something');

  console.log('✅ save failure is swallowed');

  // --- userID is forwarded on both recall and save ---

  calls.length = 0;
  const scoped = createMemoryClient({ fetch: mockFetchOk, userID: 'aristath' });
  await scoped.recall('q');
  const recallBody = JSON.parse(calls[0].init.body);
  assert.deepStrictEqual(recallBody.filters, { user_id: 'aristath' });

  // For save we need to use a fetch that doesn't depend on parsing memories
  calls.length = 0;
  const scopedSaver = createMemoryClient({ fetch: mockFetchSave, userID: 'aristath' });
  await scopedSaver.save('a fact');
  const saveBody2 = JSON.parse(calls[0].init.body);
  assert.strictEqual(saveBody2.user_id, 'aristath');

  console.log('✅ userID forwarded to recall + save');

  // --- custom baseURL ---

  calls.length = 0;
  const custom = createMemoryClient({ fetch: mockFetchSave, baseURL: 'http://192.168.1.10:9000/' });
  await custom.save('content');
  assert.strictEqual(calls[0].url, 'http://192.168.1.10:9000/memory/add', 'custom baseURL respected; trailing slash stripped');

  console.log('✅ custom baseURL');

  console.log('\n✅ All memory client tests passed');
})();
