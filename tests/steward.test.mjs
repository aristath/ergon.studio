import assert from 'assert';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

(async () => {
  const stewardPath = path.resolve(__dirname, '..', 'dist', 'steward.js');
  const {
    createStewardClient,
    REWRITE_PROMPT,
    JUDGE_PROMPT,
    DEFAULT_STEWARD_URL,
    DEFAULT_STEWARD_MODEL,
    DEFAULT_TEMPERATURE,
  } = await import(stewardPath);

  // --- prompts are exported and non-empty ---

  assert.ok(typeof REWRITE_PROMPT === 'string' && REWRITE_PROMPT.length > 50, 'REWRITE_PROMPT must be a non-trivial string');
  assert.ok(typeof JUDGE_PROMPT === 'string' && JUDGE_PROMPT.length > 50, 'JUDGE_PROMPT must be a non-trivial string');
  assert.ok(REWRITE_PROMPT.includes('NONE'), 'rewrite prompt must document NONE escape hatch');
  assert.ok(JUDGE_PROMPT.includes('"save"'), 'judge prompt must document save schema');

  console.log('✅ Prompt constants exported');

  // --- defaults are sensible ---

  assert.strictEqual(DEFAULT_STEWARD_URL, 'http://127.0.0.1:8081');
  assert.strictEqual(DEFAULT_STEWARD_MODEL, 'ergon-studio-memory-steward');
  assert.strictEqual(DEFAULT_TEMPERATURE, 0.3);

  console.log('✅ Default config matches design');

  // --- rewriteQuery: happy path ---

  let lastRequest = null;
  const mockFetchOk = async (url, init) => {
    lastRequest = { url, init };
    return {
      ok: true,
      status: 200,
      json: async () => ({
        choices: [{ message: { content: 'test implementation' } }],
      }),
    };
  };

  const client = createStewardClient({ fetch: mockFetchOk });

  const query = await client.rewriteQuery(
    'can you please do me a favor my sweet friend and test the implementation to see if there\'s anything wrong'
  );
  assert.strictEqual(query, 'test implementation', 'rewriteQuery returns trimmed completion text');

  // verify the request shape
  assert.ok(lastRequest, 'fetch should have been called');
  assert.strictEqual(lastRequest.url, 'http://127.0.0.1:8081/v1/chat/completions');
  assert.strictEqual(lastRequest.init.method, 'POST');
  const body = JSON.parse(lastRequest.init.body);
  assert.strictEqual(body.model, 'ergon-studio-memory-steward');
  assert.strictEqual(body.temperature, 0.3);
  assert.ok(Array.isArray(body.messages), 'request must have messages array');
  assert.strictEqual(body.messages[0].role, 'system');
  assert.ok(body.messages[0].content.includes('search queries'), 'system message must be the rewrite prompt');
  assert.strictEqual(body.messages[1].role, 'user');
  assert.ok(body.messages[1].content.includes('sweet friend'), 'user message must contain the original text');

  console.log('✅ rewriteQuery happy path');

  // --- rewriteQuery: NONE escape hatch returns null ---

  const mockFetchNone = async () => ({
    ok: true,
    status: 200,
    json: async () => ({ choices: [{ message: { content: 'NONE' } }] }),
  });

  const clientNone = createStewardClient({ fetch: mockFetchNone });
  const noQuery = await clientNone.rewriteQuery('thanks!');
  assert.strictEqual(noQuery, null, 'NONE response must return null');

  console.log('✅ rewriteQuery NONE → null');

  // --- rewriteQuery: handles fetch failure gracefully ---

  const mockFetchFail = async () => { throw new Error('connection refused'); };
  const clientFail = createStewardClient({ fetch: mockFetchFail });
  const failedQuery = await clientFail.rewriteQuery('hello');
  assert.strictEqual(failedQuery, null, 'fetch failure should return null, not throw');

  console.log('✅ rewriteQuery fetch failure → null (no throw)');

  // --- rewriteQuery: handles non-OK response ---

  const mockFetch500 = async () => ({ ok: false, status: 500, json: async () => ({}) });
  const client500 = createStewardClient({ fetch: mockFetch500 });
  const q500 = await client500.rewriteQuery('hello');
  assert.strictEqual(q500, null, 'non-OK response should return null');

  console.log('✅ rewriteQuery non-OK → null');

  // --- judgeSave: happy path returning a save ---

  let judgeRequest = null;
  const mockFetchJudgeSave = async (url, init) => {
    judgeRequest = { url, init };
    return {
      ok: true,
      status: 200,
      json: async () => ({
        choices: [{ message: { content: '{"save":{"content":"New Rust projects default to edition 2024"}}' } }],
      }),
    };
  };

  const clientJudge = createStewardClient({ fetch: mockFetchJudgeSave });
  const saveResult = await clientJudge.judgeSave(
    'ugh, you created a Rust project with edition 2021 again. I always want 2024.',
    'Fixing Cargo.toml to use edition 2024.'
  );
  assert.strictEqual(saveResult, 'New Rust projects default to edition 2024', 'judgeSave returns content string when save is non-null');

  // verify the user message contains both the user msg and assistant msg
  const judgeBody = JSON.parse(judgeRequest.init.body);
  assert.strictEqual(judgeBody.messages[0].role, 'system');
  assert.ok(judgeBody.messages[0].content.includes('coding exchanges'), 'system message must be the judge prompt');
  assert.ok(judgeBody.messages[1].content.includes('edition 2021'), 'user message must contain user text');
  assert.ok(judgeBody.messages[1].content.includes('Fixing Cargo.toml'), 'user message must contain assistant text');

  console.log('✅ judgeSave happy path returns save content');

  // --- judgeSave: returns null when save is null ---

  const mockFetchJudgeNull = async () => ({
    ok: true,
    status: 200,
    json: async () => ({ choices: [{ message: { content: '{"save":null}' } }] }),
  });

  const clientJudgeNull = createStewardClient({ fetch: mockFetchJudgeNull });
  const nullSave = await clientJudgeNull.judgeSave('run the tests', 'All tests passing.');
  assert.strictEqual(nullSave, null, 'null save returns null');

  console.log('✅ judgeSave null → null');

  // --- judgeSave: malformed JSON returns null (graceful) ---

  const mockFetchJudgeBad = async () => ({
    ok: true,
    status: 200,
    json: async () => ({ choices: [{ message: { content: 'not json at all lol' } }] }),
  });

  const clientJudgeBad = createStewardClient({ fetch: mockFetchJudgeBad });
  const badSave = await clientJudgeBad.judgeSave('hello', 'hi');
  assert.strictEqual(badSave, null, 'malformed JSON should return null, not throw');

  console.log('✅ judgeSave malformed JSON → null (no throw)');

  // --- custom baseURL and model are respected ---

  let customRequest = null;
  const mockFetchCustom = async (url, init) => {
    customRequest = { url, init };
    return { ok: true, status: 200, json: async () => ({ choices: [{ message: { content: 'ok' } }] }) };
  };

  const customClient = createStewardClient({
    fetch: mockFetchCustom,
    baseURL: 'http://localhost:9999',
    model: 'custom-model',
    temperature: 0.7,
  });
  await customClient.rewriteQuery('hello');
  assert.strictEqual(customRequest.url, 'http://localhost:9999/v1/chat/completions');
  const customBody = JSON.parse(customRequest.init.body);
  assert.strictEqual(customBody.model, 'custom-model');
  assert.strictEqual(customBody.temperature, 0.7);

  console.log('✅ Custom baseURL, model, temperature respected');

  console.log('\n✅ All steward client tests passed');
})();
