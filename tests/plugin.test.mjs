import assert from 'assert';
import { fileURLToPath } from 'url';
import path from 'path';
import { readFileSync, writeFileSync, existsSync, mkdirSync, rmSync } from 'fs';
import { execSync } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

(async () => {
  const pluginPath = path.resolve(__dirname, '..', 'dist', 'index.js');
  const { ErgonPlugin, createErgonPlugin } = await import(pluginPath);

  // Plugin requires a context with a client property
  let logCalled = false;
  const mockClient = {
    app: {
      log: async (data) => {
        logCalled = true;
        assert.strictEqual(data.body.service, 'ergon-plugin');
        assert.strictEqual(data.body.level, 'info');
        assert.strictEqual(data.body.message, 'Ergon session started');
      },
    },
  };

  const result = await ErgonPlugin({ client: mockClient });
  assert.strictEqual(typeof result, 'object', 'plugin should return an object');
  assert.strictEqual(typeof result.event, 'function', 'plugin should expose an event handler');

  // Test the event handler with session.created event
  await result.event({ event: { type: 'session.created' } });
  assert.ok(logCalled, 'client.app.log should be called for session.created event');

  console.log('✅ ErgonPlugin event hook test passed');

  // --- run_parallel unit tests ---

  const sessions = new Map();
  let sessionCounter = 0;

  const mockClientWithSession = {
    app: { log: async () => {} },
    session: {
      create: async ({ body }) => {
        const id = `session-${++sessionCounter}`;
        sessions.set(id, { body, deleted: false });
        return { data: { id } };
      },
      prompt: async ({ path, body }) => ({
        data: {
          parts: [{ type: 'text', text: `output from ${body.agent}` }],
        },
      }),
      delete: async ({ path }) => {
        const s = sessions.get(path.id);
        if (s) s.deleted = true;
        return { data: true };
      },
    },
  };

  const plugin = await ErgonPlugin({ client: mockClientWithSession });

  assert.ok(plugin.tool?.run_parallel, 'run_parallel tool should be registered');
  assert.strictEqual(typeof plugin.tool.run_parallel.execute, 'function', 'run_parallel should have execute');
  assert.ok(plugin.tool.run_parallel.description.length > 0, 'run_parallel should have a description');

  const context = {
    sessionID: 'parent-123',
    messageID: 'msg-1',
    agent: 'orchestrator',
    directory: '/tmp',
    worktree: '/tmp',
    abort: new AbortController().signal,
    metadata: () => {},
    ask: async () => {},
  };

  const tasks = [
    { agent: 'researcher', brief: 'question A' },
    { agent: 'critic', brief: 'question B' },
  ];

  const output = await plugin.tool.run_parallel.execute({ tasks }, context);

  assert.strictEqual(sessions.size, 2, 'should create one session per task');

  for (const [id, session] of sessions) {
    assert.ok(session.deleted, `session ${id} should be deleted after use`);
    assert.strictEqual(session.body.parentID, 'parent-123', 'sessions should carry parent ID');
  }

  assert.ok(output.includes('## researcher'), 'output should have researcher heading');
  assert.ok(output.includes('## critic'), 'output should have critic heading');
  assert.ok(output.includes('---'), 'output should have separator between results');
  assert.ok(output.includes('output from researcher'), 'output should include researcher response text');
  assert.ok(output.includes('output from critic'), 'output should include critic response text');

  console.log('✅ run_parallel unit tests passed');

  // --- scratchpad skill validation ---

  const skillPath = path.resolve(__dirname, '..', 'skills', 'scratchpad', 'SKILL.md');
  assert.ok(existsSync(skillPath), 'SKILL.md must exist at skills/scratchpad/SKILL.md');

  const skillContent = readFileSync(skillPath, 'utf8');
  assert.ok(skillContent.startsWith('---'), 'SKILL.md must have YAML frontmatter');
  assert.ok(/^name:\s*scratchpad\s*$/m.test(skillContent), 'skill name must be "scratchpad"');
  assert.ok(/^description:\s*.+/m.test(skillContent), 'skill must have a non-empty description');
  assert.ok(skillContent.includes('.ergon.studio/scratchpad.md'), 'skill must reference .ergon.studio/scratchpad.md');
  assert.ok(skillContent.includes('## Conventions'), 'skill must reference Conventions section');
  assert.ok(skillContent.includes('## Notes'), 'skill must reference Notes section');

  console.log('✅ Scratchpad skill file validation passed');

  // --- ergon init installs skill globally ---

  const tmpConfig = path.resolve(__dirname, '..', 'tmp-ergon-test');
  mkdirSync(tmpConfig, { recursive: true });
  try {
    execSync('node dist/cli.js init', {
      env: { ...process.env, XDG_CONFIG_HOME: tmpConfig },
      cwd: path.resolve(__dirname, '..'),
    });
    const installedSkill = path.join(tmpConfig, 'opencode', 'skills', 'scratchpad', 'SKILL.md');
    assert.ok(existsSync(installedSkill), 'ergon init must install SKILL.md to opencode/skills/scratchpad/');
  } finally {
    rmSync(tmpConfig, { recursive: true, force: true });
  }

  console.log('✅ ergon init installs scratchpad skill');

  // --- handoff skill validation ---

  const handoffSkillPath = path.resolve(__dirname, '..', 'skills', 'handoff', 'SKILL.md');
  assert.ok(existsSync(handoffSkillPath), 'SKILL.md must exist at skills/handoff/SKILL.md');

  const handoffSkillContent = readFileSync(handoffSkillPath, 'utf8');
  assert.ok(handoffSkillContent.startsWith('---'), 'handoff SKILL.md must have YAML frontmatter');
  assert.ok(/^name:\s*handoff\s*$/m.test(handoffSkillContent), 'skill name must be "handoff"');
  assert.ok(/^description:\s*.+/m.test(handoffSkillContent), 'skill must have a non-empty description');
  assert.ok(handoffSkillContent.includes('HANDOFF.md'), 'skill must reference HANDOFF.md');

  console.log('✅ Handoff skill file validation passed');

  // --- run_parallel error handling ---

  const mockClientWithFailure = {
    app: { log: async () => {} },
    session: {
      create: async ({ body }) => {
        const id = `session-err-${body.title.includes('researcher') ? '1' : '2'}`;
        return { data: { id } };
      },
      prompt: async ({ path: p, body }) => {
        if (body.agent === 'researcher') throw new Error('LLM unavailable');
        return { data: { parts: [{ type: 'text', text: `output from ${body.agent}` }] } };
      },
      delete: async () => ({ data: true }),
    },
  };

  const pluginWithFailure = await ErgonPlugin({ client: mockClientWithFailure, directory: '/tmp' });
  const failOutput = await pluginWithFailure.tool.run_parallel.execute(
    { tasks: [{ agent: 'researcher', brief: 'q' }, { agent: 'critic', brief: 'q' }] },
    { ...context }
  );

  assert.ok(failOutput.includes('## researcher'), 'failed task should still have a section');
  assert.ok(failOutput.includes('⚠️'), 'failed task should have error marker');
  assert.ok(failOutput.includes('LLM unavailable'), 'error message should appear in output');
  assert.ok(failOutput.includes('output from critic'), 'successful task result should still appear');

  console.log('✅ run_parallel error handling test passed');

  // --- auto-inject conventions via experimental hooks ---

  const tmpConventionsDir = path.resolve(__dirname, '..', 'tmp-conventions-test');
  const conventionsPath = path.join(tmpConventionsDir, '.ergon.studio', 'scratchpad.md');
  mkdirSync(path.dirname(conventionsPath), { recursive: true });
  writeFileSync(conventionsPath, '## Conventions\n\nFix lint issues, never suppress them\n\n## Notes\n\nCan\'t use fs.watch on NFS mounts — use polling\n');

  try {
    const pluginWithConventions = await ErgonPlugin({ client: { app: { log: async () => {} } }, directory: tmpConventionsDir });

    // system.transform should inject conventions
    const systemOutput = { system: [] };
    await pluginWithConventions['experimental.chat.system.transform']({}, systemOutput);
    assert.ok(systemOutput.system.length > 0, 'system transform should inject content when conventions.md exists');
    assert.ok(systemOutput.system.some(s => s.includes('Fix lint issues')), 'system should include scratchpad content');
    assert.ok(systemOutput.system.some(s => s.includes('NFS mounts')), 'system should include notes content');

    // compacting hook should preserve scratchpad through context compression
    const compactOutput = { context: [] };
    await pluginWithConventions['experimental.session.compacting']({ sessionID: 'ses_test' }, compactOutput);
    assert.ok(compactOutput.context.length > 0, 'compacting hook should inject scratchpad into context');
    assert.ok(compactOutput.context.some(s => s.includes('Fix lint issues')), 'compacting context should include scratchpad content');

    // no scratchpad.md → hooks should inject bootstrap stub
    const pluginNoScratchpad = await ErgonPlugin({ client: { app: { log: async () => {} } }, directory: '/tmp' });
    const stubSystem = { system: [] };
    await pluginNoScratchpad['experimental.chat.system.transform']({}, stubSystem);
    assert.strictEqual(stubSystem.system.length, 1, 'bootstrap stub injected when scratchpad.md is absent');
    assert.ok(stubSystem.system[0].includes('scratchpad.md'), 'stub should reference scratchpad.md');
  } finally {
    rmSync(tmpConventionsDir, { recursive: true, force: true });
  }

  console.log('✅ Auto-inject conventions tests passed');

  // --- ergon update copies files but does not touch opencode.json ---

  const tmpUpdateConfig = path.resolve(__dirname, '..', 'tmp-update-test');
  const updateConfigDir = path.join(tmpUpdateConfig, 'opencode');
  mkdirSync(updateConfigDir, { recursive: true });
  const existingConfig = { '$schema': 'https://opencode.ai/config.json', 'model': 'local/my-model', 'custom_key': 'preserved' };
  const updateConfigPath = path.join(updateConfigDir, 'opencode.json');
  writeFileSync(updateConfigPath, JSON.stringify(existingConfig, null, 2));

  try {
    execSync('node dist/cli.js update', {
      env: { ...process.env, XDG_CONFIG_HOME: tmpUpdateConfig },
      cwd: path.resolve(__dirname, '..'),
    });
    const afterConfig = JSON.parse(readFileSync(updateConfigPath, 'utf8'));
    assert.deepStrictEqual(afterConfig, existingConfig, 'ergon update must not modify opencode.json');

    const updatedAgent = path.join(updateConfigDir, 'agents', 'orchestrator.md');
    assert.ok(existsSync(updatedAgent), 'ergon update must copy agent files');

    const updatedSkill = path.join(updateConfigDir, 'skills', 'scratchpad', 'SKILL.md');
    assert.ok(existsSync(updatedSkill), 'ergon update must copy skill files');
  } finally {
    rmSync(tmpUpdateConfig, { recursive: true, force: true });
  }

  console.log('✅ ergon update test passed');

  // ===========================================================================
  // Memory steward integration tests
  // ===========================================================================
  //
  // The memory steward adds two integration points to the plugin:
  //   - chat.message hook   → recall path (rewrite query, search memory, inject)
  //   - event/session.idle  → save path   (judge exchange, store memory)
  //
  // Both are tested with injected stub steward and memory clients via the
  // exported createErgonPlugin factory.

  function makeStewardStub(overrides = {}) {
    const calls = { rewrite: [], judge: [] };
    const stub = {
      async rewriteQuery(text) {
        calls.rewrite.push(text);
        return overrides.rewriteResult ?? null;
      },
      async judgeSave(userMsg, assistantMsg) {
        calls.judge.push({ userMsg, assistantMsg });
        return overrides.judgeResult ?? null;
      },
    };
    return { stub, calls };
  }

  function makeMemoryStub(overrides = {}) {
    const calls = { recall: [], save: [] };
    const stub = {
      async recall(query, limit) {
        calls.recall.push({ query, limit });
        return overrides.recallResult ?? [];
      },
      async save(content) {
        calls.save.push(content);
      },
    };
    return { stub, calls };
  }

  // --- chat.message: hook is registered ---

  {
    const { stub: steward } = makeStewardStub();
    const { stub: memory } = makeMemoryStub();
    const plugin = await createErgonPlugin({ steward, memory })({
      client: { app: { log: async () => {} } },
      directory: '/tmp',
    });
    assert.strictEqual(typeof plugin['chat.message'], 'function', 'chat.message hook must be registered');
    console.log('✅ chat.message hook registered');
  }

  // --- chat.message: empty user text → no-op ---

  {
    const { stub: steward, calls: stewardCalls } = makeStewardStub();
    const { stub: memory, calls: memCalls } = makeMemoryStub();
    const plugin = await createErgonPlugin({ steward, memory })({
      client: { app: { log: async () => {} } }, directory: '/tmp',
    });
    const output = { message: { id: 'msg1' }, parts: [] };
    await plugin['chat.message']({ sessionID: 's1' }, output);
    assert.strictEqual(stewardCalls.rewrite.length, 0, 'no rewrite call on empty text');
    assert.strictEqual(memCalls.recall.length, 0, 'no recall on empty text');
    assert.strictEqual(output.parts.length, 0, 'no part injected');
    console.log('✅ chat.message empty text → no-op');
  }

  // --- chat.message: rewriteQuery returns null → no recall, no inject ---

  {
    const { stub: steward, calls: stewardCalls } = makeStewardStub({ rewriteResult: null });
    const { stub: memory, calls: memCalls } = makeMemoryStub();
    const plugin = await createErgonPlugin({ steward, memory })({
      client: { app: { log: async () => {} } }, directory: '/tmp',
    });
    const output = { message: { id: 'msg1' }, parts: [{ type: 'text', text: 'thanks!' }] };
    await plugin['chat.message']({ sessionID: 's1' }, output);
    assert.strictEqual(stewardCalls.rewrite.length, 1, 'rewrite called once');
    assert.strictEqual(stewardCalls.rewrite[0], 'thanks!');
    assert.strictEqual(memCalls.recall.length, 0, 'no recall when rewrite returns null');
    assert.strictEqual(output.parts.length, 1, 'no extra part injected');
    console.log('✅ chat.message null rewrite → no recall');
  }

  // --- chat.message: rewrite ok but recall returns [] → no inject ---

  {
    const { stub: steward } = makeStewardStub({ rewriteResult: 'test rust' });
    const { stub: memory } = makeMemoryStub({ recallResult: [] });
    const plugin = await createErgonPlugin({ steward, memory })({
      client: { app: { log: async () => {} } }, directory: '/tmp',
    });
    const output = { message: { id: 'msg1' }, parts: [{ type: 'text', text: 'test the rust thing' }] };
    await plugin['chat.message']({ sessionID: 's1' }, output);
    assert.strictEqual(output.parts.length, 1, 'no extra part injected when no memories returned');
    console.log('✅ chat.message empty recall → no inject');
  }

  // --- chat.message: full happy path stashes recall, doesn't touch parts ---
  // The recall content is consumed by experimental.chat.system.transform on
  // the same turn (see the next test). chat.message must NOT mutate
  // output.parts — that path triggered "System message must be at the
  // beginning" errors from llama.cpp's Qwen Jinja template when serializing
  // multi-content user messages.

  {
    const { stub: steward, calls: stewardCalls } = makeStewardStub({ rewriteResult: 'rust edition' });
    const { stub: memory, calls: memCalls } = makeMemoryStub({
      recallResult: [
        { id: 'm1', content: 'New Rust projects default to edition 2024', score: 0.9 },
        { id: 'm2', content: 'Use `cargo new --edition 2024` explicitly', score: 0.8 },
      ],
    });
    const plugin = await createErgonPlugin({ steward, memory })({
      client: { app: { log: async () => {} } }, directory: '/tmp',
    });
    const output = {
      message: { id: 'msg-abc' },
      parts: [{ type: 'text', text: "let's create a new rust project" }],
    };
    await plugin['chat.message']({ sessionID: 'sess-1' }, output);

    assert.strictEqual(stewardCalls.rewrite[0], "let's create a new rust project");
    assert.strictEqual(memCalls.recall[0].query, 'rust edition');
    // Critical: parts MUST stay untouched. Inserting our content as a TextPart
    // breaks the Qwen chat template downstream.
    assert.strictEqual(output.parts.length, 1, 'parts must not be mutated');
    assert.strictEqual(output.parts[0].text, "let's create a new rust project", 'user text untouched');
    console.log('✅ chat.message stashes recall, leaves parts untouched');
  }

  // --- chat.message → experimental.chat.system.transform integration ---
  // This is the cross-hook flow: chat.message stashes the recall in a
  // session-keyed map, system.transform reads it on the same turn and
  // pushes it into output.system alongside the scratchpad. After
  // consumption, the map slot is cleared so the next turn starts clean.

  {
    const { stub: steward } = makeStewardStub({ rewriteResult: 'rust edition' });
    const { stub: memory } = makeMemoryStub({
      recallResult: [
        { id: 'm1', content: 'New Rust projects default to edition 2024', score: 0.9 },
        { id: 'm2', content: 'Use `cargo new --edition 2024` explicitly', score: 0.8 },
      ],
    });
    const plugin = await createErgonPlugin({ steward, memory })({
      client: { app: { log: async () => {} } }, directory: '/tmp',
    });

    // Turn 1: chat.message stashes recall
    await plugin['chat.message'](
      { sessionID: 'sess-X' },
      { message: { id: 'm1' }, parts: [{ type: 'text', text: 'create a rust project' }] },
    );

    // Same turn: system.transform consumes it.
    // CRITICAL invariant: we must produce exactly ONE system entry, not
    // multiple, regardless of how many additions we have. Strict chat
    // templates (Qwen 3.5) reject more than one system message.
    const sysOut = { system: [] };
    await plugin['experimental.chat.system.transform']({ sessionID: 'sess-X' }, sysOut);

    assert.strictEqual(sysOut.system.length, 1, 'must produce exactly one system entry');
    const entry = sysOut.system[0];
    assert.ok(entry.includes('Project Scratchpad'), 'scratchpad content present');
    assert.ok(entry.includes('Relevant prior notes'), 'recall block present');
    assert.ok(entry.includes('edition 2024'), 'first memory present');
    assert.ok(entry.includes('cargo new'), 'second memory present');

    // Same single-entry guarantee when there's a pre-existing entry
    // (e.g. an agent identity opencode put there before our hook).
    await plugin['chat.message'](
      { sessionID: 'sess-Y' },
      { message: { id: 'm2' }, parts: [{ type: 'text', text: 'create a rust project' }] },
    );
    const sysOut1 = { system: ['## Identity\nYou are Scout.'] };
    await plugin['experimental.chat.system.transform']({ sessionID: 'sess-Y' }, sysOut1);
    assert.strictEqual(sysOut1.system.length, 1, 'pre-existing entry preserved as single entry');
    assert.ok(sysOut1.system[0].startsWith('## Identity'), 'pre-existing identity stays first');
    assert.ok(sysOut1.system[0].includes('Project Scratchpad'), 'scratchpad appended to existing entry');
    assert.ok(sysOut1.system[0].includes('Relevant prior notes'), 'recall appended to existing entry');

    // Second system.transform call for the same session must NOT re-inject recall
    const sysOut2 = { system: [] };
    await plugin['experimental.chat.system.transform']({ sessionID: 'sess-X' }, sysOut2);
    assert.strictEqual(sysOut2.system.length, 1, 'still single entry');
    assert.ok(!sysOut2.system[0].includes('Relevant prior notes'), 'recall consumed, not re-injected');

    console.log('✅ chat.message → system.transform recall injection (single entry invariant)');
  }

  // --- experimental.chat.system.transform without prior chat.message: no recall, just scratchpad ---

  {
    const { stub: steward } = makeStewardStub();
    const { stub: memory } = makeMemoryStub();
    const plugin = await createErgonPlugin({ steward, memory })({
      client: { app: { log: async () => {} } }, directory: '/tmp',
    });
    const sysOut = { system: [] };
    await plugin['experimental.chat.system.transform']({ sessionID: 'sess-quiet' }, sysOut);
    assert.strictEqual(sysOut.system.length, 1, 'one system entry: scratchpad only');
    assert.ok(!sysOut.system[0].includes('Relevant prior notes'), 'no recall content present');
    assert.ok(sysOut.system[0].includes('Project Scratchpad'), 'scratchpad stub present');
    console.log('✅ system.transform without recall → scratchpad only');
  }

  // --- chat.message: extracts text from multiple text parts ---

  {
    const { stub: steward, calls: stewardCalls } = makeStewardStub({ rewriteResult: null });
    const { stub: memory } = makeMemoryStub();
    const plugin = await createErgonPlugin({ steward, memory })({
      client: { app: { log: async () => {} } }, directory: '/tmp',
    });
    const output = {
      message: { id: 'msg1' },
      parts: [
        { type: 'text', text: 'first line' },
        { type: 'text', text: 'second line' },
        { type: 'file', mime: 'text/plain', url: 'x' }, // non-text part ignored
      ],
    };
    await plugin['chat.message']({ sessionID: 's1' }, output);
    assert.strictEqual(stewardCalls.rewrite[0], 'first line\nsecond line');
    console.log('✅ chat.message joins multiple text parts');
  }

  // --- event/session.idle: triggers save path ---

  {
    const { stub: steward, calls: stewardCalls } = makeStewardStub({ judgeResult: 'New Rust projects default to edition 2024' });
    const { stub: memory, calls: memCalls } = makeMemoryStub();

    const fakeMessages = [
      { info: { role: 'user', id: 'u1' }, parts: [{ type: 'text', text: 'create a rust project' }] },
      { info: { role: 'assistant', id: 'a1' }, parts: [{ type: 'text', text: 'Created with edition 2024.' }] },
    ];

    let messagesCallArg = null;
    const mockClient = {
      app: { log: async () => {} },
      session: {
        messages: async (arg) => {
          messagesCallArg = arg;
          return { data: fakeMessages };
        },
      },
    };

    const plugin = await createErgonPlugin({ steward, memory })({ client: mockClient, directory: '/tmp' });
    await plugin.event({ event: { type: 'session.idle', properties: { sessionID: 'sess-99' } } });

    // session.idle dispatches handler asynchronously via void+catch — give it a tick to settle
    await new Promise((r) => setImmediate(r));

    assert.deepStrictEqual(messagesCallArg, { path: { id: 'sess-99' } });
    assert.strictEqual(stewardCalls.judge.length, 1, 'judgeSave called once');
    assert.strictEqual(stewardCalls.judge[0].userMsg, 'create a rust project');
    assert.strictEqual(stewardCalls.judge[0].assistantMsg, 'Created with edition 2024.');
    assert.strictEqual(memCalls.save.length, 1, 'memory.save called once');
    assert.strictEqual(memCalls.save[0], 'New Rust projects default to edition 2024');
    console.log('✅ session.idle full save path');
  }

  // --- event/session.idle: judge returns null → no save ---

  {
    const { stub: steward, calls: stewardCalls } = makeStewardStub({ judgeResult: null });
    const { stub: memory, calls: memCalls } = makeMemoryStub();
    const fakeMessages = [
      { info: { role: 'user' }, parts: [{ type: 'text', text: 'q' }] },
      { info: { role: 'assistant' }, parts: [{ type: 'text', text: 'a' }] },
    ];
    const mockClient = {
      app: { log: async () => {} },
      session: { messages: async () => ({ data: fakeMessages }) },
    };
    const plugin = await createErgonPlugin({ steward, memory })({ client: mockClient, directory: '/tmp' });
    await plugin.event({ event: { type: 'session.idle', properties: { sessionID: 's1' } } });
    await new Promise((r) => setImmediate(r));
    assert.strictEqual(stewardCalls.judge.length, 1);
    assert.strictEqual(memCalls.save.length, 0, 'no save when judge returns null');
    console.log('✅ session.idle null judge → no save');
  }

  // --- event/session.idle: messages fetch fails → silent no-op ---

  {
    const { stub: steward, calls: stewardCalls } = makeStewardStub();
    const { stub: memory } = makeMemoryStub();
    const mockClient = {
      app: { log: async () => {} },
      session: { messages: async () => { throw new Error('boom'); } },
    };
    const plugin = await createErgonPlugin({ steward, memory })({ client: mockClient, directory: '/tmp' });
    // Must not throw
    await plugin.event({ event: { type: 'session.idle', properties: { sessionID: 's1' } } });
    await new Promise((r) => setImmediate(r));
    assert.strictEqual(stewardCalls.judge.length, 0, 'no judge call when fetch fails');
    console.log('✅ session.idle fetch failure swallowed');
  }

  // --- event/session.idle: no user/assistant pair yet → no-op ---

  {
    const { stub: steward, calls: stewardCalls } = makeStewardStub();
    const { stub: memory } = makeMemoryStub();
    const fakeMessages = [
      { info: { role: 'user' }, parts: [{ type: 'text', text: 'just a question' }] },
      // no assistant response yet
    ];
    const mockClient = {
      app: { log: async () => {} },
      session: { messages: async () => ({ data: fakeMessages }) },
    };
    const plugin = await createErgonPlugin({ steward, memory })({ client: mockClient, directory: '/tmp' });
    await plugin.event({ event: { type: 'session.idle', properties: { sessionID: 's1' } } });
    await new Promise((r) => setImmediate(r));
    assert.strictEqual(stewardCalls.judge.length, 0, 'no judge when only user message exists');
    console.log('✅ session.idle without assistant response → no-op');
  }

  // --- session.created still works after refactor (regression check) ---

  {
    let logCalled = false;
    const { stub: steward } = makeStewardStub();
    const { stub: memory } = makeMemoryStub();
    const mockClient = { app: { log: async () => { logCalled = true; } } };
    const plugin = await createErgonPlugin({ steward, memory })({ client: mockClient, directory: '/tmp' });
    await plugin.event({ event: { type: 'session.created' } });
    assert.ok(logCalled, 'session.created still logs after refactor');
    console.log('✅ session.created regression');
  }

  console.log('\n✅ All memory steward integration tests passed');
})();
