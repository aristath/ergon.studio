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

  // --- debate unit tests ---

  {
    const debateSessions = new Map();
    const debatePrompts = [];
    let debateSessionCounter = 0;

    const debateClient = {
      app: { log: async () => {} },
      session: {
        create: async ({ body }) => {
          const id = `debate-session-${++debateSessionCounter}`;
          debateSessions.set(id, { body, deleted: false });
          return { data: { id } };
        },
        prompt: async ({ path: p, body }) => {
          debatePrompts.push({ path: p, body });
          if (body.agent === 'coder') {
            return {
              data: {
                parts: [{ type: 'text', text: 'Implemented the first pass.\nVerdict: CONTINUE' }],
              },
            };
          }
          return {
            data: {
              parts: [{ type: 'text', text: 'Reviewed the first pass. This is optimal.\nVerdict: AGREE' }],
            },
          };
        },
        delete: async ({ path: p }) => {
          const s = debateSessions.get(p.id);
          if (s) s.deleted = true;
          return { data: true };
        },
      },
    };

    const debatePlugin = await ErgonPlugin({ client: debateClient, directory: '/tmp' });
    assert.ok(debatePlugin.tool?.debate, 'debate tool should be registered');
    assert.strictEqual(typeof debatePlugin.tool.debate.execute, 'function', 'debate should have execute');

    const debateOutput = await debatePlugin.tool.debate.execute(
      { agent_a: 'coder', agent_b: 'reviewer', task: 'Improve the parser.' },
      context,
    );

    assert.strictEqual(debateSessions.size, 2, 'debate should create one session per participant');
    for (const [id, session] of debateSessions) {
      assert.ok(session.deleted, `debate session ${id} should be deleted after use`);
      assert.strictEqual(session.body.parentID, 'parent-123', 'debate sessions should carry parent ID');
    }

    assert.strictEqual(debatePrompts.length, 2, 'debate should stop when the second agent agrees');
    assert.strictEqual(debatePrompts[0].body.agent, 'coder', 'agent A takes the first turn');
    assert.strictEqual(debatePrompts[1].body.agent, 'reviewer', 'agent B takes the second turn');
    assert.ok(
      debatePrompts[1].body.parts[0].text.includes('Implemented the first pass'),
      'second agent should receive the first agent output',
    );
    assert.ok(debateOutput.includes('Status: AGREE'), 'debate output should report agreement');
    assert.ok(debateOutput.includes('## Latest response'), 'debate output should include latest response');
    assert.ok(debateOutput.includes('## Transcript'), 'debate output should include transcript');

    console.log('✅ debate unit tests passed');
  }

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

  // --- quality gate prompt contract ---

  const rootAgentDir = path.resolve(__dirname, '..', 'agents');
  const qualityPrompt = readFileSync(path.join(rootAgentDir, 'quality_controller.md'), 'utf8');
  const orchestratorPrompt = readFileSync(path.join(rootAgentDir, 'orchestrator.md'), 'utf8');
  const piAgentDir = path.resolve(__dirname, '..', 'pi', 'packages', 'orchestrator-mode', 'agents');
  const piQualityPrompt = readFileSync(path.join(piAgentDir, 'quality_controller.md'), 'utf8');
  const piOrchestratorPrompt = readFileSync(path.join(piAgentDir, 'orchestrator.md'), 'utf8');

  assert.strictEqual(piQualityPrompt, qualityPrompt, 'OpenCode and Pi quality-controller prompts must stay in sync');
  assert.strictEqual(piOrchestratorPrompt, orchestratorPrompt, 'OpenCode and Pi orchestrator prompts must stay in sync');
  assert.ok(!qualityPrompt.includes('COMPLETION.md'), 'quality gate must not depend on a project-level checklist');
  assert.ok(qualityPrompt.includes('### Phase 3: Verification Evidence'), 'quality gate must verify per-task evidence');
  assert.ok(qualityPrompt.includes('exactly `Verdict: APPROVED` or `Verdict: REJECTED`'), 'quality gate must require an exact terminal verdict');
  assert.ok(qualityPrompt.includes('parent orchestrator owns retry state'), 'quality controller must not own retry state');
  assert.ok(orchestratorPrompt.includes('changes executable behavior'), 'orchestrator must scope the quality gate to behavior changes');
  assert.ok(orchestratorPrompt.includes('Track the rejection count in this parent session'), 'orchestrator must own retry state');
  assert.ok(!existsSync(path.resolve(__dirname, '..', '.ergon.studio', 'COMPLETION.md')), 'legacy completion checklist must be removed');

  console.log('✅ Quality gate prompt contract passed');

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

  // --- run_parallel does not call app.agents during plugin init ---
  // Calling OpenCode's /agent endpoint while OpenCode is itself building the
  // agent list causes recursive plugin initialization and long startup hangs.
  // Keep run_parallel's agent argument permissive at schema level; OpenCode
  // remains the source of truth when session.prompt executes.

  {
    const { z } = await import('zod');
    let agentsCalled = false;
    const clientWithAgents = {
      app: {
        log: async () => {},
        agents: async () => {
          agentsCalled = true;
          return { data: [{ name: 'researcher' }] };
        },
      },
    };

    const start = Date.now();
    const p = await ErgonPlugin({ client: clientWithAgents, directory: '/tmp' });
    const elapsed = Date.now() - start;

    assert.strictEqual(agentsCalled, false, 'plugin init must not call app.agents');
    assert.ok(elapsed < 1000, `plugin init should be immediate (took ${elapsed}ms)`);
    const argsSchema = z.object(p.tool.run_parallel.args);
    assert.doesNotThrow(
      () => argsSchema.parse({ tasks: [{ agent: 'literally-anything', brief: 'x' }] }),
      'run_parallel keeps a permissive string schema',
    );
    console.log('✅ run_parallel does not call app.agents during plugin init');
  }

  // --- run_parallel falls back to string schema when agents lookup unavailable ---
  // The mock clients used by other tests don't expose client.app.agents.
  // The plugin must still load (just without validation) so existing tests
  // and any non-opencode harness keep working.

  {
    const { z } = await import('zod');
    const argsSchema = z.object(pluginWithFailure.tool.run_parallel.args);
    assert.doesNotThrow(
      () => argsSchema.parse({ tasks: [{ agent: 'literally-anything', brief: 'x' }] }),
      'fallback string schema must accept any agent name',
    );
    console.log('✅ run_parallel falls back to string schema without app.agents');
  }

  // --- run_parallel falls back when app.agents() throws ---
  // A flaky transport, transient server hiccup, or any other thrown error
  // during the lookup must not crash plugin init — degrade to permissive
  // string schema instead.

  {
    const { z } = await import('zod');
    const throwingClient = {
      app: {
        log: async () => {},
        agents: async () => { throw new Error('boom'); },
      },
    };
    const p = await ErgonPlugin({ client: throwingClient, directory: '/tmp' });
    const argsSchema = z.object(p.tool.run_parallel.args);
    assert.doesNotThrow(
      () => argsSchema.parse({ tasks: [{ agent: 'literally-anything', brief: 'x' }] }),
      'thrown agents() must fall back to string schema',
    );
    assert.ok(
      !p.tool.run_parallel.description.includes('Valid agent names:'),
      'description should omit the live-list line when fallback is used',
    );
    console.log('✅ run_parallel falls back when app.agents() throws');
  }

  // --- run_parallel falls back when app.agents() returns non-array data ---
  // Defensive against API drift: if opencode ever changes the response shape,
  // we degrade gracefully instead of crashing.

  {
    const { z } = await import('zod');
    const weirdClient = {
      app: {
        log: async () => {},
        agents: async () => ({ data: { not: 'an array' } }),
      },
    };
    const p = await ErgonPlugin({ client: weirdClient, directory: '/tmp' });
    const argsSchema = z.object(p.tool.run_parallel.args);
    assert.doesNotThrow(
      () => argsSchema.parse({ tasks: [{ agent: 'literally-anything', brief: 'x' }] }),
      'non-array data must fall back to string schema',
    );
    console.log('✅ run_parallel falls back on non-array agents response');
  }

  // --- run_parallel falls back when filtered list is empty ---
  // Zod's enum requires a non-empty tuple. If every entry is unusable
  // (missing name, wrong type, empty string), we must not try to build
  // an empty enum — fall back to string instead.

  {
    const { z } = await import('zod');
    const emptyClient = {
      app: {
        log: async () => {},
        agents: async () => ({ data: [{}, { name: null }, { name: '' }, { name: 42 }] }),
      },
    };
    const p = await ErgonPlugin({ client: emptyClient, directory: '/tmp' });
    const argsSchema = z.object(p.tool.run_parallel.args);
    assert.doesNotThrow(
      () => argsSchema.parse({ tasks: [{ agent: 'literally-anything', brief: 'x' }] }),
      'all-invalid agent list must fall back to string schema',
    );
    console.log('✅ run_parallel falls back when filtered agent list is empty');
  }

  // --- config hook drops agent models the provider doesn't actually have ---
  // An agent pins local/foo. We resolve "local" from the config (its baseURL)
  // and ask THAT provider what it serves. local/good-model is there → keep it;
  // local/foo is not → drop it so opencode uses the default. Same for any
  // other provider, looked up the same way from the config.

  {
    const logs = [];
    const client = { app: { log: async ({ body }) => { logs.push(body); } } };
    const origFetch = globalThis.fetch;
    globalThis.fetch = async (url) => {
      const u = String(url);
      if (u === 'http://local.test/v1/models') return { ok: true, json: async () => ({ data: [{ id: 'good-model' }] }) };
      if (u === 'http://other.test/v1/models') return { ok: true, json: async () => ({ data: [{ id: 'default-model' }] }) };
      return { ok: false, json: async () => ({}) };
    };

    try {
      const p = await ErgonPlugin({ client, directory: '/tmp' });
      assert.strictEqual(typeof p.config, 'function', 'config hook must be registered');

      const config = {
        provider: {
          local: { options: { baseURL: 'http://local.test/v1' } },
          other: { options: { baseURL: 'http://other.test/v1' } },
        },
        agent: {
          coder: { model: 'local/good-model', temperature: 0.2 },
          reviewer: { model: 'local/foo', permission: { edit: 'deny' } },
          critic: { model: 'not-a-model-ref', mode: 'subagent' },
        },
        mode: {
          legacy: { model: 'other/missing-model' },
        },
      };

      await p.config(config);

      assert.strictEqual(config.agent.coder.model, 'local/good-model', 'model the provider has must be preserved');
      assert.strictEqual(config.agent.coder.temperature, 0.2, 'other agent config must be preserved');
      assert.strictEqual(config.agent.reviewer.model, undefined, 'model the provider lacks must be removed');
      assert.deepStrictEqual(config.agent.reviewer.permission, { edit: 'deny' }, 'other invalid-agent fields preserved');
      assert.strictEqual(config.agent.critic.model, undefined, 'malformed model ref must be removed');
      assert.strictEqual(config.agent.critic.mode, 'subagent', 'malformed model removal preserves other fields');
      assert.strictEqual(config.mode.legacy.model, undefined, 'every provider gets the same check');
      assert.ok(
        logs.some((l) => /local\/foo.*not found/i.test(l.message ?? '')),
        'dropping a missing model should be logged',
      );
    } finally {
      globalThis.fetch = origFetch;
    }

    console.log('✅ config hook drops models the provider does not have');
  }

  // --- config hook does not guess when the provider can't be reached ---
  // No baseURL / unreachable provider → we can't ask, so we don't touch a
  // syntactically valid reference. Only malformed strings are removed.

  {
    const logs = [];
    const client = { app: { log: async ({ body }) => { logs.push(body); } } };
    const p = await ErgonPlugin({ client, directory: '/tmp' });
    const config = {
      // 'local' has no provider entry → no baseURL → cannot be probed.
      agent: {
        coder: { model: 'local/foo' },
        critic: { model: 'broken' },
      },
    };

    await p.config(config);

    assert.strictEqual(config.agent.coder.model, 'local/foo', 'unverifiable model is left alone');
    assert.strictEqual(config.agent.critic.model, undefined, 'malformed model still removed');
    assert.ok(
      logs.some((l) => /not validated|returned no model list/i.test(l.message ?? '')),
      'inability to validate should be logged',
    );

    console.log('✅ config hook does not guess when provider is unreachable');
  }

  // --- config hook does not hang startup when the provider's endpoint stalls ---
  // The config hook runs inside opencode's awaited startup. If the provider's
  // /models request never resolves, opencode's entire launch would block with
  // it — a frozen process, no TUI, no error (the bug that actually shipped).
  // The probe MUST be bounded; on timeout we leave the reference untouched.

  {
    const logs = [];
    const client = { app: { log: async ({ body }) => { logs.push(body); } } };
    const origFetch = globalThis.fetch;
    globalThis.fetch = () => new Promise(() => { /* never resolves */ });

    try {
      // Small timeout so the suite stays fast while exercising the real path.
      const p = await createErgonPlugin({ providerLookupTimeoutMs: 200 })({
        client,
        directory: '/tmp',
      });
      const config = {
        provider: { local: { options: { baseURL: 'http://stall.test/v1' } } },
        agent: {
          coder: { model: 'local/foo' },
          critic: { model: 'broken' },
        },
      };

      const start = Date.now();
      await Promise.race([
        p.config(config),
        new Promise((_, reject) => setTimeout(() => reject(new Error('config hook hung')), 3000)),
      ]);
      const elapsed = Date.now() - start;

      assert.ok(elapsed < 3000, `config hook must complete even when the provider stalls (took ${elapsed}ms)`);
      assert.ok(elapsed >= 150, `config hook must actually wait the configured timeout (took ${elapsed}ms)`);
      assert.strictEqual(config.agent.coder.model, 'local/foo', 'unverifiable model preserved when probe times out');
      assert.strictEqual(config.agent.critic.model, undefined, 'malformed model still removed on timeout');
      assert.ok(
        logs.some((l) => /not validated|returned no model list/i.test(l.message ?? '')),
        'probe timeout must be logged so the user can diagnose it',
      );
    } finally {
      globalThis.fetch = origFetch;
    }

    console.log('✅ config hook does not hang startup when the provider endpoint stalls');
  }

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

  // --- session.idle must not attempt to re-judge the same exchange twice ---
  // session.idle fires once per turn in normal flow (Runner.onIdle in
  // sst/opencode prompt.ts), but a defensive idle re-fire (e.g. on session
  // resume, or a future opencode behaviour change) would cause the
  // steward to re-judge the SAME (user, assistant) pair. Each judge is an
  // LLM call, so duplicates are wasted cost AND risk duplicate memory
  // saves if judgeSave ever returns the same content twice on the same
  // input. Dedup by per-session "last attempted assistant message id".

  {
    const judgeCalls = [];
    const stewardSpy = {
      async rewriteQuery() { return null; },
      async judgeSave(u, a) {
        judgeCalls.push({ u, a });
        return null;
      },
    };
    const memoryNoop = { async recall() { return []; }, async save() {} };

    const fakeMessages = [
      { info: { role: 'user', id: 'u1' }, parts: [{ type: 'text', text: 'a question' }] },
      { info: { role: 'assistant', id: 'a1' }, parts: [{ type: 'text', text: 'an answer' }] },
    ];
    const mockClient = {
      app: { log: async () => {} },
      session: { messages: async () => ({ data: fakeMessages }) },
    };

    const plugin = await createErgonPlugin({ steward: stewardSpy, memory: memoryNoop })({
      client: mockClient,
      directory: '/tmp',
    });

    // Fire session.idle three times for the same exchange.
    for (let i = 0; i < 3; i++) {
      await plugin.event({ event: { type: 'session.idle', properties: { sessionID: 'sX' } } });
      await new Promise((r) => setImmediate(r));
    }

    assert.strictEqual(
      judgeCalls.length,
      1,
      `judgeSave must be called exactly once per unique exchange (got ${judgeCalls.length})`,
    );

    // After a NEW assistant message, dedup must release: the new exchange
    // gets judged.
    fakeMessages.push(
      { info: { role: 'user', id: 'u2' }, parts: [{ type: 'text', text: 'follow-up' }] },
      { info: { role: 'assistant', id: 'a2' }, parts: [{ type: 'text', text: 'follow-up answer' }] },
    );
    await plugin.event({ event: { type: 'session.idle', properties: { sessionID: 'sX' } } });
    await new Promise((r) => setImmediate(r));

    assert.strictEqual(
      judgeCalls.length,
      2,
      `new exchange must be judged (got ${judgeCalls.length})`,
    );

    console.log('✅ session.idle dedups by last attempted assistant message id');
  }

  // --- run_parallel must reject empty tasks array ---
  // An empty tasks array currently passes schema validation, execute()
  // returns "" (Promise.all([]) → []), and the LLM sees nothing came back.
  // The model has no signal to course-correct and may either hallucinate
  // success or infinite-loop trying to figure out what happened. Reject
  // at the schema layer with a meaningful error.

  {
    const { z } = await import('zod');
    const argsSchema = z.object(pluginWithFailure.tool.run_parallel.args);
    assert.throws(
      () => argsSchema.parse({ tasks: [] }),
      'empty tasks array must be rejected by schema',
    );
    console.log('✅ run_parallel rejects empty tasks array');
  }

  // --- pendingRecall map must not leak when session ends without system.transform ---
  // The chat.message hook stashes a recall block in pendingRecall keyed by
  // sessionID; system.transform consumes and clears it. If a session is
  // aborted, errors out, or is deleted before system.transform runs (e.g.
  // user cancels mid-turn), the entry is orphaned. Over many sessions in a
  // long-lived process this is an unbounded memory leak.
  //
  // The plugin must clean up the entry when opencode publishes session.deleted.

  {
    const stewardOK = {
      async rewriteQuery() { return 'a query'; },
      async judgeSave() { return null; },
    };
    const memoryWithHits = {
      async recall() { return [{ id: 'm1', content: 'note', score: 0.9 }]; },
      async save() {},
    };
    const plugin = await createErgonPlugin({
      steward: stewardOK,
      memory: memoryWithHits,
    })({
      client: { app: { log: async () => {} } },
      directory: '/tmp',
    });

    // Simulate 50 distinct sessions: each one fires chat.message (which
    // populates pendingRecall) but never reaches system.transform, then
    // gets a session.deleted event (which should clean up).
    for (let i = 0; i < 50; i++) {
      const sessionID = `leaky-session-${i}`;
      await plugin['chat.message'](
        { sessionID },
        { message: { id: `m${i}` }, parts: [{ type: 'text', text: 'hello' }] },
      );
      // session ends without ever calling system.transform
      await plugin.event({ event: { type: 'session.deleted', properties: { sessionID } } });
    }

    // The internal map is private. Probe it indirectly: if we re-enter
    // system.transform for any of those sessionIDs, no recall block should
    // be injected (because session.deleted should have cleared it).
    const sysOut = { system: [] };
    await plugin['experimental.chat.system.transform']({ sessionID: 'leaky-session-0' }, sysOut);
    assert.strictEqual(sysOut.system.length, 1, 'one entry: scratchpad only');
    assert.ok(
      !sysOut.system[0].includes('Relevant prior notes'),
      'session.deleted must purge orphaned recall — no leak into a later transform',
    );

    console.log('✅ pendingRecall is purged on session.deleted (no leak)');
  }

  // --- chat.message must not hang the user's turn when steward stalls ---
  // chat.message is awaited by opencode (unlike the fire-and-forget event
  // hook), so a hung steward.rewriteQuery() stalls every user message in
  // the session. The hook must bound external calls with a timeout and
  // fall through gracefully when they exceed it.

  {
    let logged = null;
    const stewardThatHangs = {
      async rewriteQuery() { return new Promise(() => {}); /* never */ },
      async judgeSave() { return null; },
    };
    const memoryStub = {
      async recall() { return []; },
      async save() {},
    };
    const plugin = await createErgonPlugin({
      steward: stewardThatHangs,
      memory: memoryStub,
      chatMessageTimeoutMs: 200,
    })({
      client: { app: { log: async ({ body }) => { logged = body; } } },
      directory: '/tmp',
    });

    const start = Date.now();
    const output = { message: { id: 'm1' }, parts: [{ type: 'text', text: 'hello' }] };
    await Promise.race([
      plugin['chat.message']({ sessionID: 's1' }, output),
      new Promise((_, reject) => setTimeout(() => reject(new Error('chat.message hung')), 3000)),
    ]);
    const elapsed = Date.now() - start;

    assert.ok(elapsed < 3000, `chat.message must complete even when steward hangs (took ${elapsed}ms)`);
    assert.ok(elapsed >= 150, `chat.message must wait at least the configured timeout (took ${elapsed}ms)`);
    assert.strictEqual(output.parts.length, 1, 'parts must remain untouched on timeout');
    assert.ok(
      logged && /steward.*timeout|recall.*timeout|fallback|disabled/i.test(logged.message ?? ''),
      'timeout must be logged so the user can diagnose silent recall failures',
    );
    console.log('✅ chat.message bounds steward calls with a timeout');
  }

  // --- chat.message must not hang when memory.recall stalls ---
  // Same bug class as above but for the second external call. A working
  // steward followed by a hung memory backend must still bound the hook.

  {
    let logged = null;
    const stewardOK = {
      async rewriteQuery() { return 'a query'; },
      async judgeSave() { return null; },
    };
    const memoryThatHangs = {
      async recall() { return new Promise(() => {}); /* never */ },
      async save() {},
    };
    const plugin = await createErgonPlugin({
      steward: stewardOK,
      memory: memoryThatHangs,
      chatMessageTimeoutMs: 200,
    })({
      client: { app: { log: async ({ body }) => { logged = body; } } },
      directory: '/tmp',
    });

    const start = Date.now();
    await Promise.race([
      plugin['chat.message'](
        { sessionID: 's2' },
        { message: { id: 'm2' }, parts: [{ type: 'text', text: 'hello' }] },
      ),
      new Promise((_, reject) => setTimeout(() => reject(new Error('chat.message hung on memory')), 3000)),
    ]);
    const elapsed = Date.now() - start;

    assert.ok(elapsed < 3000, `chat.message must complete even when memory hangs (took ${elapsed}ms)`);
    assert.ok(
      logged && /memory.*timeout|recall.*timeout|fallback|disabled/i.test(logged.message ?? ''),
      'memory timeout must be logged',
    );
    console.log('✅ chat.message bounds memory.recall with a timeout');
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
