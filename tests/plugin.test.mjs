import assert from 'assert';
import { fileURLToPath } from 'url';
import path from 'path';
import { readFileSync, writeFileSync, existsSync, mkdirSync, rmSync } from 'fs';
import { execSync } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

(async () => {
  const pluginPath = path.resolve(__dirname, '..', 'dist', 'index.js');
  const { ErgonPlugin } = await import(pluginPath);

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
  assert.ok(skillContent.includes('.ergon.studio/scratchpads'), 'skill must reference .ergon.studio/scratchpads');
  assert.ok(skillContent.includes('index.md'), 'skill must reference index.md');

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
  const conventionsPath = path.join(tmpConventionsDir, '.ergon.studio', 'scratchpads', 'conventions.md');
  mkdirSync(path.dirname(conventionsPath), { recursive: true });
  writeFileSync(conventionsPath, '# Project Conventions\n\n- Fix lint issues, never suppress them\n');

  try {
    const pluginWithConventions = await ErgonPlugin({ client: { app: { log: async () => {} } }, directory: tmpConventionsDir });

    // system.transform should inject conventions
    const systemOutput = { system: [] };
    await pluginWithConventions['experimental.chat.system.transform']({}, systemOutput);
    assert.ok(systemOutput.system.length > 0, 'system transform should inject content when conventions.md exists');
    assert.ok(systemOutput.system.some(s => s.includes('Fix lint issues')), 'system should include conventions content');

    // compacting hook should preserve conventions through context compression
    const compactOutput = { context: [] };
    await pluginWithConventions['experimental.session.compacting']({ sessionID: 'ses_test' }, compactOutput);
    assert.ok(compactOutput.context.length > 0, 'compacting hook should inject conventions into context');
    assert.ok(compactOutput.context.some(s => s.includes('Fix lint issues')), 'compacting context should include conventions');

    // no conventions.md → hooks should not inject anything
    const pluginNoConventions = await ErgonPlugin({ client: { app: { log: async () => {} } }, directory: '/tmp' });
    const emptySystem = { system: [] };
    await pluginNoConventions['experimental.chat.system.transform']({}, emptySystem);
    assert.strictEqual(emptySystem.system.length, 0, 'no injection when conventions.md is absent');
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
})();
