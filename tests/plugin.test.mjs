import assert from 'assert';
import { fileURLToPath } from 'url';
import path from 'path';
import { readFileSync, existsSync, mkdirSync, rmSync } from 'fs';
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
})();
