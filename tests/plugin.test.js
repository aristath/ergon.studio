const assert = require('assert');
const path = require('path');

(async () => {
  // Load the built plugin (ensure it exists)
  const pluginPath = path.resolve(__dirname, '..', 'dist', 'index.js');
  const { ErgonPlugin } = require(pluginPath);

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

  console.log('✅ ErgonPlugin test passed');
})();
