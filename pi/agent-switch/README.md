# ergon.studio Pi Agent Switch Extension

Pi extension that adds an interactive `/agent` command.

It provides:
- An interactive agent picker with cursor navigation.
- Direct selection by numeric index (`/agent 2`).
- Persisted active agent state for the current session.
- UI indicator below the editor and in the footer showing the active agent.

## Install

From a local checkout:

```bash
pi install ./pi/agent-switch
```

From npm (once published):

```bash
pi install npm:@ergon.studio/pi-agent-switch
```

The extension adds `/agent` and updates Pi UI with the active agent below the editor.

## Publish

From this package folder:

```bash
cd pi/agent-switch
git status --short   # confirm only intended package files changed
npm run pack:check
npm run release:patch
```

Preflight check:

```bash
npm run pack:check
```

Useful commands:

```bash
pi install ./pi/agent-switch            # local install
pi install npm:@ergon.studio/pi-agent-switch
```

## Command behavior

`/agent`
- Opens an interactive picker.

`/agent <name>`
- Switches to the named agent.

`/agent <number>`
- Switches by 1-based list index.

`/agent off`
- Clears active agent routing.
