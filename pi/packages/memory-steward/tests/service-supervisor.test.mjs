import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const packageDir = resolve(fileURLToPath(new URL("..", import.meta.url)));
const supervisorPath = join(
	packageDir,
	"scripts",
	"llama-server-supervisor.sh",
);

function tempBin(curlBody) {
	const dir = mkdtempSync(join(tmpdir(), "ergon-supervisor-"));
	const curlPath = join(dir, "curl");
	writeFileSync(curlPath, curlBody);
	chmodSync(curlPath, 0o755);
	return {
		dir,
		path: `${dir}:${process.env.PATH}`,
		cleanup() {
			rmSync(dir, { recursive: true, force: true });
		},
	};
}

test("supervisor accepts a healthy embedder response", () => {
	const fakeBin = tempBin(
		"#!/usr/bin/env bash\nprintf '%s\\n' '{\"data\":[{\"embedding\":[1,2,3]}]}'\n",
	);
	try {
		const result = spawnSync(
			supervisorPath,
			[
				"--name",
				"embedder-test",
				"--health",
				"embedder",
				"--url",
				"http://127.0.0.1:18092",
				"--model",
				"granite",
				"--dimensions",
				"3",
				"--",
				"bash",
				"-c",
				"sleep 0.1",
			],
			{
				env: {
					...process.env,
					PATH: fakeBin.path,
					ERGON_LLAMA_HEALTH_INTERVAL: "1",
					ERGON_LLAMA_START_PERIOD: "1",
				},
				encoding: "utf8",
				timeout: 5000,
			},
		);

		assert.equal(result.status, 0, result.stderr);
		assert.match(result.stdout, /healthcheck passed/);
	} finally {
		fakeBin.cleanup();
	}
});

test("supervisor exits when startup health never passes", () => {
	const fakeBin = tempBin("#!/usr/bin/env bash\nexit 22\n");
	try {
		const result = spawnSync(
			supervisorPath,
			[
				"--name",
				"embedder-test",
				"--health",
				"embedder",
				"--url",
				"http://127.0.0.1:18092",
				"--model",
				"granite",
				"--dimensions",
				"3",
				"--",
				"bash",
				"-c",
				"sleep 30",
			],
			{
				env: {
					...process.env,
					PATH: fakeBin.path,
					ERGON_LLAMA_START_PERIOD: "0",
				},
				encoding: "utf8",
				timeout: 5000,
			},
		);

		assert.notEqual(result.status, 0);
		assert.match(result.stderr, /healthcheck did not pass/);
	} finally {
		fakeBin.cleanup();
	}
});

test("supervisor force-kills a child that ignores restart termination", () => {
	const fakeBin = tempBin("#!/usr/bin/env bash\nexit 22\n");
	try {
		const result = spawnSync(
			supervisorPath,
			[
				"--name",
				"embedder-test",
				"--health",
				"embedder",
				"--url",
				"http://127.0.0.1:18092",
				"--model",
				"granite",
				"--dimensions",
				"3",
				"--",
				"bash",
				"-c",
				"trap '' TERM; while :; do sleep 1; done",
			],
			{
				env: {
					...process.env,
					PATH: fakeBin.path,
					ERGON_LLAMA_START_PERIOD: "0",
					ERGON_LLAMA_STOP_TIMEOUT: "0",
				},
				encoding: "utf8",
				timeout: 5000,
			},
		);

		assert.notEqual(result.status, 0);
		assert.match(result.stderr, /child did not stop/);
	} finally {
		fakeBin.cleanup();
	}
});
