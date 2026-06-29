import assert from "node:assert/strict";
import {
	chmodSync,
	mkdtempSync,
	mkdirSync,
	readFileSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { execFileSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const packageDir = resolve(fileURLToPath(new URL("..", import.meta.url)));
const scriptPath = join(packageDir, "scripts", "ergon-memory-steward");

function makeExecutable(path, body) {
	writeFileSync(path, body);
	chmodSync(path, 0o755);
}

function tempSetup() {
	const dir = mkdtempSync(join(tmpdir(), "ergon-memory-setup-"));
	const fakeBin = join(dir, "bin");
	const home = join(dir, "home");
	const config = join(dir, "xdg-config");
	const data = join(dir, "xdg-data");
	const models = join(dir, "models");
	mkdirSync(fakeBin, { recursive: true });
	mkdirSync(home, { recursive: true });
	mkdirSync(models, { recursive: true });

	makeExecutable(
		join(fakeBin, "llama-server"),
		"#!/usr/bin/env bash\nexit 0\n",
	);
	makeExecutable(
		join(fakeBin, "systemctl"),
		`#!/usr/bin/env bash\necho "$*" >> "${join(dir, "systemctl.log")}"\nexit 0\n`,
	);
	makeExecutable(join(fakeBin, "loginctl"), "#!/usr/bin/env bash\nexit 0\n");

	const stewardModel = join(models, "steward.gguf");
	const embedderModel = join(models, "embedder.gguf");
	writeFileSync(stewardModel, "steward");
	writeFileSync(embedderModel, "embedder");

	return {
		dir,
		home,
		config,
		data,
		fakeBin,
		stewardModel,
		embedderModel,
		env: {
			...process.env,
			HOME: home,
			XDG_CONFIG_HOME: config,
			XDG_DATA_HOME: data,
			PATH: `${fakeBin}:${process.env.PATH}`,
			ERGON_LLAMA_SERVER_BIN: join(fakeBin, "llama-server"),
			ERGON_STEWARD_MODEL_PATH: stewardModel,
			ERGON_EMBEDDER_MODEL_PATH: embedderModel,
		},
		cleanup() {
			rmSync(dir, { recursive: true, force: true });
		},
	};
}

test("setup writes systemd units against the resolved env file", () => {
	const fixture = tempSetup();
	try {
		execFileSync(scriptPath, ["setup"], { env: fixture.env });

		const envFile = join(fixture.config, "ergon-memory-steward.env");
		const stewardUnit = readFileSync(
			join(fixture.config, "systemd/user/ergon-steward.service"),
			"utf8",
		);
		const embedderUnit = readFileSync(
			join(fixture.config, "systemd/user/ergon-embedder.service"),
			"utf8",
		);

		assert.ok(
			readFileSync(envFile, "utf8").includes("ERGON_MEMORY_STEWARD_DIR="),
		);
		assert.ok(stewardUnit.includes(`EnvironmentFile=-${envFile}`));
		assert.ok(embedderUnit.includes(`EnvironmentFile=-${envFile}`));
		assert.ok(!stewardUnit.includes("%h/.config/ergon-memory-steward.env"));
	} finally {
		fixture.cleanup();
	}
});

test("setup reruns from the generated env without original model source files", () => {
	const fixture = tempSetup();
	try {
		execFileSync(scriptPath, ["setup"], { env: fixture.env });
		rmSync(fixture.stewardModel);
		rmSync(fixture.embedderModel);

		const rerunEnv = {
			...process.env,
			HOME: fixture.home,
			XDG_CONFIG_HOME: fixture.config,
			XDG_DATA_HOME: fixture.data,
			PATH: `${fixture.fakeBin}:${process.env.PATH}`,
		};
		execFileSync(scriptPath, ["setup"], { env: rerunEnv });

		const envText = readFileSync(
			join(fixture.config, "ergon-memory-steward.env"),
			"utf8",
		);
			assert.match(envText, /ERGON_STEWARD_MODEL_PATH=.*steward\.gguf/);
			assert.match(
				envText,
				/ERGON_EMBEDDER_MODEL_PATH=.*granite-embedding-311m-multilingual-r2-Q4_K_M\.gguf/,
			);
			assert.match(envText, /ERGON_EMBEDDER_MODEL="granite-embedding-311m"/);
		} finally {
			fixture.cleanup();
		}
	});
