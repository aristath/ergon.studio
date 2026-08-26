import assert from "node:assert/strict";
import test from "node:test";

import memoryStewardExtension, {
	selectCompletedTurn,
} from "../dist/extensions/index.js";

test("registers the save path at turn_end", () => {
	const hooks = new Map();
	memoryStewardExtension({
		on(name, handler) {
			hooks.set(name, handler);
		},
	});

	assert.equal(hooks.has("turn_end"), true);
	assert.equal(hooks.has("agent_end"), false);
});

test("rejects intermediate tool turns and accepts the completed response", () => {
	const toolTurn = selectCompletedTurn({
		message: {
			role: "assistant",
			content: [
				{ type: "text", text: "I will inspect the failing tests." },
				{ type: "toolCall", id: "call-1", name: "bash", arguments: {} },
			],
			responseId: "intermediate-response",
			stopReason: "toolUse",
		},
		toolResults: [
			{
				role: "toolResult",
				toolCallId: "call-1",
				toolName: "bash",
				content: [{ type: "text", text: "Tests failed." }],
				isError: false,
				timestamp: 2,
			},
		],
	});
	const completedTurn = selectCompletedTurn({
		message: {
			role: "assistant",
			content: [{ type: "text", text: "The build is fixed." }],
			responseId: "final-response",
			stopReason: "stop",
		},
		toolResults: [],
	});

	assert.equal(toolTurn, null);
	assert.deepEqual(completedTurn, {
		assistantText: "The build is fixed.",
		assistantId: "final-response",
	});
});

test("accepts each completed response in a queued follow-up run", () => {
	const turns = ["First answer", "Follow-up answer"].map((text, index) =>
		selectCompletedTurn({
			message: {
				role: "assistant",
				content: [{ type: "text", text }],
				responseId: `response-${index}`,
				stopReason: "stop",
			},
			toolResults: [],
		}),
	);

	assert.deepEqual(
		turns.map((turn) => turn?.assistantText),
		["First answer", "Follow-up answer"],
	);
});

test("rejects incomplete final responses", () => {
	for (const stopReason of ["aborted", "error", "length"]) {
		const turn = selectCompletedTurn({
			message: {
				role: "assistant",
				content: [{ type: "text", text: "Partial answer" }],
				stopReason,
			},
			toolResults: [],
		});

		assert.equal(turn, null);
	}
});
