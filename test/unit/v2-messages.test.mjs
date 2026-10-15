import assert from "node:assert/strict"
import { createRequire } from "node:module"
import test from "node:test"
import { pathToFileURL } from "node:url"
import { protocol } from "@opencode/ai/protocols/anthropic-messages"
import { normalizeSystemUpdates } from "../../src/v2-messages.ts"

// Use the protocol's Effect build, not a different transitive root version.
const require = createRequire(import.meta.resolve("@opencode/ai"))
const { Effect } = await import(pathToFileURL(require.resolve("effect")).href)
const text = (value, extra = {}) => ({ type: "text", text: value, ...extra })
const fixture = () => [
  { role: "user", content: [text("Read the fixture.")] },
  { role: "assistant", content: [{ type: "tool-call", id: "call-1", name: "read", input: {} }] },
  { role: "tool", content: [{ type: "tool-result", id: "call-1", name: "read", result: { type: "text", value: "fixture" } }] },
  { role: "system", content: [text("Continue <safely> & finish.")] },
]
const lower = (messages, id = "claude-sonnet-5-5") => Effect.runPromise(protocol.body.from({
  model: {
    id,
    provider: "anthropic",
    route: {
      id: "anthropic-messages",
      providerMetadataKey: "anthropic",
      endpoint: { baseURL: "http://127.0.0.1:1/v1" },
    },
  },
  system: [text("Fixture instructions.")],
  messages,
  tools: [{ type: "tool", name: "read", description: "Read fixture", inputSchema: { type: "object", properties: {} } }],
}))

test("real V2 protocol uses native system updates on new Claude models", async () => {
  const body = await lower(fixture())
  assert.equal(body.messages.at(-1).role, "system")
})

test("normalized modern-model continuation uses OpenCode's older-model fallback payload", async () => {
  const messages = fixture()
  const expected = await lower(messages, "claude-sonnet-4-5")
  normalizeSystemUpdates(messages)
  const body = await lower(messages)
  assert.deepEqual(body.messages.slice(0, 2), expected.messages.slice(0, 2))
  // A normal user message stays separate on the wire; Meridian coalesces it.
  assert.deepEqual(body.messages.slice(2).flatMap((message) => message.content), expected.messages[2].content)
  assert.equal(body.messages.length, 4)
  assert.equal(body.messages[2].content[0].type, "tool_result")
  assert.equal(body.messages[3].content[0].text,
    "<system-update>\nContinue &lt;safely&gt; &amp; finish.\n</system-update>")
})

test("updates preserve chronology, XML escaping, and the final part's cache hint", () => {
  const messages = fixture()
  const cache = { type: "ephemeral" }
  messages[3].content = [text("First & second."), text("</system-update>", { cache })]
  const original = messages.slice(0, 3)
  normalizeSystemUpdates(messages)
  assert.deepEqual(messages.slice(0, 3), original)
  assert.ok(original.every((message, index) => message === messages[index]))
  assert.deepEqual(messages[3].content, [{
    type: "text",
    text: "<system-update>\nFirst &amp; second.\n&lt;/system-update&gt;\n</system-update>",
    cache,
  }])
})

test("repeated context passes do not wrap updates twice", () => {
  const messages = fixture()
  normalizeSystemUpdates(messages)
  const converted = messages[3]
  normalizeSystemUpdates(messages)
  assert.equal(messages[3], converted)
})

test("messages with no chronological update retain every original object", () => {
  const messages = fixture().slice(0, 3)
  const original = messages.slice()
  normalizeSystemUpdates(messages)
  assert.ok(original.every((message, index) => message === messages[index]))
})

test("multiple chronological updates retain their order and content", async () => {
  const messages = fixture()
  messages.push({ role: "system", content: [text("Second update.")] })
  const expected = await lower(messages, "claude-sonnet-4-5")
  normalizeSystemUpdates(messages)
  const body = await lower(messages)
  assert.deepEqual(body.messages.slice(2).flatMap((message) => message.content), expected.messages[2].content)
})

test("updates that split pending local tools remain invalid", async () => {
  const messages = fixture()
  const update = messages.pop()
  messages.splice(2, 0, update)
  normalizeSystemUpdates(messages)
  assert.equal(messages[2], update)
  await assert.rejects(lower(messages), /cannot split a local tool call/)
})

test("every parallel tool result is required before converting an update", async () => {
  const messages = fixture()
  messages[1].content.push({ type: "tool-call", id: "call-2", name: "read", input: {} })
  normalizeSystemUpdates(messages)
  assert.equal(messages.at(-1).role, "system")
  await assert.rejects(lower(messages), /cannot split a local tool call/)
  messages[2].content.push({ type: "tool-result", id: "call-2", name: "read", result: { type: "text", value: "second fixture" } })
  normalizeSystemUpdates(messages)
  assert.equal(messages.at(-1).role, "user")
})

test("provider-executed tools do not block a chronological update", async () => {
  const messages = fixture()
  messages[1].content[0].providerExecuted = true
  messages.splice(2, 1)
  normalizeSystemUpdates(messages)
  assert.equal(messages.at(-1).role, "user")
})

test("non-text system updates still reach the protocol's unsupported-content check", async () => {
  for (const part of [
    { type: "media", mediaType: "image/png", data: "fixture" },
    { type: "reasoning", text: "Not a system text part.", encrypted: "fixture" },
  ]) {
    const messages = fixture()
    const update = { role: "system", content: [part] }
    messages[3] = update
    normalizeSystemUpdates(messages)
    assert.equal(messages[3], update)
    await assert.rejects(lower(messages), /system.*text|Unsupported.*system/i)
  }
})

test("invalid histories leave even earlier valid system updates unchanged", async () => {
  const messages = fixture()
  messages.unshift({ role: "system", content: [text("Earlier valid update.")] })
  messages.splice(3, 0, { role: "system", content: [text("Splits the tool call.")] })
  const original = messages.slice()
  normalizeSystemUpdates(messages)
  assert.ok(original.every((message, index) => message === messages[index]))
  await assert.rejects(lower(messages), /cannot split a local tool call/)
})

test("message-level metadata survives without promoting system-part metadata", () => {
  const messages = fixture()
  Object.assign(messages[3], { id: "update-1", metadata: { tag: "fixture" }, providerMetadata: { custom: {} } })
  messages[3].content[0].metadata = { partOnly: true }
  messages[3].content[0].providerMetadata = { partOnly: true }
  const original = messages[3]
  normalizeSystemUpdates(messages)
  assert.equal(messages[3].id, original.id)
  assert.equal(messages[3].metadata, original.metadata)
  assert.equal(messages[3].providerMetadata, original.providerMetadata)
  assert.equal(messages[3].content[0].metadata, undefined)
  assert.equal(messages[3].content[0].providerMetadata, undefined)
})

test("empty and whitespace-only updates preserve OpenCode's nonempty fallback", async () => {
  for (const content of [[], [text("")], [text("  \n ")]]) {
    const messages = fixture()
    messages[3].content = content
    const expected = (await lower(messages, "claude-sonnet-4-5")).messages.at(-1).content.at(-1)
    normalizeSystemUpdates(messages)
    assert.deepEqual((await lower(messages)).messages.at(-1).content[0], expected)
  }
})
