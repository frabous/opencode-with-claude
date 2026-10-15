import assert from "node:assert/strict"
import { readdirSync, readFileSync } from "node:fs"
import { createRequire } from "node:module"
import { dirname, join } from "node:path"
import test from "node:test"
import { fileURLToPath, pathToFileURL } from "node:url"
import vm from "node:vm"
import { protocol } from "@opencode/ai/protocols/anthropic-messages"
import { normalizeSystemUpdates } from "../../src/v2-messages.ts"

// Only protocol.body.from runs: no HTTP transport, Meridian import, SDK, or credentials.
// Matcher acceptance is not proof of SDK replay reconstruction or authenticated E2E.
// Match the protocol's Effect build rather than the root's other transitive version.
const require = createRequire(import.meta.resolve("@opencode/ai"))
const { Effect } = await import(pathToFileURL(require.resolve("effect")).href)
const dist = dirname(fileURLToPath(import.meta.resolve("@rynfar/meridian")))
const declaration = "function coalesceCompleteToolResultContinuation("
const bundles = readdirSync(dist).filter((name) => name.endsWith(".js"))
  .map((name) => ({ path: join(dist, name), source: readFileSync(join(dist, name), "utf8") }))
  .filter(({ source }) => source.includes(`\n${declaration}`))
assert.equal(bundles.length, 1, "expected one installed Meridian bundle containing the checkpoint matcher")
const bundle = bundles[0]

function originalFunction(name) {
  const start = bundle.source.indexOf(`\nfunction ${name}(`)
  assert.notEqual(start, -1, `missing original top-level Meridian function ${name}`)
  // Bun emits these declarations and their closing braces at column zero.
  // Compile as one expression too, so a truncated body or extra declarations fail.
  const end = bundle.source.indexOf("\n}", start + 1)
  assert.notEqual(end, -1, `missing closing brace for ${name}`)
  const source = bundle.source.slice(start + 1, end + 2)
  new vm.Script(`(${source})`, { filename: `${bundle.path}:${name}` })
  return source
}

const helperSource = originalFunction("isMidConvoEffortSystemMessage")
const matcherSource = originalFunction("coalesceCompleteToolResultContinuation")
const context = vm.createContext({}, { codeGeneration: { strings: false, wasm: false } })
for (const source of [helperSource, matcherSource]) {
  vm.runInContext(source, context, { filename: bundle.path, timeout: 1000 })
}
const coalesce = (messages, ids = [toolID]) => context.coalesceCompleteToolResultContinuation(messages, ids)
// VM-created objects have different prototypes. Compare the actual JSON wire values.
const clean = (value) => JSON.parse(JSON.stringify(value))
const text = (value) => ({ type: "text", text: value })
const toolID = "toolu_offline_1"
const update = "Continue <safely> & preserve </system-update>."
const wrappedUpdate = "<system-update>\nContinue &lt;safely&gt; &amp; preserve &lt;/system-update&gt;.\n</system-update>"
const fixture = () => [
  { role: "user", content: [text("Read the offline fixture.")] },
  {
    role: "assistant",
    content: [
      { type: "reasoning", text: "Read before answering.", encrypted: "offline-signature" },
      { type: "tool-call", id: toolID, name: "read", input: {} },
    ],
  },
  { role: "tool", content: [{ type: "tool-result", id: toolID, name: "read", result: { type: "text", value: "fixture" } }] },
  { role: "system", content: [text(update)] },
]
const lower = async (messages, id = "claude-sonnet-5-5") => clean(await Effect.runPromise(protocol.body.from({
  model: {
    id,
    provider: "anthropic",
    route: {
      id: "anthropic-messages",
      providerMetadataKey: "anthropic",
      endpoint: { baseURL: "http://127.0.0.1:1/v1" },
    },
  },
  system: [text("Offline fixture instructions.")],
  messages,
  tools: [{ type: "tool", name: "read", description: "Read fixture", inputSchema: { type: "object", properties: {} } }],
})))

test("original Meridian matcher requires its original effort-system helper", async () => {
  const body = await lower(fixture().slice(0, 3))
  const isolated = vm.createContext({})
  vm.runInContext(matcherSource, isolated, { timeout: 1000 })
  assert.throws(() => isolated.coalesceCompleteToolResultContinuation(body.messages.slice(1), [toolID]),
    /isMidConvoEffortSystemMessage is not defined/)
  vm.runInContext(helperSource, isolated, { timeout: 1000 })
  assert.deepEqual(clean(isolated.coalesceCompleteToolResultContinuation(body.messages.slice(1), [toolID])),
    [body.messages[2]])
})

test("original effort-system helper only skips empty effort updates, not chronological text", async () => {
  const body = await lower(fixture().slice(0, 3))
  const emptyEffort = { role: "system", content: [text("  ")], output_config: { effort: "high" } }
  assert.equal(context.isMidConvoEffortSystemMessage(emptyEffort), true)
  assert.equal(context.isMidConvoEffortSystemMessage({ ...emptyEffort, output_config: undefined, outputConfig: {} }), true)
  assert.deepEqual(clean(coalesce([...body.messages.slice(1), emptyEffort])), [body.messages[2]])
  const substantive = { ...emptyEffort, content: [text(update)] }
  assert.equal(context.isMidConvoEffortSystemMessage(substantive), false)
  assert.equal(coalesce([...body.messages.slice(1), substantive]), undefined)
})

test("unmodified Sonnet 5.5 native system continuation fails the installed Meridian matcher", async () => {
  const body = await lower(fixture())
  assert.equal(body.messages.at(-1).role, "system")
  assert.deepEqual(body.messages.at(-1).content, [text(update)])
  assert.equal(coalesce(body.messages.slice(1)), undefined)
})

test("normalized Sonnet 5.5 continuation matches with escaped update and signed thinking intact", async () => {
  const messages = fixture()
  const originalAssistant = messages[1]
  normalizeSystemUpdates(messages)
  assert.equal(messages[1], originalAssistant)
  const body = await lower(messages)
  assert.deepEqual(body.messages[1].content, [
    { type: "thinking", thinking: "Read before answering.", signature: "offline-signature" },
    { type: "tool_use", id: toolID, name: "read", input: {} },
  ])
  assert.deepEqual(body.messages.slice(2).map((message) => message.role), ["user", "user"])
  const expected = [{ role: "user", content: [body.messages[2].content[0], text(wrappedUpdate)] }]
  assert.deepEqual(clean(coalesce(body.messages.slice(1))), expected)

  // Older-model fallback appends the update to the tool-result turn; normalization
  // emits a separate user turn. Compare actual matcher output, not wire layout.
  const older = await lower(fixture(), "claude-sonnet-4-5")
  assert.equal(older.messages.length, 3)
  assert.deepEqual(clean(coalesce(older.messages.slice(1))), expected)
})

test("matcher accepts multiple consecutive user updates after complete tool results in order", async () => {
  const messages = fixture()
  messages.push(
    { role: "system", content: [text("Second <update>.")] },
    { role: "user", content: [text("Finish with a short answer.")] },
  )
  normalizeSystemUpdates(messages)
  const body = await lower(messages)
  assert.deepEqual(body.messages.slice(2).map((message) => message.role), ["user", "user", "user", "user"])
  assert.deepEqual(clean(coalesce(body.messages.slice(1))), [{ role: "user", content: [
    body.messages[2].content[0],
    text(wrappedUpdate),
    text("<system-update>\nSecond &lt;update&gt;.\n</system-update>"),
    text("Finish with a short answer."),
  ] }])
})

test("matcher still rejects missing, duplicate, and foreign results after normalization", async (t) => {
  const messages = fixture()
  normalizeSystemUpdates(messages)
  const body = await lower(messages)
  assert.equal(body.messages.some((message) => message.role === "system"), false)
  // Corrupt the lowered wire, not a fixture reimplementation of the matcher.
  // This isolates its ID checks from the protocol's pending-tool validation.
  for (const kind of ["missing", "duplicate", "foreign"]) {
    await t.test(kind, () => {
      const continuation = structuredClone(body.messages.slice(1))
      const results = continuation[1].content
      if (kind === "missing") results.splice(0, 1)
      if (kind === "duplicate") results.push(structuredClone(results[0]))
      if (kind === "foreign") results[0].tool_use_id = "toolu_foreign"
      assert.equal(coalesce(continuation), undefined)
    })
  }
})

test("parallel tool results must all settle before the normalized update matches", async () => {
  const secondID = "toolu_offline_2"
  const messages = fixture()
  messages[1].content.push({ type: "tool-call", id: secondID, name: "read", input: {} })
  messages[2].content.push({ type: "tool-result", id: secondID, name: "read", result: { type: "text", value: "second fixture" } })
  normalizeSystemUpdates(messages)
  const body = await lower(messages)
  const ids = [toolID, secondID]
  assert.deepEqual(clean(coalesce(body.messages.slice(1), ids)), [{ role: "user", content: [
    ...body.messages[2].content,
    text(wrappedUpdate),
  ] }])
  const incomplete = structuredClone(body.messages.slice(1))
  incomplete[1].content.pop()
  assert.equal(coalesce(incomplete, ids), undefined)
})

test("structured tool-result media and later user media survive matching alongside the update", async () => {
  const messages = fixture()
  const image = "data:image/png;base64,aW1hZ2U="
  messages[2].content[0].result = { type: "content", value: [
    text("Image from the tool."),
    { type: "file", mime: "image/png", uri: image },
    { type: "file", mime: "application/pdf", uri: "data:application/pdf;base64,cGRm", name: "fixture.pdf" },
  ] }
  messages.push({ role: "user", content: [
    text("Compare this attachment."),
    { type: "media", mediaType: "image/png", data: image },
  ] })
  normalizeSystemUpdates(messages)
  const body = await lower(messages)
  const toolResult = body.messages[2].content[0]
  assert.deepEqual(toolResult.content, [
    text("Image from the tool."),
    { type: "image", source: { type: "base64", media_type: "image/png", data: "aW1hZ2U=" } },
    { type: "document", source: { type: "base64", media_type: "application/pdf", data: "cGRm" }, title: "fixture.pdf" },
  ])
  assert.deepEqual(body.messages.at(-1).content[1], toolResult.content[1])
  assert.deepEqual(clean(coalesce(body.messages.slice(1))), [{ role: "user", content: [
    toolResult,
    text(wrappedUpdate),
    ...body.messages.at(-1).content,
  ] }])
})
