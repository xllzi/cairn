import "dotenv/config"
import OpenAI from "openai"
import { zodResponsesFunction } from "openai/helpers/zod"
import { z } from "zod"
import readline from "readline/promises"
import { readFileSync } from "node:fs"

/**
 * Cairn · Phase 1 · Explore MVP
 * 一个「感知-思考-行动」的 Agent Loop 骨架（Responses API 版）。
 *
 * 本阶段范围（已定）：不搜网、不抓 URL —— 纯靠 LLM 自身世界模型，对输入的
 *   一段文本做「单次」概念/关系抽取，调用自定义函数 constructKnowledgeGraph
 *   把图谱交回来。因此循环通常只转一圈（思考一次 → emit 一次 → 收尾）。
 *   增量建图（addConcept/addRelation 让循环真正多轮迭代）留到后续阶段，
 *   它正是从这个 emit 长出来的地方。
 *
 * 循环协议（对应 [[Agent]] 笔记的 Thought-Action-Observation）：
 *   感知 Perception  = input 数组里最新的内容（用户给的文本，或上一步的 observation）
 *   思考 Thought     = client.responses.create(...) —— 模型脑内抽取概念与关系
 *   行动 Action      = 执行模型选中的 function_call（这里就是 emit 图谱）
 *   观察 Observation = 把执行结果作为 function_call_output 回灌进 input
 */

// ─────────────────────────────────────────────────────────────────────────────
// 领域类型：知识图谱的形状。用 zod 定义一次，同时得到运行时校验（.parse）和
// 静态类型（z.infer）——这是「领域边界」，图谱结构一旦定死，下游都受它保护。
// ─────────────────────────────────────────────────────────────────────────────
const GraphNodeSchema = z.object({
    id: z.string(),          // 稳定标识，英文 kebab-case，如 "spaced-repetition"
    label: z.string(),       // 展示名（可中文）
    type: z.string(),        // 概念类别
})

const GraphEdgeSchema = z.object({
    from: z.string(),        // 源节点 id
    to: z.string(),          // 目标节点 id
    relation: z.string(),    // 关系描述，如 "对抗" | "属于"
})

const KnowledgeGraphSchema = z.object({
    nodes: z.array(GraphNodeSchema),
    edges: z.array(GraphEdgeSchema),
})

type GraphNode = z.infer<typeof GraphNodeSchema>
type GraphEdge = z.infer<typeof GraphEdgeSchema>
type KnowledgeGraph = z.infer<typeof KnowledgeGraphSchema>

// ─────────────────────────────────────────────────────────────────────────────
// 客户端。沿用 hello.ts：openai SDK 指向百炼 OpenAI 兼容端点。
// ─────────────────────────────────────────────────────────────────────────────
const client = new OpenAI({
    apiKey: process.env.DASHSCOPE_API_KEY,
    baseURL: "https://ws-op1rmhcapcp4k4fg.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
})

const MODEL = "qwen3.7-plus"
const MAX_STEPS = 7   // 兜底：防止模型陷入死循环，永远不收尾

// ─────────────────────────────────────────────────────────────────────────────
// 工具集：本阶段只有一个 —— 自定义 emit 工具。抽取是模型的「思考」，不做成工具；
// 也不放 web_search（本阶段不向外收集）。
// zodResponsesFunction 直接从 KnowledgeGraphSchema 派生 JSON Schema（strict 模式、
// additionalProperties: false 都自动处理），不必再手写一份对得上的 JSON Schema。
// ─────────────────────────────────────────────────────────────────────────────
const tools: OpenAI.Responses.Tool[] = [
    zodResponsesFunction({
        name: "constructKnowledgeGraph",
        description: "在你收集到足够信息、完成概念与关系抽取后调用它",
        parameters: KnowledgeGraphSchema,
    }),
]

// ─────────────────────────────────────────────────────────────────────────────
// 学习者画像：本阶段先写死一个常量（就是你）。它让 system prompt 能「根据背景」
// 裁剪概念——同一份资料，对新手和专家该抽出的「主要概念」并不一样。
// 可配置化（不同学习者、运行时传入）留到 Phase 5 有 UI 再说；现在写死最省事、够用。
// ─────────────────────────────────────────────────────────────────────────────
const LEARNER_PROFILE = `
- 有 TypeScript / JavaScript 编程背景。
- AI 应用开发新手，正在从原语一步步学起。
- 目的是学习理解，偏好抓住主干、先建立整体心智模型，再逐步深入细节。
`.trim()

// ─────────────────────────────────────────────────────────────────────────────
// 提示词设计（本阶段要吃透的核心概念之一）。几个刻意的取舍：
//   · 引用 LEARNER_PROFILE：让「抽什么」随学习者背景而变，而非对资料穷举。
//   · 少而精：明确要求聚焦最核心的少量概念（软上限 5-7 个），先立骨干。
//     ——直接回应「概念太多」：穷举会淹没初学者，主干清晰才谈得上心智模型。
//   · 留话头：说明后续追问时才增量补充。这为「增量建图」埋点，但本阶段不实现。
//   · 收尾契约：必须以调用 constructKnowledgeGraph 结束，别用自然语言把图谱「说」出来。
// ─────────────────────────────────────────────────────────────────────────────
const INSTRUCTIONS = `
你是学习者的学习助手与陪伴者。你的任务：对输入的学习资料明确主题，抽取核心概念、建立它们之间的关系，构建一张知识图谱。

学习者画像：
${LEARNER_PROFILE}

抽取原则：
- 少而精。只抽取该主题最核心的少量主要概念（通常 5-7 个，宁少勿多），先立起主干骨架，而不是穷举资料里出现的每一个术语。
- 因人而异。根据上面的学习者画像裁剪：对一个初学者，什么才是理解这个主题绕不开的骨干概念？次要的、进阶的细节暂时略去。
- 关系清晰。概念之间用简短的关系词连接（如「属于」「对抗」「用于」），让骨架能读出结构。
- 增量留白。学习者若在后续追问中表达疑惑，届时再补充相关的细分概念——这一轮不必求全。

收尾：完成抽取后，必须调用 constructKnowledgeGraph 交回图谱，不要用自然语言把图谱内容复述出来。
`.trim()

// ─────────────────────────────────────────────────────────────────────────────
// 行动 Action：执行模型选中的 function_call，返回一段字符串作为 Observation。
// ─────────────────────────────────────────────────────────────────────────────
async function executeFunctionCall(
    call: OpenAI.Responses.ResponseFunctionToolCall,
): Promise<string> {
    // call.arguments 是模型生成的 JSON 字符串，需要自己 parse。
    const args = JSON.parse(call.arguments)

    switch (call.name) {
        case "constructKnowledgeGraph":
            return constructKnowledgeGraph(args)
        default:
            // 模型调了个我们没注册的函数：把错误如实回灌，让它下一轮纠正。
            return `Error: unknown function "${call.name}"`
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 这里是「一块 cairn」——[[Context Version Control]] 里的可审阅检查点。要做的事：
//   a. 用 KnowledgeGraphSchema 校验 args（strict tool schema 已经把住了大部分形状，
//      这里再校验一遍是防御性的最后一道关——模型偶尔仍会吐出不合规的 JSON）。
//   b. 人工检查点：把图谱摊开打印给用户看，暂停确认（读一行 stdin），再决定是否「提交」。
//   c. 返回一段字符串作为 Observation（比如 "已保存，12 个节点 / 18 条边" 或校验失败原因）。
// ─────────────────────────────────────────────────────────────────────────────
function constructKnowledgeGraph(args: unknown): string {
    const result = KnowledgeGraphSchema.safeParse(args)
    if (!result.success) {
        return `KnowledgeGraph verification fail: ${z.prettifyError(result.error)}`
    }
    const graph = result.data
    // 可审阅：把图谱摊成人能扫读的文本，而不是一坨 JSON。这一步是「可观测性」的落点。
    console.log(renderKnowledgeGraph(graph))
    return `construct ${graph.nodes.length} nodes and ${graph.edges.length} edges`
}

// ─────────────────────────────────────────────────────────────────────────────
// 渲染：把 KnowledgeGraph 排版成人能扫一眼看懂的文本。纯函数、无副作用——
// 只吃一个 graph、吐一个 string，好测、好复用，和「执行/IO」分开。
//
// 不引入 chalk 等库，直接用 ANSI 转义码上色（先原语，不给学习骨架加噪音）。
// 呈现三块：① 摘要 ② 概念按 type 分组 ③ 关系用 起点 ──关系──▶ 终点。
// 顺带做「悬空边」检测：边引用了不存在的 node id → 标红，这正是审阅时最该看见的。
// ─────────────────────────────────────────────────────────────────────────────
const C = {
    dim: (s: string) => `\x1b[2m${s}\x1b[0m`,        // 灰：次要信息（id）
    bold: (s: string) => `\x1b[1m${s}\x1b[0m`,       // 粗：标题
    cyan: (s: string) => `\x1b[36m${s}\x1b[0m`,      // 青：概念 label
    yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,    // 黄：关系
    red: (s: string) => `\x1b[31m${s}\x1b[0m`,       // 红：悬空边告警
}

export function renderKnowledgeGraph(graph: KnowledgeGraph): string {
    const { nodes, edges } = graph
    const lines: string[] = []

    // ① 摘要
    lines.push("")
    lines.push(C.bold(`◆ 知识图谱  ${nodes.length} 个概念 · ${edges.length} 条关系`))

    // 建 id → node 索引：既给关系渲染用 label，也用来查悬空边。
    const byId = new Map(nodes.map((n) => [n.id, n]))

    // ② 概念：按 type 分组
    lines.push("")
    lines.push(C.bold("概念"))
    const groups = new Map<string, GraphNode[]>()
    for (const n of nodes) {
        const g = groups.get(n.type) ?? []
        g.push(n)
        groups.set(n.type, g)
    }
    for (const [type, members] of groups) {
        lines.push(`  ${C.dim("┌")} ${type}`)
        for (const n of members) {
            lines.push(`  ${C.dim("│")}   ${C.cyan(n.label)}  ${C.dim(n.id)}`)
        }
    }

    // ③ 关系：起点 ──关系──▶ 终点。用 label 显示（找不到就退回 id 并标红）。
    lines.push("")
    lines.push(C.bold("关系"))
    const danglingSeen = new Set<string>()
    const nameOf = (id: string): string => {
        const node = byId.get(id)
        if (node) return C.cyan(node.label)
        danglingSeen.add(id)
        return C.red(`${id}?`)   // 悬空：这个 id 没有对应节点
    }
    for (const e of edges) {
        lines.push(`  ${nameOf(e.from)} ${C.dim("──")}${C.yellow(e.relation)}${C.dim("──▶")} ${nameOf(e.to)}`)
    }

    // 悬空边告警：审阅时最该被看见的完整性问题。
    if (danglingSeen.size > 0) {
        lines.push("")
        lines.push(C.red(`⚠ ${danglingSeen.size} 个悬空引用（边指向了不存在的概念）：${[...danglingSeen].join(", ")}`))
    }

    lines.push("")
    return lines.join("\n")
}

// ─────────────────────────────────────────────────────────────────────────────
// Agent Loop：感知 → 思考 → 行动 → 观察 → 回到感知。
// ─────────────────────────────────────────────────────────────────────────────
async function runExplore(query: string): Promise<void> {
    // 感知（起点）：用户的初始目标就是第一个 observation。
    const input: OpenAI.Responses.ResponseInput = [
        { role: "user", content: query },
    ]

    for (let step = 1; step <= MAX_STEPS; step++) {
        console.log(`\n──────── step ${step} ────────`)

        // 思考 Thought：模型基于当前 input 规划下一步、选工具。
        // 提示：单次 emit 想更可靠，可加 tool_choice 逼它必须调 constructKnowledgeGraph，
        // 而不是用自然语言把图谱「说」出来。写法（可选）：
        //   tool_choice: { type: "function", name: "constructKnowledgeGraph" }
        const response = await client.responses.create({
            model: MODEL,
            input,
            tools,
            instructions: INSTRUCTIONS,
        })

        // 可见化：打印模型这一轮的自然语言部分（它的「思考快照」）。
        if (response.output_text) console.log("💭", response.output_text)

        // 从输出里挑出「行动」：我们要手动执行的 function_call 项。
        // 本阶段只有一个可能出现：constructKnowledgeGraph。
        const functionCalls = response.output.filter(
            (item): item is OpenAI.Responses.ResponseFunctionToolCall =>
                item.type === "function_call",
        )

        // 终止条件之一：模型不再行动 = 它认为任务结束（或只想说话）。
        if (functionCalls.length === 0) break

        // 回灌①：先把模型的 function_call 决策塞回 input，
        // 否则下面的 function_call_output 找不到对应的 call_id，API 会报错。
        input.push(...functionCalls)

        // 行动 + 观察：逐个执行，把结果作为 function_call_output 回灌。
        for (const call of functionCalls) {
            const observation = await executeFunctionCall(call)   // 行动
            input.push({                                          // 观察
                type: "function_call_output",
                call_id: call.call_id,
                output: observation,
            })
        }

    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI 入口。
//   · 有文件参数：`npm run explore -- path/to/file` —— 读整个文件。这是主路径，
//     适合喂大段学习资料。绕开 readline 逐行读取的坑（大段多行文本会被 rl.question
//     在第一个换行处截断，且 stdin 重定向下遇 EOF 直接返回空——正是你踩到的两个坑）。
//   · 无参数：回退到交互式单行输入，方便快速试一句话。
// ─────────────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
    // process.argv: [node, 脚本路径, ...真正的参数]。取第一个位置参数当文件路径。
    const filePath = process.argv[2]

    if (filePath) {
        const text = readFileSync(filePath, "utf-8").trim()
        if (!text) {
            console.error(`文件为空：${filePath}`)
            return
        }
        console.log(`读入资料：${filePath}（${text.length} 字）`)
        await runExplore(text)
        return
    }

    // 无文件参数 → 交互式单行输入（仅适合一句话；大段文本请用文件参数）。
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    })
    const query = await rl.question("> ")
    rl.close()
    if (query.trim()) await runExplore(query.trim())
}

// 仅在被直接运行时启动 CLI；被 import（如测试渲染）时不自动跑 main。
if (import.meta.url === `file://${process.argv[1]}`) {
    main()
}
