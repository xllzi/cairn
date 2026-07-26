import "dotenv/config"
import OpenAI from "openai"
import { zodResponsesFunction } from "openai/helpers/zod"
import { z } from "zod"
import readline from "readline/promises"
import { readFileSync } from "node:fs"
import {
    GraphNodeSchema,
    GraphEdgeSchema,
    KnowledgeGraphSchema,
    type KnowledgeGraph,
    renderKnowledgeGraph,
    beginRound,
    getCommittedGraph,
    getStaged,
    addConceptToStage,
    addRelationToStage,
    commitRound,
    discardRound,
} from "./graph.ts"

/**
 * Cairn · Phase 1-2 · Explore：单次抽取 + 追问驱动的增量建图
 * 一个「感知-思考-行动」的 Agent Loop 骨架（Responses API 版）。
 *
 * Round 0（初始抽取）：不搜网、不抓 URL —— 纯靠 LLM 自身世界模型，对输入的
 *   一段文本做「单次」概念/关系抽取，调用自定义函数 constructKnowledgeGraph
 *   把图谱交回来。
 * 增量轮（追问驱动）：CLI 抽取完不退出，持续接受用户追问；每条追问触发新一轮
 *   循环，模型改用 addConcept/addRelation 两个细粒度工具补充图谱，可在一轮内
 *   多次调用、靠 observation 反馈自我修正（重复 id / 悬空引用都会被拒绝）。
 * 每一轮（round 0 与追问轮统一）结束后，先把产出暂存，摊开给用户看一遍，
 *   人工确认是否并入正式图谱——这是 [[Cairn Agent Loop]] 说的「方向盘在人手里」。
 *
 * 循环协议（对应 [[Agent]] 笔记的 Thought-Action-Observation）：
 *   感知 Perception  = input 数组里最新的内容（用户给的文本，或上一步的 observation）
 *   思考 Thought     = client.responses.create(...) —— 模型脑内抽取概念与关系
 *   行动 Action      = 执行模型选中的 function_call（emit 图谱 / 暂存增量）
 *   观察 Observation = 把执行结果作为 function_call_output 回灌进 input
 */

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
// 工具集：round 0 只有一个批量 emit 工具；增量轮换成两个细粒度工具。
// zodResponsesFunction 直接从 graph.ts 的 schema 派生 JSON Schema（strict 模式、
// additionalProperties: false 都自动处理），不必手写一份对得上的 JSON Schema。
// ─────────────────────────────────────────────────────────────────────────────
const round0Tools: OpenAI.Responses.Tool[] = [
    zodResponsesFunction({
        name: "constructKnowledgeGraph",
        description: "在你收集到足够信息、完成概念与关系抽取后调用它",
        parameters: KnowledgeGraphSchema,
    }),
]

const incrementalTools: OpenAI.Responses.Tool[] = [
    zodResponsesFunction({
        name: "addConcept",
        description: "在追问轮次中，向图谱补充一个此前未覆盖的核心概念",
        parameters: GraphNodeSchema,
    }),
    zodResponsesFunction({
        name: "addRelation",
        description: "在追问轮次中，补充两个已存在概念（已提交或本轮已通过 addConcept 补充）之间的关系",
        parameters: GraphEdgeSchema,
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
// Round 0 提示词设计。几个刻意的取舍：
//   · 引用 LEARNER_PROFILE：让「抽什么」随学习者背景而变，而非对资料穷举。
//   · 少而精：明确要求聚焦最核心的少量概念（软上限 5-7 个），先立骨干。
//     ——直接回应「概念太多」：穷举会淹没初学者，主干清晰才谈得上心智模型。
//   · 留话头：说明后续追问时才增量补充。这为「增量建图」埋点，现已在增量轮实现。
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
// 增量轮提示词：函数而非静态常量，因为要把「当前已有图谱」内嵌进去，让模型
// 知道有哪些概念已经存在（别重复加）、可以引用哪些 id。
// ─────────────────────────────────────────────────────────────────────────────
function buildIncrementalInstructions(currentGraph: KnowledgeGraph): string {
    return `
你是学习者的学习助手与陪伴者。学习者正在追问，请围绕追问内容为已有知识图谱补充相关的概念和关系——不要重新抽取整张图，只增量补充追问相关的部分。

学习者画像：
${LEARNER_PROFILE}

当前已有的图谱（不要重复添加已存在的概念；引用已有概念时使用下面列出的 id）：
${renderKnowledgeGraph(currentGraph)}

工具：
- addConcept：补充一个新概念（id 已存在会报错，换一个 id 或跳过）。
- addRelation：补充一条新关系，from/to 必须引用已存在或本轮已通过 addConcept 添加的概念 id（引用不存在的 id 会报错）。

追加原则：
- 少而精，只补充与追问直接相关的少量概念/关系，不求全。
- 可多次调用 addConcept / addRelation；工具报错后请在下一步修正重试。

收尾：完成补充后不要再调用工具，也不要用自然语言复述你添加的内容——停止即表示这一轮结束。
`.trim()
}

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
        case "addConcept":
            return addConceptToStage(args)
        case "addRelation":
            return addRelationToStage(args)
        default:
            // 模型调了个我们没注册的函数：把错误如实回灌，让它下一轮纠正。
            return `Error: unknown function "${call.name}"`
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Round 0 的工具执行函数：校验通过后把整张图批量塞进暂存区（不是逐个调用
// addConceptToStage），是否落地交给统一的 runRound 确认流程决定。
// ─────────────────────────────────────────────────────────────────────────────
function constructKnowledgeGraph(args: unknown): string {
    const result = KnowledgeGraphSchema.safeParse(args)
    if (!result.success) {
        return `KnowledgeGraph verification fail: ${z.prettifyError(result.error)}`
    }
    const { nodes, edges } = getStaged()
    nodes.push(...result.data.nodes)
    edges.push(...result.data.edges)
    return `construct ${result.data.nodes.length} nodes and ${result.data.edges.length} edges`
}

// ─────────────────────────────────────────────────────────────────────────────
// Agent Loop：感知 → 思考 → 行动 → 观察 → 回到感知。round 0 和增量轮共用同一套
// 循环机制，只有 tools/instructions 不同。
// ─────────────────────────────────────────────────────────────────────────────
async function runAgentLoop(
    input: OpenAI.Responses.ResponseInput,
    tools: OpenAI.Responses.Tool[],
    instructions: string,
): Promise<void> {
    for (let step = 1; step <= MAX_STEPS; step++) {
        console.log(`\n──────── step ${step} ────────`)

        // 思考 Thought：模型基于当前 input 规划下一步、选工具。
        const response = await client.responses.create({
            model: MODEL,
            input,
            tools,
            instructions,
        })

        // 可见化：打印模型这一轮的自然语言部分（它的「思考快照」）。
        if (response.output_text) console.log("💭", response.output_text)

        // 从输出里挑出「行动」：我们要手动执行的 function_call 项。
        const functionCalls = response.output.filter(
            (item): item is OpenAI.Responses.ResponseFunctionToolCall =>
                item.type === "function_call",
        )

        // 终止条件：模型不再行动 = 它认为这一轮任务结束（或只想说话）。
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
// 一轮的完整生命周期：清空暂存 → 跑 agent loop → 若有产出，摊开「已提交 + 暂存」
// 的合并图谱（新增部分高亮）→ 人工确认 → 并入或丢弃。round 0 与每条追问都走这里。
// ─────────────────────────────────────────────────────────────────────────────
async function runRound(
    query: string,
    tools: OpenAI.Responses.Tool[],
    instructions: string,
    rl: readline.Interface,
): Promise<void> {
    beginRound()

    const input: OpenAI.Responses.ResponseInput = [
        { role: "user", content: query },
    ]
    await runAgentLoop(input, tools, instructions)

    const staged = getStaged()
    if (staged.nodes.length === 0 && staged.edges.length === 0) {
        console.log("（这一轮没有产出新内容）")
        return
    }

    // 确认视图：已提交图谱 + 本轮暂存内容合并渲染，新增部分标绿——
    // 这样暂存的边即使指向已提交节点，也不会被悬空边检测误判。
    const committed = getCommittedGraph()
    const merged: KnowledgeGraph = {
        nodes: [...committed.nodes, ...staged.nodes],
        edges: [...committed.edges, ...staged.edges],
    }
    console.log(renderKnowledgeGraph(merged, {
        newNodeIds: new Set(staged.nodes.map((n) => n.id)),
        newEdgeIndices: new Set(staged.edges.map((_, i) => committed.edges.length + i)),
    }))

    const answer = (await rl.question(
        `并入 ${staged.nodes.length} 个概念 / ${staged.edges.length} 条关系？(y/n) `,
    )).trim().toLowerCase()

    if (answer === "y" || answer === "yes") {
        commitRound()
        console.log("✓ 已并入")
    } else {
        discardRound()
        console.log("✗ 已丢弃")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI 入口：round 0（文件参数或单行交互输入）之后不退出，进入追问 REPL——
// 每条追问触发一轮增量建图，直到用户输入 :q 或 exit。
//   · 有文件参数：`npm run explore -- path/to/file` —— 读整个文件。这是主路径，
//     适合喂大段学习资料。绕开 readline 逐行读取的坑（大段多行文本会被 rl.question
//     在第一个换行处截断，且 stdin 重定向下遇 EOF 直接返回空——正是你踩到的两个坑）。
//   · 无参数：回退到交互式单行输入，方便快速试一句话。
// ─────────────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
    // process.argv: [node, 脚本路径, ...真正的参数]。取第一个位置参数当文件路径。
    const filePath = process.argv[2]
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    })

    let firstQuery: string
    if (filePath) {
        const text = readFileSync(filePath, "utf-8").trim()
        if (!text) {
            console.error(`文件为空：${filePath}`)
            rl.close()
            return
        }
        console.log(`读入资料：${filePath}（${text.length} 字）`)
        firstQuery = text
    } else {
        firstQuery = (await rl.question("> ")).trim()
        if (!firstQuery) {
            rl.close()
            return
        }
    }

    await runRound(firstQuery, round0Tools, INSTRUCTIONS, rl)

    // 追问 REPL：:q / exit 退出，空行只是重新提示（不退出，避免误触）。
    while (true) {
        const followUp = (await rl.question("\n追问（:q 或 exit 退出）> ")).trim()
        if (/^(:q|exit)$/i.test(followUp)) break
        if (!followUp) continue
        await runRound(followUp, incrementalTools, buildIncrementalInstructions(getCommittedGraph()), rl)
    }

    rl.close()
}

// 仅在被直接运行时启动 CLI；被 import（如测试渲染）时不自动跑 main。
if (import.meta.url === `file://${process.argv[1]}`) {
    main()
}
