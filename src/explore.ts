import "dotenv/config"
import OpenAI from "openai"
import readline from "readline/promises"

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
// 领域类型：知识图谱的形状。
// 这是「领域边界」——在这里收紧类型最值钱：图谱结构一旦定死，下游都受它保护。
// ─────────────────────────────────────────────────────────────────────────────
interface GraphNode {
    id: string          // 稳定标识，英文 kebab-case，如 "spaced-repetition"
    label: string       // 展示名（可中文）
    type: string        // 概念类别
}

interface GraphEdge {
    from: string        // 源节点 id
    to: string          // 目标节点 id
    relation: string    // 关系描述，如 "对抗" | "属于"
}

interface KnowledgeGraph {
    nodes: GraphNode[]
    edges: GraphEdge[]
}

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
// constructKnowledgeGraph 的参数 schema 就是「最终输出的形状」。
// 用 JSON Schema 描述上面的 KnowledgeGraph：一个 object，含 nodes 数组和 edges 数组，
// 每个 node 有 id/label/type，每个 edge 有 from/to/relation，全部 required，
// additionalProperties: false。写完后它必须和 interface KnowledgeGraph 对得上。
// 参考形状见文件顶部注释。
// ─────────────────────────────────────────────────────────────────────────────
const KNOWLEDGE_GRAPH_PARAMETERS: Record<string, unknown> = {
    type: "object",
    properties: {
        nodes: { 
            type: "array", 
            items: { 
                type: "object",
                properties: {
                    id: { type: "string" },
                    label: { type: "string" },
                    type: { type: "string" },
                },
                required: ["id", "label", "type"]
            }, 
        },
        edges: {
            type: "array",
            items: {
                type: "object",
                properties: {
                    from: { type: "string" },
                    to: { type: "string" },
                    relation: { type: "string" },
                },
                required: ["from", "to", "relation"]
            },
        }
    },
    required: ["nodes", "edges"],
}

// ─────────────────────────────────────────────────────────────────────────────
// 工具集：本阶段只有一个 —— 自定义 emit 工具。抽取是模型的「思考」，不做成工具；
// 也不放 web_search（本阶段不向外收集）。
// ─────────────────────────────────────────────────────────────────────────────
const tools: OpenAI.Responses.Tool[] = [
    // 自定义函数工具 = 终止动作 / emit。模型调它 = 宣告「我抽完了」。
    {
        type: "function",
        name: "constructKnowledgeGraph",
        description:
            "在你收集到足够信息、完成概念与关系抽取后调用它",
        parameters: KNOWLEDGE_GRAPH_PARAMETERS,
        strict: true,
    },
]

// ─────────────────────────────────────────────────────────────────────────────
// 想清楚要告诉模型：它是谁、任务是什么、什么时候该用 web_search、
// 以及「必须以调用 constructKnowledgeGraph 来结束」。这决定循环能不能干净收尾。
// ─────────────────────────────────────────────────────────────────────────────
const INSTRUCTIONS = `
你是用户的学习助手、陪伴者。你的任务是对输入的学习资料明确主题，抽取关键概念，建立联系，构建知识图谱。
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
//   a. 把 args 校验/收窄成 KnowledgeGraph（可信任 strict schema，但仍建议基本校验）。
//   在思考步骤，Agent抽取概念，建立联系，应当输出一个JSON作为参数，能解析为KnowledgeGraph
//   b. 人工检查点：把图谱摊开打印给用户看，暂停确认（读一行 stdin），再决定是否「提交」。
//   c. 返回一段字符串作为 Observation（比如 "已保存，12 个节点 / 18 条边" 或 "用户拒绝"）。
// ─────────────────────────────────────────────────────────────────────────────
function constructKnowledgeGraph(args: unknown): string {
    console.log("constructKnowledgeGraph receive arguments：", JSON.stringify(args, null, 2))
    try {
        const graph = parseKnowledgeGraph(args);
        console.log(`KnowledgeGraph:\n ${JSON.stringify(graph, null, 2)}`);
        return `construct ${graph.nodes.length} nodes and ${graph.edges.length} edges`;
    } catch (e) {
        return `KnowledgeGraph verificaton fail: ${(e as Error).message}`
    }
}

function isGraphNode(x: unknown): x is GraphNode {
    return typeof x === "object" && x !== null &&
        typeof (x as Record<string, unknown>).id === "string" &&
        typeof (x as Record<string, unknown>).label === "string" &&
        typeof (x as Record<string, unknown>).type === "string"
}

function isGraphEdge(x: unknown): x is GraphEdge {
    return typeof x === "object" && x !== null &&
        typeof (x as Record<string, unknown>).from === "string" &&
        typeof (x as Record<string, unknown>).to === "string" &&
        typeof (x as Record<string, unknown>).relation === "string"
}

function parseKnowledgeGraph(x: unknown): KnowledgeGraph {
    if (typeof x !== "object" || x === null) throw new Error("root is not object");
    const obj = x as Record<string, unknown>;
    if (!Array.isArray(obj.nodes)) throw new Error("lack nodes array");
    if (!Array.isArray(obj.edges)) throw new Error("lack edges array");
    if (!obj.nodes.every(isGraphNode)) throw new Error("wrong node shape");
    if (!obj.edges.every(isGraphEdge)) throw new Error("wrong edge shape");
    return { nodes: obj.nodes, edges: obj.edges };
}

// ─────────────────────────────────────────────────────────────────────────────
// Agent Loop：感知 → 思考 → 行动 → 观察 → 回到感知。
// input 数组就是「可见的记忆」——每一轮的 Thought / Action / Observation 都留痕在里面，
// 正对应你笔记里「把 observation 回灌」的图像。
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
            const observation = await executeFunctionCall(call)   // 行动 Action
            input.push({                                          // 观察 Observation
                type: "function_call_output",
                call_id: call.call_id,
                output: observation,
            })
        }

    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI 入口：读一行输入（文本或 URL 或文档）
// ─────────────────────────────────────────────────────────────────────────────
async function main(): Promise<void> {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    })

    const query = await rl.question("> ")
    rl.close()

    if (query.trim()) await runExplore(query.trim())
}

main()
