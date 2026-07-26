import { z } from "zod"

/**
 * 知识图谱的领域边界：形状（zod schema）+ 状态（已提交 / 暂存）+ 渲染。
 * 和 explore.ts 的 agent loop / CLI 编排分开，图谱本身「是什么、能不能改」
 * 不该关心是谁（round 0 的批量抽取，还是增量轮的 addConcept/addRelation）在改它。
 */

// ─────────────────────────────────────────────────────────────────────────────
// 领域类型：知识图谱的形状。zod 定义一次，同时得到运行时校验（.parse）和
// 静态类型（z.infer）。
// ─────────────────────────────────────────────────────────────────────────────
export const GraphNodeSchema = z.object({
    id: z.string(),          // 稳定标识，英文 kebab-case，如 "spaced-repetition"
    label: z.string(),       // 展示名（可中文）
    type: z.string(),        // 概念类别
})

export const GraphEdgeSchema = z.object({
    from: z.string(),        // 源节点 id
    to: z.string(),          // 目标节点 id
    relation: z.string(),    // 关系描述，如 "对抗" | "属于"
})

export const KnowledgeGraphSchema = z.object({
    nodes: z.array(GraphNodeSchema),
    edges: z.array(GraphEdgeSchema),
})

export type GraphNode = z.infer<typeof GraphNodeSchema>
export type GraphEdge = z.infer<typeof GraphEdgeSchema>
export type KnowledgeGraph = z.infer<typeof KnowledgeGraphSchema>

// ─────────────────────────────────────────────────────────────────────────────
// 状态：已提交图谱 + 当前这一轮的暂存区。本进程运行期间只有一份，无需 class；
// 只导出只读访问函数，外部不能直接改 let。
// ─────────────────────────────────────────────────────────────────────────────
let committedGraph: KnowledgeGraph = { nodes: [], edges: [] }
let stagedNodes: GraphNode[] = []
let stagedEdges: GraphEdge[] = []

export function getCommittedGraph(): KnowledgeGraph {
    return committedGraph
}

export function getStaged(): { nodes: GraphNode[]; edges: GraphEdge[] } {
    return { nodes: stagedNodes, edges: stagedEdges }
}

// ─────────────────────────────────────────────────────────────────────────────
// 暂存区生命周期：一轮开始 beginRound，期间模型可多次调用 addConceptToStage /
// addRelationToStage，结束后由调用方决定 commitRound（并入）或 discardRound（丢弃）。
// ─────────────────────────────────────────────────────────────────────────────
export function beginRound(): void {
    stagedNodes = []
    stagedEdges = []
}

function allKnownIds(): Set<string> {
    return new Set([
        ...committedGraph.nodes.map((n) => n.id),
        ...stagedNodes.map((n) => n.id),
    ])
}

export function addConceptToStage(args: unknown): string {
    const result = GraphNodeSchema.safeParse(args)
    if (!result.success) {
        return `addConcept verification fail: ${z.prettifyError(result.error)}`
    }
    const node = result.data
    if (allKnownIds().has(node.id)) {
        return `Error: concept id "${node.id}" already exists (already committed or already staged this round). Choose a different id, or skip if it already represents this concept.`
    }
    stagedNodes.push(node)
    return `staged concept "${node.label}" (${node.id})`
}

export function addRelationToStage(args: unknown): string {
    const result = GraphEdgeSchema.safeParse(args)
    if (!result.success) {
        return `addRelation verification fail: ${z.prettifyError(result.error)}`
    }
    const edge = result.data
    const known = allKnownIds()
    const missing = [edge.from, edge.to].filter((id) => !known.has(id))
    if (missing.length > 0) {
        return `Error: relation references unknown concept id(s) ${missing.map((id) => `"${id}"`).join(", ")}. It must already exist in the committed graph or be added via addConcept earlier in this round before you add a relation to/from it.`
    }
    stagedEdges.push(edge)
    return `staged relation ${edge.from} --${edge.relation}--> ${edge.to}`
}

export function commitRound(): void {
    committedGraph = {
        nodes: [...committedGraph.nodes, ...stagedNodes],
        edges: [...committedGraph.edges, ...stagedEdges],
    }
    stagedNodes = []
    stagedEdges = []
}

export function discardRound(): void {
    stagedNodes = []
    stagedEdges = []
}

// ─────────────────────────────────────────────────────────────────────────────
// 渲染：把 KnowledgeGraph 排版成人能扫一眼看懂的文本。纯函数、无副作用——
// 只吃一个 graph、吐一个 string，好测、好复用，和「执行/IO」分开。
//
// 不引入 chalk 等库，直接用 ANSI 转义码上色（先原语，不给学习骨架加噪音）。
// 呈现三块：① 摘要 ② 概念按 type 分组 ③ 关系用 起点 ──关系──▶ 终点。
// 顺带做「悬空边」检测：边引用了不存在的 node id → 标红，这正是审阅时最该看见的。
//
// 可选 highlight 参数：标出本轮新增的节点/边（用于确认视图）。调用方应传入
// 「已提交 + 暂存」合并后的图谱，这样暂存的边即使指向已提交节点也不会被
// 下面的悬空边检测误判。
// ─────────────────────────────────────────────────────────────────────────────
const C = {
    dim: (s: string) => `\x1b[2m${s}\x1b[0m`,        // 灰：次要信息（id）
    bold: (s: string) => `\x1b[1m${s}\x1b[0m`,       // 粗：标题
    cyan: (s: string) => `\x1b[36m${s}\x1b[0m`,      // 青：概念 label
    yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,    // 黄：关系
    red: (s: string) => `\x1b[31m${s}\x1b[0m`,       // 红：悬空边告警
    green: (s: string) => `\x1b[32m${s}\x1b[0m`,     // 绿：本轮新增标记
}

export function renderKnowledgeGraph(
    graph: KnowledgeGraph,
    highlight?: { newNodeIds: Set<string>; newEdgeIndices: Set<number> },
): string {
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
            const isNew = highlight?.newNodeIds.has(n.id) ?? false
            const mark = isNew ? `${C.green("+")} ` : ""
            const label = isNew ? C.green(n.label) : C.cyan(n.label)
            lines.push(`  ${C.dim("│")}   ${mark}${label}  ${C.dim(n.id)}`)
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
    edges.forEach((e, i) => {
        const isNew = highlight?.newEdgeIndices.has(i) ?? false
        const mark = isNew ? `${C.green("+")} ` : ""
        lines.push(`  ${mark}${nameOf(e.from)} ${C.dim("──")}${C.yellow(e.relation)}${C.dim("──▶")} ${nameOf(e.to)}`)
    })

    // 悬空边告警：审阅时最该被看见的完整性问题。
    if (danglingSeen.size > 0) {
        lines.push("")
        lines.push(C.red(`⚠ ${danglingSeen.size} 个悬空引用（边指向了不存在的概念）：${[...danglingSeen].join(", ")}`))
    }

    lines.push("")
    return lines.join("\n")
}
