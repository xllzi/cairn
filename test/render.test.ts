/**
 * graph.ts 的离线测试：渲染 + 暂存区/提交逻辑。
 *
 * 目的：不烧 API 额度，纯验证「图谱 → 可读文本」的渲染逻辑，以及
 * addConcept/addRelation/commitRound/discardRound 这类不依赖模型调用的
 * 纯状态逻辑。
 * 之所以能这样测，是因为 explore.ts 用 import.meta.url 守卫了 main()——
 * 被 import 时不会自动跑 CLI，渲染函数得以被单独调用。
 *
 * 运行：npx tsx test/render.test.ts
 * 这是一个零依赖的手写断言脚本（本阶段先原语，不引入测试框架）。
 */

import {
    renderKnowledgeGraph,
    beginRound,
    addConceptToStage,
    addRelationToStage,
    commitRound,
    discardRound,
    getCommittedGraph,
    getStaged,
} from "../src/graph.ts"

// ── 极简断言工具 ────────────────────────────────────────────────────────────
let passed = 0
let failed = 0

function check(name: string, cond: boolean): void {
    if (cond) {
        passed++
        console.log(`  ✓ ${name}`)
    } else {
        failed++
        console.log(`  ✗ ${name}`)
    }
}

// 去掉 ANSI 转义码，方便对纯文本做断言。
function stripAnsi(s: string): string {
    // eslint-disable-next-line no-control-regex
    return s.replace(/\x1b\[[0-9;]*m/g, "")
}

// ── 用例：一张含悬空边的图谱 ──────────────────────────────────────────────────
const graph = {
    nodes: [
        { id: "machine-learning", label: "机器学习", type: "领域" },
        { id: "supervised-learning", label: "监督学习", type: "方法" },
        { id: "unsupervised-learning", label: "无监督学习", type: "方法" },
        { id: "neural-network", label: "神经网络", type: "模型" },
    ],
    edges: [
        { from: "supervised-learning", to: "machine-learning", relation: "属于" },
        { from: "unsupervised-learning", to: "machine-learning", relation: "属于" },
        { from: "neural-network", to: "supervised-learning", relation: "用于" },
        // 故意的悬空边：ghost-node 不在 nodes 里。
        { from: "neural-network", to: "ghost-node", relation: "关联" },
    ],
}

console.log("renderKnowledgeGraph")
const out = stripAnsi(renderKnowledgeGraph(graph))

// ① 摘要行：概念数 / 关系数正确。
check("摘要显示 4 个概念 · 4 条关系", out.includes("4 个概念 · 4 条关系"))

// ② 概念按 type 分组：三种 type 都作为分组标题出现。
check("按 type 分组：领域", out.includes("领域"))
check("按 type 分组：方法", out.includes("方法"))
check("按 type 分组：模型", out.includes("模型"))

// ③ 概念以 label + id 呈现。
check("概念显示 label（机器学习）", out.includes("机器学习"))
check("概念显示 id（machine-learning）", out.includes("machine-learning"))

// ④ 关系以「起点 关系 终点」的箭头形式呈现。
check("关系箭头含关系词（属于）与箭头（──▶）", out.includes("属于") && out.includes("──▶"))

// ⑤ 悬空边：标出问号 id，并在底部汇总告警。
check("悬空引用被标出（ghost-node?）", out.includes("ghost-node?"))
check("底部悬空边告警（1 个悬空引用）", out.includes("1 个悬空引用") && out.includes("ghost-node"))

// ── 用例：一张干净的图谱不应触发告警 ─────────────────────────────────────────
const clean = {
    nodes: [
        { id: "a", label: "甲", type: "概念" },
        { id: "b", label: "乙", type: "概念" },
    ],
    edges: [{ from: "a", to: "b", relation: "相关" }],
}
const cleanOut = stripAnsi(renderKnowledgeGraph(clean))
check("干净图谱不触发悬空告警", !cleanOut.includes("悬空引用"))

// ── highlight 参数：不传时行为必须和原来完全一致 ─────────────────────────────
// 传空集合（等价于"什么都没高亮"）应该和不传参数渲染出一样的文本——
// 这是签名扩展后的回归保护，防止 highlight 逻辑悄悄改变默认路径的输出。
const outNoHighlightArg = stripAnsi(renderKnowledgeGraph(graph))
const outEmptyHighlight = stripAnsi(
    renderKnowledgeGraph(graph, { newNodeIds: new Set(), newEdgeIndices: new Set() }),
)
check("不传 highlight 与传空 highlight 集合渲染结果一致", outNoHighlightArg === outEmptyHighlight)

// ── highlight 参数：新增边引用已提交节点不应被误判为悬空 ─────────────────────
// 这是本次改动要防的核心回归：确认视图渲染的是"已提交 + 暂存"合并图，
// 暂存边即使指向一个"对暂存区来说陌生、但已提交"的节点，也不该被当悬空边。
const mergedForDiff = {
    nodes: [
        { id: "old-a", label: "旧概念", type: "概念" },
        { id: "new-b", label: "新概念", type: "概念" },
    ],
    edges: [{ from: "old-a", to: "new-b", relation: "关联" }],
}
const diffOutRaw = renderKnowledgeGraph(mergedForDiff, {
    newNodeIds: new Set(["new-b"]),
    newEdgeIndices: new Set([0]),
})
const diffOut = stripAnsi(diffOutRaw)
check("新增边引用已提交节点不应被标记为悬空", !diffOut.includes("悬空引用"))

const newNodeLine = diffOut.split("\n").find((l) => l.includes("新概念"))
const oldNodeLine = diffOut.split("\n").find((l) => l.includes("旧概念"))
check("新增节点行带 + 标记", !!newNodeLine && newNodeLine.includes("+"))
check("非新增节点行不带 + 标记", !!oldNodeLine && !oldNodeLine.includes("+"))

// ── 暂存区 / 提交逻辑 ─────────────────────────────────────────────────────────
// addConceptToStage / addRelationToStage / commitRound / discardRound 是模块级
// 可变状态，按顺序执行；每个场景前显式 beginRound() 避免用例间状态串扰。

beginRound()
check("addConcept 正常暂存", addConceptToStage({ id: "x", label: "X", type: "t" }).startsWith("staged concept"))
check("暂存区里能看到刚加的概念", getStaged().nodes.some((n) => n.id === "x"))

check(
    "addConcept 同轮重复 id 被拒绝",
    addConceptToStage({ id: "x", label: "X2", type: "t" }).includes("already exists"),
)
check("重复 id 拒绝后暂存区没有第二份", getStaged().nodes.filter((n) => n.id === "x").length === 1)

check(
    "addRelation 引用不存在 id 被拒绝",
    addRelationToStage({ from: "x", to: "not-exist", relation: "r" }).includes("unknown concept id"),
)
check("被拒绝的关系没有进暂存区", getStaged().edges.length === 0)

check(
    "addRelation 引用本轮已暂存的概念成功",
    addRelationToStage({ from: "x", to: "x", relation: "self" }).startsWith("staged relation"),
)

commitRound()
check("commitRound 后已提交图谱包含该概念", getCommittedGraph().nodes.some((n) => n.id === "x"))
check("commitRound 后暂存区清空", getStaged().nodes.length === 0 && getStaged().edges.length === 0)

beginRound()
check(
    "addConcept 对已提交的 id 也拒绝",
    addConceptToStage({ id: "x", label: "X3", type: "t" }).includes("already exists"),
)

addConceptToStage({ id: "y", label: "Y", type: "t" })
discardRound()
check("discardRound 清空暂存区", getStaged().nodes.length === 0)
check("discardRound 不影响已提交图谱", !getCommittedGraph().nodes.some((n) => n.id === "y"))

// ── 汇总 ─────────────────────────────────────────────────────────────────────
console.log(`\n${failed === 0 ? "✓ 全部通过" : "✗ 有失败"}：${passed} passed, ${failed} failed`)
process.exit(failed === 0 ? 0 : 1)
