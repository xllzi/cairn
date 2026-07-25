/**
 * renderKnowledgeGraph 的离线测试。
 *
 * 目的：不烧 API 额度，纯验证「图谱 → 可读文本」的渲染逻辑。
 * 之所以能这样测，是因为 explore.ts 用 import.meta.url 守卫了 main()——
 * 被 import 时不会自动跑 CLI，渲染函数得以被单独调用。
 *
 * 运行：npx tsx test/render.test.ts
 * 这是一个零依赖的手写断言脚本（本阶段先原语，不引入测试框架）。
 */

import { renderKnowledgeGraph } from "../src/explore.ts"

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

// ── 汇总 ─────────────────────────────────────────────────────────────────────
console.log(`\n${failed === 0 ? "✓ 全部通过" : "✗ 有失败"}：${passed} passed, ${failed} failed`)
process.exit(failed === 0 ? 0 : 1)
