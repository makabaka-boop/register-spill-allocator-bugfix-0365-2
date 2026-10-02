#!/usr/bin/env python3
"""不理解源语言语义的寄存器分配器。

输入 JSON 的结构见 README.md。分配器只使用每条指令声明的 reads / defines /
clobbers 以及变量的寄存器域、固定寄存器和带权亲和提示：

1. 在可能带循环的 CFG 上做活跃变量数据流不动点；
2. 自底向上扫描各块，建立临时变量之间的干涉图，并按「读取→改写→定义」的
   时序累计每个变量必须避开的调用改写寄存器；
3. 枚举至多 2^12 个溢出集合，并用带前向检查的回溯做合法着色；着色同时满足
   干涉、固定寄存器与调用改写限制三类硬约束；
4. 按「总溢出代价、已满足亲和权重总和（越大越优）、排序后的溢出变量序列、
   寄存器映射字典序」选择唯一答案。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional


class AllocatorError(ValueError):
    """输入程序不合法。"""


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AllocatorError(f"{description}必须是对象")
    return value


def _require_list(value: Any, description: str) -> list[Any]:
    if not isinstance(value, list):
        raise AllocatorError(f"{description}必须是数组")
    return value


def _identifier(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise AllocatorError(f"{description}必须是非空字符串")
    return value


def validate_program(program: Any) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], Optional[str]]:
    """检查原始 JSON 结构，返回变量表、基本块和可选入口块。"""
    root = _require_object(program, "程序")

    if "variables" not in root:
        raise AllocatorError("缺少 variables")
    if "blocks" not in root:
        raise AllocatorError("缺少 blocks")

    raw_variables = _require_list(root["variables"], "variables")
    raw_blocks = _require_list(root["blocks"], "blocks")

    if len(raw_variables) > 12:
        raise AllocatorError("临时变量数量不能超过 12")
    if not raw_blocks:
        raise AllocatorError("至少需要一个基本块")
    if len(raw_blocks) > 30:
        raise AllocatorError("基本块数量不能超过 30")

    variables: dict[str, dict[str, Any]] = {}
    for index, raw_variable in enumerate(raw_variables):
        item = _require_object(raw_variable, f"variables[{index}]")
        variable_id = _identifier(item.get("id"), f"variables[{index}].id")
        if variable_id in variables:
            raise AllocatorError(f"重复变量: {variable_id}")

        cost = item.get("spill_cost")
        if not _is_int(cost) or cost <= 0:
            raise AllocatorError(f"变量 {variable_id} 的 spill_cost 必须是正整数")

        registers = _require_list(item.get("registers"), f"变量 {variable_id} 的 registers")
        if not 2 <= len(registers) <= 4:
            raise AllocatorError(f"变量 {variable_id} 必须有 2 到 4 个可用寄存器")
        normalized_registers: list[str] = []
        for register in registers:
            name = _identifier(register, f"变量 {variable_id} 的寄存器")
            if name in normalized_registers:
                raise AllocatorError(f"变量 {variable_id} 的寄存器 {name} 重复")
            normalized_registers.append(name)

        pinned_register = item.get("pinned_register")
        if pinned_register is not None and pinned_register not in normalized_registers:
            raise AllocatorError(f"变量 {variable_id} 的固定寄存器不在候选集合中")
        prefer_same_as = item.get("prefer_same_as")
        if prefer_same_as is not None:
            prefer_same_as = _identifier(prefer_same_as, f"变量 {variable_id} 的亲和变量")
        affinity_weight = item.get("affinity_weight", 1)
        if not _is_int(affinity_weight) or affinity_weight <= 0:
            raise AllocatorError(f"变量 {variable_id} 的 affinity_weight 必须是正整数")

        variables[variable_id] = {
            "spill_cost": cost,
            "registers": normalized_registers,
            "pinned_register": pinned_register,
            "prefer_same_as": prefer_same_as,
            "affinity_weight": affinity_weight,
        }

    for variable_id, info in variables.items():
        partner = info["prefer_same_as"]
        if partner is not None and (partner not in variables or partner == variable_id):
            raise AllocatorError(f"变量 {variable_id} 的亲和变量无效")

    blocks: list[dict[str, Any]] = []
    block_ids: set[str] = set()
    for block_index, raw_block in enumerate(raw_blocks):
        block = _require_object(raw_block, f"blocks[{block_index}]")
        block_id = _identifier(block.get("id"), f"blocks[{block_index}].id")
        if block_id in block_ids:
            raise AllocatorError(f"重复基本块: {block_id}")
        block_ids.add(block_id)

        instructions = _require_list(block.get("instructions", []), f"块 {block_id} 的 instructions")
        normalized_instructions: list[dict[str, Optional[str]]] = []
        for instruction_index, raw_instruction in enumerate(instructions):
            instruction = _require_object(
                raw_instruction,
                f"块 {block_id} 的 instructions[{instruction_index}]",
            )

            reads_value = instruction.get("reads", [])
            reads = _require_list(
                reads_value,
                f"块 {block_id} 的 instructions[{instruction_index}].reads",
            )
            normalized_reads: list[str] = []
            for variable in reads:
                variable_id = _identifier(
                    variable,
                    f"块 {block_id} 的 instructions[{instruction_index}] 中的读取变量",
                )
                if variable_id not in variables:
                    raise AllocatorError(f"未知变量: {variable_id}")
                if variable_id not in normalized_reads:
                    normalized_reads.append(variable_id)

            defined = instruction.get("defines")
            if defined is not None:
                defined = _identifier(
                    defined,
                    f"块 {block_id} 的 instructions[{instruction_index}].defines",
                )
                if defined not in variables:
                    raise AllocatorError(f"未知变量: {defined}")

            clobbers_value = _require_list(
                instruction.get("clobbers", []),
                f"块 {block_id} 的 instructions[{instruction_index}].clobbers",
            )
            clobbers = tuple(
                _identifier(register, f"块 {block_id} 的 instructions[{instruction_index}] 中的改写寄存器")
                for register in clobbers_value
            )

            # 额外字段（例如 op）不影响分配，允许保留在输入中但不参与计算。
            normalized_instructions.append({
                "reads": tuple(normalized_reads), "defines": defined, "clobbers": clobbers,
            })

        successors_value = block.get("successors", [])
        successors = _require_list(successors_value, f"块 {block_id} 的 successors")
        normalized_successors: list[str] = []
        for successor in successors:
            successor_id = _identifier(successor, f"块 {block_id} 的后继")
            if successor_id in normalized_successors:
                raise AllocatorError(f"块 {block_id} 存在重复后继: {successor_id}")
            normalized_successors.append(successor_id)

        blocks.append(
            {
                "id": block_id,
                "instructions": normalized_instructions,
                "successors": normalized_successors,
            }
        )

    # 所有块 ID 收集后才能检查后继是否存在。
    for block in blocks:
        for successor in block["successors"]:
            if successor not in block_ids:
                raise AllocatorError(f"块 {block['id']} 有非法后继: {successor}")

    entry = None
    if "entry" in root and root["entry"] is not None:
        entry = _identifier(root["entry"], "entry")
        if entry not in block_ids:
            raise AllocatorError(f"入口块不存在: {entry}")

    return variables, blocks, entry


def loads_program(text: str) -> dict[str, Any]:
    """解析 JSON，并拒绝同一对象中的重复键。"""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AllocatorError(f"JSON 对象中存在重复键: {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(text, object_pairs_hook=reject_duplicates)
    except json.JSONDecodeError as exc:
        raise AllocatorError(f"JSON 解析失败: {exc.msg}") from exc
    return parsed


# ---------------------------------------------------------------------------
# 活跃变量不动点
# ---------------------------------------------------------------------------


def liveness_fixed_point(
    variables: dict[str, dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    live_in = {block["id"]: set() for block in blocks}
    live_out = {block["id"]: set() for block in blocks}

    while True:
        changed = False
        for block in blocks:
            block_id = block["id"]

            new_out: set[str] = set()
            for successor in block["successors"]:
                new_out.update(live_in[successor])

            live = set(new_out)
            for instruction in reversed(block["instructions"]):
                defined = instruction["defines"]
                if defined is not None:
                    live.discard(defined)
                live.update(instruction["reads"])

            if new_out != live_out[block_id] or live != live_in[block_id]:
                changed = True
                live_out[block_id] = new_out
                live_in[block_id] = live

        if not changed:
            return live_in, live_out


# ---------------------------------------------------------------------------
# 干涉图
# ---------------------------------------------------------------------------


def build_interference(
    variables: dict[str, dict[str, Any]],
    blocks: list[dict[str, Any]],
    live_out: dict[str, set[str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """建立干涉图，并计算调用改写寄存器对每个变量的限制。

    返回 (干涉图, register_restrictions)。register_restrictions[v] 是
    所有「v 的活跃区间跨越该调用点」的 clobbers 的并集：指令读取发生在改写前、
    定义发生在改写后，所以仅在本指令读取后死亡的变量、以及本指令新定义的变量
    都不算跨越调用点，只有 live_after − {defines} 中的变量必须避开这些寄存器。
    多个调用点的限制取并集，变量因此同时避开它所跨越的全部调用。
    """
    graph = {variable_id: set() for variable_id in variables}
    restrictions = {variable_id: set() for variable_id in variables}

    def add_edge(left: str, right: str) -> None:
        if left == right:
            return
        graph[left].add(right)
        graph[right].add(left)

    for block in blocks:
        live = set(live_out[block["id"]])

        # 指令语义按“先读取、后定义”处理：一个只在本指令读取后死亡的变量，
        # 与本指令新定义的变量不发生干涉，因而可以复用同一寄存器。
        for instruction in reversed(block["instructions"]):
            defined = instruction["defines"]
            clobbers = set(instruction["clobbers"])

            # live 此刻恰为 live_after(指令)。本指令读取在改写前已完成、
            # 本指令的定义在改写后才产生，二者都无需避开 clobbers；
            # 其余 live 变量必须活着穿越整条指令，不能停在被改写寄存器里。
            survivors = set(live)
            if defined is not None:
                survivors.discard(defined)
            for variable in survivors:
                restrictions[variable].update(clobbers)

            if defined is not None:
                for other in sorted(live):
                    add_edge(defined, other)
                live.discard(defined)

            # 同一指令的多个读取同时存活；排序只保证构造过程确定。
            for variable in sorted(set(instruction["reads"])):
                for other in sorted(live):
                    add_edge(variable, other)
                live.add(variable)

    return graph, restrictions


def interference_edges(graph: dict[str, set[str]]) -> list[list[str]]:
    edges = set()
    for left, neighbors in graph.items():
        for right in neighbors:
            edges.add(tuple(sorted((left, right))))
    return [list(edge) for edge in sorted(edges)]


# ---------------------------------------------------------------------------
# 着色与溢出选择
# ---------------------------------------------------------------------------


def color_without_spills(
    variables: dict[str, dict[str, Any]],
    graph: dict[str, set[str]],
    spilled: set[str],
    restrictions: Optional[dict[str, set[str]]] = None,
) -> Optional[tuple[dict[str, str], int]]:
    """对未溢出变量找合法着色。

    硬约束有三类：
    - 干涉变量不能同寄存器；
    - pinned_register 固定的变量只能取该寄存器（无法满足时该变量必须被溢出，
      由外层枚举溢出集合处理）；
    - 活跃区间跨越调用点的变量不能取该调用 clobbers 中的寄存器。

    对给定溢出集合，先最大化已满足亲和提示（prefer_same_as 与对方同色）的
    权重总和；收益相同时，寄存器仍按名称字典序枚举，搜索按变量 ID 字典序
    推进，因此第一个最优解就是字典序最小的合法映射。不存在合法着色时返回 None。
    """
    if restrictions is None:
        restrictions = {variable_id: set() for variable_id in variables}

    order = sorted(variable_id for variable_id in variables if variable_id not in spilled)
    colors: dict[str, str] = {}

    # 只统计两端都未溢出的亲和提示；任何一端被溢出都不可能满足。
    active_edges: list[tuple[str, str, int]] = []
    for variable_id in order:
        partner = variables[variable_id]["prefer_same_as"]
        if partner is not None and partner not in spilled:
            active_edges.append(
                (variable_id, partner, variables[variable_id]["affinity_weight"])
            )

    def available_colors(variable_id: str) -> list[str]:
        info = variables[variable_id]
        pinned = info["pinned_register"]
        pool = [pinned] if pinned is not None else sorted(info["registers"])
        used = {colors[neighbor] for neighbor in graph[variable_id] if neighbor in colors}
        forbidden = restrictions.get(variable_id, set())
        return [
            register
            for register in pool
            if register not in used and register not in forbidden
        ]

    def affinity_delta(variable_id: str, register: str) -> int:
        """本变量着色时恰好定局的亲和权重。

        一条亲和边在两端中较晚上色的变量被赋值时结算一次：不管它是发起方
        还是伙伴方，只要先着色的另一端颜色相同就计入收益。
        """
        gain = 0
        for source, partner, weight in active_edges:
            if variable_id == source:
                if colors.get(partner) == register:
                    gain += weight
            elif variable_id == partner and source in colors:
                if colors[source] == register:
                    gain += weight
        return gain

    best_gain: Optional[int] = None
    best_coloring: Optional[dict[str, str]] = None

    def search(index: int, gain: int) -> None:
        nonlocal best_gain, best_coloring

        # 亲和边在两端中较晚着色的变量赋值时结算：只要还有至少一端未着色，
        # 该提示就仍可能被满足（潜在权重 = 自身权重，与选择路径无关）。
        # 发起方已定且与伙伴当时颜色不同会失去收益，但那会立即在伙伴着色时
        # 反映到 gain 里；这里用「仍未定局的边」求和作为可纳上界。
        if best_gain is not None:
            pending = 0
            for source, partner, weight in active_edges:
                if source not in colors or partner not in colors:
                    pending += weight
            # 上界不超过已找到的最优即可剪枝：严格小于无法反超；等于时也无需探索，
            # 合法完整解按变量 ID、再按寄存器名的字典序依次到达（前向检查只剪掉
            # 无解分支），已记录的同收益解必然字典序更靠前。
            if gain + pending <= best_gain:
                return

        if index == len(order):
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_coloring = dict(colors)
            return

        variable_id = order[index]
        candidates = available_colors(variable_id)

        for register in candidates:
            colors[variable_id] = register

            # 只有当前变量的未着色邻居的可用颜色集合会被本次赋值改变。
            feasible = True
            for neighbor in graph[variable_id]:
                if neighbor not in colors and neighbor not in spilled:
                    if not available_colors(neighbor):
                        feasible = False
                        break

            if feasible:
                search(index + 1, gain + affinity_delta(variable_id, register))

        colors.pop(variable_id, None)

    search(0, 0)
    if best_coloring is None:
        return None
    return best_coloring, best_gain or 0


def choose_allocation(
    variables: dict[str, dict[str, Any]],
    graph: dict[str, set[str]],
    restrictions: dict[str, set[str]],
) -> tuple[set[str], dict[str, str], int, int]:
    variable_ids = sorted(variables)
    # 裁决顺序：溢出代价最小 → 已满足亲和收益最大（用负值转成比较）→
    # 排序后的溢出变量序列字典序最小 → 未溢出变量映射字典序最小。
    best_key: Optional[
        tuple[int, int, tuple[str, ...], tuple[tuple[str, str], ...]]
    ] = None
    best_spill: Optional[set[str]] = None
    best_coloring: Optional[dict[str, str]] = None
    best_cost = 0
    best_gain = 0

    # 至多 12 个变量，完整枚举 4096 个溢出集合即足够且可被测试直接对拍。
    for mask in range(1 << len(variable_ids)):
        spilled = {
            variable_id
            for bit, variable_id in enumerate(variable_ids)
            if mask & (1 << bit)
        }
        spill_sequence = tuple(sorted(spilled))
        spill_cost = sum(variables[variable_id]["spill_cost"] for variable_id in spilled)

        # 仅溢出代价更差即可跳过：后续关键字（亲和收益、序列、映射）无法挽回。
        if best_key is not None and spill_cost > best_key[0]:
            continue

        colored = color_without_spills(variables, graph, spilled, restrictions)
        if colored is None:
            continue
        coloring, affinity_gain = colored

        mapping_key = tuple(
            (variable_id, coloring[variable_id]) for variable_id in sorted(coloring)
        )
        key = (spill_cost, -affinity_gain, spill_sequence, mapping_key)
        if best_key is None or key < best_key:
            best_key = key
            best_spill = set(spilled)
            best_coloring = coloring
            best_cost = spill_cost
            best_gain = affinity_gain

    # 全部溢出总是可行（着色为空，固定与改写约束无从冲突），断言对合法输入恒成立。
    assert best_spill is not None and best_coloring is not None
    return best_spill, best_coloring, best_cost, best_gain


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------


def allocate(program: Any) -> dict[str, Any]:
    variables, blocks, _entry = validate_program(program)
    live_in, live_out = liveness_fixed_point(variables, blocks)
    graph, restrictions = build_interference(variables, blocks, live_out)
    spilled, coloring, spill_cost, affinity_gain = choose_allocation(
        variables, graph, restrictions
    )

    return {
        "liveness": {
            block["id"]: {
                "in": sorted(live_in[block["id"]]),
                "out": sorted(live_out[block["id"]]),
            }
            for block in blocks
        },
        "interference_edges": interference_edges(graph),
        # 每个变量需要避开的调用改写寄存器（其活跃区间跨越的全部调用点并集）。
        "register_restrictions": {
            variable_id: sorted(restrictions[variable_id])
            for variable_id in sorted(variables)
            if restrictions[variable_id]
        },
        "spilled": sorted(spilled),
        "spill_cost": spill_cost,
        "affinity_gain": affinity_gain,
        "allocation": {variable_id: coloring[variable_id] for variable_id in sorted(coloring)},
        "preferences": {
            variable_id: {
                "pinned_register": info["pinned_register"],
                "prefer_same_as": info["prefer_same_as"],
                "affinity_weight": info["affinity_weight"],
            }
            for variable_id, info in variables.items()
            if info["pinned_register"] is not None or info["prefer_same_as"] is not None
        },
    }


def allocate_text(text: str) -> dict[str, Any]:
    return allocate(loads_program(text))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="枚举最优溢出集合的寄存器分配器")
    parser.add_argument("input", nargs="?", help="输入 JSON 文件；省略时从标准输入读取")
    args = parser.parse_args(argv)

    try:
        if args.input:
            with open(args.input, "r", encoding="utf-8") as source:
                text = source.read()
        else:
            text = sys.stdin.read()
        result = allocate_text(text)
    except OSError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except AllocatorError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2

    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
