#!/usr/bin/env python3
"""不理解源语言语义的寄存器分配器。

输入 JSON 的结构见 README.md。分配器只使用每条指令声明的 reads / defines：

1. 在可能带循环的 CFG 上做活跃变量数据流不动点；
2. 自底向上扫描各块，建立临时变量之间的干涉图；调用类指令的 clobbers
   转成「跨越改写点的变量不得使用这些寄存器」的一元约束；
3. 枚举至多 2^12 个溢出集合，并用带前向检查和亲和收益分支限界的回溯做合法着色；
4. 按「总溢出代价、排序后的溢出变量序列、亲和收益、寄存器映射字典序」选择唯一答案。

固定寄存器（pinned_register）是硬约束：保留在寄存器中的变量只能着该色，
该色被调用改写禁用或固定条件彼此冲突时，相关变量必须溢出。
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
        if pinned_register is not None:
            pinned_register = _identifier(
                pinned_register, f"变量 {variable_id} 的 pinned_register"
            )
            if pinned_register not in normalized_registers:
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
    """建立干涉图，并返回每个变量被调用改写禁用的寄存器集合。

    指令时序为「先读取、调用改写、后定义」：本指令读取的变量在改写前已
    使用完毕，本指令定义的变量在改写后才产生；只有在改写点之后仍然存活
    的其他变量才算跨越调用，不能留在 clobbers 列出的寄存器中。
    """
    graph = {variable_id: set() for variable_id in variables}
    forbidden: dict[str, set[str]] = {variable_id: set() for variable_id in variables}

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
            # live 即 live_after(I)：改写发生在读取之后、定义之前，因此跨越
            # 改写点的是 live_after 中除定义变量外的所有变量（块内或跨块
            # 存活、以及多个调用点的约束都在 live 集合中统一体现）。
            crossing = live
            if instruction["defines"] is not None:
                crossing = live - {instruction["defines"]}
            clobbered = set(instruction["clobbers"])
            if clobbered:
                for variable_id in crossing:
                    forbidden[variable_id].update(clobbered)

            defined = instruction["defines"]
            if defined is not None:
                for other in sorted(live):
                    add_edge(defined, other)
                live.discard(defined)

            # 同一指令的多个读取同时存活；排序只保证构造过程确定。
            for variable in sorted(set(instruction["reads"])):
                for other in sorted(live):
                    add_edge(variable, other)
                live.add(variable)

    return graph, forbidden


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
    forbidden: dict[str, set[str]],
    spilled: set[str],
) -> Optional[tuple[dict[str, str], int]]:
    """为未溢出变量找亲和收益最大的合法着色。

    返回 ``(寄存器映射, 亲和收益)``；不存在合法着色时返回 None。收益相同时
    按变量 ID 排序后的映射字典序最小——回溯严格按字典序枚举，先找到的
    最优解即为字典序最小解。

    硬约束有三类：
    - 干涉邻居不能同色；
    - 跨越调用的变量不能着被该调用改写的寄存器（forbidden）；
    - 固定变量只能着 pinned_register，该色不可行时本溢出集合无解。

    亲和收益按有向提示逐条计算：变量 v 与 prefer_same_as p 都未溢出且最终
    同色时，获得 v 的 affinity_weight；双向提示的两侧分别计分。
    """
    order = sorted(variable_id for variable_id in variables if variable_id not in spilled)
    colors: dict[str, str] = {}

    # 只保留双方都未溢出的提示；溢出的变量不可能兑现亲和。
    preferences = [
        (variable_id, info["prefer_same_as"], info["affinity_weight"])
        for variable_id, info in variables.items()
        if variable_id not in spilled
        and info["prefer_same_as"] is not None
        and info["prefer_same_as"] not in spilled
    ]

    def available_colors(variable_id: str) -> list[str]:
        info = variables[variable_id]
        pinned = info["pinned_register"]
        candidates = [pinned] if pinned is not None else sorted(info["registers"])
        used = {colors[neighbor] for neighbor in graph[variable_id] if neighbor in colors}
        return [
            register
            for register in candidates
            if register not in used and register not in forbidden[variable_id]
        ]

    def affinity_upper_bound() -> int:
        """当前部分赋值下仍可能兑现的亲和收益上界（含已满足项）。"""
        total = 0
        for source, partner, weight in preferences:
            source_colored = source in colors
            partner_colored = partner in colors
            if source_colored and partner_colored:
                if colors[source] == colors[partner]:
                    total += weight
            elif source_colored:
                if colors[source] in available_colors(partner):
                    total += weight
            elif partner_colored:
                if colors[partner] in available_colors(source):
                    total += weight
            elif set(available_colors(source)) & set(available_colors(partner)):
                total += weight
        return total

    best_coloring: Optional[dict[str, str]] = None
    best_gain = -1

    def search(index: int) -> None:
        nonlocal best_coloring, best_gain

        # 上界不严格大于已知最优时，剩余子树至多追平；而字典序更小的解必然
        # 已在更早的叶子找到，可直接剪枝。
        if affinity_upper_bound() <= best_gain:
            return

        if index == len(order):
            gain = sum(
                weight
                for source, partner, weight in preferences
                if colors[source] == colors[partner]
            )
            if gain > best_gain:
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
                search(index + 1)

        colors.pop(variable_id, None)

    search(0)
    if best_coloring is None:
        return None
    return best_coloring, best_gain


def choose_allocation(
    variables: dict[str, dict[str, Any]],
    graph: dict[str, set[str]],
    forbidden: dict[str, set[str]],
) -> tuple[set[str], dict[str, str], int, int]:
    variable_ids = sorted(variables)
    best_key: Optional[
        tuple[int, tuple[str, ...], int, tuple[tuple[str, str], ...]]
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

        # 第一关键字已不可能改善时直接跳过（后续并列裁决仍需完整比较）。
        if best_key is not None and spill_cost > best_key[0]:
            continue

        colored = color_without_spills(variables, graph, forbidden, spilled)
        if colored is None:
            continue
        coloring, affinity_gain = colored

        mapping_key = tuple((variable_id, coloring[variable_id]) for variable_id in sorted(coloring))
        # 代价最小 → 溢出序列字典序最小 → 亲和收益最大 → 映射字典序最小。
        key = (spill_cost, spill_sequence, -affinity_gain, mapping_key)
        if best_key is None or key < best_key:
            best_key = key
            best_spill = set(spilled)
            best_coloring = coloring
            best_cost = spill_cost
            best_gain = affinity_gain

    # 全部溢出总是可行，因此下面断言对任何合法输入都成立。
    assert best_spill is not None and best_coloring is not None
    return best_spill, best_coloring, best_cost, best_gain


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------


def allocate(program: Any) -> dict[str, Any]:
    variables, blocks, _entry = validate_program(program)
    live_in, live_out = liveness_fixed_point(variables, blocks)
    graph, forbidden = build_interference(variables, blocks, live_out)
    spilled, coloring, spill_cost, affinity_gain = choose_allocation(variables, graph, forbidden)

    # 只回显落在该变量候选集合内的禁用寄存器；其余寄存器本来就不可选。
    clobber_restrictions = {
        variable_id: sorted(
            register
            for register in forbidden[variable_id]
            if register in variables[variable_id]["registers"]
        )
        for variable_id in sorted(variables)
    }
    clobber_restrictions = {
        variable_id: registers
        for variable_id, registers in clobber_restrictions.items()
        if registers
    }

    # 逐条说明亲和提示的兑现情况，便于解释并列裁决。
    affinity_report = {}
    for variable_id in sorted(variables):
        info = variables[variable_id]
        partner = info["prefer_same_as"]
        if partner is None:
            continue
        satisfied = (
            variable_id in coloring
            and partner in coloring
            and coloring[variable_id] == coloring[partner]
        )
        affinity_report[variable_id] = {
            "partner": partner,
            "weight": info["affinity_weight"],
            "satisfied": satisfied,
        }

    return {
        "liveness": {
            block["id"]: {
                "in": sorted(live_in[block["id"]]),
                "out": sorted(live_out[block["id"]]),
            }
            for block in blocks
        },
        "interference_edges": interference_edges(graph),
        "clobber_restrictions": clobber_restrictions,
        "spilled": sorted(spilled),
        "spill_cost": spill_cost,
        "affinity_gain": affinity_gain,
        "satisfied_affinities": affinity_report,
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
