import itertools
import json
import unittest

from allocator import AllocatorError, allocate, allocate_text


def brute_force_optimum(variables, edges):
    """独立暴力对拍：枚举每个溢出集合和每个未溢出变量的所有寄存器选择。"""
    ids = sorted(variables)
    adjacency = {variable_id: set() for variable_id in ids}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    best = None
    for mask in range(1 << len(ids)):
        spilled = {ids[i] for i in range(len(ids)) if mask & (1 << i)}
        cost = sum(variables[variable_id]["spill_cost"] for variable_id in spilled)
        kept = [variable_id for variable_id in ids if variable_id not in spilled]
        domains = [sorted(variables[variable_id]["registers"]) for variable_id in kept]

        for values in itertools.product(*domains):
            assignment = dict(zip(kept, values))
            legal = True
            for left, right in edges:
                if left in assignment and right in assignment:
                    if assignment[left] == assignment[right]:
                        legal = False
                        break
            if not legal:
                continue

            key = (
                cost,
                tuple(sorted(spilled)),
                tuple((variable_id, assignment[variable_id]) for variable_id in kept),
            )
            if best is None or key < best[0]:
                best = (key, set(spilled), assignment, cost)

    assert best is not None
    return best[1], best[2], best[3]


def edge_set(result):
    return {tuple(edge) for edge in result["interference_edges"]}


class AllocatorTests(unittest.TestCase):
    def assert_matches_brute_force(self, program, expected_liveness):
        result = allocate(program)
        variables = {
            item["id"]: {
                "spill_cost": item["spill_cost"],
                "registers": item["registers"],
            }
            for item in program["variables"]
        }
        spilled, assignment, cost = brute_force_optimum(
            variables, result["interference_edges"]
        )

        self.assertEqual(result["liveness"], expected_liveness)
        self.assertEqual(set(result["spilled"]), spilled)
        self.assertEqual(result["spill_cost"], cost)
        self.assertEqual(result["allocation"], assignment)

    def test_loop_backedge_liveness_and_two_register_coloring(self):
        program = {
            "variables": [
                {"id": "a", "spill_cost": 100, "registers": ["r0", "r1"]},
                {"id": "b", "spill_cost": 100, "registers": ["r0", "r1"]},
                {"id": "c", "spill_cost": 100, "registers": ["r0", "r1"]},
                {"id": "x", "spill_cost": 100, "registers": ["r0", "r1"]},
            ],
            "blocks": [
                {
                    "id": "entry",
                    "instructions": [{"reads": [], "defines": "x"}],
                    "successors": ["loop"],
                },
                {
                    "id": "loop",
                    "instructions": [
                        {"reads": ["x"], "defines": "a"},
                        {"reads": ["a"], "defines": "b"},
                        {"reads": ["b", "x"], "defines": "x"},
                    ],
                    # 回边和离开循环的分支同时存在。
                    "successors": ["after", "loop"],
                },
                {
                    "id": "after",
                    "instructions": [{"reads": ["x"], "defines": "c"}],
                    "successors": ["exit"],
                },
                {
                    "id": "exit",
                    "instructions": [{"reads": ["c"], "defines": None}],
                    "successors": [],
                },
            ],
        }

        expected_liveness = {
            "entry": {"in": [], "out": ["x"]},
            "loop": {"in": ["x"], "out": ["x"]},
            "after": {"in": ["x"], "out": ["c"]},
            "exit": {"in": ["c"], "out": []},
        }
        result = allocate(program)

        self.assertEqual(edge_set(result), {("a", "x"), ("b", "x")})
        self.assertEqual(result["spilled"], [])
        self.assertEqual(
            result["allocation"], {"a": "r0", "b": "r0", "c": "r0", "x": "r1"}
        )
        self.assert_matches_brute_force(program, expected_liveness)

    def test_branch_join_interference_and_cheapest_spill(self):
        program = {
            "variables": [
                {"id": "x", "spill_cost": 100, "registers": ["r0", "r1"]},
                {"id": "y", "spill_cost": 100, "registers": ["r0", "r1"]},
                {"id": "z", "spill_cost": 1, "registers": ["r0", "r1"]},
            ],
            "blocks": [
                {
                    "id": "entry",
                    "instructions": [
                        {"reads": [], "defines": "z"},
                        {"reads": [], "defines": "x"},
                        {"reads": [], "defines": "y"},
                    ],
                    "successors": ["then", "else"],
                },
                {
                    "id": "then",
                    "instructions": [{"reads": ["y"], "defines": "x"}],
                    "successors": ["join"],
                },
                {
                    "id": "else",
                    "instructions": [{"reads": ["x"], "defines": "y"}],
                    "successors": ["join"],
                },
                {
                    "id": "join",
                    "instructions": [{"reads": ["x", "y", "z"], "defines": None}],
                    "successors": ["exit"],
                },
                {
                    "id": "exit",
                    "instructions": [{"reads": ["z"], "defines": None}],
                    "successors": [],
                },
            ],
        }

        expected_liveness = {
            "entry": {"in": [], "out": ["x", "y", "z"]},
            "then": {"in": ["y", "z"], "out": ["x", "y", "z"]},
            "else": {"in": ["x", "z"], "out": ["x", "y", "z"]},
            "join": {"in": ["x", "y", "z"], "out": ["z"]},
            "exit": {"in": ["z"], "out": []},
        }
        result = allocate(program)

        self.assertEqual(
            edge_set(result), {("x", "y"), ("x", "z"), ("y", "z")}
        )
        self.assertEqual(result["spilled"], ["z"])
        self.assertEqual(result["spill_cost"], 1)
        self.assertEqual(result["allocation"], {"x": "r0", "y": "r1"})
        self.assert_matches_brute_force(program, expected_liveness)

    def test_read_before_definition_allows_register_reuse(self):
        program = {
            "variables": [
                {"id": "old", "spill_cost": 10, "registers": ["r0", "r1"]},
                {"id": "new", "spill_cost": 10, "registers": ["r0", "r1"]},
            ],
            "blocks": [
                {
                    "id": "b0",
                    # old 只在本指令读取且死亡，new 在读取后定义，二者不干涉。
                    "instructions": [{"reads": ["old"], "defines": "new"}],
                    "successors": [],
                }
            ],
        }

        result = allocate(program)
        self.assertEqual(result["interference_edges"], [])
        self.assertEqual(result["allocation"], {"new": "r0", "old": "r0"})

    def test_equal_spill_costs_choose_lexicographically_smallest_spill_and_mapping(self):
        variables = [
            {"id": "a", "spill_cost": 1, "registers": ["r0", "r1"]},
            {"id": "b", "spill_cost": 1, "registers": ["r1", "r0"]},
            {"id": "c", "spill_cost": 1, "registers": ["r0", "r1"]},
        ]
        program = {
            "variables": variables,
            "blocks": [
                {
                    "id": "start",
                    "instructions": [
                        {"reads": [], "defines": "a"},
                        {"reads": ["a"], "defines": "b"},
                        {"reads": ["a", "b"], "defines": "c"},
                        {"reads": ["a", "b", "c"], "defines": None},
                    ],
                    "successors": [],
                }
            ],
        }

        result = allocate(program)
        self.assertEqual(result["spilled"], ["a"])
        # 寄存器排序按名称，而不是输入数组顺序；b 取 r0 后 c 只能取 r1。
        self.assertEqual(result["allocation"], {"b": "r0", "c": "r1"})

    def test_restricted_register_pools_are_enumerated(self):
        program = {
            "variables": [
                {"id": "a", "spill_cost": 100, "registers": ["r0", "r2"]},
                {"id": "b", "spill_cost": 100, "registers": ["r0", "r1"]},
                {"id": "c", "spill_cost": 100, "registers": ["r1", "r2"]},
            ],
            "blocks": [
                {
                    "id": "start",
                    "instructions": [
                        {"reads": [], "defines": "a"},
                        {"reads": ["a"], "defines": "b"},
                        {"reads": ["a", "b"], "defines": "c"},
                        {"reads": ["a", "b", "c"], "defines": None},
                    ],
                    "successors": [],
                }
            ],
        }

        result = allocate(program)
        self.assertEqual(result["spilled"], [])
        # 按变量 ID 取字典序最小映射：a 先取其可用集合中最小的 r0。
        self.assertEqual(result["allocation"], {"a": "r0", "b": "r1", "c": "r2"})

    def test_reject_unknown_variables_illegal_successors_and_duplicate_blocks(self):
        valid_variable = {
            "id": "a",
            "spill_cost": 1,
            "registers": ["r0", "r1"],
        }
        valid_block = {"id": "b0", "instructions": [], "successors": []}

        bad_reads = {
            "variables": [valid_variable],
            "blocks": [
                {
                    "id": "b0",
                    "instructions": [{"reads": ["missing"], "defines": None}],
                    "successors": [],
                }
            ],
        }
        bad_defines = {
            "variables": [valid_variable],
            "blocks": [
                {
                    "id": "b0",
                    "instructions": [{"reads": [], "defines": "missing"}],
                    "successors": [],
                }
            ],
        }
        bad_successor = {
            "variables": [valid_variable],
            "blocks": [
                {"id": "b0", "instructions": [], "successors": ["missing"]}
            ],
        }
        duplicate_blocks = {
            "variables": [valid_variable],
            "blocks": [valid_block, dict(valid_block)],
        }

        with self.assertRaisesRegex(AllocatorError, "未知变量"):
            allocate(bad_reads)
        with self.assertRaisesRegex(AllocatorError, "未知变量"):
            allocate(bad_defines)
        with self.assertRaisesRegex(AllocatorError, "非法后继"):
            allocate(bad_successor)
        with self.assertRaisesRegex(AllocatorError, "重复基本块"):
            allocate(duplicate_blocks)

    def test_reject_duplicate_json_keys_and_invalid_limits(self):
        text = """
        {
          "variables": [],
          "variables": [],
          "blocks": [
            {"id": "b0", "instructions": [], "successors": []}
          ]
        }
        """
        with self.assertRaisesRegex(AllocatorError, "重复键"):
            allocate_text(text)

        too_many_variables = {
            "variables": [
                {"id": f"t{i}", "spill_cost": 1, "registers": ["r0", "r1"]}
                for i in range(13)
            ],
            "blocks": [
                {"id": "b0", "instructions": [], "successors": []}
            ],
        }
        with self.assertRaisesRegex(AllocatorError, "12"):
            allocate(too_many_variables)

        bad_cost = {
            "variables": [
                {"id": "a", "spill_cost": 0, "registers": ["r0", "r1"]}
            ],
            "blocks": [valid_block_fixture()],
        }
        with self.assertRaisesRegex(AllocatorError, "正整数"):
            allocate(bad_cost)

        one_register = {
            "variables": [
                {"id": "a", "spill_cost": 1, "registers": ["r0"]}
            ],
            "blocks": [valid_block_fixture()],
        }
        with self.assertRaisesRegex(AllocatorError, "2 到 4"):
            allocate(one_register)

    def test_output_is_json_serializable_and_sorted(self):
        program = {
            "variables": [
                {"id": "b", "spill_cost": 1, "registers": ["r1", "r0"]},
                {"id": "a", "spill_cost": 1, "registers": ["r0", "r1"]},
            ],
            "blocks": [
                {
                    "id": "second",
                    "instructions": [{"reads": ["a", "b"], "defines": None}],
                    "successors": [],
                },
                {
                    "id": "first",
                    "instructions": [{"reads": [], "defines": "a"}],
                    "successors": ["second"],
                },
                {
                    "id": "middle",
                    "instructions": [{"reads": ["a"], "defines": "b"}],
                    "successors": ["second"],
                },
            ],
        }
        # 修正 CFG 后继，使上面的块顺序仍能到达 middle。
        program["blocks"][1]["successors"] = ["middle"]

        result = allocate(program)
        json.dumps(result, ensure_ascii=False)
        self.assertEqual(list(result["liveness"]), ["second", "first", "middle"])
        self.assertEqual(result["liveness"]["second"]["in"], ["a", "b"])


def valid_block_fixture():
    return {"id": "b0", "instructions": [], "successors": []}


def brute_force_full(program):
    """带 clobber/固定寄存器/亲和的独立暴力对拍。"""
    variables = {item["id"]: item for item in program["variables"]}
    blocks = {b["id"]: b for b in program["blocks"]}
    ids = sorted(variables)

    live_in = {block_id: set() for block_id in blocks}
    live_out = {block_id: set() for block_id in blocks}
    while True:
        changed = False
        for block_id in reversed(list(blocks)):
            block = blocks[block_id]
            new_out = set()
            for successor in block.get("successors", []):
                new_out |= live_in[successor]
            live = set(new_out)
            for instruction in reversed(block.get("instructions", [])):
                if instruction.get("defines") is not None:
                    live.discard(instruction["defines"])
                live.update(instruction.get("reads", []))
            if live != live_in[block_id] or new_out != live_out[block_id]:
                changed = True
                live_in[block_id], live_out[block_id] = live, new_out
        if not changed:
            break

    adjacency = {variable_id: set() for variable_id in ids}
    forbidden = {variable_id: set() for variable_id in ids}
    for block in blocks.values():
        live = set(live_out[block["id"]])
        for instruction in reversed(block.get("instructions", [])):
            defined = instruction.get("defines")
            crossing = set(live)
            if defined is not None:
                crossing.discard(defined)
            for register in instruction.get("clobbers", []):
                for variable_id in crossing:
                    forbidden[variable_id].add(register)
            if defined is not None:
                adjacency[defined] |= live
                for other in live:
                    adjacency[other].add(defined)
                live.discard(defined)
            for variable_id in instruction.get("reads", []):
                adjacency[variable_id] |= live
                for other in live:
                    adjacency[other].add(variable_id)
                live.add(variable_id)

    best = None
    for mask in range(1 << len(ids)):
        spilled = {ids[i] for i in range(len(ids)) if mask & (1 << i)}
        kept = [variable_id for variable_id in ids if variable_id not in spilled]
        cost = sum(variables[v]["spill_cost"] for v in spilled)
        domains = []
        for variable_id in kept:
            info = variables[variable_id]
            pool = [info["pinned_register"]] if info.get("pinned_register") else sorted(info["registers"])
            domains.append([r for r in pool if r not in forbidden[variable_id]])

        for values in itertools.product(*domains):
            assignment = dict(zip(kept, values))
            legal = all(
                assignment[left] != assignment[right]
                for left in kept
                for right in adjacency[left]
                if right in assignment and left < right
            )
            if not legal:
                continue
            gain = 0
            for variable_id in kept:
                partner = variables[variable_id].get("prefer_same_as")
                if partner is not None and partner in assignment:
                    if assignment[variable_id] == assignment[partner]:
                        gain += variables[variable_id].get("affinity_weight", 1)
            key = (
                cost,
                tuple(sorted(spilled)),
                -gain,
                tuple((v, assignment[v]) for v in kept),
            )
            if best is None or key < best[0]:
                best = (key, set(spilled), assignment, cost, gain)

    assert best is not None
    return best[1], best[2], best[3], best[4]


class ClobberAndPreferenceTests(unittest.TestCase):
    def assert_matches_full_brute_force(self, program):
        result = allocate(program)
        spilled, assignment, cost, gain = brute_force_full(program)
        self.assertEqual(set(result["spilled"]), spilled)
        self.assertEqual(result["spill_cost"], cost)
        self.assertEqual(result["allocation"], assignment)
        self.assertEqual(result["affinity_gain"], gain)

    def test_clobbered_register_avoided_across_call_within_block(self):
        # a 跨越改写 r0 的调用 → 只能取 r1；随后 a 死亡，b 跨越改写 r1 的
        # 另一个调用 → 只能取 r0。两个调用点的约束分别生效。
        program = {
            "variables": [
                {"id": "a", "spill_cost": 100, "registers": ["r0", "r1"]},
                {"id": "b", "spill_cost": 100, "registers": ["r0", "r1"]},
            ],
            "blocks": [
                {
                    "id": "b0",
                    "instructions": [
                        {"reads": [], "defines": "a"},
                        {"reads": [], "defines": None, "clobbers": ["r0"]},
                        {"reads": ["a"], "defines": None},
                        {"reads": [], "defines": "b"},
                        {"reads": [], "defines": None, "clobbers": ["r1"]},
                        {"reads": ["b"], "defines": None},
                    ],
                    "successors": [],
                }
            ],
        }
        result = allocate(program)
        self.assertEqual(result["clobber_restrictions"], {"a": ["r0"], "b": ["r1"]})
        self.assertEqual(result["spilled"], [])
        self.assertEqual(result["allocation"], {"a": "r1", "b": "r0"})
        self.assert_matches_full_brute_force(program)

    def test_last_read_before_call_and_define_after_call_are_unrestricted(self):
        # a 在调用指令被最后一次读取（读取先于改写），b 在改写后才定义：
        # 二者都不跨越调用，clobber r0 不构成任何限制。
        program = {
            "variables": [
                {"id": "a", "spill_cost": 100, "registers": ["r0", "r1"]},
                {"id": "b", "spill_cost": 100, "registers": ["r0", "r1"]},
            ],
            "blocks": [
                {
                    "id": "b0",
                    "instructions": [
                        {"reads": [], "defines": "a"},
                        {"reads": ["a"], "defines": "b", "clobbers": ["r0"]},
                        {"reads": ["b"], "defines": None},
                    ],
                    "successors": [],
                }
            ],
        }
        result = allocate(program)
        self.assertEqual(result["clobber_restrictions"], {})
        self.assertEqual(result["allocation"], {"a": "r0", "b": "r0"})
        self.assert_matches_full_brute_force(program)

    def test_cross_block_live_variable_avoids_clobber_in_other_block(self):
        # x 在 b0 定义、b2 使用，跨越中间块 b1 中改写 r0 的调用。
        program = {
            "variables": [
                {"id": "x", "spill_cost": 100, "registers": ["r0", "r1"]},
            ],
            "blocks": [
                {
                    "id": "b0",
                    "instructions": [{"reads": [], "defines": "x"}],
                    "successors": ["b1"],
                },
                {
                    "id": "b1",
                    "instructions": [{"reads": [], "defines": None, "clobbers": ["r0"]}],
                    "successors": ["b2"],
                },
                {
                    "id": "b2",
                    "instructions": [{"reads": ["x"], "defines": None}],
                    "successors": [],
                },
            ],
        }
        result = allocate(program)
        self.assertEqual(result["clobber_restrictions"], {"x": ["r0"]})
        self.assertEqual(result["allocation"], {"x": "r1"})
        self.assert_matches_full_brute_force(program)

    def test_loop_with_two_calls_clobbering_all_registers_forces_spill(self):
        # 回边使 x 同时跨越改写 r0 和 r1 的两个调用；两寄存器都被禁用 → 溢出。
        program = {
            "variables": [
                {"id": "x", "spill_cost": 5, "registers": ["r0", "r1"]},
                {"id": "y", "spill_cost": 100, "registers": ["r0", "r1"]},
            ],
            "blocks": [
                {
                    "id": "loop",
                    "instructions": [
                        {"reads": ["x"], "defines": "y"},
                        {"reads": ["y"], "defines": None, "clobbers": ["r0"]},
                        {"reads": ["x"], "defines": None, "clobbers": ["r1"]},
                    ],
                    "successors": ["loop"],
                }
            ],
        }
        result = allocate(program)
        self.assertEqual(result["clobber_restrictions"], {"x": ["r0", "r1"]})
        self.assertEqual(result["spilled"], ["x"])
        self.assertEqual(result["spill_cost"], 5)
        self.assertEqual(result["allocation"], {"y": "r0"})
        self.assert_matches_full_brute_force(program)

    def test_pinned_register_is_hard_constraint(self):
        program = {
            "variables": [
                {"id": "a", "spill_cost": 100, "registers": ["r0", "r1"], "pinned_register": "r1"},
                {"id": "b", "spill_cost": 100, "registers": ["r0", "r1"]},
            ],
            "blocks": [
                {
                    "id": "b0",
                    "instructions": [
                        {"reads": [], "defines": "a"},
                        {"reads": ["a"], "defines": "b"},
                        {"reads": ["a", "b"], "defines": None},
                    ],
                    "successors": [],
                }
            ],
        }
        result = allocate(program)
        self.assertEqual(result["allocation"], {"a": "r1", "b": "r0"})
        self.assert_matches_full_brute_force(program)

    def test_conflicting_pins_spill_lexicographically_smallest_variable(self):
        # 两个固定到 r0 又相互干涉的变量只能溢出一个；代价相同溢出 ID 较小者。
        program = {
            "variables": [
                {"id": "a", "spill_cost": 1, "registers": ["r0", "r1"], "pinned_register": "r0"},
                {"id": "b", "spill_cost": 1, "registers": ["r0", "r1"], "pinned_register": "r0"},
            ],
            "blocks": [
                {
                    "id": "b0",
                    "instructions": [
                        {"reads": [], "defines": "a"},
                        {"reads": ["a"], "defines": "b"},
                        {"reads": ["a", "b"], "defines": None},
                    ],
                    "successors": [],
                }
            ],
        }
        result = allocate(program)
        self.assertEqual(result["spilled"], ["a"])
        self.assertEqual(result["allocation"], {"b": "r0"})
        self.assert_matches_full_brute_force(program)

    def test_pinned_register_clobbered_across_call_forces_spill(self):
        # a 固定 r0 且跨越改写 r0 的调用：固定色不可用，只能溢出 a。
        program = {
            "variables": [
                {"id": "a", "spill_cost": 7, "registers": ["r0", "r1"], "pinned_register": "r0"},
            ],
            "blocks": [
                {
                    "id": "b0",
                    "instructions": [
                        {"reads": [], "defines": "a"},
                        {"reads": [], "defines": None, "clobbers": ["r0"]},
                        {"reads": ["a"], "defines": None},
                    ],
                    "successors": [],
                }
            ],
        }
        result = allocate(program)
        self.assertEqual(result["spilled"], ["a"])
        self.assertEqual(result["spill_cost"], 7)
        self.assertEqual(result["allocation"], {})
        self.assert_matches_full_brute_force(program)

    def test_affinity_gain_breaks_tie_before_lexicographic_mapping(self):
        # 只有 (b,c) 一条干涉边，a 已死亡后 b、c 才定义。
        # 字典序最小映射 a=r0,b=r0,c=r1 收益 0；
        # b 让到 r1 后 c 可与 a 同 r0，获得 c→a 的亲和收益 5。
        program = {
            "variables": [
                {"id": "a", "spill_cost": 100, "registers": ["r0", "r1"]},
                {"id": "b", "spill_cost": 100, "registers": ["r0", "r1"]},
                {"id": "c", "spill_cost": 100, "registers": ["r0", "r1"],
                 "prefer_same_as": "a", "affinity_weight": 5},
            ],
            "blocks": [
                {
                    "id": "b0",
                    "instructions": [
                        {"reads": [], "defines": "a"},
                        {"reads": ["a"], "defines": None},
                        {"reads": [], "defines": "b"},
                        {"reads": [], "defines": "c"},
                        {"reads": ["b", "c"], "defines": None},
                    ],
                    "successors": [],
                }
            ],
        }
        result = allocate(program)
        self.assertEqual(result["spilled"], [])
        self.assertEqual(result["affinity_gain"], 5)
        self.assertEqual(result["allocation"], {"a": "r0", "b": "r1", "c": "r0"})
        self.assertTrue(result["satisfied_affinities"]["c"]["satisfied"])
        self.assert_matches_full_brute_force(program)

        # 去掉亲和提示后，零收益并列回退到映射字典序。
        program["variables"][2] = {"id": "c", "spill_cost": 100, "registers": ["r0", "r1"]}
        plain = allocate(program)
        self.assertEqual(plain["affinity_gain"], 0)
        self.assertEqual(plain["allocation"], {"a": "r0", "b": "r0", "c": "r1"})

    def test_spill_cost_dominates_affinity_gain(self):
        # 零溢出但亲和不满足，仍优于为兑现亲和而溢出（代价优先）。
        program = {
            "variables": [
                {"id": "a", "spill_cost": 100, "registers": ["r0", "r1"],
                 "prefer_same_as": "b", "affinity_weight": 50},
                {"id": "b", "spill_cost": 100, "registers": ["r0", "r1"]},
            ],
            "blocks": [
                {
                    "id": "b0",
                    "instructions": [
                        {"reads": [], "defines": "a"},
                        {"reads": ["a"], "defines": "b"},
                        {"reads": ["a", "b"], "defines": None},
                    ],
                    "successors": [],
                }
            ],
        }
        result = allocate(program)
        self.assertEqual(result["spilled"], [])
        self.assertEqual(result["affinity_gain"], 0)
        self.assertFalse(result["satisfied_affinities"]["a"]["satisfied"])
        self.assert_matches_full_brute_force(program)

    def test_spilled_partner_contributes_no_affinity(self):
        program = {
            "variables": [
                {"id": "a", "spill_cost": 1, "registers": ["r0", "r1"],
                 "prefer_same_as": "b", "affinity_weight": 9},
                {"id": "b", "spill_cost": 1, "registers": ["r0", "r1"]},
                {"id": "c", "spill_cost": 100, "registers": ["r0", "r1"]},
            ],
            "blocks": [
                {
                    "id": "b0",
                    "instructions": [
                        {"reads": [], "defines": "b"},
                        {"reads": [], "defines": "c"},
                        {"reads": ["b", "c"], "defines": "a"},
                        {"reads": ["a", "b", "c"], "defines": None},
                    ],
                    "successors": [],
                }
            ],
        }
        result = allocate(program)
        self.assertEqual(result["spilled"], ["a"])
        self.assertEqual(result["affinity_gain"], 0)
        self.assertFalse(result["satisfied_affinities"]["a"]["satisfied"])
        self.assert_matches_full_brute_force(program)

    def test_clobbers_pin_and_affinity_combined_matches_brute_force(self):
        # 单块内同时覆盖：c 跨越改写 r0 的调用、a 固定 r2、c 与 b 的亲和。
        # c 在「读 c 定义 b」指令后死亡，故 b、c 不干涉可同色；
        # 唯一干涉边是 (b,d)。最优方案 a=r2,b=r1,c=r1,d=r2，亲和收益 2。
        program = {
            "variables": [
                {"id": "a", "spill_cost": 100, "registers": ["r0", "r1", "r2"],
                 "pinned_register": "r2"},
                {"id": "b", "spill_cost": 100, "registers": ["r0", "r1", "r2"]},
                {"id": "c", "spill_cost": 2, "registers": ["r0", "r1"],
                 "prefer_same_as": "b", "affinity_weight": 2},
                {"id": "d", "spill_cost": 100, "registers": ["r1", "r2"]},
            ],
            "blocks": [
                {
                    "id": "entry",
                    "instructions": [
                        {"reads": [], "defines": "a"},
                        {"reads": ["a"], "defines": None},
                        {"reads": [], "defines": "c"},
                        {"reads": ["c"], "defines": None, "clobbers": ["r0"]},
                        {"reads": ["c"], "defines": "b"},
                        {"reads": [], "defines": "d"},
                        {"reads": ["b", "d"], "defines": None},
                    ],
                    "successors": [],
                }
            ],
        }
        result = allocate(program)
        self.assertEqual(result["clobber_restrictions"], {"c": ["r0"]})
        self.assertEqual(result["allocation"], {"a": "r2", "b": "r1", "c": "r1", "d": "r2"})
        self.assertEqual(result["affinity_gain"], 2)
        self.assertTrue(result["satisfied_affinities"]["c"]["satisfied"])
        self.assert_matches_full_brute_force(program)


if __name__ == "__main__":
    unittest.main()
