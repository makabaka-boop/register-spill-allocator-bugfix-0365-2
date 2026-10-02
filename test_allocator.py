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


if __name__ == "__main__":
    unittest.main()
