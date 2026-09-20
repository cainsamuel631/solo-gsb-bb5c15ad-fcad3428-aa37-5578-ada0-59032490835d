"""最短路径条数固定预算（1000 条）的回归测试。

分层菱形（每层两个分叉、逐层汇合再分叉）下，根到目标的最短路径条数
随层数指数增长（n 层 = 2**n 条），逐一构造路径列表会拖垮进程。修复后
BFS 建立最短前驱图，先按层做饱和计数：条数超过固定预算 1000 即饱和为
1001 并抛 PathCountLimit，不构造路径列表；/dependency-paths、
/license-policy、/lock-diff 任一查询超限均返回 409 status=limit
（含目标包、used=1001、limit=1000），多个目标按既有包名顺序稳定命中
首个超限项；未超限时仍返回全部最短路径且排序不变。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.paths import PATH_COUNT_LIMIT, PathCountLimit, shortest_paths

client = TestClient(app)

ENV = {
    "python_version": "3.11",
    "sys_platform": "linux",
    "platform_machine": "x86_64",
}

assert PATH_COUNT_LIMIT == 1000  # 预算固定为 1000，不随配置变化


def c(version, deps=None, **metadata):
    entry: dict = {"version": version, "dependencies": deps or []}
    if metadata:
        entry["metadata"] = metadata
    return entry


def d(name, spec="", marker=""):
    return {"name": name, "specifier": spec, "marker": marker}


def layered_graph(layers):
    """layers 层 2 路菱形的邻接表：app -> l1a/l1b -> ... -> target。

    每层两个节点，第 i 层的两个节点都连向第 i+1 层的两个节点，最后一层
    都连向 target；根到 target 的最短路径恰为 2**layers 条。
    """
    graph = {"app": ["l1a", "l1b"]}
    for i in range(1, layers + 1):
        for s in ("a", "b"):
            node = f"l{i}{s}"
            graph[node] = [f"l{i + 1}a", f"l{i + 1}b"] if i < layers else ["target"]
    graph["target"] = []
    return graph


def wide_graph(width):
    """宽度菱形的邻接表：app -> w0..w{width-1} -> target，恰 width 条最短路径。"""
    graph = {"app": [f"w{i}" for i in range(width)]}
    for i in range(width):
        graph[f"w{i}"] = ["target"]
    graph["target"] = []
    return graph


def layered_repo_lock(layers, sinks, *, sink_metadata=None):
    """分层菱形仓库快照与有效锁：app -> l1a/l1b -> ... -> 各 sink。

    sinks 为最后一层两个节点共同指向的包名列表；根到每个 sink 的最短路径
    均为 2**layers 条。所有包默认携带 MIT 许可证（allow 判定通过、不产生
    路径查询），sink 可用 sink_metadata 覆盖（如 deny 的许可证）。
    """
    sink_metadata = sink_metadata or {}
    repo = {"app": [c("1.0", [d("l1a"), d("l1b")], license="MIT")]}
    lock = {"app": "1.0"}
    for i in range(1, layers + 1):
        for s in ("a", "b"):
            node = f"l{i}{s}"
            deps = [d(f"l{i + 1}a"), d(f"l{i + 1}b")] if i < layers else [
                d(sink) for sink in sinks
            ]
            repo[node] = [c("1.0", deps, license="MIT")]
            lock[node] = "1.0"
    for sink in sinks:
        repo[sink] = [c("1.0", **sink_metadata.get(sink, {"license": "MIT"}))]
        lock[sink] = "1.0"
    return repo, lock


def query_paths(repo, lock, target, *, root="app", **overrides):
    body = {
        "repository": repo,
        "root": root,
        "target_environment": ENV,
        "current_lock": lock,
        "target": target,
    }
    body.update(overrides)
    return client.post("/dependency-paths", json=body)


def query_policy(repo, lock, *, allow=None, deny=None, **overrides):
    body = {
        "repository": repo,
        "root": "app",
        "target_environment": ENV,
        "current_lock": lock,
        "allow": ["MIT"] if allow is None else allow,
        "deny": ["GPL-3.0-only"] if deny is None else deny,
    }
    body.update(overrides)
    return client.post("/license-policy", json=body)


def query_diff(repo, before, after, **overrides):
    body = {
        "repository": repo,
        "root": "app",
        "target_environment": ENV,
        "before_lock": before,
        "after_lock": after,
    }
    body.update(overrides)
    return client.post("/lock-diff", json=body)


# --------------------------------------------------------------------- #
# 1. paths 层：按层饱和计数，超限抛 PathCountLimit，不构造路径列表
# --------------------------------------------------------------------- #
def test_layered_diamond_over_budget_raises_with_saturated_used():
    # 10 层 = 1024 条 > 1000：超限，used 饱和为 1001（而非真实条数 1024）。
    with pytest.raises(PathCountLimit) as excinfo:
        shortest_paths(layered_graph(10), "app", "target")
    exc = excinfo.value
    assert exc.target == "target"
    assert exc.used == PATH_COUNT_LIMIT + 1 == 1001
    assert exc.limit == PATH_COUNT_LIMIT == 1000


def test_deep_layered_diamond_saturates_without_explosion():
    """20 层 = 1,048,576 条：计数饱和在 1001，进程不被指数级路径拖垮。"""
    with pytest.raises(PathCountLimit) as excinfo:
        shortest_paths(layered_graph(20), "app", "target")
    assert excinfo.value.used == 1001
    assert excinfo.value.limit == 1000


def test_exactly_at_budget_returns_all_paths():
    """恰 1000 条（预算边界）：不超限，全部路径照常返回且字典序排序。"""
    paths = shortest_paths(wide_graph(1000), "app", "target")
    assert len(paths) == 1000
    assert paths == sorted(paths)
    assert paths[0] == ["app", "w0", "target"]
    assert paths[-1] == ["app", "w999", "target"]


def test_one_over_budget_raises():
    """1001 条（预算 +1）：超限，used 恰为 1001。"""
    with pytest.raises(PathCountLimit) as excinfo:
        shortest_paths(wide_graph(1001), "app", "target")
    assert excinfo.value.used == 1001
    assert excinfo.value.limit == 1000


def test_far_over_budget_still_reports_saturated_1001():
    """5000 条：饱和计数与真实条数无关，used 恒为 1001（确定性）。"""
    with pytest.raises(PathCountLimit) as excinfo:
        shortest_paths(wide_graph(5000), "app", "target")
    assert excinfo.value.used == 1001


def test_under_budget_layered_diamond_unchanged():
    """9 层 = 512 条（未超限）：全部最短路径返回，排序与既有行为一致。"""
    paths = shortest_paths(layered_graph(9), "app", "target")
    assert len(paths) == 512
    assert paths == sorted(paths)
    assert paths[0] == ["app"] + ["l1a"] + [f"l{i}a" for i in range(2, 10)] + ["target"]
    assert all(len(p) == 11 for p in paths)


def test_unreachable_and_root_target_unaffected_by_budget():
    graph = layered_graph(10)
    assert shortest_paths(graph, "app", "ghost") == []
    assert shortest_paths(graph, "app", "app") == [["app"]]


# --------------------------------------------------------------------- #
# 2. POST /dependency-paths：超限 409 status=limit
# --------------------------------------------------------------------- #
def test_dependency_paths_over_budget_409_limit_payload():
    repo, lock = layered_repo_lock(10, ["target"])
    r = query_paths(repo, lock, "target")
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["status"] == "limit"
    assert body["target"] == "target"
    assert body["used"] == 1001
    assert body["limit"] == 1000
    assert "message" in body
    # 超限时不构造路径列表：响应不含路径查询结果字段。
    assert "paths" not in body and "reachable" not in body


def test_dependency_paths_under_budget_still_returns_all_paths_sorted():
    repo, lock = layered_repo_lock(9, ["target"])
    r = query_paths(repo, lock, "target")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reachable"] is True
    assert len(body["paths"]) == 512
    assert body["paths"] == sorted(body["paths"])
    assert body["paths"][0] == (
        ["app"] + [f"l{i}a" for i in range(1, 10)] + ["target"]
    )


def test_dependency_paths_limit_is_deterministic_across_input_ordering():
    """同一超限查询：包键/锁键重排后 409 响应逐字节一致。"""
    repo, lock = layered_repo_lock(10, ["target"])
    body1 = query_paths(repo, lock, "target").json()
    body2 = query_paths(
        dict(reversed(list(repo.items()))),
        dict(reversed(list(lock.items()))),
        "target",
    ).json()
    assert body1 == body2
    assert body1["used"] == 1001 and body1["limit"] == 1000


def test_dependency_paths_invalid_lock_still_invalid_lock_not_limit():
    """锁无效优先于路径预算：仍返回 409 invalid_lock，不执行路径查询。"""
    repo, lock = layered_repo_lock(10, ["target"])
    del lock["target"]  # target 可达但未锁定 -> missing
    r = query_paths(repo, lock, "target")
    assert r.status_code == 409
    body = r.json()
    assert body["status"] == "invalid_lock"
    assert body["issues"][0]["type"] == "missing"
    assert body["issues"][0]["name"] == "target"


# --------------------------------------------------------------------- #
# 3. POST /license-policy：命中包路径超限 409，按包名顺序命中首个
# --------------------------------------------------------------------- #
def test_license_policy_over_budget_409_limit_payload():
    repo, lock = layered_repo_lock(
        10, ["denied"], sink_metadata={"denied": {"license": "GPL-3.0-only"}}
    )
    r = query_policy(repo, lock)
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["status"] == "limit"
    assert body["target"] == "denied"
    assert body["used"] == 1001
    assert body["limit"] == 1000
    assert "violations" not in body and "unknowns" not in body


def test_license_policy_first_exceeding_target_in_name_order():
    """多个命中包都超限时，按包名顺序稳定命中首个超限项。"""
    repo, lock = layered_repo_lock(
        10,
        ["denied-b", "denied-a"],
        sink_metadata={
            "denied-a": {"license": "GPL-3.0-only"},
            "denied-b": {"license": "GPL-3.0-only"},
        },
    )
    body = query_policy(repo, lock).json()
    assert body["status"] == "limit"
    assert body["target"] == "denied-a"  # denied-a < denied-b
    # 输入重排（声明顺序、锁键顺序）不改变命中的首个超限项。
    repo2, lock2 = layered_repo_lock(
        10,
        ["denied-a", "denied-b"],
        sink_metadata={
            "denied-a": {"license": "GPL-3.0-only"},
            "denied-b": {"license": "GPL-3.0-only"},
        },
    )
    body2 = query_policy(repo2, dict(reversed(list(lock2.items())))).json()
    assert body2 == body


def test_license_policy_unknown_license_over_budget_also_limit():
    """未提供许可证（unknown）的命中包同样受路径预算约束。"""
    repo, lock = layered_repo_lock(10, ["mystery"], sink_metadata={"mystery": {}})
    r = query_policy(repo, lock)
    assert r.status_code == 409
    body = r.json()
    assert body["status"] == "limit"
    assert body["target"] == "mystery"


def test_license_policy_under_budget_returns_all_paths():
    """未超限：命中包仍附全部最短路径，判定与排序不变。"""
    repo, lock = layered_repo_lock(
        9, ["denied"], sink_metadata={"denied": {"license": "GPL-3.0-only"}}
    )
    r = query_policy(repo, lock)
    assert r.status_code == 200, r.text
    body = r.json()
    assert [e["name"] for e in body["violations"]] == ["denied"]
    paths = body["violations"][0]["paths"]
    assert len(paths) == 512
    assert paths == sorted(paths)


# --------------------------------------------------------------------- #
# 4. POST /lock-diff：变化项路径超限 409，按既有包名顺序命中首个
# --------------------------------------------------------------------- #
def _diff_repo_and_locks(sinks, *, layers=10):
    """菱形中间层在前后锁中**完全相同**（同名同版本），仅末层分叉换版本。

    ``layers`` 层菱形的中间节点 l1..l{layers-1} 在前后锁中同名同版本，
    两条路径数与层数的关系仍是 2**layers 条到每个 sink。末层
    l{layers}a/l{layers}b 有两个候选：

    * 1.0（before，app 1.0）：依赖 helper（removed 项，单条路径）；
    * 2.0（after，app 2.0）：依赖各 sink（added 项，各 2**layers 条）。

    因此除 sink/helper 外只有末层两节点是 upgraded 项（各 2**(layers-1)
    条/侧），added 列表中只有 sink，路径预算的命中点与旧夹具一致。
    """
    repo, after = layered_repo_lock(layers, sinks)
    last = f"l{layers}"
    repo[f"{last}a"] = [
        c("2.0", [d(sink) for sink in sinks], license="MIT"),
        c("1.0", [d("helper")], license="MIT"),
    ]
    repo[f"{last}b"] = [
        c("2.0", [d(sink) for sink in sinks], license="MIT"),
        c("1.0", [d("helper")], license="MIT"),
    ]
    repo["app"] = [
        c("2.0", [d("l1a"), d("l1b")], license="MIT"),
        c("1.0", [d("l1a"), d("l1b")], license="MIT"),
    ]
    repo["helper"] = [c("1.0", license="MIT")]
    before = {"app": "1.0", "helper": "1.0"}
    for i in range(1, layers):  # 末层之前各层同名同版本
        for s in ("a", "b"):
            before[f"l{i}{s}"] = "1.0"
    before[f"{last}a"] = "1.0"
    before[f"{last}b"] = "1.0"
    after["app"] = "2.0"
    after[f"{last}a"] = "2.0"
    after[f"{last}b"] = "2.0"
    return repo, before, after


def test_lock_diff_added_package_over_budget_409():
    repo, before, after = _diff_repo_and_locks(["newpkg"])
    r = query_diff(repo, before, after)
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["status"] == "limit"
    assert body["target"] == "newpkg"
    assert body["used"] == 1001
    assert body["limit"] == 1000
    assert "added" not in body and "removed" not in body


def test_lock_diff_first_exceeding_target_in_name_order():
    """多个变化项都超限时，按既有包名顺序稳定命中首个超限项。"""
    repo, before, after = _diff_repo_and_locks(["newpkg-b", "newpkg-a"])
    body = query_diff(repo, before, after).json()
    assert body["status"] == "limit"
    assert body["target"] == "newpkg-a"  # newpkg-a < newpkg-b
    # 输入重排（快照键、锁键顺序）不改变命中的首个超限项与响应。
    body2 = query_diff(
        dict(reversed(list(repo.items()))),
        dict(reversed(list(before.items()))),
        dict(reversed(list(after.items()))),
    ).json()
    assert body2 == body


def test_lock_diff_under_budget_returns_all_paths():
    """未超限：变化项仍附前后图中的全部最短路径，分类与排序不变。"""
    # 6 层：sink 64 条、末层两节点各 32 条/侧、根包 app 升级各 1 条，
    # 全部目标合计 64 + 1(helper) + 1 + 1 + 64 + 64 = 195 < 1000。
    repo, before, after = _diff_repo_and_locks(["newpkg"], layers=6)
    r = query_diff(repo, before, after)
    assert r.status_code == 200, r.text
    body = r.json()
    added = {item["name"]: item for item in body["added"]}
    assert len(added["newpkg"]["after_paths"]) == 64
    assert added["newpkg"]["after_paths"] == sorted(added["newpkg"]["after_paths"])
    assert added["newpkg"]["before_paths"] == []
    assert [item["name"] for item in body["removed"]] == ["helper"]
    assert [item["name"] for item in body["upgraded"]] == ["app", "l6a", "l6b"]


def test_lock_diff_invalid_lock_still_invalid_lock_not_limit():
    """任一份锁无效优先于路径预算：仍返回 409 invalid_lock。"""
    repo, before, after = _diff_repo_and_locks(["newpkg"])
    del after["newpkg"]  # newpkg 可达但未锁定 -> missing
    r = query_diff(repo, before, after)
    assert r.status_code == 409
    body = r.json()
    assert body["status"] == "invalid_lock"
    assert body["after"]["valid"] is False
    assert body["after"]["issues"][0]["type"] == "missing"


# --------------------------------------------------------------------- #
# 5. 请求级共享预算：一次请求的所有目标共享 1000 条额度
# --------------------------------------------------------------------- #
def policy_body(repo, lock, *, deny):
    """只拒绝 deny 中标识的 /license-policy 响应（其余包 MIT 通过）。"""
    return query_policy(repo, lock, allow=["MIT"], deny=deny)


def _multi_sink_layered_repo_lock(depth_a, depth_b):
    """根下挂两棵互不相交的分层菱形，各通向独立 sink。

    子树 A：app -> a1a/a1b -> ... -> a{depth_a}* -> sink-a，sink-a 恰有
    2**depth_a 条最短路径；子树 B（节点前缀 b、sink-b）同构、深度
    depth_b。两棵子树不共享任何中间节点，故两个 sink 的条数互不影响。
    """
    repo = {"app": [c("1.0", [d("a1a"), d("a1b"), d("b1a"), d("b1b")],
                    license="MIT")]}
    lock = {"app": "1.0"}

    def build(prefix, depth, sink):
        for i in range(1, depth + 1):
            for s in ("a", "b"):
                node = f"{prefix}{i}{s}"
                if i < depth:
                    deps = [d(f"{prefix}{i + 1}a"), d(f"{prefix}{i + 1}b")]
                else:
                    deps = [d(sink)]
                repo[node] = [c("1.0", deps, license="MIT")]
                lock[node] = "1.0"
        repo[sink] = [c("1.0", license="MIT")]
        lock[sink] = "1.0"

    build("a", depth_a, "sink-a")
    build("b", depth_b, "sink-b")
    return repo, lock


def test_license_policy_two_targets_each_under_but_sum_over_limit():
    """两个命中包各 512 条（均未超单目标 1000），合计 1024：共享额度 409。

    修复前预算按目标重置：两个目标都会返回，响应 200；修复后共享 1000
    条额度，按包名顺序 sink-a（512）先计入，再加入 sink-b（512）会超限，
    立即 409，target 指向 sink-b，used 恒为 1001，且不含任何业务结果。
    """
    repo, lock = _multi_sink_layered_repo_lock(9, 9)
    for sink in ("sink-a", "sink-b"):
        repo[sink] = [c("1.0", license="GPL-3.0-only")]
    r = policy_body(repo, lock, deny=["GPL-3.0-only"])
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["status"] == "limit"
    assert body["target"] == "sink-b"  # 512 + 512 > 1000，sink-a < sink-b
    assert body["used"] == 1001
    assert body["limit"] == 1000
    assert "violations" not in body and "unknowns" not in body


def test_license_policy_shared_budget_exactly_1000_still_succeeds():
    """三个命中包合计恰 1000（512+256+232）：恰好用尽额度仍成功。"""
    repo, lock = _multi_sink_layered_repo_lock(9, 8)  # 512 + 256 = 768
    # 再加一个宽菱形子树：app -> c1..c232 -> sink-c，恰 232 条，凑满 1000。
    width = 1000 - 512 - 256
    assert width == 232
    app_deps = repo["app"][0]["dependencies"] + [d(f"c{i}") for i in range(width)]
    repo["app"] = [c("1.0", app_deps, license="MIT")]
    for i in range(width):
        node = f"c{i}"
        repo[node] = [c("1.0", [d("sink-c")], license="MIT")]
        lock[node] = "1.0"
    repo["sink-c"] = [c("1.0", license="GPL-3.0-only")]
    lock["sink-c"] = "1.0"
    for sink in ("sink-a", "sink-b"):
        repo[sink] = [c("1.0", license="GPL-3.0-only")]

    r = policy_body(repo, lock, deny=["GPL-3.0-only"])
    assert r.status_code == 200, r.text
    body = r.json()
    counts = {
        e["name"]: len(e["paths"]) for e in body["violations"]
    }
    assert counts == {"sink-a": 512, "sink-b": 256, "sink-c": 232}
    # 再加一条路径即超限：同结构把宽度加 1（合计 1001）应整体 409。
    node = f"c{width}"
    repo["app"] = [c("1.0", app_deps + [d(node)], license="MIT")]
    repo[node] = [c("1.0", [d("sink-c")], license="MIT")]
    lock[node] = "1.0"
    r2 = policy_body(repo, lock, deny=["GPL-3.0-only"])
    assert r2.status_code == 409
    over = r2.json()
    assert over["status"] == "limit"
    assert over["target"] == "sink-c"  # sink-a/sink-b 已计入 768，sink-c 233 -> 1001
    assert over["used"] == 1001 and over["limit"] == 1000


def test_license_policy_shared_limit_independent_of_input_ordering():
    """两个目标合计超限：快照键/锁键重排后 409 响应逐字节一致。"""
    repo, lock = _multi_sink_layered_repo_lock(9, 9)
    for sink in ("sink-a", "sink-b"):
        repo[sink] = [c("1.0", license="GPL-3.0-only")]
    body1 = policy_body(repo, lock, deny=["GPL-3.0-only"]).json()
    body2 = policy_body(
        dict(reversed(list(repo.items()))),
        dict(reversed(list(lock.items()))),
        deny=["GPL-3.0-only"],
    ).json()
    assert body1 == body2
    assert body1["target"] == "sink-b" and body1["used"] == 1001


def _wide_diff_repo_locks(n1, n2):
    """宽菱形 /lock-diff 夹具：added 与 removed 各为单目标，跨项共享额度。

    app 1.0（before）-> p1..p{n1} -> gone（removed 项，n1 条 before 路径）；
    app 2.0（after）-> q1..q{n2} -> new（added 项，n2 条 after 路径）。
    p*/q* 各自只在一侧锁中，因此也是 added/removed 变化项（各 1 条）。
    按包名顺序：added 先处理 new（n2 条，"new" < "q0"）再处理 q*
    （各 1 条）；removed 先处理 gone（n1 条）再处理 p*（各 1 条）。
    """
    repo = {
        "app": [
            c("2.0", [d(f"q{i}") for i in range(n2)], license="MIT"),
            c("1.0", [d(f"p{i}") for i in range(n1)], license="MIT"),
        ],
    }
    for i in range(n1):
        repo[f"p{i}"] = [c("1.0", [d("gone")], license="MIT")]
    for i in range(n2):
        repo[f"q{i}"] = [c("1.0", [d("new")], license="MIT")]
    repo["gone"] = [c("1.0", license="MIT")]
    repo["new"] = [c("1.0", license="MIT")]
    before = {"app": "1.0", **{f"p{i}": "1.0" for i in range(n1)},
              "gone": "1.0"}
    after = {"app": "2.0", **{f"q{i}": "1.0" for i in range(n2)},
             "new": "1.0"}
    return repo, before, after


def test_lock_diff_budget_accumulates_across_added_and_removed_targets():
    """added 目标 499 条、removed 目标 499 条（各自未超 1000）：合计 998。

    added 按包名先处理 new（499 条）再处理 q0..q498（各 1 条），累计
    998；随后 removed 先处理 gone（499 条），加入即超限（1497 > 1000），
    整体 409，target 指向 gone，used 恒为 1001，不含任何差异结果。
    """
    repo, before, after = _wide_diff_repo_locks(499, 499)
    r = query_diff(repo, before, after)
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["status"] == "limit"
    assert body["target"] == "gone"
    assert body["used"] == 1001
    assert body["limit"] == 1000
    assert not {"added", "removed", "upgraded", "downgraded"} & set(body)


def test_lock_diff_shared_budget_exactly_1000_across_targets_succeeds():
    """跨目标累计恰 1000：成功（恰好用尽共享额度，不返回 limit）。

    n1=1（gone 1 条 + p0 1 条），n2=498（new 498 条 + q0..q497 各 1
    条），另有根包 app 升降级的 2 条单节点路径最后计入。处理顺序
    added → removed → upgraded：added 按包名处理 new(498) 与 q*(498)
    -> 996；removed 处理 gone(1)、p0(1) -> 998；upgraded 处理 app 的
    before/after 各 1 条，计入后恰 1000，整体成功。
    """
    repo, before, after = _wide_diff_repo_locks(1, 498)
    r = query_diff(repo, before, after)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["added"]) == 499  # q0..q497 + new
    new_item = next(i for i in body["added"] if i["name"] == "new")
    assert len(new_item["after_paths"]) == 498
    assert [i["name"] for i in body["removed"]] == ["gone", "p0"]


def test_lock_diff_budget_accumulates_across_before_and_after_sides():
    """同一 upgraded 变化项的 before/after 两侧条数跨侧累计。

    中间菱形层（l1..l8）前后锁同名同版本；l9a/l9b 从 1.0（依赖
    shared 1.0）升到 2.0（依赖 shared 2.0），shared 同步升级，故
    added/removed 均为空，upgraded 项为 l9a、l9b、shared。按既有顺序
    逐项先查 before 侧再查 after 侧：

    * l9a：before 256 + after 256 = 512
    * l9b：再 +256+256 -> 1024：l9b 的 **after 侧**查询使累计超限
      （before 侧计入后恰 768，after 侧 256 加入即 1024），409 的 target
      仍指向变化项包名 l9b，used 恒为 1001。
    """
    repo, lock9 = layered_repo_lock(9, ["shared"])
    repo["shared"] = [
        c("2.0", [d("mid")], license="MIT"),
        c("1.0", [d("mid")], license="MIT"),
    ]
    repo["mid"] = [c("1.0", license="MIT")]
    before = {name: ver for name, ver in lock9.items()}
    after = dict(lock9)
    before["l9a"] = "1.0"
    before["l9b"] = "1.0"
    before["shared"] = "1.0"
    before["mid"] = "1.0"
    after["l9a"] = "2.0"
    after["l9b"] = "2.0"
    after["shared"] = "2.0"
    after["mid"] = "1.0"
    # 让 2.0 候选存在（layered_repo_lock 只放了 1.0）。
    repo["l9a"] = [
        c("2.0", [d("shared")], license="MIT"),
        c("1.0", [d("shared")], license="MIT"),
    ]
    repo["l9b"] = [
        c("2.0", [d("shared")], license="MIT"),
        c("1.0", [d("shared")], license="MIT"),
    ]
    r = query_diff(repo, before, after)
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["status"] == "limit"
    assert body["target"] == "l9b"
    assert body["used"] == 1001
    assert body["limit"] == 1000
    assert not {"added", "removed", "upgraded", "downgraded"} & set(body)


def test_lock_diff_shared_limit_independent_of_input_ordering():
    """跨 added/removed 累计超限：快照键与两侧锁键重排后 409 逐字节一致。"""
    repo, before, after = _wide_diff_repo_locks(499, 499)
    body1 = query_diff(repo, before, after).json()
    body2 = query_diff(
        dict(reversed(list(repo.items()))),
        dict(reversed(list(before.items()))),
        dict(reversed(list(after.items()))),
    ).json()
    assert body1 == body2
    assert body1["target"] == "gone" and body1["used"] == 1001
