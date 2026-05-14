"""
元件选型工具 — 封装统一混合搜索引擎为 Agent @function_tool

基于 knowledge_base/unified/hybrid_store.db (SQLite FTS5) + FAISS 向量索引
提供三路混合搜索 (Keyword + BM25 + Embedding):
  - batch_search_and_select: ★ 批量搜索+详情+子模型, 一次调用完成所有元件选型
  - search_component: 按关键词检索候选元件 (支持中英文)
  - get_component_detail: 查询元件完整信息 (含 description/ports/params)
  - recommend_submodel: 获取推荐子模型 ID
"""
from __future__ import annotations

import json
import sys
import os
import functools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from agents import function_tool
from amesim_builder.unified_search import get_searcher, can_connect
from amesim_builder.selector import recommend_submodel as _recommend_submodel


# ============================================================
# 批量选型工具 — 一次调用完成所有元件的 search+detail+submodel
# ============================================================

# RRF分数阈值: 低于此值的结果视为噪音，不在候选中呈现
_RRF_MIN_THRESHOLD = 0.005
# 每个需求返回的候选数
_CANDIDATES_PER_REQ = 3
# 端口兼容性评分权重 (叠加到 rrf_score 上)
_PORT_COMPAT_BOOST_HIGH = 0.015   # 有兼容端口对 + IO互补
_PORT_COMPAT_BOOST_MID = 0.008    # 同 port_tag 但 IO 同向
_PORT_COMPAT_BOOST_LOW = 0.003   # 仅同 port_tag


def _build_candidate(hit: dict, doc: dict | None, req_idx: int,
                     description: str, tried_query: str) -> dict:
    """将搜索命中 + 文档详情组装为单个候选元件。

    提取自原 _batch_search_and_select_impl 的组装逻辑，
    现在对每个 top-k 命中独立调用，生成统一格式的候选 dict。
    """
    lib = hit.get("library", "")
    icon_key = f"{lib}:{hit['icon_name']}"
    recommended = _recommend_submodel(icon_key)
    if not recommended and doc:
        recommended = doc.get("submodel_id", "")

    ports_out = []
    if doc:
        for p in doc.get("ports", [])[:6]:
            ports_out.append({
                "index": p.get("index"),
                "tag": p.get("tag"),
                "variables": [
                    {"norm": v.get("norm"), "io": v.get("io"),
                     "title": v.get("title"), "units": v.get("units")}
                    for v in p.get("variables", [])
                ],
            })

    domain_info = _get_domain_info(lib)

    return {
        "requirement_index": req_idx,
        "functional_description": description,
        "search_query": tried_query,
        "icon_key": icon_key,
        "icon_name": hit.get("icon_name", ""),
        "library": lib,
        "rrf_score": hit.get("rrf_score", 0),
        "recommended_submodel": recommended,
        "submodel_ids": [recommended] if recommended else [],
        "label": (doc.get("label", "") or "")[:100] if doc else "",
        "description": (doc.get("text", "") or "")[:1000] if doc else (hit.get("text", "") or "")[:1000],
        "ports": ports_out,
        "port_count": len(ports_out),
        "params": doc.get("params", [])[:10] if doc else [],
        "param_count": len(doc.get("params", [])) if doc else 0,
        "status": "found",
        "alias": _suggest_alias(description, hit.get("icon_name", "")),
        "domain": domain_info["name_cn"],
        "submodel_path": f"$AME/{lib}/submodels",
        "rotations": 0,
        "flip": False,
        "position": [0, 0],
    }


def _batch_search_and_select_impl(
    requirements: list[dict],
    topology_description: str = "",
) -> dict:
    """批量元件选型的纯函数实现 (可被代码路径直接调用)。

    改进: 每个需求返回 top-N 候选而非单一结果,
    让 LLM 审核阶段从候选中选择最佳匹配。
    支持拓扑感知: 提供 topology_description 时自动进行端口兼容性重排序。

    Args:
        requirements: 元件需求列表 [{requirement_index, keywords, library, functional_description}, ...]
        topology_description: 拓扑描述文本 (可选, 如 "A -> B -> C"),
                             提供后启用端口兼容性重排序

    Returns:
        包含 selected, failures, has_failures 的完整结果 dict
        selected 中每项新增 "candidates" 字段 (top-N 候选列表)
    """
    searcher = get_searcher()
    results = []
    failures = []

    for req in requirements:
        req_idx = req.get("requirement_index", len(results) + len(failures) + 1)
        keywords = req.get("keywords", [])
        library = req.get("library", "")
        description = req.get("functional_description", "")

        # 选择最佳搜索词: 优先英文 keywords, 其次是 functional_description
        search_terms = []
        for kw in keywords:
            if any(c.isascii() and c.isalpha() for c in kw):
                search_terms.append(kw)
        if not search_terms:
            search_terms = keywords[:]  # 纯中文 keyword 也行
        if not search_terms:
            search_terms = [description]

        # ── 搜索阶段: 收集 top-k 候选 (替代原 hits[0] 盲目取首条) ──
        all_raw_hits: list[dict] = []
        tried_queries: list[str] = []

        for term in search_terms[:3]:
            lib = library if library else None
            raw_hits = searcher.search(term, top_k=_CANDIDATES_PER_REQ * 2, library=lib)
            tried_queries.append(term)
            if raw_hits:
                # 过滤低分噪音
                for rh in raw_hits:
                    if rh.get("rrf_score", 0) >= _RRF_MIN_THRESHOLD:
                        all_raw_hits.append(rh)
                if all_raw_hits:
                    break

        if not all_raw_hits:
            # 不限定库再试一次
            for term in search_terms[:2]:
                raw_hits = searcher.search(term, top_k=_CANDIDATES_PER_REQ * 2, library=None)
                if raw_hits:
                    for rh in raw_hits:
                        if rh.get("rrf_score", 0) >= _RRF_MIN_THRESHOLD:
                            all_raw_hits.append(rh)
                    if all_raw_hits:
                        break

        if not all_raw_hits:
            failures.append({
                "requirement_index": req_idx,
                "functional_description": description,
                "keywords_used": tried_queries,
                "reason": f"在库 '{library}' 和全库中均未找到匹配元件",
            })
            continue

        # ── 去重: 同一个 icon_key 只保留最高分 ──
        seen_iconkeys: set[str] = set()
        unique_hits: list[dict] = []
        for rh in sorted(all_raw_hits, key=lambda h: -h.get("rrf_score", 0)):
            ik = f"{rh['library']}:{rh['icon_name']}"
            if ik not in seen_iconkeys:
                seen_iconkeys.add(ik)
                unique_hits.append(rh)
                if len(unique_hits) >= _CANDIDATES_PER_REQ:
                    break

        # ── 组装候选 (每个候选含完整端口/参数数据) ──
        candidates = []
        for hit in unique_hits:
            icon_key = f"{hit['library']}:{hit['icon_name']}"
            doc = searcher.get_document_by_iconkey(icon_key)
            candidates.append(_build_candidate(
                hit, doc, req_idx, description,
                tried_queries[0] if tried_queries else "",
            ))

        # 首候选作为默认 (LLM 可覆盖选择其他候选)
        primary = candidates[0]
        primary["candidates"] = candidates
        primary["candidate_count"] = len(candidates)
        results.append(primary)

    # -- 拓扑感知: 端口兼容性重排序 --
    if topology_description:
        searcher2 = get_searcher()
        _apply_port_compat_reranking(
            results, requirements, topology_description, searcher2,
        )
        # 重排序后，更新主结果 (primary) 为 rrf_score 最高的候选
        for r in results:
            cands = r.get("candidates", [r])
            if len(cands) > 1:
                cands.sort(key=lambda c: -c.get("rrf_score", 0))
                best = cands[0]
                for k in ("icon_key", "icon_name", "library", "rrf_score",
                          "recommended_submodel", "submodel_ids", "label",
                          "description", "ports", "port_count", "params",
                          "param_count", "alias"):
                    if k in best:
                        r[k] = best[k]
                r["port_compat_boost"] = best.get("port_compat_boost", 0)

    return {
        "selected": results,
        "failures": failures,
        "has_failures": len(failures) > 0,
        "total_requirements": len(requirements),
        "found_count": len(results),
        "failed_count": len(failures),
        "message": f"批量搜索完成: {len(results)}/{len(requirements)} 成功, 每项含 {_CANDIDATES_PER_REQ} 个候选",
    }


@function_tool
def batch_search_and_select(requirements_json: str) -> str:
    """★ 批量元件选型: 一次调用完成所有元件的搜索、详情查询和子模型推荐。
    这是 Selector 阶段的首选工具 — 避免 LLM 逐个调用三个工具导致的 turn 数爆炸。

    输入 requirements_json 格式:
    [
      {
        "requirement_index": 1,
        "keywords": ["hydraulic pump", "pump"],
        "library": "libhyd",
        "functional_description": "液压泵"
      },
      ...
    ]

    返回完整的选型结果, 每个元件包含:
      - 搜索命中数 (hits)
      - top-N 候选列表 (candidates), LLM 从中选择最佳匹配
      - 每个候选含完整信息 (icon_key, icon_name, library, ports, params, description)
      - 推荐子模型 ID
      - 位置坐标和别名建议 (LLM 需根据拓扑关系调整)

    Args:
        requirements_json: 元件需求列表的 JSON 字符串

    Returns:
        JSON 格式的批量选型结果
    """
    try:
        requirements = json.loads(requirements_json)
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": f"requirements_json 解析失败: {e}",
        }, ensure_ascii=False, indent=2)

    if not isinstance(requirements, list):
        return json.dumps({
            "error": "requirements_json 必须是列表格式",
        }, ensure_ascii=False, indent=2)

    result = _batch_search_and_select_impl(requirements)
    return json.dumps(result, ensure_ascii=False, indent=2)


def _suggest_alias(description: str, icon_name: str) -> str:
    """根据描述和图标名生成建议别名"""
    # 简单规则: 取 icon_name 的首个有意义的词作为别名
    parts = icon_name.replace("_", " ").split()
    if parts:
        return parts[0].capitalize()
    return description[:20].replace(" ", "_")


# ============================================================
# 拓扑感知选型 — 拓扑解析 + 端口兼容性评分
# ============================================================

# 拓扑解析: 分割用正则 (箭头 / 分隔符)
_TOPO_ARROW_RE = __import__('re').compile(r'(?:→|->|--|—)')
_TOPO_SEP_RE = __import__('re').compile(r'[;；]')


def _parse_topology_pairs_for_selector(
    topology_description: str,
) -> list[tuple[str, str]]:
    """从拓扑描述文本中解析连接对，返回 [(name_a, name_b), ...]。

    与 bridge_core.parse_topology_pairs() 逻辑一致，但不依赖已知别名列表。
    支持格式:
      - "A → B → C"         → [("A","B"), ("B","C")]
      - "A → B; C → D"      → [("A","B"), ("C","D")]
      - "A→B→C; D→E"        → [("A","B"), ("B","C"), ("D","E")]
    """
    pairs: list[tuple[str, str]] = []
    # Step 1: 按分号切分为独立链段
    segments = _TOPO_SEP_RE.split(topology_description)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # Step 2: 按箭头切分为名称列表
        names = [n.strip() for n in _TOPO_ARROW_RE.split(seg) if n.strip()]
        # Step 3: 生成相邻对 (支持链式 "A→B→C")
        for i in range(len(names) - 1):
            pair = (names[i], names[i + 1])
            if pair not in pairs:
                pairs.append(pair)
    return pairs


def _map_topology_names_to_requirements(
    pairs: list[tuple[str, str]],
    requirements: list[dict],
) -> dict[int, list[int]]:
    """将拓扑连接对映射为 requirement_index 邻接表。

    通过匹配拓扑名称与需求的 keywords/functional_description，
    建立 {req_idx: [neighbor_req_indices]} 邻接关系。

    Args:
        pairs: [("Tank", "Pipe"), ("Pipe", "Valve"), ...]
        requirements: [{"requirement_index": 1, "keywords": [...],
                        "functional_description": "..."}, ...]

    Returns:
        {req_idx: [neighbor_req_idx, ...]}
    """
    # 为每个 requirement 建立一个"匹配词集合"用于模糊匹配
    req_tokens: list[set[str]] = []
    for req in requirements:
        tokens: set[str] = set()
        for kw in req.get("keywords", []):
            tokens.update(kw.lower().split())
            tokens.add(kw.lower())
        desc = req.get("functional_description", "")
        for word in desc.lower().replace("、", " ").replace("，", " ").split():
            stripped = word.strip("，。、 ")
            if len(stripped) >= 2:
                tokens.add(stripped)
        req_tokens.append(tokens)

    # 为每个拓扑名称找到最佳匹配的 requirement
    def _match_toponame_to_req(name: str) -> int | None:
        name_lower = name.lower()
        best_idx: int | None = None
        best_score = 0
        for i, tokens in enumerate(req_tokens):
            # 精确匹配
            if name_lower in tokens:
                return requirements[i]["requirement_index"]
            # 模糊匹配：计算 token 重叠数
            overlap = sum(
                1 for t in tokens
                if t in name_lower or name_lower in t
            )
            if overlap > best_score:
                best_score = overlap
                best_idx = requirements[i]["requirement_index"]
        return best_idx

    adj: dict[int, list[int]] = {}
    for a_name, b_name in pairs:
        a_idx = _match_toponame_to_req(a_name)
        b_idx = _match_toponame_to_req(b_name)
        if a_idx is not None and b_idx is not None and a_idx != b_idx:
            adj.setdefault(a_idx, []).append(b_idx)
            adj.setdefault(b_idx, []).append(a_idx)

    # 去重
    for k in adj:
        adj[k] = list(set(adj[k]))
    return adj


def _get_ports_for_icon_key(icon_key: str, searcher, cache: dict) -> list[dict] | None:
    """获取 icon_key 对应的端口列表 (带缓存)。"""
    if icon_key in cache:
        return cache[icon_key]
    doc = searcher.get_document_by_iconkey(icon_key)
    if doc:
        ports = doc.get("ports", [])
        cache[icon_key] = ports
        return ports
    cache[icon_key] = None
    return None


def _score_port_compatibility(
    candidate_a: dict,
    candidate_b: dict,
    searcher,
    port_cache: dict,
) -> float:
    """计算两个候选元件之间的端口兼容性分数。

    对每对端口调用 can_connect() 判断是否能连接。

    Returns:
        兼容性 boost 分数 (0.0 ~ 0.015)
    """
    icon_key_a = candidate_a.get("icon_key", "")
    icon_key_b = candidate_b.get("icon_key", "")

    ports_a = _get_ports_for_icon_key(icon_key_a, searcher, port_cache)
    ports_b = _get_ports_for_icon_key(icon_key_b, searcher, port_cache)

    if not ports_a or not ports_b:
        return 0.0

    best_score = 0.0
    for pa in ports_a:
        pa_idx = pa.get("index")
        pa_tag = pa.get("tag", "")
        for pb in ports_b:
            pb_idx = pb.get("index")
            pb_tag = pb.get("tag", "")

            # port_tag 必须相同才有意义
            if not pa_tag or not pb_tag or pa_tag != pb_tag:
                continue

            ok, matched = can_connect(pa, pb)
            if ok:
                # 有兼容端口对 + IO互补 → 高分
                return _PORT_COMPAT_BOOST_HIGH

            # 同 tag 但 can_connect 失败 (IO 同向等) → 低分
            if pa_tag == pb_tag:
                best_score = max(best_score, _PORT_COMPAT_BOOST_LOW)

    return best_score


def _apply_port_compat_reranking(
    results: list[dict],
    requirements: list[dict],
    topology_description: str,
    searcher,
) -> None:
    """对候选列表进行端口兼容性重排序：在 rrf_score 上叠加 port_compat_boost。

    修改 results 中每个候选的 rrf_score (原地修改)。
    """
    if not topology_description or len(results) < 2:
        return

    # Step 1: 解析拓扑
    pairs = _parse_topology_pairs_for_selector(topology_description)
    if not pairs:
        return
    adj = _map_topology_names_to_requirements(pairs, requirements)
    if not adj:
        return

    # Step 2: 按 requirement_index 索引 candidates
    candidates_by_req: dict[int, list[dict]] = {}
    for r in results:
        req_idx = r["requirement_index"]
        for cand in r.get("candidates", [r]):
            candidates_by_req.setdefault(req_idx, []).append(cand)

    # Step 3: 对每对相邻元件，计算端口兼容加分
    port_cache: dict[str, list[dict] | None] = {}
    boosts: dict[str, float] = {}  # icon_key → total boost

    for req_a, neighbors in adj.items():
        cands_a = candidates_by_req.get(req_a, [])
        if not cands_a:
            continue
        for req_b in neighbors:
            cands_b = candidates_by_req.get(req_b, [])
            if not cands_b:
                continue
            for ca in cands_a:
                for cb in cands_b:
                    boost = _score_port_compatibility(
                        ca, cb, searcher, port_cache,
                    )
                    if boost > 0:
                        ik_a = ca.get("icon_key", "")
                        ik_b = cb.get("icon_key", "")
                        boosts[ik_a] = boosts.get(ik_a, 0.0) + boost
                        boosts[ik_b] = boosts.get(ik_b, 0.0) + boost

    if not boosts:
        return

    # Step 4: 应用到 rrf_score
    for r in results:
        for cand in r.get("candidates", [r]):
            ik = cand.get("icon_key", "")
            b = boosts.get(ik, 0.0)
            if b > 0:
                cand["rrf_score"] = cand.get("rrf_score", 0.0) + b
                cand["port_compat_boost"] = round(b, 6)


@functools.lru_cache(maxsize=1)
def _load_domain_map() -> dict:
    """加载库→域映射 (domain_map.json)"""
    from pathlib import Path
    p = (Path(__file__).resolve().parent.parent.parent.parent /
         "knowledge_base" / "json" / "index" / "domain_map.json")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_domain_info(library: str) -> dict:
    """根据库名获取 domain/domain_cn 等信息"""
    dm = _load_domain_map()
    info = dm.get(library, {})
    return {
        "domain": info.get("domain", ""),
        "name_cn": info.get("name_cn", ""),
        "name_en": info.get("name_en", ""),
    }


@function_tool
def search_component(query: str, library: str = "", top_k: int = 8,
                     english_query: str = "") -> str:
    """按关键词搜索 Amesim 元件，支持中文和英文关键词。
    使用三路混合搜索 (Keyword倒排 + BM25全文 + Embedding语义)。
    提供英文翻译时可显著提升中文查询的召回率。

    Args:
        query: 中文或英文关键词，如 "质量块"、"orifice"、"thermal capacity pipe"
        library: 限定库名，如 "libmec"、"libtpf"，留空搜全部库
        top_k: 返回候选数量，默认 8
        english_query: ★ 中文查询的专业英文翻译，如 "mass block"。
                       提供后与 query 并行搜索并 RRF 融合，大幅提升中文命中率。

    Returns:
        JSON 格式的候选元件列表，含 doc_id、label、description_preview、submodel_id、ports 等
    """
    lib = library if library else None
    searcher = get_searcher()
    results = searcher.search(query, top_k=top_k, library=lib,
                              english_query=english_query if english_query else None)

    if not results:
        return json.dumps({
            "query": query,
            "hits": 0,
            "message": f"未找到匹配 '{query}' 的元件，请尝试其他关键词",
        }, ensure_ascii=False, indent=2)

    simplified = []
    for r in results:
        ports_summary = []
        for p in r.get("ports", [])[:4]:
            ports_summary.append({
                "index": p.get("index"),
                "tag": p.get("tag"),
                "var_count": len(p.get("variables", [])),
            })

        simplified.append({
            "icon_key": f"{r['library']}:{r['icon_name']}",
            "doc_id": r.get("doc_id", ""),
            "icon_name": r["icon_name"],
            "library": r["library"],
            "submodel_id": r.get("submodel_id", ""),
            "label": (r.get("label", "") or "")[:80],
            "description_preview": (r.get("text", "") or "")[:200],
            "rrf_score": r.get("rrf_score", 0),
            "port_count": len(r.get("ports", [])),
            "ports_summary": ports_summary,
            "param_count": len(r.get("params", [])),
        })
    return json.dumps({
        "query": query,
        "hits": len(simplified),
        "search_type": "hybrid(keyword+bm25+embedding)",
        "results": simplified,
    }, ensure_ascii=False, indent=2)


@function_tool
def get_component_detail(icon_key: str) -> str:
    """获取指定元件的完整信息，包括描述文本、参数、端口变量细节等。

    Args:
        icon_key: 格式为 "library:icon_name"，如 "libmec:mass_friction_endstops"
                  从 search_component 的返回结果中获取

    Returns:
        JSON 格式的元件详细信息 (含 description, ports{variables}, params)
    """
    searcher = get_searcher()
    doc = searcher.get_document_by_iconkey(icon_key)

    if not doc:
        return json.dumps({
            "icon_key": icon_key,
            "found": False,
            "message": f"未在统一文档中找到元件: {icon_key}",
        }, ensure_ascii=False, indent=2)

    # 简化 ports 输出 (保留关键字段)
    ports_out = []
    for p in doc.get("ports", []):
        ports_out.append({
            "index": p.get("index"),
            "tag": p.get("tag"),
            "variables": [
                {"norm": v.get("norm"), "io": v.get("io"),
                 "title": v.get("title"), "units": v.get("units")}
                for v in p.get("variables", [])
            ],
        })

    return json.dumps({
        "icon_key": icon_key,
        "found": True,
        "doc_id": doc.get("doc_id"),
        "library": doc.get("library"),
        "icon_name": doc.get("icon_name"),
        "submodel_id": doc.get("submodel_id"),
        "label": doc.get("label"),
        "description": doc.get("text", "")[:3000],
        "ports": ports_out,
        "params": doc.get("params", []),
    }, ensure_ascii=False, indent=2)


@function_tool
def recommend_submodel(icon_key: str) -> str:
    """获取元件的推荐子模型 ID。

    Args:
        icon_key: 格式为 "library:icon_name"，如 "libmec:mass_friction_endstops"

    Returns:
        推荐子模型 ID，如 "MAS000"
    """
    result = _recommend_submodel(icon_key)
    if result:
        return json.dumps({
            "icon_key": icon_key,
            "recommended_submodel": result,
        }, ensure_ascii=False, indent=2)
    return json.dumps({
        "icon_key": icon_key,
        "recommended_submodel": None,
        "message": f"未找到推荐子模型: {icon_key}",
    }, ensure_ascii=False, indent=2)
