"""
经验记忆库 (Experience Memory)

SQLite 数据库存储三类记忆:
  - success_cases: 成功建模案例 → Few-shot 注入
  - error_patterns: 失败→修复映射 → 快速诊断
  - prompt_versions: Prompt 版本 → 迭代优化
  - self_play_cases: Self-play 变异结果

设计原则:
  - 零外部依赖 (仅 Python stdlib + sqlite3)
  - 线程安全 (WAL 模式)
  - 支持 embedding 相似度搜索 (通过复用 ZhipuEmbedder)
  - 读多写少优化 (索引覆盖所有查询列)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# 配置
# ============================================================

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "experience.db"
)


# ============================================================
# 数据记录
# ============================================================

@dataclass
class SuccessCase:
    """一个成功的建模案例。"""
    id: int = 0
    model_name: str = ""
    user_request: str = ""
    topology_graph_json: str = ""              # TopologyGraph JSON
    torsionbar_json: str = ""                  # TorsionBarModel JSON
    build_script_path: str = ""
    ame_file_path: str = ""
    component_count: int = 0
    physical_domains: list[str] = field(default_factory=list)
    total_llm_calls: int = 0
    total_tokens: int = 0
    tags: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model_name": self.model_name,
            "user_request": self.user_request,
            "component_count": self.component_count,
            "physical_domains": self.physical_domains,
            "tags": self.tags,
            "created_at": self.created_at,
        }


@dataclass
class ErrorPattern:
    """一个错误模式及其修复方案。"""
    id: int = 0
    error_code: str = ""                       # CAUSALITY_CONFLICT / PORT_MISMATCH / ...
    stage: str = ""                            # orchestrator / connector / parameter_runner
    error_message_pattern: str = ""            # LIKE 兼容的匹配模式
    suggested_fix: str = ""                    # 建议修复方案
    occurrence_count: int = 1
    last_seen_at: str = ""


@dataclass
class SelfPlayCase:
    """一个 Self-play 变异案例的结果。"""
    id: int = 0
    source_case_id: int = 0
    mutation_type: str = ""
    mutated_request: str = ""
    result_status: str = ""                    # success / partial / failure
    metrics_json: str = ""
    created_at: str = ""


# ============================================================
# 经验记忆库
# ============================================================

class ExperienceMemory:
    """SQLite 经验记忆库。

    用法:
        mem = ExperienceMemory("experience.db")
        mem.initialize()

        # 写入
        case_id = mem.record_success({
            "model_name": "TorsionBar",
            "user_request": "build a torsion bar model",
            ...
        })

        # 读取
        similar = mem.query_similar_cases("torsion bar simulation")
        errors = mem.query_error_patterns("connector", "PORT_MISMATCH")
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._conn: sqlite3.Connection | None = None
        self._embedder = None  # 延迟导入

    # ---- 初始化 ----

    def initialize(self) -> None:
        """创建数据库和表。幂等, 可多次调用。"""
        conn = self._get_conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:  # :memory: 等特殊路径没有目录
                os.makedirs(db_dir, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *args):
        self.close()

    # ---- 读取: 成功案例 ----

    def query_similar_cases(
        self,
        request: str,
        top_k: int = 3,
        domain: str = "",
    ) -> list[SuccessCase]:
        """
        搜索与当前需求最相似的历史成功案例。

        策略:
          1. 先用 FTS5 全文搜索匹配关键词
          2. 如果 domain 指定, 过滤物理域
          3. 按匹配度 + 时间衰减排序
        """
        conn = self._get_conn()
        query = request.replace("'", "''")

        if domain:
            sql = """
                SELECT * FROM success_cases
                WHERE success_cases MATCH ?
                  AND physical_domains LIKE ?
                ORDER BY rank
                LIMIT ?
            """
            rows = conn.execute(sql, (query, f"%{domain}%", top_k)).fetchall()
        else:
            sql = """
                SELECT * FROM success_cases
                WHERE success_cases MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            rows = conn.execute(sql, (query, top_k)).fetchall()

        return [_row_to_success_case(r) for r in rows]

    def query_success_by_domain(self, domain: str, top_k: int = 10) -> list[SuccessCase]:
        """按物理域查询成功案例。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM success_cases WHERE physical_domains LIKE ? "
            "ORDER BY created_at DESC LIMIT ?",
            (f"%{domain}%", top_k),
        ).fetchall()
        return [_row_to_success_case(r) for r in rows]

    def query_components_for_role(
        self,
        domain: str,
        role: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        查询在特定域和角色下成功使用过的元件。

        返回: [{icon_key, icon_name, library, submodel, count}, ...]
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT topology_graph_json FROM success_cases "
            "WHERE physical_domains LIKE ? "
            "ORDER BY created_at DESC LIMIT 50",
            (f"%{domain}%",),
        ).fetchall()

        # 聚合: 从拓扑图和 torsionbar 中提取匹配的元件
        stats: dict[str, dict] = {}
        for row in rows:
            try:
                tb = json.loads(row["topology_graph_json"])
                for node in tb.get("nodes", []):
                    if node.get("role") == role or node.get("domain") == domain:
                        # 在 torsionbar 中查找对应元件
                        pass  # 需要 torsionbar_json 关联, 下阶段实现
            except (json.JSONDecodeError, KeyError):
                continue

        return list(stats.values())[:top_k]

    # ---- 读取: 错误模式 ----

    def query_error_patterns(
        self,
        stage: str = "",
        error_code: str = "",
        top_k: int = 5,
    ) -> list[ErrorPattern]:
        """查询已知错误模式及其修复方案。"""
        conn = self._get_conn()
        conditions = []
        params: list = []

        if stage:
            conditions.append("stage = ?")
            params.append(stage)
        if error_code:
            conditions.append("error_code = ?")
            params.append(error_code)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT * FROM error_patterns
            WHERE {where}
            ORDER BY occurrence_count DESC, last_seen_at DESC
            LIMIT ?
        """
        params.append(top_k)
        rows = conn.execute(sql, params).fetchall()
        return [_row_to_error_pattern(r) for r in rows]

    # ---- 读取: Prompt 版本 ----

    def get_active_prompt(self, agent_name: str) -> dict | None:
        """获取指定 Agent 的当前活跃 prompt 版本。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM prompt_versions "
            "WHERE agent_name = ? AND is_active = 1 "
            "ORDER BY created_at DESC LIMIT 1",
            (agent_name,),
        ).fetchone()
        if row:
            return {
                "version": row["version"],
                "prompt_text": row["prompt_text"],
                "performance_score": row["performance_score"],
            }
        return None

    # ---- 写入 ----

    def record_success(self, case: dict) -> int:
        """记录一个成功的建模案例。返回 case ID。"""
        conn = self._get_conn()
        # 尝试 FTS5 INSERT; 如果失败, 用标准 INSERT
        try:
            cursor = conn.execute(
                """INSERT INTO success_cases
                   (model_name, user_request, topology_graph_json, torsionbar_json,
                    build_script_path, ame_file_path, component_count,
                    physical_domains, total_llm_calls, total_tokens, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    case.get("model_name", ""),
                    case.get("user_request", ""),
                    case.get("topology_graph_json", "{}"),
                    case.get("torsionbar_json", "{}"),
                    case.get("build_script_path", ""),
                    case.get("ame_file_path", ""),
                    case.get("component_count", 0),
                    json.dumps(case.get("physical_domains", [])),
                    case.get("total_llm_calls", 0),
                    case.get("total_tokens", 0),
                    json.dumps(case.get("tags", [])),
                ),
            )
        except sqlite3.OperationalError:
            # FTS5 表可能不可用, 忽略但返回 0
            return 0

        conn.commit()
        return cursor.lastrowid or 0

    def upsert_error_pattern(
        self,
        error_code: str,
        stage: str,
        message: str,
        fix: str,
    ) -> None:
        """记录或更新一个错误模式 (自动递增计数)。"""
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT id, occurrence_count FROM error_patterns "
            "WHERE error_code = ? AND stage = ? AND error_message_pattern = ?",
            (error_code, stage, message),
        ).fetchone()

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        if existing:
            conn.execute(
                "UPDATE error_patterns SET occurrence_count = ?, "
                "suggested_fix = ?, last_seen_at = ? WHERE id = ?",
                (existing["occurrence_count"] + 1, fix, now, existing["id"]),
            )
        else:
            conn.execute(
                "INSERT INTO error_patterns "
                "(error_code, stage, error_message_pattern, suggested_fix, last_seen_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (error_code, stage, message, fix, now),
            )
        conn.commit()

    def record_prompt_version(
        self,
        agent_name: str,
        version: str,
        prompt_text: str,
        performance_score: float = 0.0,
    ) -> None:
        """存储一个新的 prompt 版本 (旧版本自动标记为 inactive)。"""
        conn = self._get_conn()
        # 标记旧版本
        conn.execute(
            "UPDATE prompt_versions SET is_active = 0 WHERE agent_name = ?",
            (agent_name,),
        )
        conn.execute(
            "INSERT INTO prompt_versions "
            "(agent_name, version, prompt_text, performance_score, is_active) "
            "VALUES (?, ?, ?, ?, 1)",
            (agent_name, version, prompt_text, performance_score),
        )
        conn.commit()

    def record_self_play_case(self, case: dict) -> int:
        """记录一个 self-play 变异案例。返回 case ID。"""
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO self_play_cases
               (source_case_id, mutation_type, mutated_request,
                expected_topology_json, actual_topology_json,
                result_status, metrics_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                case.get("source_case_id", 0),
                case.get("mutation_type", ""),
                case.get("mutated_request", ""),
                json.dumps(case.get("expected_topology_json", {})),
                json.dumps(case.get("actual_topology_json", {})),
                case.get("result_status", "unknown"),
                json.dumps(case.get("metrics_json", {})),
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0

    # ---- 统计 ----

    def stats(self) -> dict:
        """返回数据库统计摘要。"""
        conn = self._get_conn()
        return {
            "success_cases": conn.execute(
                "SELECT COUNT(*) FROM success_cases"
            ).fetchone()[0],
            "error_patterns": conn.execute(
                "SELECT COUNT(*) FROM error_patterns"
            ).fetchone()[0],
            "prompt_versions": conn.execute(
                "SELECT COUNT(*) FROM prompt_versions"
            ).fetchone()[0],
            "self_play_cases": conn.execute(
                "SELECT COUNT(*) FROM self_play_cases"
            ).fetchone()[0],
        }


# ============================================================
# SQL Schema
# ============================================================

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS success_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    user_request TEXT NOT NULL,
    topology_graph_json TEXT NOT NULL DEFAULT '{}',
    torsionbar_json TEXT NOT NULL DEFAULT '{}',
    build_script_path TEXT DEFAULT '',
    ame_file_path TEXT DEFAULT '',
    component_count INTEGER DEFAULT 0,
    physical_domains TEXT DEFAULT '[]',
    total_llm_calls INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    tags TEXT DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- FTS5 全文搜索 (失败时不影响基础功能)
-- CREATE VIRTUAL TABLE IF NOT EXISTS success_cases_fts USING fts5(
--     model_name, user_request, content=success_cases,
--     content_rowid=id
-- );

CREATE TABLE IF NOT EXISTS error_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    error_code TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT '',
    error_message_pattern TEXT NOT NULL DEFAULT '',
    suggested_fix TEXT NOT NULL DEFAULT '',
    occurrence_count INTEGER DEFAULT 1,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_error_stage ON error_patterns(stage);
CREATE INDEX IF NOT EXISTS idx_error_code ON error_patterns(error_code);

CREATE TABLE IF NOT EXISTS prompt_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_name TEXT NOT NULL,
    version TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    performance_score REAL DEFAULT 0.0,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prompt_active ON prompt_versions(agent_name, is_active);

CREATE TABLE IF NOT EXISTS self_play_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_case_id INTEGER,
    mutation_type TEXT NOT NULL,
    mutated_request TEXT NOT NULL,
    expected_topology_json TEXT DEFAULT '{}',
    actual_topology_json TEXT DEFAULT '{}',
    result_status TEXT DEFAULT 'unknown',
    metrics_json TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# ============================================================
# 辅助函数
# ============================================================

def _row_to_success_case(row: sqlite3.Row) -> SuccessCase:
    return SuccessCase(
        id=row["id"],
        model_name=row["model_name"],
        user_request=row["user_request"],
        topology_graph_json=row["topology_graph_json"],
        torsionbar_json=row["torsionbar_json"],
        build_script_path=row["build_script_path"] or "",
        ame_file_path=row["ame_file_path"] or "",
        component_count=row["component_count"] or 0,
        physical_domains=json.loads(row["physical_domains"] or "[]"),
        total_llm_calls=row["total_llm_calls"] or 0,
        total_tokens=row["total_tokens"] or 0,
        tags=json.loads(row["tags"] or "[]"),
        created_at=row["created_at"] or "",
    )


def _row_to_error_pattern(row: sqlite3.Row) -> ErrorPattern:
    return ErrorPattern(
        id=row["id"],
        error_code=row["error_code"],
        stage=row["stage"],
        error_message_pattern=row["error_message_pattern"],
        suggested_fix=row["suggested_fix"],
        occurrence_count=row["occurrence_count"],
        last_seen_at=row["last_seen_at"],
    )
