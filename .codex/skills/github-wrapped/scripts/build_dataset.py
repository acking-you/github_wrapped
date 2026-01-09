import argparse
import json
import math
import re
import datetime as dt
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


# These globals are set in main() from CLI arguments.
BASE = Path(".")
RAW = Path(".")
OUT_DIR = Path(".")
YEAR = 0
TZ = ZoneInfo("Asia/Shanghai")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_iso8601(ts: str) -> dt.datetime:
    if not ts:
        raise ValueError("empty timestamp")
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return dt.datetime.fromisoformat(ts)


def to_local(ts: str) -> dt.datetime:
    return parse_iso8601(ts).astimezone(TZ)


def safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return float(value) / float(max_value)


def compact_number(n: int) -> str:
    if n >= 100_000_000:
        return f"{n/100_000_000:.1f}亿"
    if n >= 10_000:
        return f"{n/10_000:.1f}万"
    return str(n)


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    emoji: str
    color: str
    description: str


CATEGORIES = [
    Category(
        key="systems",
        label="系统/性能",
        emoji="🧠",
        color="#45f7c7",
        description="Rust/C/C++/Zig、性能、异步、底层实现",
    ),
    Category(
        key="database",
        label="数据库/查询",
        emoji="🗄️",
        color="#ffcc66",
        description="ClickHouse/DataFusion/SQL/OLAP/存储/索引",
    ),
    Category(
        key="data",
        label="数据工程",
        emoji="🧩",
        color="#d29922",
        description="ETL/ELT、流式、数据管道、分析工程",
    ),
    Category(
        key="ai",
        label="AI/LLM",
        emoji="🧬",
        color="#ff4fd8",
        description="LLM、RAG、Agent、推理、OpenAI、MCP",
    ),
    Category(
        key="cli",
        label="命令行/终端",
        emoji="⌨️",
        color="#58a6ff",
        description="CLI/TUI、终端工具、脚本与自动化",
    ),
    Category(
        key="editor",
        label="编辑器/IDE",
        emoji="📝",
        color="#bc8cff",
        description="Neovim/Vim/VScode/Emacs、LSP、插件生态",
    ),
    Category(
        key="tooling",
        label="开发工具链",
        emoji="🛠️",
        color="#7aa7ff",
        description="构建/调试/格式化/测试、开发效率与工程化",
    ),
    Category(
        key="frontend",
        label="前端/客户端",
        emoji="🪟",
        color="#a855f7",
        description="Web/TypeScript、桌面/移动端、Tauri/WASM",
    ),
    Category(
        key="infra",
        label="工程/基础设施",
        emoji="🛰️",
        color="#22c55e",
        description="CI/CD、DevOps、分布式、可观测性、云",
    ),
    Category(
        key="observability",
        label="可观测性",
        emoji="📡",
        color="#45f7c7",
        description="Logging/Metrics/Tracing、OpenTelemetry、监控体系",
    ),
    Category(
        key="network",
        label="网络/协议",
        emoji="🕸️",
        color="#ffcc66",
        description="HTTP/DNS/TLS/Proxy、网关、协议栈",
    ),
    Category(
        key="security",
        label="安全/加密",
        emoji="🛡️",
        color="#f85149",
        description="Auth/OAuth/JWT、密码学与安全工具",
    ),
    Category(
        key="docs",
        label="文档/学习",
        emoji="📚",
        color="#d29922",
        description="教程/手册/awesome/知识整理（基于仓库描述/话题）",
    ),
    Category(
        key="other",
        label="未标注/其它",
        emoji="🗂️",
        color="#8b949e",
        description="缺少 topics/description 或无法可靠归类",
    ),
]


CATEGORY_BY_KEY = {c.key: c for c in CATEGORIES}


TOPIC_KEYWORDS = {
    "ai": {
        "ai",
        "llm",
        "llms",
        "openai",
        "rag",
        "agent",
        "agents",
        "mcp",
        "claude",
        "claude-code",
        "codex",
        "deepseek",
        "qwen",
        "llama",
        "chatbot",
        "generative-ai",
        "artificial-intelligence",
        "machine-learning",
        "vector-search",
        "semantic-search",
        "embeddings",
        "inference",
        "transformer",
    },
    "database": {
        "database",
        "db",
        "sql",
        "olap",
        "analytics",
        "storage",
        "index",
        "indexing",
        "arrow",
        "datafusion",
        "clickhouse",
        "duckdb",
        "postgres",
        "mysql",
        "sqlite",
        "vector-database",
        "vector-search",
    },
    "data": {
        "etl",
        "elt",
        "pipeline",
        "data-pipeline",
        "data-engineering",
        "streaming",
        "kafka",
        "flink",
        "spark",
        "dbt",
        "airflow",
        "lakehouse",
        "analytics-engineering",
    },
    "systems": {
        # Avoid classifying purely by language; prefer capability/intent words.
        "performance",
        "async",
        "wasm",
        "compiler",
        "runtime",
        "benchmark",
        "profiling",
        "systems-programming",
        "concurrency",
        "kernel",
        "operating-system",
        "osdev",
        "memory",
        "allocator",
    },
    "cli": {
        "cli",
        "terminal",
        "shell",
        "tui",
        "command-line",
        "tmux",
        "bash",
        "zsh",
        "fish",
        "powershell",
    },
    "editor": {
        "editor",
        "neovim",
        "vim",
        "vscode",
        "emacs",
        "lsp",
        "treesitter",
        "plugin",
        "plugins",
    },
    "tooling": {
        "devtools",
        "workflow",
        "automation",
        "linter",
        "formatter",
        "debugger",
        "testing",
        "benchmark",
        "build",
        "build-tool",
        "tool",
        "tools",
    },
    "frontend": {
        "frontend",
        "web",
        "tauri",
        "electron",
        "react",
        "vue",
        "svelte",
        "ui",
        "ux",
        "wasm",
        "mobile",
        "ios",
        "android",
        "browser",
    },
    "infra": {
        "kubernetes",
        "docker",
        "devops",
        "ci",
        "cd",
        "github-actions",
        "distributed-systems",
        "cloud",
        "infra",
        "server",
        "iac",
        "terraform",
    },
    "observability": {
        "observability",
        "monitoring",
        "metrics",
        "logging",
        "tracing",
        "opentelemetry",
        "otel",
        "prometheus",
        "grafana",
        "jaeger",
        "loki",
        "sentry",
    },
    "network": {
        "network",
        "networking",
        "proxy",
        "gateway",
        "http",
        "https",
        "dns",
        "tls",
        "quic",
        "grpc",
        "nginx",
        "envoy",
    },
    "security": {
        "security",
        "auth",
        "authentication",
        "authorization",
        "oauth",
        "openid",
        "jwt",
        "crypto",
        "cryptography",
        "ssh",
        "vulnerability",
    },
    "docs": {
        "docs",
        "documentation",
        "guide",
        "tutorial",
        "handbook",
        "book",
        "awesome",
        "cheatsheet",
        "notes",
        "learning",
    },
}


LANG_FALLBACK_WEIGHTS = {}


TEXT_KEYWORDS = {
    "ai": {
        "llm",
        "rag",
        "openai",
        "anthropic",
        "claude",
        "chatgpt",
        "agent",
        "agents",
        "mcp",
        "inference",
        "transformer",
        "embeddings",
        "embedding",
        "vector",
        "prompt",
    },
    "database": {
        "sql",
        "database",
        "olap",
        "analytics",
        "query",
        "query engine",
        "datafusion",
        "clickhouse",
        "duckdb",
        "postgres",
        "postgresql",
        "mysql",
        "sqlite",
        "parquet",
        "arrow",
        "storage",
        "index",
        "lsm",
        "btree",
        "rocksdb",
    },
    "data": {
        "etl",
        "elt",
        "pipeline",
        "data pipeline",
        "data engineering",
        "streaming",
        "kafka",
        "flink",
        "airflow",
        "dbt",
        "lakehouse",
    },
    "systems": {
        "performance",
        "perf",
        "async",
        "runtime",
        "concurrency",
        "thread",
        "benchmark",
        "profiling",
        "compiler",
        "kernel",
        "operating system",
        "os",
        "allocator",
        "memory",
        "low-level",
        "systems programming",
    },
    "cli": {
        "cli",
        "terminal",
        "shell",
        "tmux",
        "tui",
        "prompt",
        "command",
        "ripgrep",
        "grep",
        "fzf",
        "curl",
        "wget",
    },
    "editor": {
        "neovim",
        "vim",
        "vscode",
        "zed",
        "emacs",
        "lsp",
        "treesitter",
        "plugin",
    },
    "tooling": {
        "devtools",
        "automation",
        "workflow",
        "linter",
        "formatter",
        "debugger",
        "build tool",
        "build system",
        "packaging",
        "package manager",
        "testing",
        "benchmark",
    },
    "frontend": {
        "frontend",
        "web",
        "ui",
        "ux",
        "react",
        "vue",
        "svelte",
        "tauri",
        "electron",
        "browser",
        "css",
        "html",
        "typescript",
        "javascript",
        "wasm",
        "mobile",
        "android",
        "ios",
        "flutter",
    },
    "infra": {
        "kubernetes",
        "docker",
        "devops",
        "ci",
        "cd",
        "github actions",
        "distributed",
        "cloud",
        "server",
        "terraform",
        "iac",
    },
    "observability": {
        "observability",
        "monitoring",
        "metrics",
        "logging",
        "tracing",
        "opentelemetry",
        "otel",
        "prometheus",
        "grafana",
        "jaeger",
        "loki",
        "sentry",
    },
    "network": {
        "proxy",
        "network",
        "dns",
        "http",
        "tls",
        "quic",
        "grpc",
        "gateway",
        "nginx",
        "envoy",
        "reverse proxy",
        "load balancer",
    },
    "security": {
        "security",
        "auth",
        "oauth",
        "openid",
        "jwt",
        "crypto",
        "cryptography",
        "encryption",
        "ssh",
        "vulnerability",
    },
    "docs": {
        "docs",
        "documentation",
        "guide",
        "tutorial",
        "handbook",
        "book",
        "awesome",
        "cheatsheet",
        "notes",
        "learning",
    },
}


def _contains_kw(text: str, kw: str) -> bool:
    kw = (kw or "").strip().lower()
    if not kw:
        return False
    t = text or ""
    if re.fullmatch(r"[a-z0-9]+", kw) and len(kw) <= 3:
        return re.search(rf"(?<![a-z0-9]){re.escape(kw)}(?![a-z0-9])", t) is not None
    return kw in t


def classify_repo(*, name: str, language: str | None, topics: list[str], description: str | None) -> dict:
    name_l = (name or "").lower()
    desc_l = (description or "").lower()
    topics_l = [str(t or "").strip().lower() for t in (topics or []) if str(t or "").strip()]
    text = f" {name_l} {desc_l} {' '.join(topics_l)} "

    score = {c.key: 0 for c in CATEGORIES if c.key != "other"}
    signals = defaultdict(list)

    # Topics: strong deterministic hint (but we avoid pure language tags above).
    for cat_key, kws in TOPIC_KEYWORDS.items():
        for t in topics_l:
            if t in kws:
                score[cat_key] += 4
                signals[cat_key].append(f"topic:{t}")

    # Description/name: real text signal (primary source for "索引卡").
    for cat_key, kws in TEXT_KEYWORDS.items():
        for kw in kws:
            if _contains_kw(text, kw):
                score[cat_key] += 2
                signals[cat_key].append(f"text:{kw}")

    if not any(score.values()):
        return {
            "primary": "other",
            "scores": {**score, "other": 0},
            "tags": [],
            "signals": ["none:text/topics"],
            "usedLanguageFallback": False,
        }

    best_key = max(score, key=lambda k: (score[k], k))
    sorted_keys = sorted(score.keys(), key=lambda k: score[k], reverse=True)
    tags = [k for k in sorted_keys if score[k] > 0][:3]
    top_signals = []
    for k in sorted_keys:
        if not score[k]:
            continue
        top_signals.extend(signals.get(k, [])[:8])
        if len(top_signals) >= 12:
            break
    return {
        "primary": best_key,
        "scores": {**score, "other": 0},
        "tags": tags,
        "signals": top_signals,
        "usedLanguageFallback": False,
    }


TOPIC_CANON_ALIASES = {
    "golang": "go",
    "cpp": "c++",
    "c-plus-plus": "c++",
    "cxx": "c++",
    "js": "javascript",
    "ts": "typescript",
    "llms": "llm",
}


def canon_topic(topic: str) -> str:
    t = (topic or "").strip().lower()
    if not t:
        return ""
    return TOPIC_CANON_ALIASES.get(t, t)


def stable_palette_color(key: str) -> str:
    # Deterministic color assignment without relying on salted Python hashes.
    palette = [
        "#58a6ff",
        "#bc8cff",
        "#3fb950",
        "#ffcc66",
        "#ff4fd8",
        "#45f7c7",
        "#f85149",
        "#d29922",
        "#7aa7ff",
        "#22c55e",
        "#a855f7",
        "#0ea5e9",
        "#fb7185",
        "#eab308",
    ]
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    idx = int(h[:8], 16) % len(palette)
    return palette[idx]


def streak_from_days(days_sorted: list[dict]) -> dict:
    max_streak = 0
    max_start = None
    max_end = None
    current = 0
    current_start = None
    prev_date = None

    for d in days_sorted:
        date = dt.date.fromisoformat(d["date"])
        if d["count"] > 0:
            if current == 0:
                current_start = date
            current += 1
        else:
            if current > max_streak:
                max_streak = current
                max_start = current_start
                max_end = prev_date
            current = 0
            current_start = None
        prev_date = date

    if current > max_streak:
        max_streak = current
        max_start = current_start
        max_end = prev_date

    return {
        "count": max_streak,
        "start": max_start.isoformat() if max_start else None,
        "end": max_end.isoformat() if max_end else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Build processed/dataset.json from saved gh API raw JSON.")
    ap.add_argument("--year", type=int, required=True, help="Report year, e.g. 2025")
    ap.add_argument("--timezone", default="Asia/Shanghai", help="IANA timezone, e.g. Asia/Shanghai")
    ap.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Base directory (default: data/github-wrapped-<year>)",
    )
    args = ap.parse_args()

    global BASE, RAW, OUT_DIR, YEAR, TZ
    YEAR = int(args.year)
    TZ = ZoneInfo(str(args.timezone))
    BASE = args.base if args.base else Path(f"data/github-wrapped-{YEAR}")
    RAW = BASE / "raw"
    OUT_DIR = BASE / "processed"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW.exists():
        raise SystemExit(f"raw dir not found: {RAW} (run scripts/collect_raw.sh first)")

    user = load(RAW / "user.json")
    login = user["login"]

    repos = load(RAW / "user_repos.json")
    own_repos = [r for r in repos if not r.get("fork")]
    first_repo = min(own_repos, key=lambda r: r.get("created_at") or "9999") if own_repos else None

    contrib = load(RAW / "contributions.json")
    cc = contrib["data"]["user"]["contributionsCollection"]
    calendar = cc["contributionCalendar"]
    weeks = calendar["weeks"]

    days = []
    for week in weeks:
        for day in week.get("contributionDays", []):
            days.append(
                {
                    "date": day["date"],
                    "count": safe_int(day["contributionCount"]),
                    "weekday": safe_int(day["weekday"]),
                }
            )

    days_sorted = sorted(days, key=lambda d: d["date"])
    active_days = sum(1 for d in days_sorted if d["count"] > 0)
    total_days = len(days_sorted)
    inactive_days = total_days - active_days

    busiest_count = max((d["count"] for d in days_sorted), default=0)
    busiest_day = next((d for d in days_sorted if d["count"] == busiest_count), {"date": None, "count": 0})
    top_days = sorted([d for d in days_sorted if d["count"] > 0], key=lambda d: d["count"], reverse=True)[:8]

    streak = streak_from_days(days_sorted)

    weekday_sums = [0] * 7
    for d in days_sorted:
        weekday_sums[d["weekday"]] += d["count"]
    weekend_sum = weekday_sums[0] + weekday_sums[6]
    total_contrib = sum(weekday_sums)
    weekend_ratio = (weekend_sum / total_contrib) if total_contrib else 0.0
    if total_contrib == 0:
        pattern = "暂无数据"
    elif weekend_ratio >= 0.40:
        pattern = "周末战士"
    elif weekend_ratio <= 0.20:
        pattern = "工作日重度"
    else:
        pattern = "均衡型"

    # Monthly activity
    month_contrib = defaultdict(int)
    month_active_days = defaultdict(int)
    for d in days_sorted:
        month = d["date"][:7]
        month_contrib[month] += d["count"]
        if d["count"] > 0:
            month_active_days[month] += 1
    month_contrib_sorted = sorted(month_contrib.items())
    most_active_month = max(month_contrib.items(), key=lambda kv: kv[1])[0] if month_contrib else None

    # Starred repos (all-time) + selected year slice
    star_pages = load(RAW / "starred_repos_pages.json")
    star_edges = []
    for page in star_pages:
        star_edges.extend(page["data"]["user"]["starredRepositories"]["edges"])

    stars = []
    for edge in star_edges:
        node = edge["node"]
        topics_raw = [t["topic"]["name"] for t in (node.get("repositoryTopics") or {}).get("nodes", [])]
        topics_canon = sorted({canon_topic(t) for t in topics_raw if canon_topic(t)})
        stars.append(
            {
                "starredAt": edge["starredAt"],
                "nameWithOwner": node["nameWithOwner"],
                "description": node.get("description"),
                "stargazerCount": safe_int(node.get("stargazerCount")),
                "forkCount": safe_int(node.get("forkCount")),
                "primaryLanguage": (node.get("primaryLanguage") or {}).get("name"),
                "topics": topics_raw,
                "topicsCanonical": topics_canon,
                "url": node.get("url") or f"https://github.com/{node['nameWithOwner']}",
            }
        )

    stars_year = [s for s in stars if s["starredAt"].startswith(f"{YEAR}-")]
    stars_before = [s for s in stars if s["starredAt"] < f"{YEAR}-01-01T00:00:00Z"]

    stars_by_year = defaultdict(int)
    for s in stars:
        year = s["starredAt"][:4]
        stars_by_year[year] += 1

    star_month_counts = Counter([s["starredAt"][:7] for s in stars_year])
    star_month_top_repo = {}
    for s in stars_year:
        key = s["starredAt"][:7]
        prev = star_month_top_repo.get(key)
        if not prev or s["stargazerCount"] > prev["stargazerCount"]:
            star_month_top_repo[key] = s

    star_month_repos = defaultdict(list)
    for s in stars_year:
        star_month_repos[s["starredAt"][:7]].append(s)
    for month_key in star_month_repos:
        star_month_repos[month_key].sort(key=lambda x: (x.get("stargazerCount", 0), x.get("starredAt") or ""), reverse=True)

    # Star events by month (for timeline charts)
    star_month_events = defaultdict(list)
    for s in stars_year:
        key = s["starredAt"][:7]
        star_month_events[key].append(
            {
                "starredAt": s["starredAt"],
                "nameWithOwner": s["nameWithOwner"],
                "stars": safe_int(s.get("stargazerCount")),
                "url": s.get("url"),
            }
        )
    for month_key in star_month_events:
        star_month_events[month_key].sort(key=lambda x: x.get("starredAt") or "")

    star_hour_local = [0] * 24
    for s in stars_year:
        hour = to_local(s["starredAt"]).hour
        star_hour_local[hour] += 1

    star_lang_year = Counter([s["primaryLanguage"] for s in stars_year if s.get("primaryLanguage")])
    star_topic_year = Counter()
    star_topic_canon_year = Counter()
    for s in stars_year:
        for t in s.get("topics", []):
            star_topic_year[t.lower()] += 1
        for t in s.get("topicsCanonical", []):
            star_topic_canon_year[t] += 1

    star_topic_before = Counter()
    star_topic_canon_before = Counter()
    for s in stars_before:
        for t in s.get("topics", []):
            star_topic_before[t.lower()] += 1
        for t in s.get("topicsCanonical", []):
            star_topic_canon_before[t] += 1

    new_topics = []
    rising_topics = []
    count_key = f"count{YEAR}"
    for topic, cur in star_topic_year.most_common():
        prev = star_topic_before.get(topic, 0)
        if prev == 0 and cur >= 3:
            new_topics.append({"topic": topic, "countBefore": 0, "countInYear": cur, count_key: cur})
        elif prev > 0 and cur >= max(5, prev * 2):
            rising_topics.append(
                {
                    "topic": topic,
                    "countBefore": prev,
                    "countInYear": cur,
                    "ratio": round(cur / prev, 2) if prev else None,
                    count_key: cur,
                }
            )

    new_topics = new_topics[:12]
    rising_topics = sorted(rising_topics, key=lambda x: (-(x.get("ratio") or 0), -x.get(count_key, 0)))[:12]

    new_topics_canon = []
    rising_topics_canon = []
    for topic, cur in star_topic_canon_year.most_common():
        prev = star_topic_canon_before.get(topic, 0)
        if prev == 0 and cur >= 3:
            new_topics_canon.append({"topic": topic, "countBefore": 0, "countInYear": cur, count_key: cur})
        elif prev > 0 and cur >= max(5, prev * 2):
            rising_topics_canon.append(
                {
                    "topic": topic,
                    "countBefore": prev,
                    "countInYear": cur,
                    "ratio": round(cur / prev, 2) if prev else None,
                    count_key: cur,
                }
            )
    new_topics_canon = new_topics_canon[:12]
    rising_topics_canon = sorted(rising_topics_canon, key=lambda x: (-(x.get("ratio") or 0), -x.get(count_key, 0)))[:12]

    lang_before = Counter([s["primaryLanguage"] for s in stars_before if s.get("primaryLanguage")])
    new_langs = []
    for lang, cur in star_lang_year.most_common():
        prev = lang_before.get(lang, 0)
        if prev == 0 and cur >= 2:
            new_langs.append({"language": lang, "countBefore": 0, "countInYear": cur, count_key: cur})
    new_langs = new_langs[:8]

    first_star_ever = min(stars, key=lambda s: s["starredAt"]) if stars else None
    first_star_year = min(stars_year, key=lambda s: s["starredAt"]) if stars_year else None
    latest_star_year = max(stars_year, key=lambda s: s["starredAt"]) if stars_year else None

    # Repo creation by year (own repos)
    repos_created_by_year = Counter([(r.get("created_at") or "")[:4] for r in own_repos if r.get("created_at")])

    # Contribution repos (top)
    commit_repos = cc.get("commitContributionsByRepository", [])
    pr_repos = cc.get("pullRequestContributionsByRepository", [])
    issue_repos = cc.get("issueContributionsByRepository", [])

    commit_repo_list = [
        {
            "nameWithOwner": item["repository"]["nameWithOwner"],
            "count": safe_int(item["contributions"]["totalCount"]),
        }
        for item in commit_repos
    ]
    pr_repo_list = [
        {
            "nameWithOwner": item["repository"]["nameWithOwner"],
            "count": safe_int(item["contributions"]["totalCount"]),
        }
        for item in pr_repos
    ]
    issue_repo_list = [
        {
            "nameWithOwner": item["repository"]["nameWithOwner"],
            "count": safe_int(item["contributions"]["totalCount"]),
        }
        for item in issue_repos
    ]

    # Focus project (most commit contributions in own repos)
    focus_project = None
    for item in sorted(commit_repo_list, key=lambda x: x["count"], reverse=True):
        if item["nameWithOwner"].lower().startswith(login.lower() + "/"):
            focus_project = item
            break

    # Merged PRs (year slice)
    pr_pages_path = RAW / f"prs_{YEAR}_pages.json"
    pr_pages = load(pr_pages_path) if pr_pages_path.exists() else []
    merged_prs = []
    for page in pr_pages:
        for node in page["data"]["search"]["nodes"]:
            if not node:
                continue
            merged_prs.append(
                {
                    "title": node["title"],
                    "url": node["url"],
                    "createdAt": node.get("createdAt"),
                    "mergedAt": node.get("mergedAt"),
                    "additions": safe_int(node.get("additions")),
                    "deletions": safe_int(node.get("deletions")),
                    "repo": node["repository"]["nameWithOwner"],
                    "repoStars": safe_int(node["repository"].get("stargazerCount")),
                    "repoLanguage": (node["repository"].get("primaryLanguage") or {}).get("name"),
                    "repoTopics": [
                        t["topic"]["name"]
                        for t in (node["repository"].get("repositoryTopics") or {}).get("nodes", [])
                    ],
                }
            )

    pr_total = len(merged_prs)
    pr_add_total = sum(pr["additions"] for pr in merged_prs)
    pr_del_total = sum(pr["deletions"] for pr in merged_prs)
    pr_lines_total = pr_add_total + pr_del_total

    latest_pr_created = None
    if merged_prs:
        latest_pr_created = max(
            merged_prs,
            key=lambda pr: pr["createdAt"] or "0000",
        )

    biggest_pr = None
    if merged_prs:
        biggest_pr = max(merged_prs, key=lambda pr: pr["additions"] + pr["deletions"])

    pr_by_repo = defaultdict(lambda: {"count": 0, "lines": 0, "add": 0, "del": 0})
    for pr in merged_prs:
        agg = pr_by_repo[pr["repo"]]
        agg["count"] += 1
        agg["add"] += pr["additions"]
        agg["del"] += pr["deletions"]
        agg["lines"] += pr["additions"] + pr["deletions"]

    oss_award_repo = None
    if pr_by_repo:
        oss_award_repo = max(pr_by_repo.items(), key=lambda kv: (kv[1]["count"], kv[1]["lines"]))[0]

    oss_award = None
    if oss_award_repo:
        oss_award = {"repo": oss_award_repo, **pr_by_repo[oss_award_repo]}

    # External contributed repos sample
    contrib_pages = load(RAW / "contributed_repos_pages.json")
    contributed_repos = []
    for page in contrib_pages:
        contributed_repos.extend(page["data"]["user"]["repositoriesContributedTo"]["nodes"])
    external_contrib = [
        r
        for r in contributed_repos
        if (r.get("owner") or {}).get("login", "").lower() != login.lower()
    ]
    external_contrib_sorted = sorted(external_contrib, key=lambda r: safe_int(r.get("stargazerCount")), reverse=True)

    # Holiday stars (Chinese-focused; fixed-date + known lunar-date mappings).
    # For years we don't have a verified mapping for lunar holidays, we only emit fixed-date ones.
    CN_HOLIDAYS_BY_YEAR = {
        2023: {
            "spring_festival": dt.date(2023, 1, 22),
            "qingming": dt.date(2023, 4, 5),
            "dragon_boat": dt.date(2023, 6, 22),
            "qixi": dt.date(2023, 8, 22),
            "mid_autumn": dt.date(2023, 9, 29),
        },
        2024: {
            "spring_festival": dt.date(2024, 2, 10),
            "qingming": dt.date(2024, 4, 4),
            "dragon_boat": dt.date(2024, 6, 10),
            "qixi": dt.date(2024, 8, 10),
            "mid_autumn": dt.date(2024, 9, 17),
        },
        2025: {
            "spring_festival": dt.date(2025, 1, 29),
            "qingming": dt.date(2025, 4, 4),
            "dragon_boat": dt.date(2025, 5, 31),
            "qixi": dt.date(2025, 8, 29),
            "mid_autumn": dt.date(2025, 10, 6),
        },
    }

    def holidays_for_year(year: int) -> list[tuple[str, str, dt.date]]:
        holidays: list[tuple[str, str, dt.date]] = [
            ("new_year", "元旦", dt.date(year, 1, 1)),
            ("labor_day", "劳动节", dt.date(year, 5, 1)),
            ("national_day", "国庆节", dt.date(year, 10, 1)),
            ("programmer_day", "程序员节", dt.date(year, 10, 24)),
            ("singles_day", "双十一", dt.date(year, 11, 11)),
            ("new_year_eve", "跨年夜(12/31)", dt.date(year, 12, 31)),
        ]

        extra = CN_HOLIDAYS_BY_YEAR.get(year, {})
        spring_festival = extra.get("spring_festival")
        if spring_festival:
            holidays.append(("chuxi", "除夕", spring_festival - dt.timedelta(days=1)))
            holidays.append(("spring_festival", "春节", spring_festival))
        if extra.get("qingming"):
            holidays.append(("qingming", "清明节", extra["qingming"]))
        if extra.get("dragon_boat"):
            holidays.append(("dragon_boat", "端午节", extra["dragon_boat"]))
        if extra.get("qixi"):
            holidays.append(("qixi", "七夕", extra["qixi"]))
        if extra.get("mid_autumn"):
            holidays.append(("mid_autumn", "中秋节", extra["mid_autumn"]))

        holidays.sort(key=lambda x: x[2].isoformat())
        return holidays

    holidays = holidays_for_year(YEAR)

    holiday_cards = []
    for key, label, date in holidays:
        repos = [s for s in stars_year if s["starredAt"][:10] == date.isoformat()]
        if not repos:
            continue
        repos_sorted = sorted(repos, key=lambda s: s["stargazerCount"], reverse=True)
        holiday_cards.append(
            {
                "key": key,
                "label": label,
                "date": date.isoformat(),
                "count": len(repos_sorted),
                "repos": [
                    {
                        "nameWithOwner": r["nameWithOwner"],
                        "stars": r["stargazerCount"],
                        "language": r.get("primaryLanguage"),
                        "url": r.get("url"),
                    }
                    for r in repos_sorted[:6]
                ],
            }
        )

    # Category stats based on stars in the selected year
    stars_year_with_cat = []
    category_counts = Counter()
    for s in stars_year:
        cat = classify_repo(
            name=s["nameWithOwner"],
            language=s.get("primaryLanguage"),
            topics=s.get("topicsCanonical") or s.get("topics") or [],
            description=s.get("description"),
        )
        stars_year_with_cat.append({**s, "category": cat})
        category_counts[cat["primary"]] += 1

    category_top_repos = defaultdict(list)
    for s in stars_year_with_cat:
        category_top_repos[s["category"]["primary"]].append(s)
    for key in category_top_repos:
        category_top_repos[key].sort(key=lambda s: s.get("stargazerCount", 0), reverse=True)

    # Contribution-side category aggregation (merged PRs)
    pr_category_lines = Counter()
    pr_category_count = Counter()
    for pr in merged_prs:
        cat = classify_repo(
            name=pr["repo"],
            language=pr.get("repoLanguage"),
            topics=pr.get("repoTopics", []),
            description=None,
        )
        pr_category_count[cat["primary"]] += 1
        pr_category_lines[cat["primary"]] += pr["additions"] + pr["deletions"]

    # Radar values: combine star count + PR lines signal
    max_star_cat = max(category_counts.values(), default=0)
    max_pr_lines_cat = max(pr_category_lines.values(), default=0)

    radar = {}
    for cat in CATEGORIES:
        if cat.key == "other":
            radar[cat.key] = 0
            continue
        star_score = normalize(category_counts.get(cat.key, 0), max_star_cat)
        pr_score = normalize(pr_category_lines.get(cat.key, 0), max_pr_lines_cat)
        value = (0.65 * star_score) + (0.35 * pr_score)
        radar[cat.key] = int(round(clamp(value, 0.0, 1.0) * 100))

    radar_main = [(k, v) for k, v in radar.items() if k != "other"]
    primary_track = max(radar_main, key=lambda kv: kv[1])[0] if radar_main else "systems"

    # 90-day events: deep-night push (best-effort)
    events = load(RAW / "events_90d.json") if (RAW / "events_90d.json").exists() else []
    push_events = [e for e in events if e.get("type") == "PushEvent"]

    def night_score(local_dt: dt.datetime) -> float:
        # Higher is "deeper night": prefer 00:00-05:59 and 23:00-23:59
        h = local_dt.hour + local_dt.minute / 60.0
        if h >= 23:
            return 1.0 + (h - 23) / 1.0
        if h < 6:
            return 1.0 + (6 - h) / 6.0
        return 0.0

    deep_night_push = None
    if push_events:
        def key_fn(ev):
            local_dt = to_local(ev.get("created_at"))
            return (night_score(local_dt), local_dt.isoformat())

        deep_night_push = max(push_events, key=key_fn)

    deep_night_push_card = None
    if deep_night_push:
        local_dt = to_local(deep_night_push.get("created_at"))
        commits = (deep_night_push.get("payload") or {}).get("commits") or []
        deep_night_push_card = {
            "localTime": local_dt.isoformat(),
            "repo": (deep_night_push.get("repo") or {}).get("name"),
            "commitCount": len(commits),
            "sampleMessages": [c.get("message", "")[:80] for c in commits[:3]],
        }

    # Identity (data-driven, year-specific)
    # Keep it short but evidence-backed: languages come from stars-in-year; tracks come from radar; OSS highlight from merged PRs.
    star_langs = [lang for lang, _ in star_lang_year.most_common(6) if lang]
    lang_line = " / ".join(star_langs[:3]) if star_langs else ""

    track_labels = []
    for key, value in sorted(radar.items(), key=lambda kv: kv[1], reverse=True):
        if value <= 0:
            continue
        cat = CATEGORY_BY_KEY.get(key)
        if not cat:
            continue
        track_labels.append(cat.label)
        if len(track_labels) >= 2:
            break
    track_line = " × ".join(track_labels) if track_labels else ""

    oss_line = ""
    if oss_award_repo:
        short = oss_award_repo.split("/")[-1]
        lines = safe_int(pr_by_repo.get(oss_award_repo, {}).get("lines", 0), 0)
        if lines > 0:
            oss_line = f"PR@{short} {lines}行"
        else:
            oss_line = f"PR@{short}"

    identity_parts = [p for p in (lang_line, track_line, oss_line) if p]
    identity = " · ".join(identity_parts[:3]) if identity_parts else "开发者"

    # Meet GitHub duration (as of YEAR-12-31)
    created_date = parse_iso8601(user.get("created_at")).date()
    report_end = dt.date(YEAR, 12, 31)
    days_since = (report_end - created_date).days
    years_approx = round(days_since / 365.2425, 1)

    # Enrich starred repos with classification signals and local timestamps (deterministic).
    for s in stars_year:
        cat = classify_repo(
            name=s["nameWithOwner"],
            language=s.get("primaryLanguage"),
            topics=s.get("topicsCanonical") or s.get("topics") or [],
            description=s.get("description"),
        )
        s["track"] = cat["primary"]
        s["trackTags"] = cat["tags"]
        s["trackSignals"] = list(cat.get("signals") or [])
        s["trackUsedLanguageFallback"] = bool(cat.get("usedLanguageFallback"))
        s["starredAtLocal"] = to_local(s["starredAt"]).isoformat()

    # Dynamic categories (data-driven; derived from starred repos topics/languages).
    topic_candidates = [t for t, c in star_topic_canon_year.most_common() if c >= 4][:18]
    lang_candidates_raw = [l for l, c in star_lang_year.most_common() if c >= 6][:8]
    lang_to_topic = {l: canon_topic(l) for l in lang_candidates_raw if canon_topic(l) in topic_candidates}
    lang_candidates = [l for l in lang_candidates_raw if l not in lang_to_topic]

    def dynamic_key_for_repo(s: dict) -> str:
        topics = s.get("topicsCanonical") or []
        if topics:
            choices = [t for t in topics if t in topic_candidates]
            if choices:
                choices.sort(key=lambda t: (-star_topic_canon_year.get(t, 0), t))
                return f"t:{choices[0]}"
        lang = s.get("primaryLanguage")
        if lang:
            merged = lang_to_topic.get(lang)
            if merged:
                return f"t:{merged}"
            if lang in lang_candidates:
                return f"l:{lang}"
        return "other"

    for s in stars_year:
        s["dynamicCategoryKey"] = dynamic_key_for_repo(s)

    stars_year_sorted = sorted(
        stars_year,
        key=lambda s: (s.get("stargazerCount", 0), s.get("starredAt") or ""),
        reverse=True,
    )

    dyn_repo_names = defaultdict(list)
    for s in stars_year_sorted:
        dyn_repo_names[s["dynamicCategoryKey"]].append(s["nameWithOwner"])

    dyn_defs = []
    for t in topic_candidates:
        key = f"t:{t}"
        dyn_defs.append(
            {
                "key": key,
                "kind": "topic",
                "label": f"#{t}",
                "count": len(dyn_repo_names.get(key, [])),
                "color": stable_palette_color(key),
                "repoNames": dyn_repo_names.get(key, []),
            }
        )
    for l in lang_candidates:
        key = f"l:{l}"
        dyn_defs.append(
            {
                "key": key,
                "kind": "language",
                "label": f"语言 · {l}",
                "count": len(dyn_repo_names.get(key, [])),
                "color": stable_palette_color(key),
                "repoNames": dyn_repo_names.get(key, []),
            }
        )
    dyn_defs.append(
        {
            "key": "other",
            "kind": "other",
            "label": "其他/未标注",
            "count": len(dyn_repo_names.get("other", [])),
            "color": stable_palette_color("other"),
            "repoNames": dyn_repo_names.get("other", []),
        }
    )
    dyn_defs.sort(key=lambda d: (-d["count"], d["key"]))

    stars_block = {
        "totalAllTime": len(stars),
        "byYear": dict(sorted(stars_by_year.items())),
        "totalInYear": len(stars_year),
        "reposInYear": [
            {
                "nameWithOwner": s["nameWithOwner"],
                "stars": safe_int(s.get("stargazerCount")),
                "forks": safe_int(s.get("forkCount")),
                "language": s.get("primaryLanguage"),
                "starredAt": s.get("starredAt"),
                "starredAtLocal": s.get("starredAtLocal"),
                "url": s.get("url"),
                "description": s.get("description"),
                "topics": list(s.get("topics", []) or []),
                "topicsCanonical": list(s.get("topicsCanonical", []) or []),
                "track": s.get("track"),
                "trackTags": list(s.get("trackTags", []) or []),
                "trackSignals": list(s.get("trackSignals", []) or []),
                "trackUsedLanguageFallback": bool(s.get("trackUsedLanguageFallback")),
                "dynamicCategoryKey": s.get("dynamicCategoryKey"),
            }
            for s in sorted(
                stars_year_sorted,
                key=lambda s: (s.get("stargazerCount", 0), s.get("starredAt") or ""),
                reverse=True,
            )
        ],
        "byMonthInYear": [
            {
                "month": month,
                "count": star_month_counts.get(month, 0),
                "topRepo": {
                    "nameWithOwner": (star_month_top_repo.get(month) or {}).get("nameWithOwner"),
                    "stars": (star_month_top_repo.get(month) or {}).get("stargazerCount"),
                    "url": (star_month_top_repo.get(month) or {}).get("url"),
                },
                "repos": [
                    {
                        "nameWithOwner": r.get("nameWithOwner"),
                        "stars": safe_int(r.get("stargazerCount")),
                        "url": r.get("url"),
                    }
                    for r in (star_month_repos.get(month) or [])[:12]
                ],
                "events": [
                    {
                        "starredAt": e.get("starredAt"),
                        "starredAtLocal": to_local(e.get("starredAt")).isoformat() if e.get("starredAt") else None,
                        "nameWithOwner": e.get("nameWithOwner"),
                        "stars": safe_int(e.get("stars")),
                        "url": e.get("url"),
                    }
                    for e in (star_month_events.get(month) or [])
                ],
            }
            for month, _ in sorted(month_contrib_sorted)
        ],
        "byHourLocalInYear": star_hour_local,
        "topLanguagesInYear": [{"name": k, "count": v} for k, v in star_lang_year.most_common(12)],
        "topTopicsInYear": [{"name": k, "count": v} for k, v in star_topic_year.most_common(24)],
        "topTopicsCanonicalInYear": [{"name": k, "count": v} for k, v in star_topic_canon_year.most_common(24)],
        "topStarredReposInYear": [
            {
                "nameWithOwner": s["nameWithOwner"],
                "stars": s["stargazerCount"],
                "language": s.get("primaryLanguage"),
                "starredAt": s["starredAt"],
                "starredAtLocal": s.get("starredAtLocal"),
                "url": s["url"],
            }
            for s in sorted(stars_year_sorted, key=lambda s: s["stargazerCount"], reverse=True)[:20]
        ],
        "dynamicCategoriesInYear": dyn_defs,
        "firstStarInYear": {
            "starredAt": first_star_year["starredAt"] if first_star_year else None,
            "starredAtLocal": to_local(first_star_year["starredAt"]).isoformat() if first_star_year else None,
            "repo": first_star_year["nameWithOwner"] if first_star_year else None,
            "url": first_star_year.get("url") if first_star_year else None,
        },
        "latestStarInYear": {
            "starredAt": latest_star_year["starredAt"] if latest_star_year else None,
            "starredAtLocal": to_local(latest_star_year["starredAt"]).isoformat() if latest_star_year else None,
            "repo": latest_star_year["nameWithOwner"] if latest_star_year else None,
            "url": latest_star_year.get("url") if latest_star_year else None,
        },
    }

    # Backwards-compat aliases for older year-specific renderers (2024/2025 HTML).
    stars_block[f"total{YEAR}"] = stars_block["totalInYear"]
    stars_block[f"repos{YEAR}"] = stars_block["reposInYear"]
    stars_block[f"byMonth{YEAR}"] = stars_block["byMonthInYear"]
    stars_block[f"byHourLocal{YEAR}"] = stars_block["byHourLocalInYear"]
    stars_block[f"topLanguages{YEAR}"] = stars_block["topLanguagesInYear"]
    stars_block[f"topTopics{YEAR}"] = stars_block["topTopicsInYear"]
    stars_block[f"topStarredRepos{YEAR}"] = stars_block["topStarredReposInYear"]
    stars_block[f"firstStar{YEAR}"] = stars_block["firstStarInYear"]
    stars_block[f"latestStar{YEAR}"] = stars_block["latestStarInYear"]

    categories_block = {
        "definitions": [
            {
                "key": c.key,
                "label": c.label,
                "emoji": c.emoji,
                "color": c.color,
                "description": c.description,
            }
            for c in CATEGORIES
        ],
        "starCountsInYear": {k: category_counts.get(k, 0) for k in CATEGORY_BY_KEY.keys()},
        "topReposInYear": {
            k: [
                {
                    "nameWithOwner": s["nameWithOwner"],
                    "stars": s["stargazerCount"],
                    "language": s.get("primaryLanguage"),
                    "url": s["url"],
                }
                for s in category_top_repos.get(k, [])[:6]
            ]
            for k in CATEGORY_BY_KEY.keys()
        },
        "radar": radar,
        "primaryTrack": primary_track,
        "prCountsByCategory": dict(pr_category_count),
        "prLinesByCategory": dict(pr_category_lines),
    }
    categories_block[f"starCounts{YEAR}"] = categories_block["starCountsInYear"]
    categories_block[f"topRepos{YEAR}"] = categories_block["topReposInYear"]

    dataset = {
        "meta": {
            "year": YEAR,
            "timezone": str(TZ),
            "generatedAt": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
            "dataProvenance": {
                "rawDir": str(RAW),
                "notes": [
                    "所有数值来自 gh api 的原始 JSON（仓库内已保存），页面仅做统计与可视化。",
                    "Star 列表来自当前 starredRepositories；取消 Star 的历史无法从公开 API 追溯，可能低估当年 Star 数。",
                    "Star 仓库 track 分类为启发式：基于 description/name/topics 的关键词规则生成，并在 reposInYear[].trackSignals 展示命中依据；缺少元数据时会落入「未标注/其它」。",
                    "GitHub Events API 仅保留近 90 天；深夜提交彩蛋为 best-effort。",
                    f"仓库收到的 Stars/Forks 为当前快照，不代表 {YEAR} 新增。",
                ],
            },
        },
        "user": {
            "login": login,
            "name": user.get("name"),
            "avatarUrl": user.get("avatar_url"),
            "createdAt": user.get("created_at"),
            "followers": safe_int(user.get("followers")),
            "following": safe_int(user.get("following")),
            "meetGitHub": {
                "asOf": report_end.isoformat(),
                "days": days_since,
                "yearsApprox": years_approx,
            },
            "firstRepo": {
                "name": first_repo.get("name") if first_repo else None,
                "createdAt": first_repo.get("created_at") if first_repo else None,
            },
            "firstStarEver": {
                "starredAt": first_star_ever["starredAt"] if first_star_ever else None,
                "repo": first_star_ever["nameWithOwner"] if first_star_ever else None,
                "url": first_star_ever.get("url") if first_star_ever else None,
            },
            "identity": identity,
        },
        "year": {
            "totals": {
                "contributions": safe_int(calendar.get("totalContributions")),
                "commits": safe_int(cc.get("totalCommitContributions")),
                "prs": safe_int(cc.get("totalPullRequestContributions")),
                "issues": safe_int(cc.get("totalIssueContributions")),
                "reposContributedTo": safe_int(cc.get("totalRepositoryContributions")),
            },
            "activity": {
                "totalDays": total_days,
                "activeDays": active_days,
                "inactiveDays": inactive_days,
                "activeRate": round(active_days / total_days, 3) if total_days else 0,
                "busiestDay": busiest_day,
                "topDays": top_days,
                "longestStreak": streak,
                "weekdaySums": weekday_sums,
                "weekendRatio": round(weekend_ratio, 3),
                "patternLabel": pattern,
                "mostActiveMonth": most_active_month,
                "byMonth": [
                    {
                        "month": m,
                        "contributions": month_contrib[m],
                        "activeDays": month_active_days.get(m, 0),
                    }
                    for m, _ in month_contrib_sorted
                ],
            },
            "repos": {
                "total": len(repos),
                "ownTotal": len(own_repos),
                "createdByYear": dict(repos_created_by_year),
                "createdInYear": sum(1 for r in own_repos if (r.get("created_at") or "").startswith(f"{YEAR}-")),
                "ownStarsTotalSnapshot": sum(safe_int(r.get("stargazers_count")) for r in own_repos),
                "ownForksTotalSnapshot": sum(safe_int(r.get("forks_count")) for r in own_repos),
                "topSnapshot": [
                    {
                        "name": r.get("name"),
                        "stars": safe_int(r.get("stargazers_count")),
                        "forks": safe_int(r.get("forks_count")),
                        "language": r.get("language"),
                        "createdAt": r.get("created_at"),
                        "url": r.get("html_url"),
                    }
                    for r in sorted(own_repos, key=lambda r: safe_int(r.get("stargazers_count")), reverse=True)[:10]
                ],
            },
            "contributions": {
                "calendarWeeks": weeks,
                "topCommitRepos": sorted(commit_repo_list, key=lambda x: x["count"], reverse=True)[:12],
                "topPrRepos": sorted(pr_repo_list, key=lambda x: x["count"], reverse=True)[:12],
                "topIssueRepos": sorted(issue_repo_list, key=lambda x: x["count"], reverse=True)[:12],
                "focusProject": focus_project,
            },
            "stars": stars_block,
            "discoveries": {
                "newTopics": new_topics,
                "risingTopics": rising_topics,
                "newTopicsCanonical": new_topics_canon,
                "risingTopicsCanonical": rising_topics_canon,
                "newLanguages": new_langs,
            },
            "categories": categories_block,
            "openSource": {
                "mergedPrs": {
                    "total": pr_total,
                    "additions": pr_add_total,
                    "deletions": pr_del_total,
                    "lines": pr_lines_total,
                },
                "mergedPrsList": sorted(
                    merged_prs,
                    key=lambda pr: pr.get("mergedAt") or pr.get("createdAt") or "",
                    reverse=True,
                ),
                "ossAward": oss_award,
                "biggestPr": biggest_pr,
                "latestPrCreated": latest_pr_created,
                "highlights": sorted(merged_prs, key=lambda pr: pr["additions"] + pr["deletions"], reverse=True)[:8],
                "externalContributedRepos": [
                    {
                        "nameWithOwner": r.get("nameWithOwner"),
                        "stars": safe_int(r.get("stargazerCount")),
                        "language": (r.get("primaryLanguage") or {}).get("name"),
                        "topics": [t["topic"]["name"] for t in (r.get("repositoryTopics") or {}).get("nodes", [])],
                        "owner": (r.get("owner") or {}).get("login"),
                    }
                    for r in external_contrib_sorted[:18]
                ],
            },
            "specialDates": {
                "holidayStars": holiday_cards,
                "deepNightPush90d": deep_night_push_card,
            },
        },
    }

    out_path = OUT_DIR / "dataset.json"
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print("dataset written to", out_path)


if __name__ == "__main__":
    main()
