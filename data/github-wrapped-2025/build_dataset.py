import json
import math
import re
import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


BASE = Path("data/github-wrapped-2025")
RAW = BASE / "raw"
OUT_DIR = BASE / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2025
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
        key="ai",
        label="AI/LLM",
        emoji="🧬",
        color="#ff4fd8",
        description="LLM、RAG、Agent、推理、OpenAI、MCP",
    ),
    Category(
        key="tooling",
        label="工具链/效率",
        emoji="🛠️",
        color="#7aa7ff",
        description="CLI、终端、编辑器、开发效率与自动化",
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
    "systems": {
        "rust",
        "cpp",
        "c++",
        "c",
        "zig",
        "performance",
        "async",
        "wasm",
        "compiler",
        "runtime",
        "benchmark",
        "profiling",
        "systems-programming",
    },
    "tooling": {
        "cli",
        "terminal",
        "editor",
        "zed",
        "vscode",
        "neovim",
        "tmux",
        "git",
        "productivity",
        "automation",
        "devtools",
        "workflow",
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
        "observability",
        "monitoring",
        "distributed-systems",
        "cloud",
        "infra",
        "server",
        "proxy",
        "networking",
    },
}


LANG_WEIGHTS = {
    "Rust": {"systems": 2, "database": 1, "ai": 1},
    "C": {"systems": 2},
    "C++": {"systems": 2, "database": 1},
    "Zig": {"systems": 2},
    "Go": {"infra": 2, "tooling": 1},
    "Python": {"ai": 2, "tooling": 1},
    "TypeScript": {"frontend": 2, "tooling": 1},
    "JavaScript": {"frontend": 2, "tooling": 1},
    "Dart": {"frontend": 2},
    "Swift": {"frontend": 2},
    "Shell": {"tooling": 2, "infra": 1},
}


def classify_repo(*, name: str, language: str | None, topics: list[str], description: str | None) -> dict:
    name_l = (name or "").lower()
    desc_l = (description or "").lower()
    topics_l = [t.lower() for t in (topics or [])]

    score = {c.key: 0 for c in CATEGORIES}

    for cat_key, kws in TOPIC_KEYWORDS.items():
        for t in topics_l:
            if t in kws:
                score[cat_key] += 3

    if language:
        for cat_key, weight in LANG_WEIGHTS.get(language, {}).items():
            score[cat_key] += weight

    # Lightweight keyword hints from name/description.
    if re.search(r"\\b(llm|rag|openai|agent|mcp|claude|codex|deepseek|qwen)\\b", name_l + " " + desc_l):
        score["ai"] += 2
    if re.search(r"\\b(sql|database|clickhouse|datafusion|olap|arrow)\\b", name_l + " " + desc_l):
        score["database"] += 2
    if re.search(r"\\b(cli|terminal|zed|vscode|tool)\\b", name_l + " " + desc_l):
        score["tooling"] += 1
    if re.search(r"\\b(tauri|wasm|react|vue|svelte|frontend)\\b", name_l + " " + desc_l):
        score["frontend"] += 1
    if re.search(r"\\b(docker|kubernetes|ci|cd|devops|proxy)\\b", name_l + " " + desc_l):
        score["infra"] += 1

    best_key = max(score, key=lambda k: (score[k], k))
    sorted_keys = sorted(score.keys(), key=lambda k: score[k], reverse=True)
    tags = [k for k in sorted_keys if score[k] > 0][:3]
    return {"primary": best_key, "scores": score, "tags": tags}


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

    # Starred repos (all-time) + 2025 slice
    star_pages = load(RAW / "starred_repos_pages.json")
    star_edges = []
    for page in star_pages:
        star_edges.extend(page["data"]["user"]["starredRepositories"]["edges"])

    stars = []
    for edge in star_edges:
        node = edge["node"]
        stars.append(
            {
                "starredAt": edge["starredAt"],
                "nameWithOwner": node["nameWithOwner"],
                "description": node.get("description"),
                "stargazerCount": safe_int(node.get("stargazerCount")),
                "forkCount": safe_int(node.get("forkCount")),
                "primaryLanguage": (node.get("primaryLanguage") or {}).get("name"),
                "topics": [t["topic"]["name"] for t in (node.get("repositoryTopics") or {}).get("nodes", [])],
                "url": f"https://github.com/{node['nameWithOwner']}",
            }
        )

    stars_2025 = [s for s in stars if s["starredAt"].startswith(f"{YEAR}-")]
    stars_before = [s for s in stars if s["starredAt"] < f"{YEAR}-01-01T00:00:00Z"]

    stars_by_year = defaultdict(int)
    for s in stars:
        year = s["starredAt"][:4]
        stars_by_year[year] += 1

    star_month_counts = Counter([s["starredAt"][:7] for s in stars_2025])
    star_month_top_repo = {}
    for s in stars_2025:
        key = s["starredAt"][:7]
        prev = star_month_top_repo.get(key)
        if not prev or s["stargazerCount"] > prev["stargazerCount"]:
            star_month_top_repo[key] = s

    star_month_repos = defaultdict(list)
    for s in stars_2025:
        star_month_repos[s["starredAt"][:7]].append(s)
    for month_key in star_month_repos:
        star_month_repos[month_key].sort(key=lambda x: (x.get("stargazerCount", 0), x.get("starredAt") or ""), reverse=True)

    # Star events by month (for timeline charts)
    star_month_events = defaultdict(list)
    for s in stars_2025:
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
    for s in stars_2025:
        hour = to_local(s["starredAt"]).hour
        star_hour_local[hour] += 1

    star_lang_2025 = Counter([s["primaryLanguage"] for s in stars_2025 if s.get("primaryLanguage")])
    star_topic_2025 = Counter()
    for s in stars_2025:
        for t in s.get("topics", []):
            star_topic_2025[t.lower()] += 1

    star_topic_before = Counter()
    for s in stars_before:
        for t in s.get("topics", []):
            star_topic_before[t.lower()] += 1

    new_topics = []
    rising_topics = []
    for topic, cur in star_topic_2025.most_common():
        prev = star_topic_before.get(topic, 0)
        if prev == 0 and cur >= 3:
            new_topics.append({"topic": topic, "count2025": cur, "countBefore": 0})
        elif prev > 0 and cur >= max(5, prev * 2):
            rising_topics.append(
                {
                    "topic": topic,
                    "count2025": cur,
                    "countBefore": prev,
                    "ratio": round(cur / prev, 2) if prev else None,
                }
            )

    new_topics = new_topics[:12]
    rising_topics = sorted(rising_topics, key=lambda x: (-(x["ratio"] or 0), -x["count2025"]))[:12]

    lang_before = Counter([s["primaryLanguage"] for s in stars_before if s.get("primaryLanguage")])
    new_langs = []
    for lang, cur in star_lang_2025.most_common():
        prev = lang_before.get(lang, 0)
        if prev == 0 and cur >= 2:
            new_langs.append({"language": lang, "count2025": cur, "countBefore": 0})
    new_langs = new_langs[:8]

    first_star_ever = min(stars, key=lambda s: s["starredAt"]) if stars else None
    first_star_2025 = min(stars_2025, key=lambda s: s["starredAt"]) if stars_2025 else None
    latest_star_2025 = max(stars_2025, key=lambda s: s["starredAt"]) if stars_2025 else None

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

    # Merged PRs (2025)
    pr_pages = load(RAW / "prs_2025_pages.json")
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

    # Holiday stars (Chinese-focused, 2025)
    spring_festival = dt.date(2025, 1, 29)
    chuxi = spring_festival - dt.timedelta(days=1)
    holidays = [
        ("new_year", "元旦", dt.date(2025, 1, 1)),
        ("chuxi", "除夕", chuxi),
        ("spring_festival", "春节", spring_festival),
        ("qingming", "清明节", dt.date(2025, 4, 4)),
        ("labor_day", "劳动节", dt.date(2025, 5, 1)),
        ("dragon_boat", "端午节", dt.date(2025, 5, 31)),
        ("qixi", "七夕", dt.date(2025, 8, 29)),
        ("national_day", "国庆节", dt.date(2025, 10, 1)),
        ("mid_autumn", "中秋节", dt.date(2025, 10, 6)),
        ("programmer_day", "程序员节", dt.date(2025, 10, 24)),
        ("singles_day", "双十一", dt.date(2025, 11, 11)),
        ("new_year_eve", "跨年夜(12/31)", dt.date(2025, 12, 31)),
    ]

    holiday_cards = []
    for key, label, date in holidays:
        repos = [s for s in stars_2025 if s["starredAt"][:10] == date.isoformat()]
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

    # Category stats based on 2025 stars
    stars_2025_with_cat = []
    category_counts = Counter()
    for s in stars_2025:
        cat = classify_repo(
            name=s["nameWithOwner"],
            language=s.get("primaryLanguage"),
            topics=s.get("topics", []),
            description=s.get("description"),
        )
        stars_2025_with_cat.append({**s, "category": cat})
        category_counts[cat["primary"]] += 1

    category_top_repos = defaultdict(list)
    for s in stars_2025_with_cat:
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
        star_score = normalize(category_counts.get(cat.key, 0), max_star_cat)
        pr_score = normalize(pr_category_lines.get(cat.key, 0), max_pr_lines_cat)
        value = (0.65 * star_score) + (0.35 * pr_score)
        radar[cat.key] = int(round(clamp(value, 0.0, 1.0) * 100))

    primary_track = max(radar.items(), key=lambda kv: kv[1])[0] if radar else "systems"

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

    # Identity (avoid over-assertive single label)
    primary_langs = [lang for lang, _ in Counter([r.get("language") for r in own_repos if r.get("language")]).most_common(5)]
    top_topics = [t for t, _ in star_topic_2025.most_common(8)]

    identity_lines = []
    if primary_langs:
        identity_lines.append(" / ".join(primary_langs[:3]))
    if "rust" in top_topics:
        identity_lines.append("Rust 生态重度关注")
    if "ai" in top_topics or "llm" in top_topics:
        identity_lines.append("AI 工具链探索")
    if any(repo["nameWithOwner"].startswith("apache/datafusion") for repo in external_contrib_sorted[:10]) or (oss_award_repo == "apache/datafusion"):
        identity_lines.append("DataFusion 开源贡献")

    identity = " · ".join(identity_lines[:3]) if identity_lines else "开发者"

    # Meet GitHub duration (as of 2025-12-31)
    created_date = parse_iso8601(user.get("created_at")).date()
    report_end = dt.date(YEAR, 12, 31)
    days_since = (report_end - created_date).days
    years_approx = round(days_since / 365.2425, 1)

    dataset = {
        "meta": {
            "year": YEAR,
            "timezone": "Asia/Shanghai",
            "generatedAt": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
            "dataProvenance": {
                "rawDir": str(RAW),
                "notes": [
                    "所有数值来自 gh api 的原始 JSON（仓库内已保存），页面仅做统计与可视化。",
                    "GitHub Events API 仅保留近 90 天；深夜提交彩蛋为 best-effort。",
                    "仓库收到的 Stars/Forks 为当前快照，不代表 2025 新增。",
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
            "stars": {
                "total2025": len(stars_2025),
                "totalAllTime": len(stars),
                "repos2025": [
                    {
                        "nameWithOwner": s["nameWithOwner"],
                        "stars": safe_int(s.get("stargazerCount")),
                        "forks": safe_int(s.get("forkCount")),
                        "language": s.get("primaryLanguage"),
                        "starredAt": s.get("starredAt"),
                        "url": s.get("url"),
                        "description": s.get("description"),
                        "topics": list(s.get("topics", []) or []),
                    }
                    for s in sorted(stars_2025, key=lambda s: (s.get("stargazerCount", 0), s.get("starredAt") or ""), reverse=True)
                ],
                "byYear": dict(sorted(stars_by_year.items())),
                "byMonth2025": [
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
                                "nameWithOwner": e.get("nameWithOwner"),
                                "stars": safe_int(e.get("stars")),
                                "url": e.get("url"),
                            }
                            for e in (star_month_events.get(month) or [])
                        ],
                    }
                    for month, _ in sorted(month_contrib_sorted)
                ],
                "byHourLocal2025": star_hour_local,
                "topLanguages2025": [{"name": k, "count": v} for k, v in star_lang_2025.most_common(12)],
                "topTopics2025": [{"name": k, "count": v} for k, v in star_topic_2025.most_common(24)],
                "topStarredRepos2025": [
                    {
                        "nameWithOwner": s["nameWithOwner"],
                        "stars": s["stargazerCount"],
                        "language": s.get("primaryLanguage"),
                        "starredAt": s["starredAt"],
                        "url": s["url"],
                    }
                    for s in sorted(stars_2025, key=lambda s: s["stargazerCount"], reverse=True)[:20]
                ],
                "firstStar2025": {
                    "starredAt": first_star_2025["starredAt"] if first_star_2025 else None,
                    "repo": first_star_2025["nameWithOwner"] if first_star_2025 else None,
                    "url": first_star_2025.get("url") if first_star_2025 else None,
                },
                "latestStar2025": {
                    "starredAt": latest_star_2025["starredAt"] if latest_star_2025 else None,
                    "repo": latest_star_2025["nameWithOwner"] if latest_star_2025 else None,
                    "url": latest_star_2025.get("url") if latest_star_2025 else None,
                },
            },
            "discoveries": {
                "newTopics": new_topics,
                "risingTopics": rising_topics,
                "newLanguages": new_langs,
            },
            "categories": {
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
                "starCounts2025": {k: category_counts.get(k, 0) for k in CATEGORY_BY_KEY.keys()},
                "topRepos2025": {
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
            },
            "openSource": {
                "mergedPrs": {
                    "total": pr_total,
                    "additions": pr_add_total,
                    "deletions": pr_del_total,
                    "lines": pr_lines_total,
                },
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
