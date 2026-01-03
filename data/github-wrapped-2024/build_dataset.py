import json
import math
import re
import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


BASE = Path("data/github-wrapped-2024")
RAW = BASE / "raw"
OUT_DIR = BASE / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2024
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


"""
Star 技术雷达：领域分类

说明：
- 这里的分类完全基于仓库的 `description` + `topics`（其次才参考语言/名字）。
- 目标是把 2024 年 Star 的 300+ 项目做出“领域”划分，而不是语言/生态细分。
"""


# 26 个领域分类（可扩展；保持 key 稳定便于前端渲染）
CATEGORIES = [
    # AI / LLM
    Category(key="ai_inference", label="模型推理/运行时", emoji="⚙️", color="#f97316", description="ollama, llama.cpp, mistral.rs, LocalAI"),
    Category(key="ai_rag_agent", label="RAG/Agent 工具链", emoji="🧠", color="#8b5cf6", description="LangChain, GraphRAG, MCP, tool-calling"),
    Category(key="ai_platform", label="AI 平台/工作流", emoji="🧩", color="#0ea5e9", description="Dify, Langflow, Flowise, FastGPT"),
    Category(key="ai_apps", label="AI 应用/客户端", emoji="💬", color="#f43f5e", description="Open WebUI, NextChat, Tabby, AnythingLLM"),

    # 数据 / 数据库
    Category(key="db_systems", label="数据库系统", emoji="🗄️", color="#22c55e", description="Postgres/Neon, 时序/图/分布式数据库"),
    Category(key="db_storage", label="存储引擎/嵌入式 DB", emoji="🧱", color="#eab308", description="RocksDB/LSM, embedded database, storage engine"),
    Category(key="db_tooling", label="SQL/ORM/数据库工具", emoji="🧰", color="#fb923c", description="sqlx, sea-orm, sqlglot, Chat2DB"),
    Category(key="vector_search", label="向量检索/Embedding", emoji="🧲", color="#a855f7", description="Qdrant, Milvus, pgvector, ANN/embedding"),
    Category(key="data_analytics", label="分析引擎/数据计算", emoji="📈", color="#06b6d4", description="OLAP, DataFusion, Polars, data warehouse/lakehouse"),
    Category(key="data_formats", label="数据格式/解析/序列化", emoji="🧾", color="#14b8a6", description="Arrow/Parquet, serde/protobuf, parser/format"),
    Category(key="messaging_stream", label="消息队列/流处理", emoji="📬", color="#38bdf8", description="MQ, streaming, pubsub, event processing"),
    Category(key="search_index", label="搜索/索引/检索", emoji="🔎", color="#f472b6", description="search engine, index, full-text, filters"),

    # 基础设施
    Category(key="storage_fs", label="对象存储/文件系统", emoji="🪣", color="#60a5fa", description="S3/MinIO, Ceph, JuiceFS, SeaweedFS"),
    Category(key="cloud_devops", label="容器/K8s/DevOps", emoji="☁️", color="#93c5fd", description="Docker/Podman, Kubernetes, CI/CD, cloud-native"),
    Category(key="proxy_vpn", label="代理/VPN/网络客户端", emoji="🛡️", color="#34d399", description="Clash, proxy/VPN clients, tunneling"),
    Category(key="network_protocols", label="网络协议/传输库", emoji="🌐", color="#22d3ee", description="QUIC/HTTP3, gRPC, WebSocket, TCP/IP, RDMA"),

    # 系统 / 性能 / 安全
    Category(key="observability_perf", label="可观测性/性能分析", emoji="📊", color="#fbbf24", description="benchmark, flamegraph/pprof, tracing/metrics"),
    Category(key="security", label="安全/网络扫描", emoji="🔐", color="#ef4444", description="port scanner, pentest, traffic inspection"),
    Category(key="systems_os", label="操作系统/虚拟化/内核", emoji="🧬", color="#a78bfa", description="kernel, WSL/VM, LibOS, OS/runtime"),
    Category(key="systems_libs", label="高性能系统库", emoji="⚙️", color="#3b82f6", description="folly/abseil, hashmaps, coroutine/async primitives"),

    # 开发工具
    Category(key="lang_tooling", label="语言工具链/编译器", emoji="🧪", color="#c084fc", description="compiler/interpreter, linter/formatter, LLVM/Cranelift"),
    Category(key="build_release", label="构建/包管理/发布", emoji="📦", color="#fdba74", description="cargo, rye/pixi/uv, cross-compile, build tooling"),
    Category(key="editor_ide", label="编辑器/IDE", emoji="✏️", color="#a3e635", description="Zed, Helix, Neovim, VS Code tooling"),
    Category(key="cli_terminal", label="CLI/终端工具", emoji="⌨️", color="#67e8f9", description="nushell, atuin, gitui, terminal utilities"),
    Category(key="ui_frameworks", label="UI/跨端框架", emoji="🖼️", color="#fb7185", description="Tauri/Flutter/Avalonia, React/Svelte/Solid, GUI frameworks"),

    # 应用与资料
    Category(key="apps_productivity", label="应用/效率工具", emoji="🧩", color="#f9a8d4", description="PDF/翻译/OCR/文件管理/桌面工具"),
    Category(key="media", label="媒体/影音/娱乐", emoji="🎬", color="#fda4af", description="IPTV, music/video apps, media utilities"),
    Category(key="learning", label="学习/资料/清单", emoji="📚", color="#fde047", description="books, tutorials, papers, curated lists"),
]


CATEGORY_BY_KEY = {c.key: c for c in CATEGORIES}


def _tokenize(text: str) -> set[str]:
    # Keep hyphenated tokens used by GitHub topics (e.g., "vector-database").
    raw = set(re.findall(r"[a-z0-9][a-z0-9_+.#\\-]*", text.lower()))
    parts: set[str] = set()
    for t in raw:
        parts.update(x for x in re.split(r"[-_.+/]+", t) if x)
    return raw | parts


def _phrase_match(*, phrase: str, text: str, tokens: set[str]) -> bool:
    p = (phrase or "").strip().lower()
    if not p:
        return False

    # Chinese (and other non-ASCII) phrases: substring match is usually intended.
    if any(ord(ch) > 127 for ch in p):
        return p in text

    # Multi-word ASCII phrases: keep substring semantics.
    if re.search(r"\\s", p):
        return p in text

    # Short ASCII tokens like IDE/ORM/SQL: require token match to avoid false positives
    # (e.g. "orm" in "performance", "ide" in "side").
    if len(p) <= 3:
        return p in tokens

    return p in tokens or p in text


# 分类规则（topics 为强信号；tokens/phrases 为弱信号；尽量用 description 来决定“领域”）
RULES: dict[str, dict] = {
    "ai_inference": {
        "topics": {"ollama", "llama", "ggml", "inference", "cuda", "tensorrt", "triton", "onnx", "gguf", "diffusion", "stable-diffusion"},
        "tokens": {"ollama", "llama", "mistral", "inference", "ggml", "gguf", "diffusion", "stable-diffusion", "quantization", "quantize", "tensorrt", "triton", "onnx", "cuda", "openvino", "localai", "deepseek", "qwen", "gemma"},
        "phrases": {"模型推理", "推理引擎", "推理加速", "模型部署", "离线运行", "本地模型", "量化"},
        "regex": [re.compile(r"\b(inference|quantiz|ggml|gguf|llama\.cpp|ollama|mistral)\b", re.I)],
    },
    "ai_rag_agent": {
        "topics": {"rag", "agent", "agents", "mcp", "prompt", "langchain", "llamaindex", "graphrag"},
        "tokens": {"rag", "agent", "agents", "prompt", "tool-calling", "function-calling", "langchain", "llamaindex", "graphrag", "mcp", "openai", "claude", "anthropic"},
        "phrases": {"知识库", "检索增强", "检索增强生成", "向量检索", "提示词", "工具调用", "智能体", "代理"},
        "regex": [re.compile(r"\b(rag|retrieval[- ]augmented|agent|tool[- ]calling|prompt)\b", re.I)],
    },
    "ai_platform": {
        "topics": {"dify", "langflow", "flowise", "low-code", "no-code", "workflow-automation", "agentic-workflow", "agentic-ai"},
        "tokens": {"workflow", "orchestration", "studio", "visual", "builder", "low-code", "nocode", "deploy", "production", "agentic", "dify", "langflow", "flowise", "fastgpt"},
        "phrases": {"工作流", "编排", "低代码", "应用开发"},
        "regex": [re.compile(r"\b(workflow|orchestrat|low[- ]code|no[- ]code|visual)\b", re.I)],
    },
    "ai_apps": {
        "topics": set(),
        "tokens": {"assistant", "chat", "client", "webui", "ui", "desktop", "copilot", "tabby", "anything-llm", "nextchat", "open-webui", "jan"},
        "phrases": {"AI 助手", "聊天", "客户端", "桌面应用", "本地运行", "自托管", "知识库应用"},
        "regex": [re.compile(r"\b(ai assistant|chat(\s|-)?(ui|client)|self-hosted)\b", re.I)],
    },
    "vector_search": {
        "topics": {"vector-database", "vector-search", "embedding", "ann", "qdrant", "milvus", "faiss", "chroma", "pgvector"},
        "tokens": {"vector", "embedding", "ann", "faiss", "qdrant", "milvus", "chroma", "pgvector", "similarity", "semantic"},
        "phrases": {"向量数据库", "向量检索", "语义检索", "相似度检索", "嵌入"},
        "regex": [re.compile(r"\b(vector (database|search)|embedding|ann|nearest neighbor)\b", re.I)],
    },
    "data_analytics": {
        "topics": {"olap", "analytics", "big-data", "datawarehouse", "lakehouse", "clickhouse", "duckdb", "datafusion", "polars", "delta-lake", "mpp", "sql"},
        "tokens": {"olap", "analytics", "warehouse", "lakehouse", "query-engine", "query", "datafusion", "polars", "duckdb", "clickhouse", "trino", "presto", "spark", "delta", "mpp"},
        "phrases": {"数据仓库", "湖仓", "分析引擎", "查询引擎", "列式", "OLAP"},
        "regex": [re.compile(r"\b(olap|data warehouse|lakehouse|query engine|columnar)\b", re.I)],
    },
    "data_formats": {
        "topics": {"arrow", "parquet", "protobuf", "serialization", "parser"},
        "tokens": {"arrow", "parquet", "protobuf", "serde", "serialization", "serialize", "serializer", "deserialize", "deserializer", "json", "yaml", "toml", "csv", "bincode", "flatbuffers", "capnp", "capnproto", "parser", "lexer", "grammar", "format"},
        "phrases": {"序列化", "反序列化", "解析器", "语法", "格式", "编码", "协议"},
        "regex": [re.compile(r"\b(arrow|parquet|protobuf|serializ|deserializ|flatbuffers)\b", re.I)],
    },
    "db_systems": {
        "topics": {"database", "dbms", "distributed-systems", "graph-database", "timeseries", "time-series"},
        "tokens": {"database", "dbms", "postgres", "postgresql", "mysql", "redis", "serverless", "cluster", "distributed", "replication", "sharding", "timeseries", "time-series"},
        "phrases": {"数据库", "分布式数据库", "时序数据库", "图数据库", "存储系统"},
        "regex": [re.compile(r"\b(database|dbms|time series|graph database|distributed database)\b", re.I)],
    },
    "db_storage": {
        "topics": {"storage-engine", "embedded-database", "rocksdb", "lsm-tree", "key-value", "kv", "cache"},
        "tokens": {"rocksdb", "leveldb", "lmdb", "lsm", "lsm-tree", "btree", "b-tree", "sstable", "wal", "in-process", "storage-engine", "key-value", "kv", "wisckey", "titan", "pebble"},
        "phrases": {"存储引擎", "嵌入式数据库", "键值", "KV", "LSM"},
        "regex": [re.compile(r"\b(rocksdb|lsm|embedded (database|storage)|storage engine|key[- ]value)\b", re.I)],
    },
    "db_tooling": {
        "topics": {"sql", "orm", "database"},
        "tokens": {"sql", "orm", "migration", "schema", "driver", "connector", "client", "admin", "toolkit", "query-builder", "sqlglot", "sqlx", "sea-orm", "chat2db"},
        "phrases": {"SQL 解析", "数据库客户端", "数据库管理", "迁移", "ORM", "SQL 工具"},
        "regex": [re.compile(r"\b(sql (parser|transpil|client)|orm|migration|database client)\b", re.I)],
    },
    "messaging_stream": {
        "topics": {"streaming", "message-queue", "mq", "kafka", "nats", "rabbitmq", "pubsub"},
        "tokens": {"kafka", "nats", "rabbitmq", "mq", "message", "messaging", "queue", "stream", "streaming", "broker", "pubsub", "pulsar"},
        "phrases": {"消息队列", "消息系统", "流处理", "消息中间件", "事件流"},
        "regex": [re.compile(r"\b(message queue|stream processing|event streaming|pubsub)\b", re.I)],
    },
    "search_index": {
        "topics": {"search", "search-engine", "full-text", "lucene", "search-engine"},
        "tokens": {"search", "index", "lucene", "inverted", "meilisearch", "zincsearch", "tantivy", "full-text", "filter", "bloom", "xor"},
        "phrases": {"搜索", "检索", "索引", "全文检索"},
        "regex": [re.compile(r"\b(search engine|full[- ]text|inverted index)\b", re.I)],
    },
    "storage_fs": {
        "topics": {"s3", "object-storage", "object-store", "distributed-storage", "filesystem", "fuse", "posix", "hdfs", "ceph", "minio"},
        "tokens": {"s3", "object-storage", "objectstore", "object-store", "filesystem", "file-system", "fuse", "posix", "hdfs", "nfs", "ceph", "minio", "seaweedfs", "juicefs", "mountpoint"},
        "phrases": {"对象存储", "文件系统", "分布式存储", "云存储", "块存储"},
        "regex": [re.compile(r"\b(s3|object (store|storage)|file system|filesystem|fuse|ceph)\b", re.I)],
    },
    "cloud_devops": {
        "topics": {"kubernetes", "k8s", "container", "cloud-native", "helm", "terraform", "podman"},
        "tokens": {"docker", "kubernetes", "k8s", "container", "containers", "helm", "terraform", "podman", "colima", "ci", "cd", "devops", "cloud-native", "runtime", "image"},
        "phrases": {"容器", "Kubernetes", "K8s", "云原生", "部署", "镜像", "集群"},
        "regex": [re.compile(r"\b(docker|kubernetes|k8s|container runtime|helm|terraform|ci/cd)\b", re.I)],
    },
    "proxy_vpn": {
        "topics": {"proxy", "vpn", "clash", "shadowsocks", "v2ray", "wireguard"},
        "tokens": {"proxy", "vpn", "clash", "shadowsocks", "v2ray", "wireguard", "socks", "tun", "tunnel"},
        "phrases": {"代理", "VPN", "科学上网", "翻墙", "代理客户端", "隧道"},
        "regex": [re.compile(r"\b(proxy|vpn|clash|wireguard|shadowsocks|v2ray)\b", re.I)],
    },
    "network_protocols": {
        "topics": {"networking", "quic", "grpc", "websocket", "http3", "http2", "tcp", "udp", "dns", "rdma", "kcp"},
        "tokens": {"network", "networking", "protocol", "service", "quic", "http3", "http2", "grpc", "websocket", "tcp", "udp", "dns", "rdma", "kcp", "tls", "socket", "ipstack", "http"},
        "phrases": {"网络协议", "传输协议", "网络栈", "TCP/IP", "RDMA"},
        "regex": [re.compile(r"\b(quic|http/3|grpc|websocket|tcp/ip|rdma|kcp)\b", re.I)],
    },
    "observability_perf": {
        "topics": {"observability", "monitoring", "logging", "metrics", "tracing", "benchmark", "profiling", "pprof"},
        "tokens": {"observability", "monitoring", "metrics", "logging", "tracing", "opentelemetry", "otel", "pprof", "flamegraph", "profiler", "profiling", "benchmark", "benchmarking", "wrk", "perf", "valgrind", "debugger", "debug"},
        "phrases": {"可观测性", "监控", "指标", "日志", "链路追踪", "性能分析", "压测", "基准测试"},
        "regex": [re.compile(r"\b(observability|monitoring|metrics|logging|tracing|pprof|flamegraph|benchmark)\b", re.I)],
    },
    "security": {
        "topics": {"security", "pentesting", "hacking", "security-tools", "packet-sniffer"},
        "tokens": {"security", "pentest", "pentesting", "hacking", "scanner", "scan", "nmap", "packet", "sniffer", "traffic", "vulnerability"},
        "phrases": {"安全", "端口扫描", "抓包", "漏洞", "渗透"},
        "regex": [re.compile(r"\b(port scanner|pentest|packet sniffer|vulnerability)\b", re.I)],
    },
    "systems_os": {
        "topics": {"kernel", "os", "linux-kernel", "virtualization", "vm", "wsl", "hypervisor", "libos"},
        "tokens": {"kernel", "os", "linux-kernel", "wsl", "virtualization", "virtual-machine", "vm", "hypervisor", "libos"},
        "phrases": {"内核", "操作系统", "虚拟机", "虚拟化", "子系统", "LibOS"},
        "regex": [re.compile(r"\b(kernel|operating system|virtual machine|hypervisor|wsl|libos)\b", re.I)],
    },
    "systems_libs": {
        "topics": {"coroutines", "io-uring"},
        "tokens": {"library", "crate", "folly", "abseil", "boost", "stl", "hashmap", "btree", "b-tree", "allocator", "jemalloc", "mimalloc", "coroutine", "coroutines", "concurrency", "concurrent", "async", "runtime", "io-uring", "lock-free", "lockfree", "simd", "datastructure", "data-structure", "error", "diagnostic", "bytes", "string", "low-latency"},
        "phrases": {"高性能", "无锁", "协程", "并发", "数据结构", "内存分配", "io_uring", "错误处理", "诊断"},
        "regex": [re.compile(r"\b(hashmap|b-?tree|coroutine|concurren|io[-_ ]uring|allocator|jemalloc|mimalloc|std::)\b", re.I)],
    },
    "lang_tooling": {
        "topics": {"parser", "compiler", "interpreter"},
        "tokens": {"compiler", "interpreter", "linter", "formatter", "rustpython", "cranelift", "llvm", "miri", "rustc", "parser", "lexer", "transpiler"},
        "phrases": {"编译器", "解释器", "语言实现", "静态分析", "格式化", "语法解析"},
        "regex": [re.compile(r"\b(compiler|interpreter|linter|formatter|llvm|cranelift|rustpython)\b", re.I)],
    },
    "build_release": {
        "topics": {"build", "package-manager", "cargo", "pnpm", "npm", "yarn"},
        "tokens": {"build", "package-manager", "dependency", "cargo", "cmake", "bazel", "make", "ninja", "pnpm", "npm", "yarn", "pip", "rye", "pixi", "uv", "cross", "release"},
        "phrases": {"构建", "包管理", "依赖", "发布", "交叉编译"},
        "regex": [re.compile(r"\b(package manager|dependency|cross[- ]compile|build system)\b", re.I)],
    },
    "editor_ide": {
        "topics": {"editor", "ide", "vscode", "neovim", "vim", "emacs", "helix", "zed"},
        "tokens": {"editor", "ide", "vscode", "neovim", "vim", "emacs", "helix", "zed", "lsp", "language-server", "treesitter", "tree-sitter"},
        "phrases": {"编辑器", "IDE", "插件", "LSP"},
        "regex": [re.compile(r"\b(editor|ide|vscode|neovim|helix|zed)\b", re.I)],
    },
    "cli_terminal": {
        "topics": {"cli", "terminal", "shell", "tui"},
        "tokens": {"cli", "terminal", "shell", "tui", "command-line", "prompt", "bash", "zsh", "fish", "nushell", "atuin", "gitui", "ripgrep", "fd", "bat", "tokei"},
        "phrases": {"命令行", "终端", "Shell", "TUI"},
        "regex": [re.compile(r"\b(cli|command[- ]line|terminal|tui|shell)\b", re.I)],
    },
    "ui_frameworks": {
        "topics": {"tauri", "flutter", "react", "svelte", "solidjs", "wasm", "webassembly", "avalonia", "egui", "iced"},
        "tokens": {"tauri", "flutter", "electron", "avalonia", "iced", "egui", "dioxus", "leptos", "yew", "react", "svelte", "solid", "solidjs", "wasm", "webassembly", "component", "components", "ui-framework", "gui"},
        "phrases": {"UI 框架", "GUI 框架", "组件库", "前端框架", "桌面应用框架"},
        "regex": [re.compile(r"\b(ui framework|gui framework|component library)\b", re.I)],
    },
    "apps_productivity": {
        "topics": set(),
        "tokens": {"pdf", "translator", "translation", "ocr", "resume", "filebrowser", "cleaner", "optimizer", "screenshot", "launcher", "clipboard", "desktop-environment"},
        "phrases": {"翻译", "OCR", "PDF", "文件管理", "清理", "垃圾", "隐私", "空间", "内存", "优化", "截图", "桌面环境", "效率工具"},
        "regex": [re.compile(r"\b(pdf|ocr|translation|resume|file browser|windows optimizer|desktop environment)\b", re.I)],
    },
    "media": {
        "topics": {"iptv", "music", "video", "streaming"},
        "tokens": {"iptv", "music", "video", "youtube", "bilibili", "streaming"},
        "phrases": {"IPTV", "音乐", "视频", "直播", "播放器"},
        "regex": [re.compile(r"\b(iptv|music|video|youtube|streaming)\b", re.I)],
    },
    "learning": {
        "topics": {"awesome", "tutorial", "book", "course", "learning", "papers"},
        "tokens": {"tutorial", "book", "course", "learning", "awesome", "papers", "reference", "handbook", "guide", "beginners", "beginner", "examples", "cheatsheet", "list", "collection"},
        "phrases": {"教程", "课程", "书", "学习", "指南", "参考", "清单", "资料", "入门", "英语"},
        "regex": [re.compile(r"\b(tutorial|course|book|papers|for beginners|curated list|collection of)\b", re.I)],
    },
}


_PRIORITY = {c.key: i for i, c in enumerate(CATEGORIES)}


def classify_repo(*, name: str, language: str | None, topics: list[str], description: str | None) -> dict:
    # Language is intentionally not used as a primary signal:
    # it tends to distort domain classification (e.g. many Go projects are not "build tools").
    text = f"{name or ''} {description or ''}".strip().lower()
    tokens = _tokenize(text)
    topics_l = {t.lower() for t in (topics or [])}

    score = {c.key: 0 for c in CATEGORIES}

    for cat_key, rule in RULES.items():
        score[cat_key] += 8 * len(topics_l.intersection(rule.get("topics", set())))
        score[cat_key] += 3 * len(tokens.intersection(rule.get("tokens", set())))
        for p in rule.get("phrases", set()):
            if _phrase_match(phrase=p, text=text, tokens=tokens):
                score[cat_key] += 3
        for rx in rule.get("regex", []):
            if rx.search(text):
                score[cat_key] += 4

    # Cross-category compound boosts (improves precision for common overlaps)
    ai_indicators = {"ai", "llm", "llms", "chatgpt", "gpt", "openai", "ollama", "agent", "agents", "agentic", "rag"}
    ai_app_indicators = {"assistant", "chat", "client", "webui", "desktop", "ui", "gui", "copilot"}
    # Prefer description/name signals over topics: some non-AI tools may tag ChatGPT etc. for marketing/integration.
    ai_app_topic_indicators = {"webui", "ui", "desktop", "tauri", "self-hosted", "client", "chat", "chatbot", "assistant"}
    ai_boost_indicators = (tokens & ai_indicators) or (topics_l & {"ai", "llm", "llms", "ollama", "openai", "stable-diffusion"})
    if ai_boost_indicators and ((tokens & ai_app_indicators) or (topics_l & ai_app_topic_indicators)):
        score["ai_apps"] += 14
        # If it looks like an end-user UI/client, down-weight inference category even when topics mention runtimes.
        score["ai_inference"] = int(score["ai_inference"] * 0.35)
        score["ai_rag_agent"] = int(score["ai_rag_agent"] * 0.75)

    # Learning-first hint: courses/books/tutorials should not be dominated by AI keywords.
    learning_indicators = {"tutorial", "course", "courses", "lesson", "lessons", "beginners", "beginner", "handbook", "reference", "book", "books", "papers"}
    if (tokens & learning_indicators) or ("教程" in text) or ("课程" in text) or ("入门" in text):
        score["learning"] += 12
        score["ai_inference"] = int(score["ai_inference"] * 0.6)
        score["ai_apps"] = int(score["ai_apps"] * 0.7)

    ai_platform_indicators = {"platform", "workflow", "orchestration", "studio", "visual", "builder", "low-code", "nocode", "agentic"}
    ai_platform_names = {"dify", "langflow", "flowise", "fastgpt"}
    if tokens & ai_platform_names:
        score["ai_platform"] += 20
    if (tokens & ai_indicators or topics_l & ai_indicators) and (tokens & ai_platform_indicators or ("工作流" in text) or ("编排" in text) or ("低代码" in text)):
        score["ai_platform"] += 12
    if tokens & ai_platform_indicators:
        score["ai_rag_agent"] = int(score["ai_rag_agent"] * 0.75)

    db_indicators = {"database", "dbms", "sql", "postgres", "postgresql", "mysql"}
    db_tool_indicators = {"orm", "migration", "client", "driver", "connector", "parser", "transpiler", "admin", "tool", "toolkit", "sqlx", "sea-orm", "sqlglot"}
    if (tokens & db_indicators or topics_l & db_indicators or ("数据库" in text)) and (tokens & db_tool_indicators):
        score["db_tooling"] += 8
        score["db_systems"] = int(score["db_systems"] * 0.65)

    # Embedded / KV databases should prefer db_storage even if other DB signals exist.
    embedded_kv_topics = {"embedded-kv", "kv", "key-value", "embedded-database", "storage-engine", "lsm-tree", "rocksdb"}
    embedded_kv_tokens = {"key-value", "kv", "lsm", "rocksdb"}
    if ("embedded" in tokens or "in-process" in tokens) and ("database" in tokens or "databases" in tokens or "db" in tokens) and (
        (tokens & embedded_kv_tokens) or (topics_l & embedded_kv_topics) or ("嵌入式数据库" in text)
    ):
        score["db_storage"] += 12
        score["db_systems"] = int(score["db_systems"] * 0.6)
        score["db_tooling"] = int(score["db_tooling"] * 0.6)

    # Vector databases should prefer vector_search over generic db_systems.
    if (topics_l & {"vector-database", "vector-search"}) or (("vector" in tokens) and ("database" in tokens)):
        score["vector_search"] += 12
        score["db_systems"] = int(score["db_systems"] * 0.7)

    # Avoid classifying UI apps as frameworks
    if "ui_frameworks" in score and ("application" in tokens or "app" in tokens or "客户端" in text or "桌面应用" in text):
        score["ui_frameworks"] = int(score["ui_frameworks"] * 0.75)

    max_score = max(score.values()) if score else 0
    if max_score <= 0:
        # Fallback heuristics (best-effort, avoid defaulting to the first category)
        if {"tutorial", "course", "book", "papers", "reference", "algorithm", "algorithms"} & tokens:
            best_key = "learning"
        elif {"kernel", "os", "wsl", "vm", "hypervisor", "libos"} & tokens or ("操作系统" in text) or ("内核" in text):
            best_key = "systems_os"
        elif {"benchmark", "bench", "profil", "perf", "fio", "ior"} & tokens or ("基准" in text) or ("压测" in text):
            best_key = "observability_perf"
        elif {"iptv", "music", "video"} & tokens or ("音乐" in text) or ("视频" in text):
            best_key = "media"
        elif language in {"Rust", "C", "C++", "Zig"} or "library" in tokens or "crate" in tokens:
            best_key = "systems_libs"
        else:
            best_key = "apps_productivity"
    else:
        best_key = max(score.keys(), key=lambda k: (score[k], -_PRIORITY.get(k, 10_000), k))
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

    # Starred repos (all-time) + 2024 slice
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

    stars_2024 = [s for s in stars if s["starredAt"].startswith(f"{YEAR}-")]
    stars_before = [s for s in stars if s["starredAt"] < f"{YEAR}-01-01T00:00:00Z"]

    stars_by_year = defaultdict(int)
    for s in stars:
        year = s["starredAt"][:4]
        stars_by_year[year] += 1

    star_month_counts = Counter([s["starredAt"][:7] for s in stars_2024])
    star_month_top_repo = {}
    for s in stars_2024:
        key = s["starredAt"][:7]
        prev = star_month_top_repo.get(key)
        if not prev or s["stargazerCount"] > prev["stargazerCount"]:
            star_month_top_repo[key] = s

    star_month_repos = defaultdict(list)
    for s in stars_2024:
        star_month_repos[s["starredAt"][:7]].append(s)
    for month_key in star_month_repos:
        star_month_repos[month_key].sort(key=lambda x: (x.get("stargazerCount", 0), x.get("starredAt") or ""), reverse=True)

    # Star events by month (for timeline charts)
    star_month_events = defaultdict(list)
    for s in stars_2024:
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
    for s in stars_2024:
        hour = to_local(s["starredAt"]).hour
        star_hour_local[hour] += 1

    star_lang_2024 = Counter([s["primaryLanguage"] for s in stars_2024 if s.get("primaryLanguage")])
    star_topic_2024 = Counter()
    for s in stars_2024:
        for t in s.get("topics", []):
            star_topic_2024[t.lower()] += 1

    star_topic_before = Counter()
    for s in stars_before:
        for t in s.get("topics", []):
            star_topic_before[t.lower()] += 1

    new_topics = []
    rising_topics = []
    for topic, cur in star_topic_2024.most_common():
        prev = star_topic_before.get(topic, 0)
        if prev == 0 and cur >= 3:
            new_topics.append({"topic": topic, "count2024": cur, "countBefore": 0})
        elif prev > 0 and cur >= max(5, prev * 2):
            rising_topics.append(
                {
                    "topic": topic,
                    "count2024": cur,
                    "countBefore": prev,
                    "ratio": round(cur / prev, 2) if prev else None,
                }
            )

    new_topics = new_topics[:12]
    rising_topics = sorted(rising_topics, key=lambda x: (-(x["ratio"] or 0), -x["count2024"]))[:12]

    lang_before = Counter([s["primaryLanguage"] for s in stars_before if s.get("primaryLanguage")])
    new_langs = []
    for lang, cur in star_lang_2024.most_common():
        prev = lang_before.get(lang, 0)
        if prev == 0 and cur >= 2:
            new_langs.append({"language": lang, "count2024": cur, "countBefore": 0})
    new_langs = new_langs[:8]

    first_star_ever = min(stars, key=lambda s: s["starredAt"]) if stars else None
    first_star_2024 = min(stars_2024, key=lambda s: s["starredAt"]) if stars_2024 else None
    latest_star_2024 = max(stars_2024, key=lambda s: s["starredAt"]) if stars_2024 else None

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

    # Merged PRs (2024)
    pr_pages = load(RAW / "prs_2024_pages.json")
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

    # Holiday stars (Chinese-focused, 2024)
    spring_festival = dt.date(2024, 2, 10)
    chuxi = spring_festival - dt.timedelta(days=1)
    holidays = [
        ("new_year", "元旦", dt.date(2024, 1, 1)),
        ("chuxi", "除夕", chuxi),
        ("spring_festival", "春节", spring_festival),
        ("qingming", "清明节", dt.date(2024, 4, 4)),
        ("labor_day", "劳动节", dt.date(2024, 5, 1)),
        ("dragon_boat", "端午节", dt.date(2024, 6, 10)),
        ("qixi", "七夕", dt.date(2024, 8, 10)),
        ("national_day", "国庆节", dt.date(2024, 10, 1)),
        ("mid_autumn", "中秋节", dt.date(2024, 9, 17)),
        ("programmer_day", "程序员节", dt.date(2024, 10, 24)),
        ("singles_day", "双十一", dt.date(2024, 11, 11)),
        ("new_year_eve", "跨年夜(12/31)", dt.date(2024, 12, 31)),
    ]

    holiday_cards = []
    for key, label, date in holidays:
        repos = [s for s in stars_2024 if s["starredAt"][:10] == date.isoformat()]
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

    # Category stats based on 2024 stars
    stars_2024_with_cat = []
    category_counts = Counter()
    for s in stars_2024:
        cat = classify_repo(
            name=s["nameWithOwner"],
            language=s.get("primaryLanguage"),
            topics=s.get("topics", []),
            description=s.get("description"),
        )
        stars_2024_with_cat.append({**s, "category": cat})
        category_counts[cat["primary"]] += 1

    category_top_repos = defaultdict(list)
    for s in stars_2024_with_cat:
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
    top_topics = [t for t, _ in star_topic_2024.most_common(8)]

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

    # Meet GitHub duration (as of 2024-12-31)
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
                    "仓库收到的 Stars/Forks 为当前快照，不代表 2024 新增。",
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
                "total2024": len(stars_2024),
                "totalAllTime": len(stars),
                "byYear": dict(sorted(stars_by_year.items())),
                "byMonth2024": [
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
                "byHourLocal2024": star_hour_local,
                "topLanguages2024": [{"name": k, "count": v} for k, v in star_lang_2024.most_common(12)],
                "topTopics2024": [{"name": k, "count": v} for k, v in star_topic_2024.most_common(24)],
                "topStarredRepos2024": [
                    {
                        "nameWithOwner": s["nameWithOwner"],
                        "stars": s["stargazerCount"],
                        "language": s.get("primaryLanguage"),
                        "starredAt": s["starredAt"],
                        "url": s["url"],
                    }
                    for s in sorted(stars_2024, key=lambda s: s["stargazerCount"], reverse=True)[:20]
                ],
                "firstStar2024": {
                    "starredAt": first_star_2024["starredAt"] if first_star_2024 else None,
                    "repo": first_star_2024["nameWithOwner"] if first_star_2024 else None,
                    "url": first_star_2024.get("url") if first_star_2024 else None,
                },
                "latestStar2024": {
                    "starredAt": latest_star_2024["starredAt"] if latest_star_2024 else None,
                    "repo": latest_star_2024["nameWithOwner"] if latest_star_2024 else None,
                    "url": latest_star_2024.get("url") if latest_star_2024 else None,
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
                "starCounts2024": {k: category_counts.get(k, 0) for k in CATEGORY_BY_KEY.keys()},
                "topRepos2024": {
                    k: [
                        {
                            "nameWithOwner": s["nameWithOwner"],
                            "stars": s["stargazerCount"],
                            "language": s.get("primaryLanguage"),
                            "url": s["url"],
                        }
                        for s in category_top_repos.get(k, [])
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
