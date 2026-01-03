# GitHub Wrapped 2025 — 数据来源与可核验性

本项目的年度报告**不伪造数据**：页面里展示的所有数字都来自 GitHub API 的原始响应（通过 `gh api` 获取），并已保存到 `data/github-wrapped-2025/raw/` 目录中，方便复核。

## 1) 前置条件

- `gh` 已登录：`gh auth status`
- 仅使用 GitHub API（GraphQL/REST）可公开访问的数据。

## 2) 原始数据文件（仓库内保存）

> 你可以直接打开这些 JSON 文件核对页面显示的数字。

- `data/github-wrapped-2025/raw/user.json`
  - 来源：`gh api user`
  - 用途：用户名、头像、followers/following、账号创建时间（用于“相遇多久”）

- `data/github-wrapped-2025/raw/contributions.json`
  - 来源：GraphQL `contributionsCollection`（`from=2025-01-01` 到 `to=2025-12-31`）
  - 用途：
    - 2025 年总贡献：`contributionCalendar.totalContributions`
    - commits / PRs / issues / reposContributedTo：`total*Contributions`
    - 全年热力图：`contributionCalendar.weeks[].contributionDays[]`
    - Top 仓库（按 commit/PR/issue 贡献）：`*ContributionsByRepository`

- `data/github-wrapped-2025/raw/user_repos.json`
  - 来源：REST `users/<user>/repos?per_page=100 --paginate`
  - 用途：
    - 你自己的仓库数量、语言分布（按仓库数）
    - 2025 年新建仓库数（按 `created_at` 年份过滤）
    - **注意**：仓库当前 `stargazers_count/forks_count` 是“快照”，不等同于 2025 年新增

- `data/github-wrapped-2025/raw/starred_repos_pages.json`
  - 来源：GraphQL `starredRepositories(orderBy: STARRED_AT, direction: DESC)` + `--paginate --slurp`
  - 用途：
    - 2025 年新增 Star 数量（按 `starredAt` 年份过滤）
    - Star 的 topics/语言偏好
    - “节日当天 Star 了哪些项目”（按 `starredAt` 日期匹配）
    - “解锁新兴趣”（对比 2025 之前 vs 2025 当年 topics 计数）

- `data/github-wrapped-2025/raw/prs_2025_pages.json`
  - 来源：GraphQL Search（示例 query：`author:<user> is:pr is:merged merged:2025-01-01..2025-12-31`）+ `--paginate --slurp`
  - 用途：
    - 2025 年合并 PR 数量
    - PR 代码变更行：`additions/deletions`
    - “年度开源奖”（按合并 PR 数 + 代码变更行）

- `data/github-wrapped-2025/raw/contributed_repos_pages.json`
  - 来源：GraphQL `repositoriesContributedTo(...)` + `--paginate --slurp`
  - 用途：你参与贡献过的外部仓库列表（用于“外部贡献 Top”展示）

- `data/github-wrapped-2025/raw/events_90d.json`
  - 来源：REST `users/<user>/events?per_page=100 --paginate`
  - 用途：仅作为“深夜彩蛋”（GitHub 事件 API 只保留近 90 天历史）

## 3) 统计结果（可复现）

- `data/github-wrapped-2025/processed/dataset.json`
  - 由脚本 `data/github-wrapped-2025/build_dataset.py` 从 `raw/` 计算得到
  - 页面 `frontend/standalone/github-wrapped-2025.html` 会把 `dataset.json` **内嵌**进单文件 HTML
  - 重新内嵌脚本：`data/github-wrapped-2025/embed_dataset_into_html.py`

## 4) 已知限制（不做伪造）

- “某一天具体做了什么（精确到仓库/commit 列表）”：公开贡献日历只给日级计数，不提供公开的仓库级明细；Events API 也只有近 90 天。
- “你的仓库在 2025 年新增了多少 stars/forks”：GitHub API 默认只提供当前快照；若要精确到年份需要额外抓取 stargazers 时间线（本次未纳入）。
