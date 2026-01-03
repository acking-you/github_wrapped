# Data Sources & Verifiability (Raw `gh api` JSON)

This project is built around a strict rule:
**everything displayed must be verifiable from saved GitHub API responses**.

## Prerequisites

- GitHub CLI authenticated: `gh auth status`
- Use only public GitHub APIs (GraphQL + REST) via `gh api`
- Always paginate when applicable

## Canonical Collector

Use `scripts/collect_raw.sh` (do not hand-roll ad-hoc queries).

It writes raw files to:

```
data/github-wrapped-$YEAR/raw/
  queries/   # snapshot of GraphQL queries used
  *.json     # raw API responses
```

## Required Raw Files

### 1) `raw/user.json`

Source:

```bash
gh api user
```

Used for:
- login, name, avatar
- followers / following
- `createdAt` (for “how long since we met GitHub”)

### 2) `raw/contributions.json`

Source: GraphQL `contributionsCollection(from,to)` for the selected year.

Used for:
- total contributions (calendar)
- commits / PRs / issues totals
- per-day heatmap data
- top repositories by commit/PR/issue contributions

### 3) `raw/starred_repos_pages.json`

Source: GraphQL `starredRepositories(orderBy: STARRED_AT)` with pagination:

- `--paginate --slurp`

Used for:
- total stars in the selected year (filter by `starredAt`)
- interests: topics / languages / descriptions
- holiday star easter eggs (date match on `starredAt`)
- “new interests” (compare year vs previous years)
- category inference (heuristics over topics/desc/lang)

### 4) `raw/prs_${YEAR}_pages.json`

Source: GraphQL search for merged PRs in the year:

```
author:<user> is:pr is:merged merged:<year>-01-01..<year>-12-31
```

Used for:
- merged PR count
- additions/deletions for “impact” storytelling
- external OSS award candidates (external repos only)

### 5) `raw/contributed_repos_pages.json`

Source: GraphQL `repositoriesContributedTo(...)` with pagination.

Used for:
- external contribution highlights

### 6) `raw/user_repos.json`

Source: REST `users/<user>/repos?per_page=100 --paginate`

Used for:
- your repositories list
- language distribution (by repo count)
- repos created in the year

Important limitation:
- `stargazers_count` and `forks_count` are **current snapshots**, not “stars gained this year”.

## Optional Raw Files

### 7) `raw/events_90d.json` (best-effort only)

Source: REST events API:

```bash
gh api users/<user>/events?per_page=100 --paginate
```

Used for:
- “night owl” easter eggs

Hard limit:
- GitHub keeps only ~90 days of events.

## Known Data Limits (Must Disclose In The Page)

- Contribution calendar is day-level totals only; you cannot reconstruct “exactly what happened” per repo/commit.
- Events API is ~90 days; do not pretend it is a full-year log.
- Stars/forks on repos are snapshots, not year deltas.
- Private contributions and hidden settings can skew the picture.

