# Dataset Schema (Renderer Contract)

Treat `processed/dataset.json` as a **public API** between your data pipeline and the single-file HTML.
Keep it stable, additive, and resilient to missing data.

This repo’s canonical builder is `scripts/build_dataset.py`.

## Principles

- Prefer **additive changes** (add fields, keep old fields).
- Use `null` for optional fields instead of omitting them (simplifies rendering).
- Put caveats in `meta.dataProvenance.notes[]` and also show them in-page.

## Top-level

- `meta`
  - `year`: number
  - `timezone`: string (IANA, e.g. `Asia/Shanghai`)
  - `generatedAt`: ISO timestamp (UTC)
  - `dataProvenance`
    - `rawDir`: string (path to saved raw JSON)
    - `notes`: string[] (limitations / best-effort disclosures)
- `user`
  - `login`, `name`, `avatarUrl`, `createdAt`
  - `followers`, `following`
  - `meetGitHub`: `{ asOf, days, yearsApprox }`
  - `firstRepo`: `{ name, createdAt }`
  - `firstStarEver`: `{ starredAt, repo, url }`
  - `identity`: string (short human-friendly label; should be conservative)
- `year`
  - `totals`: `{ contributions, commits, prs, issues, reposContributedTo }`
  - `activity`
    - `totalDays`, `activeDays`, `inactiveDays`, `activeRate`
    - `busiestDay`: `{ date, count }`
    - `topDays`: list of `{ date, count, weekday }`
    - `longestStreak`: `{ count, start, end }`
    - `weekdaySums`: 7 ints (Sun..Sat as returned by GitHub)
    - `weekendRatio`: number
    - `patternLabel`: string
    - `mostActiveMonth`: `YYYY-MM` | null
    - `byMonth[]`: `{ month, contributions, activeDays }`
  - `repos`
    - `total`, `ownTotal`
    - `createdByYear`: map of `YYYY -> count`
    - `createdInYear`: number
    - `ownStarsTotalSnapshot`, `ownForksTotalSnapshot` (current snapshots)
    - `topSnapshot[]`: `{ name, stars, forks, language, createdAt, url }`
  - `contributions`
    - `calendarWeeks`: pass-through `contributionCalendar.weeks` (for heatmaps)
    - `topCommitRepos[]`, `topPrRepos[]`, `topIssueRepos[]`: `{ nameWithOwner, count }`
    - `focusProject`: `{ nameWithOwner, count }` | null
  - `stars`
    - **Stable keys (recommended for new HTML)**
      - `totalInYear`, `totalAllTime`
      - `byYear`: map of `YYYY -> count`
      - `reposInYear[]`: `{ nameWithOwner, stars, forks, language, starredAt, url, description, topics[] }`
      - `byMonthInYear[]`: `{ month, count, topRepo, repos[], events[] }`
      - `byHourLocalInYear`: 24 ints
      - `topLanguagesInYear[]`: `{ name, count }`
      - `topTopicsInYear[]`: `{ name, count }`
      - `topStarredReposInYear[]`: `{ nameWithOwner, stars, language, starredAt, url }`
      - `firstStarInYear`, `latestStarInYear`: `{ starredAt, repo, url }`
    - **Back-compat aliases (older year-specific renderers)**
      - `total2024`, `byMonth2024`, `topTopics2024`, ... (same payload as `*InYear`)
  - `discoveries`
    - `newTopics[]`, `risingTopics[]`, `newLanguages[]`
    - Each entry includes stable `countInYear` and also `count<YEAR>` for back-compat.
  - `categories`
    - `definitions[]`: `{ key, label, emoji, color, description }`
    - `starCountsInYear`: map `categoryKey -> count`
    - `topReposInYear`: map `categoryKey -> repo[]` (top N)
    - `radar`: map `categoryKey -> 0..100` (signal mix)
    - `primaryTrack`: string category key
    - `prCountsByCategory`, `prLinesByCategory`: maps
    - Back-compat: `starCounts<YEAR>`, `topRepos<YEAR>`
  - `openSource`
    - `mergedPrs`: `{ total, additions, deletions, lines }`
    - `ossAward`: `{ repo, count, lines, add, del }` | null
    - `biggestPr`, `latestPrCreated`: PR objects | null
    - `highlights[]`: subset of merged PRs
    - `externalContributedRepos[]`: list of `{ nameWithOwner, stars, language, topics[], owner }`
  - `specialDates`
    - `holidayStars[]`: `{ key, label, date, count, repos[] }`
    - `deepNightPush90d`: `{ localTime, repo, commitCount, sampleMessages[] }` | null

## Versioning (Recommended)

If you expect frequent evolution, add:

- `meta.schemaVersion` (e.g. `1`)

Then keep the renderer compatible with older schema versions.

