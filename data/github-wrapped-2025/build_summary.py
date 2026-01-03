import json
import datetime as dt
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('data/github-wrapped-2025')
RAW = BASE / 'raw'
OUT = BASE / 'processed'
OUT.mkdir(parents=True, exist_ok=True)


def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

user = load(RAW / 'user.json')
login = user.get('login')

contrib = load(RAW / 'contributions.json')
cc = contrib['data']['user']['contributionsCollection']
calendar = cc['contributionCalendar']
weeks = calendar['weeks']

days = []
for week in weeks:
    for day in week['contributionDays']:
        days.append({
            'date': day['date'],
            'count': day['contributionCount'],
            'weekday': day['weekday'],
        })

# Sort days

days_sorted = sorted(days, key=lambda d: d['date'])

# Busiest day
max_count = max((d['count'] for d in days_sorted), default=0)
busiest_days = [d for d in days_sorted if d['count'] == max_count and max_count > 0]
busiest_day = busiest_days[0] if busiest_days else {'date': None, 'count': 0}

# Longest streak
max_streak = 0
max_start = None
max_end = None
current = 0
current_start = None
prev_date = None

for d in days_sorted:
    date = dt.date.fromisoformat(d['date'])
    if d['count'] > 0:
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

# Final streak check
if current > max_streak:
    max_streak = current
    max_start = current_start
    max_end = prev_date

# Weekday distribution
weekday_counts = [0] * 7
for d in days_sorted:
    weekday_counts[d['weekday']] += d['count']

# Monthly distribution
month_counts = defaultdict(int)
for d in days_sorted:
    month = d['date'][:7]
    month_counts[month] += d['count']
months_sorted = sorted(month_counts.items())

# Contribution repos (top)
commit_repos = cc.get('commitContributionsByRepository', [])
pr_repos = cc.get('pullRequestContributionsByRepository', [])
issue_repos = cc.get('issueContributionsByRepository', [])

# Contributed repos (external)
contrib_pages = load(RAW / 'contributed_repos_pages.json')
contrib_nodes = []
for page in contrib_pages:
    nodes = page['data']['user']['repositoriesContributedTo']['nodes']
    contrib_nodes.extend(nodes)

external_contrib = [r for r in contrib_nodes if r.get('owner', {}).get('login') and r['owner']['login'].lower() != login.lower()]

# PRs in 2025 (merged)
pr_pages = load(RAW / 'prs_2025_pages.json')
pr_nodes = []
for page in pr_pages:
    nodes = page['data']['search']['nodes']
    pr_nodes.extend([n for n in nodes if n])

# Compute PR totals
pr_total = len(pr_nodes)
pr_additions = sum(pr.get('additions', 0) for pr in pr_nodes)
pr_deletions = sum(pr.get('deletions', 0) for pr in pr_nodes)

# External PRs (outside own repos)
external_prs = [pr for pr in pr_nodes if pr.get('repository', {}).get('owner', {}).get('login', '').lower() != login.lower()]

# User repos
user_repos = load(RAW / 'user_repos.json')

# Starred repos
star_pages = load(RAW / 'starred_repos_pages.json')
star_edges = []
for page in star_pages:
    edges = page['data']['user']['starredRepositories']['edges']
    star_edges.extend(edges)

# Flatten stars
stars = []
for edge in star_edges:
    node = edge['node']
    stars.append({
        'starredAt': edge['starredAt'],
        'nameWithOwner': node['nameWithOwner'],
        'description': node.get('description'),
        'stargazerCount': node.get('stargazerCount', 0),
        'forkCount': node.get('forkCount', 0),
        'primaryLanguage': (node.get('primaryLanguage') or {}).get('name'),
        'topics': [t['topic']['name'] for t in node.get('repositoryTopics', {}).get('nodes', [])],
        'owner': (node.get('owner') or {}).get('login'),
    })

stars_2025 = [s for s in stars if s['starredAt'].startswith('2025-')]

# Star languages/topics counts (2025 only)
star_lang_counts = Counter([s['primaryLanguage'] for s in stars_2025 if s['primaryLanguage']])

star_topic_counts = Counter()
for s in stars_2025:
    for t in s['topics']:
        star_topic_counts[t.lower()] += 1

# Fallback to all stars if 2025 topics too small
if sum(star_topic_counts.values()) < 5:
    star_topic_counts = Counter()
    for s in stars:
        for t in s['topics']:
            star_topic_counts[t.lower()] += 1

# User primary languages (own repos)
repo_lang_counts = Counter()
for r in user_repos:
    lang = r.get('language')
    if lang:
        repo_lang_counts[lang] += 1

# Top repos by stars (own repos)
own_repos = [r for r in user_repos if not r.get('fork')]
own_repos_sorted = sorted(own_repos, key=lambda r: r.get('stargazers_count', 0), reverse=True)
own_stars_total = sum(r.get('stargazers_count', 0) for r in own_repos)
own_forks_total = sum(r.get('forks_count', 0) for r in own_repos)
created_in_year = sum(1 for r in own_repos if (r.get('created_at') or '').startswith('2025-'))

# Determine profile
primary_languages = [lang for lang, _ in repo_lang_counts.most_common(5)]
interest_topics = [topic for topic, _ in star_topic_counts.most_common(10)]

identity = "开发者"
if 'rust' in interest_topics and ('database' in interest_topics or 'db' in interest_topics):
    identity = "Rust 数据/系统工程师"
elif 'rust' in interest_topics:
    identity = "Rust 爱好者"
elif 'llm' in interest_topics or 'ai' in interest_topics:
    identity = "AI/LLM 探索者"
elif primary_languages:
    identity = f"{primary_languages[0]} 开发者"

# Coding pattern
weekday_total = sum(weekday_counts)
weekend_total = weekday_counts[0] + weekday_counts[6]
pattern = "均衡型"
if weekday_total > 0:
    weekend_ratio = weekend_total / weekday_total
    if weekend_ratio >= 0.4:
        pattern = "周末战士"
    elif weekend_ratio <= 0.2:
        pattern = "工作日重度"

# Year story
most_active_month = max(month_counts.items(), key=lambda kv: kv[1])[0] if month_counts else None

# Special holiday stars (dates will be filled later by separate step)

summary = {
    'user': {
        'login': login,
        'name': user.get('name'),
        'avatar_url': user.get('avatar_url'),
        'followers': user.get('followers'),
        'following': user.get('following'),
    },
    'year': 2025,
    'totals': {
        'totalContributions': calendar.get('totalContributions', 0),
        'commits': cc.get('totalCommitContributions', 0),
        'prs': cc.get('totalPullRequestContributions', 0),
        'issues': cc.get('totalIssueContributions', 0),
        'repos': cc.get('totalRepositoryContributions', 0),
    },
    'busiestDay': {
        'date': busiest_day['date'],
        'count': busiest_day['count'],
    },
    'longestStreak': {
        'count': max_streak,
        'start': max_start.isoformat() if max_start else None,
        'end': max_end.isoformat() if max_end else None,
    },
    'calendar': {
        'weeks': weeks,
    },
    'weekdayCounts': weekday_counts,
    'monthlyCounts': [{'month': m, 'count': c} for m, c in months_sorted],
    'topCommitRepos': [
        {
            'nameWithOwner': r['repository']['nameWithOwner'],
            'count': r['contributions']['totalCount'],
        }
        for r in sorted(commit_repos, key=lambda r: r['contributions']['totalCount'], reverse=True)[:8]
    ],
    'topPrRepos': [
        {
            'nameWithOwner': r['repository']['nameWithOwner'],
            'count': r['contributions']['totalCount'],
        }
        for r in sorted(pr_repos, key=lambda r: r['contributions']['totalCount'], reverse=True)[:8]
    ],
    'topIssueRepos': [
        {
            'nameWithOwner': r['repository']['nameWithOwner'],
            'count': r['contributions']['totalCount'],
        }
        for r in sorted(issue_repos, key=lambda r: r['contributions']['totalCount'], reverse=True)[:8]
    ],
    'openSource': {
        'contributedReposTotal': len(contrib_nodes),
        'externalReposTotal': len(external_contrib),
        'externalReposSample': [
            {
                'nameWithOwner': r['nameWithOwner'],
                'stars': r.get('stargazerCount', 0),
                'language': (r.get('primaryLanguage') or {}).get('name'),
            }
            for r in sorted(external_contrib, key=lambda r: r.get('stargazerCount', 0), reverse=True)[:6]
        ],
    },
    'prs2025': {
        'total': pr_total,
        'additions': pr_additions,
        'deletions': pr_deletions,
        'externalTotal': len(external_prs),
        'highlights': [
            {
                'title': pr['title'],
                'url': pr['url'],
                'repo': pr['repository']['nameWithOwner'],
                'additions': pr.get('additions', 0),
                'deletions': pr.get('deletions', 0),
                'createdAt': pr.get('createdAt'),
                'mergedAt': pr.get('mergedAt'),
            }
            for pr in sorted(pr_nodes, key=lambda p: (p.get('additions', 0) + p.get('deletions', 0)), reverse=True)[:5]
        ],
    },
    'stars': {
        'totalAllTime': len(stars),
        'total2025': len(stars_2025),
        'languages': [
            {'name': lang, 'count': count}
            for lang, count in star_lang_counts.most_common(8)
        ],
        'topics': [
            {'name': topic, 'count': count}
            for topic, count in star_topic_counts.most_common(20)
        ],
        'topStarred2025': [
            {
                'nameWithOwner': s['nameWithOwner'],
                'stars': s['stargazerCount'],
                'language': s['primaryLanguage'],
                'starredAt': s['starredAt'],
            }
            for s in sorted(stars_2025, key=lambda s: s['stargazerCount'], reverse=True)[:8]
        ],
        'stars2025Sample': [
            {
                'nameWithOwner': s['nameWithOwner'],
                'stars': s['stargazerCount'],
                'language': s['primaryLanguage'],
                'starredAt': s['starredAt'],
                'topics': s['topics'][:5],
            }
            for s in stars_2025[:50]
        ],
    },
    'repos': {
        'total': len(user_repos),
        'ownTotal': len(own_repos),
        'ownStarsTotal': own_stars_total,
        'ownForksTotal': own_forks_total,
        'createdInYear': created_in_year,
        'top': [
            {
                'name': r['name'],
                'stars': r.get('stargazers_count', 0),
                'forks': r.get('forks_count', 0),
                'language': r.get('language'),
                'created_at': r.get('created_at'),
            }
            for r in own_repos_sorted[:8]
        ],
    },
    'profile': {
        'identity': identity,
        'primary_languages': primary_languages,
        'interests': interest_topics[:8],
        'pattern': pattern,
        'most_active_month': most_active_month,
    }
}

with open(OUT / 'summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print('summary written to', OUT / 'summary.json')
