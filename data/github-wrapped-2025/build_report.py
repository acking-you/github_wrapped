import json
import datetime as dt
from collections import Counter
from pathlib import Path

BASE = Path('data/github-wrapped-2025')
RAW = BASE / 'raw'
PROCESSED = BASE / 'processed'


def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

summary = load(PROCESSED / 'summary.json')
login = summary['user']['login']

# Load stars from raw pages
star_pages = load(RAW / 'starred_repos_pages.json')
star_edges = []
for page in star_pages:
    edges = page['data']['user']['starredRepositories']['edges']
    star_edges.extend(edges)

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
        'url': f"https://github.com/{node['nameWithOwner']}",
    })

stars_2025 = [s for s in stars if s['starredAt'].startswith('2025-')]

# Topic counts (2025)
star_topic_counts = Counter()
for s in stars_2025:
    for t in s['topics']:
        star_topic_counts[t.lower()] += 1

# If not enough, fallback to all-time
if sum(star_topic_counts.values()) < 5:
    star_topic_counts = Counter()
    for s in stars:
        for t in s['topics']:
            star_topic_counts[t.lower()] += 1

# Language counts for stars (2025)
star_lang_counts = Counter()
for s in stars_2025:
    if s['primaryLanguage']:
        star_lang_counts[s['primaryLanguage']] += 1

# Holidays
YEAR = 2025

# Lunar-based festival dates (2025)
spring_festival = dt.date(2025, 1, 29)  # Chinese New Year
chuxi = spring_festival - dt.timedelta(days=1)  # Lunar New Year's Eve
qingming = dt.date(2025, 4, 4)
dragon_boat = dt.date(2025, 5, 31)
qixi = dt.date(2025, 8, 29)
mid_autumn = dt.date(2025, 10, 6)

fixed_holidays = {
    'new_year': f"{YEAR}-01-01",
    'chuxi': chuxi.isoformat(),
    'spring_festival': spring_festival.isoformat(),
    'womens_day': f"{YEAR}-03-08",
    'qingming': qingming.isoformat(),
    'labor_day': f"{YEAR}-05-01",
    'youth_day': f"{YEAR}-05-04",
    'childrens_day': f"{YEAR}-06-01",
    'dragon_boat': dragon_boat.isoformat(),
    'qixi': qixi.isoformat(),
    'mid_autumn': mid_autumn.isoformat(),
    'national_day': f"{YEAR}-10-01",
    'programmer_day': f"{YEAR}-10-24",
    'singles_day': f"{YEAR}-11-11",
    'new_year_eve': f"{YEAR}-12-31",
}

holiday_labels = {
    'new_year': '元旦',
    'chuxi': '除夕',
    'spring_festival': '春节',
    'womens_day': '妇女节',
    'qingming': '清明节',
    'labor_day': '劳动节',
    'youth_day': '青年节',
    'childrens_day': '儿童节',
    'dragon_boat': '端午节',
    'qixi': '七夕',
    'mid_autumn': '中秋节',
    'national_day': '国庆节',
    'programmer_day': '程序员节',
    'singles_day': '双十一',
    'new_year_eve': '跨年夜(12/31)',
}

# Starred repos on holidays (2025 only)
holiday_stars = {}
for key, date in fixed_holidays.items():
    holiday_stars[key] = [s for s in stars_2025 if s['starredAt'][:10] == date]

# Select interesting holidays (3-5)
# Scoring: notable stars or interest match
interests = set(summary['profile'].get('interests', []))
primary_langs = set(summary['profile'].get('primary_languages', []))


def score_repo(s):
    score = 0
    lang = (s.get('primaryLanguage') or '').lower()
    if s.get('primaryLanguage') in primary_langs:
        score += 3
    if lang in interests:
        score += 2
    for t in s.get('topics', []):
        if t.lower() in interests:
            score += 2
    stars_count = s.get('stargazerCount', 0)
    if stars_count >= 10000:
        score += 3
    elif stars_count >= 1000:
        score += 1
    return score

holiday_scores = []
for key, repos in holiday_stars.items():
    if not repos:
        continue
    score = sum(score_repo(r) for r in repos)
    holiday_scores.append((key, score, repos))

holiday_scores.sort(key=lambda x: (-x[1], -len(x[2])))
selected_holidays = holiday_scores[:5]

holiday_cards = []
for key, score, repos in selected_holidays:
    repos_sorted = sorted(repos, key=lambda r: r.get('stargazerCount', 0), reverse=True)
    holiday_cards.append({
        'key': key,
        'label': holiday_labels.get(key, key),
        'date': fixed_holidays[key],
        'count': len(repos),
        'topRepo': {
            'nameWithOwner': repos_sorted[0]['nameWithOwner'],
            'stars': repos_sorted[0]['stargazerCount'],
            'language': repos_sorted[0]['primaryLanguage'],
            'starredAt': repos_sorted[0]['starredAt'],
        },
        'repos': [
            {
                'nameWithOwner': r['nameWithOwner'],
                'stars': r['stargazerCount'],
                'language': r['primaryLanguage'],
            }
            for r in repos_sorted[:3]
        ]
    })

# AI Picks (from starred repos in 2025)
# Scoring based on interests + language + stars + AI/Rust relevance

def ai_pick_score(s):
    score = 0
    lang = (s.get('primaryLanguage') or '').lower()
    topics = [t.lower() for t in s.get('topics', [])]
    name = (s.get('nameWithOwner') or '').lower()
    desc = (s.get('description') or '').lower()

    # Interest match
    for t in topics:
        if t in interests:
            score += 3
    if lang in interests:
        score += 2

    # Language match
    if s.get('primaryLanguage') in primary_langs:
        score += 3

    # Popularity
    stars_count = s.get('stargazerCount', 0)
    if stars_count >= 50000:
        score += 4
    elif stars_count >= 10000:
        score += 3
    elif stars_count >= 1000:
        score += 1

    # AI/Rust signals
    if 'ai' in interests or 'llm' in interests:
        if 'ai' in topics or 'llm' in topics or 'openai' in topics or 'rag' in topics:
            score += 4
        if 'ai' in name or 'llm' in name or 'openai' in name or 'rag' in name:
            score += 2
        if 'ai' in desc or 'llm' in desc or 'openai' in desc or 'rag' in desc:
            score += 1

    if 'rust' in interests:
        if 'rust' in topics or lang == 'rust':
            score += 3

    return score

candidates = stars_2025[:] if stars_2025 else stars
candidates_sorted = sorted(candidates, key=ai_pick_score, reverse=True)

# pick top 3 unique repos
ai_picks = []
seen = set()
for s in candidates_sorted:
    if s['nameWithOwner'] in seen:
        continue
    if ai_pick_score(s) == 0:
        continue
    ai_picks.append(s)
    seen.add(s['nameWithOwner'])
    if len(ai_picks) >= 3:
        break

# Build reasons for AI picks
ai_pick_cards = []
for s in ai_picks:
    reasons = []
    if s.get('primaryLanguage') in primary_langs:
        reasons.append(f"与你常用语言 {s['primaryLanguage']} 高度匹配")
    topic_hit = [t for t in s.get('topics', []) if t.lower() in interests]
    if topic_hit:
        reasons.append("关注主题契合：" + "、".join(topic_hit[:3]))
    if s.get('stargazerCount', 0) >= 10000:
        reasons.append(f"人气项目（★{s['stargazerCount']}）")
    if s.get('description'):
        reasons.append(s['description'][:60])

    ai_pick_cards.append({
        'nameWithOwner': s['nameWithOwner'],
        'url': s['url'],
        'stars': s['stargazerCount'],
        'language': s['primaryLanguage'],
        'starredAt': s['starredAt'],
        'reasons': reasons[:3],
    })

report = {
    'summary': summary,
    'holidays': holiday_cards,
    'holidayMeta': {
        'checkedDates': fixed_holidays,
        'note': '节日日期包含农历节日（春节/除夕/端午/七夕/中秋）对应的公历日期。',
    },
    'aiPicks': ai_pick_cards,
    'starTopicsTop': [
        {'name': k, 'count': v} for k, v in star_topic_counts.most_common(30)
    ],
    'starLanguagesTop': [
        {'name': k, 'count': v} for k, v in star_lang_counts.most_common(10)
    ],
}

with open(PROCESSED / 'report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print('report written to', PROCESSED / 'report.json')
