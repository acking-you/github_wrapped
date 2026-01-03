# GitHub Wrapped Skill (Codex / Claude)

This repository is structured as a **repo-scoped Codex skill (Way B)**.

## Where the skill lives

- `.codex/skills/github-wrapped/SKILL.md`

## Install (one command)

```bash
./install-skill.sh         # install to BOTH Codex + Claude
./install-skill.sh --codex # Codex only
./install-skill.sh --claude # Claude only
```

## Use

- In Codex, invoke: `$github-wrapped`
- Requires: `gh auth status` to be OK

