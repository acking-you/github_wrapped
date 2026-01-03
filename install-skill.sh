#!/usr/bin/env bash
set -euo pipefail

SKILL_NAME="github-wrapped"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${ROOT_DIR}/.codex/skills/${SKILL_NAME}"

CODEX_HOME_DEFAULT="${CODEX_HOME:-${HOME}/.codex}"
CLAUDE_HOME_DEFAULT="${CLAUDE_HOME:-${HOME}/.claude}"

usage() {
  cat <<'USAGE'
Install this repo-scoped skill to Codex and/or Claude.

Usage:
  ./install-skill.sh               # install to BOTH Codex + Claude (default)
  ./install-skill.sh --codex       # install to Codex only
  ./install-skill.sh --claude      # install to Claude only

Options:
  --codex-home <dir>               # default: $CODEX_HOME or ~/.codex
  --claude-home <dir>              # default: $CLAUDE_HOME or ~/.claude
  --no-backup                      # overwrite in-place (no .bak-<ts>)
  --dry-run                        # show what would be copied
  -h, --help

Notes:
  - This script copies files (no symlinks), which is the most reliable mode for Codex skills.
  - Source of truth is: .codex/skills/github-wrapped/
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

backup_dir() {
  local target="$1"
  local ts
  ts="$(date +%Y%m%d-%H%M%S)"
  mv "$target" "${target}.bak-${ts}"
}

sync_dir() {
  local from="$1"
  local to="$2"
  local dry_run="$3"

  local rsync_flags=(-a --delete --exclude "__pycache__/" --exclude "*.pyc")
  if [[ "${dry_run}" == "1" ]]; then
    rsync_flags+=(--dry-run)
  fi

  mkdir -p "$to"
  rsync "${rsync_flags[@]}" "${from}/" "${to}/"
}

main() {
  local install_codex="1"
  local install_claude="1"
  local codex_home="${CODEX_HOME_DEFAULT}"
  local claude_home="${CLAUDE_HOME_DEFAULT}"
  local do_backup="1"
  local dry_run="0"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --codex)
        install_codex="1"
        install_claude="0"
        shift
        ;;
      --claude)
        install_codex="0"
        install_claude="1"
        shift
        ;;
      --codex-home)
        [[ $# -ge 2 ]] || die "--codex-home requires a value"
        codex_home="$2"
        shift 2
        ;;
      --claude-home)
        [[ $# -ge 2 ]] || die "--claude-home requires a value"
        claude_home="$2"
        shift 2
        ;;
      --no-backup)
        do_backup="0"
        shift
        ;;
      --dry-run)
        dry_run="1"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown arg: $1 (use --help)"
        ;;
    esac
  done

  [[ -d "${SRC_DIR}" ]] || die "skill source not found: ${SRC_DIR}"
  [[ -f "${SRC_DIR}/SKILL.md" ]] || die "missing SKILL.md in: ${SRC_DIR}"

  if [[ "${install_codex}" == "1" ]]; then
    local codex_dst="${codex_home}/skills/${SKILL_NAME}"
    mkdir -p "${codex_home}/skills"
    if [[ "${do_backup}" == "1" && -d "${codex_dst}" && "${dry_run}" != "1" ]]; then
      backup_dir "${codex_dst}"
    fi
    sync_dir "${SRC_DIR}" "${codex_dst}" "${dry_run}"
    echo "Installed to Codex: ${codex_dst}"
  fi

  if [[ "${install_claude}" == "1" ]]; then
    local claude_dst="${claude_home}/skills/${SKILL_NAME}"
    mkdir -p "${claude_home}/skills"
    if [[ "${do_backup}" == "1" && -d "${claude_dst}" && "${dry_run}" != "1" ]]; then
      backup_dir "${claude_dst}"
    fi
    sync_dir "${SRC_DIR}" "${claude_dst}" "${dry_run}"
    echo "Installed to Claude: ${claude_dst}"
  fi
}

main "$@"

