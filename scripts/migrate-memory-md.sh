#!/usr/bin/env bash
# migrate-memory-md.sh — Migrate OpenClaw MEMORY.md / memory/*.md files into Conch
#
# Usage:
#   ./migrate-memory-md.sh                          # auto-detect ~/.openclaw/workspace
#   ./migrate-memory-md.sh /path/to/workspace       # explicit workspace
#   ./migrate-memory-md.sh --dry-run                # preview without writing
#
# What it does:
#   1. Reads MEMORY.md and memory/*.md files
#   2. Parses sections into episodes (## headers → episodes, bullet points → facts)
#   3. Ingests everything into conch via `conch remember` / `conch remember-episode`
#
# Requirements: conch CLI in PATH, bash 4+

set -uo pipefail

DRY_RUN=false
WORKSPACE=""
CONCH="conch"
FACTS=0
EPISODES=0
SKIPPED=0

# Parse args
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --help|-h)
      echo "Usage: $0 [--dry-run] [workspace-path]"
      echo ""
      echo "Migrates OpenClaw MEMORY.md and memory/*.md files into Conch."
      echo ""
      echo "Options:"
      echo "  --dry-run    Preview what would be ingested without writing"
      echo "  --help       Show this help"
      echo ""
      echo "If no workspace path is given, defaults to ~/.openclaw/workspace"
      exit 0
      ;;
    *) WORKSPACE="$arg" ;;
  esac
done

# Default workspace
if [ -z "$WORKSPACE" ]; then
  WORKSPACE="${HOME}/.openclaw/workspace"
fi

# Verify conch is available
if ! command -v "$CONCH" &>/dev/null; then
  echo "❌ conch not found in PATH. Install: cargo install --path crates/conch-cli"
  exit 1
fi

# Verify workspace exists
if [ ! -d "$WORKSPACE" ]; then
  echo "❌ Workspace not found: $WORKSPACE"
  exit 1
fi

echo "🐚 Conch Memory Migration"
echo "   Workspace: $WORKSPACE"
echo "   Dry run: $DRY_RUN"
echo ""

# Helper: ingest a fact
remember_fact() {
  local subject="$1" relation="$2" object="$3"
  if [ ${#object} -lt 3 ]; then
    ((SKIPPED++))
    return
  fi
  if [ "$DRY_RUN" = true ]; then
    echo "  [FACT] $subject | $relation | $object"
  else
    $CONCH remember "$subject" "$relation" "$object" 2>/dev/null || true
  fi
  ((FACTS++))
}

# Helper: ingest an episode
remember_episode() {
  local text="$1"
  if [ ${#text} -lt 10 ]; then
    ((SKIPPED++))
    return
  fi
  if [ "$DRY_RUN" = true ]; then
    echo "  [EPISODE] $text"
  else
    $CONCH remember-episode "$text" 2>/dev/null || true
  fi
  ((EPISODES++))
}

# Process a markdown file
process_file() {
  local file="$1"
  local filename
  filename=$(basename "$file" .md)
  local current_section=""
  local episode_buffer=""

  echo "📄 Processing: $file"

  while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines
    [[ -z "${line// }" ]] && continue

    # H1 header — file title, skip
    if [[ "$line" =~ ^#\ (.+) ]]; then
      continue
    fi

    # H2 header — new section = new episode context
    if [[ "$line" =~ ^##\ (.+) ]]; then
      # Flush previous episode buffer
      if [ -n "$episode_buffer" ]; then
        remember_episode "$episode_buffer"
        episode_buffer=""
      fi
      current_section="${BASH_REMATCH[1]}"
      continue
    fi

    # H3 header — subsection
    if [[ "$line" =~ ^###\ (.+) ]]; then
      if [ -n "$episode_buffer" ]; then
        remember_episode "$episode_buffer"
        episode_buffer=""
      fi
      current_section="${BASH_REMATCH[1]}"
      continue
    fi

    # Bullet point — could be a fact or part of an episode
    if [[ "$line" =~ ^[[:space:]]*[-*]\ (.+) ]]; then
      local content="${BASH_REMATCH[1]}"
      # Strip markdown bold
      content="${content//\*\*/}"

      # If it looks like a key-value fact (contains → or : or "is")
      if [[ "$content" =~ ^([^:→]+)[→:]\ *(.+)$ ]]; then
        local key="${BASH_REMATCH[1]}"
        local val="${BASH_REMATCH[2]}"
        # Clean up
        key="${key#"${key%%[![:space:]]*}"}"
        key="${key%"${key##*[![:space:]]}"}"
        val="${val#"${val%%[![:space:]]*}"}"
        val="${val%"${val##*[![:space:]]}"}"

        if [ ${#val} -gt 2 ] && [ ${#key} -lt 60 ]; then
          remember_fact "$key" "is" "$val"
        else
          # Append to episode buffer
          if [ -n "$episode_buffer" ]; then
            episode_buffer="$episode_buffer. $content"
          else
            episode_buffer="${filename}: ${current_section:+$current_section — }$content"
          fi
        fi
      else
        # General bullet — append to episode
        if [ -n "$episode_buffer" ]; then
          episode_buffer="$episode_buffer. $content"
        else
          episode_buffer="${filename}: ${current_section:+$current_section — }$content"
        fi
      fi
      continue
    fi

    # Plain text paragraph — episode content
    if [ -n "$line" ] && [[ ! "$line" =~ ^[\`\|] ]]; then
      if [ -n "$episode_buffer" ]; then
        episode_buffer="$episode_buffer. $line"
      else
        episode_buffer="${filename}: ${current_section:+$current_section — }$line"
      fi
    fi

  done < "$file"

  # Flush remaining buffer
  if [ -n "$episode_buffer" ]; then
    remember_episode "$episode_buffer"
  fi
}

# Process MEMORY.md
if [ -f "$WORKSPACE/MEMORY.md" ]; then
  process_file "$WORKSPACE/MEMORY.md"
fi

# Process memory/*.md files
if [ -d "$WORKSPACE/memory" ]; then
  for f in "$WORKSPACE/memory"/*.md; do
    [ -f "$f" ] && process_file "$f"
  done
  # Process subdirectories
  for dir in "$WORKSPACE/memory"/*/; do
    [ -d "$dir" ] || continue
    for f in "$dir"*.md; do
      [ -f "$f" ] && process_file "$f"
    done
  done
fi

echo ""
echo "🐚 Migration complete!"
echo "   Facts ingested:    $FACTS"
echo "   Episodes ingested: $EPISODES"
echo "   Skipped (too short): $SKIPPED"
if [ "$DRY_RUN" = true ]; then
  echo ""
  echo "   (Dry run — nothing was written. Remove --dry-run to migrate.)"
fi
