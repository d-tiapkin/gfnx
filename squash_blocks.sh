#!/usr/bin/env bash
set -euo pipefail

# original branch
SRC=main
# new rewritten branch
DST=rewritten

# 1. create orphan target
git checkout --orphan "$DST"
git rm -rf .
GIT_AUTHOR_NAME="Daniil Tiapkin" \
GIT_AUTHOR_EMAIL="daniil.tiapkin@gmail.com" \
GIT_COMMITTER_NAME="Daniil Tiapkin" \
GIT_COMMITTER_EMAIL="daniil.tiapkin@gmail.com" \
git commit --allow-empty -m "Initial commit"

# 2. walk original history in order
mapfile -t COMMITS < <(git log --reverse --pretty=format:'%H' "$SRC")

current_author=""
block_last_commit=""

finish_block() {
    [ -z "$block_last_commit" ] && return

    author_name=$(git log -1 --pretty='%an' "$block_last_commit")
    author_email=$(git log -1 --pretty='%ae' "$block_last_commit")
    committer_name=$(git log -1 --pretty='%cn' "$block_last_commit")
    committer_email=$(git log -1 --pretty='%ce' "$block_last_commit")
    commit_message=$(git log -1 --pretty='%B' "$block_last_commit")

    git checkout "$DST"

    git rm -rf . >/dev/null 2>&1 || true
    git checkout "$block_last_commit" -- .

    git add -A

    # --- minimal fix: skip empty diffs ---
    if git diff --cached --quiet; then
        block_last_commit=""
        return
    fi
    # --------------------------------------

    GIT_AUTHOR_NAME="$author_name" \
    GIT_AUTHOR_EMAIL="$author_email" \
    GIT_COMMITTER_NAME="$committer_name" \
    GIT_COMMITTER_EMAIL="$committer_email" \
    git commit -m "Initial commit"

    block_last_commit=""
}

for c in "${COMMITS[@]}"; do
    a=$(git log -1 --pretty='%ae' "$c")

    if [[ "$a" != "$current_author" ]]; then
        finish_block
        current_author="$a"
    fi

    block_last_commit="$c"
done

finish_block