#!/usr/bin/env bash
# Run the entailment verification on Lab-IA instead of your laptop.
#
#   ./nli_sync.sh push     # ship code + articles + judgment logs to the cluster (~1 MB)
#   ./nli_sync.sh run      # submit the job and follow its log until it finishes
#   ./nli_sync.sh pull     # bring the nli_verdicts.json files back
#   ./nli_sync.sh all      # push, run, pull
#
# Then run sections 5-7 of notebooks/0_shared/9c_nli_verification.ipynb locally: they read the
# verdict files, need no GPU and no model, and finish instantly.
#
# Transfers use tar-over-ssh rather than rsync: macOS still ships rsync 2.6.9, which does not
# survive the ProxyJump this cluster needs. The payload is ~1 MB, so there is nothing to gain from
# rsync's delta transfer anyway.
#
# Only what the scorer needs is synced — the articles, the judgment logs nb9/nb9b wrote, and the
# two code files. Graphs, viz, and everything else stay on your machine.
set -euo pipefail

HOST=${HOST:-slurm}
REMOTE=${REMOTE:-Phase2_LLM_Extraction}          # path relative to $HOME on the cluster
RESULTS_MODEL=${RESULTS_MODEL:-gemini-2.5-flash}
TRACK=${TRACK:-e2e}
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# macOS tar warns about xattrs and clock skew, which are not errors — filter those two patterns
# rather than redirecting stderr to /dev/null, so a REAL tar failure is still visible. (Silencing
# it entirely means a broken archive still reports "unpacked" and the cluster runs stale code.)
# COPYFILE_DISABLE stops tar writing an AppleDouble "._name" beside every file on the far end.
# Note the argument order: the mode (czf/xzf) must come first, then --exclude. bsdtar rejects
# `tar --exclude=X czf -`, and it fails in a way that still exits 0 through a pipe.
_denoise() { grep -v -e 'Ignoring unknown extended header' -e 'time stamp' -e 'Removing leading' || true; }
_tarc() { COPYFILE_DISABLE=1 tar czf - --exclude='.DS_Store' --exclude='._*' "$@" 2> >(_denoise >&2); }
_tarx() { COPYFILE_DISABLE=1 tar xzf - "$@" 2> >(_denoise >&2); }

push() {
  echo "==> pushing to $HOST:$REMOTE"
  local logs
  logs=$(cd "$LOCAL" && find "results/$RESULTS_MODEL" \
           \( -name grounding_judgments.json -o -name grounding_fulltext_judgments.json \) 2>/dev/null)
  if [ -z "$logs" ]; then
    echo "    !! no judgment logs under results/$RESULTS_MODEL — run nb9 / nb9b first"; exit 1
  fi
  echo "    $(printf '%s\n' "$logs" | wc -l | tr -d ' ') judgment log(s), \
$(ls "$LOCAL/dataset/articles" | wc -l | tr -d ' ') article(s)"
  # one stream: code + articles + the judgment logs
  (cd "$LOCAL" && _tarc \
      notebooks/nli_core.py cluster/nli_verify.py cluster/nli_verify.sbatch \
      cluster/verbalise_check.py cluster/verbalise_check.sbatch \
      dataset/articles $logs) \
    | ssh "$HOST" "mkdir -p '$REMOTE' && tar xzf - -C '$REMOTE'"
  # prove the far end really got THIS copy — a silently broken archive would otherwise leave the
  # cluster running stale code, which is invisible until the job fails in a confusing way
  local want have
  want=$(cd "$LOCAL" && shasum notebooks/nli_core.py cluster/nli_verify.py | awk '{print $1}' | sort | md5)
  have=$(ssh "$HOST" "cd '$REMOTE' && sha1sum notebooks/nli_core.py cluster/nli_verify.py \
         | awk '{print \$1}' | sort | md5sum | cut -d' ' -f1")
  if [ "$want" != "$have" ]; then
    echo "    !! pushed code does not match local — transfer failed"; exit 1
  fi
  echo "    ok (code verified identical)"
}

run() {
  echo "==> submitting"
  local jid
  jid=$(ssh "$HOST" "cd '$REMOTE/cluster' && RESULTS_MODEL='$RESULTS_MODEL' TRACK='$TRACK' \
        ROOT=\$HOME/'$REMOTE' sbatch --parsable nli_verify.sbatch")
  echo "    job $jid — following its log (Ctrl-C stops watching, not the job)"
  ssh "$HOST" "cd '$REMOTE/cluster'
    for _ in \$(seq 1 120); do [ -f nli-$jid.log ] && break; sleep 5; done
    if [ ! -f nli-$jid.log ]; then echo '    still queued; check: squeue -j $jid'; exit 0; fi
    tail -f nli-$jid.log & TAILPID=\$!
    while squeue -h -j $jid -o %T 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING'; do sleep 10; done
    sleep 3; kill \$TAILPID 2>/dev/null || true"
  echo "    job $jid finished"
}

pull() {
  echo "==> pulling verdicts"
  ssh "$HOST" "cd '$REMOTE' && find results/$RESULTS_MODEL -name nli_verdicts.json 2>/dev/null \
      | tar czf - -T -" | _tarx -C "$LOCAL"
  find "$LOCAL/results/$RESULTS_MODEL" -name nli_verdicts.json 2>/dev/null \
    | sed "s|$LOCAL/||" | sort | sed 's/^/    /'
}

case "${1:-all}" in
  push) push ;;
  run)  run ;;
  pull) pull ;;
  all)  push; run; pull ;;
  *)    echo "usage: $0 {push|run|pull|all}"; exit 1 ;;
esac
