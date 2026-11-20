"""Pretty-print `codex exec --json` events from stdin (best effort across Codex versions)."""

import json
import sys


def _short(text, n=400):
    text = str(text).strip().replace("\n", "\n    ")
    return text if len(text) <= n else text[:n] + " ..."


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            print(line, flush=True)
            continue
        etype = ev.get("type", "")
        item = ev.get("item") or ev.get("msg") or {}
        itype = item.get("type", "") if isinstance(item, dict) else ""
        if itype in ("command_execution", "exec_command") or "command" in item:
            if etype.endswith("started"):
                print(f"$ {_short(item.get('command'), 300)}", flush=True)
            elif etype.endswith("completed"):
                out = item.get("aggregated_output") or item.get("output") or ""
                code = item.get("exit_code")
                print(f"  -> exit {code}\n    {_short(out, 600)}", flush=True)
        elif itype in ("agent_message", "assistant_message", "message"):
            if etype.endswith("completed") or not etype.endswith("started"):
                print(f"[agent] {_short(item.get('text') or item.get('message') or '', 2000)}", flush=True)
        elif itype in ("reasoning",):
            continue
        elif itype in ("file_change", "patch"):
            if etype.endswith("completed"):
                changes = item.get("changes") or []
                files = ", ".join(c.get("path", "?") for c in changes if isinstance(c, dict))
                print(f"[edit] {files}", flush=True)
        elif etype in ("turn.completed", "turn.failed", "error", "thread.started"):
            print(f"[{etype}] {_short(json.dumps({k: v for k, v in ev.items() if k != 'type'}), 300)}", flush=True)


if __name__ == "__main__":
    main()
