import json
import re
from pathlib import Path


DATASET_PATH = Path("data/github-wrapped-2024/processed/dataset.json")
HTML_PATH = Path("frontend/standalone/github-wrapped-2024.html")


def main() -> None:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    json_text = json.dumps(data, ensure_ascii=False, indent=2).replace("<", "\\u003c")

    html = HTML_PATH.read_text(encoding="utf-8")

    # Case 1: Placeholder-based template.
    if "__DATASET_JSON__" in html:
        html2 = html.replace("__DATASET_JSON__", json_text)
        HTML_PATH.write_text(html2, encoding="utf-8")
        print(f"embedded {DATASET_PATH} -> {HTML_PATH} (placeholder)")
        return

    # Case 2: Normal dataset <script> block exists.
    block_pattern = r'(<script id="dataset" type="application/json">\s*)\{.*?\}(\s*</script>)'
    if re.search(block_pattern, html, flags=re.S):
        def repl(match: re.Match) -> str:
            return match.group(1) + json_text + match.group(2)

        html2 = re.sub(block_pattern, repl, html, flags=re.S)
        HTML_PATH.write_text(html2, encoding="utf-8")
        print(f"embedded {DATASET_PATH} -> {HTML_PATH} (script block)")
        return

    # Case 3: Corrupted block (previous bad replacement inserted literal \\1/\\2).
    marker = "<!-- Embedded dataset (generated from saved gh JSON). -->"
    marker_idx = html.find(marker)
    if marker_idx >= 0:
        after_marker = marker_idx + len(marker)
        next_script = html.find("<script>", after_marker)
        if next_script > after_marker:
            dataset_block = (
                f"{marker}\n"
                f'    <script id="dataset" type="application/json">\n'
                f"{json_text}\n"
                f"    </script>\n\n"
            )
            html2 = html[:marker_idx] + dataset_block + html[next_script:]
            HTML_PATH.write_text(html2, encoding="utf-8")
            print(f"embedded {DATASET_PATH} -> {HTML_PATH} (repaired)")
            return

    raise SystemExit("dataset block not found; cannot embed dataset")
    # unreachable


if __name__ == "__main__":
    main()
