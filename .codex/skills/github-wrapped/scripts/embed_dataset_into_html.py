import argparse
import json
import re
from pathlib import Path


def embed(dataset_path: Path, html_path: Path) -> None:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    json_text = json.dumps(data, ensure_ascii=False, indent=2).replace("<", "\\u003c")

    html = html_path.read_text(encoding="utf-8")

    # Case 1: Placeholder-based template.
    if "__DATASET_JSON__" in html:
        html2 = html.replace("__DATASET_JSON__", json_text)
        html_path.write_text(html2, encoding="utf-8")
        print(f"embedded {dataset_path} -> {html_path} (placeholder)")
        return

    # Case 2: Normal dataset <script> block exists.
    block_pattern = r'(<script id="dataset" type="application/json">\s*)\{.*?\}(\s*</script>)'
    if re.search(block_pattern, html, flags=re.S):

        def repl(match: re.Match) -> str:
            return match.group(1) + json_text + match.group(2)

        html2 = re.sub(block_pattern, repl, html, flags=re.S)
        html_path.write_text(html2, encoding="utf-8")
        print(f"embedded {dataset_path} -> {html_path} (script block)")
        return

    # Case 3: If there's a marker, try to repair by re-inserting the block.
    marker = "<!-- Embedded dataset (generated from saved gh JSON). -->"
    marker_idx = html.find(marker)
    if marker_idx >= 0:
        after_marker = marker_idx + len(marker)
        next_script = html.find("<script>", after_marker)
        if next_script > after_marker:
            dataset_block = (
                f"{marker}\\n"
                f'    <script id="dataset" type="application/json">\\n'
                f"{json_text}\\n"
                f"    </script>\\n\\n"
            )
            html2 = html[:marker_idx] + dataset_block + html[next_script:]
            html_path.write_text(html2, encoding="utf-8")
            print(f"embedded {dataset_path} -> {html_path} (repaired)")
            return

    raise SystemExit("dataset block not found; cannot embed dataset")


def main() -> None:
    p = argparse.ArgumentParser(description="Embed dataset.json into a single-file HTML report.")
    p.add_argument("--dataset", required=True, type=Path, help="Path to processed/dataset.json")
    p.add_argument("--html", required=True, type=Path, help="Path to output HTML file")
    args = p.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"dataset not found: {args.dataset}")
    if not args.html.exists():
        raise SystemExit(f"html not found: {args.html}")

    embed(args.dataset, args.html)


if __name__ == "__main__":
    main()
