import re
import argparse
from pathlib import Path

def find_ignored_intervals(text):
    intervals = []
    # fenced code blocks ``` or ~~~
    for m in re.finditer(r'(?m)^(`{3,}|~{3,}).*?\n.*?^\1\s*$',
                         text, flags=re.DOTALL):
        intervals.append((m.start(), m.end()))
    # display math $$...$$
    for m in re.finditer(r'\$\$.*?\$\$', text, flags=re.DOTALL):
        intervals.append((m.start(), m.end()))
    # inline code spans `...` (simple heuristic)
    for m in re.finditer(r'`+[^`]*?`+', text, flags=re.DOTALL):
        intervals.append((m.start(), m.end()))
    intervals.sort()
    return intervals

def inside_intervals(pos, intervals):
    # binary search-like (linear ok for typical files)
    for a,b in intervals:
        if a <= pos < b:
            return True
        if pos < a:
            return False
    return False

def convert_single_dollars(text):
    intervals = find_ignored_intervals(text)
    # Regex for paired single $...$ (not $$) with no newline inside (safer)
    pattern = re.compile(r'(?<!\$)\$(?!\$)([^$\n]+?)(?<!\$)\$(?!\$)', flags=re.DOTALL)
    changes = 0

    def repl(m):
        nonlocal changes
        s = m.start()
        if inside_intervals(s, intervals):
            return m.group(0)
        # don't convert if inside parentheses like $100 (common currency) — heuristic:
        inner = m.group(1)
        # skip if inner starts or ends with whitespace only (to avoid accidental)
        if inner.strip() == '':
            return m.group(0)
        changes += 1
        return '$$' + inner + '$$'

    new_text = pattern.sub(repl, text)
    return new_text, changes

def process_file(path: Path, apply: bool):
    text = path.read_text(encoding='utf-8')
    new_text, changes = convert_single_dollars(text)
    if changes == 0:
        return False, 0
    if apply:
        bak = path.with_suffix(path.suffix + '.bak')
        bak.write_text(text, encoding='utf-8')
        path.write_text(new_text, encoding='utf-8')
    return True, changes

def main():
    p = argparse.ArgumentParser(description="Convert Obsidian inline $...$ to $$...$$ (safe).")
    p.add_argument('--path', '-p', type=str, default='notes', help='File or root path to scan (file or directory).')
    p.add_argument('--ext', '-e', type=str, default='.md', help='File extension to process when path is a directory (default: .md)')
    p.add_argument('--apply', action='store_true', help='Apply changes (writes files). Without this, just dry-run.')
    args = p.parse_args()

    root = Path(args.path)
    if not root.exists():
        print("Path not found:", root)
        return

    total_files = 0
    total_changes = 0

    if root.is_file():
        # single file mode
        ok, changes = process_file(root, apply=args.apply)
        if ok:
            total_files += 1
            total_changes += changes
            status = "MODIFIED" if args.apply else "WILL CHANGE"
            print(f"{status}: {root}  ({changes} replacements)")
    else:
        # directory mode (previous behavior)
        files = list(root.rglob(f'*{args.ext}'))
        for f in files:
            ok, changes = process_file(f, apply=args.apply)
            if ok:
                total_files += 1
                total_changes += changes
                status = "MODIFIED" if args.apply else "WILL CHANGE"
                print(f"{status}: {f}  ({changes} replacements)")

    if total_files == 0:
        print("No files to change.")
    else:
        print(f"\nFiles affected: {total_files}. Total replacements: {total_changes}.")
        if not args.apply:
            print("Run with --apply to write changes (backups *.md.bak created).")

if __name__ == '__main__':
    main()

# Example usage:
# Dry-run on a single file:
# python3 scripts/convert_obsidian_math.py --path notes/stats-basis/descriptive-statistics.md
#
# Apply on a single file:
# python3 scripts/convert_obsidian_math.py --path notes/stats-basis/descriptive-statistics.md --apply
#
# Apply recursively on a directory:
# python3 scripts/convert_obsidian_math.py --path notes --ext .md --apply