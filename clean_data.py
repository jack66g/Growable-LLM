"""
Data cleaning script for GrowableLLM training datasets.

Removes or fixes rows flagged by audit_data.py:
  1. Translate-not-translated  → DROP (instruction says translate, but output is same lang as input)
  2. Input residue             → FIX  (strip machine-translation residue from input field)
  3. Cooking mismatch          → DROP (instruction asks for method A, output describes method B)
  4. Output == instruction     → DROP (model just copied the instruction)
  5. Empty output              → DROP (blank output)
  6. Lang mismatch (output)    → DROP (target lang != actual output lang, for translate tasks only)

Semantic shortcut rows are NOT auto-dropped — they need human review.

Usage:
    python clean_data.py [path1.jsonl path2.jsonl ...]
    # default: Experiment_Replication/*.jsonl

Outputs:
    <name>_cleaned.jsonl   — cleaned dataset
    <name>_dropped.jsonl  — dropped rows with drop reason
"""

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Re-use detection logic from audit_data.py (inline to avoid import dependency)
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r'[一-鿿㐀-䶿豈-﫿]')
_HIRAGANA_RE = re.compile(r'[぀-ゟ]')
_KATAKANA_RE = re.compile(r'[゠-ヿ]')
_LATIN_RE = re.compile(r'[A-Za-z]')


def detect_lang(text: str) -> str:
    if not text or not text.strip():
        return "empty"
    cjk = len(_CJK_RE.findall(text))
    hira = len(_HIRAGANA_RE.findall(text))
    kata = len(_KATAKANA_RE.findall(text))
    lat = len(_LATIN_RE.findall(text))
    if hira + kata > 0 and hira + kata >= cjk * 0.3:
        return "ja"
    total = cjk + lat
    if total == 0:
        return "other"
    if cjk > 0 and lat > 0:
        ratio = cjk / total
        if ratio > 0.7: return "zh"
        elif ratio < 0.3: return "en"
        else: return "mixed"
    elif cjk > 0: return "zh"
    elif lat > 0: return "en"
    return "other"


_TARGET_LANG_PATTERNS = {
    'en': [r'翻译成英文', r'翻译为英文', r'译成英文', r'译为英文',
           r'Translate.*to English', r'translate.*into English',
           r'in English', r'用英文', r'用英语', r'英语翻译', r'英文翻译'],
    'zh': [r'翻译成中文', r'翻译为中文', r'译成中文', r'译为中文',
           r'Translate.*to Chinese', r'translate.*into Chinese',
           r'in Chinese', r'用中文', r'用汉语', r'中文翻译'],
    'ja': [r'翻译成日文', r'翻译成日语', r'译成日文', r'译成日语',
           r'Translate.*to Japanese', r'translate.*into Japanese',
           r'in Japanese', r'用日文', r'用日语'],
    'fr': [r'翻译成法文', r'翻译成法语', r'译成法文', r'译成法语',
           r'Translate.*to French', r'translate.*into French',
           r'in French', r'用法文', r'用法语'],
    'de': [r'翻译成德文', r'翻译成德语', r'译成德文', r'译成德语',
           r'Translate.*to German', r'translate.*into German',
           r'in German', r'用德文', r'用德语'],
    'es': [r'翻译成西班牙文', r'翻译成西班牙语', r'译成西班牙文',
           r'Translate.*to Spanish', r'translate.*into Spanish',
           r'in Spanish', r'用西班牙文', r'用西班牙语'],
}

_TRANSLATE_TASK_RE = re.compile(r'翻译|translate|译成|译为|译出', re.IGNORECASE)

_COOK_METHODS = {
    '煮': ['煮', '水煮', '白煮'],
    '煎': ['煎', '香煎', '干煎', '平底锅煎'],
    '炒': ['炒', '爆炒', '翻炒'],
    '烤': ['烤', '烘烤', '烤箱'],
    '蒸': ['蒸', '清蒸'],
    '炸': ['炸', '油炸', '深炸'],
    '炖': ['炖', '红烧', '焖'],
}

# Patterns to strip machine-translation residue from input
# Match '的中文翻译为' or '的英文翻译为' and everything after
_RESIDUE_PATTERNS = [
    re.compile(r'的中文翻译为.*$'),
    re.compile(r'的英文翻译为.*$'),
    re.compile(r'翻译为[：:].*$'),
]


def extract_target_lang(instruction: str) -> str | None:
    for lang, patterns in _TARGET_LANG_PATTERNS.items():
        for p in patterns:
            if re.search(p, instruction, re.IGNORECASE):
                return lang
    return None


def is_translate_task(instruction: str) -> bool:
    return bool(_TRANSLATE_TASK_RE.search(instruction))


# ---------------------------------------------------------------------------
# Cleaning decisions
# ---------------------------------------------------------------------------

def should_drop(row: dict) -> str | None:
    """Return drop reason, or None if row should be kept."""
    inst = row.get('instruction', '').strip()
    inp = row.get('input', '').strip()
    out = row.get('output', '').strip()

    # 1. Empty output
    if not out:
        return "Empty output"

    # 2. Output == instruction (copy)
    if out == inst:
        return "Output == instruction (copy)"

    # 3. Translate-not-translated
    if is_translate_task(inst) and inp:
        input_lang = detect_lang(inp)
        output_lang = detect_lang(out)
        if (input_lang not in ('empty', 'other', 'mixed')
            and output_lang not in ('empty', 'other', 'mixed')
            and input_lang == output_lang):
            target = extract_target_lang(inst) or '?'
            return f"Translate not translated: input & output both '{input_lang}', target '{target}'"

    # 4. Lang mismatch for translate tasks (output lang != target lang)
    if is_translate_task(inst):
        target = extract_target_lang(inst)
        if target:
            output_lang = detect_lang(out)
            if output_lang not in ('empty', 'other', 'mixed') and output_lang != target:
                return f"Lang mismatch: target '{target}', output is '{output_lang}'"

    # 5. Cooking mismatch
    inst_methods = set()
    for method, keywords in _COOK_METHODS.items():
        if any(kw in inst for kw in keywords):
            inst_methods.add(method)
    if inst_methods:
        out_methods = set()
        for method, keywords in _COOK_METHODS.items():
            if any(kw in out for kw in keywords):
                out_methods.add(method)
        if out_methods and not (inst_methods & out_methods):
            return f"Cooking mismatch: instruction '{'/'.join(inst_methods)}', output '{'/'.join(out_methods)}'"

    return None


def fix_input_residue(row: dict) -> dict:
    """Strip machine-translation residue from input field.
    Returns a new dict (does not mutate original)."""
    inp = row.get('input', '')

    for residue_re in _RESIDUE_PATTERNS:
        m = residue_re.search(inp)
        if m:
            # Keep everything before the residue marker
            clean_input = inp[:m.start()].strip()
            if clean_input and clean_input != inp.strip():
                new_row = dict(row)
                new_row['input'] = clean_input
                return new_row

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def clean_file(filepath: str):
    path = Path(filepath)
    if not path.exists():
        print(f"  File not found: {filepath}")
        return

    clean_path = path.with_name(path.stem + '_cleaned.jsonl')
    dropped_path = path.with_name(path.stem + '_dropped.jsonl')

    print(f"\n{'='*70}")
    print(f"  Cleaning: {path.name}")
    print(f"{'='*70}")

    total = 0
    kept = 0
    dropped = 0
    fixed = 0
    drop_reasons = {}

    with open(path, 'r', encoding='utf-8') as fin, \
         open(clean_path, 'w', encoding='utf-8') as fclean, \
         open(dropped_path, 'w', encoding='utf-8') as fdrop:

        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                reason = "Invalid JSON"
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                fdrop.write(json.dumps({
                    "line": lineno, "reason": reason, "raw": line[:200]
                }, ensure_ascii=False) + '\n')
                continue

            total += 1

            # Check if should drop
            reason = should_drop(row)
            if reason:
                dropped += 1
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                fdrop.write(json.dumps({
                    "line": lineno, "reason": reason,
                    "instruction": row.get('instruction', '')[:200],
                    "input": row.get('input', '')[:200],
                    "output": row.get('output', '')[:200],
                }, ensure_ascii=False) + '\n')
                continue

            # Fix input residue
            original_input = row.get('input', '')
            row = fix_input_residue(row)
            if row['input'] != original_input:
                fixed += 1

            # Keep
            kept += 1
            fclean.write(json.dumps(row, ensure_ascii=False) + '\n')

    # Summary
    print(f"\n  Total:   {total}")
    print(f"  Kept:    {kept}  ({kept/total*100:.1f}%)" if total else "  Kept:    0")
    print(f"  Dropped: {dropped}  ({dropped/total*100:.1f}%)" if total else "  Dropped: 0")
    print(f"  Fixed:   {fixed}  (input residue stripped)")
    print(f"\n  Drop reasons:")
    for reason, cnt in sorted(drop_reasons.items(), key=lambda x: -x[1]):
        print(f"    {cnt:4d}  {reason}")

    print(f"\n  Output files:")
    print(f"    Clean:   {clean_path}")
    print(f"    Dropped: {dropped_path}")


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        base = Path(__file__).parent / "Experiment_Replication"
        # Skip files generated by this script (suffix _cleaned or _dropped)
        files = sorted(
            f for f in base.glob("*.jsonl")
            if not f.name.endswith('_cleaned.jsonl') and not f.name.endswith('_dropped.jsonl')
        )

    if not files:
        print("No JSONL files found.")
        sys.exit(1)

    print("GrowableLLM Data Cleaner")
    print("=" * 70)

    for f in files:
        clean_file(str(f))


if __name__ == "__main__":
    main()
