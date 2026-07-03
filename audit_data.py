"""
Data quality audit script for GrowableLLM training datasets.

Detects:
  1. Translate-not-translated  — instruction is a translate task, but output is in the same language as input
  2. Lang mismatch              — instruction asks for target lang X, but output/input is lang Y
  3. Input residue              — input contains machine-translation residue (e.g. "的中文翻译为")
  4. Semantic shortcut           — instruction asks to generate/describe, output only gives a label
  5. Cooking mismatch           — instruction asks for method A, output describes method B
  6. Output == instruction      — model just copied the instruction
  7. Empty output               — blank output field
  8. Output too short (real)    — output suspiciously short for a generative task (not a label)

Usage:
    python audit_data.py [path1.jsonl path2.jsonl ...]
    # default: Experiment_Replication/*.jsonl
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Language detection (lightweight, no external dependency)
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r'[一-鿿㐀-䶿豈-﫿]')
_HIRAGANA_RE = re.compile(r'[぀-ゟ]')
_KATAKANA_RE = re.compile(r'[゠-ヿ]')
_LATIN_RE = re.compile(r'[A-Za-z]')
_CYRILLIC_RE = re.compile(r'[Ѐ-ӿ]')


def detect_lang(text: str) -> str:
    """Return dominant language tag: 'zh', 'ja', 'en', 'ru', or 'mixed'."""
    if not text or not text.strip():
        return "empty"

    cjk = len(_CJK_RE.findall(text))
    hira = len(_HIRAGANA_RE.findall(text))
    kata = len(_KATAKANA_RE.findall(text))
    lat = len(_LATIN_RE.findall(text))
    cyr = len(_CYRILLIC_RE.findall(text))

    if hira + kata > 0 and hira + kata >= cjk * 0.3:
        return "ja"
    if cyr > lat and cyr > cjk:
        return "ru"

    total = cjk + lat
    if total == 0:
        return "other"

    if cjk > 0 and lat > 0:
        ratio = cjk / total
        if ratio > 0.7:
            return "zh"
        elif ratio < 0.3:
            return "en"
        else:
            return "mixed"
    elif cjk > 0:
        return "zh"
    elif lat > 0:
        return "en"
    return "other"


# ---------------------------------------------------------------------------
# Instruction intent extraction
# ---------------------------------------------------------------------------

# Target language from instruction
_TARGET_LANG_PATTERNS = {
    'en': [
        r'翻译成英文', r'翻译为英文', r'译成英文', r'译为英文',
        r'Translate.*to English', r'translate.*into English',
        r'in English', r'用英文', r'用英语',
        r'英语翻译', r'英文翻译', r'English translation',
    ],
    'zh': [
        r'翻译成中文', r'翻译为中文', r'译成中文', r'译为中文',
        r'Translate.*to Chinese', r'translate.*into Chinese',
        r'in Chinese', r'用中文', r'用汉语',
        r'中文翻译', r'Chinese translation',
    ],
    'ja': [
        r'翻译成日文', r'翻译成日语', r'译成日文', r'译成日语',
        r'Translate.*to Japanese', r'translate.*into Japanese',
        r'in Japanese', r'用日文', r'用日语',
    ],
    'fr': [
        r'翻译成法文', r'翻译成法语', r'译成法文', r'译成法语',
        r'Translate.*to French', r'translate.*into French',
        r'in French', r'用法文', r'用法语',
    ],
    'de': [
        r'翻译成德文', r'翻译成德语', r'译成德文', r'译成德语',
        r'Translate.*to German', r'translate.*into German',
        r'in German', r'用德文', r'用德语',
    ],
    'es': [
        r'翻译成西班牙文', r'翻译成西班牙语', r'译成西班牙文',
        r'Translate.*to Spanish', r'translate.*into Spanish',
        r'in Spanish', r'用西班牙文', r'用西班牙语',
    ],
}

# Source language from instruction (for translate A→B detection)
_SOURCE_LANG_PATTERNS = {
    'en': [
        r'从英语翻译', r'从英文翻译', r'英语翻译成', r'英文翻译成',
        r'from English', r'English to', r'英译',
    ],
    'zh': [
        r'从中文翻译', r'从汉语翻译', r'中文翻译成', r'汉语翻译成',
        r'from Chinese', r'Chinese to', r'中译',
    ],
    'fr': [
        r'从法语翻译', r'从法文翻译', r'法语翻译成',
        r'from French', r'French to', r'法译',
    ],
    'de': [
        r'从德语翻译', r'从德文翻译', r'德语翻译成',
        r'from German', r'German to', r'德译',
    ],
    'es': [
        r'从西班牙语翻译', r'西班牙语翻译成',
        r'from Spanish', r'Spanish to',
    ],
    'ja': [
        r'从日语翻译', r'从日文翻译', r'日语翻译成',
        r'from Japanese', r'Japanese to', r'日译',
    ],
}

# Patterns that indicate a translate task
_TRANSLATE_TASK_RE = re.compile(
    r'翻译|translate|译成|译为|译出',
    re.IGNORECASE,
)

# Cooking method keywords
_COOK_METHODS = {
    '煮': ['煮', '水煮', '白煮'],
    '煎': ['煎', '香煎', '干煎', '平底锅煎'],
    '炒': ['炒', '爆炒', '翻炒'],
    '烤': ['烤', '烘烤', '烤箱'],
    '蒸': ['蒸', '清蒸'],
    '炸': ['炸', '油炸', '深炸'],
    '炖': ['炖', '红烧', '焖'],
}

# Short-label patterns — outputs that are legitimate short answers (not errors)
_LABEL_PATTERNS = re.compile(
    r'^(真|假|true|false|是|否|yes|no|对|错|'
    r'积极|消极|中性|正面|负面|'
    r'陈述句|疑问句|感叹句|祈使句|'
    r'小说|非小说|虚构|非虚构|'
    r'主观|客观|观察|推断|评价|'
    r'动物|植物|人|地点|'
    r'工具|玩具|家具|家电|电子产品|'
    r'肉食动物|草食动物|杂食动物|'
    r'政治|经济|社会|娱乐|体育|'
    r'促销|交易|事实|观点|'
    r'正式|非正式|'
    r'同义词|反义词|明喻|暗喻|比喻|'
    r'其他|主角|反派|'
    r'[A-E]\)?\s*\S+|'   # multiple choice: A) xxx
    r'分类[：:].+|'
    r'#[一-鿿]+|'  # hashtag
    r'\S{1,4}[。.！？]?$'  # single short word + punctuation
    r')\s*[。.！？]?$',
    re.IGNORECASE,
)

# Generative task keywords — instructions that expect a long output
_GENERATIVE_RE = re.compile(
    r'生成|写|描述|解释|创建|列出|提供|给出.*步骤|如何|怎么|'
    r'generate|write|describe|explain|create|list|provide|how to',
    re.IGNORECASE,
)


def extract_target_lang(instruction: str) -> str | None:
    """If instruction explicitly asks for a target language, return its code."""
    for lang, patterns in _TARGET_LANG_PATTERNS.items():
        for p in patterns:
            if re.search(p, instruction, re.IGNORECASE):
                return lang
    return None


def extract_source_lang(instruction: str) -> str | None:
    """If instruction mentions a source language, return its code."""
    for lang, patterns in _SOURCE_LANG_PATTERNS.items():
        for p in patterns:
            if re.search(p, instruction, re.IGNORECASE):
                return lang
    return None


def is_translate_task(instruction: str) -> bool:
    """Check if instruction is a translation task."""
    return bool(_TRANSLATE_TASK_RE.search(instruction))


# ---------------------------------------------------------------------------
# Audit checks
# ---------------------------------------------------------------------------

def check_translate_not_translated(row: dict) -> list[str]:
    """Instruction is a translate task, but output is in the same language as input
    (i.e. nothing was actually translated)."""
    inst = row.get('instruction', '')
    inp = row.get('input', '').strip()
    out = row.get('output', '').strip()

    if not is_translate_task(inst):
        return []

    # Need input to compare
    if not inp:
        return []

    input_lang = detect_lang(inp)
    output_lang = detect_lang(out)

    # Both are the same non-empty language → not translated
    if (input_lang not in ('empty', 'other', 'mixed')
        and output_lang not in ('empty', 'other', 'mixed')
        and input_lang == output_lang):
        target = extract_target_lang(inst) or '?'
        return [f"Translate not translated: input & output both '{input_lang}', target was '{target}'"]

    return []


def check_lang_mismatch(row: dict) -> list[str]:
    """Instruction asks for target lang X but output/input is lang Y."""
    issues = []
    target = extract_target_lang(row.get('instruction', ''))
    if target is None:
        return issues

    output_lang = detect_lang(row.get('output', ''))
    inp = row.get('input', '').strip()
    input_lang = detect_lang(inp) if inp else None

    if output_lang not in ('empty', 'other', 'mixed') and output_lang != target:
        issues.append(f"Lang mismatch: target '{target}', output is '{output_lang}'")

    if input_lang and input_lang not in ('empty', 'other', 'mixed') and input_lang != target:
        # Source lang mismatch is only a problem if input should be in target lang
        # (e.g. "translate to English" but input is already English → weird)
        source = extract_source_lang(row.get('instruction', ''))
        if source is None or source == target:
            issues.append(f"Lang mismatch: target '{target}', input is '{input_lang}'")

    return issues


def check_input_residue(row: dict) -> list[str]:
    """Input contains machine-translation residue like '的中文翻译为'."""
    inp = row.get('input', '')
    if not inp.strip():
        return []

    residue_patterns = [
        (r'的中文翻译为', '中文翻译残留'),
        (r'的英文翻译为', '英文翻译残留'),
        (r'翻译为[：:]', '翻译残留'),
    ]

    issues = []
    for pattern, label in residue_patterns:
        if re.search(pattern, inp):
            issues.append(f"Input residue: {label}")
    return issues


def check_semantic_shortcut(row: dict) -> list[str]:
    """Instruction asks for a generative task, but output is just a short label."""
    inst = row.get('instruction', '')
    out = row.get('output', '').strip()

    # Only flag if instruction is clearly generative
    if not _GENERATIVE_RE.search(inst):
        return []

    # And output is very short (< 15 chars) and looks like a label
    if len(out) >= 15:
        return []

    # Skip if it's a legitimate short answer
    if _LABEL_PATTERNS.match(out):
        return []

    # This looks like a shortcut: generative instruction but tiny non-label output
    return [f"Semantic shortcut: generative instruction, but output is only {len(out)} chars"]


def check_cooking_mismatch(row: dict) -> list[str]:
    """Instruction asks for cooking method A but output describes method B."""
    inst = row.get('instruction', '')
    out = row.get('output', '')

    inst_methods = set()
    for method, keywords in _COOK_METHODS.items():
        if any(kw in inst for kw in keywords):
            inst_methods.add(method)

    if not inst_methods:
        return []

    out_methods = set()
    for method, keywords in _COOK_METHODS.items():
        if any(kw in out for kw in keywords):
            out_methods.add(method)

    if not out_methods:
        return []

    if not inst_methods & out_methods:
        return [f"Cooking mismatch: instruction '{'/'.join(inst_methods)}', output '{'/'.join(out_methods)}'"]
    return []


def check_copy(row: dict) -> list[str]:
    """Output exactly copies instruction."""
    out = row.get('output', '').strip()
    inst = row.get('instruction', '').strip()
    if out and out == inst:
        return ["Output == instruction (copy)"]
    return []


def check_empty_output(row: dict) -> list[str]:
    """Output is blank."""
    out = row.get('output', '').strip()
    if not out:
        return ["Empty output"]
    return []


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

_SEVERITY = {
    'Translate not translated': 'critical',
    'Lang mismatch': 'critical',
    'Cooking mismatch': 'high',
    'Input residue': 'high',
    'Semantic shortcut': 'high',
    'Output == instruction': 'high',
    'Empty output': 'medium',
}


def classify_severity(issue: str) -> str:
    for prefix, sev in _SEVERITY.items():
        if issue.startswith(prefix):
            return sev
    return 'low'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_CHECKS = [
    check_translate_not_translated,
    check_lang_mismatch,
    check_input_residue,
    check_semantic_shortcut,
    check_cooking_mismatch,
    check_copy,
    check_empty_output,
]


def audit_file(filepath: str):
    """Run all checks on a single JSONL file and print findings."""
    path = Path(filepath)
    if not path.exists():
        print(f"  File not found: {filepath}")
        return

    print(f"\n{'='*70}")
    print(f"  Auditing: {path.name}  ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"{'='*70}")

    issues_by_type = defaultdict(list)
    total = 0

    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                issues_by_type["JSON parse error"].append((lineno, "Invalid JSON", {}))
                continue

            total += 1
            all_issues = []
            for check in ALL_CHECKS:
                all_issues.extend(check(row))

            for issue in all_issues:
                category = issue.split(':')[0]
                issues_by_type[category].append((lineno, issue, row))

    # --- Print results ---
    print(f"\n  Total rows: {total}")
    print(f"  Issue categories: {len(issues_by_type)}\n")

    # Sort by severity (critical first)
    def sort_key(cat):
        sample = issues_by_type[cat][0][1] if issues_by_type[cat] else ""
        sev = classify_severity(sample)
        order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        return order.get(sev, 4)

    for category in sorted(issues_by_type, key=sort_key):
        entries = issues_by_type[category]
        severity = classify_severity(entries[0][1])
        sev_icon = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}[severity]

        print(f"  {sev_icon} [{severity.upper()}] {category} — {len(entries)} cases")
        print(f"  {'-'*60}")

        for lineno, issue, row in entries[:15]:
            inst_preview = row.get('instruction', '')[:80]
            out_preview = row.get('output', '')[:80]
            inp = row.get('input', '').strip()
            inp_preview = inp[:60] if inp else '(empty)'
            print(f"    Line {lineno}: {issue}")
            print(f"      instruction: {inst_preview}")
            print(f"      input:       {inp_preview}")
            print(f"      output:      {out_preview}")
            print()

        if len(entries) > 15:
            print(f"    ... and {len(entries) - 15} more\n")

    # --- Summary ---
    total_issues = sum(len(v) for v in issues_by_type.values())
    dirty_lines = set()
    for entries in issues_by_type.values():
        for lineno, *_ in entries:
            dirty_lines.add(lineno)
    clean_pct = (total - len(dirty_lines)) / total * 100 if total else 0

    # Severity breakdown
    by_sev = defaultdict(int)
    for entries in issues_by_type.values():
        for _, issue, _ in entries:
            by_sev[classify_severity(issue)] += 1

    print(f"  {'='*60}")
    print(f"  Summary: {total_issues} issues in {len(dirty_lines)} rows ({clean_pct:.1f}% clean)")
    print(f"  Severity: " + " | ".join(f"{sev}: {cnt}" for sev, cnt in sorted(by_sev.items())))

    # --- Export report ---
    report_path = path.with_suffix('.audit.json')
    report = {
        "file": str(path),
        "total_rows": total,
        "total_issues": total_issues,
        "dirty_rows": len(dirty_lines),
        "clean_pct": round(clean_pct, 2),
        "severity_breakdown": dict(by_sev),
        "issues": {
            cat: [
                {
                    "line": lineno,
                    "severity": classify_severity(issue),
                    "issue": issue,
                    "instruction": row.get('instruction', '')[:300],
                    "input": row.get('input', '')[:300],
                    "output": row.get('output', '')[:300],
                }
                for lineno, issue, row in entries
            ]
            for cat, entries in issues_by_type.items()
        },
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Report saved: {report_path}")


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        base = Path(__file__).parent / "Experiment_Replication"
        files = sorted(base.glob("*.jsonl"))

    if not files:
        print("No JSONL files found.")
        sys.exit(1)

    print("GrowableLLM Data Quality Audit")
    print("=" * 70)

    for f in files:
        audit_file(str(f))


if __name__ == "__main__":
    main()
