"""
apply_review.py
===============
Apply a completed review_sheet.csv to the JSON sources, then rebuild.

    python datasets/clean_synthetic/apply_review.py [review_sheet.csv]

For each reviewed row:
  * corrected_student_answer filled in      → the answer text is replaced
  * fact_and_type_correct = 'n', no correction → the answer is removed
  * anything else                             → unchanged

The JSON files are rewritten in place (keep them under version control so a
review can be undone), then build.py regenerates data/dataset_clean_*.csv.
Re-train afterwards: run_all_datasets.py --datasets data/dataset_*_plus_clean.csv
"""

import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    'single_sentence':  ['single_part1.json', 'single_part2.json'],
    'medium_sentences': ['medium.json'],
    'long_sentences':   ['long.json'],
}


def main():
    sheet_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'review_sheet.csv')
    sheet = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
    edits = {}
    for r in sheet.to_dict('records'):
        fix = r['corrected_student_answer'].strip()
        verdict = r['fact_and_type_correct (y/n)'].strip().lower()
        key = (r['tier'], r['question'], r['student_answer'])
        if fix:
            edits[key] = fix
        elif verdict.startswith('n'):
            edits[key] = None

    changed = removed = 0
    for tier, files in FILES.items():
        for name in files:
            path = os.path.join(HERE, name)
            with open(path, encoding='utf-8') as fh:
                items = json.load(fh)
            for item in items:
                kept = []
                for atype, text in item['answers']:
                    key = (tier, item['question'], text)
                    if key in edits and edits[key] is None:
                        removed += 1
                        continue
                    if key in edits:
                        text, changed = edits[key], changed + 1
                    kept.append([atype, text])
                item['answers'] = kept
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(items, fh, ensure_ascii=False, indent=2)
                fh.write('\n')

    print(f'{changed} answer(s) corrected, {removed} removed.')
    subprocess.run([sys.executable, os.path.join(HERE, 'build.py')], check=True)


if __name__ == '__main__':
    main()
