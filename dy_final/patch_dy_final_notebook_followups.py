# -*- coding: utf-8 -*-
"""Small follow-up patches for downstream cells in the dy_final notebook."""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_DIR = Path(r"C:\Users\Admin\Documents\GitHub\Ml_Baseball\dy_final")
NOTEBOOK_PATH = [p for p in NOTEBOOK_DIR.glob("*.ipynb") if "backup_before" not in p.name][0]


def main() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    changed = False

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))

        if "=== 역대 연도별 예측 검증 ===" in src and "hist_accs" not in src:
            src = src.replace(
                'print("\\n=== 역대 연도별 예측 검증 ===")\nfor yr in COMPLETE_YRS:',
                'print("\\n=== 역대 연도별 예측 검증 ===")\nhist_accs = []\nfor yr in COMPLETE_YRS:',
            )
            src = src.replace(
                "    acc = (sub['예측가을야구']==sub['가을야구']).mean()\n",
                "    acc = (sub['예측가을야구']==sub['가을야구']).mean()\n    hist_accs.append(acc)\n",
            )
            cell["source"] = src.splitlines(keepends=True)
            changed = True

        if "ML 앙상블 역대 검증 정확도" in src and "hist_accs" not in src:
            old = """print(f'  📈 ML 앙상블 역대 검증 정확도:  ~{(sum([(team_master[team_master["연도"]==yr].dropna(subset=FEATURES+[TARGET]).shape[0]) for yr in COMPLETE_YRS if yr in team_master["연도"].values]>0)*100):.0f}% (연도별 결과 참조)')"""
            new = """if 'hist_accs' in dir() and len(hist_accs) > 0:
    print(f'  📈 ML 앙상블 역대 검증 정확도:  ~{np.mean(hist_accs):.1%} (연도별 결과 참조)')
else:
    print('  📈 ML 앙상블 역대 검증 정확도:  CELL 8의 연도별 결과 참조')"""
            if old in src:
                src = src.replace(old, new)
                cell["source"] = src.splitlines(keepends=True)
                changed = True

    if not changed:
        print("no downstream patch needed")
    else:
        NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"patched downstream cells: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
