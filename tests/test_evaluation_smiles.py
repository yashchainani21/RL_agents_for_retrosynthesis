import csv
from pathlib import Path

import pytest
from rdkit import Chem


EVALUATION_DIR = Path(__file__).resolve().parents[1] / "data" / "evaluation"
SKIP_ROWS = {
    "Roucairol_et_al.csv": {18, 28, 29, 31},
}


def _find_smiles_column(fieldnames: list[str]) -> str | None:
    for name in fieldnames:
        if name is None:
            continue
        if "smiles" in name.lower():
            return name
    return None


@pytest.mark.parametrize("csv_path", sorted(EVALUATION_DIR.glob("*.csv")))
def test_evaluation_smiles_sanitize(csv_path: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            pytest.fail(f"{csv_path} has no header row")

        smiles_col = _find_smiles_column(reader.fieldnames)
        if smiles_col is None:
            pytest.fail(f"{csv_path} has no SMILES column (headers: {reader.fieldnames})")

        failures: list[str] = []
        skipped_rows: list[int] = []
        skip_rows = SKIP_ROWS.get(csv_path.name, set())
        for row_index, row in enumerate(reader, start=2):
            if row_index in skip_rows:
                skipped_rows.append(row_index)
                continue
            smiles = (row.get(smiles_col) or "").strip()
            if not smiles:
                failures.append(f"{csv_path} row {row_index} has empty SMILES")
                continue
            try:
                mol = Chem.MolFromSmiles(smiles, sanitize=True)
            except Exception as exc:  # RDKit sanitization errors raise exceptions
                failures.append(
                    f"{csv_path} row {row_index} failed sanitization: {smiles} ({exc})"
                )
                continue

            if mol is None:
                failures.append(f"{csv_path} row {row_index} invalid SMILES: {smiles}")

        if skipped_rows:
            skipped_str = ", ".join(str(r) for r in sorted(skipped_rows))
            print(f"[SMILES test] Skipping rows in {csv_path.name}: {skipped_str}")

        if failures:
            formatted = "\n".join(failures)
            pytest.fail(f"Invalid SMILES detected:\n{formatted}")
