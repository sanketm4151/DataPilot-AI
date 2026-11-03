"""Sanity checks that the repository is ready to deploy."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_PACKAGES = ("pandas", "numpy", "openpyxl", "scikit-learn", "streamlit", "plotly",
                    "matplotlib", "seaborn", "google-genai", "python-dotenv", "reportlab")


def test_required_files_exist():
    for relative in ("app.py", "requirements.txt", "assets/style.css",
                     "sample_data/sample_sales.csv", ".streamlit/config.toml", ".env.example"):
        assert (ROOT / relative).exists(), f"missing file: {relative}"


def test_requirements_are_pinned_and_complete():
    lines = [line.strip() for line in
             (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.strip().startswith("#")]
    assert all("==" in line for line in lines), "every requirement must be pinned with =="
    names = {line.split("==")[0].lower() for line in lines}
    for package in RUNTIME_PACKAGES:
        assert package in names, f"{package} is missing from requirements.txt"
    assert "pytest" not in names, "pytest belongs in requirements-dev.txt"


def test_secrets_are_git_ignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in ignored and ".streamlit/secrets.toml" in ignored


def test_no_api_key_is_hardcoded():
    key_pattern = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")     # shape of a Google API key
    for path in list(ROOT.glob("*.py")) + list((ROOT / "modules").glob("*.py")) \
            + list((ROOT / "tests").glob("*.py")):
        assert not key_pattern.search(path.read_text(encoding="utf-8")), f"key found in {path}"