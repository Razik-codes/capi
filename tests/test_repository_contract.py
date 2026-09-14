from __future__ import annotations

import ast
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_public_documentation_exists(self) -> None:
        required_paths = [
            "README.md",
            "CONTRIBUTING.md",
            "LICENSE",
            "docs/PROJECT_SCOPE.md",
            "docs/REPRODUCIBILITY.md",
            "scripts/check_environment.py",
            "scripts/validate_pretrained.py",
            ".github/workflows/quality.yml",
        ]
        missing = [path for path in required_paths if not (ROOT / path).exists()]
        self.assertEqual([], missing)

    def test_readme_states_provenance_and_no_publish_placeholders(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        forbidden_placeholders = ["OWNER/REPOSITORY", "<your-fork-url>"]
        for placeholder in forbidden_placeholders:
            self.assertNotIn(placeholder, readme)

        readme_lower = readme.lower()
        for phrase in [
            "original capi authors",
            "project scope",
            "have not reproduced full",
            "4 gb vram",
        ]:
            self.assertIn(phrase, readme_lower)

    def test_pytest_collects_only_fast_contract_tests(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        pytest_options = pyproject["tool"]["pytest"]["ini_options"]
        self.assertEqual(["tests"], pytest_options["testpaths"])

    def test_validation_scripts_parse_without_project_dependencies(self) -> None:
        for relative_path in ["scripts/check_environment.py", "scripts/validate_pretrained.py"]:
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            ast.parse(source, filename=relative_path)


if __name__ == "__main__":
    unittest.main()
