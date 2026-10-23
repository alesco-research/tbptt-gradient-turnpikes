"""Keep direct imports covered by the package's dependency metadata."""

import ast
from pathlib import Path
import re
import sys
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DependencyDeclarationTests(unittest.TestCase):
    def test_direct_third_party_imports_are_declared(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        runtime = {re.split(r"[<>=!~\[; ]", value, maxsplit=1)[0].lower().replace("_", "-")
                   for value in project["dependencies"]}
        test = runtime | {re.split(r"[<>=!~\[; ]", value, maxsplit=1)[0].lower().replace("_", "-")
                          for value in project["optional-dependencies"]["dev"]}
        for directory in ("src", "experiments", "tests"):
            declared = test if directory == "tests" else runtime
            for path in (ROOT / directory).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                    if isinstance(node, ast.Import):
                        names = [item.name.split(".")[0] for item in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                        names = [node.module.split(".")[0]]
                    else:
                        continue
                    for name in names:
                        if name not in sys.stdlib_module_names and name != "gradient_turnpike":
                            with self.subTest(path=str(path.relative_to(ROOT)), package=name):
                                self.assertIn(name.lower().replace("_", "-"), declared)


if __name__ == "__main__":
    unittest.main()
