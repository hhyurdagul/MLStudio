import ast
import unittest
from pathlib import Path


class ArchitectureTests(unittest.TestCase):
    def test_backend_does_not_import_frontend_or_streamlit(self) -> None:
        backend = Path("mlstudio/backend")
        forbidden = ("streamlit", "mlstudio.frontend")

        for path in backend.glob("*.py"):
            tree = ast.parse(path.read_text())
            imports = [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            imported_names = []
            for node in imports:
                if isinstance(node, ast.Import):
                    imported_names.extend(alias.name for alias in node.names)
                elif node.module is not None:
                    imported_names.append(node.module)
            self.assertFalse(
                any(
                    name.startswith(prefix)
                    for name in imported_names
                    for prefix in forbidden
                ),
                f"{path} imports a frontend dependency",
            )


if __name__ == "__main__":
    unittest.main()
