import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.utils import copy_all_src


class CopyAllSrcTests(unittest.TestCase):
    def test_skips_modules_whose_source_file_does_not_exist(self):
        project_root = Path(__file__).resolve().parents[1]
        missing_source_module = types.ModuleType("missing_source_module")
        missing_source_module.__file__ = "_ops.py"

        with tempfile.TemporaryDirectory() as destination:
            with patch.dict(sys.modules, {"missing_source_module": missing_source_module}):
                with patch.object(sys, "argv", [str(project_root / "train.py")]):
                    with patch.object(sys, "path", [str(project_root), str(project_root)]):
                        with patch("os.getcwd", return_value=os.fspath(project_root)):
                            copy_all_src(destination)

            self.assertTrue((Path(destination) / "src" / "utils.py").is_file())


if __name__ == "__main__":
    unittest.main()
