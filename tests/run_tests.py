"""
Standalone test runner for project 1.
Executes test cases in test_schema.py and test_harness.py.
"""

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_schema import TestValidItems, TestInvalidItems, TestStrictValidation, TestFileValidation
from tests.test_harness import TestPromptFormatting, TestLabelExtraction, TestMetrics, TestMcNemarsTest, TestValidLabels


def run_test_class(cls):
    instance = cls()
    passed = 0
    failed = 0
    methods = [m for m in dir(cls) if m.startswith("test_")]

    print(f"\n--- Running {cls.__name__} ({len(methods)} tests) ---")
    for method_name in methods:
        method = getattr(instance, method_name)
        try:
            # Handle methods that require fixtures like tmp_path
            if "tmp_path" in method.__code__.co_varnames:
                import tempfile
                with tempfile.TemporaryDirectory() as tmp_dir:
                    method(Path(tmp_dir))
            else:
                method()
            print(f"  [PASS] {method_name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {method_name}: {e}")
            traceback.print_exc()
            failed += 1

    return passed, failed


def main():
    test_classes = [
        TestValidItems,
        TestInvalidItems,
        TestStrictValidation,
        TestFileValidation,
        TestPromptFormatting,
        TestLabelExtraction,
        TestMetrics,
        TestMcNemarsTest,
        TestValidLabels,
    ]

    total_passed = 0
    total_failed = 0

    for cls in test_classes:
        p, f = run_test_class(cls)
        total_passed += p
        total_failed += f

    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {total_passed} Passed, {total_failed} Failed")
    print("=" * 50)

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
