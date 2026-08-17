from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = [ROOT / name for name in ("app", "pct", "training", "legacy")]


def _python_sources():
    for directory in SOURCE_DIRS:
        yield from directory.rglob("*.py")


def test_no_known_local_machine_paths_in_python_sources():
    forbidden = ["D:" + "\\Teacher", "C:" + "\\Users", "F:" + "\\", "E:" + "\\"]
    offenders = []
    for path in _python_sources():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in forbidden:
            if marker in text:
                offenders.append((str(path.relative_to(ROOT)), marker))
    assert offenders == []


def test_all_python_sources_compile():
    for path in _python_sources():
        py_compile.compile(str(path), doraise=True)
