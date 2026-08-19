"""Execute every notebook, failing on the first error.

The ``%pip install`` cells are skipped: whoever runs this already has the
environment. Everything else runs, so a notebook that drifts from the API is a
CI failure rather than a surprise for the next reader.

    uv run python scripts/run_notebooks.py [notebook ...]
    uv run python scripts/run_notebooks.py --save [notebook ...]

``--save`` writes the fresh outputs back. The docs render *stored* outputs, so
a notebook whose code was updated but whose outputs were not will show a reader
an API that no longer exists -- run with ``--save`` after any API change.
"""

import copy
import pathlib
import sys
import time

import nbformat
from nbclient import NotebookClient

NOTEBOOKS = pathlib.Path(__file__).resolve().parent.parent / "notebooks"
TIMEOUT_S = 1800


def execute(path: pathlib.Path, save: bool = False) -> float:
    """Execute one notebook and return how long it took.

    Args:
        path: The notebook.
        save: Whether to write the fresh outputs back.

    Returns:
        Wall-clock seconds.

    Raises:
        Exception: Whatever the notebook raised, unchanged.

    """
    notebook = nbformat.reads(path.read_text(), as_version=4)
    runnable = copy.deepcopy(notebook)
    skipped = [
        index
        for index, cell in enumerate(runnable.cells)
        if cell.cell_type == "code" and "%pip" in "".join(cell.source)
    ]
    for index in reversed(skipped):
        del runnable.cells[index]

    start = time.perf_counter()
    NotebookClient(runnable, timeout=TIMEOUT_S, kernel_name="python3").execute()
    seconds = time.perf_counter() - start

    if save:
        # Put the install cells back where they were, output-free as they were.
        for index in skipped:
            runnable.cells.insert(index, notebook.cells[index])
        path.write_text(nbformat.writes(runnable))
    return seconds


def main() -> int:
    """Run the notebooks named on the command line, or all of them."""
    arguments = sys.argv[1:]
    save = "--save" in arguments
    arguments = [argument for argument in arguments if argument != "--save"]
    names = arguments or sorted(path.name for path in NOTEBOOKS.glob("*.ipynb"))
    failed = []

    for name in names:
        try:
            seconds = execute(NOTEBOOKS / name, save=save)
        except Exception as error:  # noqa: BLE001 - the notebook's error is the report
            print(f"FAIL  {name}\n{error}", file=sys.stderr)
            failed.append(name)
        else:
            print(f"pass  {name:32} {seconds:6.1f}s")

    if failed:
        names = ", ".join(failed)
        print(f"\n{len(failed)} notebook(s) failed: {names}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
