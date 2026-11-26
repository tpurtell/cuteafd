"""Import ERNIE by hard link, verify its eight-way placement, and retire owned scratch."""
from hy4_flash.handoff import Placement, run_cli
from .publish import TARGET

LAYOUT = Placement(TARGET, "ernie55-flash-k4-mc", keep_ref="keep/ernie55-flash-k4-mc",
                   task_intermediates=("bootstrap", "bootstrap-packed", "preflight-fixtures",
                                       "pilot/reconstruct", "env-build", "corpus"))


if __name__ == "__main__":
    run_cli(LAYOUT)
