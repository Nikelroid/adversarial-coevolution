"""Tiny pipeline smoke test for run_one (CPU; ~30s)."""
import sys
import tempfile

import sweep.configs as C

C.BASE["total_timesteps"] = 4096
C.BASE["num_env"] = 4
C.BASE["n_steps"] = 256
C.BASE["batch_size"] = 256
C.BASE["eval_episodes"] = 20
C.BASE["turns_limit"] = 50

import sweep.run_one as RO  # noqa: E402


if __name__ == "__main__":
    sys.argv = ["_smoke", "--combo", "0", "--save-root", tempfile.mkdtemp()]
    RO.main()
    print("SMOKE OK")
