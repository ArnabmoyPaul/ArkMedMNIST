"""Crash-proof checkpoint primitives: atomic writes (tmp -> fsync ->
os.replace, so a power cut can only ever leave the previous good file
intact -- os.replace is MoveFileExW with MOVEFILE_REPLACE_EXISTING on
Windows, atomic for same-volume renames) and full RNG state capture/restore,
so training resumes byte-for-byte rather than restarting the RNG stream."""
import os
import random

import numpy as np
import torch


def save_checkpoint_atomic(state, path_latest, path_prev):
    tmp_path = path_latest + ".tmp"
    with open(tmp_path, "wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    if os.path.isfile(path_latest):
        os.replace(path_latest, path_prev)
    os.replace(tmp_path, path_latest)


def load_checkpoint_with_fallback(path_latest, path_prev):
    """Try latest; if missing or fails to load (e.g. killed between the two
    os.replace() calls, or disk corruption), fall back to the previous one."""
    for path, tag in [(path_latest, "latest"), (path_prev, "previous")]:
        if os.path.isfile(path):
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                print(f">>> Loaded {tag} checkpoint: {path}")
                return ckpt
            except Exception as e:
                print(f"WARNING: failed to load {tag} checkpoint '{path}': {e}")
    return None


def capture_rng_state():
    return {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(rng):
    torch.set_rng_state(rng['torch'])
    np.random.set_state(rng['numpy'])
    random.setstate(rng['python'])
    if rng['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng['cuda'])
