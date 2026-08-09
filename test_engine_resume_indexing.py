"""test_engine_resume_indexing.py — Run: python test_engine_resume_indexing.py"""
from engine import _resume_dataset_range


def test_no_checkpoint_starts_at_zero():
    first_i = _resume_dataset_range(checkpoint=None, current_epoch=0, n_datasets=5)
    assert first_i == 0


def test_mid_epoch_resume_skips_completed_datasets():
    # last_completed=2 means datasets 0,1,2 fully finished (train+EMA) this
    # epoch -- resume must continue at index 3, not redo or skip one.
    ckpt = {'epoch': 3, 'last_completed': 2}
    first_i = _resume_dataset_range(checkpoint=ckpt, current_epoch=3, n_datasets=5)
    assert first_i == 3


def test_resume_only_applies_to_the_checkpointed_epoch():
    # A later epoch (e.g. after a full-epoch checkpoint) must start at 0,
    # not re-apply the old epoch's last_completed position.
    ckpt = {'epoch': 3, 'last_completed': 2}
    first_i = _resume_dataset_range(checkpoint=ckpt, current_epoch=4, n_datasets=5)
    assert first_i == 0


def test_checkpoint_saved_after_last_dataset_resumes_at_n_and_loop_is_empty():
    # If the checkpoint landed right after the LAST dataset of an epoch
    # finished (before validation ran), first_i should equal n_datasets so
    # range(first_i, n_datasets) is empty and control falls straight to
    # validation -- no separate "epoch already done" branch needed.
    ckpt = {'epoch': 3, 'last_completed': 4}
    first_i = _resume_dataset_range(checkpoint=ckpt, current_epoch=3, n_datasets=5)
    assert first_i == 5


if __name__ == "__main__":
    test_no_checkpoint_starts_at_zero()
    test_mid_epoch_resume_skips_completed_datasets()
    test_resume_only_applies_to_the_checkpointed_epoch()
    test_checkpoint_saved_after_last_dataset_resumes_at_n_and_loop_is_empty()
    print("test_engine_resume_indexing.py: all checks passed")
