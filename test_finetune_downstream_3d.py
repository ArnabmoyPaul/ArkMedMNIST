"""test_finetune_downstream_3d.py — Run: python test_finetune_downstream_3d.py"""
from finetune_downstream_3d import _early_stop, _compare_to_benchmark, parse_args, BENCHMARKS_3D


def test_no_stop_while_still_improving():
    assert not _early_stop([0.9, 0.8, 0.7, 0.6, 0.5], patience=2)


def test_no_stop_before_patience_window_elapses():
    assert not _early_stop([0.5, 0.6, 0.7], patience=3)


def test_stops_after_patience_epochs_without_improvement():
    assert _early_stop([0.9, 0.4, 0.5, 0.6, 0.7], patience=3)


def test_stops_exactly_at_patience_boundary():
    assert _early_stop([0.5, 0.6, 0.7], patience=2)


def test_compare_pass_when_auc_meets_or_beats_target():
    assert _compare_to_benchmark(0.994, 0.994) == "PASS"
    assert _compare_to_benchmark(0.999, 0.994) == "PASS"


def test_compare_within_1pct_when_short_by_at_most_one_point():
    assert _compare_to_benchmark(0.984, 0.994) == "WITHIN_1PCT"


def test_compare_fail_when_short_by_more_than_one_point():
    assert _compare_to_benchmark(0.90, 0.994) == "FAIL"


def test_compare_no_target_generic():
    assert _compare_to_benchmark(0.80, None) == "NO_TARGET"


def test_synapse_target_is_0_851():
    assert BENCHMARKS_3D["SynapseMNIST3D"] == 0.851


def test_parse_args_requires_checkpoint():
    try:
        parse_args(["--dataset", "OrganMNIST3D"])
        assert False, "expected SystemExit for missing required --checkpoint"
    except SystemExit:
        pass


def test_parse_args_defaults():
    args = parse_args(["--dataset", "OrganMNIST3D", "--checkpoint", "fake.pth.tar"])
    assert args.dataset == "OrganMNIST3D"
    assert args.checkpoint == "fake.pth.tar"
    assert args.epochs == 20
    assert args.lr == 1e-3
    assert args.patience == 5


if __name__ == "__main__":
    test_no_stop_while_still_improving()
    test_no_stop_before_patience_window_elapses()
    test_stops_after_patience_epochs_without_improvement()
    test_stops_exactly_at_patience_boundary()
    test_compare_pass_when_auc_meets_or_beats_target()
    test_compare_within_1pct_when_short_by_at_most_one_point()
    test_compare_fail_when_short_by_more_than_one_point()
    test_compare_no_target_generic()
    test_synapse_target_is_0_851()
    test_parse_args_requires_checkpoint()
    test_parse_args_defaults()
    print("test_finetune_downstream_3d.py: all checks passed")
