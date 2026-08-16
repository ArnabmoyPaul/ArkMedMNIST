"""test_ema_buffer.py — confirms ema_update_teacher's BatchNorm buffer EMA
(added alongside the existing parameter EMA) actually moves running_mean/
running_var toward the student's by the expected momentum-weighted amount,
leaves num_batches_tracked (an int counter, not a stat) untouched, and
doesn't regress the existing parameter EMA. Run: python test_ema_buffer.py"""
import torch
import torch.nn as nn

from trainer import ema_update_teacher


def test_buffer_ema_moves_by_expected_amount():
    student = nn.BatchNorm3d(4)
    teacher = nn.BatchNorm3d(4)

    student.running_mean.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    student.running_var.data = torch.tensor([1.0, 1.0, 1.0, 1.0])
    teacher.running_mean.data = torch.zeros(4)
    teacher.running_var.data = torch.full((4,), 2.0)

    m = 0.9
    ema_update_teacher(student, teacher, momentum_schedule=[m], it=0)

    expected_mean = m * 0.0 + (1 - m) * student.running_mean
    expected_var = m * 2.0 + (1 - m) * student.running_var
    assert torch.allclose(teacher.running_mean, expected_mean), \
        f"expected {expected_mean}, got {teacher.running_mean}"
    assert torch.allclose(teacher.running_var, expected_var), \
        f"expected {expected_var}, got {teacher.running_var}"


def test_num_batches_tracked_untouched():
    student = nn.BatchNorm3d(4)
    teacher = nn.BatchNorm3d(4)
    student.num_batches_tracked.fill_(50)
    teacher.num_batches_tracked.fill_(7)

    ema_update_teacher(student, teacher, momentum_schedule=[0.9], it=0)

    assert teacher.num_batches_tracked.item() == 7, \
        "num_batches_tracked is an int counter, EMA must not touch it"


def test_parameter_ema_still_works():
    student = nn.BatchNorm3d(4)
    teacher = nn.BatchNorm3d(4)
    student.weight.data.fill_(2.0)
    teacher.weight.data.fill_(0.0)

    ema_update_teacher(student, teacher, momentum_schedule=[0.5], it=0)

    assert torch.allclose(teacher.weight.data, torch.full((4,), 1.0)), \
        f"parameter EMA regressed: {teacher.weight.data}"


if __name__ == "__main__":
    test_buffer_ema_moves_by_expected_amount()
    test_num_batches_tracked_untouched()
    test_parameter_ema_still_works()
    print("test_ema_buffer.py: all checks passed")
