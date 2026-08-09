"""test_get_args_parser_argv.py — Run: python test_get_args_parser_argv.py"""
import sys
from main_ark import get_args_parser


def test_explicit_empty_argv_ignores_process_sys_argv():
    old_argv = sys.argv
    try:
        sys.argv = ['ipykernel_launcher.py', '-f', '/some/kernel-connection.json']
        args = get_args_parser(argv=[])
        assert args.model_name == "swin_base"  # default, not crashed on '-f'
    finally:
        sys.argv = old_argv


def test_argv_none_still_reads_sys_argv_for_backward_compat():
    old_argv = sys.argv
    try:
        sys.argv = ['main_ark.py', '--model', 'swin_tiny']
        args = get_args_parser(argv=None)
        assert args.model_name == "swin_tiny"
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    test_explicit_empty_argv_ignores_process_sys_argv()
    test_argv_none_still_reads_sys_argv_for_backward_compat()
    print("test_get_args_parser_argv.py: all checks passed")
