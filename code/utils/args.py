import argparse

def check_bool_arg(arg):
    if arg.lower() not in {"true", "false"}:
        raise argparse.ArgumentTypeError("true or false expected")
