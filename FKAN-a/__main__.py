"""
FKAN-a: DTI Prediction
"""

import argparse

from fkan import __version__, cli


def main():
    parser = argparse.ArgumentParser(description=__doc__)   # argparse.ArgumentParser 是 Python 标准库 argparse 模块中的一个类，它用于处理命令行参数和选项

    parser.add_argument(
        "-v", "--version", action="version", version=f"fkan {__version__}"
    )
    # parser.add_argument(
    #     "-c",
    #     "--citation",
    #     action=CitationAction,
    #     nargs=0,
    #     help="show program's citation and exit",
    # )

    subparsers = parser.add_subparsers(title="fkan Commands", dest="cmd")
    subparsers.required = True

    modules = {
        "train": (cli.train, cli.train_parser),
        "download": (cli.download, cli.download_parser),
        # "embed": embed,
        # "evaluate": evaluate,
        # "predict": predict,
    }

    for name, (main_func, args_func) in modules.items():
        sp = subparsers.add_parser(name, description=main_func.__doc__)
        args_func(sp)
        sp.set_defaults(main_func=main_func)

    args = parser.parse_args()
    args.main_func(args)


if __name__ == "__main__":
    main()
