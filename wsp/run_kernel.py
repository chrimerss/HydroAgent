"""Execute an editor script with the kernel API preloaded (wsp/wsp py ...).

Like a notebook cell: if the script's last statement is an expression, its
value is printed.
"""

import ast
import sys

from wsp import kernel


def main(path: str) -> None:
    src = open(path).read()
    tree = ast.parse(src, filename=path)
    ns = {name: getattr(kernel, name) for name in kernel.__all__}
    ns["__name__"] = "__main__"
    ns["__file__"] = path
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, path, "exec"), ns)
    if last is not None:
        value = eval(compile(last, path, "eval"), ns)
        if value is not None:
            print(repr(value))


if __name__ == "__main__":
    main(sys.argv[1])
