"""Match stock qmv arithmetic before and after MLX f8aaf49d (0.32.4).

Use distribution metadata so selecting a transcription never initializes Metal.
The development build at f8aaf49d already carries the 0.32.4 version number.
"""

import re
from importlib.metadata import version

from packaging.version import Version

FLOAT_SUMS = Version(Version(version("mlx")).base_version) >= Version("0.32.4")


def stock_load_vector(header: str, float_sums: bool = FLOAT_SUMS) -> str:
    """Promote each source element before addition, preserving association.

    Restrict rewriting to sum statements: weight products and scaled x_thread
    loads must retain their original arithmetic. Already-promoted sums stay intact.
    """
    if not float_sums:
        return header

    def promote(match):
        return re.sub(r"(?<!float\()\bx\[([^]]+)\]", r"float(x[\1])", match[0])

    return re.sub(r"\bsum\s*\+=.*?;", promote, header, flags=re.DOTALL)
