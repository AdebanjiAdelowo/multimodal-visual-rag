"""Run the Kaggle command-line client with hostname resolution restricted to IPv4.

Uploads go to Google Cloud Storage. On a network whose IPv6 route silently drops
packets the client hangs there instead of falling back, so `kaggle datasets
create` and `kaggle kernels push` never return. Usage:

    python kaggle/kaggle_ipv4.py kernels status <owner>/<slug>

Run it with an interpreter that has the `kaggle` package installed.
"""

import socket
import sys

_resolve = socket.getaddrinfo


def _ipv4_only(host, port, family=0, *args, **kwargs):
    return _resolve(host, port, socket.AF_INET, *args, **kwargs)


socket.getaddrinfo = _ipv4_only

from kaggle.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = "kaggle"
    main()
