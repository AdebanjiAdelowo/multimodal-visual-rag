"""mmrag: multimodal visual retrieval-augmented generation."""

import os

__version__ = "0.1.0"


def force_ipv4() -> None:
    """Resolve hostnames to IPv4 only.

    Some networks advertise an IPv6 route that silently drops packets. Python's
    HTTP clients then hang on every connection instead of falling back, which
    stalls dataset and model downloads. Enable with MMRAG_FORCE_IPV4=1.
    """
    import socket

    if getattr(socket.getaddrinfo, "_mmrag_ipv4", False):
        return
    original = socket.getaddrinfo

    def ipv4_only(host, port, family=0, *args, **kwargs):
        return original(host, port, socket.AF_INET, *args, **kwargs)

    ipv4_only._mmrag_ipv4 = True
    socket.getaddrinfo = ipv4_only


if os.environ.get("MMRAG_FORCE_IPV4") == "1":
    force_ipv4()
