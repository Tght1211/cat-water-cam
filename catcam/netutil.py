from __future__ import annotations

import socket
import subprocess


def is_private_lan(ip: str) -> bool:
    """是否 RFC1918 内网地址（10/8、172.16/12、192.168/16）。

    顺带排除 VPN/代理常用的 198.18/15 基准网段——开了 TUN 模式代理时，
    连 8.8.8.8 拿到的会是隧道地址而非真实局域网 IP。
    """
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    return a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168)


def _socket_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return ""
    finally:
        s.close()


def _route_ip() -> str:
    """macOS：查默认路由网卡再取它的 IPv4（绕开 VPN 隧道地址）。其它平台失败即返空。"""
    try:
        out = subprocess.run(
            ["route", "-n", "get", "default"], capture_output=True, text=True, timeout=2
        ).stdout
        iface = ""
        for line in out.splitlines():
            if "interface:" in line:
                iface = line.split(":", 1)[1].strip()
        if iface:
            return subprocess.run(
                ["ipconfig", "getifaddr", iface], capture_output=True, text=True, timeout=2
            ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return ""


def lan_ip() -> str:
    """本机在局域网里的 IP（用于邮件里给出可点的平台地址）。

    优先用真实内网地址：先试 socket 探测，若拿到的是隧道/公网地址，再查默认网卡。
    """
    ip = _socket_ip()
    if is_private_lan(ip):
        return ip
    ip2 = _route_ip()
    if is_private_lan(ip2):
        return ip2
    return ip or ip2 or "127.0.0.1"
