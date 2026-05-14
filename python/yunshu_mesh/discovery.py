"""Yunshu Mesh — Node discovery via mDNS (zeroconf) or UDP broadcast.

Discovers peer Yunshu nodes on the local network for cluster formation.
Primary: zeroconf (mDNS/DNS-SD) for automatic discovery.
Fallback: UDP broadcast for environments without mDNS support.
"""
from __future__ import annotations

import json
import logging
import socket
import struct
import threading
import time
from typing import Callable, Optional

from .node import MeshNode, MeshNodeState, NodeCapabilities

logger = logging.getLogger(__name__)

_SERVICE_TYPE = "_yunshu._tcp.local."
_DISCOVERY_PORT = 7998


class NodeDiscovery:
    """Discovers Yunshu nodes on the local network.

    Strategy:
    1. Try zeroconf (mDNS) for automatic service discovery
    2. Fall back to UDP broadcast if zeroconf unavailable
    3. Advertise local node + browse for peers
    """

    def __init__(self, port: int = 8000, discovery_port: int = _DISCOVERY_PORT):
        self.port = port
        self.discovery_port = discovery_port
        self._discovered_nodes: dict[str, MeshNode] = {}
        self._on_discovered_callbacks: list[Callable] = []
        self._on_lost_callbacks: list[Callable] = []
        self._running = False
        self._local_node: Optional[MeshNode] = None
        self._zeroconf = None
        self._browser = None
        self._udp_socket: Optional[socket.socket] = None
        self._udp_thread: Optional[threading.Thread] = None
        self._use_zeroconf = False

    def start(self, local_node: MeshNode) -> None:
        self._local_node = local_node
        self._running = True
        self._try_start_zeroconf(local_node)
        if not self._use_zeroconf:
            self._start_udp(local_node)

    def _try_start_zeroconf(self, local_node: MeshNode) -> None:
        try:
            import zeroconf
            from zeroconf import Zeroconf, ServiceInfo, ServiceBrowser
        except ImportError:
            logger.info("zeroconf not available, using UDP broadcast discovery")
            return

        try:
            self._zeroconf = Zeroconf()
            service_name = f"yunshu-{local_node.node_id}.{_SERVICE_TYPE}"
            info = ServiceInfo(
                _SERVICE_TYPE,
                name=service_name,
                addresses=[socket.inet_aton(local_node.ip)],
                port=local_node.port,
                properties={
                    "node_id": local_node.node_id,
                    "hostname": local_node.hostname,
                    "chip": local_node.capabilities.chip,
                    "memory_gb": str(local_node.capabilities.total_memory_gb),
                    "gpu_cores": str(local_node.capabilities.gpu_cores),
                },
            )
            self._zeroconf.register_service(info)

            class _Listener:
                def __init__(self, discovery):
                    self._discovery = discovery

                def add_service(self, zc, type_, name):
                    info = zc.get_service_info(type_, name)
                    if info:
                        node = self._discovery._service_info_to_node(info)
                        if node and node.node_id != local_node.node_id:
                            self._discovery._add_discovered(node)

                def update_service(self, zc, type_, name):
                    pass

                def remove_service(self, zc, type_, name):
                    node_id = name.split("-")[1].split(".")[0] if "-" in name else name
                    self._discovery._remove_discovered(node_id)

            self._browser = ServiceBrowser(self._zeroconf, _SERVICE_TYPE, _Listener(self))
            self._use_zeroconf = True
            logger.info(f"mDNS discovery started: {service_name}")
        except Exception as e:
            logger.warning(f"Failed to start mDNS discovery: {e}")
            if self._zeroconf:
                self._zeroconf.close()
                self._zeroconf = None

    def _service_info_to_node(self, info) -> Optional[MeshNode]:
        try:
            props = {}
            if info.properties:
                for k, v in info.properties.items():
                    props[k.decode() if isinstance(k, bytes) else k] = v.decode() if isinstance(v, bytes) else v

            addresses = info.parsed_addresses()
            ip = addresses[0] if addresses else "0.0.0.0"

            return MeshNode(
                node_id=props.get("node_id", "unknown"),
                hostname=props.get("hostname", ""),
                ip=ip,
                port=info.port or 8000,
                state=MeshNodeState.READY,
                capabilities=NodeCapabilities(
                    chip=props.get("chip", ""),
                    total_memory_gb=float(props.get("memory_gb", 0)),
                    gpu_cores=int(props.get("gpu_cores", 0)),
                ),
            )
        except Exception as e:
            logger.debug(f"Failed to parse service info: {e}")
            return None

    def _start_udp(self, local_node: MeshNode) -> None:
        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._udp_socket.bind(("", self.discovery_port))
        except OSError:
            logger.warning(f"Failed to bind UDP discovery port {self.discovery_port}")
            return

        self._udp_socket.settimeout(2.0)

        def _listen():
            announce = json.dumps({
                "node_id": local_node.node_id,
                "hostname": local_node.hostname,
                "ip": local_node.ip,
                "port": local_node.port,
                "chip": local_node.capabilities.chip,
                "memory_gb": local_node.capabilities.total_memory_gb,
                "gpu_cores": local_node.capabilities.gpu_cores,
            }).encode()

            while self._running:
                # Send announcement
                try:
                    self._udp_socket.sendto(announce, ("<broadcast>", self.discovery_port))
                except Exception:
                    logger.debug("failed to send UDP announcement", exc_info=True)
                # Listen for peers
                try:
                    data, addr = self._udp_socket.recvfrom(4096)
                    peer = json.loads(data.decode())
                    if peer.get("node_id") != local_node.node_id:
                        node = MeshNode(
                            node_id=peer["node_id"],
                            hostname=peer.get("hostname", ""),
                            ip=peer.get("ip", addr[0]),
                            port=peer.get("port", 8000),
                            state=MeshNodeState.READY,
                            capabilities=NodeCapabilities(
                                chip=peer.get("chip", ""),
                                total_memory_gb=peer.get("memory_gb", 0),
                                gpu_cores=peer.get("gpu_cores", 0),
                            ),
                        )
                        self._add_discovered(node)
                except socket.timeout:
                    continue
                except Exception:
                    logger.debug("failed to receive UDP peer packet", exc_info=True)
                    continue

        self._udp_thread = threading.Thread(target=_listen, daemon=True)
        self._udp_thread.start()
        logger.info("UDP broadcast discovery started")

    def _add_discovered(self, node: MeshNode) -> None:
        is_new = node.node_id not in self._discovered_nodes
        self._discovered_nodes[node.node_id] = node
        if is_new:
            logger.info(f"Discovered node: {node.hostname} ({node.ip}:{node.port})")
            for cb in self._on_discovered_callbacks:
                try:
                    cb(node)
                except Exception:
                    logger.debug("on_discovered callback failed", exc_info=True)

    def _remove_discovered(self, node_id: str) -> None:
        node = self._discovered_nodes.pop(node_id, None)
        if node:
            for cb in self._on_lost_callbacks:
                try:
                    cb(node)
                except Exception:
                    logger.debug("on_lost callback failed", exc_info=True)

    def stop(self) -> None:
        self._running = False
        if self._zeroconf:
            try:
                self._zeroconf.close()
            except Exception:
                logger.debug("failed to close zeroconf", exc_info=True)
            self._zeroconf = None
        if self._udp_socket:
            try:
                self._udp_socket.close()
            except Exception:
                logger.debug("failed to close UDP socket", exc_info=True)
        if self._udp_thread:
            self._udp_thread.join(timeout=3)
        logger.info("Node discovery stopped")

    def on_node_discovered(self, callback: Callable) -> None:
        self._on_discovered_callbacks.append(callback)

    def on_node_lost(self, callback: Callable) -> None:
        self._on_lost_callbacks.append(callback)

    def get_discovered_nodes(self) -> list[MeshNode]:
        return list(self._discovered_nodes.values())
