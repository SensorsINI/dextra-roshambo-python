"""jAER SharedMemoryDVSFrameSender client: mmap pixels + optional localhost TCP control.

Protocol (little-endian), two slots:
  offset 0  : 4s magic b'JAER'
  4         : u16 version
  6         : u16 flags
  8         : u32 seq          (written last as publication fence)
  12        : u32 width
  16        : u32 height
  20        : u32 stride_bytes
  24        : u16 channels (1)
  26        : u16 dtype (1 = U8)
  28        : i64 timestamp_us
  36..63    : reserved
  64        : uint8[height*width] row-major, y * width + x

TCP JSON lines on 127.0.0.1 (default port 14100):
  HELLO / FRAME_READY. If TCP is unused, poll both slot headers for a new seq.
"""

from __future__ import annotations

import json
import mmap
import os
import socket
import struct
import time
import atexit
from typing import Optional, Tuple

import numpy as np

from my_logger import my_logger

log = my_logger(__name__)

MAGIC = b"JAER"
HEADER_STRUCT = struct.Struct("<4sHHIIIIHHq")  # 36 bytes; rest of 64 is reserved
HEADER_BYTES = 64
NUM_BUFFERS = 2
DTYPE_U8 = 1
DEFAULT_TCP = "127.0.0.1:14100"


def slot_size(width: int, height: int) -> int:
    return HEADER_BYTES + max(0, width) * max(0, height)


class JaerFrameShm:
    """Attach to a jAER SharedMemoryDVSFrameSender mmap file."""

    def __init__(
        self,
        path: str,
        tcp: Optional[str] = DEFAULT_TCP,
        wait_file_s: float = 60.0,
    ):
        self.path = path
        self.tcp_addr = tcp
        self.wait_file_s = wait_file_s
        self._file = None
        self.mm: Optional[mmap.mmap] = None
        self.sock: Optional[socket.socket] = None
        self.width = 0
        self.height = 0
        self._slot_size = 0
        self._tcp_buf = b""

    def open(self) -> None:
        self._wait_for_file()
        self._map()
        if self.tcp_addr:
            self._connect_tcp()
        atexit.register(self.close)

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self._unmap()

    def _unmap(self) -> None:
        if self.mm is not None:
            try:
                self.mm.close()
            except BufferError:
                pass
            self.mm = None
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def next_frame(
        self, last_seq: int = 0, timeout_s: float = 0.1
    ) -> Optional[Tuple[int, float, np.ndarray]]:
        """Return (seq, receive_time, uint8 HxW copy) or None on timeout.

        receive_time is wall-clock ``time.time()`` when the frame was obtained
        (AER timestamps are not wall-clock).
        """
        deadline = time.time() + timeout_s
        if self.sock is not None:
            remaining = max(0.0, deadline - time.time())
            msg = self._read_tcp_json(timeout_s=remaining)
            if msg and msg.get("type") == "FRAME_READY":
                slot = int(msg["buffer_index"])
                seq, img = self._read_slot(slot)
                if img is not None and seq > last_seq:
                    return seq, time.time(), img
        return self._poll_slots(last_seq, deadline)

    def _wait_for_file(self) -> None:
        t0 = time.time()
        last_log = 0.0
        while True:
            try:
                if os.path.isfile(self.path) and os.path.getsize(self.path) >= HEADER_BYTES * NUM_BUFFERS:
                    return
            except OSError:
                pass
            if time.time() - t0 > self.wait_file_s:
                raise FileNotFoundError(
                    f"jAER mmap file not found or empty after {self.wait_file_s:.0f}s: {self.path}. "
                    "Enable SharedMemoryDVSFrameSender in jAER first."
                )
            if time.time() - last_log > 2.0:
                log.info(f"waiting for jAER mmap file {self.path} ...")
                last_log = time.time()
            time.sleep(0.1)

    def _map(self) -> None:
        self._unmap()
        self._file = open(self.path, "rb")
        self.mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        magic, ver, flags, seq, w, h, stride, ch, dtype, ts = HEADER_STRUCT.unpack(
            self.mm[0 : HEADER_STRUCT.size]
        )
        if magic != MAGIC and seq == 0 and w == 0:
            # file created but first header not filled yet
            w, h = 64, 64
        if w > 0 and h > 0:
            self.width, self.height = w, h
            self._slot_size = slot_size(w, h)
        else:
            self.width = self.height = 64
            self._slot_size = slot_size(64, 64)
        log.info(
            f"mapped {self.path} size={len(self.mm)} slot={self._slot_size} "
            f"(w={self.width} h={self.height} magic={magic!r} ver={ver})"
        )

    def _connect_tcp(self) -> None:
        host, port_s = self.tcp_addr.rsplit(":", 1)
        port = int(port_s)
        try:
            s = socket.create_connection((host, port), timeout=2.0)
            s.settimeout(0.05)
            self.sock = s
            hello = self._read_tcp_json(timeout_s=2.0)
            if hello and hello.get("type") == "HELLO":
                log.info(f"jAER TCP HELLO: {hello}")
                w, h = int(hello.get("width", self.width)), int(hello.get("height", self.height))
                if w > 0 and h > 0:
                    self.width, self.height = w, h
                    self._slot_size = slot_size(w, h)
            else:
                log.info(f"connected to jAER control {self.tcp_addr} (no HELLO yet)")
        except OSError as e:
            log.warning(
                f"could not connect to jAER TCP {self.tcp_addr} ({e}); polling mmap seq instead"
            )
            self.sock = None

    def _read_tcp_json(self, timeout_s: float) -> Optional[dict]:
        if self.sock is None:
            return None
        deadline = time.time() + timeout_s
        while True:
            nl = self._tcp_buf.find(b"\n")
            if nl >= 0:
                line, self._tcp_buf = self._tcp_buf[:nl], self._tcp_buf[nl + 1 :]
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as e:
                    log.warning(f"bad JSON from jAER: {line!r} ({e})")
                    return None
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            try:
                self.sock.settimeout(remaining)
                chunk = self.sock.recv(4096)
            except socket.timeout:
                return None
            except OSError as e:
                log.warning(f"jAER TCP closed ({e}); falling back to mmap poll")
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None
                return None
            if not chunk:
                log.warning("jAER TCP EOF; falling back to mmap poll")
                try:
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None
                return None
            self._tcp_buf += chunk

    def _poll_slots(
        self, last_seq: int, deadline: float
    ) -> Optional[Tuple[int, float, np.ndarray]]:
        while time.time() < deadline:
            best_seq = last_seq
            best_img = None
            for slot in range(NUM_BUFFERS):
                seq, img = self._read_slot(slot)
                if img is not None and seq > best_seq:
                    best_seq = seq
                    best_img = img
            if best_img is not None:
                return best_seq, time.time(), best_img
            time.sleep(0.005)
        return None

    def _read_slot(self, slot: int) -> Tuple[int, Optional[np.ndarray]]:
        if self.mm is None:
            return 0, None
        size = len(self.mm)
        if self._slot_size <= 0 or (slot + 1) * self._slot_size > size:
            try:
                self._map()
            except OSError:
                return 0, None
        off = slot * self._slot_size
        if off + HEADER_STRUCT.size > len(self.mm):
            return 0, None
        magic, ver, flags, seq, w, h, stride, ch, dtype, ts = HEADER_STRUCT.unpack(
            self.mm[off : off + HEADER_STRUCT.size]
        )
        if magic != MAGIC or seq <= 0 or w <= 0 or h <= 0:
            return 0, None
        if w != self.width or h != self.height:
            self.width, self.height = w, h
            self._slot_size = slot_size(w, h)
            off = slot * self._slot_size
        pix_off = off + HEADER_BYTES
        n = w * h
        if pix_off + n > len(self.mm):
            return seq, None
        img = np.frombuffer(self.mm, dtype=np.uint8, count=n, offset=pix_off).reshape(h, w).copy()
        return seq, img
