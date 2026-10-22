#!/usr/bin/env python3
"""Deterministic INVP v1 process fixture. This is not cuRobo."""

import math
import os
import struct
import time


FD = 3
HEADER = struct.Struct("<4sHBBIi")
MODE = os.environ.get("INVAR_PROCESS_FIXTURE_MODE", "normal")


def read_exact(size: int) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        chunk = os.read(FD, size - len(data))
        if not chunk:
            if not data:
                return None
            raise EOFError("truncated input")
        data.extend(chunk)
    return bytes(data)


def read_frame() -> bytes | None:
    prefix = read_exact(4)
    if prefix is None:
        return None
    (length,) = struct.unpack("<I", prefix)
    if not 16 <= length <= 65536:
        raise ValueError("invalid frame length")
    payload = read_exact(length)
    if payload is None:
        raise EOFError("truncated frame")
    return payload


def write_all(data: bytes) -> None:
    while data:
        written = os.write(FD, data)
        if written <= 0:
            raise OSError("short write")
        data = data[written:]


def send_frame(op: int, sequence: int, status: int = 0, body: bytes = b"") -> None:
    payload = bytearray(HEADER.pack(b"INVP", 1, op, 1, sequence, status) + body)
    if MODE == "bad_magic_step" and op == 3:
        payload[0] = 0
    if MODE == "bad_version_step" and op == 3:
        payload[4] = 2
    if MODE == "bad_op_step" and op == 3:
        payload[6] = 4
    if MODE == "bad_role_step" and op == 3:
        payload[7] = 0
    if MODE == "bad_sequence_step" and op == 3:
        payload[8] ^= 1
    if MODE == "bad_status_step" and op == 3:
        payload[12] = 99
    if MODE == "bad_body_step" and op == 3:
        payload.pop()
    if MODE == "nan_pose_step" and op == 3:
        payload[16:24] = struct.pack("<d", math.nan)
    if MODE == "bad_quaternion_step" and op == 3:
        payload[-8:] = struct.pack("<d", 2.0)
    if MODE == "oversized_step" and op == 3:
        write_all(struct.pack("<I", 65537))
        return
    if MODE == "truncated_step" and op == 3:
        write_all(struct.pack("<I", len(payload)) + payload[:8])
        return
    write_all(struct.pack("<I", len(payload)) + payload)


def run() -> None:
    started = False
    configured = 0
    last_sequence = 0
    while True:
        payload = read_frame()
        if payload is None:
            return
        if len(payload) < HEADER.size:
            raise ValueError("short payload")
        magic, version, op, role, sequence, status = HEADER.unpack_from(payload)
        if (magic != b"INVP" or version != 1 or role != 0 or status != 0 or
                sequence <= last_sequence):
            raise ValueError("invalid request header")
        last_sequence = sequence
        body = payload[HEADER.size:]
        if op == 1:
            if started or body:
                send_frame(op, sequence, 2)
            else:
                started = True
                send_frame(op, sequence)
        elif op == 2:
            if not started or configured:
                send_frame(op, sequence, 2)
            elif len(body) < 8:
                send_frame(op, sequence, 1)
            else:
                model_kind, path_len = struct.unpack_from("<II", body)
                path = body[8:]
                try:
                    decoded = path.decode("utf-8", errors="strict")
                    valid_path = bool(decoded) and "\x00" not in decoded
                except UnicodeError:
                    valid_path = False
                if model_kind not in (6, 7) or path_len != len(path) or path_len > 4096 or not valid_path:
                    send_frame(op, sequence, 1)
                else:
                    configured = model_kind
                    send_frame(op, sequence)
        elif op == 3:
            if not configured:
                send_frame(op, sequence, 2)
                continue
            if len(body) < 4:
                send_frame(op, sequence, 1)
                continue
            (count,) = struct.unpack_from("<I", body)
            if count != configured or len(body) != 4 + 8 * count:
                send_frame(op, sequence, 1)
                continue
            joints = struct.unpack_from("<" + "d" * count, body, 4)
            if not all(map(math.isfinite, joints)):
                send_frame(op, sequence, 1)
                continue
            if MODE == "crash_step":
                os._exit(23)
            if MODE == "hang_step":
                stepping_file = os.environ.get("INVAR_PROCESS_FIXTURE_STEPPING_FILE")
                if stepping_file:
                    with open(stepping_file, "w", encoding="ascii") as output:
                        output.write("entered\n")
                time.sleep(10)
            if MODE == "component_error_step":
                send_frame(op, sequence, 3)
                continue
            if MODE == "bad_failure_body_step":
                send_frame(op, sequence, 3, b"x")
                continue
            pose = (*joints[:3], 0.0, 0.0, 0.0, 1.0)
            send_frame(op, sequence, body=struct.pack("<7d", *pose))
            if MODE in ("truncated_step", "oversized_step"):
                return
        elif op == 4:
            if not started or body:
                send_frame(op, sequence, 2)
            else:
                send_frame(op, sequence)
                if MODE == "hang_shutdown":
                    time.sleep(10)
                if MODE == "crash_shutdown":
                    os._exit(24)
                return
        else:
            send_frame(op, sequence, 1)


if __name__ == "__main__":
    pidfile = os.environ.get("INVAR_PROCESS_FIXTURE_PIDFILE")
    if pidfile:
        with open(pidfile, "w", encoding="ascii") as output:
            output.write(f"{os.getpid()}\n")
    run()
