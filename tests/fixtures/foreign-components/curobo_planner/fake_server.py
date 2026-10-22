#!/usr/bin/env python3
"""Explicit deterministic test double; never imports cuRobo."""

import os
import math
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[4] /
                       "examples/cells/reach_place/components/curobo_planner"))
import planner_server as protocol


class FakePlanner:
    def __init__(self, config):
        self.config = config

    def plan(self, request):
        mode = os.environ.get("INVAR_PLANNER_FAKE_MODE", "normal")
        if mode == "crash":
            os._exit(23)
        if mode == "hang":
            marker = os.environ.get("INVAR_PLANNER_FAKE_ENTERED")
            if marker:
                Path(marker).write_text("entered\n")
            time.sleep(10)
        if mode == "fail":
            raise RuntimeError("deterministic planning failure")
        q = list(request["q"])
        last_q, last_dq = q.copy(), [0.0] * len(q)
        if mode in ("edge", "outside_q"):
            last_q[-1] = self.config["limits"][-1][1]
            if mode == "outside_q":
                last_q[-1] = math.nextafter(last_q[-1], math.inf)
        if mode in ("edge", "outside_dq"):
            last_dq[-1] = protocol.VELOCITY_LIMITS[len(q)][-1]
            if mode == "outside_dq":
                last_dq[-1] = math.nextafter(last_dq[-1], math.inf)
        return 0.1, [(0.0, q, [0.0] * len(q)), (0.1, last_q, last_dq)]


if __name__ == "__main__":
    original_send = protocol.send

    def send(op, sequence, status=0, body=b""):
        mode = os.environ.get("INVAR_PLANNER_FAKE_MODE")
        if op == 3 and status == 0:
            if mode == "bad_reply":
                body = struct.pack("<d", float("nan")) + body[8:]
            elif mode in ("reply_bad_q", "reply_bad_dq"):
                changed = bytearray(body)
                count = (len(body) - 28) // 32
                stride = 8 * (1 + 2 * count)
                index = 12 + stride + 8
                if mode == "reply_bad_dq":
                    index += 8 * count
                    value = math.nextafter(protocol.VELOCITY_LIMITS[count][0], math.inf)
                else:
                    value = math.nextafter(2.0, math.inf)
                struct.pack_into("<d", changed, index, value)
                body = bytes(changed)
            elif mode in ("reply_start_edge", "reply_bad_start"):
                changed = bytearray(body)
                value = 1e-6
                if mode == "reply_bad_start":
                    value = math.nextafter(value, math.inf)
                struct.pack_into("<d", changed, 20, value)
                body = bytes(changed)
            if mode in ("reply_bad_length", "reply_bad_version"):
                if mode == "reply_bad_length":
                    body += b"x"
                payload = protocol.HEADER.pack(
                    b"INVP", 99 if mode == "reply_bad_version" else protocol.VERSION,
                    op, 1, sequence, status) + body
                data = memoryview(struct.pack("<I", len(payload)) + payload)
                while data:
                    data = data[os.write(protocol.FD, data):]
                return
        original_send(op, sequence, status, body)

    protocol.send = send
    protocol.serve(FakePlanner)
