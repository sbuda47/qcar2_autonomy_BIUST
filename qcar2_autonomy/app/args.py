#!/usr/bin/env python3
import argparse

def build_parser():
    p = argparse.ArgumentParser()

    # Keep your existing args here later, but for now we only need these:
    p.add_argument("--model-path", type=str, default=None, help="Path to YOLO .pt model")
    p.add_argument("--no-yolo", action="store_true", help="Disable YOLO road-sign detection (lane-follow only)")

    # (Optional) add your usual args later (qcar_id, ports, etc.)
    return p

def parse_args(argv=None):
    return build_parser().parse_args(argv)
