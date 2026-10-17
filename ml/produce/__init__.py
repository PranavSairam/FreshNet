"""Separate FreshNet produce-type classifier (identity only; not freshness).

Nothing in this package imports or modifies the freshness training code,
checkpoints, bundle or manifests. It reads the v3 manifests and writes its own
manifests, runs, checkpoints, metrics and bundle under produce-specific paths.
"""
