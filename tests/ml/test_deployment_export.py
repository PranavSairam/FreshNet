from pathlib import Path

import torch

from deployment.export_bundle import (
    ORIGINAL_SPLIT_FINGERPRINT,
    V3_FINGERPRINT,
    hold_block_reason,
    main,
)


def test_hold_blocks_original_fingerprint_only(monkeypatch) -> None:
    monkeypatch.setattr("deployment.export_bundle.downstream_hold_reason", lambda: "paused")
    assert hold_block_reason(ORIGINAL_SPLIT_FINGERPRINT) == "paused"
    assert hold_block_reason(None) == "paused"
    assert hold_block_reason("") == "paused"
    assert hold_block_reason(V3_FINGERPRINT) is None


def test_export_cli_refuses_original_fingerprint(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("deployment.export_bundle.downstream_hold_reason", lambda: "paused")
    checkpoint = tmp_path / "original.pt"
    torch.save({"split_fingerprint": ORIGINAL_SPLIT_FINGERPRINT, "smoke_test": False}, checkpoint)
    bundle = tmp_path / "bundle"
    code = main(["--checkpoint", str(checkpoint), "--bundle-dir", str(bundle)])
    assert code == 3
    assert "downstream_hold" in capsys.readouterr().out
    assert not bundle.exists()
