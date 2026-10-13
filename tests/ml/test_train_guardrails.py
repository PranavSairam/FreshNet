from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from train import count_images, main


def test_train_aborts_without_dataset(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-data"
    code = main(["--data-dir", str(missing), "--num-classes", "3"])
    assert code == 2
    assert not list(tmp_path.glob("**/*.pt"))


def test_count_images_on_empty_class_folders(tmp_path: Path) -> None:
    (tmp_path / "Fresh").mkdir()
    (tmp_path / "Spoiled").mkdir()
    names, total = count_images(tmp_path)
    assert names == ["Fresh", "Spoiled"]
    assert total == 0
