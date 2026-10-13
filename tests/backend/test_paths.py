from app.core.config import BACKEND_DIR, REPO_ROOT, Settings


def test_repo_root_is_the_monorepo() -> None:
    assert BACKEND_DIR.name == "backend"
    assert (REPO_ROOT / "docs" / "PROJECT_SPEC.md").is_file()
    settings = Settings(sqlite_path="data/local/freshnet.sqlite")
    assert settings.sqlite_file == (REPO_ROOT / "data" / "local" / "freshnet.sqlite").resolve()
