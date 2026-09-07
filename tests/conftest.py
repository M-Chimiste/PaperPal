"""Set isolated test configuration before imports; no database is needed for unit tests."""
import os
import json
import pathlib
import pytest

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://theseus:theseus@localhost:5434/theseus_test"
)
TEST_DB_NAME = TEST_DB_URL.rsplit("/", 1)[-1]
# Refuse to run against anything that isn't an expendable test database.
assert TEST_DB_NAME.endswith("_test"), (
    f"Test database name must end with '_test', got {TEST_DB_NAME!r}"
)

os.environ["DATABASE_URL"] = TEST_DB_URL
os.environ["DB_POOL_MIN_SIZE"] = "1"
os.environ["DB_POOL_MAX_SIZE"] = "5"
os.environ["DB_POOL_TIMEOUT"] = "10"  # fail fast instead of the 60s default
os.environ["APP_SECRET_KEY"] = "test_secret"
os.environ.setdefault("HF_HUB_OFFLINE", "1")


GOLDEN_DIR = pathlib.Path(__file__).parent / "goldens"

@pytest.fixture(scope="session")
def golden():
    """Golden-file comparator.

    Missing golden -> written and the test passes (review before committing).
    UPDATE_GOLDENS=1 -> rewrite all goldens.
    """

    def check(name: str, value):
        path = GOLDEN_DIR / f"{name}.json"
        serialized = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
        if os.environ.get("UPDATE_GOLDENS") == "1" :
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(serialized)
            print(f"[golden] wrote {path.name}")
            return
        assert path.exists(), f"Missing golden {path.name}; explicitly run UPDATE_GOLDENS=1 to create it"
        expected = path.read_text()
        assert serialized == expected, (
            f"Golden mismatch for {name!r}. If the change is intentional, "
            f"rerun with UPDATE_GOLDENS=1 and review the diff."
        )

    return check
