from pathlib import Path

# Assemble private markers so the test file does not trip broader text scans itself.
FORBIDDEN_CORE_STRINGS = (
    "her" + "mes",
    "ta" + "rs",
    "hind" + "sight",
    "/home/" + "manfred",
    "/home/" + "pollen",
    "192" + ".168.",
)


def test_community_core_has_no_private_or_agent_import_strings():
    core_dir = Path("src/reachy_agent/runtime/core")
    assert core_dir.exists()
    offenders: list[str] = []
    for path in core_dir.rglob("*.py"):
        text = path.read_text()
        for needle in FORBIDDEN_CORE_STRINGS:
            if needle in text.lower():
                offenders.append(f"{path}:{needle}")
    assert offenders == []
