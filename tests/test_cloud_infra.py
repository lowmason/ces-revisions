import re
from pathlib import Path

NETWORK_TF = Path(__file__).parents[1] / "infra" / "env" / "network.tf"
AWS_SECURITY_GROUP_DESCRIPTION = re.compile(r"[A-Za-z0-9 ._\-:/()#,@\[\]+=&;{}!$*]+")


def test_security_group_descriptions_use_aws_character_set() -> None:
    descriptions = re.findall(
        r'^\s*description\s*=\s*"([^"]*)"', NETWORK_TF.read_text(), re.MULTILINE
    )

    assert descriptions
    assert [
        description
        for description in descriptions
        if AWS_SECURITY_GROUP_DESCRIPTION.fullmatch(description) is None
    ] == []
