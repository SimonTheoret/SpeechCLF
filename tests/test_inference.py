import pytest

from inference import download_limited, get_html_content, parse_content, split_strings


def test_parse_content():
    with open("tests/Active_learning.html", "r") as f:
        content = f.read()
        text: str = parse_content(content)
        assert text is not None  # Verify if content is correctly parsed
        assert "Active learning" in text
        assert "Definitions" in text
        assert "Balance exploration and exploitation" in text
        assert "Settles, Burr" in text
        assert "There are situations in which" in text


def test_download_limited():
    with pytest.raises(Exception):
        download_limited(
            "https://en.wikipedia.org/wiki/Active_learning_(machine_learning)",
            20,  # 20 bytes
        )
    download_limited(
        "https://en.wikipedia.org/wiki/Active_learning_(machine_learning)",  # ~250KB
        300_000,  # 300,000 bytes
    )


def test_split_strings():
    text = "this is a very long string without much content, with exactly 75 characters"
    assert len(text) == 75
    assert len(split_strings(text, 25)) == 3
    assert len(split_strings(text, 50)) == 2
    assert len(split_strings(text, 50)[1]) == 25
