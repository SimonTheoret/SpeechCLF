import pytest

from inference import download_limited, get_html_content, parse_content


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


def test_get_html_content():
    with pytest.raises(Exception):  # will raise an exception
        get_html_content(
            "https://en.wikipedia.org/wiki/Active_learning_(machine_learning)", 20
        )

    text = get_html_content(  # will not raise an exception
        "https://en.wikipedia.org/wiki/Active_learning_(machine_learning)", 300_000
    )
    assert text is not None  # Verify if content is correctly parsed
    assert "Active learning" in text
    assert "Definitions" in text
    assert "Balance exploration and exploitation" in text
    assert "Settles, Burr" in text
    assert "There are situations in which" in text
