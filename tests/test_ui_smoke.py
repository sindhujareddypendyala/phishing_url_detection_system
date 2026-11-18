"""
UI Smoke Tests using streamlit.testing.v1.AppTest.
Verifies that each page renders without unhandled exceptions and input validation behaves correctly.
"""

import pytest
from streamlit.testing.v1 import AppTest


def test_app_main_navigation_loads():
    """Verify that main app.py entrypoint loads with navigation without exceptions."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"app.py crashed: {at.exception}"


def test_page_scan_url_loads():
    """Verify Scan URL page renders title and inputs cleanly."""
    at = AppTest.from_file("pages/scan_url.py", default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.title) > 0
    assert "URL Threat Inspector" in at.title[0].value


def test_page_bulk_scan_loads():
    """Verify Bulk Scan page renders file uploader without exceptions."""
    at = AppTest.from_file("pages/bulk_scan.py", default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.title) > 0
    assert "Bulk URL Scanner" in at.title[0].value


def test_page_model_performance_loads():
    """Verify Model Performance page renders reports and metrics without exceptions."""
    at = AppTest.from_file("pages/model_performance.py", default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.title) > 0
    assert "Model Performance" in at.title[0].value


def test_page_about_loads():
    """Verify About page renders architecture and specs without exceptions."""
    at = AppTest.from_file("pages/about.py", default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.title) > 0
    assert "About CyberShield" in at.title[0].value


def test_empty_url_validation_message():
    """Verify that clicking scan on an empty input displays a user-friendly warning."""
    at = AppTest.from_file("pages/scan_url.py", default_timeout=30)
    at.run()
    assert not at.exception

    inspect_btn = next((b for b in at.button if b.label == "Scan URL"), None)
    assert inspect_btn is not None
    inspect_btn.click().run()

    assert not at.exception
    assert len(at.warning) > 0
    assert "Please provide a URL string" in at.warning[0].value


def test_invalid_url_validation_message():
    """Malformed URL input should receive a friendly validation message."""
    at = AppTest.from_file("pages/scan_url.py", default_timeout=30)
    at.run()
    at.text_input[0].input("not a url").run()
    inspect_btn = next((b for b in at.button if b.label == "Scan URL"), None)
    assert inspect_btn is not None
    inspect_btn.click().run()
    assert not at.exception
    assert any("Enter a valid URL" in item.value for item in at.error)


def test_excessive_url_length_validation_message():
    """Verify that submitting an overlong URL (>2048 chars) displays a clear error."""
    at = AppTest.from_file("pages/scan_url.py", default_timeout=30)
    at.run()
    assert not at.exception

    long_url = "http://example.com/" + ("a" * 2050)
    at.text_input[0].input(long_url).run()

    inspect_btn = next((b for b in at.button if b.label == "Scan URL"), None)
    assert inspect_btn is not None
    inspect_btn.click().run()

    assert not at.exception
    assert len(at.error) > 0
    assert "maximum permissible length" in at.error[0].value
