"""Tests for tool validation."""
import pytest

from app.tools.validation import validate_and_fix_tool_calls
from app.tools.continue_defaults import DEFAULT_CONTINUE_TOOLS


def test_validate_tool_calls_name_alias():
    """Test tool name alias resolution."""
    tool_calls = [
        {
            "function": {
                "name": "edit_file",
                "arguments": '{"filepath": "test.py", "changes": "print(1)"}'
            }
        }
    ]
    result = validate_and_fix_tool_calls(tool_calls, DEFAULT_CONTINUE_TOOLS)
    assert len(result) == 1
    assert result[0]["function"]["name"] == "edit_existing_file"


def test_validate_tool_calls_quote_stripping():
    """Test quote stripping from string arguments."""
    tool_calls = [
        {
            "function": {
                "name": "edit_existing_file",
                "arguments": '{"filepath": "\\"test.py\\"", "changes": "\\"print(1)\\""}'
            }
        }
    ]
    result = validate_and_fix_tool_calls(tool_calls, DEFAULT_CONTINUE_TOOLS)
    args = result[0]["function"]["arguments"]
    assert '"test.py"' not in args
    assert "test.py" in args


def test_validate_tool_calls_create_to_edit():
    """Test create_new_file to edit_existing_file conversion."""
    tool_calls = [
        {
            "function": {
                "name": "create_new_file",
                "arguments": '{"filepath": "test.py", "contents": "print(1)"}'
            }
        }
    ]
    result = validate_and_fix_tool_calls(tool_calls, DEFAULT_CONTINUE_TOOLS)
    assert len(result) == 1
    assert result[0]["function"]["name"] == "edit_existing_file"
    args = result[0]["function"]["arguments"]
    assert "changes" in args
    assert "contents" not in args


def test_validate_tool_calls_filepath_normalization():
    """Test filepath normalization (strip ./ prefix)."""
    tool_calls = [
        {
            "function": {
                "name": "edit_existing_file",
                "arguments": '{"filepath": "./test.py", "changes": "print(1)"}'
            }
        }
    ]
    result = validate_and_fix_tool_calls(tool_calls, DEFAULT_CONTINUE_TOOLS)
    args = result[0]["function"]["arguments"]
    assert args["filepath"] == "test.py"


def test_validate_tool_calls_empty_changes_placeholder():
    """Test placeholder for empty changes in edit_existing_file."""
    tool_calls = [
        {
            "function": {
                "name": "edit_existing_file",
                "arguments": '{"filepath": "test.py", "changes": ""}'
            }
        }
    ]
    result = validate_and_fix_tool_calls(tool_calls, DEFAULT_CONTINUE_TOOLS)
    args = result[0]["function"]["arguments"]
    assert args["changes"] == "Please describe the changes to make"


def test_validate_tool_calls_old_string_new_string_merge():
    """Test merging old_string/new_string into changes."""
    tool_calls = [
        {
            "function": {
                "name": "edit_existing_file",
                "arguments": '{"filepath": "test.py", "old_string": "old", "new_string": "new"}'
            }
        }
    ]
    result = validate_and_fix_tool_calls(tool_calls, DEFAULT_CONTINUE_TOOLS)
    args = result[0]["function"]["arguments"]
    assert "changes" in args
    assert "old_string" not in args
    assert "new_string" not in args
    assert "Replace:" in args["changes"]


def test_validate_tool_calls_ls_dirpath_fix():
    """Test fixing bare '.' for ls dirPath."""
    tool_calls = [
        {
            "function": {
                "name": "ls",
                "arguments": '{"dirPath": "."}'
            }
        }
    ]
    result = validate_and_fix_tool_calls(tool_calls, DEFAULT_CONTINUE_TOOLS)
    args = result[0]["function"]["arguments"]
    assert args["dirPath"] == "./"


def test_validate_tool_calls_unknown_tool_skipped():
    """Test that unknown tools are skipped."""
    tool_calls = [
        {
            "function": {
                "name": "unknown_tool",
                "arguments": '{}'
            }
        }
    ]
    result = validate_and_fix_tool_calls(tool_calls, DEFAULT_CONTINUE_TOOLS)
    assert len(result) == 0
