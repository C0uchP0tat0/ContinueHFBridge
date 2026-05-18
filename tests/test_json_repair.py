"""Tests for JSON repair utilities."""
import pytest

from app.parsing.json_repair import (
    unwrap_json_from_markdown,
    extract_first_json_object,
    strip_concatenated_json_echoes,
    repair_json_string,
    repair_truncated_json,
)


def test_unwrap_json_from_markdown():
    """Test unwrapping JSON from markdown code blocks."""
    # Simple case
    assert unwrap_json_from_markdown('```json\n{"key": "value"}\n```') == '{"key": "value"}'
    
    # No markdown
    assert unwrap_json_from_markdown('{"key": "value"}') == '{"key": "value"}'
    
    # With language tag
    assert unwrap_json_from_markdown('```python\n{"key": "value"}\n```') == '{"key": "value"}'
    
    # Nested markdown
    assert unwrap_json_from_markdown('```\n```json\n{"key": "value"}\n```\n```') == '{"key": "value"}'


def test_extract_first_json_object():
    """Test extracting first JSON object from text."""
    # Simple case
    text = '{"key": "value"} some text'
    result = extract_first_json_object(text)
    assert result == '{"key": "value"}'
    
    # Multiple JSON objects
    text = '{"first": 1} {"second": 2}'
    result = extract_first_json_object(text)
    assert result == '{"first": 1}'
    
    # No JSON
    text = 'just plain text'
    result = extract_first_json_object(text)
    assert result == ''


def test_strip_concatenated_json_echoes():
    """Test stripping concatenated JSON echoes."""
    # Simple echo
    text = '{"key": "value"} {"key": "value"}'
    result = strip_concatenated_json_echoes(text)
    assert result == '{"key": "value"}'
    
    # No echo
    text = '{"key": "value"}'
    result = strip_concatenated_json_echoes(text)
    assert result == '{"key": "value"}'
    
    # Different objects
    text = '{"first": 1} {"second": 2}'
    result = strip_concatenated_json_echoes(text)
    assert result == '{"first": 1}'


def test_repair_json_string():
    """Test repairing JSON strings."""
    # Missing comma
    result = repair_json_string('{"key": "value" "key2": "value2"}')
    assert '"key2"' in result
    
    # Trailing comma
    result = repair_json_string('{"key": "value",}')
    assert result == '{"key": "value"}'
    
    # Valid JSON
    result = repair_json_string('{"key": "value"}')
    assert result == '{"key": "value"}'


def test_repair_truncated_json():
    """Test repairing truncated JSON."""
    # Truncated mid-string
    result = repair_truncated_json('{"key": "value", "key2": "')
    assert result.endswith('}')
    
    # Truncated mid-object
    result = repair_truncated_json('{"key": "value", "key2":')
    assert result.endswith('}')
    
    # Valid JSON
    result = repair_truncated_json('{"key": "value"}')
    assert result == '{"key": "value"}'
    
    # Truncated with multiple nested structures
    result = repair_truncated_json('{"key": {"nested": "value", "key2":')
    assert result.count('}') >= 2
