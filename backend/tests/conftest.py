"""Shared test fixtures for RAG chatbot tests."""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from anthropic.types import Message, ContentBlock, TextBlock, ToolUseBlock, Usage
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


@pytest.fixture
def sample_course():
    """Sample course with 3 lessons for testing."""
    return Course(
        title="Test Course",
        link="https://example.com/course",
        instructor="Test Instructor",
        lessons=[
            Lesson(
                lesson_number=0,
                title="Introduction",
                lesson_link="https://example.com/lesson/0"
            ),
            Lesson(
                lesson_number=1,
                title="Advanced Topics",
                lesson_link=None
            ),
            Lesson(
                lesson_number=2,
                title="Best Practices",
                lesson_link="https://example.com/lesson/2"
            )
        ]
    )


@pytest.fixture
def sample_course_file():
    """Path to sample course text file."""
    return Path(__file__).parent / "fixtures" / "sample_course.txt"


@pytest.fixture
def mock_anthropic_response_direct():
    """Mock Anthropic response with direct text (no tool use)."""
    mock_response = Mock(spec=Message)
    mock_response.id = "msg_123"
    mock_response.model = "claude-sonnet-4-20250514"
    mock_response.role = "assistant"
    mock_response.stop_reason = "end_turn"

    # Create text block
    text_block = Mock(spec=TextBlock)
    text_block.type = "text"
    text_block.text = "This is a test response."

    mock_response.content = [text_block]
    mock_response.usage = Mock(spec=Usage)
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50

    return mock_response


@pytest.fixture
def mock_anthropic_response_tool_use():
    """Mock Anthropic response requesting tool use."""
    mock_response = Mock(spec=Message)
    mock_response.id = "msg_456"
    mock_response.model = "claude-sonnet-4-20250514"
    mock_response.role = "assistant"
    mock_response.stop_reason = "tool_use"

    # Create tool use block
    tool_block = Mock(spec=ToolUseBlock)
    tool_block.type = "tool_use"
    tool_block.id = "toolu_789"
    tool_block.name = "search_course_content"
    tool_block.input = {
        "query": "test query",
        "course_name": None,
        "lesson_number": None
    }

    mock_response.content = [tool_block]
    mock_response.usage = Mock(spec=Usage)
    mock_response.usage.input_tokens = 150
    mock_response.usage.output_tokens = 75

    return mock_response


@pytest.fixture
def mock_anthropic_response_empty_content():
    """Mock Anthropic response with empty content list (error case)."""
    mock_response = Mock(spec=Message)
    mock_response.id = "msg_error1"
    mock_response.model = "claude-sonnet-4-20250514"
    mock_response.role = "assistant"
    mock_response.stop_reason = "end_turn"
    mock_response.content = []  # Empty content
    mock_response.usage = Mock(spec=Usage)
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 0

    return mock_response


@pytest.fixture
def mock_anthropic_response_no_text():
    """Mock Anthropic response with content but no text attribute (error case)."""
    mock_response = Mock(spec=Message)
    mock_response.id = "msg_error2"
    mock_response.model = "claude-sonnet-4-20250514"
    mock_response.role = "assistant"
    mock_response.stop_reason = "end_turn"

    # Create malformed content block without text attribute
    malformed_block = Mock()
    malformed_block.type = "unknown"
    # Deliberately not setting .text attribute

    mock_response.content = [malformed_block]
    mock_response.usage = Mock(spec=Usage)
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 0

    return mock_response


@pytest.fixture
def mock_search_results():
    """Mock search results with documents."""
    return SearchResults(
        documents=[
            "This is lesson 0 content about the basics of testing.",
            "This is lesson 1 content covering advanced testing strategies."
        ],
        metadata=[
            {"course_title": "Test Course", "lesson_number": 0, "chunk_index": 0},
            {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0}
        ],
        distances=[0.2, 0.3]
    )


@pytest.fixture
def mock_search_results_empty():
    """Mock empty search results."""
    return SearchResults.empty("No results found")


@pytest.fixture
def mock_chromadb_client():
    """Mock ChromaDB client with collections."""
    mock_client = MagicMock()

    # Mock course_catalog collection
    mock_catalog = MagicMock()
    mock_catalog.count.return_value = 1
    mock_catalog.query.return_value = {
        "ids": [["test_course"]],
        "distances": [[0.1]],
        "metadatas": [[{"title": "Test Course"}]]
    }

    # Mock course_content collection
    mock_content = MagicMock()
    mock_content.count.return_value = 3
    mock_content.query.return_value = {
        "ids": [["chunk_0", "chunk_1"]],
        "distances": [[0.2, 0.3]],
        "documents": [
            [
                "This is lesson 0 content about the basics of testing.",
                "This is lesson 1 content covering advanced testing strategies."
            ]
        ],
        "metadatas": [
            [
                {"course_title": "Test Course", "lesson_number": 0, "chunk_index": 0},
                {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0}
            ]
        ]
    }

    # Mock get_or_create_collection
    def get_collection_side_effect(name):
        if name == "course_catalog":
            return mock_catalog
        elif name == "course_content":
            return mock_content
        raise ValueError(f"Unknown collection: {name}")

    mock_client.get_or_create_collection.side_effect = get_collection_side_effect

    return mock_client


@pytest.fixture
def mock_anthropic_client(mock_anthropic_response_direct):
    """Mock Anthropic client."""
    mock_client = MagicMock()
    mock_messages = MagicMock()
    mock_messages.create.return_value = mock_anthropic_response_direct
    mock_client.messages = mock_messages

    return mock_client


@pytest.fixture
def api_client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from app import app

    return TestClient(app)
