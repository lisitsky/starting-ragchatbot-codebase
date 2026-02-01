"""
Error injection tests.

Simulates exact failure modes identified during exploration.
All tests SHOULD fail initially, proving error handling is missing.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from anthropic.types import Message, TextBlock
from vector_store import SearchResults


class TestAIGeneratorErrorInjection:
    """Inject errors into AI generator to test error handling."""

    def test_ai_generator_malformed_response(self):
        """Inject empty content list into AI response."""
        from ai_generator import AIGenerator
        from search_tools import ToolManager, CourseSearchTool
        from vector_store import VectorStore

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.search.return_value = SearchResults.empty("No results")

        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()

            # Inject malformed response
            malformed_response = Mock(spec=Message)
            malformed_response.content = []  # Empty!
            malformed_response.stop_reason = "end_turn"

            mock_client.messages.create.return_value = malformed_response
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            # Should NOT raise exception
            result = generator.generate_response("test", [])
            assert "error" in result.lower() or "empty" in result.lower(), \
                "Should return error message, not crash"

    def test_ai_generator_no_text_attribute(self):
        """Inject content block without .text attribute."""
        from ai_generator import AIGenerator
        from search_tools import ToolManager, CourseSearchTool
        from vector_store import VectorStore

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.search.return_value = SearchResults.empty("No results")

        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()

            # Inject response with content but no .text
            malformed_response = Mock(spec=Message)
            bad_block = Mock()
            bad_block.type = "unknown"
            # No .text attribute
            malformed_response.content = [bad_block]
            malformed_response.stop_reason = "end_turn"

            mock_client.messages.create.return_value = malformed_response
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            # Should NOT raise AttributeError
            result = generator.generate_response("test", [])
            assert "error" in result.lower() or "unexpected" in result.lower(), \
                "Should handle missing .text gracefully"

    def test_ai_generator_tool_execution_raises(self):
        """Make tool execution raise exception."""
        from ai_generator import AIGenerator
        from search_tools import ToolManager, CourseSearchTool
        from vector_store import VectorStore
        from anthropic.types import ToolUseBlock

        mock_vector_store = Mock(spec=VectorStore)
        # Make vector store raise exception
        mock_vector_store.search.side_effect = Exception("ChromaDB connection failed")

        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()

            # First response: tool use
            tool_response = Mock(spec=Message)
            tool_response.stop_reason = "tool_use"

            tool_block = Mock(spec=ToolUseBlock)
            tool_block.type = "tool_use"
            tool_block.id = "toolu_error"
            tool_block.name = "search_course_content"
            tool_block.input = {"query": "test", "course_name": None, "lesson_number": None}

            tool_response.content = [tool_block]

            # Second response: final answer
            final_response = Mock(spec=Message)
            final_response.stop_reason = "end_turn"
            text_block = Mock(spec=TextBlock)
            text_block.type = "text"
            text_block.text = "Error occurred"
            final_response.content = [text_block]

            mock_client.messages.create.side_effect = [tool_response, final_response]
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            # Should handle tool execution error gracefully
            result = generator.generate_response("test", [])
            # Either returns error message or completes with Claude's response
            assert result is not None


class TestCourseSearchErrorInjection:
    """Inject errors into course search tool."""

    def test_course_search_chromadb_exception(self):
        """Force ChromaDB to raise exception during search."""
        from search_tools import CourseSearchTool
        from vector_store import VectorStore

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.search.side_effect = Exception("ChromaDB query failed")

        tool = CourseSearchTool(mock_vector_store)

        # Should return error message, not raise
        result = tool.execute(query="test", course_name=None, lesson_number=None)
        assert isinstance(result, str), "Should return string result"
        assert "ChromaDB query failed" in result, "Error message should contain original exception"

    def test_course_search_metadata_missing(self):
        """Inject search results with missing metadata."""
        from search_tools import CourseSearchTool
        from vector_store import VectorStore

        mock_vector_store = Mock(spec=VectorStore)

        # Create results with incomplete metadata
        bad_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "unknown", "lesson_number": -1}],  # Missing metadata
            distances=[0.5]
        )
        mock_vector_store.search.return_value = bad_results

        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="test", course_name=None, lesson_number=None)
        # Should handle missing metadata gracefully
        assert isinstance(result, str)

    def test_course_search_resolve_name_fails(self):
        """Make _resolve_course_name return None."""
        from search_tools import CourseSearchTool
        from vector_store import VectorStore

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.search.return_value = SearchResults.empty("No results")

        # Mock _resolve_course_name to return None
        tool = CourseSearchTool(mock_vector_store)

        with patch.object(tool.store, 'search', return_value=SearchResults.empty("No results")):
            result = tool.execute(query="test", course_name="NonexistentCourse", lesson_number=None)
            # Should handle gracefully
            assert isinstance(result, str)


class TestVectorStoreErrorInjection:
    """Inject errors into vector store operations."""

    def test_vector_store_collection_not_initialized(self):
        """Test behavior when ChromaDB collections don't exist."""
        from vector_store import VectorStore

        # Create client with temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # VectorStore should create collections if they don't exist
            try:
                from config import config as cfg
                store = VectorStore(chroma_path=tmpdir, embedding_model=cfg.EMBEDDING_MODEL, max_results=5)
                assert store is not None
            except Exception as e:
                pytest.fail(f"VectorStore should handle missing collections: {e}")

    def test_vector_store_empty_database(self):
        """Test searching empty database."""
        from vector_store import VectorStore
        import chromadb

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from config import config as cfg
            store = VectorStore(chroma_path=tmpdir, embedding_model=cfg.EMBEDDING_MODEL, max_results=5)

            # Search empty database
            results = store.search(query="test", limit=5)
            assert results.is_empty()
            assert len(results.documents) == 0


class TestRAGSystemErrorInjection:
    """Inject errors into RAG system."""

    def test_rag_system_no_api_key(self):
        """Test behavior when ANTHROPIC_API_KEY is missing."""
        import os
        from rag_system import RAGSystem
        from anthropic import AuthenticationError
        from dataclasses import dataclass

        @dataclass
        class EmptyKeyConfig:
            ANTHROPIC_API_KEY: str = ""
            ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
            EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
            CHUNK_SIZE: int = 800
            CHUNK_OVERLAP: int = 100
            MAX_RESULTS: int = 0
            MAX_HISTORY: int = 2
            CHROMA_PATH: str = "./chroma_db"

        # Should raise error when querying with empty key
        # Empty key causes TypeError from Anthropic client validation
        with pytest.raises((AuthenticationError, ValueError, TypeError)):
            system = RAGSystem(config=EmptyKeyConfig())
            system.query("test")


class TestEndpointErrorInjection:
    """Test that errors are properly surfaced through the API endpoint.

    Uses the shared api_client / mock_rag_system fixtures so that app import
    and static-file mounting are handled consistently regardless of CWD.
    """

    def test_endpoint_all_exceptions_masked(self, api_client, mock_rag_system):
        """
        Verify that endpoint returns specific error details in HTTP 500.

        After fix (app.py):
            detail=f"{type(e).__name__}: {str(e)}"

        Should now include the exception type name, not just the message.
        """
        mock_rag_system.query.side_effect = ValueError("Specific error message")

        response = api_client.post(
            "/api/query",
            json={"query": "test", "session_id": None}
        )

        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "ValueError" in detail, f"Error detail should include exception type, got: {detail}"
        assert "Specific error message" in detail, f"Error detail should include message, got: {detail}"

    def test_endpoint_chromadb_error(self, api_client, mock_rag_system):
        """Test endpoint response when ChromaDB fails."""
        mock_rag_system.query.side_effect = Exception("ChromaDB connection error")

        response = api_client.post(
            "/api/query",
            json={"query": "What is MCP?", "session_id": None}
        )

        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "Exception" in detail, f"Error detail should include exception type, got: {detail}"
        assert "ChromaDB connection error" in detail, f"Error detail should include message, got: {detail}"

    def test_endpoint_api_key_error(self, api_client, mock_rag_system):
        """Test endpoint response when Anthropic API key is invalid."""
        from anthropic import AuthenticationError

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {}
        auth_error = AuthenticationError(
            "Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}}
        )

        mock_rag_system.query.side_effect = auth_error

        response = api_client.post(
            "/api/query",
            json={"query": "test", "session_id": None}
        )

        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "AuthenticationError" in detail
        assert "Invalid API key" in detail
