"""
AI Generator unit tests.

Tests Claude API interaction with mocked client.
Critical tests target identified error handling gaps at lines 96, 120, 144.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from anthropic.types import Message, TextBlock, ToolUseBlock
from ai_generator import AIGenerator, MAX_TOOL_ROUNDS
from search_tools import ToolManager, CourseSearchTool
from vector_store import SearchResults


class TestAIGeneratorBasic:
    """Test basic AI generator functionality."""

    @pytest.fixture
    def tool_manager(self):
        """Create tool manager with search tool."""
        from vector_store import VectorStore

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.search.return_value = SearchResults.empty("No results")

        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        return manager

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AI generator with mocked client."""
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_anthropic_class.return_value = mock_anthropic_client
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            return generator

    def test_generate_response_no_tools(
        self, ai_generator, mock_anthropic_client, mock_anthropic_response_direct
    ):
        """Test basic response generation without tool use."""
        mock_anthropic_client.messages.create.return_value = mock_anthropic_response_direct

        response = ai_generator.generate_response(query="What is 2+2?", conversation_history="")

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        assert response == "This is a test response."

    def test_generate_response_with_history(
        self, ai_generator, mock_anthropic_client, mock_anthropic_response_direct
    ):
        """Test response generation with conversation history."""
        mock_anthropic_client.messages.create.return_value = mock_anthropic_response_direct

        history = "Previous conversation:\nUser: Hello\nAssistant: Hi there!"

        response = ai_generator.generate_response(
            query="How are you?", conversation_history=history
        )

        assert response is not None
        # Verify history was included in system prompt
        call_args = mock_anthropic_client.messages.create.call_args
        assert "Previous conversation" in call_args.kwargs["system"]


class TestAIGeneratorToolUse:
    """Test AI generator tool use pattern."""

    @pytest.fixture
    def tool_manager_with_results(self, mock_search_results):
        """Create tool manager that returns search results."""
        from vector_store import VectorStore

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.search.return_value = mock_search_results

        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        return manager

    @pytest.fixture
    def ai_generator_with_tool(self):
        """Create AI generator with working search tool."""
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            generator.client = mock_client

            return generator

    def test_generate_response_tool_use(
        self,
        ai_generator_with_tool,
        tool_manager_with_results,
        mock_anthropic_response_tool_use,
        mock_anthropic_response_direct,
    ):
        """Test two-phase pattern: tool request -> tool execution -> final response."""
        mock_client = ai_generator_with_tool.client

        # First call: Claude requests tool use
        # Second call: Claude provides final answer
        mock_client.messages.create.side_effect = [
            mock_anthropic_response_tool_use,
            mock_anthropic_response_direct,
        ]

        response = ai_generator_with_tool.generate_response(
            query="What is in the test course?",
            conversation_history="",
            tools=tool_manager_with_results.get_tool_definitions(),
            tool_manager=tool_manager_with_results,
        )

        assert response is not None
        assert isinstance(response, str)
        # Should have made 2 API calls
        assert mock_client.messages.create.call_count == 2


class TestAIGeneratorErrorHandling:
    """Test error handling in AI generator - these tests target known gaps."""

    @pytest.fixture
    def basic_tool_manager(self):
        """Create basic tool manager."""
        from vector_store import VectorStore

        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.search.return_value = SearchResults.empty("No results")

        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(search_tool)

        return manager

    def test_response_content_missing(
        self, basic_tool_manager, mock_anthropic_response_empty_content
    ):
        """
        ERROR CASE: response.content is empty list.

        Tests line 96 in ai_generator.py:
            return response.content[0].text

        This will raise IndexError if content is empty.
        Expected to FAIL initially, confirming missing error handling.
        """
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response_empty_content
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            # This should not raise an exception
            # Instead should return error message or handle gracefully
            try:
                response = generator.generate_response(query="Test query", conversation_history="")
                # If we get here, error handling was added
                assert (
                    "error" in response.lower() or "empty" in response.lower()
                ), "Should return error message for empty content"
            except IndexError as e:
                pytest.fail(
                    f"IndexError raised - missing error handling at line 96: {e}\n"
                    "Should handle empty response.content gracefully"
                )
            except Exception as e:
                pytest.fail(f"Unexpected exception: {e}")

    def test_response_content_no_text(self, basic_tool_manager, mock_anthropic_response_no_text):
        """
        ERROR CASE: content[0] has no .text attribute.

        Tests line 96 and 144 in ai_generator.py:
            return response.content[0].text

        This will raise AttributeError if content block doesn't have .text.
        Expected to FAIL initially, confirming missing error handling.
        """
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_anthropic_response_no_text
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            try:
                response = generator.generate_response(query="Test query", conversation_history="")
                # If we get here, error handling was added
                assert (
                    "error" in response.lower() or "unexpected" in response.lower()
                ), "Should return error message for malformed response"
            except AttributeError as e:
                pytest.fail(
                    f"AttributeError raised - missing error handling at lines 96/144: {e}\n"
                    "Should validate response.content[0] has .text attribute"
                )
            except Exception as e:
                pytest.fail(f"Unexpected exception: {e}")

    def test_tool_execution_exception(self, basic_tool_manager):
        """
        ERROR CASE: Tool.execute() raises exception.

        Tests lines 118-129 in ai_generator.py where tools are executed.
        If tool execution fails, should be caught and handled gracefully.
        Expected to FAIL initially, confirming missing error handling.
        """
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()

            # Create tool use response
            tool_response = Mock(spec=Message)
            tool_response.stop_reason = "tool_use"

            tool_block = Mock(spec=ToolUseBlock)
            tool_block.type = "tool_use"
            tool_block.id = "toolu_error"
            tool_block.name = "search_course_content"
            tool_block.input = {"query": "test", "course_name": None, "lesson_number": None}

            tool_response.content = [tool_block]

            mock_client.messages.create.return_value = tool_response
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            # Make tool execution fail
            with patch.object(
                basic_tool_manager, "execute_tool", side_effect=Exception("Tool failed")
            ):
                try:
                    response = generator.generate_response(
                        query="Test query", conversation_history=""
                    )
                    # If we get here, error handling was added
                    assert (
                        "error" in response.lower() or "failed" in response.lower()
                    ), "Should return error message when tool execution fails"
                except Exception as e:
                    pytest.fail(
                        f"Exception propagated - missing error handling at lines 118-129: {e}\n"
                        "Should catch and handle tool execution exceptions"
                    )

    def test_tool_result_wrong_format(self, basic_tool_manager):
        """
        ERROR CASE: Tool returns dict instead of string.

        Tests assumption that tool results are strings.
        If tool returns non-string, should handle gracefully.
        """
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()

            # Create tool use response
            tool_response = Mock(spec=Message)
            tool_response.stop_reason = "tool_use"

            tool_block = Mock(spec=ToolUseBlock)
            tool_block.type = "tool_use"
            tool_block.id = "toolu_dict"
            tool_block.name = "search_course_content"
            tool_block.input = {"query": "test", "course_name": None, "lesson_number": None}

            tool_response.content = [tool_block]

            # Create second response
            final_response = Mock(spec=Message)
            final_response.stop_reason = "end_turn"
            text_block = Mock(spec=TextBlock)
            text_block.type = "text"
            text_block.text = "Final answer"
            final_response.content = [text_block]

            mock_client.messages.create.side_effect = [tool_response, final_response]
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            # Make tool return dict instead of string
            with patch.object(
                basic_tool_manager, "execute_tool", return_value={"error": "not a string"}
            ):
                try:
                    response = generator.generate_response(
                        query="Test query", conversation_history=""
                    )
                    # Should handle non-string result
                    assert response is not None
                except TypeError as e:
                    pytest.fail(
                        f"TypeError raised - should handle non-string tool results: {e}\n"
                        "Should convert tool result to string if needed"
                    )

    def test_api_key_invalid(self, basic_tool_manager):
        """Test handling of invalid API key."""
        from anthropic import AuthenticationError

        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            # AuthenticationError requires response and body kwargs
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.headers = {}
            mock_client.messages.create.side_effect = AuthenticationError(
                "Invalid API key",
                response=mock_response,
                body={"error": {"message": "Invalid API key"}},
            )
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            with pytest.raises(AuthenticationError):
                generator.generate_response(query="Test query", conversation_history="")

    def test_api_timeout(self, basic_tool_manager):
        """Test handling of API timeout."""
        from anthropic import APITimeoutError

        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = APITimeoutError("Request timed out")
            mock_anthropic_class.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            with pytest.raises(APITimeoutError):
                generator.generate_response(query="Test query", conversation_history="")


class TestAIGeneratorSequentialToolUse:
    """Test sequential (multi-round) tool calling behaviour."""

    @pytest.fixture
    def ai_generator_with_mock_client(self):
        """Create AI generator and expose the mocked client."""
        with patch("anthropic.Anthropic") as mock_anthropic_class:
            mock_client = MagicMock()
            mock_anthropic_class.return_value = mock_client
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            generator.client = mock_client
            return generator, mock_client

    @pytest.fixture
    def tool_manager(self):
        """Create a tool manager whose execute_tool can be patched per-test."""
        manager = MagicMock(spec=ToolManager)
        manager.execute_tool.return_value = "mock tool result"
        manager.get_tool_definitions.return_value = [
            {
                "name": "search_course_content",
                "description": "Search course content",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_course_outline",
                "description": "Get course outline",
                "input_schema": {
                    "type": "object",
                    "properties": {"course_name": {"type": "string"}},
                    "required": ["course_name"],
                },
            },
        ]
        return manager

    def test_two_sequential_tool_rounds(
        self,
        ai_generator_with_mock_client,
        tool_manager,
        mock_anthropic_response_tool_use,
        mock_anthropic_response_tool_use_outline,
        mock_anthropic_response_direct,
    ):
        """Happy path: Claude uses two tools across two rounds, then synthesizes."""
        generator, mock_client = ai_generator_with_mock_client

        mock_client.messages.create.side_effect = [
            mock_anthropic_response_tool_use,  # round 1: search
            mock_anthropic_response_tool_use_outline,  # round 2: outline
            mock_anthropic_response_direct,  # synthesis
        ]

        response = generator.generate_response(
            query="Tell me about the test course",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # 3 API calls: tool_use -> tool_use -> direct
        assert mock_client.messages.create.call_count == 3
        # execute_tool called once per round
        assert tool_manager.execute_tool.call_count == 2
        # Response is the final synthesized text
        assert response == "This is a test response."
        # Second API call still included tools (tools persist through the loop)
        second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        assert "tools" in second_call_kwargs

    def test_max_rounds_forces_synthesis_without_tools(
        self,
        ai_generator_with_mock_client,
        tool_manager,
        mock_anthropic_response_tool_use,
        mock_anthropic_response_tool_use_outline,
        mock_anthropic_response_direct,
    ):
        """Rounds exhausted: loop breaks, forced synthesis call omits tools."""
        generator, mock_client = ai_generator_with_mock_client

        # Both rounds return tool_use; post-loop forced synthesis returns text
        mock_client.messages.create.side_effect = [
            mock_anthropic_response_tool_use,  # round 1
            mock_anthropic_response_tool_use_outline,  # round 2 (hits MAX_TOOL_ROUNDS)
            mock_anthropic_response_direct,  # forced synthesis
        ]

        response = generator.generate_response(
            query="Tell me everything",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert mock_client.messages.create.call_count == 3
        assert response == "This is a test response."
        # Third (forced synthesis) call must NOT include tools
        third_call_kwargs = mock_client.messages.create.call_args_list[2].kwargs
        assert "tools" not in third_call_kwargs
        assert "tool_choice" not in third_call_kwargs

    def test_tool_failure_terminates_loop(
        self,
        ai_generator_with_mock_client,
        tool_manager,
        mock_anthropic_response_tool_use,
        mock_anthropic_response_direct,
    ):
        """All tools fail in a round: loop breaks early, forced synthesis follows."""
        generator, mock_client = ai_generator_with_mock_client

        tool_manager.execute_tool.side_effect = Exception("Tool failed")

        mock_client.messages.create.side_effect = [
            mock_anthropic_response_tool_use,  # round 1: tool_use (tool will fail)
            mock_anthropic_response_direct,  # forced synthesis
        ]

        response = generator.generate_response(
            query="Search for something",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Loop broke after round 1 (all tools failed), then forced synthesis
        assert mock_client.messages.create.call_count == 2
        # Forced synthesis call must NOT include tools
        second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        assert "tools" not in second_call_kwargs
        # Response still returned successfully
        assert response == "This is a test response."

    def test_single_round_unchanged(
        self,
        ai_generator_with_mock_client,
        tool_manager,
        mock_anthropic_response_tool_use,
        mock_anthropic_response_direct,
    ):
        """Regression: single tool round followed by direct response works as before."""
        generator, mock_client = ai_generator_with_mock_client

        mock_client.messages.create.side_effect = [
            mock_anthropic_response_tool_use,  # round 1: tool_use
            mock_anthropic_response_direct,  # Claude synthesizes directly (no more tools)
        ]

        response = generator.generate_response(
            query="What is in the test course?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        assert mock_client.messages.create.call_count == 2
        assert tool_manager.execute_tool.call_count == 1
        assert response == "This is a test response."
