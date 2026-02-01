"""
API endpoint integration tests.

Covers the request/response contracts for all three routes with the RAG system
mocked at the module level via the api_client / mock_rag_system fixtures:

    POST   /api/query            – query processing and response format
    GET    /api/courses          – course catalog statistics
    DELETE /api/session/{id}     – session lifecycle
"""

import pytest


class TestQueryEndpoint:
    """POST /api/query — the primary user-facing endpoint."""

    def test_query_returns_200_with_full_response(self, api_client):
        """Successful query returns answer, sources, and session_id."""
        response = api_client.post(
            "/api/query",
            json={"query": "What is taught in the test course?"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "This is a mock answer about the course content."
        assert len(body["sources"]) == 2
        assert body["session_id"] == "session_test_1"

    def test_query_creates_session_when_none_provided(self, api_client, mock_rag_system):
        """When session_id is null, endpoint creates one via session_manager."""
        response = api_client.post(
            "/api/query",
            json={"query": "test question", "session_id": None},
        )

        assert response.status_code == 200
        assert response.json()["session_id"] == "session_test_1"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_uses_provided_session_id(self, api_client, mock_rag_system):
        """When session_id is provided, it bypasses create_session and is
        passed directly to rag_system.query."""
        response = api_client.post(
            "/api/query",
            json={"query": "follow-up question", "session_id": "session_existing_42"},
        )

        assert response.status_code == 200
        assert response.json()["session_id"] == "session_existing_42"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with(
            "follow-up question", "session_existing_42"
        )

    def test_query_sources_dict_format(self, api_client):
        """Dict sources are serialized as SourceCitation objects preserving
        both text and url fields."""
        response = api_client.post("/api/query", json={"query": "test"})

        sources = response.json()["sources"]
        assert sources[0] == {
            "text": "Test Course - Lesson 0",
            "url": "https://example.com/lesson/0",
        }
        # url=None is preserved (not omitted)
        assert sources[1] == {"text": "Test Course - Lesson 1", "url": None}

    def test_query_sources_string_format(self, api_client, mock_rag_system):
        """Plain string sources are wrapped as SourceCitation with null url.
        This exercises the else-branch of the source conversion in app.py."""
        mock_rag_system.query.return_value = (
            "Answer with string source.",
            ["Course A - Lesson 1"],
        )

        response = api_client.post("/api/query", json={"query": "test"})

        assert response.status_code == 200
        sources = response.json()["sources"]
        assert len(sources) == 1
        assert sources[0]["text"] == "Course A - Lesson 1"
        assert sources[0]["url"] is None

    def test_query_empty_sources(self, api_client, mock_rag_system):
        """When the RAG system returns no sources, response has an empty list."""
        mock_rag_system.query.return_value = ("Answer with no sources.", [])

        response = api_client.post(
            "/api/query", json={"query": "general question"}
        )

        assert response.status_code == 200
        assert response.json()["sources"] == []

    def test_query_missing_body_returns_422(self, api_client):
        """POST with no JSON body returns 422 Unprocessable Entity."""
        response = api_client.post("/api/query")
        assert response.status_code == 422

    def test_query_missing_required_field_returns_422(self, api_client):
        """POST body without the required 'query' field returns 422."""
        response = api_client.post(
            "/api/query", json={"session_id": "session_1"}
        )
        assert response.status_code == 422

    def test_query_error_detail_format(self, api_client, mock_rag_system):
        """HTTP 500 detail includes both the exception type name and message.
        app.py formats it as '{ExceptionType}: {message}'."""
        mock_rag_system.query.side_effect = RuntimeError("something broke")

        response = api_client.post(
            "/api/query", json={"query": "trigger error"}
        )

        assert response.status_code == 500
        detail = response.json()["detail"]
        assert "RuntimeError" in detail
        assert "something broke" in detail


class TestCoursesEndpoint:
    """GET /api/courses — course catalog statistics."""

    def test_courses_returns_stats(self, api_client):
        """Returns total course count and list of course titles."""
        response = api_client.get("/api/courses")

        assert response.status_code == 200
        body = response.json()
        assert body["total_courses"] == 2
        assert body["course_titles"] == ["Test Course", "Advanced Course"]

    def test_courses_response_schema(self, api_client):
        """Response body contains exactly total_courses and course_titles."""
        response = api_client.get("/api/courses")

        assert response.status_code == 200
        assert set(response.json().keys()) == {"total_courses", "course_titles"}

    def test_courses_empty_catalog(self, api_client, mock_rag_system):
        """When no courses are indexed, returns zero count and empty list."""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        response = api_client.get("/api/courses")

        assert response.status_code == 200
        body = response.json()
        assert body["total_courses"] == 0
        assert body["course_titles"] == []


class TestSessionEndpoint:
    """DELETE /api/session/{session_id} — session lifecycle."""

    def test_delete_session_returns_success(self, api_client, mock_rag_system):
        """Deleting a session returns 200 with status: success and forwards
        the session_id to clear_session."""
        response = api_client.delete("/api/session/session_test_1")

        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_rag_system.session_manager.clear_session.assert_called_once_with(
            "session_test_1"
        )

    def test_delete_nonexistent_session_still_succeeds(
        self, api_client, mock_rag_system
    ):
        """Deleting an unknown session still returns 200.

        app.py treats session cleanup as non-critical — the endpoint swallows
        exceptions and returns success to avoid surfacing internal state to
        clients.
        """
        mock_rag_system.session_manager.clear_session.side_effect = KeyError(
            "not found"
        )

        response = api_client.delete("/api/session/nonexistent_id")

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_delete_session_forwards_path_parameter(
        self, api_client, mock_rag_system
    ):
        """The session_id extracted from the URL path is forwarded correctly."""
        api_client.delete("/api/session/my_custom_session_id")

        mock_rag_system.session_manager.clear_session.assert_called_once_with(
            "my_custom_session_id"
        )
