"""
Environment diagnostic tests.

Run these tests FIRST to verify system prerequisites before testing code logic.
Tests check for API keys, database state, dependencies, and configuration.
"""

import pytest
import os
from pathlib import Path
import chromadb
from anthropic import Anthropic


class TestEnvironmentSetup:
    """Test that the environment is properly configured."""

    def test_anthropic_api_key_exists(self):
        """Verify ANTHROPIC_API_KEY is set in environment."""
        from config import config

        api_key = config.ANTHROPIC_API_KEY
        assert api_key is not None, "ANTHROPIC_API_KEY not found in config"
        assert len(api_key) > 0, "ANTHROPIC_API_KEY is empty"
        assert api_key.startswith("sk-ant-"), "ANTHROPIC_API_KEY has invalid format"

    def test_anthropic_api_key_valid(self):
        """Test that API key works with a simple API call."""
        from config import config

        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        try:
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            assert response.content, "API returned empty response"
        except Exception as e:
            pytest.fail(f"API key validation failed: {e}")

    def test_chromadb_directory_exists(self):
        """Verify ChromaDB directory exists."""
        chroma_path = Path(__file__).parent.parent / "chroma_db"
        assert chroma_path.exists(), f"ChromaDB directory not found at {chroma_path}"
        assert chroma_path.is_dir(), f"{chroma_path} is not a directory"

    def test_chromadb_collections_structure(self):
        """Validate ChromaDB has required collections with correct structure."""
        chroma_path = Path(__file__).parent.parent / "chroma_db"
        if not chroma_path.exists():
            pytest.skip("ChromaDB directory does not exist")

        client = chromadb.PersistentClient(path=str(chroma_path))

        # Check course_catalog collection
        try:
            catalog = client.get_collection("course_catalog")
            assert catalog is not None, "course_catalog collection not found"
        except Exception as e:
            pytest.fail(f"Failed to access course_catalog: {e}")

        # Check course_content collection
        try:
            content = client.get_collection("course_content")
            assert content is not None, "course_content collection not found"
        except Exception as e:
            pytest.fail(f"Failed to access course_content: {e}")

    def test_chromadb_has_data(self):
        """Verify ChromaDB collections contain documents."""
        chroma_path = Path(__file__).parent.parent / "chroma_db"
        if not chroma_path.exists():
            pytest.skip("ChromaDB directory does not exist")

        client = chromadb.PersistentClient(path=str(chroma_path))

        try:
            catalog = client.get_collection("course_catalog")
            catalog_count = catalog.count()
            assert catalog_count > 0, f"course_catalog is empty (count: {catalog_count})"

            content = client.get_collection("course_content")
            content_count = content.count()
            assert content_count > 0, f"course_content is empty (count: {content_count})"

            print(f"\nChromaDB status:")
            print(f"  - course_catalog: {catalog_count} courses")
            print(f"  - course_content: {content_count} chunks")

        except Exception as e:
            pytest.fail(f"Failed to query ChromaDB collections: {e}")

    def test_course_documents_exist(self):
        """Check that course documents directory exists and has files."""
        docs_path = Path(__file__).parent.parent.parent / "docs"
        assert docs_path.exists(), f"docs directory not found at {docs_path}"
        assert docs_path.is_dir(), f"{docs_path} is not a directory"

        # Check for .txt files
        txt_files = list(docs_path.glob("*.txt"))
        assert len(txt_files) > 0, f"No .txt files found in {docs_path}"

        print(f"\nFound {len(txt_files)} course document(s):")
        for f in txt_files:
            print(f"  - {f.name}")

    def test_embedding_model_available(self):
        """Verify embedding model can be loaded."""
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            assert model is not None, "Failed to load embedding model"

            # Test encoding
            test_text = "This is a test sentence."
            embedding = model.encode([test_text])
            assert embedding is not None, "Failed to generate embedding"
            assert len(embedding) > 0, "Embedding is empty"

        except Exception as e:
            pytest.fail(f"Embedding model test failed: {e}")

    def test_dependencies_installed(self):
        """Verify all required dependencies can be imported."""
        required_imports = [
            "fastapi",
            "uvicorn",
            "anthropic",
            "chromadb",
            "sentence_transformers",
            "pydantic",
        ]

        missing = []
        for module_name in required_imports:
            try:
                __import__(module_name)
            except ImportError:
                missing.append(module_name)

        assert len(missing) == 0, f"Missing dependencies: {', '.join(missing)}"


class TestConfiguration:
    """Test configuration values are correct."""

    def test_config_values_loaded(self):
        """Verify config.py loads successfully with expected values."""
        try:
            from config import config

            # Check critical config values exist
            assert hasattr(config, "ANTHROPIC_MODEL"), "ANTHROPIC_MODEL not defined"
            assert hasattr(config, "CHUNK_SIZE"), "CHUNK_SIZE not defined"
            assert hasattr(config, "MAX_RESULTS"), "MAX_RESULTS not defined"
            assert hasattr(config, "CHROMA_PATH"), "CHROMA_PATH not defined"

            # Validate values
            assert config.CHUNK_SIZE > 0, f"Invalid CHUNK_SIZE: {config.CHUNK_SIZE}"
            # Note: MAX_RESULTS is 0 in current config (means unlimited)

            print(f"\nConfiguration:")
            print(f"  - Model: {config.ANTHROPIC_MODEL}")
            print(f"  - Chunk size: {config.CHUNK_SIZE}")
            print(f"  - Max results: {config.MAX_RESULTS}")
            print(f"  - ChromaDB path: {config.CHROMA_PATH}")

        except Exception as e:
            pytest.fail(f"Failed to load config: {e}")

    def test_chroma_path_matches_directory(self):
        """Verify CHROMA_PATH in config matches actual directory."""
        from config import config

        config_path = Path(config.CHROMA_PATH)
        expected_path = Path(__file__).parent.parent / "chroma_db"

        # Check if paths match (resolve to absolute for comparison)
        assert (
            config_path.resolve() == expected_path.resolve()
        ), f"CHROMA_PATH mismatch: config={config_path}, expected={expected_path}"
