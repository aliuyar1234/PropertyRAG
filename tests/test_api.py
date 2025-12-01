"""Integration tests for the API."""

import pytest
from httpx import AsyncClient


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient) -> None:
        """Test health check returns healthy status."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_root_endpoint(self, client: AsyncClient) -> None:
        """Test root endpoint returns app info."""
        response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "PropertyRAG"
        assert "version" in data


class TestProjectEndpoints:
    """Tests for project endpoints."""

    @pytest.mark.asyncio
    async def test_create_project(self, client: AsyncClient) -> None:
        """Test creating a project."""
        response = await client.post(
            "/api/v1/projects",
            json={"name": "Test Project", "description": "A test project"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Project"
        assert data["description"] == "A test project"
        assert "id" in data

    @pytest.mark.asyncio
    async def test_create_project_minimal(self, client: AsyncClient) -> None:
        """Test creating a project with only required fields."""
        response = await client.post(
            "/api/v1/projects",
            json={"name": "Minimal Project"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Minimal Project"
        assert data["description"] is None

    @pytest.mark.asyncio
    async def test_create_project_validation_error(self, client: AsyncClient) -> None:
        """Test project creation with invalid data."""
        response = await client.post(
            "/api/v1/projects",
            json={"name": ""},  # Empty name should fail
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_list_projects(self, client: AsyncClient) -> None:
        """Test listing projects."""
        # Create a project first
        await client.post("/api/v1/projects", json={"name": "Test"})

        response = await client.get("/api/v1/projects")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_get_project(self, client: AsyncClient) -> None:
        """Test getting a specific project."""
        # Create a project
        create_response = await client.post(
            "/api/v1/projects", json={"name": "Get Test"}
        )
        project_id = create_response.json()["id"]

        response = await client.get(f"/api/v1/projects/{project_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Get Test"

    @pytest.mark.asyncio
    async def test_get_project_not_found(self, client: AsyncClient) -> None:
        """Test getting a non-existent project."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.get(f"/api/v1/projects/{fake_id}")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_update_project(self, client: AsyncClient) -> None:
        """Test updating a project."""
        # Create a project
        create_response = await client.post(
            "/api/v1/projects", json={"name": "Original Name"}
        )
        project_id = create_response.json()["id"]

        response = await client.patch(
            f"/api/v1/projects/{project_id}",
            json={"name": "Updated Name"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"

    @pytest.mark.asyncio
    async def test_delete_project(self, client: AsyncClient) -> None:
        """Test deleting a project."""
        # Create a project
        create_response = await client.post(
            "/api/v1/projects", json={"name": "To Delete"}
        )
        project_id = create_response.json()["id"]

        response = await client.delete(f"/api/v1/projects/{project_id}")

        assert response.status_code == 204

        # Verify it's gone
        get_response = await client.get(f"/api/v1/projects/{project_id}")
        assert get_response.status_code == 404


class TestDocumentEndpoints:
    """Tests for document endpoints."""

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, client: AsyncClient) -> None:
        """Test listing documents when none exist."""
        response = await client.get("/api/v1/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, client: AsyncClient) -> None:
        """Test getting a non-existent document."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.get(f"/api/v1/documents/{fake_id}")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_upload_non_pdf(self, client: AsyncClient) -> None:
        """Test uploading a non-PDF file."""
        response = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"Hello World", "text/plain")},
        )

        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]


class TestQueryEndpoints:
    """Tests for query endpoints."""

    @pytest.mark.asyncio
    async def test_query_validation(self, client: AsyncClient) -> None:
        """Test query validation."""
        # Empty question should fail
        response = await client.post(
            "/api/v1/query",
            json={"question": ""},
        )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_top_k_validation(self, client: AsyncClient) -> None:
        """Test query top_k validation."""
        # top_k out of range should fail
        response = await client.post(
            "/api/v1/query",
            json={"question": "Test question", "top_k": 100},
        )

        assert response.status_code == 422
