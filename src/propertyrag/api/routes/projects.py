"""Project API routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from propertyrag.api.dependencies import get_db
from propertyrag.api.schemas import ProjectCreate, ProjectResponse, ProjectUpdate
from propertyrag.db.repository import DocumentRepository, ProjectRepository

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    request: ProjectCreate,
    session: AsyncSession = Depends(get_db),
) -> ProjectResponse:
    """
    Create a new project.

    Projects are used to group related documents together.

    Args:
        request: Project creation request.

    Returns:
        Created project.
    """
    repo = ProjectRepository(session)
    project = await repo.create(
        name=request.name,
        description=request.description,
    )

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        document_count=0,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    session: AsyncSession = Depends(get_db),
) -> list[ProjectResponse]:
    """
    List all projects.

    Returns:
        List of projects with document counts.
    """
    project_repo = ProjectRepository(session)
    doc_repo = DocumentRepository(session)

    projects = await project_repo.get_all()

    result = []
    for project in projects:
        docs = await doc_repo.get_by_project(project.id)
        result.append(
            ProjectResponse(
                id=project.id,
                name=project.name,
                description=project.description,
                document_count=len(docs),
                created_at=project.created_at,
                updated_at=project.updated_at,
            )
        )

    return result


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: UUID,
    session: AsyncSession = Depends(get_db),
) -> ProjectResponse:
    """
    Get a project by ID.

    Args:
        project_id: Project ID.

    Returns:
        Project details.
    """
    project_repo = ProjectRepository(session)
    doc_repo = DocumentRepository(session)

    project = await project_repo.get_by_id(project_id)

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    docs = await doc_repo.get_by_project(project_id)

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        document_count=len(docs),
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: UUID,
    request: ProjectUpdate,
    session: AsyncSession = Depends(get_db),
) -> ProjectResponse:
    """
    Update a project.

    Args:
        project_id: Project ID.
        request: Fields to update.

    Returns:
        Updated project.
    """
    project_repo = ProjectRepository(session)
    doc_repo = DocumentRepository(session)

    project = await project_repo.get_by_id(project_id)

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )

    # Update fields
    if request.name is not None:
        project.name = request.name
    if request.description is not None:
        project.description = request.description

    await session.flush()

    docs = await doc_repo.get_by_project(project_id)

    return ProjectResponse(
        id=project.id,
        name=project.name,
        description=project.description,
        document_count=len(docs),
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: UUID,
    session: AsyncSession = Depends(get_db),
) -> None:
    """
    Delete a project and optionally its documents.

    Note: Documents are kept but unlinked from the project.

    Args:
        project_id: Project ID.
    """
    repo = ProjectRepository(session)
    deleted = await repo.delete(project_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found",
        )
