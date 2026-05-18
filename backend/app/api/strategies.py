"""GET /api/strategies — list available preset strategies + their param schema."""

from fastapi import APIRouter

from app.core.runner import list_strategies

router = APIRouter(prefix="/api", tags=["strategies"])


@router.get("/strategies")
def get_strategies():
    return {"strategies": list_strategies()}
