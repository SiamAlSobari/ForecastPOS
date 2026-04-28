from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from app.helpers.config import settings
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        auth_header = request.headers.get("Authorization")

        if not auth_header or auth_header != f"Bearer {settings.api_token}":
            return JSONResponse(
                status_code=401,
                content={"message": "Unauthorized"}
            )

        return await call_next(request)