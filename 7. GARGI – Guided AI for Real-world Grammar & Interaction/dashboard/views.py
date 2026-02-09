from fastapi import Depends, Request
from fastapi.templating import Jinja2Templates

from auth.dependencies import get_current_teacher
from dashboard.analytics import load_all_users

templates = Jinja2Templates(directory="templates")


def register_dashboard_routes(app):

    @app.get("/dashboard/ui")
    def dashboard_ui(
        request: Request,
        teacher=Depends(get_current_teacher)
    ):
        users = load_all_users()
        return templates.TemplateResponse(
            "dashboard.html",
            {"request": request, "users": users}
        )

    @app.get("/dashboard/ui/{user_id}")
    def learner_ui(
        user_id: str,
        request: Request,
        teacher=Depends(get_current_teacher)
    ):
        users = load_all_users()
        return templates.TemplateResponse(
            "learner.html",
            {
                "request": request,
                "user_id": user_id,
                "sessions": users.get(user_id, [])
            }
        )
