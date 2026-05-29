from app.services.persistence_service import persistence_service
from app.services.redis_service import redis_service


class TaskService:
    def get_task_status(self, task_id: str, user_id: int) -> dict | None:
        cached = redis_service.get_task_status(task_id)
        record = persistence_service.get_task_record(task_id, user_id=user_id)
        if cached and cached.get("user_id") is not None and int(cached["user_id"]) != user_id:
            cached = None

        if record is None and cached is None:
            return None

        if record is None:
            return {**cached, "redis_cached": True}

        if cached:
            merged = {**record, **cached}
            merged["redis_cached"] = True
            return merged

        record["redis_cached"] = False
        return record


task_service = TaskService()


def get_task_service() -> TaskService:
    return task_service
