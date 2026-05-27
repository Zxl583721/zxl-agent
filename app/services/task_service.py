from app.services.persistence_service import persistence_service
from app.services.redis_service import redis_service


class TaskService:
    def get_task_status(self, task_id: str) -> dict | None:
        cached = redis_service.get_task_status(task_id)
        record = persistence_service.get_task_record(task_id)

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

