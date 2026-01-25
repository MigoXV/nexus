from fastapi import APIRouter

from .depends import configure, shutdown
from .endpoint import realtime_endpoint_worker

router = APIRouter(tags=["Realtime"])

realtime_endpoint = router.websocket("/realtime")(realtime_endpoint_worker)
