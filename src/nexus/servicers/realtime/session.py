import queue
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RealtimeSession:
    """实时会话状态"""

    session_id: Optional[str] = None
    model: str = "gpt-4o-realtime-preview"
    audio_queue: queue.Queue[np.ndarray] = field(default_factory=queue.Queue)
    result_queue: queue.Queue[str] = field(default_factory=queue.Queue)
    sample_rate: int = 16000
