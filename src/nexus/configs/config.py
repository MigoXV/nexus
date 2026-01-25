from dataclasses import dataclass, field


@dataclass
class NexusConfig:
    asr_grpc_addr: str
    tts_base_url: str
    tts_api_key: str
    chat_base_url: str
    chat_api_key: str
    
    asr_interim_results: bool = False
