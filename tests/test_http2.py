"""Unit tests for building HTTP transcription requests without hitting a live server."""

import io
from email.parser import BytesParser
from email.policy import default

import requests


def _parse_multipart_request(prepared_request: requests.PreparedRequest):
    """Parse a prepared multipart request into a dict of part name -> payload bytes."""
    content_type = prepared_request.headers["Content-Type"]
    body = prepared_request.body
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + body
    )

    parts: dict[str, bytes] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="Content-Disposition")
        parts[name] = part.get_payload(decode=True)
    return parts


def test_transcription_request_payload():
    """Ensure the transcription request includes expected fields and file content."""
    url = "http://127.0.0.1:8000/v1/audio/transcriptions"
    audio_bytes = b"RIFF....WAVEfmt "  # minimal placeholder bytes; no real audio needed
    file_obj = io.BytesIO(audio_bytes)

    request = requests.Request(
        "POST",
        url,
        data={"model": "ux_speech_grpc_proxy"},
        files={"file": ("speaker1_a_cn_16k.wav", file_obj, "audio/wav")},
        headers={"Authorization": "Bearer <token>"},
    )

    prepared = request.prepare()

    assert prepared.method == "POST"
    assert prepared.url == url
    assert prepared.headers["Authorization"] == "Bearer <token>"
    assert "multipart/form-data" in prepared.headers["Content-Type"]

    parts = _parse_multipart_request(prepared)

    assert parts["model"] == b"ux_speech_grpc_proxy"
    assert parts["file"] == audio_bytes
