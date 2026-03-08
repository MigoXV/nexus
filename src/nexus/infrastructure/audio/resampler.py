"""
流式 PCM16 重采样器，基于 libsamplerate (samplerate) 实现。

使用有状态的 samplerate.Resampler，在连续的 chunk 之间维护滤波器状态,
避免逐块独立重采样导致的拼接伪影和音质退化。
"""

import asyncio

import numpy as np
import samplerate


class StreamingResampler:
    """
    流式 PCM16 重采样器。

    内部使用 libsamplerate 的有状态 API，每次 ``process()`` 调用之间
    会保留滤波器历史状态，从而实现真正的在线（online）重采样，
    不会在 chunk 边界产生不连续。

    Parameters
    ----------
    input_rate : int
        输入音频采样率，例如 48000。
    output_rate : int
        输出音频采样率，例如 24000。
    channels : int
        声道数，默认 1（单声道）。
    converter_type : str
        libsamplerate 转换器类型，默认 ``"sinc_fastest"``。
        可选: ``"sinc_best"``, ``"sinc_medium"``, ``"sinc_fastest"``,
        ``"zero_order_hold"``, ``"linear"``。
        对于实时语音场景，``"sinc_fastest"`` 在延迟和音质之间取得了
        最佳平衡；``"sinc_best"`` 适用于离线高保真场景。
    """

    # PCM16: 每个采样点 2 字节 (int16)
    BYTES_PER_SAMPLE = 2

    def __init__(
        self,
        input_rate: int,
        output_rate: int,
        channels: int = 1,
        converter_type: str = "sinc_fastest",
    ):
        if input_rate <= 0 or output_rate <= 0:
            raise ValueError(f"Sample rates must be positive: input={input_rate}, output={output_rate}")

        self._ratio = output_rate / input_rate
        self._channels = channels
        self._resampler = samplerate.Resampler(converter_type, channels=channels)
        self._leftover = b""

    @property
    def ratio(self) -> float:
        return self._ratio

    def process(self, pcm_bytes: bytes, end_of_data: bool = False) -> bytes:
        """
        处理一段 PCM16 音频数据并返回重采样后的 PCM16 数据。

        Parameters
        ----------
        pcm_bytes : bytes
            输入的 PCM16 LE 音频数据（int16 little-endian）。
        end_of_data : bool
            是否为最后一块数据。当为 True 时，会冲刷滤波器内部缓冲区。

        Returns
        -------
        bytes
            重采样后的 PCM16 LE 音频数据。可能返回 b"" 如果输入过短。
        """
        # 拼接上次残留的不完整采样字节
        data = self._leftover + pcm_bytes

        # 确保字节数是 sample 对齐的 (channels * 2 bytes per sample)
        frame_size = self.BYTES_PER_SAMPLE * self._channels
        remainder = len(data) % frame_size
        if remainder:
            self._leftover = data[-remainder:]
            data = data[:-remainder]
        else:
            self._leftover = b""

        if not data and not end_of_data:
            return b""

        if data:
            # int16 -> float32，归一化到 [-1.0, 1.0]
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            if self._channels > 1:
                # reshape 为 (frames, channels) 以满足 libsamplerate 要求
                samples = samples.reshape(-1, self._channels)
        else:
            # end_of_data 但无新数据——用空数组冲刷
            if self._channels > 1:
                samples = np.empty((0, self._channels), dtype=np.float32)
            else:
                samples = np.empty(0, dtype=np.float32)

        # libsamplerate 有状态处理
        resampled = self._resampler.process(samples, self._ratio, end_of_input=end_of_data)

        if resampled.size == 0:
            return b""

        # float32 -> int16，clip 防止溢出
        resampled_int16 = np.clip(resampled * 32768.0, -32768, 32767).astype(np.int16)
        return resampled_int16.tobytes()

    def flush(self) -> bytes:
        """
        冲刷滤波器内部状态，返回剩余的重采样数据。

        应在音频流结束时调用一次。
        """
        return self.process(b"", end_of_data=True)

    # ------------------------------------------------------------------
    # 异步接口 – 将 CPU 密集的重采样卸载到线程池，避免阻塞事件循环
    # ------------------------------------------------------------------

    async def aprocess(self, pcm_bytes: bytes, end_of_data: bool = False) -> bytes:
        """``process()`` 的异步版本，在线程池中执行以避免阻塞事件循环。"""
        return await asyncio.to_thread(self.process, pcm_bytes, end_of_data)

    async def aflush(self) -> bytes:
        """``flush()`` 的异步版本，在线程池中执行以避免阻塞事件循环。"""
        return await asyncio.to_thread(self.flush)

    def reset(self) -> None:
        """重置滤波器状态，可用于开始新的音频流。"""
        self._resampler.reset()
        self._leftover = b""
