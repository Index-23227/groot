"""
STT 기반 음성 명령 입력 — Level 2

마이크 입력 → Whisper → (선택) LLM 정제 → instruction 문자열 반환

사용법:
  # 단독 테스트
  python utils/stt_instruction.py
  python utils/stt_instruction.py --use-llm  # LLM으로 instruction 정제
"""

import os, sys, argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class STTInstruction:
    """마이크 → Whisper → instruction 문자열"""

    def __init__(self, model_size="base", language="ko", use_llm=False):
        self.language = language
        self.use_llm = use_llm
        self._whisper_model = None
        self._model_size = model_size

    def _load_whisper(self):
        if self._whisper_model is None:
            import whisper
            self._whisper_model = whisper.load_model(self._model_size)
            print(f"[STT] Whisper '{self._model_size}' loaded")

    def listen(self, duration=5, sample_rate=16000) -> str:
        """마이크에서 duration초 녹음 → 텍스트 반환"""
        import numpy as np
        try:
            import sounddevice as sd
        except ImportError:
            print("[STT] sounddevice 없음 — 텍스트 입력 fallback")
            return input("[STT] instruction 입력: ").strip()

        print(f"[STT] 🎙️  {duration}초 녹음 시작...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                       channels=1, dtype="float32")
        sd.wait()
        print("[STT] 녹음 완료, 인식 중...")

        audio = audio.flatten()
        # 무음 감지
        if np.abs(audio).max() < 0.01:
            print("[STT] 무음 감지됨")
            return ""

        return self._transcribe(audio)

    def from_file(self, path: str) -> str:
        """WAV 파일 → 텍스트"""
        self._load_whisper()
        result = self._whisper_model.transcribe(path, language=self.language)
        return self._postprocess(result["text"])

    def _transcribe(self, audio) -> str:
        import whisper
        self._load_whisper()
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self._whisper_model.device)
        result = whisper.decode(self._whisper_model, mel,
                                whisper.DecodingOptions(language=self.language))
        return self._postprocess(result.text)

    def _postprocess(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        print(f"[STT] 인식: \"{text}\"")

        if self.use_llm:
            text = self._refine_with_llm(text)

        return text

    def _refine_with_llm(self, raw_text: str) -> str:
        """한국어 음성 → 영어 robot instruction으로 변환"""
        try:
            from anthropic import Anthropic
            client = Anthropic()
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=100,
                messages=[{"role": "user", "content":
                    f'로봇에게 내릴 명령을 영어로 바꿔줘. 짧고 명확하게.\n'
                    f'입력: "{raw_text}"\n'
                    f'영어 명령만 답해:'}],
            )
            refined = resp.content[0].text.strip().strip('"')
            print(f"[STT] 정제: \"{raw_text}\" → \"{refined}\"")
            return refined
        except Exception as e:
            print(f"[STT] LLM 정제 실패 ({e}), 원본 사용")
            return raw_text


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", default="base",
                    choices=["tiny", "base", "small", "medium"])
    p.add_argument("--duration", type=float, default=5)
    p.add_argument("--use-llm", action="store_true")
    p.add_argument("--file", default=None, help="WAV 파일 경로")
    args = p.parse_args()

    stt = STTInstruction(model_size=args.model_size, use_llm=args.use_llm)

    if args.file:
        text = stt.from_file(args.file)
    else:
        text = stt.listen(duration=args.duration)

    print(f"\n결과: \"{text}\"")
