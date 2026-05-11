def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"


class TranscriptFormatter:
    def to_string(self, segments: list[dict]) -> str:
        lines = []
        current_speaker = None

        for seg in segments:
            speaker = seg.get("speaker", "INCONNU")
            start = _fmt_time(seg["start"])
            end = _fmt_time(seg["end"])
            text = seg["text"]

            if speaker != current_speaker:
                lines.append(f"\n[{speaker}]")
                current_speaker = speaker

            lines.append(f"  [{start} --> {end}]  {text}")

        return "\n".join(lines).strip()
