"""
download_librispeech_ru.py
==========================
Скачивает записи нормативных дикторов из istupakov/russian_librispeech
и сохраняет как wav файлы по папкам дикторов + CSV манифест с транскриптами.

Использование:
    # Тест — только 20 записей на диктора:
    python download_librispeech_ru.py ^
        --out УКАЖИТЕ_ПАПКУ ^
        --max_per_speaker 20

    # Полная загрузка:
    python download_librispeech_ru.py ^
        --out УКАЖИТЕ_ПАПКУ ^
        --max_per_speaker 500

Установка:
    pip install datasets soundfile numpy tqdm
"""

import argparse
import csv
import io
import warnings
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────────────────────

DATASET_ID   = "istupakov/russian_librispeech"
SPLIT        = "train"
SPEAKERS     = ["8086", "9014", "8169", "295"]

MIN_DURATION = 1.0    # сек
MAX_DURATION = 15.0   # сек
MIN_SCORE    = -0.5   # чем ближе к 0, тем чище запись


# ─────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────

def get_speaker_id(audio_filepath: str) -> str:
    """audio/SPEAKER/CHAPTER/file.wav → SPEAKER"""
    parts = audio_filepath.replace("\\", "/").split("/")
    return parts[1] if len(parts) > 2 else "unknown"


def decode_audio(audio_field: dict):
    """
    Декодирует аудио из HuggingFace Audio поля (decode=False).
    Возвращает (numpy array float32, sample_rate) или (None, None).
    """
    try:
        audio_bytes = audio_field.get("bytes")
        if audio_bytes:
            buf = io.BytesIO(audio_bytes)
            y, sr = sf.read(buf, dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            return y, sr
    except Exception:
        pass
    return None, None


# ─────────────────────────────────────────────────────────────
# ОСНОВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────

def download(out_root: Path, max_per_speaker: int):
    from datasets import load_dataset, Audio

    out_root.mkdir(parents=True, exist_ok=True)

    for spk in SPEAKERS:
        (out_root / f"librispeech_{spk}").mkdir(exist_ok=True)

    print(f"Подключаемся к {DATASET_ID} (streaming)...")
    ds = load_dataset(DATASET_ID, split=SPLIT, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    saved   = {spk: 0 for spk in SPEAKERS}
    skipped = {spk: 0 for spk in SPEAKERS}
    errors  = 0

    # Манифест пишем построчно — не теряем данные при прерывании
    manifest_path = out_root / "manifest_librispeech.csv"
    manifest_file = open(manifest_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(manifest_file, fieldnames=[
        "utterance_id", "rel_path", "speaker_id",
        "dialect_group", "duration_sec", "score", "transcript",
    ])
    writer.writeheader()

    print(f"\nДикторы: {SPEAKERS}")
    print(f"Фильтры: duration {MIN_DURATION}–{MAX_DURATION}с, score ≥ {MIN_SCORE}")
    print(f"Лимит: {max_per_speaker} записей на диктора\n")

    pbar = tqdm(desc="Скачивание", unit="файл")

    try:
        for row in ds:
            filepath = row.get("audio_filepath", "")
            spk      = get_speaker_id(filepath)

            if spk not in SPEAKERS:
                continue

            if saved[spk] >= max_per_speaker:
                if all(saved[s] >= max_per_speaker for s in SPEAKERS):
                    break
                continue

            duration = float(row.get("duration", 0))
            if duration < MIN_DURATION or duration > MAX_DURATION:
                skipped[spk] += 1
                continue

            score = float(row.get("score", -999))
            if score < MIN_SCORE:
                skipped[spk] += 1
                continue

            transcript = str(row.get("text", "")).strip()
            if not transcript:
                skipped[spk] += 1
                continue

            y, sr = decode_audio(row["audio"])
            if y is None:
                errors += 1
                continue

            stem     = Path(filepath).stem
            utt_id   = f"librispeech_{spk}_{stem}"
            wav_name = f"{utt_id}.wav"
            out_path = out_root / f"librispeech_{spk}" / wav_name
            rel_path = f"librispeech_{spk}/{wav_name}"

            try:
                sf.write(str(out_path), y, sr, subtype="PCM_16")
            except Exception:
                errors += 1
                continue

            writer.writerow({
                "utterance_id":  utt_id,
                "rel_path":      rel_path,
                "speaker_id":    f"librispeech_{spk}",
                "dialect_group": "standard",
                "duration_sec":  round(duration, 4),
                "score":         round(score, 4),
                "transcript":    transcript,
            })
            manifest_file.flush()

            saved[spk] += 1
            pbar.update(1)
            pbar.set_postfix({s: saved[s] for s in SPEAKERS})

    finally:
        pbar.close()
        manifest_file.close()

    # ── Итоговая статистика ───────────────────────────────────
    print(f"\n{'='*52}")
    print("ИТОГО:")
    total = 0
    for spk in SPEAKERS:
        n = saved[spk]
        total += n
        print(f"  librispeech_{spk:<6}  {n:>4} файлов  "
              f"(пропущено: {skipped[spk]})")

    print(f"\n  Всего сохранено:      {total} файлов")
    print(f"  Ошибок декодирования: {errors}")
    print(f"  Манифест:             {manifest_path}")
    print(f"\n  Структура:")
    for spk in SPEAKERS:
        print(f"    {out_root}/librispeech_{spk}/")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Скачивание нормативных дикторов из Russian LibriSpeech"
    )
    parser.add_argument(
        "--out", required=True,
        help="Выходная папка (напр. D:/DIPLOMA/data_scraping/librispeech_ru)"
    )
    parser.add_argument(
        "--max_per_speaker", type=int, default=500,
        help="Макс. записей на диктора (по умолчанию: 500)"
    )
    args = parser.parse_args()

    download(
        out_root=Path(args.out),
        max_per_speaker=args.max_per_speaker,
    )


if __name__ == "__main__":
    main()
