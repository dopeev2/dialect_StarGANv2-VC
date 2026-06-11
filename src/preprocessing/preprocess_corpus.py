"""
preprocess_corpus.py
====================
Предобработка диалектного корпуса для StarGANv2-VC.

Требования:
    pip install librosa soundfile pandas numpy tqdm
    pip install deepfilternet          # деноизинг
    pip install torch torchaudio       # для silero-VAD

Использование:
    python preprocess_corpus.py \
        --manifest   "D:/DIPLOMA/dialect_corpus/manifest.csv" \
        --corpus_root "D:/DIPLOMA/dialect_corpus" \
        --out_root    "D:/DIPLOMA/corpus_24k" \
        --min_snr     10.0 \
        --denoise_snr_max 20.0 \
        --target_rms  -20.0 \
        --min_dur     1.0 \
        --max_dur     10.0
"""

import os
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# ПАРАМЕТРЫ
# ─────────────────────────────────────────────────────────────
TARGET_SR   = 24_000   # Гц — требование StarGANv2-VC
TARGET_RMS  = -20.0    # дБ — целевой уровень громкости
RMS_FLOOR   = 1e-5     # нижний порог для нормализации

# ─────────────────────────────────────────────────────────────
# ШАГ 0: Загрузка и фильтрация манифеста
# ─────────────────────────────────────────────────────────────

def load_and_filter_manifest(manifest_path: str,
                              min_snr: float,
                              min_dur: float,
                              max_dur: float) -> pd.DataFrame:
    """
    Загружает manifest.csv и применяет жёсткие фильтры:
    - SNR >= min_snr дБ
    - duration >= min_dur и <= max_dur секунд
    - нет [нрзб] в транскрипте (опционально)
    """
    df = pd.read_csv(manifest_path)
    total = len(df)
    
    # Фильтр 1: SNR
    df_filtered = df[df["snr_db"] >= min_snr].copy()
    n_snr = total - len(df_filtered)
    
    # Фильтр 2: длительность (нижняя граница)
    df_filtered = df_filtered[df_filtered["duration_sec"] >= min_dur]
    n_dur_low = len(df[df["snr_db"] >= min_snr]) - len(df_filtered)
    
    # Фильтр 3: клиппирование
    df_filtered = df_filtered[df_filtered["clip_ratio"] < 0.005]
    n_clip = 0  # у нас всего 3 таких записи

    # Примечание: длинные (> max_dur) НЕ отбрасываем сейчас —
    # они будут сегментированы на этапе обработки
    
    print(f"\n{'─'*50}")
    print(f"Загружено записей:        {total:>6}")
    print(f"Отброшено (SNR < {min_snr}дБ):  {n_snr:>6}")
    print(f"Отброшено (dur < {min_dur}с):   {n_dur_low:>6}")
    print(f"Осталось после фильтрации: {len(df_filtered):>6}")
    print(f"  из них длиннее {max_dur}с: {(df_filtered['duration_sec'] > max_dur).sum():>6} (будут сегментированы)")
    print(f"{'─'*50}")
    
    return df_filtered


# ─────────────────────────────────────────────────────────────
# ШАГ 1: Ресемплинг
# ─────────────────────────────────────────────────────────────

def resample_audio(y: np.ndarray, orig_sr: int,
                   target_sr: int = TARGET_SR) -> np.ndarray:
    """
    Ресемплинг аудио до target_sr.
    Используем librosa с kaiser_best качеством.
    Если sr уже совпадает — возвращаем без изменений.
    """
    if orig_sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr,
                            res_type="soxr_hq")


# ─────────────────────────────────────────────────────────────
# ШАГ 2: Деноизинг (DeepFilterNet)
# ─────────────────────────────────────────────────────────────

def load_denoiser():
    """
    Загружает DeepFilterNet модель.
    Возвращает (model, df_state) или None если не установлен.
    """
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        model, df_state, _ = init_df()
        print("✅ DeepFilterNet загружен")
        return model, df_state
    except ImportError:
        print("⚠️  DeepFilterNet не установлен. Деноиз пропускается.")
        print("   Установите: pip install deepfilternet")
        return None, None


def denoise_audio(y: np.ndarray, sr: int,
                  model, df_state) -> np.ndarray:
    """
    Применяет DeepFilterNet деноиз.
    Если модель не загружена — возвращает исходный сигнал.
    """
    if model is None:
        return y
    
    try:
        import torch
        from df.enhance import enhance
        
        # DeepFilterNet ожидает тензор [channels, samples]
        audio_tensor = torch.from_numpy(y).float().unsqueeze(0)
        enhanced = enhance(model, df_state, audio_tensor)
        return enhanced.squeeze(0).numpy()
    except Exception as e:
        # Если деноиз упал — возвращаем оригинал
        return y


# ─────────────────────────────────────────────────────────────
# ШАГ 3: VAD-обрезка тишины
# ─────────────────────────────────────────────────────────────

def trim_silence_vad(y: np.ndarray,
                     top_db: float = 30.0,
                     pad_ms: int = 50) -> np.ndarray:
    """
    Обрезает тишину в начале и конце записи.
    pad_ms: оставляем небольшой отступ (миллисекунды)
    
    Используем librosa.effects.trim (быстро и без зависимостей).
    Для продакшена можно заменить на silero-VAD.
    """
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    
    # Добавляем небольшой паддинг
    pad_samples = int(TARGET_SR * pad_ms / 1000)
    pad = np.zeros(pad_samples, dtype=y.dtype)
    return np.concatenate([pad, y_trimmed, pad])


# ─────────────────────────────────────────────────────────────
# ШАГ 4: Нормализация RMS
# ─────────────────────────────────────────────────────────────

def normalize_rms(y: np.ndarray,
                  target_db: float = TARGET_RMS) -> np.ndarray:
    """
    Нормализует RMS уровень к целевому значению в дБ.
    
    Формула: gain = 10^((target_db - current_db) / 20)
    Затем применяем gain с ограничением пика до 0.99.
    """
    rms = np.sqrt(np.mean(y ** 2))
    
    if rms < RMS_FLOOR:
        return y  # слишком тихий — не трогаем
    
    current_db = 20.0 * np.log10(rms)
    gain = 10.0 ** ((target_db - current_db) / 20.0)
    
    y_normalized = y * gain
    
    # Предотвращаем клиппирование после усиления
    peak = np.max(np.abs(y_normalized))
    if peak > 0.99:
        y_normalized = y_normalized * (0.99 / peak)
    
    return y_normalized


# ─────────────────────────────────────────────────────────────
# ШАГ 5: Сегментация длинных записей
# ─────────────────────────────────────────────────────────────

def segment_long_audio(y: np.ndarray, sr: int,
                       max_dur: float = 10.0,
                       min_seg_dur: float = 1.0) -> list[np.ndarray]:
    """
    Разбивает длинную запись на сегменты по паузам.
    
    Алгоритм:
    1. Находим речевые интервалы через librosa.effects.split
    2. Объединяем близкие интервалы в сегменты
    3. Возвращаем список сегментов длиной min_seg_dur..max_dur
    
    Если запись <= max_dur — возвращаем как есть.
    """
    duration = len(y) / sr
    
    if duration <= max_dur:
        return [y]
    
    # Получаем речевые интервалы (в сэмплах)
    intervals = librosa.effects.split(y, top_db=35, 
                                      frame_length=2048,
                                      hop_length=512)
    
    if len(intervals) == 0:
        return [y]
    
    # Жадное объединение интервалов в сегменты
    max_samples = int(max_dur * sr)
    min_samples = int(min_seg_dur * sr)
    gap_threshold = int(0.5 * sr)   # паузы > 500 мс — граница сегмента
    
    segments = []
    seg_start = intervals[0][0]
    seg_end   = intervals[0][1]
    
    for start, end in intervals[1:]:
        gap = start - seg_end
        candidate_end = end
        
        if (gap > gap_threshold or 
                (candidate_end - seg_start) > max_samples):
            # Сохраняем текущий сегмент
            seg = y[seg_start:seg_end]
            if len(seg) >= min_samples:
                segments.append(seg)
            seg_start = start
        
        seg_end = end
    
    # Последний сегмент
    seg = y[seg_start:seg_end]
    if len(seg) >= min_samples:
        segments.append(seg)
    
    return segments if segments else [y]


# ─────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ ОБРАБОТКИ ОДНОЙ ЗАПИСИ
# ─────────────────────────────────────────────────────────────

def process_utterance(row: pd.Series,
                      corpus_root: Path,
                      out_root: Path,
                      denoiser,
                      denoise_threshold: float,
                      max_dur: float) -> list[dict]:
    """
    Применяет полный пайплайн к одной записи.
    Возвращает список словарей для обновлённого манифеста
    (может быть > 1 при сегментации).
    """
    wav_path = corpus_root / row["rel_path"]
    
    if not wav_path.exists():
        return []
    
    # ── Загрузка ──────────────────────────────────────────────
    try:
        y, orig_sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)  # стерео → моно
    except Exception as e:
        print(f"  [ERR load] {wav_path.name}: {e}")
        return []
    
    # ── Ресемплинг ────────────────────────────────────────────
    y = resample_audio(y, orig_sr, TARGET_SR)
    
    # ── Деноизинг (только для medium-качества) ───────────────
    model, df_state = denoiser
    if model is not None and row["snr_db"] < denoise_threshold:
        y = denoise_audio(y, TARGET_SR, model, df_state)
    
    # ── VAD-обрезка ───────────────────────────────────────────
    y = trim_silence_vad(y)
    
    # ── Нормализация RMS ──────────────────────────────────────
    y = normalize_rms(y)
    
    # ── Сегментация длинных ───────────────────────────────────
    segments = segment_long_audio(y, TARGET_SR, max_dur=max_dur)
    
    # ── Сохранение сегментов ──────────────────────────────────
    speaker_id = row["speaker_id"]
    out_spk_dir = out_root / speaker_id
    out_spk_dir.mkdir(parents=True, exist_ok=True)
    
    new_records = []
    base_utt_id = row["utterance_id"]
    
    for seg_idx, seg in enumerate(segments):
        # Формируем новый ID
        if len(segments) == 1:
            new_utt_id = base_utt_id
        else:
            new_utt_id = f"{base_utt_id}_seg{seg_idx:02d}"
        
        out_wav_name = f"{new_utt_id}.wav"
        out_wav_path = out_spk_dir / out_wav_name
        rel_path     = f"{speaker_id}/{out_wav_name}"
        
        # Сохраняем как PCM 16-bit WAV (стандарт для речи)
        sf.write(str(out_wav_path), seg, TARGET_SR,
                 subtype="PCM_16")
        
        duration = len(seg) / TARGET_SR
        rms_db = 20.0 * np.log10(max(np.sqrt(np.mean(seg**2)), 1e-10))
        
        new_records.append({
            "utterance_id":  new_utt_id,
            "rel_path":      rel_path,
            "speaker_id":    speaker_id,
            "domain_id":     row["domain_id"],
            "dialect_group": row["dialect_group"],
            "gender":        row["gender"],
            "birth_year":    row["birth_year"],
            "duration_sec":  round(duration, 4),
            "snr_db":        row["snr_db"],     # SNR исходника
            "rms_db":        round(rms_db, 2),  # RMS после нормализации
            "quality_tier":  row["quality_tier"],
            "has_nrzb":      row["has_nrzb"],
            "transcript":    row["transcript"],
            "split":         row["split"],
        })
    
    return new_records


# ─────────────────────────────────────────────────────────────
# ГЕНЕРАЦИЯ train_list.txt / val_list.txt
# ─────────────────────────────────────────────────────────────

def generate_stargan_lists(manifest_path: str,
                            out_root: Path,
                            base_abs_path: str = None):
    """
    Читает обновлённый манифест и генерирует
    train_list.txt и val_list.txt в формате StarGANv2-VC:
    
        /абс/путь/к/файлу.wav|domain_id
    
    base_abs_path: абсолютный путь к корню corpus_24k
                   (можно задать позже при загрузке в Colab)
    """
    df = pd.read_csv(manifest_path)
    
    # Исключаем интервьюеров из диалектных списков (опционально)
    # Закомментируйте следующую строку если нужно включить интервьюеров
    # df_dialect = df[df["dialect_group"] != "interviewer"]
    
    base = base_abs_path or str(out_root)
    
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        
        list_path = out_root / f"{split}_list.txt"
        with open(list_path, "w", encoding="utf-8") as f:
            for _, row in split_df.iterrows():
                abs_path = f"{base}/{row['rel_path']}"
                f.write(f"{abs_path}|{row['domain_id']}\n")
        
        print(f"  {split}_list.txt: {len(split_df)} записей")
    
    # Статистика по доменам в train
    train_df = df[df["split"] == "train"]
    print(f"\nРаспределение по domain_id в train:")
    print(train_df.groupby(["domain_id", "speaker_id"])["utterance_id"]
          .count().to_string())


# ─────────────────────────────────────────────────────────────
# ТОЧКА ВХОДА
# ─────────────────────────────────────────────────────────────

def main(args):
    manifest_path = Path(args.manifest)
    corpus_root   = Path(args.corpus_root)
    out_root      = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # ── Загрузка и фильтрация ─────────────────────────────────
    df = load_and_filter_manifest(
        str(manifest_path),
        min_snr=args.min_snr,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
    )
    
    # ── Загружаем деноизер ────────────────────────────────────
    denoiser = load_denoiser()
    
    # ── Обрабатываем каждую запись ────────────────────────────
    all_records = []
    errors = 0
    
    print(f"\nОбработка {len(df)} записей...")
    
    for _, row in tqdm(df.iterrows(), total=len(df),
                        desc="Preprocessing", unit="utt"):
        try:
            records = process_utterance(
                row=row,
                corpus_root=corpus_root,
                out_root=out_root,
                denoiser=denoiser,
                denoise_threshold=args.denoise_snr_max,
                max_dur=args.max_dur,
            )
            all_records.extend(records)
        except Exception as e:
            errors += 1
            if errors <= 10:  # печатаем только первые 10 ошибок
                print(f"\n  [ERR] {row['utterance_id']}: {e}")
    
    print(f"\nГотово! Записей: {len(all_records)}, ошибок: {errors}")
    
    # ── Сохраняем обновлённый манифест ───────────────────────
    df_out = pd.DataFrame(all_records)
    
    manifest_out = out_root / "manifest_24k.csv"
    df_out.to_csv(manifest_out, index=False, encoding="utf-8")
    print(f"\nМанифест сохранён: {manifest_out}")
    
    # ── Статистика ────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"ИТОГО:")
    print(f"  Записей:      {len(df_out)}")
    print(f"  Общая длит.:  {df_out['duration_sec'].sum()/60:.1f} мин")
    print(f"  По сплитам:")
    print(df_out["split"].value_counts().to_string())
    print(f"\n  По диктору:")
    print(df_out.groupby("speaker_id")["duration_sec"]
          .agg(["count", "sum"])
          .rename(columns={"count": "n_utt", "sum": "total_sec"})
          .assign(total_min=lambda x: (x["total_sec"]/60).round(1))
          .drop(columns="total_sec")
          .to_string())
    
    # ── Генерируем списки для StarGANv2-VC ───────────────────
    print(f"\n{'─'*50}")
    print("Генерируем train/val/test списки...")
    generate_stargan_lists(
        str(manifest_out),
        out_root,
        base_abs_path=args.colab_base_path  # None если локально
    )
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",        required=True,
                        help="Путь к manifest.csv")
    parser.add_argument("--corpus_root",     required=True,
                        help="Корень dialect_corpus/ (содержит audio/)")
    parser.add_argument("--out_root",        required=True,
                        help="Куда сохранять обработанные файлы")
    parser.add_argument("--min_snr",         type=float, default=10.0,
                        help="Минимальный SNR (дБ)")
    parser.add_argument("--denoise_snr_max", type=float, default=20.0,
                        help="Деноизить если SNR < этого порога")
    parser.add_argument("--target_rms",      type=float, default=-20.0,
                        help="Целевой RMS (дБ)")
    parser.add_argument("--min_dur",         type=float, default=1.0,
                        help="Минимальная длина записи (с)")
    parser.add_argument("--max_dur",         type=float, default=10.0,
                        help="Максимальная длина сегмента (с)")
    parser.add_argument("--colab_base_path", type=str,   default=None,
                        help="Абсолютный путь в Colab (напр. /content/drive/...)")
    
    args = parser.parse_args()
    main(args)
