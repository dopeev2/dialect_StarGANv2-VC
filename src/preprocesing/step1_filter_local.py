"""
step1_filter_local.py  (v2 — под реальный manifest_raw.csv)
=============================================================
ЭТАП 1 — Локальная фильтрация и аудитирование корпуса.

Поскольку в manifest_raw.csv поля snr_db / duration_sec / rms_db
полностью пустые, скрипт ВЫЧИСЛЯЕТ их из аудиофайлов напрямую,
а затем применяет фильтры.

Итог: manifest_filtered.csv — готов к передаче в Colab для деноизинга.

Требования (локально, без GPU):
    pip install librosa soundfile pandas numpy tqdm

Использование:
    python step1_filter_local.py \
        --manifest   "D:/DIPLOMA/training_corpus/manifest_raw.csv" \
        --corpus_root "D:/DIPLOMA/training_corpus" \
        --out         "D:/DIPLOMA/training_corpus/manifest_filtered.csv" \
        --min_dur 1.5 \
        --max_dur 10.0 \
        --snr_floor 7.0

Примечание о скорости:
    Скрипт читает каждый WAV только для вычисления метрик — без
    записи на диск. На 20к файлов с медианой ~3с займёт ~15–25 мин
    локально (без GPU). Для ускорения используется --workers N.
"""

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ─────────────────────────────────────────────────────────────

TARGET_SR = 24_000

# Дикторы с неоднородным / неречевым контентом — исключаем всегда
EXCLUDE_SPEAKERS = {
    "don_Activity",   # "Смеётся.", "Кашляет.", "Лает собака." и т.п.
    "don_Others",     # агрегированные случайные дикторы
    "pyoza_Other",    # аналогично
    "pyoza_AT",       # всего 64 записи — слишком мало
}

# Транскрипты, указывающие на неречевой контент (паралингвистика)
NONVERBAL_TRANSCRIPTS = {
    "смеётся.", "смеётся", "смех.", "смех",
    "вздыхает.", "вздыхает", "вздох.", "вздох",
    "кашляет.", "кашляет",
    "усмехается.", "усмехается",
    "вспоминает.", "вспоминает",
    "разговоры на фоне.", "разговоры на фоне",
    "лает собака.",
}

# Минимальный объём на диктора для включения в обучение
MIN_SPEAKER_MINUTES = 15.0


# ─────────────────────────────────────────────────────────────
# ВЫЧИСЛЕНИЕ МЕТРИК ИЗ АУДИО
# ─────────────────────────────────────────────────────────────

def compute_audio_metrics(wav_path: str) -> dict:
    """
    Загружает WAV и вычисляет: duration_sec, snr_db, rms_db, clip_ratio.
    Возвращает словарь или None если файл не читается.
    """
    try:
        import soundfile as sf
        import librosa

        y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)

        # Ресемплинг если нужно (только для метрик — не сохраняем)
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR,
                                  res_type="soxr_hq")
            sr = TARGET_SR

        duration = len(y) / sr

        # RMS
        rms = float(np.sqrt(np.mean(y ** 2)))
        rms_db = float(20 * np.log10(max(rms, 1e-10)))

        # SNR — сравниваем громкие и тихие фреймы
        frame_len = 2048
        if len(y) >= frame_len * 2:
            frames = np.lib.stride_tricks.sliding_window_view(
                y, frame_len)[::frame_len // 2]
            rms_frames = np.sqrt(np.mean(frames ** 2, axis=1))
            n = len(rms_frames)
            sorted_r = np.sort(rms_frames)
            signal_rms = float(np.median(sorted_r[int(0.75 * n):]))
            noise_rms  = float(np.median(sorted_r[:max(1, int(0.25 * n))]))
            snr_db = float(np.clip(
                20 * np.log10(max(signal_rms, 1e-10) / max(noise_rms, 1e-10)),
                0, 60
            ))
        else:
            # Короткий файл — SNR через общий RMS
            snr_db = float(np.clip(
                20 * np.log10(max(rms, 1e-10) / 1e-3),
                0, 60
            ))

        # Клиппирование
        clip_ratio = float(np.mean(np.abs(y) > 0.99))

        return {
            "duration_sec": round(duration, 4),
            "snr_db":        round(snr_db, 2),
            "rms_db":        round(rms_db, 2),
            "clip_ratio":    round(clip_ratio, 5),
            "error":         None,
        }

    except Exception as e:
        return {
            "duration_sec": None,
            "snr_db":        None,
            "rms_db":        None,
            "clip_ratio":    None,
            "error":         str(e),
        }


def _worker(args):
    """Воркер для мультипроцессинга."""
    idx, wav_path = args
    metrics = compute_audio_metrics(wav_path)
    return idx, metrics


# ─────────────────────────────────────────────────────────────
# ФИЛЬТРАЦИЯ
# ─────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame,
                  min_snr: float,
                  min_dur: float,
                  max_dur: float) -> pd.DataFrame:

    print(f"\n{'═'*60}")
    print("  ПРИМЕНЕНИЕ ФИЛЬТРОВ")
    print(f"{'═'*60}")
    total = len(df)
    print(f"  Исходно: {total} записей\n")

    def step(name, mask_keep):
        nonlocal df
        n_before = len(df)
        df = df[mask_keep].copy()
        dropped = n_before - len(df)
        icon = "✅" if dropped == 0 else "🗑️ "
        print(f"  {icon} {name:<42} −{dropped:>5} ({dropped/total*100:.1f}%)")

    # 1. Файлы, которые не удалось прочитать
    step("Нечитаемые файлы (ошибка загрузки)",
         df["error"].isna())

    # 2. Исключённые спикеры (неречевые / агрегированные)
    step("Исключённые спикеры (Activity/Others/AT)",
         ~df["speaker_id"].isin(EXCLUDE_SPEAKERS))

    # 3. Неречевые транскрипты (паралингвистика)
    nonverbal_mask = df["transcript"].str.strip().str.lower().isin(NONVERBAL_TRANSCRIPTS)
    step("Паралингвистика (Смеётся/Вздыхает/...)",
         ~nonverbal_mask)

    # 4. [нрзб] в транскрипте
    nrzb_mask = df["has_nrzb"].astype(str).str.lower() == "true"
    step("[нрзб] в транскрипте",
         ~nrzb_mask)

    # 5. Слишком короткие
    step(f"Длительность < {min_dur} с",
         df["duration_sec"] >= min_dur)

    # 6. Слишком низкий SNR (деноизинг не поможет)
    step(f"SNR < {min_snr} дБ (даже после деноизинга)",
         df["snr_db"] >= min_snr)

    # 7. Сильное клиппирование
    step("Клиппирование > 0.5%",
         df["clip_ratio"] < 0.005)

    # 8. Спикеры с малым объёмом данных после всех фильтров
    spk_min = df.groupby("speaker_id")["duration_sec"].sum() / 60
    bad_spk = spk_min[spk_min < MIN_SPEAKER_MINUTES].index
    if len(bad_spk) > 0:
        print(f"\n  ⚠️  Дикторы с объёмом < {MIN_SPEAKER_MINUTES} мин после фильтрации:")
        for sp in sorted(bad_spk):
            print(f"       {sp}: {spk_min[sp]:.1f} мин")
        step(f"Дикторы < {MIN_SPEAKER_MINUTES} мин данных",
             ~df["speaker_id"].isin(bad_spk))
    else:
        print(f"  ✅ {'Все дикторы ≥ '+str(MIN_SPEAKER_MINUTES)+' мин данных':<42}")

    dropped_total = total - len(df)
    print(f"\n  {'─'*54}")
    print(f"  Итого отброшено: {dropped_total} ({dropped_total/total*100:.1f}%)")
    print(f"  Осталось:        {len(df)}")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST СПЛИТ
# ─────────────────────────────────────────────────────────────

def assign_splits(df: pd.DataFrame,
                  val_ratio: float = 0.05,
                  test_ratio: float = 0.05,
                  seed: int = 42) -> pd.DataFrame:
    """
    Стратифицированный сплит по speaker_id.
    Гарантирует, что каждый диктор представлен во всех трёх сплитах.
    """
    import random
    rng = random.Random(seed)

    splits = []
    for spk, grp in df.groupby("speaker_id"):
        idx = grp.index.tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_val  = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))
        for i, orig_idx in enumerate(idx):
            if i < n_test:
                splits.append((orig_idx, "test"))
            elif i < n_test + n_val:
                splits.append((orig_idx, "val"))
            else:
                splits.append((orig_idx, "train"))

    split_series = pd.Series(dict(splits), name="split")
    df["split"] = split_series
    return df


# ─────────────────────────────────────────────────────────────
# СТАТИСТИКА
# ─────────────────────────────────────────────────────────────

def print_statistics(df: pd.DataFrame):
    print(f"\n{'═'*60}")
    print("  СТАТИСТИКА ПОСЛЕ ФИЛЬТРАЦИИ")
    print(f"{'═'*60}")

    total_min = df["duration_sec"].sum() / 60
    print(f"  Записей:        {len(df)}")
    print(f"  Длит. итого:    {total_min:.1f} мин ({total_min/60:.2f} ч)")
    print(f"  Медиана (с):    {df['duration_sec'].median():.2f}")

    print(f"\n  SNR (дБ):")
    print(f"    Среднее:  {df['snr_db'].mean():.1f}")
    print(f"    Медиана:  {df['snr_db'].median():.1f}")
    for lo, hi, label in [(0,10,"<10 🔴"), (10,15,"10–15 🟡"),
                           (15,20,"15–20 🟡"), (20,25,"20–25 ✅"), (25,99,">25 ✅")]:
        cnt = ((df["snr_db"] >= lo) & (df["snr_db"] < hi)).sum()
        bar = "█" * (cnt // 60)
        print(f"    {label:>9}: {cnt:>5}  {bar}")

    print(f"\n  По domain_id:")
    for d, grp in df.groupby("domain_id"):
        mins = grp["duration_sec"].sum() / 60
        print(f"    {d:<20} {len(grp):>5} уттер. | {mins:>6.1f} мин")

    print(f"\n  По дикторам:")
    spk = (df.groupby("speaker_id")["duration_sec"]
             .agg(["count","sum"])
             .assign(mins=lambda x: x["sum"]/60)
             .sort_values("mins", ascending=False))
    for sp, row in spk.iterrows():
        icon = "✅" if row["mins"] >= 45 else ("🟡" if row["mins"] >= 15 else "🔴")
        print(f"    {icon} {sp:<32} {row['count']:>5} уттер. | {row['mins']:>6.1f} мин")

    n_need = (df["snr_db"] < 20).sum()
    n_high = (df["snr_db"] >= 20).sum()
    print(f"\n  Для Colab-деноизинга:")
    print(f"    Нужен деноизинг (SNR < 20 дБ):    {n_need} записей")
    print(f"    Только нормализация (SNR ≥ 20 дБ): {n_high} записей")
    est_min = n_need * 0.35 / 60
    print(f"    Оценочное время на A100:           ~{est_min:.0f} мин")

    if "split" in df.columns:
        print(f"\n  Сплит:")
        for s, cnt in df["split"].value_counts().sort_index().items():
            print(f"    {s:<6}: {cnt}")


# ─────────────────────────────────────────────────────────────
# ТОЧКА ВХОДА
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Этап 1: аудиометрики + фильтрация корпуса"
    )
    parser.add_argument("--manifest",    required=True,
                        help="Путь к manifest_raw.csv")
    parser.add_argument("--corpus_root", required=True,
                        help="Корень training_corpus/ (где лежит папка raw/)")
    parser.add_argument("--out",         required=True,
                        help="Путь для manifest_filtered.csv")
    parser.add_argument("--min_dur",     type=float, default=1.5,
                        help="Мин. длительность записи (с) [1.5]")
    parser.add_argument("--max_dur",     type=float, default=10.0,
                        help="Макс. длительность (с) — длиннее сегментируется в Colab [10.0]")
    parser.add_argument("--snr_floor",   type=float, default=7.0,
                        help="Мин. SNR для включения — ниже денойз не поможет [7.0]")
    parser.add_argument("--workers",     type=int, default=4,
                        help="Параллельных воркеров для чтения WAV [4]")
    parser.add_argument("--cache",       type=str, default=None,
                        help="Путь для кэша метрик (чтобы не пересчитывать при повторном запуске)")
    args = parser.parse_args()

    corpus_root = Path(args.corpus_root)
    out_path    = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Загружаем манифест ────────────────────────────────────
    print(f"\n📂 Загружаем: {args.manifest}")
    df = pd.read_csv(args.manifest)
    print(f"   Загружено {len(df)} записей")
    print(f"   Домены: {dict(df['domain_id'].value_counts())}")

    # ── Кэш метрик ────────────────────────────────────────────
    cache_path = Path(args.cache) if args.cache else out_path.with_suffix(".metrics_cache.csv")
    if cache_path.exists():
        print(f"\n🔄 Загружаем кэш метрик: {cache_path}")
        metrics_df = pd.read_csv(cache_path, index_col="utterance_id")
        # Добавляем только те, которых ещё нет
        missing = df[~df["utterance_id"].isin(metrics_df.index)]
        print(f"   В кэше: {len(metrics_df)}, нужно вычислить ещё: {len(missing)}")
    else:
        metrics_df = pd.DataFrame()
        missing = df
        print(f"\n⚙️  Кэш не найден, вычисляем метрики для всех {len(df)} файлов")

    # ── Вычисляем метрики ─────────────────────────────────────
    if len(missing) > 0:
        tasks = [
            (i, str(corpus_root / row["rel_path"]))
            for i, (_, row) in enumerate(missing.iterrows())
        ]
        results = {}
        print(f"\n⚙️  Читаем {len(tasks)} WAV-файлов ({args.workers} воркеров)...")
        print("   (это займёт ~15–25 мин локально, можно прерваться — кэш сохранится)")

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks),
                               desc="Аудиоаудит", unit="файл"):
                local_idx, metrics = future.result()
                orig_idx = missing.index[local_idx]
                utt_id = missing.loc[orig_idx, "utterance_id"]
                results[utt_id] = metrics

        new_metrics = pd.DataFrame(results).T
        new_metrics.index.name = "utterance_id"

        # Объединяем с кэшем
        if len(metrics_df) > 0:
            metrics_df = pd.concat([metrics_df, new_metrics])
        else:
            metrics_df = new_metrics

        # Сохраняем кэш
        metrics_df.to_csv(cache_path)
        print(f"   💾 Кэш сохранён: {cache_path}")

    # ── Объединяем метрики с манифестом ───────────────────────
    df = df.drop(columns=["duration_sec","snr_db","rms_db",
                           "quality_tier","split"], errors="ignore")
    metrics_aligned = metrics_df.reindex(df["utterance_id"].values)
    df["duration_sec"] = metrics_aligned["duration_sec"].values
    df["snr_db"]       = metrics_aligned["snr_db"].values
    df["rms_db"]       = metrics_aligned["rms_db"].values
    df["clip_ratio"]   = metrics_aligned["clip_ratio"].values
    df["error"]        = metrics_aligned["error"].values

    # Флаг для Colab: нужен ли деноизинг
    df["needs_denoise"] = (df["snr_db"] < 20.0).astype(bool)

    # ── Применяем фильтры ─────────────────────────────────────
    df_filtered = apply_filters(
        df,
        min_snr=args.snr_floor,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
    )

    # ── Assign splits ─────────────────────────────────────────
    df_filtered = assign_splits(df_filtered)

    # ── Статистика ────────────────────────────────────────────
    print_statistics(df_filtered)

    # ── Сохраняем ─────────────────────────────────────────────
    cols_out = [
        "utterance_id", "rel_path", "speaker_id",
        "domain_id", "dialect_group", "birth_year",
        "duration_sec", "snr_db", "rms_db", "clip_ratio",
        "has_nrzb", "needs_denoise", "transcript", "split",
    ]
    cols_out = [c for c in cols_out if c in df_filtered.columns]
    df_filtered[cols_out].to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n✅ Сохранено: {out_path}")
    print(f"   Записей: {len(df_filtered)}")
    print(f"\n   Следующий шаг → загрузи на Google Drive и запусти step2_denoise_colab.ipynb")


if __name__ == "__main__":
    main()
