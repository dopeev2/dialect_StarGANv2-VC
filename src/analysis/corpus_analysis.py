#!/usr/bin/env python3
"""
corpus_analysis.py — детальный анализ диалектного корпуса перед очисткой.

Структура:
    Блок 0.  Установка и импорты
    Блок 1.  Загрузка manifest и сбор акустических метрик
    Блок 2.  Общая статистика (таблица)
    Блок 3.  Распределение длительностей
    Блок 4.  SNR — отношение сигнал/шум
    Блок 5.  RMS — уровень громкости
    Блок 6.  VAD-ratio — доля речи в записи
    Блок 7.  Клиппирование
    Блок 8.  Сравнение дикторов (violin plot)
    Блок 9.  Корреляционная матрица метрик
    Блок 10. Анализ транскриптов ([нрзб], длина)
    Блок 11. Перекрытия реплик (временны́е метки)
    Блок 12. Итоговый дашборд (все метрики на одном листе)
    Блок 13. Сохранение отчёта

Запуск:
    python corpus_analysis.py --manifest corpus/manifest.csv
    python corpus_analysis.py --manifest corpus/manifest.csv --domain dialect
    python corpus_analysis.py --manifest corpus/manifest.csv --limit 500

Зависимости:
    pip install librosa soundfile tqdm seaborn pandas
"""

import argparse
import csv
import os
import warnings
from collections import defaultdict
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi":       120,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.facecolor": "white",
})

DOMAIN_PALETTE = {
    "dialect":     "#E05C4B",
    "interviewer": "#4B8BE0",
    "standard":    "#27AE60",
    "unknown":     "#888888",
}


# =============================================================================
# БЛОК 1. ЗАГРУЗКА MANIFEST И РАЗРЕШЕНИЕ ПУТЕЙ
# =============================================================================

def resolve_path(file_path_str: str, manifest_dir: Path) -> Path | None:
    """
    Надёжно разрешает путь к файлу из manifest.

    Поддерживает три варианта записи file_path:
      1. Абсолютный путь   /home/user/corpus/dialect/EIV1939/audios/file.wav
      2. Относительный     dialect/EIV1939/audios/file.wav
      3. Просто имя файла  file.wav  (ищем рекурсивно)
    """
    if not file_path_str or file_path_str in ("", "None"):
        return None

    p = Path(file_path_str)

    # Вариант 1: абсолютный путь
    if p.is_absolute():
        return p if p.exists() else None

    # Вариант 2: относительный от директории manifest
    candidate = manifest_dir / p
    if candidate.exists():
        return candidate

    # Вариант 3: относительный от родителя директории manifest
    candidate2 = manifest_dir.parent / p
    if candidate2.exists():
        return candidate2

    return None


def load_manifest(manifest_path: str,
                  domain_filter: str | None = None,
                  limit: int | None = None) -> tuple[list[dict], Path]:
    """
    Загружает manifest, возвращает (rows, manifest_dir).
    Сразу диагностирует проблемы с путями.
    """
    manifest_path = Path(manifest_path)
    manifest_dir  = manifest_path.parent

    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Загружено записей из manifest: {len(rows)}")

    if domain_filter:
        rows = [r for r in rows if r.get("domain") == domain_filter]
        print(f"После фильтра домена '{domain_filter}': {len(rows)}")

    if limit:
        rows = rows[:limit]
        print(f"Ограничение --limit: {len(rows)}")

    # ── Диагностика путей (проверяем первые 5 файлов) ────────────────────
    print("\nДиагностика путей (первые 5 записей):")
    found_count = 0
    for row in rows[:5]:
        fp  = row.get("file_path", "")
        res = resolve_path(fp, manifest_dir)
        status = "✓" if res and res.exists() else "✗"
        print(f"  {status}  {fp[:80]}")
        if res and res.exists():
            found_count += 1

    if found_count == 0 and len(rows) > 0:
        print("\n  ⚠ Ни один из первых 5 файлов не найден!")
        print("  Проверьте: запущен ли скрипт из правильной директории,")
        print("  и правильно ли указан --manifest.")

    return rows, manifest_dir


# =============================================================================
# МЕТРИКИ
# =============================================================================

def compute_snr(y: np.ndarray) -> float:
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    noise_floor = np.percentile(rms, 20)
    valid = rms[rms > noise_floor * 2 + 1e-10]
    if len(valid) == 0 or noise_floor < 1e-10:
        return 60.0
    return float(np.clip(
        20 * np.log10(np.mean(valid) / (noise_floor + 1e-10)), -5, 70
    ))


def compute_vad_ratio(y: np.ndarray) -> float:
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    return float(np.mean(rms > np.percentile(rms, 30)))


def compute_rms_db(y: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-10))


def compute_clip_ratio(y: np.ndarray) -> float:
    return float(np.mean(np.abs(y) > 0.99))


def compute_spectral_centroid(y: np.ndarray, sr: int) -> float:
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    return float(np.mean(sc))


def analyze_file(filepath: Path) -> dict | None:
    try:
        info = sf.info(str(filepath))
        y, sr = librosa.load(str(filepath), sr=16000, mono=True)
        return {
            "duration_sec":       round(info.duration, 4),
            "sample_rate":        info.samplerate,
            "snr_db":             round(compute_snr(y), 2),
            "rms_db":             round(compute_rms_db(y), 2),
            "vad_ratio":          round(compute_vad_ratio(y), 4),
            "clip_ratio":         round(compute_clip_ratio(y), 6),
            "spectral_centroid":  round(compute_spectral_centroid(y, sr), 1),
        }
    except Exception:
        return None


def collect_metrics(rows: list[dict], manifest_dir: Path) -> list[dict]:
    """
    Для каждой строки manifest:
      - пробует взять длительность из manifest (быстро, не читая файл)
      - если файл доступен — считает полные акустические метрики
    """
    results = []
    files_found    = 0
    files_missing  = 0
    metrics_from_manifest = 0

    for row in tqdm(rows, desc="Анализ файлов"):
        entry = dict(row)

        # ── Длительность из manifest ──────────────────────────────────────
        # Берём из duration_ms или duration_sec если уже есть
        dur_from_manifest = None
        for field in ("duration_sec", "duration_ms"):
            v = row.get(field)
            if v not in (None, "", "None"):
                try:
                    val = float(v)
                    dur_from_manifest = val / 1000 if field == "duration_ms" else val
                    break
                except (ValueError, TypeError):
                    pass

        if dur_from_manifest is not None:
            entry["duration_sec"] = round(dur_from_manifest, 4)

        # ── Акустические метрики — читаем файл ───────────────────────────
        resolved = resolve_path(row.get("file_path", ""), manifest_dir)

        if resolved and resolved.exists():
            files_found += 1
            metrics = analyze_file(resolved)
            if metrics:
                # Обновляем duration_sec реальным значением
                entry.update(metrics)
        else:
            files_missing += 1
            # Если длительность есть в manifest — используем её,
            # остальные метрики остаются None
            if dur_from_manifest is not None:
                metrics_from_manifest += 1

        # ── Транскрипт ────────────────────────────────────────────────────
        transcript = row.get("transcript") or ""
        entry["has_nrzb"]       = "[нрзб]" in transcript
        entry["transcript_len"] = len(transcript.split())

        results.append(entry)

    print(f"\n  Файлов найдено:          {files_found}")
    print(f"  Файлов не найдено:       {files_missing}")
    if metrics_from_manifest:
        print(f"  Длит. из manifest:       {metrics_from_manifest} "
              f"(акустические метрики недоступны)")

    if files_missing > 0 and files_found == 0:
        print("\n  ⚠ Все файлы не найдены. Возможные причины:")
        print("    - manifest содержит абсолютные пути,")
        print("      которые не существуют на этой машине")
        print("    - запустите скрипт из той же директории,")
        print("      где запускали rebuild_manifest.py")
        print("    - или используйте --manifest с абсолютным путём")

    return results


def extract_numeric(data: list[dict], key: str) -> np.ndarray:
    vals = []
    for d in data:
        v = d.get(key)
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            pass
    return np.array(vals)


# =============================================================================
# БЛОК 2. ОБЩАЯ СТАТИСТИКА
# =============================================================================

def print_summary(data: list[dict]) -> None:
    from collections import Counter

    print("\n" + "═" * 58)
    print("  СВОДКА ПО КОРПУСУ")
    print("═" * 58)
    print(f"  Всего записей: {len(data)}")

    dur = extract_numeric(data, "duration_sec")
    if len(dur):
        print(f"\n  ДЛИТЕЛЬНОСТЬ:")
        print(f"    Сумма:    {dur.sum()/60:.1f} мин  ({dur.sum()/3600:.2f} ч)")
        print(f"    Среднее:  {dur.mean():.2f} сек")
        print(f"    Медиана:  {np.median(dur):.2f} сек")
        print(f"    Мин/Макс: {dur.min():.2f} / {dur.max():.2f} сек")

    print(f"\n  ДОМЕНЫ:")
    domains = Counter(d.get("domain", "?") for d in data)
    for dom, cnt in sorted(domains.items()):
        spk_cnt = len({d["speaker_id"] for d in data
                       if d.get("domain") == dom})
        d_dur   = extract_numeric(
            [x for x in data if x.get("domain") == dom], "duration_sec"
        )
        print(f"    {dom:<15} {cnt:>5} записей | "
              f"{spk_cnt} дикторов | {d_dur.sum()/60:.1f} мин")

    snr = extract_numeric(data, "snr_db")
    if len(snr):
        print(f"\n  SNR (дБ): mean={snr.mean():.1f}  "
              f"median={np.median(snr):.1f}  "
              f"p5={np.percentile(snr,5):.1f}  "
              f"p95={np.percentile(snr,95):.1f}")

    nrzb = sum(1 for d in data if d.get("has_nrzb"))
    print(f"\n  [нрзб]: {nrzb} из {len(data)} ({100*nrzb/len(data):.1f}%)")
    print("═" * 58 + "\n")


# =============================================================================
# БЛОКИ 3–11. ГРАФИКИ
# =============================================================================

def plot_duration_distribution(data, save_path=None):
    dur_all = extract_numeric(data, "duration_sec")
    if not len(dur_all):
        print("  Длительность: нет данных")
        return

    domain_data = defaultdict(list)
    for d in data:
        v = d.get("duration_sec")
        if v is not None:
            try:
                domain_data[d.get("domain", "unknown")].append(float(v))
            except (ValueError, TypeError):
                pass

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Распределение длительностей записей", fontsize=14, y=1.02)

    ax = axes[0]
    ax.hist(dur_all, bins=60, color="#4B8BE0", edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(dur_all), color="#E05C4B", lw=1.8, linestyle="--",
               label=f"Медиана {np.median(dur_all):.2f}с")
    ax.axvline(dur_all.mean(), color="#F5A623", lw=1.8, linestyle=":",
               label=f"Среднее {dur_all.mean():.2f}с")
    ax.set_xlabel("Длительность (сек)")
    ax.set_ylabel("Кол-во записей")
    ax.set_title("Весь корпус")
    ax.legend(fontsize=8)

    ax = axes[1]
    for dom, vals in domain_data.items():
        ax.hist(vals, bins=40, alpha=0.65,
                color=DOMAIN_PALETTE.get(dom, "#888"),
                edgecolor="white", linewidth=0.3, label=dom)
    ax.set_xlabel("Длительность (сек)")
    ax.set_ylabel("Кол-во записей")
    ax.set_title("По доменам")
    ax.legend(fontsize=8)

    ax = axes[2]
    sorted_dur = np.sort(dur_all)
    cdf = np.arange(1, len(sorted_dur) + 1) / len(sorted_dur)
    ax.plot(sorted_dur, cdf, color="#4B8BE0", lw=2)
    for threshold, label, color in [
        (0.5, "0.5с", "#E05C4B"),
        (1.0, "1.0с", "#F5A623"),
        (2.0, "2.0с", "#27AE60"),
    ]:
        frac = float(np.mean(sorted_dur >= threshold))
        ax.axvline(threshold, color=color, lw=1.4, linestyle="--",
                   label=f"≥{label}: {100*frac:.0f}%")
    ax.set_xlabel("Длительность (сек)")
    ax.set_ylabel("Доля записей (CDF)")
    ax.set_title("Накопленное распределение")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_snr_distribution(data, save_path=None):
    snr_all = extract_numeric(data, "snr_db")
    if not len(snr_all):
        print("  SNR: нет данных (файлы не найдены)")
        return

    domain_snr = defaultdict(list)
    for d in data:
        v = d.get("snr_db")
        if v is not None:
            try:
                domain_snr[d.get("domain", "unknown")].append(float(v))
            except (ValueError, TypeError):
                pass

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("SNR — отношение сигнал/шум (дБ)", fontsize=14, y=1.02)

    ax = axes[0]
    ax.hist(snr_all, bins=50, color="#27AE60", edgecolor="white", linewidth=0.4)
    ax.axvspan(-10, 10, alpha=0.12, color="#E05C4B", label="Плохо (<10 дБ)")
    ax.axvspan(10,  20, alpha=0.12, color="#F5A623", label="Денойз (10–20 дБ)")
    ax.axvspan(20,  80, alpha=0.08, color="#27AE60", label="Хорошо (>20 дБ)")
    ax.axvline(np.median(snr_all), color="#333", lw=1.5, linestyle="--",
               label=f"Медиана {np.median(snr_all):.1f} дБ")
    ax.set_xlabel("SNR (дБ)")
    ax.set_ylabel("Кол-во записей")
    ax.set_title("Распределение SNR")
    ax.legend(fontsize=7)

    ax = axes[1]
    for dom, vals in domain_snr.items():
        if len(vals) > 5:
            sns.kdeplot(vals, ax=ax, label=dom,
                        color=DOMAIN_PALETTE.get(dom, "#888"),
                        fill=True, alpha=0.25, linewidth=2)
    ax.set_xlabel("SNR (дБ)")
    ax.set_ylabel("Плотность")
    ax.set_title("KDE по доменам")
    ax.legend(fontsize=8)

    ax = axes[2]
    bins_snr   = [-10, 5, 10, 15, 20, 30, 70]
    labels_snr = ["<5", "5–10", "10–15", "15–20", "20–30", ">30"]
    counts = np.histogram(snr_all, bins=bins_snr)[0]
    colors = ["#C0392B", "#E05C4B", "#F5A623", "#F0C040", "#7BC67E", "#27AE60"]
    bars = ax.bar(labels_snr, counts, color=colors, edgecolor="white")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                str(cnt), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Диапазон SNR (дБ)")
    ax.set_ylabel("Кол-во записей")
    ax.set_title("Файлы по SNR-диапазонам")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_rms_distribution(data, save_path=None):
    rms_all = extract_numeric(data, "rms_db")
    if not len(rms_all):
        print("  RMS: нет данных")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("RMS — уровень громкости (дБ)", fontsize=14, y=1.02)

    ax = axes[0]
    ax.hist(rms_all, bins=50, color="#9B59B6", edgecolor="white", linewidth=0.4)
    ax.axvline(-45, color="#E05C4B", lw=1.5, linestyle="--",
               label="Порог отсева (−45 дБ)")
    ax.axvline(-20, color="#27AE60", lw=1.5, linestyle="--",
               label="Цель нормализации (−20 дБ)")
    ax.axvline(np.median(rms_all), color="#333", lw=1.5, linestyle=":",
               label=f"Медиана {np.median(rms_all):.1f} дБ")
    ax.set_xlabel("RMS (дБ)")
    ax.set_ylabel("Кол-во записей")
    ax.set_title("Распределение RMS")
    ax.legend(fontsize=8)

    ax = axes[1]
    snr_all = extract_numeric(data, "snr_db")
    if len(snr_all) > 0:
        n = min(len(snr_all), len(rms_all))
        c_list = [DOMAIN_PALETTE.get(d.get("domain", "unknown"), "#888")
                  for d in data
                  if d.get("rms_db") is not None and d.get("snr_db") is not None]
        ax.scatter(rms_all[:n], snr_all[:n], alpha=0.35, s=8,
                   c=c_list[:n])
        ax.axvline(-45, color="#E05C4B", lw=1.2, linestyle="--", alpha=0.7)
        ax.axhline(10,  color="#E05C4B", lw=1.2, linestyle="--", alpha=0.7,
                   label="Пороги фильтрации")
        ax.set_xlabel("RMS (дБ)")
        ax.set_ylabel("SNR (дБ)")
        ax.set_title("Зависимость SNR от RMS")
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=c, markersize=8, label=dom)
                   for dom, c in DOMAIN_PALETTE.items()
                   if any(d.get("domain") == dom for d in data)]
        ax.legend(handles=handles, fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_vad_distribution(data, save_path=None):
    vad_all = extract_numeric(data, "vad_ratio")
    if not len(vad_all):
        print("  VAD: нет данных")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("VAD-ratio — доля речевых фреймов", fontsize=14, y=1.02)

    ax = axes[0]
    ax.hist(vad_all, bins=40, color="#E67E22", edgecolor="white", linewidth=0.4)
    ax.axvline(0.25, color="#E05C4B", lw=1.5, linestyle="--",
               label="Порог отсева (0.25)")
    ax.axvline(np.median(vad_all), color="#333", lw=1.5, linestyle=":",
               label=f"Медиана {np.median(vad_all):.2f}")
    ax.set_xlabel("VAD-ratio")
    ax.set_ylabel("Кол-во записей")
    ax.set_title("Распределение VAD-ratio")
    ax.legend(fontsize=8)

    ax = axes[1]
    bins_vad   = [0, 0.25, 0.4, 0.6, 0.8, 1.01]
    labels_vad = ["<0.25\n(мало речи)", "0.25–0.4",
                  "0.4–0.6", "0.6–0.8", "0.8–1.0\n(плотная речь)"]
    counts = np.histogram(vad_all, bins=bins_vad)[0]
    colors = ["#C0392B", "#E05C4B", "#F5A623", "#7BC67E", "#27AE60"]
    bars   = ax.bar(labels_vad, counts, color=colors, edgecolor="white")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                str(cnt), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Кол-во записей")
    ax.set_title("Записи по диапазонам VAD-ratio")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_clipping(data, save_path=None):
    clip = extract_numeric(data, "clip_ratio")
    if not len(clip):
        print("  Клиппирование: нет данных")
        return

    n_clipped = int(np.sum(clip > 0.005))
    n_total   = len(clip)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Клиппирование сигнала", fontsize=14, y=1.02)

    ax = axes[0]
    nonzero = clip[clip > 0]
    if len(nonzero):
        ax.hist(nonzero, bins=50, color="#C0392B",
                edgecolor="white", linewidth=0.4)
        ax.axvline(0.005, color="#F5A623", lw=1.5, linestyle="--",
                   label="Порог отсева (0.005)")
        ax.set_xscale("log")
        ax.set_xlabel("Доля клиппированных сэмплов (log)")
        ax.set_ylabel("Кол-во записей")
        ax.set_title("Распределение clip_ratio (ненулевые)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Клиппирование не обнаружено",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Клиппирование")

    ax = axes[1]
    ax.pie(
        [n_clipped, n_total - n_clipped],
        labels=[f"Клиппировано\n{n_clipped} ({100*n_clipped/n_total:.1f}%)",
                f"Норма\n{n_total-n_clipped} ({100*(n_total-n_clipped)/n_total:.1f}%)"],
        colors=["#E05C4B", "#7BC67E"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax.set_title("Доля клиппированных записей")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_speaker_comparison(data, save_path=None):
    speaker_metrics = defaultdict(lambda: defaultdict(list))
    for d in data:
        spk = d.get("speaker_id", "?")
        for key in ("duration_sec", "snr_db", "rms_db", "vad_ratio"):
            v = d.get(key)
            if v is not None:
                try:
                    speaker_metrics[spk][key].append(float(v))
                except (ValueError, TypeError):
                    pass

    speakers = sorted(speaker_metrics.keys())
    if not speakers:
        print("  Дикторы: нет данных")
        return

    metrics = ["duration_sec", "snr_db", "rms_db", "vad_ratio"]
    titles  = ["Длительность (сек)", "SNR (дБ)", "RMS (дБ)", "VAD-ratio"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle("Сравнение дикторов по акустическим метрикам",
                 fontsize=14, y=1.01)

    for ax, metric, title in zip(axes.flat, metrics, titles):
        plot_data  = []
        plot_labels = []
        valid_spks  = []
        for spk in speakers:
            vals = speaker_metrics[spk].get(metric, [])
            if len(vals) >= 3:
                plot_data.append(vals)
                plot_labels.append(spk[:14])
                valid_spks.append(spk)

        if not plot_data:
            ax.set_title(title + " (нет данных)")
            continue

        parts = ax.violinplot(plot_data, showmedians=True, showextrema=True)
        for i, (spk, body) in enumerate(zip(valid_spks, parts["bodies"])):
            dom = next((d.get("domain", "unknown") for d in data
                        if d.get("speaker_id") == spk), "unknown")
            body.set_facecolor(DOMAIN_PALETTE.get(dom, "#888"))
            body.set_alpha(0.7)

        ax.set_xticks(range(1, len(plot_labels) + 1))
        ax.set_xticklabels(plot_labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(title)
        ax.set_title(title)

        all_vals = [v for sublist in plot_data for v in sublist]
        ax.axhline(np.median(all_vals), color="#333", lw=1,
                   linestyle=":", alpha=0.6, label="Медиана корпуса")
        ax.legend(fontsize=7)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=c, label=dom)
        for dom, c in DOMAIN_PALETTE.items()
        if any(d.get("domain") == dom for d in data)
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               fontsize=9, title="Домен")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_correlation_matrix(data, save_path=None):
    import pandas as pd

    keys   = ["duration_sec", "snr_db", "rms_db", "vad_ratio",
              "clip_ratio", "spectral_centroid"]
    labels = ["Длит. (с)", "SNR (дБ)", "RMS (дБ)",
              "VAD-ratio", "Clip-ratio", "Spect.centroid"]

    matrix_data = []
    for d in data:
        row = []
        for k in keys:
            v = d.get(k)
            try:
                row.append(float(v))
            except (TypeError, ValueError):
                row.append(np.nan)
        matrix_data.append(row)

    df   = pd.DataFrame(matrix_data, columns=labels).dropna()
    if df.empty:
        print("  Корреляция: недостаточно данных")
        return

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr, ax=ax,
        annot=True, fmt=".2f", linewidths=0.5,
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Корреляция между акустическими метриками", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_transcript_analysis(data, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Анализ транскриптов", fontsize=14, y=1.02)

    ax = axes[0]
    speaker_nrzb = defaultdict(lambda: [0, 0])
    for d in data:
        spk = d.get("speaker_id", "?")
        speaker_nrzb[spk][1] += 1
        if d.get("has_nrzb"):
            speaker_nrzb[spk][0] += 1

    spks   = sorted(speaker_nrzb.keys())
    ratios = [speaker_nrzb[s][0] / max(speaker_nrzb[s][1], 1) for s in spks]
    colors = [DOMAIN_PALETTE.get(
        next((d.get("domain") for d in data
              if d.get("speaker_id") == s), "unknown"), "#888")
              for s in spks]
    bars = ax.barh([s[:16] for s in spks], ratios,
                   color=colors, edgecolor="white")
    ax.set_xlabel("Доля реплик с [нрзб]")
    ax.set_title("Частота [нрзб] по дикторам")
    for bar, ratio in zip(bars, ratios):
        if ratio > 0:
            ax.text(ratio + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{100*ratio:.1f}%", va="center", fontsize=8)

    ax = axes[1]
    tlen = extract_numeric(data, "transcript_len")
    if len(tlen):
        ax.hist(tlen, bins=40, color="#1ABC9C",
                edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(tlen), color="#333", lw=1.5, linestyle="--",
                   label=f"Медиана {np.median(tlen):.0f} слов")
        ax.set_xlabel("Слов в транскрипте")
        ax.set_ylabel("Кол-во записей")
        ax.set_title("Длина транскрипта")
        ax.legend(fontsize=8)

    ax = axes[2]
    dur = extract_numeric(data, "duration_sec")
    if len(dur) > 0 and len(tlen) > 0:
        n = min(len(dur), len(tlen))
        ax.scatter(dur[:n], tlen[:n], alpha=0.3, s=8, color="#9B59B6")
        try:
            z = np.polyfit(dur[:n], tlen[:n], 1)
            p = np.poly1d(z)
            xline = np.linspace(dur.min(), dur.max(), 100)
            ax.plot(xline, p(xline), "r--", lw=1.5, alpha=0.7, label="Тренд")
        except Exception:
            pass
        ax.set_xlabel("Длительность (сек)")
        ax.set_ylabel("Слов в транскрипте")
        ax.set_title("Длит. vs длина транскрипта")
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_overlap_analysis(data, save_path=None):
    sessions = defaultdict(list)
    for d in data:
        sess  = d.get("session") or d.get("page_name", "?")
        start = d.get("start_ms")
        end   = d.get("end_ms")
        if start not in (None, "", "None") and end not in (None, "", "None"):
            try:
                sessions[sess].append({
                    "start":  int(float(start)),
                    "end":    int(float(end)),
                    "spk":    d.get("speaker_id", "?"),
                    "domain": d.get("domain", "?"),
                })
            except (ValueError, TypeError):
                pass

    if not sessions:
        print("  Перекрытия: временны́е метки не найдены в manifest")
        return

    overlap_ratios = {}
    for sess, utterances in sessions.items():
        u_sorted = sorted(utterances, key=lambda x: x["start"])
        overlaps = sum(
            1 for i in range(1, len(u_sorted))
            if u_sorted[i]["start"] < u_sorted[i-1]["end"]
        )
        overlap_ratios[sess] = overlaps / max(len(utterances) - 1, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Анализ перекрытий реплик", fontsize=13, y=1.02)

    ax = axes[0]
    sess_names = [s[:22] for s in overlap_ratios.keys()]
    ratios     = list(overlap_ratios.values())
    colors_bar = ["#E05C4B" if r > 0.15 else
                  "#F5A623" if r > 0.05 else
                  "#7BC67E" for r in ratios]
    ax.barh(sess_names, ratios, color=colors_bar, edgecolor="white")
    ax.axvline(0.1, color="#E05C4B", lw=1.3, linestyle="--",
               label="Порог внимания (10%)")
    ax.set_xlabel("Доля перекрывающихся реплик")
    ax.set_title("Перекрытия по беседам")
    ax.legend(fontsize=8)

    ax = axes[1]
    worst_sess = max(overlap_ratios, key=overlap_ratios.get)
    utterances = sorted(sessions[worst_sess], key=lambda x: x["start"])[:40]
    speaker_ids = list(dict.fromkeys(u["spk"] for u in utterances))
    spk_y       = {spk: i for i, spk in enumerate(speaker_ids)}

    for utt in utterances:
        y     = spk_y[utt["spk"]]
        color = DOMAIN_PALETTE.get(utt["domain"], "#888")
        ax.barh(y,
                (utt["end"] - utt["start"]) / 1000,
                left=utt["start"] / 1000,
                height=0.5, color=color, alpha=0.75, edgecolor="white")

    ax.set_yticks(range(len(speaker_ids)))
    ax.set_yticklabels([s[:14] for s in speaker_ids], fontsize=9)
    ax.set_xlabel("Время (сек)")
    ax.set_title(f"Timeline: {worst_sess[:25]}\n"
                 f"(перекрытий: {100*overlap_ratios[worst_sess]:.0f}%)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# =============================================================================
# БЛОК 12. ДАШБОРД
# =============================================================================

def plot_dashboard(data, save_path=None):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Анализ корпуса диалектной речи — сводный дашборд",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

    dur  = extract_numeric(data, "duration_sec")
    snr  = extract_numeric(data, "snr_db")
    rms  = extract_numeric(data, "rms_db")
    vad  = extract_numeric(data, "vad_ratio")

    for col, (vals, title, color, unit) in enumerate([
        (dur, "Длительность",  "#4B8BE0", "сек"),
        (snr, "SNR",           "#27AE60", "дБ"),
        (rms, "RMS",           "#9B59B6", "дБ"),
        (vad, "VAD-ratio",     "#E67E22", ""),
    ]):
        ax = fig.add_subplot(gs[0, col])
        if len(vals):
            ax.hist(vals, bins=35, color=color, edgecolor="white", linewidth=0.3)
            ax.axvline(np.median(vals), color="#333", lw=1.4, linestyle="--",
                       label=f"Md={np.median(vals):.2f}")
            ax.legend(fontsize=7)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(unit, fontsize=9)
        ax.set_ylabel("N", fontsize=9)

    ax2a = fig.add_subplot(gs[1, 0:2])
    if len(snr) and len(rms):
        n = min(len(snr), len(rms))
        c_list = [DOMAIN_PALETTE.get(d.get("domain", "?"), "#888")
                  for d in data if d.get("snr_db") and d.get("rms_db")]
        ax2a.scatter(rms[:len(c_list)], snr[:len(c_list)],
                     alpha=0.3, s=6, c=c_list)
        ax2a.axvline(-45, color="#E05C4B", lw=1.2, linestyle="--", alpha=0.7)
        ax2a.axhline(10,  color="#E05C4B", lw=1.2, linestyle="--", alpha=0.7,
                     label="Пороги фильтрации")
        ax2a.set_xlabel("RMS (дБ)")
        ax2a.set_ylabel("SNR (дБ)")
        ax2a.set_title("SNR vs RMS")
        ax2a.legend(fontsize=7)

    ax2b = fig.add_subplot(gs[1, 2:4])
    if len(dur) and len(vad):
        c_list2 = [DOMAIN_PALETTE.get(d.get("domain", "?"), "#888")
                   for d in data
                   if d.get("duration_sec") and d.get("vad_ratio")]
        ax2b.scatter(dur[:len(c_list2)], vad[:len(c_list2)],
                     alpha=0.3, s=6, c=c_list2)
        ax2b.axhline(0.25, color="#E05C4B", lw=1.2, linestyle="--",
                     alpha=0.7, label="Порог VAD")
        ax2b.set_xlabel("Длительность (сек)")
        ax2b.set_ylabel("VAD-ratio")
        ax2b.set_title("Длительность vs VAD-ratio")
        ax2b.legend(fontsize=7)

    ax3a = fig.add_subplot(gs[2, 0:2])
    spk_snr = defaultdict(list)
    for d in data:
        v = d.get("snr_db")
        if v is not None:
            try:
                spk_snr[d.get("speaker_id", "?")].append(float(v))
            except (ValueError, TypeError):
                pass
    if spk_snr:
        spks_sorted = sorted(spk_snr.keys())
        ax3a.boxplot(
            [spk_snr[s] for s in spks_sorted],
            labels=[s[:10] for s in spks_sorted],
            patch_artist=True,
            boxprops={"facecolor": "#4B8BE055"},
            medianprops={"color": "#E05C4B", "linewidth": 2},
            whiskerprops={"linewidth": 1},
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        )
        ax3a.axhline(10, color="#E05C4B", lw=1.2, linestyle="--",
                     label="SNR=10 дБ")
        ax3a.set_ylabel("SNR (дБ)")
        ax3a.set_title("SNR по дикторам")
        ax3a.tick_params(axis="x", rotation=35, labelsize=7)
        ax3a.legend(fontsize=7)

    ax3b = fig.add_subplot(gs[2, 2])
    spk_nrzb = defaultdict(lambda: [0, 0])
    for d in data:
        s = d.get("speaker_id", "?")
        spk_nrzb[s][1] += 1
        if d.get("has_nrzb"):
            spk_nrzb[s][0] += 1
    spks2   = sorted(spk_nrzb.keys())
    nrzb_r  = [spk_nrzb[s][0]/max(spk_nrzb[s][1], 1) for s in spks2]
    colors2 = [DOMAIN_PALETTE.get(
        next((d.get("domain") for d in data
              if d.get("speaker_id") == s), "?"), "#888")
               for s in spks2]
    ax3b.bar([s[:10] for s in spks2], nrzb_r, color=colors2, edgecolor="white")
    ax3b.set_ylabel("Доля [нрзб]")
    ax3b.set_title("[нрзб] по дикторам")
    ax3b.tick_params(axis="x", rotation=35, labelsize=7)

    ax3c = fig.add_subplot(gs[2, 3])
    if len(dur):
        sd = np.sort(dur)
        ax3c.plot(sd, np.arange(1, len(sd)+1)/len(sd), color="#4B8BE0", lw=2)
        for thr, col in [(0.5, "#C0392B"), (1.0, "#E67E22"), (2.0, "#27AE60")]:
            frac = float(np.mean(sd >= thr))
            ax3c.axvline(thr, color=col, lw=1.2, linestyle="--",
                         label=f"≥{thr}с: {100*frac:.0f}%")
        ax3c.set_xlabel("Длит. (сек)")
        ax3c.set_ylabel("CDF")
        ax3c.set_title("CDF длительностей")
        ax3c.legend(fontsize=7)

    from matplotlib.patches import Patch
    legend_h = [Patch(facecolor=c, label=dom)
                for dom, c in DOMAIN_PALETTE.items()
                if any(d.get("domain") == dom for d in data)]
    fig.legend(handles=legend_h, loc="upper right", fontsize=9,
               title="Домен", bbox_to_anchor=(0.99, 0.97))

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Дашборд сохранён: {save_path}")
    plt.show()


# =============================================================================
# БЛОК 13. ТЕКСТОВЫЙ ОТЧЁТ
# =============================================================================

def save_text_report(data, output_path):
    from collections import Counter
    lines = ["=" * 58,
             "  ОТЧЁТ ПО КОРПУСУ ДИАЛЕКТНОЙ РЕЧИ",
             "=" * 58,
             f"Всего записей: {len(data)}"]

    dur = extract_numeric(data, "duration_sec")
    if len(dur):
        lines += [
            "\nДЛИТЕЛЬНОСТЬ (сек):",
            f"  Сумма:    {dur.sum()/60:.1f} мин",
            f"  Среднее:  {dur.mean():.2f}",
            f"  Медиана:  {np.median(dur):.2f}",
            f"  Мин/Макс: {dur.min():.2f} / {dur.max():.2f}",
            f"  < 0.5с:   {int(np.sum(dur < 0.5))} ({100*np.mean(dur < 0.5):.1f}%)",
            f"  0.5–2с:   {int(np.sum((dur >= 0.5) & (dur < 2)))} "
            f"({100*np.mean((dur >= 0.5) & (dur < 2)):.1f}%)",
            f"  ≥ 2с:     {int(np.sum(dur >= 2))} ({100*np.mean(dur >= 2):.1f}%)",
        ]

    snr = extract_numeric(data, "snr_db")
    if len(snr):
        lines += [
            "\nSNR (дБ):",
            f"  Среднее:  {snr.mean():.1f}",
            f"  Медиана:  {np.median(snr):.1f}",
            f"  P5/P95:   {np.percentile(snr,5):.1f} / {np.percentile(snr,95):.1f}",
            f"  < 5 дБ:   {int(np.sum(snr < 5))} — будут отброшены",
            f"  5–10 дБ:  {int(np.sum((snr >= 5) & (snr < 10)))} — под вопросом",
            f"  10–20 дБ: {int(np.sum((snr >= 10) & (snr < 20)))} — нужен денойз",
            f"  > 20 дБ:  {int(np.sum(snr >= 20))} — хорошее качество",
        ]
    else:
        lines.append(
            "\nSNR: нет данных "
            "(файлы не найдены — используется только информация из manifest)"
        )

    nrzb = sum(1 for d in data if d.get("has_nrzb"))
    lines.append(f"\n[нрзб]: {nrzb} из {len(data)} ({100*nrzb/len(data):.1f}%)")

    lines.append("\nДИКТОРЫ:")
    spk_counts = Counter(d.get("speaker_id", "?") for d in data)
    for spk, cnt in sorted(spk_counts.items(), key=lambda x: -x[1]):
        spk_dur = extract_numeric(
            [x for x in data if x.get("speaker_id") == spk], "duration_sec"
        )
        lines.append(f"  {spk:<28} {cnt:>5} реплик | {spk_dur.sum()/60:.1f} мин")

    report = "\n".join(lines)
    print(report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nОтчёт сохранён: {output_path}")


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def run_analysis(manifest_path, output_dir="analysis_output",
                 domain_filter=None, limit=None):

    manifest_path = Path(manifest_path)
    output_dir    = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, manifest_dir = load_manifest(
        str(manifest_path), domain_filter, limit
    )

    print("\nСбор акустических метрик...")
    data = collect_metrics(rows, manifest_dir)

    print_summary(data)
    print("Построение графиков...\n")

    plot_duration_distribution(data, str(output_dir / "01_duration.png"))
    plot_snr_distribution(      data, str(output_dir / "02_snr.png"))
    plot_rms_distribution(      data, str(output_dir / "03_rms.png"))
    plot_vad_distribution(      data, str(output_dir / "04_vad.png"))
    plot_clipping(              data, str(output_dir / "05_clipping.png"))
    plot_speaker_comparison(    data, str(output_dir / "06_speakers.png"))
    plot_correlation_matrix(    data, str(output_dir / "07_correlation.png"))
    plot_transcript_analysis(   data, str(output_dir / "08_transcripts.png"))
    plot_overlap_analysis(      data, str(output_dir / "09_overlaps.png"))
    plot_dashboard(             data, str(output_dir / "00_dashboard.png"))
    save_text_report(           data, str(output_dir / "corpus_report.txt"))

    print(f"\n✅ Все графики сохранены в: {output_dir}/")
    return data


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Анализ и визуализация диалектного корпуса.",
    )
    parser.add_argument("--manifest", "-m", required=True,
                        help="Путь к manifest.csv")
    parser.add_argument("--out",      "-o", default="analysis_output",
                        help="Папка для сохранения графиков")
    parser.add_argument("--domain",   default=None,
                        choices=["dialect", "interviewer", "standard"])
    parser.add_argument("--limit",    type=int, default=None,
                        help="Ограничить кол-во файлов (для отладки)")
    args = parser.parse_args()

    run_analysis(args.manifest, args.out, args.domain, args.limit)


if __name__ == "__main__":
    main()
