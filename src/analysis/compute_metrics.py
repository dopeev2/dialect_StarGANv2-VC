"""
compute_metrics.py  (v2 — ускоренная версия)
============================================
Ключевые оптимизации относительно v1:
  1. MFCC пула предвычисляется ОДИН РАЗ на сценарий (не для каждого файла)
  2. DTW-поиск работает по уже готовым матрицам в памяти (нет повторных загрузок)
  3. pyin заменён на yin (в 5–10x быстрее, достаточно для сравнения доменов)
  4. n_candidates снижен до 25 (вместо 80)
  5. Результаты кэшируются: если metrics_per_file.csv уже есть — пропускаем

Ожидаемое ускорение: ~8–15x → вместо 2–3 часов ~15–25 минут.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import scipy.signal
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr

from experiment_config import (
    SR, N_FFT, HOP, WIN, N_MELS,
    SCENARIOS, SCENARIO_MAP, METRICS, BASELINES,
    ACTUAL_TEST_COUNTS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Загрузка аудио
# ─────────────────────────────────────────────────────────────

def load_audio(path: Path, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# MFCC
# ─────────────────────────────────────────────────────────────

def extract_mfcc(path: Path, n_mfcc: int = 13) -> np.ndarray:
    """Возвращает MFCC (T, n_mfcc)."""
    y = load_audio(path)
    mfcc = librosa.feature.mfcc(
        y=y, sr=SR, n_mfcc=n_mfcc,
        n_fft=N_FFT, hop_length=HOP, win_length=WIN, n_mels=N_MELS,
    )
    return mfcc.T  # (T, n_mfcc)


def _mcd_from_matrices(
    mc_c: np.ndarray,
    mc_r: np.ndarray,
    use_dtw: bool = True,
) -> float:
    """MCD из двух уже загруженных матриц MFCC."""
    if use_dtw:
        try:
            from fastdtw import fastdtw
            _, path_dtw = fastdtw(mc_c, mc_r, dist=euclidean, radius=10)
            mc_c = np.array([mc_c[i] for i, _ in path_dtw])
            mc_r = np.array([mc_r[j] for _, j in path_dtw])
        except ImportError:
            t = min(len(mc_c), len(mc_r))
            mc_c, mc_r = mc_c[:t], mc_r[:t]
    else:
        t = min(len(mc_c), len(mc_r))
        mc_c, mc_r = mc_c[:t], mc_r[:t]

    diff = mc_c[:, 1:] - mc_r[:, 1:]
    return float(np.mean((10.0 / np.log(10)) * np.sqrt(2 * np.sum(diff**2, axis=1))))


def compute_mcd(conv_path: Path, ref_path: Path, n_mfcc: int = 13) -> float:
    """MCD между двумя файлами (загружает оба с нуля — используется только для G0)."""
    return _mcd_from_matrices(
        extract_mfcc(conv_path, n_mfcc),
        extract_mfcc(ref_path,  n_mfcc),
    )


# ─────────────────────────────────────────────────────────────
# ОПТИМИЗАЦИЯ 1: Индекс MFCC пула — предвычисляется один раз
# ─────────────────────────────────────────────────────────────

def build_mfcc_index(
    pool: List[Path],
    n_mfcc: int = 13,
    cache: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """
    Предвычисляет MFCC для всего пула целевого домена.
    Возвращает {str(path): mfcc_matrix (T, n_mfcc)}.
    Кэшируется на диск — при повторном запуске загружается мгновенно.
    """
    if cache and cache.exists():
        logger.info("Загружаю MFCC-индекс из кэша: %s", cache)
        data = np.load(str(cache), allow_pickle=True).item()
        return data

    logger.info("Предвычисляю MFCC для %d файлов пула...", len(pool))
    index: Dict[str, np.ndarray] = {}
    t0 = time.time()
    for i, p in enumerate(pool):
        try:
            index[str(p)] = extract_mfcc(p, n_mfcc)
        except Exception as e:
            logger.debug("Ошибка MFCC для %s: %s", p, e)
        if (i + 1) % 20 == 0:
            logger.info("  MFCC индекс: %d/%d (%.1f с)", i+1, len(pool), time.time()-t0)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache), index)
        logger.info("MFCC-индекс сохранён: %s (%d файлов)", cache, len(index))

    return index


def find_dtw_nearest_indexed(
    query_mfcc: np.ndarray,
    mfcc_index: Dict[str, np.ndarray],
    n_candidates: int = 25,
    seed: int = 42,
) -> Tuple[Path, float]:
    """
    ОПТИМИЗАЦИЯ 2: Поиск ближайшего референса по уже готовому индексу.
    Не читает файлы с диска — только матричные операции в памяти.
    n_candidates=25 вместо 80 — достаточно для стабильного результата.
    """
    rng = np.random.default_rng(seed)
    keys = list(mfcc_index.keys())

    if len(keys) > n_candidates:
        chosen = rng.choice(len(keys), size=n_candidates, replace=False)
        keys = [keys[i] for i in chosen]

    best_path = Path(keys[0])
    best_mcd  = float("inf")

    for k in keys:
        try:
            mcd = _mcd_from_matrices(query_mfcc, mfcc_index[k], use_dtw=True)
            if mcd < best_mcd:
                best_mcd = mcd
                best_path = Path(k)
        except Exception:
            pass

    return best_path, best_mcd


# ─────────────────────────────────────────────────────────────
# ОПТИМИЗАЦИЯ 3: yin вместо pyin (в 5–10x быстрее)
# ─────────────────────────────────────────────────────────────

def extract_f0(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    F0 через librosa.yin — быстрее pyin в 5–10x.
    Для сравнения доменных контуров точности достаточно.
    Returns: f0 (Гц, 0=unvoiced), voiced_mask (bool).
    """
    y = load_audio(path)
    f0 = librosa.yin(
        y,
        fmin=METRICS.f0_fmin,
        fmax=METRICS.f0_fmax,
        sr=SR,
        hop_length=HOP,
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    voiced = (f0 > METRICS.f0_fmin).astype(bool)
    return f0, voiced


def build_domain_f0_median(
    file_list: List[Path],
    target_len: int = 300,
    cache: Optional[Path] = None,
) -> np.ndarray:
    """Медианный F0-контур домена (нормализован к target_len). Кэшируется."""
    if cache and cache.exists():
        return np.load(str(cache))

    matrix = []
    for p in file_list:
        try:
            f0, voiced = extract_f0(p)
            if voiced.sum() < 5:
                continue
            norm = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(f0)),
                f0,
            )
            matrix.append(norm)
        except Exception:
            pass

    if not matrix:
        return np.zeros(target_len, np.float32)

    result = np.median(np.array(matrix), axis=0).astype(np.float32)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(cache), result)
    return result


def _align_f0(f0_a: np.ndarray, f0_b: np.ndarray):
    t = min(len(f0_a), len(f0_b))
    interp = lambda s: np.interp(
        np.linspace(0, 1, t), np.linspace(0, 1, len(s)), s,
    )
    return interp(f0_a), interp(f0_b)


def compute_f0_rmse(conv_path: Path, domain_f0_median: np.ndarray) -> float:
    f0_c, voiced_c = extract_f0(conv_path)
    f0_c, f0_ref   = _align_f0(f0_c, domain_f0_median)
    mask = voiced_c[:len(f0_c)] & (f0_ref[:len(f0_c)] > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((f0_c[mask] - f0_ref[mask])**2)))


def compute_f0_pcc(conv_path: Path, domain_f0_median: np.ndarray) -> float:
    f0_c, voiced_c = extract_f0(conv_path)
    f0_c, f0_ref   = _align_f0(f0_c, domain_f0_median)
    mask = voiced_c[:len(f0_c)] & (f0_ref[:len(f0_c)] > 0)
    if mask.sum() < 2:
        return float("nan")
    a, b = f0_c[mask], f0_ref[mask]
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return float("nan")
    pcc, _ = pearsonr(a, b)
    return float(pcc)


# ─────────────────────────────────────────────────────────────
# WER (Whisper)
# ─────────────────────────────────────────────────────────────

_WHISPER_CACHE: Dict = {}

def transcribe(path: Path) -> str:
    import whisper
    if "model" not in _WHISPER_CACHE:
        logger.info("Загрузка Whisper %s...", METRICS.whisper_model)
        _WHISPER_CACHE["model"] = whisper.load_model(METRICS.whisper_model)
    model = _WHISPER_CACHE["model"]
    y = load_audio(path, sr=16_000)
    return model.transcribe(y, language=METRICS.whisper_lang)["text"].strip()


def _edit_distance(r: List[str], h: List[str]) -> int:
    d = [[0] * (len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i][j] = d[i-1][j-1] if r[i-1]==h[j-1] \
                       else 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[len(r)][len(h)]


def compute_wer(ref_text: str, hyp_path: Path) -> float:
    import re
    clean = lambda s: re.sub(r"[^\w\s]", "", s.lower()).split()
    ref_w = clean(ref_text)
    hyp_w = clean(transcribe(hyp_path))
    return _edit_distance(ref_w, hyp_w) / len(ref_w) if ref_w else 0.0


# ─────────────────────────────────────────────────────────────
# ΔMFCC — Z-расстояние формант
# ─────────────────────────────────────────────────────────────

def estimate_formants(path: Path) -> np.ndarray:
    """Медианные F1, F2 через LPC. Returns (2,) в Гц."""
    y = load_audio(path)
    y = librosa.effects.preemphasis(y, coef=0.97)
    frames = librosa.util.frame(y, frame_length=WIN, hop_length=HOP)
    tracks = []
    for frame in frames.T:
        w = frame * scipy.signal.windows.hamming(len(frame))
        try:
            a = librosa.lpc(w, order=METRICS.lpc_order)
            roots = np.roots(a)
            roots = roots[np.imag(roots) >= 0]
            freqs  = np.sort(np.angle(roots) * (SR / (2 * np.pi)))
            freqs  = freqs[(freqs >= 200) & (freqs <= 4000)]
            if len(freqs) >= 2:
                tracks.append(freqs[:2])
        except Exception:
            pass
    if not tracks:
        return np.full(2, np.nan)
    return np.nanmedian(np.array(tracks), axis=0)


def build_standard_formant_stats(
    std_files: List[Path],
    cache: Optional[Path] = None,
) -> Dict:
    """Статистики формант нормативного домена. Кэшируются."""
    if cache and cache.exists():
        with open(cache) as f:
            return json.load(f)

    medians = []
    for p in std_files:
        try:
            m = estimate_formants(p)
            if not np.any(np.isnan(m)):
                medians.append(m)
        except Exception:
            pass

    arr = np.array(medians)
    stats = {
        "F1": {"mean": float(arr[:,0].mean()), "std": float(arr[:,0].std())},
        "F2": {"mean": float(arr[:,1].mean()), "std": float(arr[:,1].std())},
    }
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as f:
            json.dump(stats, f, indent=2)
    return stats


def compute_delta_mfcc(path: Path, std_stats: Dict) -> Dict[str, float]:
    med = estimate_formants(path)
    result = {}
    for i, label in enumerate(["F1", "F2"]):
        mu, sigma = std_stats[label]["mean"], std_stats[label]["std"]
        result[f"delta_{label}"] = float(abs(med[i]-mu)/sigma) \
                                   if sigma > 1e-6 and not np.isnan(med[i]) \
                                   else float("nan")
    vals = [v for v in result.values() if not np.isnan(v)]
    result["delta_mean"] = float(np.mean(vals)) if vals else float("nan")
    return result


# ─────────────────────────────────────────────────────────────
# Агрегация метрик одного файла
# ─────────────────────────────────────────────────────────────

def compute_file_metrics(
    conv_path:        Path,
    orig_path:        Path,
    conv_mfcc:        np.ndarray,   # предвычисленный MFCC конвертации
    dtw_ref_path:     Path,
    dtw_ref_mfcc:     np.ndarray,   # предвычисленный MFCC референса
    domain_f0_median: np.ndarray,
    std_stats:        Dict,
    reference_text:   str,
    compute_wer_flag: bool = False,
) -> Dict:
    m = {}

    # MCD конвертация (матрицы уже в памяти — быстро)
    try:
        m["mcd_conv"] = _mcd_from_matrices(conv_mfcc, dtw_ref_mfcc)
    except Exception:
        m["mcd_conv"] = float("nan")

    # MCD passthrough (оригинал → тот же референс)
    try:
        orig_mfcc = extract_mfcc(orig_path)
        m["mcd_pass"] = _mcd_from_matrices(orig_mfcc, dtw_ref_mfcc)
    except Exception:
        m["mcd_pass"] = float("nan")

    # F0 RMSE и PCC (yin — быстро)
    try:
        m["f0_rmse"] = compute_f0_rmse(conv_path, domain_f0_median)
    except Exception:
        m["f0_rmse"] = float("nan")

    try:
        m["f0_pcc"] = compute_f0_pcc(conv_path, domain_f0_median)
    except Exception:
        m["f0_pcc"] = float("nan")

    # WER (только если явно включён)
    if compute_wer_flag and reference_text:
        try:
            wer_orig = compute_wer(reference_text, orig_path)
            wer_conv = compute_wer(reference_text, conv_path)
            m["wer_orig"]     = wer_orig
            m["wer_conv"]     = wer_conv
            m["wer_delta_pp"] = (wer_conv - wer_orig) * 100
        except Exception:
            m["wer_orig"] = m["wer_conv"] = m["wer_delta_pp"] = float("nan")
    else:
        m["wer_orig"] = m["wer_conv"] = m["wer_delta_pp"] = float("nan")

    # ΔMFCC
    try:
        m.update(compute_delta_mfcc(conv_path, std_stats))
    except Exception:
        m["delta_F1"] = m["delta_F2"] = m["delta_mean"] = float("nan")

    return m


# ─────────────────────────────────────────────────────────────
# Полный пайплайн оценки одного сценария
# ─────────────────────────────────────────────────────────────

def evaluate_scenario(
    scenario_id:      str,
    test_df:          pd.DataFrame,
    audio_dir:        Path,
    conv_dir:         Path,
    results_dir:      Path,
    std_stats:        Dict,
    f0_medians:       Dict[str, np.ndarray],
    compute_wer_flag: bool = False,
    n_candidates:     int = 25,
    resume:           bool = True,
) -> pd.DataFrame:
    sc = SCENARIO_MAP[scenario_id]
    scenario_dir = results_dir / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # ОПТИМИЗАЦИЯ 4: resume — пропускаем уже посчитанные
    out_csv = scenario_dir / "metrics_per_file.csv"
    if resume and out_csv.exists():
        existing = pd.read_csv(out_csv)
        logger.info("Сценарий %s: загружен кэш (%d файлов)", scenario_id, len(existing))
        return existing

    src_df = test_df[test_df["domain_id"] == sc.src]
    tgt_df = test_df[test_df["domain_id"] == sc.tgt]

    tgt_pool = [
        audio_dir / r["rel_path"]
        for _, r in tgt_df.iterrows()
        if (audio_dir / r["rel_path"]).exists()
    ]

    # ОПТИМИЗАЦИЯ 1: предвычисляем MFCC пула один раз
    mfcc_cache = scenario_dir / "tgt_mfcc_index.npy"
    mfcc_index = build_mfcc_index(tgt_pool, cache=mfcc_cache)

    domain_f0 = f0_medians.get(sc.tgt, np.zeros(300))
    conv_scenario_dir = conv_dir / scenario_id

    rows = []
    t0 = time.time()
    logger.info("Оцениваю %s (%s): %d файлов...", scenario_id, sc.label, len(src_df))

    for i, (_, row) in enumerate(src_df.iterrows()):
        orig_path = audio_dir / row["rel_path"]
        conv_path = conv_scenario_dir / (Path(row["rel_path"]).stem + ".wav")

        if not conv_path.exists():
            continue

        # MFCC конвертации (нужен для DTW-поиска и MCD)
        try:
            conv_mfcc = extract_mfcc(conv_path)
        except Exception:
            continue

        # DTW-поиск по индексу (ОПТИМИЗАЦИЯ 2)
        try:
            dtw_ref, _ = find_dtw_nearest_indexed(
                conv_mfcc, mfcc_index, n_candidates=n_candidates,
            )
            dtw_ref_mfcc = mfcc_index[str(dtw_ref)]
        except Exception:
            dtw_ref = tgt_pool[0] if tgt_pool else conv_path
            dtw_ref_mfcc = extract_mfcc(dtw_ref)

        m = compute_file_metrics(
            conv_path=conv_path,
            orig_path=orig_path,
            conv_mfcc=conv_mfcc,
            dtw_ref_path=dtw_ref,
            dtw_ref_mfcc=dtw_ref_mfcc,
            domain_f0_median=domain_f0,
            std_stats=std_stats,
            reference_text=str(row.get("transcript", "")),
            compute_wer_flag=compute_wer_flag,
        )
        m["utterance_id"] = row["utterance_id"]
        m["speaker_id"]   = row["speaker_id"]
        m["scenario_id"]  = scenario_id
        rows.append(m)

        if (i + 1) % 20 == 0:
            rate = (i+1) / max(time.time()-t0, 1e-6)
            eta  = (len(src_df)-i-1) / rate
            logger.info("  %d/%d | %.1f файл/с | ETA %.0f с",
                        i+1, len(src_df), rate, eta)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    logger.info("Сценарий %s: сохранено %d строк → %s", scenario_id, len(df), out_csv)
    return df


# ─────────────────────────────────────────────────────────────
# Сводный отчёт
# ─────────────────────────────────────────────────────────────

def build_report(all_dfs: Dict[str, pd.DataFrame], results_dir: Path):
    COLS = ["mcd_conv", "mcd_pass", "f0_rmse", "f0_pcc",
            "wer_delta_pp", "delta_mean"]
    COL_LABELS = {
        "mcd_conv":    "MCD конв. (dB) ↓",
        "mcd_pass":    "MCD pass. (dB)",
        "f0_rmse":     "F0 RMSE (Гц) ↓",
        "f0_pcc":      "F0 PCC ↑",
        "wer_delta_pp":"ΔWER (п.п.) ↓",
        "delta_mean":  "ΔFormant ↓",
    }

    summary_rows = []
    for sid, df in all_dfs.items():
        sc  = SCENARIO_MAP.get(sid)
        row = {
            "Сценарий": f"{sid} ({sc.label if sc else ''})",
            "N": len(df),
            "Основной": "✓" if (sc and sc.is_primary) else "",
        }
        for col in COLS:
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals):
                    row[COL_LABELS[col]] = f"{vals.mean():.3f} ± {vals.std():.3f}"
                else:
                    row[COL_LABELS[col]] = "—"
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    csv_path = results_dir / "summary_table.csv"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Сводная таблица: %s", csv_path)

    print("\n" + "="*65)
    print("СВОДНЫЕ РЕЗУЛЬТАТЫ (Таблица X ВКР)")
    print("="*65)
    print(summary.to_string(index=False))
    print()

    for sid, df in all_dfs.items():
        sc = SCENARIO_MAP.get(sid)
        if sc is None or not sc.is_primary:
            continue
        print(f"--- {sid} {sc.label} ---")

        if "mcd_conv" in df.columns and "mcd_pass" in df.columns:
            mcd_c = df["mcd_conv"].dropna().mean()
            mcd_p = df["mcd_pass"].dropna().mean()
            impr  = mcd_p - mcd_c
            print(f"  MCD:    pass={mcd_p:.3f} → conv={mcd_c:.3f} "
                  f"| улучшение {impr:+.3f} dB ({'✓' if impr > 0 else '✗'})")

        if "f0_pcc" in df.columns:
            pcc = df["f0_pcc"].dropna().mean()
            thr = METRICS.f0_pcc_good
            print(f"  F0 PCC: {pcc:.3f} "
                  f"({'≥' if pcc>=thr else '<'}{thr} — "
                  f"{'хорошо ✓' if pcc>=thr else 'ниже порога ✗'})")

        if "wer_delta_pp" in df.columns:
            wd = df["wer_delta_pp"].dropna()
            if len(wd):
                print(f"  ΔWER:   {wd.mean():+.1f} п.п. "
                      f"({'≤5 ✓' if wd.mean()<=5 else '>5 ✗'})")
    print("="*65)

    latex = _to_latex(summary)
    (results_dir / "summary_latex.tex").write_text(latex, encoding="utf-8")
    logger.info("LaTeX: %s", results_dir / "summary_latex.tex")


def _to_latex(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    col_spec = "l" + "c" * (len(cols)-1)
    header = " & ".join(f"\\textbf{{{c}}}" for c in cols) + " \\\\"
    lines = [
        "\\begin{table}[h]", "\\centering",
        "\\caption{Объективные метрики оценки конверсии}",
        "\\label{tab:results}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule", header, "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",      type=Path, required=True)
    parser.add_argument("--audio-dir",     type=Path, required=True)
    parser.add_argument("--converted-dir", type=Path, required=True)
    parser.add_argument("--results-dir",   type=Path,
                        default=Path("/content/drive/MyDrive/diploma/results"))
    parser.add_argument("--scenarios",     nargs="+", default=["S1","S2","S3","S4"])
    parser.add_argument("--with-wer",      action="store_true")
    parser.add_argument("--no-resume",     action="store_true",
                        help="Пересчитать даже если CSV уже есть")
    parser.add_argument("--n-candidates",  type=int, default=25,
                        help="Кол-во DTW-кандидатов (меньше = быстрее)")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    df      = pd.read_csv(args.manifest)
    test_df = df[df["split"] == "test"].copy()

    std_files = [
        args.audio_dir / r["rel_path"]
        for _, r in test_df[test_df["domain_id"]=="standard"].iterrows()
        if (args.audio_dir / r["rel_path"]).exists()
    ][:100]

    std_stats = build_standard_formant_stats(
        std_files, cache=args.results_dir / "std_formant_stats.json",
    )

    f0_medians = {}
    for group in ["don_dialect", "pyoza_dialect", "standard"]:
        pool = [
            args.audio_dir / r["rel_path"]
            for _, r in test_df[test_df["domain_id"]==group].iterrows()
            if (args.audio_dir / r["rel_path"]).exists()
        ][:150]
        f0_medians[group] = build_domain_f0_median(
            pool, cache=args.results_dir / f"f0_median_{group}.npy",
        )

    metric_dfs = {}
    for sid in args.scenarios:
        if sid not in SCENARIO_MAP:
            continue
        df_m = evaluate_scenario(
            scenario_id=sid,
            test_df=test_df,
            audio_dir=args.audio_dir,
            conv_dir=args.converted_dir,
            results_dir=args.results_dir,
            std_stats=std_stats,
            f0_medians=f0_medians,
            compute_wer_flag=args.with_wer,
            n_candidates=args.n_candidates,
            resume=not args.no_resume,
        )
        metric_dfs[sid] = df_m
        logger.info("Сценарий %s: %d файлов", sid, len(df_m))

    build_report(metric_dfs, args.results_dir)


if __name__ == "__main__":
    main()
