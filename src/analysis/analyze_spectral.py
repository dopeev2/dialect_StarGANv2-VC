# analyze_spectral_honest.py
# Честный пайплайн анализа спектральных признаков диалектного корпуса.
#
# Контролируемые факторы:
#   1. Пол спикера        — только женщины (карта SPEAKER_GENDER + верификация через F0)
#   2. Фонемный контекст  — только файлы с предударным /о/ и частотными словами
#   3. Длительность       — 2–6 сек
#   4. Качество звука     — SNR >= 15 дБ
#   5. Баланс спикеров    — не более MAX_PER_SPEAKER файлов на информанта
#
# Запуск: локально, без GPU.
# pip install librosa praat-parselmouth pandas numpy scipy matplotlib seaborn tqdm

import os
import re
import warnings
import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  ПАРАМЕТРЫ — измените под свои пути
# ══════════════════════════════════════════════════════════════

MANIFEST_PATH = "D:/DIPLOMA/data_scraping/corpus_22k/manifest_clean_partial.csv"
CORPUS_ROOT   = "D:/DIPLOMA/data_scraping/corpus_22k"
OUTPUT_DIR    = "D:/DIPLOMA/data_scraping/analysis_honest"

GROUP_COL = "dialect_group"   # don / pyoza / standard
SR        = 24_000
N_MFCC    = 13

# ── Фильтры ──────────────────────────────────────────────────
DUR_MIN         = 2.0    # сек
DUR_MAX         = 6.0    # сек
SNR_MIN         = 15.0   # дБ
MAX_PER_SPEAKER = 60     # защита от доминирования одного информанта

# ── Пол спикеров ─────────────────────────────────────────────
# Определено по инициалам (ЛингКонЛаб) и LibriSpeech speakers.tsv.
# Верификация через медианный F0 выполняется автоматически на шаге 3.
SPEAKER_GENDER = {
    "don_EIV1939":        "F",
    "don_GLT1934":        "F",
    "don_KVA1948":        "F",
    "don_MLI1941":        "F",
    "don_SVS1939":        "F",
    "don_TNG1957":        "F",
    "don_VIK1941":        "F",
    "pyoza_AME1920":      "F",
    "pyoza_GGS1932":      "F",
    "pyoza_GLA1926":      "F",
    "pyoza_GVA1914":      "F",
    "pyoza_KE1919":       "F",
    "pyoza_MAN1910":      "F",
    "pyoza_MAT1915":      "F",
    "pyoza_MGG1932":      "F",
    "don_interviewer":    "F",
    "pyoza_interviewer":  "F",
    "librispeech_295":    "F",   # LibriSpeech speakers.tsv
    "librispeech_8086":   "F",
    "librispeech_8169":   "F",
    "librispeech_9014":   "M",   # мужчина — исключаем
}

# ── Целевые слова ─────────────────────────────────────────────
# Отбираем файлы, содержащие слова с /о/ в предударной позиции
# (контекст для проверки аканья/оканья) + высокочастотные слова
# с предударными гласными, общие для всех трёх доменов.
TARGET_WORDS = [
    # Предударный /о/ — ключевой контекст аканья/оканья
    "потом", "тогда", "потому", "поэтому",
    "хорошо", "хорошая", "хороший",
    "молодой", "молодая", "молодые",
    "голова", "голове", "голову",
    "дорога", "дороге", "дорогу",
    "корова", "коровы",
    "колхоз", "колхозе",
    "одного", "одной", "одному",
    "своего", "своей", "своему",
    "того",
    # Дополнительно: частотные слова с предударными гласными
    # (расширяем покрытие для домена standard / LibriSpeech)
    "когда", "было", "человек",
    "большой", "большая",
    "сказала", "говорит",
]

# Паттерн без групп захвата — избегаем предупреждений pandas
_PATTERN = "|".join(re.escape(w) for w in TARGET_WORDS)


# ══════════════════════════════════════════════════════════════
#  1. ФИЛЬТРАЦИЯ МАНИФЕСТА
# ══════════════════════════════════════════════════════════════

def filter_manifest(manifest_path: str) -> pd.DataFrame:
    df    = pd.read_csv(manifest_path)
    n_all = len(df)
    log   = []

    # 0. Только train-сплит
    df = df[df["split"] == "train"].copy()
    log.append(("split == train", len(df)))

    # 1. Только женщины
    df["gender"] = df["speaker_id"].map(SPEAKER_GENDER).fillna("?")
    df = df[df["gender"] == "F"].copy()
    log.append(("gender == F", len(df)))

    # 2. Диапазон длительности
    df = df[df["duration_sec"].between(DUR_MIN, DUR_MAX)].copy()
    log.append((f"duration [{DUR_MIN}–{DUR_MAX}] сек", len(df)))

    # 3. Минимальный SNR
    df = df[df["snr_db"] >= SNR_MIN].copy()
    log.append((f"snr_db >= {SNR_MIN} дБ", len(df)))

    # 4. Наличие целевых слов (фонемный контекст)
    df = df[df["transcript"].str.lower()
              .str.contains(_PATTERN, regex=True)].copy()
    log.append(("содержит целевое слово", len(df)))

    # 5. Стратификация по спикеру
    parts = []
    for _, spk_df in df.groupby("speaker_id"):
        n = min(len(spk_df), MAX_PER_SPEAKER)
        parts.append(spk_df.sample(n=n, random_state=42))
    df = pd.concat(parts, ignore_index=True)
    log.append((f"max {MAX_PER_SPEAKER} файлов/спикер", len(df)))

    # Лог
    print(f"\n{'='*55}\nФИЛЬТРАЦИЯ МАНИФЕСТА\n{'='*55}")
    print(f"  Исходно:  {n_all}")
    for step, n in log:
        print(f"  → {step:<36} {n:5d}")

    print(f"\n  По доменам после фильтрации:")
    print(df[GROUP_COL].value_counts().to_string())
    print(f"\n  Уникальных спикеров по доменам:")
    print(df.groupby(GROUP_COL)["speaker_id"].nunique().to_string())

    return df


# ══════════════════════════════════════════════════════════════
#  2. ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
# ══════════════════════════════════════════════════════════════

def extract_spectral(wav_path: str) -> dict | None:
    try:
        y, _ = librosa.load(wav_path, sr=SR, mono=True)
    except Exception:
        return None
    if len(y) < SR * 0.3:
        return None
    mfcc  = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=256)
    feats = {f"mfcc_{i+1}": float(mfcc[i].mean()) for i in range(N_MFCC)}
    feats["spectral_centroid"]  = float(librosa.feature.spectral_centroid(y=y, sr=SR).mean())
    feats["spectral_bandwidth"] = float(librosa.feature.spectral_bandwidth(y=y, sr=SR).mean())
    feats["spectral_rolloff"]   = float(librosa.feature.spectral_rolloff(y=y, sr=SR).mean())
    feats["zcr"]                = float(librosa.feature.zero_crossing_rate(y).mean())
    feats["rms_energy"]         = float(librosa.feature.rms(y=y).mean())
    return feats

_FORMANT_EMPTY = {k: np.nan for k in
                  ["F1_median","F2_median","F3_median","F1_std","F2_std"]}

def extract_formants(wav_path: str) -> dict:
    try:
        snd     = parselmouth.Sound(wav_path)
        formant = call(snd, "To Formant (burg)", 0.0, 3, 5500, 0.025, 50)
        dur     = call(snd, "Get total duration")
        times   = np.arange(0.025, dur - 0.025, 0.01)
        f1, f2, f3 = [], [], []
        for t in times:
            v1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            v2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            v3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
            if not np.isnan(v1): f1.append(v1)
            if not np.isnan(v2): f2.append(v2)
            if not np.isnan(v3): f3.append(v3)
        return {
            "F1_median": np.median(f1) if f1 else np.nan,
            "F2_median": np.median(f2) if f2 else np.nan,
            "F3_median": np.median(f3) if f3 else np.nan,
            "F1_std":    np.std(f1)    if f1 else np.nan,
            "F2_std":    np.std(f2)    if f2 else np.nan,
        }
    except Exception:
        return _FORMANT_EMPTY.copy()

_PROSODY_EMPTY = {k: np.nan for k in
                  ["F0_mean","F0_std","F0_range","F0_p10","F0_p90","voiced_ratio"]}

def extract_prosody(wav_path: str) -> dict:
    try:
        snd   = parselmouth.Sound(wav_path)
        # 100–500 Гц — диапазон женского голоса
        pitch = call(snd, "To Pitch", 0.0, 100, 500)
        dur   = call(snd, "Get total duration")
        times = np.arange(0.0, dur, 0.01)
        f0 = []
        for t in times:
            v = call(pitch, "Get value at time", t, "Hertz", "Linear")
            if not np.isnan(v) and v > 0:
                f0.append(v)
        if len(f0) < 5:
            return _PROSODY_EMPTY.copy()
        return {
            "F0_mean":      float(np.mean(f0)),
            "F0_std":       float(np.std(f0)),
            "F0_range":     float(np.max(f0) - np.min(f0)),
            "F0_p10":       float(np.percentile(f0, 10)),
            "F0_p90":       float(np.percentile(f0, 90)),
            "voiced_ratio": float(len(f0) / max(len(times), 1)),
        }
    except Exception:
        return _PROSODY_EMPTY.copy()


def build_feature_table(df: pd.DataFrame, corpus_root: str) -> pd.DataFrame:
    records, errors = [], 0
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Признаки", unit="утт"):
        wav_path = os.path.join(corpus_root, row["rel_path"])
        if not os.path.exists(wav_path):
            errors += 1
            continue
        spec = extract_spectral(wav_path)
        if spec is None:
            errors += 1
            continue
        record = {
            "utterance_id": row["utterance_id"],
            "speaker_id":   row["speaker_id"],
            GROUP_COL:      row[GROUP_COL],
            "domain_id":    row["domain_id"],
            "duration_sec": row["duration_sec"],
            "snr_db":       row["snr_db"],
            "birth_year":   row.get("birth_year", np.nan),
        }
        record.update(spec)
        record.update(extract_formants(wav_path))
        record.update(extract_prosody(wav_path))
        records.append(record)
    print(f"Извлечено: {len(records)}, пропущено: {errors}")
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════
#  3. ВЕРИФИКАЦИЯ ПОЛА ЧЕРЕЗ F0
#  Женский голос: медианный F0 спикера > 150 Гц.
#  Если спикер ниже порога — исключаем и предупреждаем.
# ══════════════════════════════════════════════════════════════

def verify_gender_by_f0(feat_df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*55}\nВЕРИФИКАЦИЯ ПОЛА ЧЕРЕЗ F0\n{'='*55}")
    spk_f0   = feat_df.groupby("speaker_id")["F0_mean"].median().sort_values()
    suspects = []
    print(f"  {'Спикер':<30} {'F0 медиана':>11}  {'Статус':>8}")
    print(f"  {'-'*52}")
    for spk, f0 in spk_f0.items():
        if   f0 > 160: flag = "✓  женский"
        elif f0 < 145: flag = "⚠  МУЖСКОЙ?"; suspects.append(spk)
        else:          flag = "?  граница"
        print(f"  {spk:<30} {f0:>9.1f} Гц  {flag}")
    if suspects:
        print(f"\n  Исключаем подозрительных: {suspects}")
        feat_df = feat_df[~feat_df["speaker_id"].isin(suspects)].copy()
    else:
        print("\n  Все спикеры подтверждены.")
    return feat_df


# ══════════════════════════════════════════════════════════════
#  4. СТАТИСТИКА
# ══════════════════════════════════════════════════════════════

ANALYSIS_FEATURES = [
    "F1_median", "F2_median", "F3_median",
    "F0_mean", "F0_std", "F0_range", "F0_p10", "F0_p90", "voiced_ratio",
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4",
    "spectral_centroid", "spectral_rolloff", "spectral_bandwidth", "zcr",
]

def domain_statistics(feat_df: pd.DataFrame) -> pd.DataFrame:
    groups  = sorted(feat_df[GROUP_COL].unique())
    results = []
    for feat in ANALYSIS_FEATURES:
        if feat not in feat_df.columns:
            continue
        row        = {"feature": feat}
        group_data = []
        for g in groups:
            vals = feat_df.loc[feat_df[GROUP_COL] == g, feat].dropna().values
            row[f"{g}_mean"] = round(float(np.mean(vals)), 3) if len(vals) else np.nan
            row[f"{g}_std"]  = round(float(np.std(vals)),  3) if len(vals) else np.nan
            row[f"{g}_n"]    = len(vals)
            group_data.append(vals)

        # Краскел–Уоллис (общий тест)
        valid = [g for g in group_data if len(g) > 5]
        if len(valid) >= 2:
            H, p = stats.kruskal(*valid)
            row["kruskal_H"] = round(H, 3)
            row["p_value"]   = round(p, 6)
            row["sig"]       = ("***" if p < 0.001 else "**" if p < 0.01
                                 else "*" if p < 0.05 else "ns")
            # Попарные тесты Манна–Уитни с поправкой Бонферрони
            n_pairs = len(groups) * (len(groups) - 1) // 2
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    a = feat_df.loc[feat_df[GROUP_COL] == groups[i], feat].dropna().values
                    b = feat_df.loc[feat_df[GROUP_COL] == groups[j], feat].dropna().values
                    if len(a) > 5 and len(b) > 5:
                        _, p_pair = stats.mannwhitneyu(a, b, alternative="two-sided")
                        p_adj = min(p_pair * n_pairs, 1.0)
                        row[f"{groups[i]}_vs_{groups[j]}"] = round(p_adj, 5)
        else:
            row.update({"kruskal_H": np.nan, "p_value": np.nan, "sig": "—"})
        results.append(row)
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════
#  5. ВИЗУАЛИЗАЦИИ
# ══════════════════════════════════════════════════════════════

PALETTE      = {"don": "#e74c3c", "pyoza": "#3498db", "standard": "#2ecc71"}
DOMAIN_ORDER = ["don", "pyoza", "standard"]

def _clr(d): return PALETTE.get(d, "#95a5a6")

def _order(feat_df):
    return [d for d in DOMAIN_ORDER if d in feat_df[GROUP_COL].unique()]

def plot_confounders_check(df_filtered: pd.DataFrame, out_dir: str):
    """Диагностика: длительность и SNR должны быть выровнены после фильтрации."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    order = _order(df_filtered)
    sns.boxplot(data=df_filtered, x=GROUP_COL, y="duration_sec",
                palette=PALETTE, order=order, ax=axes[0])
    sns.boxplot(data=df_filtered, x=GROUP_COL, y="snr_db",
                palette=PALETTE, order=order, ax=axes[1])
    axes[0].set_title("Длительность после фильтра (сек)")
    axes[1].set_title("SNR после фильтра (дБ)")
    for ax in axes: ax.set_xlabel("")
    plt.suptitle("Контроль конфаундеров: длительность и качество аудио", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out_dir, "00_confounders_check.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  {p}")

def plot_vowel_space(feat_df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    for domain, grp in feat_df.groupby(GROUP_COL):
        clean = grp[["F2_median","F1_median"]].dropna()
        if len(clean) < 3: continue
        ax.scatter(clean["F2_median"], clean["F1_median"],
                   alpha=0.3, s=16, color=_clr(domain), label=domain)
        mx, my = clean["F2_median"].mean(), clean["F1_median"].mean()
        ax.scatter(mx, my, s=200, marker="X", color=_clr(domain),
                   edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(domain, (mx, my), textcoords="offset points",
                    xytext=(6, 4), fontsize=10, fontweight="bold")
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("F2 (Гц)", fontsize=12)
    ax.set_ylabel("F1 (Гц)", fontsize=12)
    ax.set_title("Пространство гласных F1×F2\n"
                 "(только Ж, 2–6 сек, предударный /о/)", fontsize=12)
    ax.legend(fontsize=11)
    plt.tight_layout()
    p = os.path.join(out_dir, "01_vowel_space.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  {p}")

def plot_f0_by_speaker(feat_df: pd.DataFrame, out_dir: str):
    """F0 по каждому спикеру — показывает возрастную вариацию внутри домена."""
    domains = _order(feat_df)
    fig, axes = plt.subplots(1, len(domains), figsize=(5 * len(domains), 5), sharey=True)
    if len(domains) == 1: axes = [axes]
    for ax, domain in zip(axes, domains):
        sub = feat_df[feat_df[GROUP_COL] == domain]
        if sub.empty: continue
        sns.boxplot(data=sub, x="speaker_id", y="F0_mean",
                    color=_clr(domain), ax=ax)
        ax.set_title(domain, fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=50)
    axes[0].set_ylabel("F0_mean (Гц)", fontsize=11)
    plt.suptitle("F0 по спикерам — контроль возрастного конфаундера\n"
                 "(внутридоменная вариация должна быть меньше междоменной)", fontsize=11)
    plt.tight_layout()
    p = os.path.join(out_dir, "02_f0_by_speaker.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  {p}")

def plot_f0_kde(feat_df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for domain, grp in feat_df.groupby(GROUP_COL):
        c = _clr(domain)
        grp["F0_mean"].dropna().plot.kde(ax=axes[0], label=domain, color=c, linewidth=2)
        grp["F0_range"].dropna().plot.kde(ax=axes[1], label=domain, color=c, linewidth=2)
    titles = ["Средний F0 (Гц)", "Диапазон F0 (Гц)"]
    for ax, t in zip(axes, titles):
        ax.set_title(t); ax.set_xlabel("Гц"); ax.legend()
    plt.suptitle("Просодика по доменам (только женщины, контролируемые условия)", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out_dir, "03_f0_kde.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  {p}")

def plot_mfcc_boxplots(feat_df: pd.DataFrame, out_dir: str):
    order = _order(feat_df)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, [f"mfcc_{i}" for i in range(1, 7)]):
        sns.boxplot(data=feat_df, x=GROUP_COL, y=col,
                    palette=PALETTE, order=order, ax=ax)
        ax.set_title(col); ax.set_xlabel("")
    plt.suptitle("MFCC 1–6 (только женщины, предударный /о/)", fontsize=13)
    plt.tight_layout()
    p = os.path.join(out_dir, "04_mfcc_boxplots.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  {p}")

def plot_pairwise_significance(stats_df: pd.DataFrame, out_dir: str):
    """Тепловая карта p-значений попарных сравнений (−log10 шкала)."""
    pair_cols = [c for c in stats_df.columns if "_vs_" in c]
    if not pair_cols: return
    hmap  = stats_df.set_index("feature")[pair_cols].copy().astype(float)
    logp  = -np.log10(hmap.clip(lower=1e-10))
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    sns.heatmap(logp, cmap="YlOrRd", ax=axes[0],
                linewidths=0.4, cbar_kws={"label": "−log₁₀(p adj.)"})
    axes[0].set_title("−log₁₀(p) попарных сравнений\n(больше = значимее)")
    sig_mark = hmap.applymap(
        lambda v: "***" if v < 0.001 else "**" if v < 0.01
                  else "*" if v < 0.05 else "ns" if not np.isnan(v) else ""
    )
    sns.heatmap(hmap.round(4), cmap="YlOrRd", ax=axes[1],
                annot=sig_mark, fmt="", linewidths=0.4,
                cbar_kws={"label": "p adj. (Бонферрони)"})
    axes[1].set_title("p-значения с метками значимости")
    plt.suptitle("Попарные сравнения доменов (Манн–Уитни + поправка Бонферрони)",
                 fontsize=12)
    plt.tight_layout()
    p = os.path.join(out_dir, "05_pairwise_significance.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  {p}")

def plot_feature_heatmap(stats_df: pd.DataFrame, out_dir: str):
    mean_cols = [c for c in stats_df.columns if c.endswith("_mean")]
    hmap = stats_df.set_index("feature")[mean_cols].copy()
    hmap = hmap.apply(lambda r: (r - r.mean()) / (r.std() + 1e-8), axis=1)
    hmap.columns = [c.replace("_mean", "") for c in hmap.columns]
    # Сортируем по H-статистике
    if "kruskal_H" in stats_df.columns:
        order = stats_df.sort_values("kruskal_H", ascending=False)["feature"].tolist()
        hmap  = hmap.reindex([f for f in order if f in hmap.index])
    plt.figure(figsize=(7, 10))
    sns.heatmap(hmap, cmap="RdBu_r", center=0,
                annot=True, fmt=".2f", linewidths=0.5,
                cbar_kws={"label": "Z-score"})
    plt.title("Z-нормализованные признаки по доменам\n"
              "(отсортировано по убыванию H-статистики)", fontsize=12)
    plt.tight_layout()
    p = os.path.join(out_dir, "06_feature_heatmap.png")
    plt.savefig(p, dpi=150); plt.close(); print(f"  {p}")


# ══════════════════════════════════════════════════════════════
#  6. ИТОГОВЫЙ ОТЧЁТ
# ══════════════════════════════════════════════════════════════

def print_final_report(feat_df: pd.DataFrame, stats_df: pd.DataFrame):
    print(f"\n{'='*60}\nИТОГОВЫЙ ОТЧЁТ\n{'='*60}")

    print("\n── Состав выборки ──────────────────────────────────────")
    summary = feat_df.groupby(GROUP_COL).agg(
        файлов     = ("utterance_id", "count"),
        спикеров   = ("speaker_id",   "nunique"),
        дур_mean   = ("duration_sec",  "mean"),
        дур_std    = ("duration_sec",  "std"),
        snr_mean   = ("snr_db",        "mean"),
    ).round(2)
    print(summary.to_string())

    print("\n── Признаки, значимо различающие домены (p < 0.05) ────")
    sig = stats_df[stats_df["sig"].isin(["*","**","***"])]\
              .sort_values("kruskal_H", ascending=False)
    mean_cols = [c for c in stats_df.columns if c.endswith("_mean")]
    print(sig[["feature"] + mean_cols + ["kruskal_H","p_value","sig"]]
          .to_string(index=False))

    print("\n── Лингвистическая интерпретация ────────────────────────")
    notes = {
        "F1_median":         "Открытость гласных → аканье (high F1) vs оканье (low F1)",
        "F2_median":         "Передне-заднее положение → вокализм диалекта",
        "F3_median":         "3-й формант → огубленность, ретрофлексия",
        "F0_mean":           "Высота голоса (возраст контролирован частично)",
        "F0_range":          "Мелодический диапазон → пословный контур Пёзы",
        "zcr":               "Аффрикаты/шумные → цоканье, консонантизм",
        "voiced_ratio":      "Доля звонких фреймов → диереза, редукция",
        "spectral_centroid": "Центр тяжести спектра → тембральные различия",
        "mfcc_3":            "3-й MFCC → структура 2-й форманты",
    }
    for feat, note in notes.items():
        row = stats_df[stats_df["feature"] == feat]
        if not row.empty:
            sig_v = row["sig"].values[0]
            h_v   = row["kruskal_H"].values[0]
            if not pd.isna(h_v):
                print(f"  {feat:<22} [{sig_v:3s}  H={h_v:6.1f}]  {note}")


# ══════════════════════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*55)
    print("ШАГ 1: Фильтрация манифеста")
    df_filtered = filter_manifest(MANIFEST_PATH)

    print(f"\n{'='*55}")
    print("ШАГ 2: Извлечение признаков")
    feat_df = build_feature_table(df_filtered, CORPUS_ROOT)

    print(f"\n{'='*55}")
    print("ШАГ 3: Верификация пола через F0")
    feat_df = verify_gender_by_f0(feat_df)

    feat_csv = os.path.join(OUTPUT_DIR, "features_honest.csv")
    feat_df.to_csv(feat_csv, index=False)
    print(f"\nТаблица признаков: {feat_csv}")

    print(f"\n{'='*55}")
    print("ШАГ 4: Статистика (Краскел–Уоллис + Манн–Уитни + Бонферрони)")
    stats_df = domain_statistics(feat_df)

    mean_cols    = [c for c in stats_df.columns if c.endswith("_mean")]
    display_cols = ["feature"] + mean_cols + ["kruskal_H","p_value","sig"]
    print(stats_df[[c for c in display_cols if c in stats_df.columns]]
          .to_string(index=False))

    stats_csv = os.path.join(OUTPUT_DIR, "statistics_honest.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"Статистика: {stats_csv}")

    print(f"\n{'='*55}")
    print("ШАГ 5: Визуализации")
    plot_confounders_check(df_filtered, OUTPUT_DIR)
    plot_vowel_space(feat_df, OUTPUT_DIR)
    plot_f0_by_speaker(feat_df, OUTPUT_DIR)
    plot_f0_kde(feat_df, OUTPUT_DIR)
    plot_mfcc_boxplots(feat_df, OUTPUT_DIR)
    plot_pairwise_significance(stats_df, OUTPUT_DIR)
    plot_feature_heatmap(stats_df, OUTPUT_DIR)

    print_final_report(feat_df, stats_df)
