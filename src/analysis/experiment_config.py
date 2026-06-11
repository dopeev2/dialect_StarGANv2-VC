"""
experiment_config.py
====================
Единая конфигурация для всего экспериментального пайплайна.

Согласована с:
  - manifest_clean_partial.csv (фактические данные)
  - INFERENCE_v2.ipynb (sr=22050, 3 домена, параметры мела)
  - Раздел 3.1 ВКР (сценарии, метрики, контрольные группы)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# ─────────────────────────────────────────────────────────────
# Пути (адаптировать под Colab)
# ─────────────────────────────────────────────────────────────
DRIVE = Path("/content/drive/MyDrive/diploma")

AUDIO_DIR        = DRIVE / "corpus_22k"
MANIFEST_PATH    = AUDIO_DIR / "manifest_clean_partial.csv"
CONVERTED_DIR    = DRIVE / "converted"
RESULTS_DIR      = DRIVE / "results"

STARGAN_CKPT     = DRIVE / "stargan_22k_v2/Models/dialect/epoch_00102.pth"
STARGAN_CFG      = DRIVE / "stargan_22k_v2/Models/dialect/config.yml"
HIFIGAN_CKPT     = DRIVE / "hifigan_dialect/checkpoints_universal/g_02530000"
HIFIGAN_CFG      = DRIVE / "hifigan_dialect/config_finetune.json"

STARGAN_REPO     = "/content/StarGANv2-VC"
HIFIGAN_REPO     = "/content/hifi-gan"

# ─────────────────────────────────────────────────────────────
# Аудио параметры (из INFERENCE_v2.ipynb — НЕ менять!)
# ─────────────────────────────────────────────────────────────
SR        = 22_050   # ← 22050, не 24000
N_FFT     = 1024
HOP       = 256
WIN       = 1024
N_MELS    = 80
F_MIN     = 0
F_MAX     = 8000

# Нормализация мела
STARGAN_NORM_MEAN  = -4.0
STARGAN_NORM_STD   =  4.0
CORPUS_MEAN        = -2.5696
CORPUS_STD         =  4.1923
HIFI_MEAN          = -4.9008
HIFI_STD           =  2.1360

# ─────────────────────────────────────────────────────────────
# Домены (Таблица 3 ВКР + нотбук)
# ─────────────────────────────────────────────────────────────
# Индексы как в нотбуке: 0=pyoza, 1=don, 2=standard
DOMAIN_IDX: Dict[str, int] = {
    "pyoza_dialect": 0,
    "don_dialect":   1,
    "standard":      2,
}

# Группы speaker_id → domain_id
SPEAKER_TO_DOMAIN: Dict[str, str] = {
    # Донские говоры
    "don_TNG1957": "don_dialect",
    "don_EIV1939": "don_dialect",
    "don_VIK1941": "don_dialect",
    "don_GLT1934": "don_dialect",
    "don_KVA1948": "don_dialect",
    "don_MLI1941": "don_dialect",
    "don_SVS1939": "don_dialect",
    # Пёзские говоры
    "pyoza_MAN1910": "pyoza_dialect",
    "pyoza_GGS1932": "pyoza_dialect",
    "pyoza_GLA1926": "pyoza_dialect",
    "pyoza_MGG1932": "pyoza_dialect",
    "pyoza_KE1919":  "pyoza_dialect",
    "pyoza_AME1920": "pyoza_dialect",
    "pyoza_GVA1914": "pyoza_dialect",
    "pyoza_MAT1915": "pyoza_dialect",
    # Нормативный домен
    "don_interviewer":     "standard",
    "pyoza_interviewer":   "standard",
    "librispeech_295":     "standard",
    "librispeech_8086":    "standard",
    "librispeech_8169":    "standard",
    "librispeech_9014":    "standard",
}

# ─────────────────────────────────────────────────────────────
# Сценарии конверсии (Таблица 3 ВКР)
# ─────────────────────────────────────────────────────────────
@dataclass
class Scenario:
    id:         str
    src:        str   # domain_id
    tgt:        str   # domain_id
    src_idx:    int   # индекс домена в модели
    tgt_idx:    int
    label:      str
    task:       str
    is_primary: bool

SCENARIOS: List[Scenario] = [
    Scenario("S1", "don_dialect",   "standard",      1, 2,
             "Don → Std",   "Нормализация южнорусского говора",   True),
    Scenario("S2", "pyoza_dialect", "standard",      0, 2,
             "Pyoza → Std", "Нормализация севернорусского говора", True),
    Scenario("S3", "don_dialect",   "pyoza_dialect", 1, 0,
             "Don → Pyoza", "Межд. конверсия (контроль)",          False),
    Scenario("S4", "standard",      "don_dialect",   2, 1,
             "Std → Don",   "Обратная конверсия (контроль)",       False),
]
SCENARIO_MAP = {s.id: s for s in SCENARIOS}

# ─────────────────────────────────────────────────────────────
# Фактический состав тестовой выборки (из manifest_clean_partial.csv)
# ─────────────────────────────────────────────────────────────
# ВАЖНО: В манифесте 710 файлов в тесте ВСЕГО (не на домен!)
# don_dialect: 327 | pyoza_dialect: 187 | standard: 196
# Это расходится с Таблицей 4 ВКР (710 на домен)
# Возможные причины: manifest_clean_partial — частичный срез

ACTUAL_TEST_COUNTS = {
    "don_dialect":   327,
    "pyoza_dialect": 187,
    "standard":      196,
}

# Дикторы в тестовой выборке (из анализа манифеста)
TEST_SPEAKERS = {
    "don_dialect":   ["don_EIV1939", "don_GLT1934", "don_KVA1948",
                      "don_MLI1941", "don_SVS1939", "don_TNG1957", "don_VIK1941"],
    "pyoza_dialect": ["pyoza_AME1920", "pyoza_GGS1932", "pyoza_GLA1926",
                      "pyoza_GVA1914", "pyoza_KE1919", "pyoza_MAN1910",
                      "pyoza_MAT1915", "pyoza_MGG1932"],
    "standard":      ["don_interviewer", "librispeech_295", "librispeech_8086",
                      "librispeech_8169", "librispeech_9014", "pyoza_interviewer"],
}

# ─────────────────────────────────────────────────────────────
# Параметры метрик (Таблица 5 ВКР)
# ─────────────────────────────────────────────────────────────
@dataclass
class MetricsCfg:
    # MCD
    n_mfcc:     int   = 13
    dtw_radius: int   = 10

    # F0
    f0_method:  str   = "pyin"
    f0_fmin:    float = 75.0
    f0_fmax:    float = 500.0
    f0_voiced_thr: float = 0.1

    # WER
    whisper_model: str = "base"
    whisper_lang:  str = "ru"
    wer_delta_max: float = 5.0   # п.п.

    # ΔMFCC
    n_formants:    int = 2
    lpc_order:     int = 14

    # F0 PCC порог "хорошо"
    f0_pcc_good: float = 0.70

    # Воспроизводимость
    seed: int = 42

METRICS = MetricsCfg()

# ─────────────────────────────────────────────────────────────
# Контрольные группы (раздел 3.1.2)
# ─────────────────────────────────────────────────────────────
BASELINES = {
    "G0": "Passthrough — исходный сигнал без конверсии (верхняя граница MCD)",
    "G1": "Domain centroid — ближайший по DTW файл целевого домена (нижняя граница)",
    "G2": "Self-conversion — Std → Std (тест стабильности генератора)",
}
