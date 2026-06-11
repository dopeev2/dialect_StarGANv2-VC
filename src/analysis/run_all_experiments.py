# =====================================================================
# run_all_experiments.py
# Главный скрипт для запуска в Google Colab
# Полный пайплайн: inference + метрики + отчёт
#
# Запускать ячейками или целиком:
#   %run run_all_experiments.py
# =====================================================================

# ─────────────────────────────────────────────────────────────────────
# ЯЧЕЙКА 0: Монтирование и установка
# ─────────────────────────────────────────────────────────────────────
"""
from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'librosa', 'soundfile', 'munch', 'torchaudio',
    'fastdtw', 'openai-whisper', 'scipy',
])
print('✓ Зависимости установлены')
"""

# ─────────────────────────────────────────────────────────────────────
# ЯЧЕЙКА 1: Клонирование репозиториев (как в нотбуке INFERENCE_v2)
# ─────────────────────────────────────────────────────────────────────
"""
import os, shutil

if not os.path.exists('/content/hifi-gan'):
    os.system('git clone https://github.com/jik876/hifi-gan.git')

if not os.path.exists('/content/StarGANv2-VC'):
    shutil.copytree(
        '/content/drive/MyDrive/diploma/StarGANv2-VC_v2',
        '/content/StarGANv2-VC'
    )
print('✓ Репозитории готовы')
"""

# ─────────────────────────────────────────────────────────────────────
# ЯЧЕЙКА 2: Патч HiFi-GAN (из нотбука INFERENCE_v2, ячейка 3)
# ─────────────────────────────────────────────────────────────────────
"""
with open('/content/hifi-gan/meldataset.py', 'r') as f:
    content = f.read()

content = content.replace(
    'mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)',
    'mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)'
)
content = content.replace(
    "center=center, pad_mode='reflect', normalized=False, onesided=True)",
    "center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)\n    spec = torch.view_as_real(spec)"
)
with open('/content/hifi-gan/meldataset.py', 'w') as f:
    f.write(content)
print('✓ Патч применён')
"""

# ─────────────────────────────────────────────────────────────────────
# ЯЧЕЙКА 3: Копирование кода эксперимента
# ─────────────────────────────────────────────────────────────────────
"""
import shutil, sys
from pathlib import Path

# Копируем experiment_config.py, batch_inference.py, compute_metrics.py
CODE_SRC = Path('/content/drive/MyDrive/diploma/code')
for fname in ['experiment_config.py', 'batch_inference.py', 'compute_metrics.py']:
    src = CODE_SRC / fname
    dst = Path('/content') / fname
    if src.exists():
        shutil.copy(str(src), str(dst))
        print(f'✓ Скопирован: {fname}')
    else:
        print(f'✗ Не найден: {src}')

sys.path.insert(0, '/content')
"""

# ─────────────────────────────────────────────────────────────────────
# ЯЧЕЙКА 4: Загрузка моделей
# ─────────────────────────────────────────────────────────────────────
"""
from batch_inference import load_models, Converter
from experiment_config import *

print('Загружаю модели...')
generator, style_enc, mapping_net, F0_model, hifigan, mp, device = load_models(
    stargan_ckpt = STARGAN_CKPT,
    stargan_cfg  = STARGAN_CFG,
    hifigan_ckpt = HIFIGAN_CKPT,
    hifigan_cfg  = HIFIGAN_CFG,
    stargan_repo = STARGAN_REPO,
    hifigan_repo = HIFIGAN_REPO,
    device       = 'cuda',
)

converter = Converter(generator, style_enc, mapping_net, F0_model, hifigan, mp, device)
print('✓ Все модели загружены')
"""

# ─────────────────────────────────────────────────────────────────────
# ЯЧЕЙКА 5: Проверка inference на одном файле (санити-чек)
# ─────────────────────────────────────────────────────────────────────
"""
import pandas as pd, random
from pathlib import Path
from IPython.display import Audio, display

df      = pd.read_csv(MANIFEST_PATH)
test_df = df[df['split'] == 'test']

# Берём один файл из don_dialect
don_sample = test_df[test_df['domain_id']=='don_dialect'].sample(1).iloc[0]
src_path   = AUDIO_DIR / don_sample['rel_path']

# Референс из standard
std_sample = test_df[test_df['domain_id']=='standard'].sample(1).iloc[0]
ref_path   = AUDIO_DIR / std_sample['rel_path']

out_path = '/content/test_conversion.wav'

print(f'Источник:  {src_path.name}')
print(f'Референс:  {ref_path.name}')
print(f'Транскрипция: {don_sample["transcript"]}')

converter.convert(
    input_wav=str(src_path),
    target_domain=2,           # 2 = standard
    output_wav=out_path,
    reference_wav=str(ref_path),
)

print('\\nОригинал:')
display(Audio(str(src_path)))
print('Конвертация (Don → Std):')
display(Audio(out_path))
"""

# ─────────────────────────────────────────────────────────────────────
# ЯЧЕЙКА 6: Батч inference — все 4 сценария
# ─────────────────────────────────────────────────────────────────────
"""
import pandas as pd
from batch_inference import run_scenario_inference, run_self_conversion
from experiment_config import MANIFEST_PATH, AUDIO_DIR, CONVERTED_DIR, SCENARIOS

df      = pd.read_csv(MANIFEST_PATH)
test_df = df[df['split'] == 'test'].copy()

CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
all_logs = {}

# Основные сценарии
for sc in SCENARIOS:
    print(f'\\n{"="*50}')
    print(f'Сценарий {sc.id}: {sc.label} ({sc.task})')
    print('='*50)
    log = run_scenario_inference(
        scenario_id  = sc.id,
        test_df      = test_df,
        audio_dir    = AUDIO_DIR,
        out_dir      = CONVERTED_DIR,
        converter    = converter,
        resume       = True,    # Продолжить если прервалось
        seed         = 42,
    )
    all_logs[sc.id] = log
    ok = (log['status'] == 'ok').sum()
    print(f'  → {ok}/{len(log)} файлов конвертировано')

# G2: self-conversion (Std → Std)
print('\\nG2: Self-conversion...')
g2_log = run_self_conversion(test_df, AUDIO_DIR, CONVERTED_DIR, converter)
all_logs['G2'] = g2_log

print('\\n✓ Inference завершён')
print('Итог:')
for sid, log in all_logs.items():
    ok   = (log['status'] == 'ok').sum()
    skip = (log['status'] == 'skipped_exists').sum() if 'status' in log.columns else 0
    print(f'  {sid}: {ok} OK, {skip} пропущено')
"""

# ─────────────────────────────────────────────────────────────────────
# ЯЧЕЙКА 7: Вычисление метрик
# ─────────────────────────────────────────────────────────────────────
"""
import pandas as pd, json
from pathlib import Path
from compute_metrics import (
    build_standard_formant_stats, build_domain_f0_median,
    evaluate_scenario, build_report,
)
from experiment_config import (
    MANIFEST_PATH, AUDIO_DIR, CONVERTED_DIR, RESULTS_DIR, SCENARIO_MAP,
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df      = pd.read_csv(MANIFEST_PATH)
test_df = df[df['split'] == 'test'].copy()

# Нормативные файлы для формантных статистик
std_files = [
    AUDIO_DIR / r['rel_path']
    for _, r in test_df[test_df['domain_id']=='standard'].iterrows()
    if (AUDIO_DIR / r['rel_path']).exists()
][:100]

print(f'Вычисляю формантные статистики ({len(std_files)} файлов)...')
std_stats = build_standard_formant_stats(
    std_files,
    cache = RESULTS_DIR / 'std_formant_stats.json',
)
print('Результат:', std_stats)

# Медианный F0 по доменам
f0_medians = {}
for group in ['don_dialect', 'pyoza_dialect', 'standard']:
    pool = [
        AUDIO_DIR / r['rel_path']
        for _, r in test_df[test_df['domain_id']==group].iterrows()
        if (AUDIO_DIR / r['rel_path']).exists()
    ][:150]
    f0_medians[group] = build_domain_f0_median(
        pool,
        cache = RESULTS_DIR / f'f0_median_{group}.npy',
    )
    v = f0_medians[group]
    v_voiced = v[v > 0]
    print(f'F0 {group}: {v_voiced.mean():.1f} Гц (voiced mean)')

# Оценка всех сценариев
metric_dfs = {}
for sid in ['S1', 'S2', 'S3', 'S4']:
    print(f'\\nОцениваю {sid}...')
    df_m = evaluate_scenario(
        scenario_id = sid,
        test_df     = test_df,
        audio_dir   = AUDIO_DIR,
        conv_dir    = CONVERTED_DIR,
        results_dir = RESULTS_DIR,
        std_stats   = std_stats,
        f0_medians  = f0_medians,
        compute_wer_flag = False,    # WER отдельно ниже
    )
    metric_dfs[sid] = df_m
    print(f'  {sid}: {len(df_m)} файлов оценено')

# Сводный отчёт
build_report(metric_dfs, RESULTS_DIR)
"""

# ─────────────────────────────────────────────────────────────────────
# ЯЧЕЙКА 8: WER (отдельно — медленно, нужен GPU + Whisper)
# ─────────────────────────────────────────────────────────────────────
"""
import pandas as pd
from compute_metrics import compute_wer
from experiment_config import RESULTS_DIR, AUDIO_DIR, CONVERTED_DIR

# Запускаем WER только для S1 и S2 (основные сценарии)
for sid in ['S1', 'S2']:
    per_file_path = RESULTS_DIR / sid / 'metrics_per_file.csv'
    df_m = pd.read_csv(per_file_path)

    wer_rows = []
    for i, row in df_m.iterrows():
        conv_path = CONVERTED_DIR / sid / (Path(row.get('utterance_id','')).name + '.wav')
        # или восстановить путь из manifest...
        # Здесь нужно подставить свою логику маппинга utterance_id → путь
        pass

    print(f'{sid} WER будет добавлен после маппинга путей')
"""

# ─────────────────────────────────────────────────────────────────────
# ЯЧЕЙКА 9: Анализ ошибок (best/worst по MCD)
# ─────────────────────────────────────────────────────────────────────
"""
import pandas as pd
from pathlib import Path
from IPython.display import Audio, display
from experiment_config import RESULTS_DIR, AUDIO_DIR, CONVERTED_DIR

def show_extremes(sid: str, n: int = 3):
    csv = RESULTS_DIR / sid / 'metrics_per_file.csv'
    df  = pd.read_csv(csv).dropna(subset=['mcd_conv'])

    print(f'\\n===== {sid}: ЛУЧШИЕ (низкий MCD) =====')
    for _, r in df.nsmallest(n, 'mcd_conv').iterrows():
        print(f'  {r["utterance_id"]} | MCD={r["mcd_conv"]:.3f} | '
              f'F0_PCC={r.get("f0_pcc", float("nan")):.3f} | '
              f'spk={r["speaker_id"]}')

    print(f'\\n===== {sid}: ХУДШИЕ (высокий MCD) =====')
    for _, r in df.nlargest(n, 'mcd_conv').iterrows():
        print(f'  {r["utterance_id"]} | MCD={r["mcd_conv"]:.3f} | '
              f'F0_PCC={r.get("f0_pcc", float("nan")):.3f} | '
              f'spk={r["speaker_id"]}')

show_extremes('S1')
show_extremes('S2')
"""

# ─────────────────────────────────────────────────────────────────────
# ЯЧЕЙКА 10: Быстрый просмотр итоговых результатов
# ─────────────────────────────────────────────────────────────────────
"""
import pandas as pd
from pathlib import Path
from experiment_config import RESULTS_DIR

summary = pd.read_csv(RESULTS_DIR / 'summary_table.csv')
print(summary.to_string(index=False))

print('\\nLaTeX таблица:')
print((RESULTS_DIR / 'summary_latex.tex').read_text())
"""

print("run_all_experiments.py загружен. Запускайте ячейки последовательно.")
print("Структура пайплайна:")
print("  Ячейка 0-3: окружение и модели")
print("  Ячейка 5:   санити-чек одного файла")
print("  Ячейка 6:   батч inference (S1–S4 + G2)")
print("  Ячейка 7:   метрики (MCD, F0, ΔMFCC)")
print("  Ячейка 8:   WER (медленно, опционально)")
print("  Ячейка 9:   анализ best/worst")
print("  Ячейка 10:  итоговая таблица")
