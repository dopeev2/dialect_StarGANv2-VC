"""
step3_verify_quality.py
=======================
ЭТАП 3 — Локальная верификация качества очищенного корпуса.

Запускается после скачивания corpus_clean/ с Google Drive.
Генерирует подробный отчёт и выявляет оставшиеся проблемы.

Установка:
    pip install pandas numpy librosa soundfile matplotlib tqdm

Использование:
    python step3_verify_quality.py \
        --manifest  "D:/DIPLOMA/corpus_clean/manifest_clean.csv" \
        --corpus    "D:/DIPLOMA/corpus_clean" \
        --out_report "D:/DIPLOMA/corpus_clean/quality_report.txt" \
        --sample_check 200
"""

import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

warnings.filterwarnings('ignore')


def compute_snr_db(y: np.ndarray, frame_len: int = 2048) -> float:
    if len(y) < frame_len * 2:
        return float(np.clip(20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-10), 0, 60))
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=frame_len//2)
    rms_frames = np.sqrt(np.mean(frames**2, axis=0))
    n = len(rms_frames)
    sorted_rms = np.sort(rms_frames)
    sig = np.median(sorted_rms[int(0.75*n):]) + 1e-10
    noi = np.median(sorted_rms[:max(1, int(0.25*n))]) + 1e-10
    return float(np.clip(20 * np.log10(sig / noi), 0, 60))


def check_audio_integrity(wav_path: str, expected_sr: int = 24000) -> dict:
    """Проверяет физическую целостность аудиофайла."""
    result = {
        'ok': True,
        'issues': [],
        'duration_sec': 0,
        'snr_db': 0,
        'rms_db': -99,
        'actual_sr': 0,
    }
    try:
        y, sr = sf.read(wav_path, dtype='float32', always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        result['actual_sr'] = sr
        result['duration_sec'] = len(y) / sr

        if sr != expected_sr:
            result['issues'].append(f'SR={sr} (ожидалось {expected_sr})')
        if len(y) == 0:
            result['issues'].append('пустой файл')
            result['ok'] = False
            return result

        rms = np.sqrt(np.mean(y**2))
        result['rms_db'] = float(20 * np.log10(rms + 1e-10))

        # Клиппирование
        clip_ratio = np.mean(np.abs(y) > 0.99)
        if clip_ratio > 0.001:
            result['issues'].append(f'клиппирование {clip_ratio*100:.2f}%')

        # Тишина (слишком тихий сигнал)
        if rms < 1e-4:
            result['issues'].append(f'слишком тихо (RMS={result["rms_db"]:.1f} дБ)')
            result['ok'] = False
            return result

        # SNR
        result['snr_db'] = compute_snr_db(y)

        if result['issues']:
            result['ok'] = False

    except Exception as e:
        result['ok'] = False
        result['issues'].append(str(e))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--out_report', default=None)
    parser.add_argument('--sample_check', type=int, default=300,
                        help='Количество файлов для физической проверки')
    args = parser.parse_args()

    corpus_root = Path(args.corpus)
    df = pd.read_csv(args.manifest)

    lines = []

    def log(s=''):
        print(s)
        lines.append(s)

    log('=' * 62)
    log('  ОТЧЁТ О КАЧЕСТВЕ КОРПУСА ПОСЛЕ ДЕНОИЗИНГА')
    log('=' * 62)
    log(f'  Манифест:  {args.manifest}')
    log(f'  Записей:   {len(df)}')

    # ── Общая статистика ─────────────────────────────────────
    total_min = df['duration_sec'].sum() / 60
    log(f'\n── ДЛИТЕЛЬНОСТЬ ────────────────────────────────────────')
    log(f'  Итого:     {total_min:.1f} мин ({total_min/60:.2f} ч)')
    log(f'  Среднее:   {df["duration_sec"].mean():.2f} с')
    log(f'  Медиана:   {df["duration_sec"].median():.2f} с')
    log(f'  1–3 с:     {((df["duration_sec"]>=1)&(df["duration_sec"]<3)).sum()}')
    log(f'  3–6 с:     {((df["duration_sec"]>=3)&(df["duration_sec"]<6)).sum()}')
    log(f'  6–10 с:    {((df["duration_sec"]>=6)&(df["duration_sec"]<=10)).sum()}')

    # ── SNR ───────────────────────────────────────────────────
    log(f'\n── SNR (из манифеста) ───────────────────────────────────')
    log(f'  Среднее:   {df["snr_db"].mean():.1f} дБ')
    log(f'  Медиана:   {df["snr_db"].median():.1f} дБ')
    log(f'  P5/P95:    {df["snr_db"].quantile(0.05):.1f} / {df["snr_db"].quantile(0.95):.1f} дБ')
    bins = [0, 10, 15, 20, 25, 60]
    labels = ['10–15', '15–20', '20–25', '>25']
    snr_cut = pd.cut(df['snr_db'], bins=[0,10,15,20,25,60],
                     labels=['<10','10–15','15–20','20–25','>25'])
    for label, count in snr_cut.value_counts().sort_index().items():
        bar = '█' * (count // 80)
        log(f'  {label:>6} дБ: {count:>5}  {bar}')

    # ── Дикторы ───────────────────────────────────────────────
    log(f'\n── ДИКТОРЫ ─────────────────────────────────────────────')
    spk = (
        df.groupby('speaker_id')['duration_sec']
        .agg(['count', 'sum'])
        .assign(total_min=lambda x: (x['sum'] / 60).round(1))
        .drop(columns='sum')
        .sort_values('total_min', ascending=False)
    )
    for sp, row in spk.iterrows():
        icon = '✅' if row['total_min'] >= 45 else ('🟡' if row['total_min'] >= 15 else '🔴')
        log(f'  {icon} {sp:<30} {row["count"]:>5} уттер. | {row["total_min"]:>6.1f} мин')

    # ── По диалектным группам ─────────────────────────────────
    log(f'\n── ДИАЛЕКТНЫЕ ГРУППЫ ────────────────────────────────────')
    grp = df.groupby('dialect_group')['duration_sec'].agg(['count', 'sum'])
    for g, r in grp.iterrows():
        log(f'  {g:<20} {r["count"]:>5} уттер. | {r["sum"]/60:>6.1f} мин')

    # ── Физическая проверка файлов ────────────────────────────
    log(f'\n── ФИЗИЧЕСКАЯ ПРОВЕРКА ({args.sample_check} случ. файлов) ──────────────')
    sample_df = df.sample(min(args.sample_check, len(df)), random_state=42)

    broken = []
    snr_recomputed = []

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df),
                        desc='Проверка файлов'):
        wav_path = corpus_root / row['rel_path']
        result = check_audio_integrity(str(wav_path))
        snr_recomputed.append(result['snr_db'])
        if not result['ok'] or result['issues']:
            broken.append({
                'id': row['utterance_id'],
                'issues': ', '.join(result['issues']),
                'path': str(wav_path),
            })

    if broken:
        log(f'\n  ⚠️  Найдено {len(broken)} проблемных файлов:')
        for b in broken[:10]:
            log(f'     {b["id"]}: {b["issues"]}')
        if len(broken) > 10:
            log(f'     ... и ещё {len(broken) - 10}')
    else:
        log(f'  ✅ Проблем не обнаружено')

    snr_arr = np.array([s for s in snr_recomputed if s > 0])
    if len(snr_arr) > 0:
        log(f'\n  Пересчитанный SNR по выборке:')
        log(f'    Среднее:  {snr_arr.mean():.1f} дБ')
        log(f'    Медиана:  {np.median(snr_arr):.1f} дБ')
        log(f'    Разница с манифестом: '
              f'{snr_arr.mean() - df["snr_db"].mean():+.1f} дБ')

    # ── Финальное заключение ──────────────────────────────────
    log(f'\n── ЗАКЛЮЧЕНИЕ ───────────────────────────────────────────')
    total_min = df['duration_sec'].sum() / 60
    n_good = (df['snr_db'] >= 20).sum()
    n_ok   = ((df['snr_db'] >= 10) & (df['snr_db'] < 20)).sum()
    n_bad  = (df['snr_db'] < 10).sum()

    log(f'  SNR ≥ 20 дБ (отличное): {n_good} ({n_good/len(df)*100:.1f}%)')
    log(f'  SNR 10–20 дБ (хорошее): {n_ok}  ({n_ok/len(df)*100:.1f}%)')
    log(f'  SNR < 10 дБ (плохое):   {n_bad}  ({n_bad/len(df)*100:.1f}%)')

    if n_bad > len(df) * 0.05:
        log(f'\n  ⚠️  Рекомендуется дополнительно отфильтровать {n_bad} записей')
        log(f'      с SNR < 10 дБ перед обучением.')
    else:
        log(f'\n  ✅ Корпус готов к обучению!')

    log(f'\n  Итоговый объём: {total_min:.1f} мин ({total_min/60:.2f} ч)')
    log('=' * 62)

    # Сохраняем отчёт
    if args.out_report:
        out_path = Path(args.out_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f'\n📄 Отчёт сохранён: {out_path}')


if __name__ == '__main__':
    main()
