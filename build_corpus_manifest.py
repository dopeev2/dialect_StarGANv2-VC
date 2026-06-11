"""
build_corpus_manifest.py
========================
Сборка единого обучающего корпуса из трёх источников:
  1. don_corpus_output    -- Донской диалектный корпус (сырые файлы)
  2. pyoza_corpus_output  -- Корпус Пёзы (сырые файлы)
  3. librispeech_ru       -- LibriSpeech русский

Что делает скрипт:
  - Копирует все WAV-файлы в training_corpus/raw/<speaker_id>/<utt_id>.wav
  - Собирает manifest_raw.csv с метаданными по каждой записи
  - Сохраняет отчёт о записях без транскрипта -> missing_transcripts.csv
  - Печатает подробную статистику

Итоговая структура:
  training_corpus/
    raw/
      don_EIV1939/
        don_EIV1939_utt001.wav
        ...
      pyoza_XXX/
        pyoza_XXX_utt001.wav
        ...
      don_interviewer/
        don_interviewer_utt001.wav
        ...
      librispeech_295/
        librispeech_295_utt001.wav
        ...
    manifest_raw.csv
    missing_transcripts.csv   (если есть записи без транскрипта)

Схема доменов:
  don_dialect    -- dialect/ из don_corpus_output
  pyoza_dialect  -- dialect/ из pyoza_corpus_output
  standard       -- interviewer/ из обоих корпусов + librispeech_ru

Колонки manifest_raw.csv (совместимы с preprocess_corpus.py):
  utterance_id, rel_path, speaker_id, domain_id, dialect_group,
  gender, birth_year, duration_sec, snr_db, rms_db,
  quality_tier, has_nrzb, transcript, split

  Примечания:
  - snr_db, rms_db, duration_sec, gender, quality_tier, split
    заполняются NaN -- вычисляются на этапе препроцессинга
  - rel_path -- ОТНОСИТЕЛЬНО training_corpus/
    (т.е. raw/<speaker_id>/<file>.wav)

Использование:
  python build_corpus_manifest.py \
      --don_root       "D:/DIPLOMA/don_corpus_output" \
      --pyoza_root     "D:/DIPLOMA/pyoza_corpus_output" \
      --libri_root     "D:/DIPLOMA/librispeech_ru" \
      --libri_manifest "D:/DIPLOMA/librispeech_ru/manifest_librispeech.csv" \
      --out_dir        "D:/DIPLOMA/training_corpus"

  Флаги:
      --dry_run   -- только напечатать статистику, не копировать файлы
      --overwrite -- перезаписывать уже скопированные файлы
"""

import argparse
import re
import shutil
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------
# КОНСТАНТЫ
# -----------------------------------------------------------------

NRZB_PATTERN = re.compile(r'\[нрзб\]', re.IGNORECASE)
NAN = float('nan')


# -----------------------------------------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -----------------------------------------------------------------

def detect_nrzb(text: str) -> bool:
    """True если транскрипт содержит маркер [нрзб]."""
    return bool(NRZB_PATTERN.search(text))


def extract_birth_year(folder_name: str) -> float:
    """
    Извлекает год рождения из имени папки диктора.
    Формат: любые символы + 4-значный год в конце, например EIV1939 -> 1939.0
    Возвращает NaN если год не найден или вне диапазона 1880-2000.
    """
    match = re.search(r'(\d{4})$', folder_name)
    if match:
        year = int(match.group(1))
        if 1880 <= year <= 2000:
            return float(year)
    return NAN


def read_transcript(txt_path: Path) -> str:
    """Читает транскрипт из .txt файла. Возвращает '' если файл не найден."""
    if not txt_path.exists():
        return ""
    return txt_path.read_text(encoding='utf-8', errors='replace').strip()


def make_record(
    utterance_id: str,
    rel_path: str,
    speaker_id: str,
    domain_id: str,
    dialect_group: str,
    birth_year: float,
    transcript: str,
) -> dict:
    """Создаёт строку манифеста с NaN-полями для препроцессинга."""
    return {
        "utterance_id":  utterance_id,
        "rel_path":      rel_path,
        "speaker_id":    speaker_id,
        "domain_id":     domain_id,
        "dialect_group": dialect_group,
        "gender":        NAN,
        "birth_year":    birth_year,
        "duration_sec":  NAN,
        "snr_db":        NAN,
        "rms_db":        NAN,
        "quality_tier":  NAN,
        "has_nrzb":      detect_nrzb(transcript),
        "transcript":    transcript,
        "split":         NAN,
    }


def copy_file(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> bool:
    """
    Копирует src -> dst.
    Возвращает True если файл скопирован (или было бы скопировано в dry_run).
    False если файл уже существует и overwrite=False.
    """
    if dry_run:
        return True
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


# -----------------------------------------------------------------
# ПАРСИНГ ДИАЛЕКТНОГО КОРПУСА (don / pyoza)
# -----------------------------------------------------------------

def parse_dialect_corpus(
    corpus_root:    Path,
    corpus_prefix:  str,
    dialect_domain: str,
    dialect_group:  str,
    raw_out:        Path,
    overwrite:      bool,
    dry_run:        bool,
) -> list:
    """
    Обходит dialect/ и interviewer/ внутри corpus_root.
    Копирует WAV в raw/<speaker_id>/<utterance_id>.wav
    Возвращает список записей для манифеста.
    """
    records = []
    skipped = 0

    # -- 1. dialect/ ------------------------------------------
    dialect_root = corpus_root / "dialect"
    if not dialect_root.exists():
        print(f"  [WARN] Не найдена папка: {dialect_root}")
    else:
        speaker_dirs = sorted([d for d in dialect_root.iterdir() if d.is_dir()])
        print(f"  [{corpus_prefix}] dialect/ -- дикторов: {len(speaker_dirs)}")

        for spk_dir in tqdm(speaker_dirs, desc=f"  {corpus_prefix}/dialect", leave=False):
            raw_spk_name = spk_dir.name
            speaker_id   = f"{corpus_prefix}_{raw_spk_name}"
            birth_year   = extract_birth_year(raw_spk_name)

            audios_dir = spk_dir / "audios"
            texts_dir  = spk_dir / "texts"

            if not audios_dir.exists():
                print(f"    [WARN] Нет audios/: {spk_dir}")
                continue

            wav_files = sorted(audios_dir.glob("*.wav"))
            if not wav_files:
                print(f"    [WARN] Нет WAV файлов: {audios_dir}")
                continue

            dst_spk_dir = raw_out / speaker_id

            for wav_src in wav_files:
                stem = wav_src.stem

                # Новое имя файла: speaker_id + оригинальный stem для уникальности
                new_filename = f"{speaker_id}_{stem}.wav"
                utterance_id = f"{speaker_id}_{stem}"
                dst_wav      = dst_spk_dir / new_filename
                rel_path     = f"raw/{speaker_id}/{new_filename}"

                txt_path   = texts_dir / f"{stem}.txt"
                transcript = read_transcript(txt_path)

                copied = copy_file(wav_src, dst_wav, overwrite, dry_run)
                if not copied:
                    skipped += 1

                records.append(make_record(
                    utterance_id  = utterance_id,
                    rel_path      = rel_path,
                    speaker_id    = speaker_id,
                    domain_id     = dialect_domain,
                    dialect_group = dialect_group,
                    birth_year    = birth_year,
                    transcript    = transcript,
                ))

    # -- 2. interviewer/ --------------------------------------
    interviewer_audio = corpus_root / "interviewer" / "interviewer" / "audios"
    interviewer_texts = corpus_root / "interviewer" / "interviewer" / "texts"

    if not interviewer_audio.exists():
        print(f"  [WARN] Не найдена папка интервьюера: {interviewer_audio}")
    else:
        speaker_id  = f"{corpus_prefix}_interviewer"
        wav_files   = sorted(interviewer_audio.glob("*.wav"))
        dst_spk_dir = raw_out / speaker_id

        print(f"  [{corpus_prefix}] interviewer/ -- записей: {len(wav_files)}")

        for wav_src in tqdm(wav_files, desc=f"  {corpus_prefix}/interviewer", leave=False):
            stem = wav_src.stem

            new_filename = f"{speaker_id}_{stem}.wav"
            utterance_id = f"{speaker_id}_{stem}"
            dst_wav      = dst_spk_dir / new_filename
            rel_path     = f"raw/{speaker_id}/{new_filename}"

            txt_path   = interviewer_texts / f"{stem}.txt"
            transcript = read_transcript(txt_path)

            copied = copy_file(wav_src, dst_wav, overwrite, dry_run)
            if not copied:
                skipped += 1

            records.append(make_record(
                utterance_id  = utterance_id,
                rel_path      = rel_path,
                speaker_id    = speaker_id,
                domain_id     = "standard",
                dialect_group = "standard",
                birth_year    = NAN,
                transcript    = transcript,
            ))

    if skipped > 0:
        print(f"  [{corpus_prefix}] Пропущено (уже существуют): {skipped}")

    return records


# -----------------------------------------------------------------
# ПАРСИНГ LIBRISPEECH
# -----------------------------------------------------------------

def parse_librispeech(
    libri_root:     Path,
    libri_manifest: Path,
    raw_out:        Path,
    overwrite:      bool,
    dry_run:        bool,
) -> list:
    """
    Читает manifest_librispeech.csv.
    Копирует WAV в raw/<speaker_id>/<utterance_id>.wav
    Все записи -> домен 'standard'.
    """
    df = pd.read_csv(libri_manifest)
    print(f"  [librispeech] записей в манифесте: {len(df)}")

    records = []
    missing = 0
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="  librispeech", leave=False):

        utterance_id = str(row["utterance_id"])
        speaker_id   = str(row["speaker_id"])
        transcript   = str(row.get("transcript", "")).strip()
        src_rel_path = str(row["rel_path"])

        wav_src = libri_root / src_rel_path
        if not wav_src.exists():
            missing += 1
            continue

        # utterance_id в LibriSpeech уже глобально уникален
        new_filename = f"{utterance_id}.wav"
        dst_wav      = raw_out / speaker_id / new_filename
        rel_path     = f"raw/{speaker_id}/{new_filename}"

        copied = copy_file(wav_src, dst_wav, overwrite, dry_run)
        if not copied:
            skipped += 1

        records.append(make_record(
            utterance_id  = utterance_id,
            rel_path      = rel_path,
            speaker_id    = speaker_id,
            domain_id     = "standard",
            dialect_group = "standard",
            birth_year    = NAN,
            transcript    = transcript,
        ))

    if missing > 0:
        print(f"  [librispeech] WARN: не найдено WAV: {missing}")
    if skipped > 0:
        print(f"  [librispeech] Пропущено (уже существуют): {skipped}")

    return records


# -----------------------------------------------------------------
# СТАТИСТИКА
# -----------------------------------------------------------------

def print_statistics(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"ИТОГОВАЯ СТАТИСТИКА КОРПУСА")
    print(f"{'='*60}")
    print(f"Всего записей: {len(df)}")

    print(f"\n-- По domain_id {'--'*20}")
    print(df.groupby("domain_id")["utterance_id"].count()
            .rename("n_records").to_string())

    print(f"\n-- По dialect_group {'--'*18}")
    print(df.groupby("dialect_group")["utterance_id"].count()
            .rename("n_records").to_string())

    print(f"\n-- По speaker_id {'--'*20}")
    spk = (
        df.groupby(["domain_id", "speaker_id"])["utterance_id"]
        .count().rename("n_records").reset_index()
        .sort_values(["domain_id", "n_records"], ascending=[True, False])
    )
    print(spk.to_string(index=False))

    print(f"\n-- Транскрипты {'--'*22}")
    print(f"  Без транскрипта: {(df['transcript'] == '').sum()}")
    print(f"  Содержат [нрзб]: {df['has_nrzb'].sum()}")
    print(f"{'='*60}\n")


# -----------------------------------------------------------------
# ТОЧКА ВХОДА
# -----------------------------------------------------------------

def main(args):
    don_root       = Path(args.don_root)
    pyoza_root     = Path(args.pyoza_root)
    libri_root     = Path(args.libri_root)
    libri_manifest = Path(args.libri_manifest)
    out_dir        = Path(args.out_dir)
    raw_out        = out_dir / "raw"

    dry_run   = args.dry_run
    overwrite = args.overwrite

    if dry_run:
        print("[DRY RUN] Файлы копироваться не будут.\n")

    # -- Проверка путей ---------------------------------------
    for p, name in [
        (don_root,       "--don_root"),
        (pyoza_root,     "--pyoza_root"),
        (libri_root,     "--libri_root"),
        (libri_manifest, "--libri_manifest"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Путь не найден ({name}): {p}")

    if not dry_run:
        raw_out.mkdir(parents=True, exist_ok=True)

    all_records = []

    # -- 1. Don -----------------------------------------------
    print(f"\n[1/3] Don corpus: {don_root}")
    don_records = parse_dialect_corpus(
        corpus_root    = don_root,
        corpus_prefix  = "don",
        dialect_domain = "don_dialect",
        dialect_group  = "don",
        raw_out        = raw_out,
        overwrite      = overwrite,
        dry_run        = dry_run,
    )
    print(f"  -> {len(don_records)} записей")
    all_records.extend(don_records)

    # -- 2. Pyoza ---------------------------------------------
    print(f"\n[2/3] Pyoza corpus: {pyoza_root}")
    pyoza_records = parse_dialect_corpus(
        corpus_root    = pyoza_root,
        corpus_prefix  = "pyoza",
        dialect_domain = "pyoza_dialect",
        dialect_group  = "pyoza",
        raw_out        = raw_out,
        overwrite      = overwrite,
        dry_run        = dry_run,
    )
    print(f"  -> {len(pyoza_records)} записей")
    all_records.extend(pyoza_records)

    # -- 3. LibriSpeech ---------------------------------------
    print(f"\n[3/3] LibriSpeech: {libri_root}")
    libri_records = parse_librispeech(
        libri_root     = libri_root,
        libri_manifest = libri_manifest,
        raw_out        = raw_out,
        overwrite      = overwrite,
        dry_run        = dry_run,
    )
    print(f"  -> {len(libri_records)} записей")
    all_records.extend(libri_records)

    # -- DataFrame --------------------------------------------
    df = pd.DataFrame(all_records)

    dupes = df["utterance_id"].duplicated().sum()
    if dupes > 0:
        print(f"\n[WARN] Дублирующихся utterance_id: {dupes}")
        df = df.drop_duplicates(subset="utterance_id", keep="first")
        print(f"  Дубликаты удалены, осталось: {len(df)}")

    print_statistics(df)

    # -- Сохранение -------------------------------------------
    if not dry_run:
        manifest_path = out_dir / "manifest_raw.csv"
        df.to_csv(manifest_path, index=False, encoding="utf-8")
        print(f"Манифест сохранён: {manifest_path}")

        empty = df[df["transcript"] == ""]
        if len(empty) > 0:
            report_path = out_dir / "missing_transcripts.csv"
            empty[["utterance_id", "rel_path", "speaker_id", "domain_id"]] \
                .to_csv(report_path, index=False, encoding="utf-8")
            print(f"[WARN] {len(empty)} записей без транскрипта -> {report_path}")
    else:
        print("[DRY RUN] Манифест не сохранён.")

    print(f"\nГотово. Всего записей в корпусе: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Сборка единого обучающего корпуса из диалектных данных и LibriSpeech"
    )
    parser.add_argument("--don_root",       required=True,
                        help="Путь к don_corpus_output/")
    parser.add_argument("--pyoza_root",     required=True,
                        help="Путь к pyoza_corpus_output/")
    parser.add_argument("--libri_root",     required=True,
                        help="Путь к librispeech_ru/")
    parser.add_argument("--libri_manifest", required=True,
                        help="Путь к manifest_librispeech.csv")
    parser.add_argument("--out_dir",        required=True,
                        help="Выходная папка (внутри создаётся raw/ и manifest_raw.csv)")
    parser.add_argument("--dry_run",        action="store_true",
                        help="Только статистика, без копирования файлов")
    parser.add_argument("--overwrite",      action="store_true",
                        help="Перезаписывать уже существующие файлы")
    args = parser.parse_args()
    main(args)
