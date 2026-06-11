#!/usr/bin/env python3
"""
Модифицированный парсер Донского корпуса.
Изменения по сравнению с оригиналом:
  1. Сохраняет оригинальные имена файлов (с временны́ми метками)
  2. Генерирует сводный manifest.csv
  3. Явно разделяет домены: dialect / interviewer / unknown
  4. Извлекает метаданные информанта из имени страницы
"""

import argparse
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Извлечение метаданных из имени страницы
# ---------------------------------------------------------------------------

def parse_page_name(page_name: str) -> dict:
    """
    Разбирает имя вида '050722_EIV1939_Melikhovskaya_2'.
    
    Возвращает словарь с метаданными информанта.
    """
    parts = page_name.split("_")
    meta = {
        "page_name": page_name,
        "date": None,
        "speaker_code": None,
        "birth_year": None,
        "location": None,
        "session_part": None,
    }
    
    # Дата: первый элемент вида DDMMYY
    if len(parts) > 0 and re.match(r"^\d{6}$", parts[0]):
        d = parts[0]
        meta["date"] = f"20{d[4:6]}-{d[2:4]}-{d[0:2]}"  # → 2022-07-05
    
    # Код информанта: второй элемент, содержит буквы + год рождения
    if len(parts) > 1:
        code = parts[1]
        meta["speaker_code"] = code
        # Год рождения — последние 4 цифры кода (напр. EIV1939 → 1939)
        year_match = re.search(r"(\d{4})$", code)
        if year_match:
            meta["birth_year"] = int(year_match.group(1))
            # Буквенная часть — инициалы
            meta["speaker_initials"] = code[:year_match.start()]
    
    # Населённый пункт: третий элемент
    if len(parts) > 2:
        # Если последний элемент — цифра, это номер части
        if parts[-1].isdigit():
            meta["location"] = "_".join(parts[2:-1])
            meta["session_part"] = int(parts[-1])
        else:
            meta["location"] = "_".join(parts[2:])
    
    return meta


# ---------------------------------------------------------------------------
# Парсинг HTML (минимальные изменения логики, добавлено сохранение audio_id)
# ---------------------------------------------------------------------------

def parse_page(html_path: Path) -> dict:
    html = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    h3 = soup.find("h3")
    page_name = h3.get_text(strip=True) if h3 else html_path.stem
    parts = page_name.split("_")
    interlocutor = parts[1] if len(parts) >= 2 else "speaker"

    # Извлекаем метаданные из имени
    page_meta = parse_page_name(page_name)

    utterances = []
    container = soup.find("div", class_="container") or soup.body or soup

    current_speaker = None
    text_buffer = []

    for node in container.children:
        tag_name = getattr(node, "name", None)

        if tag_name == "b":
            raw = node.get_text()
            if "interviewer" in raw.lower():
                current_speaker = "interviewer"
            else:
                m = re.match(r"([^:]+):", raw.strip())
                current_speaker = m.group(1).strip() if m else interlocutor
            text_buffer = []
            continue

        if tag_name == "audio":
            src = node.get("src", "")
            uid = node.get("id", Path(src).stem if src else "")
            
            # --- ИЗМЕНЕНИЕ: извлекаем временны́е метки из audio_id ---
            start_ms, end_ms = None, None
            time_match = re.search(r"-(\d+)-(\d+)$", uid)
            if time_match:
                start_ms = int(time_match.group(1))
                end_ms   = int(time_match.group(2))
            
            utterances.append({
                "speaker":    current_speaker,
                "text":       " ".join(text_buffer).strip(),
                "audio_id":   uid,
                "audio_src":  src,
                # --- НОВЫЕ ПОЛЯ ---
                "start_ms":   start_ms,
                "end_ms":     end_ms,
                "duration_ms": (end_ms - start_ms) if (start_ms is not None and end_ms is not None) else None,
                # Домен: явная разметка
                "domain": "interviewer" if current_speaker == "interviewer" else "dialect",
            })
            text_buffer = []
            continue

        if tag_name is None:
            t = re.sub(r"[♪]", "", str(node)).strip()
            if t:
                text_buffer.append(t)

    return {
        "page_name":   page_name,
        "interlocutor": interlocutor,
        "page_meta":   page_meta,
        "utterances":  [u for u in utterances if u["speaker"] is not None],
    }


# ---------------------------------------------------------------------------
# Создание структуры + manifest
# ---------------------------------------------------------------------------

def build_audio_url(audio_src: str, page_url: str | None) -> str | None:
    if not audio_src:
        return None
    if audio_src.startswith("http"):
        return audio_src
    if page_url:
        base = page_url.rsplit("/", 1)[0]
        abs_url = base + "/" + audio_src
        parts = abs_url.split("/")
        clean = []
        for p in parts:
            if p == ".." and clean:
                clean.pop()
            elif p != ".":
                clean.append(p)
        return "/".join(clean)
    return None


def create_structure(parsed: dict, output_root: Path,
                     page_url: str | None = None,
                     download: bool = True,
                     delay: float = 0.3) -> list[dict]:
    """
    Возвращает список записей для manifest.csv.
    
    ИЗМЕНЕНИЕ: файлы сохраняются под оригинальным именем (audio_id),
    а не sample1, sample2, ...
    """
    page_name    = parsed["page_name"]
    interlocutor = parsed["interlocutor"]
    page_meta    = parsed["page_meta"]

    manifest_rows = []

    for utt in parsed["utterances"]:
        speaker = utt["speaker"] or interlocutor
        domain  = utt["domain"]
        
        # Путь: corpus_output / dialect|interviewer / speaker_code / audio_id.wav
        # Плоская структура по доменам удобнее для DataLoader
        domain_dir = output_root / domain / speaker
        (domain_dir / "audios").mkdir(parents=True, exist_ok=True)
        (domain_dir / "texts").mkdir(parents=True, exist_ok=True)

        # --- ИЗМЕНЕНИЕ: используем оригинальный audio_id как имя файла ---
        audio_id   = utt["audio_id"] or f"utt_{len(manifest_rows):06d}"
        audio_file = domain_dir / "audios" / f"{audio_id}.wav"
        text_file  = domain_dir / "texts"  / f"{audio_id}.txt"

        # Текст
        text_file.write_text(utt["text"], encoding="utf-8")

        # Аудио
        if download:
            url = build_audio_url(utt["audio_src"], page_url)
            if url:
                try:
                    urllib.request.urlretrieve(url, audio_file)
                    time.sleep(delay)
                except Exception as exc:
                    print(f"  ✗ {url}: {exc}", file=sys.stderr)

        # --- Запись для manifest ---
        manifest_rows.append({
            # Идентификация
            "audio_id":       audio_id,
            "file_path":      str(audio_file.relative_to(output_root)),
            "text_path":      str(text_file.relative_to(output_root)),
            # Домен и спикер
            "domain":         domain,
            "speaker_id":     speaker,
            # Временны́е метки из оригинала
            "start_ms":       utt["start_ms"],
            "end_ms":         utt["end_ms"],
            "duration_ms":    utt["duration_ms"],
            # Метаданные информанта
            "session":        page_name,
            "location":       page_meta.get("location"),
            "birth_year":     page_meta.get("birth_year"),
            "recording_date": page_meta.get("date"),
            "session_part":   page_meta.get("session_part"),
            # Текст реплики (дублируем в manifest для удобства)
            "transcript":     utt["text"],
            # Флаги качества (заполняются позже на этапе скоринга)
            "snr_db":         None,
            "vad_ratio":      None,
            "quality_ok":     None,
        })

    return manifest_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Модифицированный парсер Донского корпуса."
    )
    parser.add_argument("input")
    parser.add_argument("--out", "-o", default="corpus_output")
    parser.add_argument("--url", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_root = Path(args.out)

    html_files = (sorted(input_path.glob("*.html"))
                  if input_path.is_dir() else [input_path])
    if not html_files:
        print("Нет HTML-файлов.", file=sys.stderr)
        sys.exit(1)

    all_manifest_rows = []

    for html_file in html_files:
        print(f"\n→ {html_file.name}")
        parsed = parse_page(html_file)
        rows = create_structure(
            parsed,
            output_root=output_root,
            page_url=args.url,
            download=not args.no_download,
            delay=args.delay,
        )
        all_manifest_rows.extend(rows)
        print(f"  Реплик: {len(rows)}")

    # --- Сохраняем сводный manifest ---
    manifest_path = output_root / "manifest.csv"
    if all_manifest_rows:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_manifest_rows)
        print(f"\n✅ Manifest сохранён: {manifest_path}")
        print(f"   Всего записей: {len(all_manifest_rows)}")
        
        # Статистика по доменам
        from collections import Counter
        domains = Counter(r["domain"] for r in all_manifest_rows)
        for domain, count in domains.items():
            print(f"   {domain}: {count} реплик")


if __name__ == "__main__":
    main()
