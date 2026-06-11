# Данные

В этой папке хранятся **только манифесты** (CSV-файлы с метаданными).  
Аудиофайлы не включены в репозиторий — их нужно получить самостоятельно по инструкции ниже.

---

## Структура папки

```
data/
├── README.md               # этот файл
├── manifests/
│   ├── manifest_raw.csv        # исходный манифест до препроцессинга
│   ├── manifest_24k.csv        # после препроцессинга (22 050 Гц)
│   └── manifest_librispeech.csv  # Russian LibriSpeech (1884 реплики)
└── splits/
    ├── train_list.txt      # для StarGANv2-VC (формат: путь|domain_id)
    ├── val_list.txt
    └── test_list.txt
```

---

## Источник 1: Корпус LingConLab (диалектный материал)

### Что это

Полевые аудиозаписи из двух корпусов [Международной лаборатории языковой конвергенции (НИУ ВШЭ)](https://lingconlab.ru):

| Корпус | Ареал | Наречие | Ссылка |
|---|---|---|---|
| Корпус донских диалектов | Донские говоры | Южное | https://lingconlab.ru/don_rnd/ |
| Корпус Средней Пёзы | Пёзские говоры | Северное | https://lingconlab.ru |

### Как получить

Аудио скачивается скриптом парсинга:

```bash
python ../src/data_collection/parse_lingconlab.py \
    --corpus don \
    --output_dir /path/to/dialect_corpus
```

Результат — папка с аудиофайлами и исходный манифест `manifest_raw.csv`.

### Структура исходного манифеста

| Колонка | Описание |
|---|---|
| `utterance_id` | Уникальный ID реплики |
| `rel_path` | Относительный путь к аудиофайлу |
| `speaker_id` | ID диктора (например, `don_TNG1957`) |
| `dialect_group` | Диалектная группа (`don` / `pyoza` / `interviewer`) |
| `duration_sec` | Длительность в секундах |
| `snr_db` | Отношение сигнал/шум (дБ) |
| `transcript` | Транскрипт реплики |

---

## Источник 2: Russian LibriSpeech (нормативный домен)

### Что это

Аудиокниги, начитанные носителями нормативного русского языка.  
HuggingFace: [istupakov/russian_librispeech](https://huggingface.co/datasets/istupakov/russian_librispeech)

В проекте используются 4 диктора: **8086, 9014, 8169, 295** — по ~450–500 реплик каждый.

### Как получить

```bash
python ../src/data_collection/download_librispeech.py \
    --speakers 8086 9014 8169 295 \
    --max_per_speaker 500 \
    --output_dir /path/to/librispeech
```

Или вручную через HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("istupakov/russian_librispeech", split="train", streaming=True)
```

### Критерии фильтрации

При загрузке применялись следующие фильтры:
- длительность: от 1,0 до 15,0 секунд
- оценка качества (`score`): не ниже −0,5
- не более 500 реплик на диктора

Итого после фильтрации: **1884 реплики**, ~2,5 ч (см. `manifest_librispeech.csv`).

---

## Препроцессинг

После сбора данных запусти препроцессинг:

```bash
python ../src/preprocessing/preprocess_corpus.py \
    --manifest manifests/manifest_raw.csv \
    --corpus_root /path/to/dialect_corpus \
    --out_root /path/to/corpus_24k
```

Пайплайн включает:
1. Ресэмплинг до 22 050 Гц
2. Обрезку тишины (VAD, порог −40 дБ)
3. Нормализацию амплитуды (RMS → −23 дБ)
4. Шумоподавление (`noisereduce`, для SNR < 20 дБ)
5. Фильтрацию: длительность 0,5–15 с, SNR ≥ 5 дБ, без [нрзб]
6. Сегментацию длинных записей (> 15 с) по точкам тишины

Результат сохраняется в `manifest_24k.csv`.

---

## Структура доменов для StarGANv2-VC

| `domain_id` | Описание | Дикторы |
|---|---|---|
| `don_dialect` | Донские говоры | don_TNG1957, don_EIV1939, don_VIK1941, don_GLT1934, don_KVA1948, don_MLI1941, don_SVS1939 |
| `pyoza_dialect` | Говоры Средней Пёзы | pyoza_MAN1910, pyoza_GGS1932, pyoza_GLA1926, pyoza_MGG1932, pyoza_KE1919 |
| `standard` | Нормативная речь | don_interviewer, pyoza_interviewer, librispeech_8086, librispeech_9014, librispeech_8169, librispeech_295 |

Разбивка на сплиты — **90:5:5** (train/val/test) **по диктору**: все реплики одного говорящего попадают в один сплит. Это исключает утечку данных при оценке.

---

## Примечание об авторских правах

Материалы LingConLab используются в некоммерческих исследовательских целях.  
Russian LibriSpeech — открытый корпус, допускающий свободное использование в научных целях.
