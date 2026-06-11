# Адаптация StarGANv2-VC для диалектно-нормативного преобразования речи

> Выпускная квалификационная работа бакалавра  
> Трунов Илья Олегович, ЮФУ, 2026  
> Направление: 45.03.04 — Интеллектуальные системы в гуманитарной сфере  
> Научный руководитель: проф. д.ф.н. Северина Е. М.

---

## О проекте

Система непараллельного преобразования диалектной русской речи в нормативную на основе архитектуры **StarGANv2-VC**. Модель обучена на двух контрастных диалектных ареалах — донских говорах (южное наречие) и говорах Средней Пёзы (северное наречие) — и нормативном домене, расширенном материалами Russian LibriSpeech.

Ключевое методологическое решение: **домен переопределяется от диктора к речевой разновидности**, что позволяет использовать нейросетевую конверсию не только как инженерный инструмент, но и как аналитический — для изучения фонетических различий между диалектной и нормативной речью.

---

## Структура репозитория

```
dialect_StarGANv2-VC/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md               # Инструкция по получению данных
│   └── manifests/
│       ├── manifest_raw.csv    # Исходный манифест корпуса LingConLab
│       ├── manifest_24k.csv    # После препроцессинга (22 050 Гц)
│       └── manifest_librispeech.csv  # Russian LibriSpeech (1884 реплики)
│
├── src/
│   ├── data_collection/        # Сбор данных
│   │   ├── parse_lingconlab.py # Парсинг аудио из LingConLab
│   │   └── download_librispeech.py
│   │
│   ├── analysis/               # Анализ корпуса
│   │   ├── corpus_stats.py     # Статистики: длительность, SNR, дикторы
│   │   └── spectral_analysis.py
│   │
│   ├── preprocessing/          # Препроцессинг
│   │   └── preprocess_corpus.py
│   │
│   └── inference/              # Инференс и оценка
│       ├── convert.py
│       └── evaluate.py
│
├── src/models/
│   └── StarGANv2-VC/           # Адаптированная архитектура модели
│
├── notebooks/                  # Colab-ноутбуки
│   ├── 01_corpus_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_stargan_training.ipynb
│   └── 04_evaluation.ipynb
│
├── configs/
│   ├── preprocess_config.yaml
│   └── stargan_config.yaml
│
└── docs/
    └── corpus_report.txt
```

---

## Данные

### Диалектный корпус (LingConLab)

| Ареал | Наречие | Дикторов | Реплик | Длительность |
|---|---|---|---|---|
| Донские говоры | Южное | 7 | ~7 000 | ~5,5 ч |
| Средняя Пёза | Северное | 5 | ~4 000 | ~3,2 ч |
| Интервьюеры | Нормативный | 2 | ~800 | ~2,3 ч |

Источники: [Корпус донских диалектов](https://lingconlab.ru/don_rnd/) и [корпус Средней Пёзы](https://lingconlab.ru) (LingConLab, НИУ ВШЭ).

### Нормативный корпус (Russian LibriSpeech)

4 диктора (8086, 9014, 8169, 295), ~1884 реплики, ~2,5 ч.  
Источник: [istupakov/russian_librispeech](https://huggingface.co/datasets/istupakov/russian_librispeech) на HuggingFace.

> **Аудиофайлы не хранятся в репозитории.** Инструкция по получению данных — в [`data/README.md`](data/README.md).

---

## Домены модели

Модель обучена в режиме **many-to-many** с тремя доменами:

| Домен | Описание |
|---|---|
| `don_dialect` | Донские говоры (южное наречие) |
| `pyoza_dialect` | Говоры Средней Пёзы (северное наречие) |
| `standard` | Нормативная речь (интервьюеры + LibriSpeech) |

---

## Архитектура и адаптации

На основе [StarGANv2-VC (Li et al., 2021)](https://arxiv.org/abs/2107.10394) с двумя ключевыми изменениями:

1. **Входное представление**: мел-спектрограмма (torchaudio) вместо MCC (вокодер WORLD), параметры согласованы с HiFi-GAN (`n_fft=1024`, `hop=256`, `n_mels=80`, `sr=22050`)
2. **Функция потерь**: повышен вес ASR-потери (`λ_asr=10.0`) для сохранения лингвистического содержания при изменении произносительного стиля

Вокодер **HiFi-GAN** дообучен на материале корпуса (50 000 шагов).

---

## Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Препроцессинг

```bash
python src/preprocessing/preprocess_corpus.py \
    --manifest data/manifests/manifest_raw.csv \
    --corpus_root /path/to/audio \
    --out_root /path/to/output
```

### Инференс (конвертация)

```bash
python src/inference/convert.py \
    --input audio.wav \
    --source_domain don_dialect \
    --target_domain standard \
    --checkpoint /path/to/checkpoint.pt
```

### Обучение (Google Colab)

Открыть [`notebooks/03_stargan_training.ipynb`](notebooks/03_stargan_training.ipynb) в Colab и следовать инструкциям внутри ноутбука.

---

## Результаты

Оценка на тестовой выборке (712 файлов × 3 домена):

| Сценарий | F0 RMSE (Гц) | F0 PCC | ΔFormant |
|---|---|---|---|
| Don → Std (С1) | 143,1 ± 51,5 | 0,265 ± 0,184 | 0,661 ± 0,352 |
| Pyoza → Std (С2) | 141,7 ± 38,7 | 0,321 ± 0,195 | 0,714 ± 0,456 |
| Don → Pyoza (С3) | 155,8 ± 40,7 | 0,238 ± 0,186 | 0,653 ± 0,381 |
| Std → Don (С4) | 152,5 ± 36,9 | 0,050 ± 0,172 | 0,672 ± 0,457 |

Сценарий С1 подтверждает принципиальную реализуемость задачи: погрешность просодического сдвига — 2,6 Гц (6,7% от ожидаемого) при непараллельном обучении.

---

## Чекпоинты

Финальные чекпоинты размещены на Google Drive:  
🔗 *(ссылка добавляется после публикации)*

---

## Ссылки

- [StarGANv2-VC (оригинальный репозиторий)](https://github.com/yl4579/StarGANv2-VC)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [Корпус донских диалектов (LingConLab)](https://lingconlab.ru/don_rnd/)
- Li, Y. A. et al. StarGANv2-VC: A Diverse, Unsupervised, Non-parallel Framework for Natural-Sounding Voice Conversion. *Interspeech 2021*.

---

## Лицензия

Код проекта: MIT. Данные LingConLab используются в некоммерческих исследовательских целях согласно условиям корпусов.
