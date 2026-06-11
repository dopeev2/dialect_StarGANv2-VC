"""
batch_inference.py
==================
Пакетный inference StarGANv2-VC для всех 4 сценариев (раздел 3.1.1 ВКР).

Код полностью согласован с INFERENCE_v2.ipynb:
  - sr = 22050 (НЕ 24000)
  - 3 домена: 0=pyoza_dialect, 1=don_dialect, 2=standard
  - параметры мела: n_fft=1024, hop=256, win=1024, n_mels=80
  - нормализация: MEAN=-4, STD=4 (StarGAN) → ремасштаб для HiFi-GAN
  - reference-стиль: style_encoder по случайному файлу целевого домена

Запуск в Google Colab:
    !python batch_inference.py \
        --manifest /content/drive/MyDrive/diploma/manifest_clean_partial.csv \
        --audio-dir /content/drive/MyDrive/diploma/corpus_22k \
        --stargan-ckpt /content/drive/MyDrive/diploma/StarGANv2-VC_v2/Models/dialect/epoch_00128.pth \
        --stargan-cfg /content/StarGANv2-VC/Configs/config.yml \
        --hifigan-ckpt /content/drive/MyDrive/diploma/hifigan_dialect/checkpoints_universal/g_02530000 \
        --hifigan-cfg /content/drive/MyDrive/diploma/hifigan_dialect/config_finetune.json \
        --out-dir /content/drive/MyDrive/diploma/converted \
        --scenarios S1 S2 S3 S4

Структура выхода:
    converted/
      S1/  (don_dialect → standard)
      S2/  (pyoza_dialect → standard)
      S3/  (don_dialect → pyoza_dialect)
      S4/  (standard → don_dialect)
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml
from munch import Munch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Маппинг доменов (из нотбука INFERENCE_v2.ipynb)
# ---------------------------------------------------------------------------

DOMAIN_IDX: Dict[str, int] = {
    "pyoza_dialect": 0,
    "don_dialect":   1,
    "standard":      2,
}
IDX_DOMAIN = {v: k for k, v in DOMAIN_IDX.items()}

# Сценарии конверсии (Таблица 3 ВКР)
SCENARIOS_DEF = {
    "S1": {"src": "don_dialect",   "tgt": "standard",      "label": "Don → Std"},
    "S2": {"src": "pyoza_dialect", "tgt": "standard",      "label": "Pyoza → Std"},
    "S3": {"src": "don_dialect",   "tgt": "pyoza_dialect", "label": "Don → Pyoza"},
    "S4": {"src": "standard",      "tgt": "don_dialect",   "label": "Std → Don"},
}

# Параметры мела (из нотбука)
SR       = 22050
N_FFT    = 1024
HOP      = 256
WIN      = 1024
N_MELS   = 80
F_MIN    = 0
F_MAX    = 8000

# Нормализация StarGAN (из нотбука, ячейка 7)
MEAN = -4
STD  = 4

# Точные статистики корпуса (из нотбука)
STARGAN_MEAN = -2.5696
STARGAN_STD  =  4.1923
HIFI_MEAN    = -4.9008
HIFI_STD     =  2.1360


# ---------------------------------------------------------------------------
# Загрузка моделей (воспроизводит ячейки 4–6 нотбука)
# ---------------------------------------------------------------------------

def setup_paths(stargan_repo: str = "/content/StarGANv2-VC",
                hifigan_repo: str = "/content/hifi-gan"):
    """Настраивает sys.path как в нотбуке."""
    # Сначала чистим
    sys.path = [p for p in sys.path
                if "hifi-gan" not in p and "StarGANv2-VC" not in p]
    for mod in list(sys.modules.keys()):
        if mod in ["models", "meldataset", "env"]:
            del sys.modules[mod]

    sys.path.insert(0, stargan_repo)
    sys.path.insert(1, hifigan_repo)
    os.chdir(stargan_repo)


def load_models(
    stargan_ckpt: Path,
    stargan_cfg:  Path,
    hifigan_ckpt: Path,
    hifigan_cfg:  Path,
    stargan_repo: str = "/content/StarGANv2-VC",
    hifigan_repo: str = "/content/hifi-gan",
    device: str = "cuda",
):
    """
    Загружает все модели в том же порядке, что ячейка 6 нотбука.
    Returns: (generator, style_encoder, mapping_network, F0_model, hifigan, mp, h)
    """
    setup_paths(stargan_repo, hifigan_repo)

    # --- StarGAN импорты ---
    from models import Generator, MappingNetwork, StyleEncoder  # type: ignore
    from Utils.JDC.model import JDCNet                          # type: ignore

    # --- HiFi-GAN импорты (через манипуляцию sys.modules) ---
    for mod in list(sys.modules.keys()):
        if mod in ["models", "meldataset"]:
            del sys.modules[mod]
    sys.path.insert(0, hifigan_repo)
    from env import AttrDict                            # type: ignore
    from models import Generator as HifiGenerator      # type: ignore
    # Восстанавливаем StarGAN models
    for mod in list(sys.modules.keys()):
        if mod == "models":
            del sys.modules[mod]
    sys.path.insert(0, stargan_repo)
    from models import Generator, MappingNetwork, StyleEncoder  # type: ignore (noqa F811)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("Устройство: %s", dev)

    # --- StarGAN конфиг ---
    with open(stargan_cfg) as f:
        cfg = yaml.safe_load(f)
    mp = cfg["model_params"]
    logger.info("StarGAN: num_domains=%s, dim_in=%s, style_dim=%s",
                mp["num_domains"], mp["dim_in"], mp["style_dim"])

    # --- Загружаем StarGAN чекпоинт ---
    state = torch.load(str(stargan_ckpt), map_location=dev, weights_only=False)
    ema = state["model_ema"]

    generator = Generator(
        dim_in=mp["dim_in"], style_dim=mp["style_dim"],
        max_conv_dim=mp["max_conv_dim"], w_hpf=mp["w_hpf"],
        F0_channel=mp["F0_channel"],
    ).to(dev)
    generator.load_state_dict(ema["generator"])
    generator.eval()

    style_enc = StyleEncoder(
        dim_in=mp["dim_in"], style_dim=mp["style_dim"],
        num_domains=mp["num_domains"], max_conv_dim=mp["max_conv_dim"],
    ).to(dev)
    style_enc.load_state_dict(ema["style_encoder"])
    style_enc.eval()

    mapping_net = MappingNetwork(
        latent_dim=mp["latent_dim"], style_dim=mp["style_dim"],
        num_domains=mp["num_domains"], hidden_dim=mp["max_conv_dim"],
    ).to(dev)
    mapping_net.load_state_dict(ema["mapping_network"])
    mapping_net.eval()

    F0_model = JDCNet(num_class=1, seq_len=192)
    f0_params = torch.load(
        os.path.join(stargan_repo, "Utils/JDC/bst.t7"),
        weights_only=False,
    )["net"]
    F0_model.load_state_dict(f0_params)
    F0_model.eval().to(dev)

    # --- HiFi-GAN ---
    with open(hifigan_cfg) as f:
        h = AttrDict(json.load(f))
    hifigan = HifiGenerator(h).to(dev)
    hifi_state = torch.load(str(hifigan_ckpt), map_location=dev, weights_only=False)
    hifigan.load_state_dict(hifi_state["generator"])
    hifigan.eval()

    logger.info("Все модели загружены успешно")
    return generator, style_enc, mapping_net, F0_model, hifigan, mp, dev


# ---------------------------------------------------------------------------
# Мел-спектрограмма и конверсия (воспроизводит ячейку 7 нотбука)
# ---------------------------------------------------------------------------

class Converter:
    """
    Инкапсулирует логику compute_mel + convert из нотбука.
    Полностью воспроизводит ячейку 7.
    """

    def __init__(self, generator, style_enc, mapping_net, F0_model,
                 hifigan, mp, device):
        self.generator    = generator
        self.style_enc    = style_enc
        self.mapping_net  = mapping_net
        self.F0_model     = F0_model
        self.hifigan      = hifigan
        self.mp           = mp
        self.device       = device

        self.to_melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            win_length=WIN,
            hop_length=HOP,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
        ).to(device)

    def compute_mel(self, wav_path: str) -> torch.Tensor:
        """Ячейка 7 нотбука: compute_mel."""
        wav, sr = sf.read(wav_path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        # Ресемплинг если нужно
        if sr != SR:
            wav_t = torch.FloatTensor(wav).unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, sr, SR)
            wav = wav_t.squeeze().numpy()
        wav_tensor = torch.FloatTensor(wav).to(self.device)
        mel = self.to_melspec(wav_tensor)                     # [80, T]
        mel = (torch.log(1e-5 + mel) - MEAN) / STD           # нормализация
        return mel.unsqueeze(0)                               # [1, 80, T]

    @torch.no_grad()
    def convert(
        self,
        input_wav:    str,
        target_domain: int,
        output_wav:   str,
        reference_wav: Optional[str] = None,
    ) -> np.ndarray:
        """Ячейка 7 нотбука: convert."""
        mel    = self.compute_mel(input_wav).to(self.device)
        source = mel.unsqueeze(1)                             # [1,1,80,T]

        label = torch.LongTensor([target_domain]).to(self.device)

        if reference_wav is not None:
            ref_mel = self.compute_mel(reference_wav).to(self.device)
            ref = self.style_enc(ref_mel.unsqueeze(1), label)
        else:
            latent = torch.randn(1, self.mp["latent_dim"]).to(self.device)
            ref = self.mapping_net(latent, label)

        f0_feat   = self.F0_model.get_feature_GAN(source)
        converted = self.generator(source, ref, F0=f0_feat)

        # Денормализация StarGAN → ремасштаб HiFi-GAN
        mel_denorm   = converted.squeeze(1) * STD + MEAN
        mel_rescaled = (mel_denorm - STARGAN_MEAN) / STARGAN_STD * HIFI_STD + HIFI_MEAN

        audio = self.hifigan(mel_rescaled).squeeze().cpu().numpy()
        sf.write(output_wav, audio, SR)
        return audio


# ---------------------------------------------------------------------------
# Батч-конверсия для одного сценария
# ---------------------------------------------------------------------------

def run_scenario_inference(
    scenario_id:   str,
    test_df:       pd.DataFrame,
    audio_dir:     Path,
    out_dir:       Path,
    converter:     "Converter",
    n_references:  int = 5,
    resume:        bool = True,
    seed:          int = 42,
) -> pd.DataFrame:
    """
    Конвертирует все тестовые файлы для одного сценария.

    Reference-стиль: для каждого source-файла берётся случайный файл
    из тестового подмножества целевого домена (как в ячейке 8 нотбука).

    Parameters
    ----------
    resume : если True, пропускает уже сконвертированные файлы
    """
    rng = random.Random(seed)

    cfg = SCENARIOS_DEF[scenario_id]
    src_group = cfg["src"]
    tgt_group = cfg["tgt"]
    tgt_idx   = DOMAIN_IDX[tgt_group]

    src_files = test_df[test_df["domain_id"] == src_group].copy()
    tgt_files = test_df[test_df["domain_id"] == tgt_group].copy()

    logger.info(
        "\n%s\nСценарий %s: %s → %s (domain_idx=%d)\n"
        "Исходных файлов: %d, целевых (референс): %d\n%s",
        "="*55, scenario_id, src_group, tgt_group, tgt_idx,
        len(src_files), len(tgt_files), "="*55,
    )

    # Список референс-файлов целевого домена
    ref_pool: List[Path] = [
        audio_dir / row["rel_path"]
        for _, row in tgt_files.iterrows()
        if (audio_dir / row["rel_path"]).exists()
    ]
    if not ref_pool:
        logger.error("Нет референс-файлов для %s в %s", tgt_group, audio_dir)
        return pd.DataFrame()

    scenario_out = out_dir / scenario_id
    scenario_out.mkdir(parents=True, exist_ok=True)

    log_rows = []
    errors   = 0
    t_start  = time.time()

    for i, (_, row) in enumerate(src_files.iterrows()):
        src_path  = audio_dir / row["rel_path"]
        out_name  = Path(row["rel_path"]).stem + ".wav"
        out_path  = scenario_out / out_name

        # Resume: пропускаем уже готовые
        if resume and out_path.exists():
            log_rows.append({
                "utterance_id": row["utterance_id"],
                "speaker_id":   row["speaker_id"],
                "domain_id":    src_group,
                "target_domain": tgt_group,
                "scenario_id":  scenario_id,
                "input_path":   str(src_path),
                "output_path":  str(out_path),
                "reference_path": "",
                "status":       "skipped_exists",
                "duration_sec": row["duration_sec"],
            })
            continue

        if not src_path.exists():
            logger.warning("Файл не найден: %s", src_path)
            errors += 1
            continue

        # Выбираем случайный референс
        ref_path = rng.choice(ref_pool)

        try:
            converter.convert(
                input_wav=str(src_path),
                target_domain=tgt_idx,
                output_wav=str(out_path),
                reference_wav=str(ref_path),
            )
            status = "ok"
        except Exception as e:
            logger.warning("Ошибка конверсии %s: %s", src_path.name, e)
            status = f"error: {e}"
            errors += 1

        log_rows.append({
            "utterance_id":   row["utterance_id"],
            "speaker_id":     row["speaker_id"],
            "domain_id":      src_group,
            "target_domain":  tgt_group,
            "scenario_id":    scenario_id,
            "input_path":     str(src_path),
            "output_path":    str(out_path),
            "reference_path": str(ref_path),
            "status":         status,
            "duration_sec":   row["duration_sec"],
        })

        # Прогресс каждые 25 файлов
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta  = (len(src_files) - i - 1) / max(rate, 1e-6)
            logger.info(
                "  %d/%d файлов | %.1f файл/с | ETA %.0f с | ошибок: %d",
                i + 1, len(src_files), rate, eta, errors,
            )

    log_df = pd.DataFrame(log_rows)
    log_path = scenario_out / "inference_log.csv"
    log_df.to_csv(log_path, index=False)

    ok_count = (log_df["status"] == "ok").sum()
    skip_count = (log_df["status"] == "skipped_exists").sum()
    logger.info(
        "Сценарий %s завершён: %d OK, %d пропущено, %d ошибок. Лог: %s",
        scenario_id, ok_count, skip_count, errors, log_path,
    )
    return log_df


# ---------------------------------------------------------------------------
# Baseline G2: Self-conversion (Std → Std)
# ---------------------------------------------------------------------------

def run_self_conversion(
    test_df:   pd.DataFrame,
    audio_dir: Path,
    out_dir:   Path,
    converter: "Converter",
    seed:      int = 42,
) -> pd.DataFrame:
    """
    G2: конвертирует нормативные файлы сами в себя (Std → Std).
    Тест стабильности генератора.
    """
    rng = random.Random(seed)
    std_files = test_df[test_df["domain_id"] == "standard"].copy()

    out_path_g2 = out_dir / "G2_self_conversion"
    out_path_g2.mkdir(parents=True, exist_ok=True)

    logger.info("G2: self-conversion (%d файлов)...", len(std_files))
    rows = []

    std_pool = [
        audio_dir / r["rel_path"]
        for _, r in std_files.iterrows()
        if (audio_dir / r["rel_path"]).exists()
    ]

    for _, row in std_files.iterrows():
        src_path = audio_dir / row["rel_path"]
        out_name = Path(row["rel_path"]).stem + ".wav"
        out_path = out_path_g2 / out_name

        if not src_path.exists():
            continue

        # Референс — другой нормативный файл (не сам себя)
        others = [p for p in std_pool if p != src_path]
        ref_path = rng.choice(others) if others else src_path

        try:
            converter.convert(
                input_wav=str(src_path),
                target_domain=DOMAIN_IDX["standard"],
                output_wav=str(out_path),
                reference_wav=str(ref_path),
            )
            status = "ok"
        except Exception as e:
            status = f"error: {e}"

        rows.append({
            "utterance_id":  row["utterance_id"],
            "input_path":    str(src_path),
            "output_path":   str(out_path),
            "reference_path": str(ref_path),
            "status": status,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path_g2 / "g2_log.csv", index=False)
    ok = (df["status"] == "ok").sum()
    logger.info("G2 завершён: %d/%d файлов OK", ok, len(df))
    return df


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Батч-inference для всех сценариев (раздел 3.1.1 ВКР)"
    )
    parser.add_argument("--manifest",     type=Path, required=True,
                        help="manifest_clean_partial.csv")
    parser.add_argument("--audio-dir",    type=Path, required=True,
                        help="Директория с wav (corpus_22k/)")
    parser.add_argument("--stargan-ckpt", type=Path, required=True,
                        help="epoch_XXXXX.pth")
    parser.add_argument("--stargan-cfg",  type=Path, required=True,
                        help="Configs/config.yml")
    parser.add_argument("--hifigan-ckpt", type=Path, required=True,
                        help="g_XXXXXXXX")
    parser.add_argument("--hifigan-cfg",  type=Path, required=True,
                        help="config_finetune.json")
    parser.add_argument("--stargan-repo", type=str,
                        default="/content/StarGANv2-VC")
    parser.add_argument("--hifigan-repo", type=str,
                        default="/content/hifi-gan")
    parser.add_argument("--out-dir",      type=Path, required=True,
                        help="Директория для конвертированных файлов")
    parser.add_argument("--scenarios",    nargs="+",
                        default=["S1", "S2", "S3", "S4"],
                        choices=["S1", "S2", "S3", "S4"])
    parser.add_argument("--self-conv",    action="store_true",
                        help="Запустить G2 self-conversion")
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--no-resume",    action="store_true",
                        help="Пересчитать все файлы, игнорируя существующие")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    # --- Загрузка манифеста ---
    df = pd.read_csv(args.manifest)
    test_df = df[df["split"] == "test"].copy()
    logger.info("Тестовая выборка: %d файлов", len(test_df))

    for dom in ["don_dialect", "pyoza_dialect", "standard"]:
        n = len(test_df[test_df["domain_id"] == dom])
        logger.info("  %s: %d файлов", dom, n)

    # --- Загрузка моделей ---
    logger.info("Загрузка моделей...")
    generator, style_enc, mapping_net, F0_model, hifigan, mp, device = load_models(
        stargan_ckpt=args.stargan_ckpt,
        stargan_cfg=args.stargan_cfg,
        hifigan_ckpt=args.hifigan_ckpt,
        hifigan_cfg=args.hifigan_cfg,
        stargan_repo=args.stargan_repo,
        hifigan_repo=args.hifigan_repo,
        device=args.device,
    )

    converter = Converter(
        generator, style_enc, mapping_net, F0_model,
        hifigan, mp, device,
    )

    # --- Запуск сценариев ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_logs: Dict[str, pd.DataFrame] = {}

    for sid in args.scenarios:
        log_df = run_scenario_inference(
            scenario_id=sid,
            test_df=test_df,
            audio_dir=args.audio_dir,
            out_dir=args.out_dir,
            converter=converter,
            resume=not args.no_resume,
            seed=args.seed,
        )
        all_logs[sid] = log_df

    # --- G2 self-conversion ---
    if args.self_conv:
        g2_df = run_self_conversion(
            test_df=test_df,
            audio_dir=args.audio_dir,
            out_dir=args.out_dir,
            converter=converter,
            seed=args.seed,
        )
        all_logs["G2"] = g2_df

    # --- Итоговая статистика ---
    print("\n" + "="*55)
    print("ИТОГ INFERENCE")
    print("="*55)
    for sid, log_df in all_logs.items():
        if log_df.empty:
            print(f"  {sid}: нет данных")
            continue
        ok   = (log_df["status"] == "ok").sum()
        skip = (log_df["status"] == "skipped_exists").sum()
        err  = len(log_df) - ok - skip
        print(f"  {sid}: {ok} OK | {skip} пропущено | {err} ошибок")
    print(f"\nВыходная директория: {args.out_dir}")


if __name__ == "__main__":
    main()
