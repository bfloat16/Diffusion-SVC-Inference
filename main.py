import torch
import librosa
import argparse
import soundfile as sf
from tools.infer_tools import DiffusionSVC


def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model",
        "--model",
        type=str,
        required=True,
        help="path to the diffusion model checkpoint",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        required=False,
        help="cpu or cuda, auto if not set")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="path to the input audio file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path to the output audio file",
    )
    parser.add_argument(
        "-ref",
        "--refer_audio",
        type=str,
        required=True,
        default=1,
        help="refer audio path",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        required=False,
        default=0,
        help="key changed (number of semitones) | default: 0",
    )
    parser.add_argument(
        "-f",
        "--formant_shift_key",
        type=str,
        required=False,
        default=0,
        help="formant changed (number of semitones) , only for pitch-augmented model| default: 0",
    )
    parser.add_argument(
        "-pe",
        "--pitch_extractor",
        type=str,
        required=False,
        default='crepe',
        help="pitch extrator type: parselmouth, dio, harvest, crepe (default) or rmvpe",
    )
    parser.add_argument(
        "-fmin",
        "--f0_min",
        type=str,
        required=False,
        default=50,
        help="min f0 (Hz) | default: 50",
    )
    parser.add_argument(
        "-fmax",
        "--f0_max",
        type=str,
        required=False,
        default=1100,
        help="max f0 (Hz) | default: 1100",
    )
    parser.add_argument(
        "-th",
        "--threhold",
        type=str,
        required=False,
        default=-60,
        help="response threhold (dB) | default: -60",
    )
    parser.add_argument(
        "-th4sli",
        "--threhold_for_split",
        type=str,
        required=False,
        default=-40,
        help="threhold for split (dB) | default: -40",
    )
    parser.add_argument(
        "-min_len",
        "--min_len",
        type=str,
        required=False,
        default=5000,
        help="min split len | default: 5000",
    )
    parser.add_argument(
        "-speedup",
        "--speedup",
        type=str,
        required=False,
        default=10,
        help="speed up | default: 10",
    )
    parser.add_argument(
        "-method",
        "--method",
        type=str,
        required=False,
        default='dpm-solver',
        help="ddim, pndm, dpm-solver or unipc | default: dpm-solver",
    )
    return parser.parse_args(args=args, namespace=namespace)

if __name__ == '__main__':
    cmd = parse_args()

    device = cmd.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    diffusion_svc = DiffusionSVC(device=device)
    diffusion_svc.load_model(model_path=cmd.model, f0_model=cmd.pitch_extractor, f0_max=cmd.f0_max, f0_min=cmd.f0_min)

    in_wav, in_sr = librosa.load(cmd.input, sr=None)
    if len(in_wav.shape) > 1:
        in_wav = librosa.to_mono(in_wav)

    in_refer, in_rsr = librosa.load(cmd.refer_audio, sr=None)
    if len(in_wav.shape) > 1:
        in_wav = librosa.to_mono(in_wav)

    out_wav, out_sr = diffusion_svc.infer_from_long_audio(
        in_wav, sr=(in_sr,in_rsr),
        key=float(cmd.key),
        refer_audio=str(cmd.refer_audio),
        aug_shift=int(cmd.formant_shift_key),
        infer_speedup=int(cmd.speedup),
        method=cmd.method,
        use_tqdm=True,
        threhold=float(cmd.threhold),
        threhold_for_split=float(cmd.threhold_for_split),
        min_len=int(cmd.min_len),
    )
    
    sf.write(cmd.output, out_wav, out_sr)
