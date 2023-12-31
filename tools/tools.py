import numpy as np
import torch
import torch.nn as nn
import pyworld as pw
import parselmouth
import librosa
from fairseq import checkpoint_utils
from torchaudio.transforms import Resample

class F0_Extractor:
    def __init__(self, f0_extractor, sample_rate=44100, hop_size=512, f0_min=65, f0_max=800,
                 block_size=None, model_sampling_rate=None):
        self.block_size = block_size
        self.model_sampling_rate = model_sampling_rate
        self.f0_extractor = f0_extractor
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.transformer_f0 = None
        self.rmvpe = None
        if (self.block_size is not None) or (self.model_sampling_rate is not None):
            assert (self.block_size is not None) and (self.model_sampling_rate is not None)
            self.hop_size_follow_input = True
        else:
            self.hop_size_follow_input = False

    def extract(self, audio, uv_interp=False, device=None, silence_front=0, sr=None):
        if sr is not None:
            assert self.hop_size_follow_input
            self.hop_size = self.block_size * sr / self.model_sampling_rate
            self.sample_rate = sr

        raw_audio = audio
        n_frames = int(len(audio) // self.hop_size) + 1

        start_frame = int(silence_front * self.sample_rate / self.hop_size)
        real_silence_front = start_frame * self.hop_size / self.sample_rate
        audio = audio[int(np.round(real_silence_front * self.sample_rate)):]

        if self.f0_extractor == 'parselmouth':
            f0 = parselmouth.Sound(audio, self.sample_rate).to_pitch_ac(
                time_step=self.hop_size / self.sample_rate,
                voicing_threshold=0.6,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max).selected_array['frequency']
            pad_size = start_frame + (int(len(audio) // self.hop_size) - len(f0) + 1) // 2
            f0 = np.pad(f0, (pad_size, n_frames - len(f0) - pad_size))

        elif self.f0_extractor == 'harvest':
            f0, _ = pw.harvest(
                audio.astype('double'),
                self.sample_rate,
                f0_floor=self.f0_min,
                f0_ceil=self.f0_max,
                frame_period=(1000 * self.hop_size / self.sample_rate))
            f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))

        elif self.f0_extractor == "fcpe":
            _JUMP_SAFE_PAD = False
            if self.transformer_f0 is None:
                from encoder.fcpe.model import FCPEInfer
                self.transformer_f0 = FCPEInfer(model_path='pretrain/fcpe/fcpe.pt')
            if _JUMP_SAFE_PAD:
                raw_audio = audio
            f0 = self.transformer_f0(audio=raw_audio, sr=self.sample_rate)
            f0 = f0.transpose(1, 2)
            if not _JUMP_SAFE_PAD:
                f0 = torch.nn.functional.interpolate(f0, size=int(n_frames), mode='nearest')
            f0 = f0.transpose(1, 2)
            f0 = f0.squeeze().cpu().numpy()
            if _JUMP_SAFE_PAD:
                f0 = np.array(
                    [f0[int(min(int(np.round(n * self.hop_size / self.sample_rate / 0.01)), len(f0) - 1))] for n in
                     range(n_frames - start_frame)])
                f0 = np.pad(f0.astype('float'), (start_frame, n_frames - len(f0) - start_frame))

        elif self.f0_extractor == "rmvpe":
            if self.rmvpe is None:
                from encoder.rmvpe import RMVPE
                self.rmvpe = RMVPE('pretrain/rmvpe/model.pt', hop_length=160)
            f0 = self.rmvpe.infer_from_audio(audio, self.sample_rate, device=device, thred=0.03, use_viterbi=False)
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            origin_time = 0.01 * np.arange(len(f0))
            target_time = self.hop_size / self.sample_rate * np.arange(n_frames - start_frame)
            f0 = np.interp(target_time, origin_time, f0)
            uv = np.interp(target_time, origin_time, uv.astype(float)) > 0.5
            f0[uv] = 0
            f0 = np.pad(f0, (start_frame, 0))
        else:
            raise ValueError(f" [x] Unknown f0 extractor: {self.f0_extractor}")

        if uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min
        return f0


class Volume_Extractor:
    def __init__(self, hop_size=512, block_size=None, model_sampling_rate=None):
        self.block_size = block_size
        self.model_sampling_rate = model_sampling_rate
        self.hop_size = hop_size
        if (self.block_size is not None) or (self.model_sampling_rate is not None):
            assert (self.block_size is not None) and (self.model_sampling_rate is not None)
            self.hop_size_follow_input = True
        else:
            self.hop_size_follow_input = False

    def extract(self, audio, sr=None):
        if sr is not None:
            assert self.hop_size_follow_input
            self.hop_size = self.block_size * sr / self.model_sampling_rate
        n_frames = int(len(audio) // self.hop_size) + 1
        audio2 = audio ** 2
        audio2 = np.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode='reflect')
        volume = np.array(
            [np.mean(audio2[int(n * self.hop_size): int((n + 1) * self.hop_size)]) for n in range(n_frames)])
        volume = np.sqrt(volume)
        return volume

    def get_mask_from_volume(self, volume, threhold=-60.0,device='cpu'):
        mask = (volume > 10 ** (float(threhold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n: n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.block_size).squeeze(-1)
        return mask

class Units_Encoder:
    def __init__(self, encoder, encoder_ckpt, encoder_sample_rate=16000, encoder_hop_size=320, device=None, units_forced_mode='nearest'):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.units_forced_mode = units_forced_mode

        is_loaded_encoder = False
        if encoder == 'contentvec768l12':
            self.model = Audio2ContentVec768L12(encoder_ckpt, device=device)
            is_loaded_encoder = True
        if not is_loaded_encoder:
            raise ValueError(f" [x] Unknown units encoder: {encoder}")
        print(f"[INFO] Units Forced Mode:{self.units_forced_mode}")

        if self.units_forced_mode == 'rfa512to441':
            encoder_sample_rate = encoder_sample_rate * 441 / 512
        if self.units_forced_mode == 'rfa441to512':
            encoder_sample_rate = encoder_sample_rate * 512 / 441

        self.resample_kernel = {}
        self.encoder_sample_rate = encoder_sample_rate
        self.encoder_hop_size = encoder_hop_size

    def encode(self, audio, sample_rate, hop_size, padding_mask=None):
        if self.units_forced_mode not in ('rfa441to512', 'rfa512to441'):
            if sample_rate == self.encoder_sample_rate:
                audio_res = audio
            else:
                key_str = str(sample_rate)
                if key_str not in self.resample_kernel:
                    self.resample_kernel[key_str] = Resample(sample_rate, self.encoder_sample_rate,
                                                             lowpass_filter_width=128).to(self.device)
                audio_res = self.resample_kernel[key_str](audio)
        else:
            if isinstance(audio, np.ndarray):
                _audio = audio
            else:
                _audio = audio.cpu().numpy()
            audio_res = librosa.resample(_audio, orig_sr=sample_rate, target_sr=self.encoder_sample_rate)
            audio_res = torch.from_numpy(audio_res).to(self.device)

        if audio_res.size(-1) < 400:
            audio_res = torch.nn.functional.pad(audio, (0, 400 - audio_res.size(-1)))
        units = self.model(audio_res, padding_mask=padding_mask)

        if self.units_forced_mode == 'left':
            n_frames = audio.size(-1) // hop_size + 1
            ratio = (hop_size / sample_rate) / (self.encoder_hop_size / self.encoder_sample_rate)
            index = torch.clamp(torch.round(ratio * torch.arange(n_frames).to(self.device)).long(), max=units.size(1) - 1)
            units_aligned = torch.gather(units, 1, index.unsqueeze(0).unsqueeze(-1).repeat([1, 1, units.size(-1)]))

        elif self.units_forced_mode == 'nearest':
            n_frames = int(audio.size(-1) // hop_size + 1)
            units = units.transpose(1, 2)
            units_aligned = torch.nn.functional.interpolate(units, size=int(n_frames), mode='nearest')
            units_aligned = units_aligned.transpose(1, 2)

        elif self.units_forced_mode in ('rfa441to512', 'rfa512to441'):
            n_frames = int(audio.size(-1) // hop_size + 1)
            units = units.transpose(1, 2)
            units_aligned = torch.nn.functional.interpolate(units, size=int(n_frames), mode='nearest')
            units_aligned = units_aligned.transpose(1, 2)

        else:
            raise ValueError(f'Unknow units_forced_mode:{self.units_forced_mode}')
        return units_aligned

class Audio2ContentVec768L12():
    def __init__(self, path, device='cpu'):
        self.device = device
        self.models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task([path], suffix="", )
        self.hubert = self.models[0]
        self.hubert = self.hubert.to(self.device)
        self.hubert.eval()

    def __call__(self, audio, padding_mask=None):
        wav_tensor = audio
        feats = wav_tensor.view(1, -1)
        if padding_mask is None:
            padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        else:
            padding_mask = padding_mask.bool()
            padding_mask = ~padding_mask if torch.all(padding_mask) else padding_mask
        inputs = {
            "source": feats.to(wav_tensor.device),
            "padding_mask": padding_mask.to(wav_tensor.device),
            "output_layer": 12,
        }
        with torch.no_grad():
            logits = self.hubert.extract_features(**inputs)
            feats = logits[0]
        units = feats
        return units

class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(torch.cat((signal, signal[:, :, -1:]), 2), size=signal.shape[-1] * factor + 1, mode='linear', align_corners=True)
    signal = signal[:, :, :-1]
    return signal.permute(0, 2, 1)


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx: a.shape[0]] = (1 - k) * a[idx:] + k * b[: fade_len]
    np.copyto(dst=result[a.shape[0]:], src=b[fade_len:])
    return result
