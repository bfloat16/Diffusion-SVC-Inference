import numpy as np
import torch
import torch.nn.functional
import torchaudio
from diffusion.unit2mel import load_model_vocoder
from tools.slicer import split
from tools.tools import F0_Extractor, Volume_Extractor, Units_Encoder, cross_fade
import tqdm


class DiffusionSVC:
    def __init__(self, device=None):
        if device is not None:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = None
        self.model = None
        self.vocoder = None
        self.args = None
        self.units_encoder = None
        self.f0_extractor = None
        self.f0_model = None
        self.f0_min = None
        self.f0_max = None
        self.volume_extractor = None
        self.resample_dict_16000 = {}
        self.use_combo_model = False

    def load_model(self, model_path, f0_model=None, f0_min=None, f0_max=None):

        self.model_path = model_path
        self.model, self.vocoder, self.args = load_model_vocoder(model_path, device=self.device)

        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            self.args.data.encoder_ckpt,
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size,
            device=self.device,
            units_forced_mode=self.args.data.units_forced_mode
        )

        self.volume_extractor = Volume_Extractor(
            hop_size=512,
            block_size=self.args.data.block_size,
            model_sampling_rate=self.args.data.sampling_rate
        )

        self.load_f0_extractor(f0_model=f0_model, f0_min=f0_min, f0_max=f0_max)

    def flush(self, model_path=None, f0_model=None, f0_min=None, f0_max=None):
        assert model_path is not None
        if ((self.model_path != model_path) or (self.f0_model != f0_model) or (self.f0_min != f0_min) or (self.f0_max != f0_max)):
            self.load_model(model_path, f0_model=f0_model, f0_min=f0_min, f0_max=f0_max)

    def flush_f0_extractor(self, f0_model):
        if (f0_model != self.f0_model) and (f0_model is not None):
            self.load_f0_extractor(f0_model)

    def load_f0_extractor(self, f0_model, f0_min=None, f0_max=None):
        self.f0_model = f0_model if (f0_model is not None) else self.args.data.f0_extractor
        self.f0_min = f0_min if (f0_min is not None) else self.args.data.f0_min
        self.f0_max = f0_max if (f0_max is not None) else self.args.data.f0_max
        self.f0_model = f0_model
        self.f0_extractor = F0_Extractor(
            f0_extractor=self.f0_model,
            sample_rate=44100,
            hop_size=512,
            f0_min=self.f0_min,
            f0_max=self.f0_max,
            block_size=self.args.data.block_size,
            model_sampling_rate=self.args.data.sampling_rate
        )

    @torch.no_grad()
    def encode_units(self, audio, sr=44100, padding_mask=None):
        assert self.units_encoder is not None
        hop_size = self.args.data.block_size * sr / self.args.data.sampling_rate
        return self.units_encoder.encode(audio, sr, hop_size, padding_mask=padding_mask)

    @torch.no_grad()
    def extract_f0(self, audio, key=0, sr=44100, silence_front=0):
        assert self.f0_extractor is not None
        f0 = self.f0_extractor.extract(audio, uv_interp=True, device=self.device, silence_front=silence_front, sr=sr)
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(key) / 12)
        return f0

    @torch.no_grad()
    def extract_volume_and_mask(self, audio, sr=44100, threhold=-60.0):
        assert self.volume_extractor is not None
        volume = self.volume_extractor.extract(audio, sr)
        mask = self.volume_extractor.get_mask_from_volume(volume, threhold=threhold, device=self.device)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        return volume, mask

    @torch.no_grad()
    def extract_mel(self, audio, sr=44100):
        assert sr == 441000
        mel = self.vocoder.extract(audio, self.args.data.sampling_rate)
        return mel

    @torch.no_grad()
    def encode_spk(self, audio, sr=44100):
        assert self.speaker_encoder is not None
        return self.speaker_encoder(audio=audio, sample_rate=sr)

    @torch.no_grad()
    def mel2wav(self, mel, f0, start_frame=0):
        if start_frame == 0:
            return self.vocoder.infer(mel, f0)
        else:  # for realtime speedup
            mel = mel[:, start_frame:, :]
            f0 = f0[:, start_frame:, :]
            out_wav = self.vocoder.infer(mel, f0)
            return torch.nn.functional.pad(out_wav, (start_frame * self.vocoder.vocoder_hop_size, 0))

    @torch.no_grad()
    def __call__(self, units, f0, volume, refer_spec=None, aug_shift=0,
                 gt_spec=None, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True):

        aug_shift = torch.from_numpy(np.array([[float(aug_shift)]])).float().to(self.device)
        return self.model(units, f0, volume, refer_spec, aug_shift=aug_shift,
                          gt_spec=gt_spec, infer=True, infer_speedup=infer_speedup, method=method, k_step=k_step,
                          use_tqdm=use_tqdm)

    @torch.no_grad()
    def infer(self, units, f0, volume, gt_spec=None, refer_spec = None, aug_shift=0,
              infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True):
        if k_step is not None:
            assert gt_spec is not None
            k_step = int(k_step)
            gt_spec = gt_spec
        else:
            gt_spec = None

        out_mel = self.__call__(units, f0, volume, refer_spec=refer_spec, aug_shift=aug_shift, gt_spec=gt_spec, infer_speedup=infer_speedup, method=method, k_step=k_step,
                                use_tqdm=use_tqdm)
        return self.mel2wav(out_mel, f0)

    @torch.no_grad()
    def infer_from_long_audio(self, audio, sr=(44100, 44100), key=0, refer_audio=None, aug_shift=0, infer_speedup=10, method='dpm-solver',
                              k_step=None, use_tqdm=True, threhold=-60, threhold_for_split=-40, min_len=5000):
        in_sr,in_rsr = sr
        hop_size = self.args.data.block_size * in_sr / self.args.data.sampling_rate
        segments = split(audio, in_sr, hop_size, db_thresh=threhold_for_split, min_len=min_len)

        f0 = self.extract_f0(audio, key=key, sr=in_sr)
        volume, mask = self.extract_volume_and_mask(audio, in_sr, threhold=float(threhold))

        refer_audio = torchaudio.load(refer_audio)[0].float().to(self.device)

        if refer_audio.shape[0] == 2:
            refer_audio = torch.mean(refer_audio, dim=0, keepdim=True)

        refer_spec = self.vocoder.extract(refer_audio, in_rsr)
        if k_step is not None:
            assert 0 < int(k_step) <= 1000
            k_step = int(k_step)
            audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            gt_spec = self.vocoder.extract(audio_t, in_sr)
            gt_spec = torch.cat((gt_spec, gt_spec[:, -1:, :]), 1)
        else:
            gt_spec = None

        result = np.zeros(0)
        current_length = 0
        for segment in tqdm.tqdm(segments, desc="Processing Audio"):
            start_frame = segment[0]
            seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(self.device)
            seg_units = self.units_encoder.encode(seg_input, in_sr, hop_size)
            seg_f0 = f0[:, start_frame: start_frame + seg_units.size(1), :]
            seg_volume = volume[:, start_frame: start_frame + seg_units.size(1), :]
            if gt_spec is not None:
                seg_gt_spec = gt_spec[:, start_frame: start_frame + seg_units.size(1), :]
            else:
                seg_gt_spec = None
            seg_output = self.infer(seg_units, seg_f0, seg_volume, gt_spec=seg_gt_spec, refer_spec = refer_spec, aug_shift=aug_shift, infer_speedup=infer_speedup, method=method, k_step=k_step,
                                    use_tqdm=use_tqdm)
            _left = start_frame * self.args.data.block_size
            _right = (start_frame + seg_units.size(1)) * self.args.data.block_size
            seg_output *= mask[:, _left:_right]
            seg_output = seg_output.squeeze().cpu().numpy()
            silent_length = round(start_frame * self.args.data.block_size) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_output)
            else:
                result = cross_fade(result, seg_output, current_length + silent_length)
            current_length = current_length + silent_length + len(seg_output)

        return result, self.args.data.sampling_rate
