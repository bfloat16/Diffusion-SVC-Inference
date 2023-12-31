import os
import yaml
import torch
import torch.nn as nn
from .diffusion import GaussianDiffusion
from .vocoder import Vocoder
from .unet1d.unet_1d_condition import UNet1DConditionModel
from .mrte_model import MRTE

class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_model_vocoder(model_path, device='cpu', loaded_vocoder=None):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)

    if loaded_vocoder is None:
        vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    else:
        vocoder = loaded_vocoder

    model = load_svc_model(args=args, vocoder_dimension=vocoder.dimension)

    print('[Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, vocoder, args

def load_svc_model(args, vocoder_dimension):
    if args.model.type == 'Diffusion':
        model = Unit2Mel(
                    args.data.encoder_out_channels, 
                    args.model.use_pitch_aug,
                    vocoder_dimension,
                    args.model.n_layers,
                    args.model.block_out_channels,
                    args.model.n_heads,
                    args.model.n_hidden,
                    mrte_layer=args.model.mrte_layer,
                    mrte_hident_size=args.model.mrte_hident_size
                    )
    else:
        raise ("Unknow model")
    return model


class Unit2Mel(nn.Module):
    def __init__(
            self,
            input_channel,
            use_pitch_aug=False,
            out_dims=128,
            n_layers=2,
            block_out_channels=(256,384,512,512),
            n_heads=8,
            n_hidden=256,
            mrte_layer=5,
            mrte_hident_size = 512
            ):
        super().__init__()
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        self.mrte = MRTE(
            out_dims,
            n_hidden,
            mrte_layer,
            mrte_hident_size,
            n_hidden,
            5,
            4,
            2
        )
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
            
        self.decoder = GaussianDiffusion(UNet1DConditionModel(
            in_channels=out_dims + n_hidden,
            out_channels=out_dims,
            block_out_channels=block_out_channels,
            norm_num_groups=8,
            cross_attention_dim = out_dims,
            attention_head_dim = n_heads,
            layers_per_block = n_layers,
            addition_embed_type='text',
            resnet_time_scale_shift='scale_shift'), out_dims=out_dims)

    def forward(self, units, f0, volume, reference_audio_mel,aug_shift=None, gt_spec=None, infer=True, infer_speedup=10, method='dpm-solver', k_step=None, use_tqdm=True):
        x = self.mrte(self.unit_embed(units), reference_audio_mel)
        x += self.f0_embed((1 + f0 / 700).log()) + self.volume_embed(volume)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)
        x = self.decoder(x, gt_spec=gt_spec,reference_mel = reference_audio_mel, infer=infer, infer_speedup=infer_speedup, method=method, k_step=k_step,
                         use_tqdm=use_tqdm)

        return x
