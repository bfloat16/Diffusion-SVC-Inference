import torch
import os
import gradio as gr
import librosa
import soundfile as sf
from datetime import datetime
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from tools.infer_tools import DiffusionSVC

def generate_filename(input_wav, type):
    time = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    file_name = os.path.basename(input_wav).split('.')[0]
    if type == 'in':
        output_file_name = f'{time}' + "_in_" + f'{file_name}' + ".mp3"
    if type == 'ref':
        output_file_name = f'{time}' + "_ref_" + f'{file_name}' + ".mp3"
    if type == 'out':
        output_file_name = f'{time}' + "_out_" + f'{file_name}' + ".mp3"

    output_file_path = os.path.join("results", output_file_name)
    return output_file_path

def inference(input_wav, reference_wav, key, threhold, speedup, menthod, progress=gr.Progress(track_tqdm=True)):
    if input_wav == None or reference_wav == None:
        raise gr.Error("未输入音频")

    in_wav, in_sr = librosa.load(input_wav, sr=None)
    in_refer, in_rsr = librosa.load(reference_wav, sr=None)

    if int(len(in_wav)) <= int(in_sr * 305):
        in_wav = librosa.to_mono(in_wav)
    else:
        raise gr.Error("输入音频长度不能超过5分钟")

    if int(len(in_refer)) > int(in_sr * 35):
        raise gr.Error("参考音频长度不能超过30秒")
    #rec_result = inference_pipeline(audio_in=input_wav)
    #print(rec_result)
    print(datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
    out_wav, out_sr= diffusion_svc.infer_from_long_audio(in_wav, sr=(in_sr, in_rsr), key=float(key), refer_audio=str(reference_wav), aug_shift=0, infer_speedup=int(speedup),
                                                            method=menthod, use_tqdm=True, threhold=-60, threhold_for_split=float(threhold), min_len=5000)

    input_wav_path = generate_filename(input_wav, 'in')
    reference_wav_path = generate_filename(reference_wav, 'ref')
    output_wav_path = generate_filename(input_wav, 'out')

    sf.write(input_wav_path, in_wav, out_sr, format='mp3')
    sf.write(reference_wav_path, in_refer, out_sr, format='mp3')
    sf.write(output_wav_path, out_wav, out_sr, format='mp3')
    return output_wav_path

def main_ui():
    with gr.Blocks(theme=gr.themes.Base(primary_hue=gr.themes.colors.purple)) as ui:
        gr.Markdown('# Diffusion-SVC&nbsp;&nbsp;&nbsp;♬ヽ(*・ω・)ﾉ&nbsp;&nbsp;&nbsp;&nbsp;𝒁𝒆𝒓𝒐𝒔𝒉𝒐𝒕-Inference')
        gr.Markdown("### 推理音频限制5分钟，参考音频限制30秒")
        with gr.Row():
            input_wav = gr.Audio(type='filepath', label='推理音频', source='upload')
            reference_wav = gr.Audio(type='filepath', label='参考音频', source='upload')
        with gr.Column():
            with gr.Row():
                key = gr.Slider(minimum=-12, maximum=12, step=1, value=0, label='变调', interactive=True)
                threhold = gr.Slider(minimum=-50, maximum=-30, step=1, value=-40, label='切片阈值', interactive=True)
            speedup = gr.Slider(minimum=10, maximum=100, step=1, value=10, label='加速倍数', interactive=True)
            with gr.Row():
                f0_extractor = gr.Dropdown(choices=['fcpe', 'rmvpe', 'parselmouth', 'harvest'], value='fcpe', label='音高提取器', interactive=False)
                menthod = gr.Dropdown(choices=['unipc', 'dpm-solver'], value='unipc', label='采样方法', interactive=True)
        out_wav = gr.Audio(label='输出音频', format='mp3',interactive=False)
        submit = gr.Button(value='开始推理', variant="primary")

        submit.click(inference, [input_wav, reference_wav, key, threhold, speedup, menthod], [out_wav])

    ui.queue(status_update_rate=10, max_size=5)
    ui.launch(server_name='0.0.0.0', server_port=2233, share=False)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffusion_svc = DiffusionSVC(device=device)
    diffusion_svc.load_model(model_path='model_210000.pt', f0_model='fcpe', f0_max=1100, f0_min=50)
    '''
    param_dict = dict()
    param_dict['use_timestamp'] = False
    inference_pipeline = pipeline(
       task=Tasks.auto_speech_recognition,
       model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
       vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
       punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
       lm_model='damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch',
       lm_weight=0.15,
       beam_size=10,
       param_dict=param_dict
     )
     '''
    main_ui()
