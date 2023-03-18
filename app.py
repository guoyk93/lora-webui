#!/usr/bin/env python
import json
import os
import subprocess
import tempfile
import threading
import time
from os import path
from pathlib import Path
from shutil import rmtree
from typing import List, TypeVar, Dict

import gradio as gr

from convert import convert_lora_file_to_safetensors

TITLE = "DreamBooth LoRA WebUI"

working_dir = path.dirname(__file__)

cache_file = path.join(working_dir, "config.cache.json")

FIELDS = []

T = TypeVar("T")


def save_config_cache(data):
    with open(cache_file, 'w') as f:
        json.dump(data, f)


def load_config_cache() -> dict:
    if not path.exists(cache_file):
        return {}
    with open(cache_file) as f:
        return json.load(f)


def register_input(name, block: T) -> T:
    FIELDS.append({
        'name': name,
        'block': block,
    })
    return block


def extract_input_names() -> List[str]:
    return [item['name'] for item in FIELDS]


def extract_inputs() -> List[any]:
    return [item['block'] for item in FIELDS]


def extract_input_values(*args) -> Dict[str, any]:
    names = extract_input_names()
    return {name: args[idx] for idx, name in enumerate(names)}


def create_textbox(name: str, **kwargs) -> gr.Textbox:
    kwargs['label'] = name.replace('_', ' ').capitalize()
    if 'max_lines' not in kwargs:
        kwargs['max_lines'] = 1
    if 'interactive' not in kwargs:
        kwargs['interactive'] = True
    return register_input(name, gr.Textbox(**kwargs))


def create_files(name: str, **kwargs) -> gr.Files:
    kwargs['label'] = name.replace('_', ' ').capitalize()
    if 'interactive' not in kwargs:
        kwargs['interactive'] = True
    return register_input(name, gr.Files(**kwargs))


def create_radio(name: str, choices: List[str], **kwargs) -> gr.Radio:
    kwargs['label'] = name.replace('_', ' ').capitalize()
    if 'choices' not in kwargs:
        kwargs['choices'] = choices
    if 'interactive' not in kwargs:
        kwargs['interactive'] = True
    return register_input(name, gr.Radio(**kwargs))


def create_checkbox(name: str, value: bool, **kwargs) -> gr.Checkbox:
    kwargs['label'] = name.replace('_', ' ').capitalize()
    if 'interactive' not in kwargs:
        kwargs['interactive'] = True
    if 'value' not in kwargs:
        kwargs['value'] = value
    return register_input(name, gr.Checkbox(**kwargs))


def create_number(name: str, value: any, **kwargs) -> gr.Number:
    kwargs['label'] = name.replace('_', ' ').capitalize()
    if 'value' not in kwargs:
        kwargs['value'] = value
    if 'interactive' not in kwargs:
        kwargs['interactive'] = True
    return register_input(name, gr.Number(**kwargs))


def wrap_vargs_method_with_progress(m: str, num: int):
    args = ''.join([f'v{i}, ' for i in range(1, num + 1)])

    exec(f"""
def wrapped_method({args}progress=gr.Progress()):
    return {m}({args}progress=progress)
    """)

    return locals().get('wrapped_method')


def do_load():
    data = load_config_cache()

    ret = []

    for item in FIELDS:
        if item['name'] in data:
            ret.append(data[item['name']])
        else:
            ret.append(item['block'].value)

    return ret


def do_train(*args, progress=gr.Progress()):
    arg_train = [
        "accelerate",
        "launch",
        path.join(path.dirname(__file__), "scripts", "train_dreambooth_lora.py")
    ]

    values = extract_input_values(*args)
    save_config_cache(values)

    for key, val in values.items():
        if val:
            if val is True:
                arg_train.append('--' + key)
            else:
                arg_train.append('--' + key)
                arg_train.append(str(val))
        else:
            if val == 0 and val is not False:
                arg_train.append('--' + key)
                arg_train.append(str(val))

    prg_path = tempfile.mktemp(suffix="train_dreambooth_lora.progress")
    Path(prg_path).write_text('0.01, Starting')

    arg_train.append("--progress_file")
    arg_train.append(prg_path)

    def show_progress(_prg_path, _progress):
        p = Path(_prg_path)
        while True:
            if not path.exists(_prg_path):
                return
            try:
                content = p.read_text().strip()
                v1, v2 = content.split(',', 2)
                _progress(float(v1.strip()) * 0.8 + 0.1, v2.strip())
            except ValueError:
                pass
            time.sleep(1)

    progress(0.1, "Training")
    threading.Thread(target=show_progress, args=(prg_path, progress)).start()

    try:
        subprocess.run(arg_train, check=True)
    except subprocess.CalledProcessError as e:
        return str(e)
    finally:
        os.remove(prg_path)

    progress(0.9, "Converting to .safetensors")

    convert_input = path.join(values['output_dir'], 'pytorch_lora_weights.bin')
    convert_output = path.join(values['output_dir'], 'lora.safetensors')

    convert_lora_file_to_safetensors(convert_input, convert_output)

    return f'\n\nCompleted:\n\n{convert_output}'


def create_training():
    with gr.Row():
        with gr.Column():
            with gr.Box():
                gr.Markdown('**Input**')
                create_textbox(
                    'instance_data_dir',
                    value=path.join(working_dir, "src"),
                )
                create_textbox(
                    'instance_prompt',
                    value='a photo of guoyk',
                )
                gr.Markdown(
                    '''
                    - Upload images or specify a path of instance images
                    - For an instance prompt, use a unique, made up word to avoid collisions.
                    '''
                )
            with gr.Box():
                gr.Markdown('Output')
                create_textbox(
                    'output_dir',
                    value=path.join(working_dir, "out")
                )
                create_textbox(
                    'validation_prompt',
                    value='a photo of guoyk in red dress',
                )
                with gr.Row():
                    create_number(
                        'num_validation_images', 4,
                        precision=0,
                    )
                    create_number(
                        "validation_epochs", 50,
                        precision=0,
                    )

            with gr.Box():
                with gr.Row():
                    button_start = gr.Button("Start")
                    button_load = gr.Button("Load")

            with gr.Box():
                gr.Markdown('Output message')
                with gr.Box():
                    output_message = gr.Markdown()

        with gr.Column():
            with gr.Box():
                with gr.Row():
                    gr.Markdown("**🛠 Model & Loading**")
                with gr.Row():
                    create_textbox(
                        'pretrained_model_name_or_path',
                        value='runwayml/stable-diffusion-v1-5'
                    )
                    create_textbox('revision')
                with gr.Row():
                    create_radio(
                        'resolution', ['512', '768'],
                        value='512'
                    )
                    create_checkbox('center_crop', False)
                with gr.Row():
                    create_number(
                        'train_batch_size', 1,
                        precision=0,
                    )
                    create_number(
                        'gradient_accumulation_steps', 1,
                        precision=0,
                    )
                    create_number(
                        'dataloader_num_workers', 0,
                        precision=0,
                    )

                with gr.Row():
                    gr.Markdown("**🛠 Learning Rate & Steps**")

                with gr.Row():
                    with gr.Column():
                        create_radio(
                            'lr_scheduler',
                            ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                             "constant_with_warmup"],
                            value='constant',
                        )
                    with gr.Column():
                        create_number(
                            'lr_num_cycles', 1,
                            precision=0,
                        )
                        create_number(
                            'lr_power', 1
                        )

                with gr.Row():
                    create_number(
                        'max_train_steps', 1000,
                        precision=0,
                    )
                    create_number(
                        'learning_rate', 0.0001
                    )
                    create_number(
                        'lr_warmup_steps', 200,
                        precision=0,
                    )

                with gr.Row():
                    gr.Markdown("**🛠 Checkpointing & Seed**")

                with gr.Row():
                    with gr.Column():
                        create_number(
                            'checkpointing_steps', 100,
                            precision=0,
                        )
                        create_textbox('resume_from_checkpoint')
                    with gr.Column():
                        create_number(
                            'seed', 0,
                            precision=0,
                        )

                with gr.Row():
                    gr.Markdown("**🛠 Precision**")

                with gr.Row():
                    with gr.Column():
                        create_radio('mixed_precision', ['no', 'fp16', 'bf16'], value="fp16")
                        create_radio('prior_generation_precision', ['no', 'fp32', 'fp16', 'bf16'], value="fp16")
                    with gr.Column():
                        create_checkbox('use_8bit_adam', False)
                        create_checkbox('enable_xformers_memory_efficient_attention', True)

    button_start.click(
        fn=wrap_vargs_method_with_progress('do_train', len(FIELDS)),
        inputs=extract_inputs(),
        outputs=output_message
    )

    button_load.click(
        fn=do_load,
        outputs=extract_inputs(),
    )


def create_app() -> gr.Blocks:
    with gr.Blocks(
            title=TITLE,
            css=path.join(path.dirname(__file__), 'style.css'),
    ) as app:
        with gr.Row():
            with gr.Column():
                gr.Markdown(f'**{TITLE}**')
        create_training()
    return app


def main():
    app = create_app()
    app.queue(max_size=1).launch(share=True)


if __name__ == "__main__":
    main()
