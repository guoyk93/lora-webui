#!/usr/bin/env python
import subprocess
from os import path
from typing import List, TypeVar, Dict

import gradio as gr

TITLE = "DreamBooth LoRA WebUI"

working_dir = path.dirname(__file__)

BLOCKS = []

T = TypeVar("T")


def register_input(name, block: T) -> T:
    BLOCKS.append({
        'name': name,
        'block': block,
    })
    return block


def extract_input_names() -> List[str]:
    return [item['name'] for item in BLOCKS]


def extract_inputs() -> List[any]:
    return [item['block'] for item in BLOCKS]


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


def do_train(*args):
    arg_train = [
        "accelerate",
        "launch",
        path.join(path.dirname(__file__), "scripts", "train_dreambooth_lora.py")
    ]
    values = extract_input_values(*args)
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

    subprocess.run(arg_train, check=True)

    arg_convert = [
        "python",
        path.join(path.dirname(__file__), "scripts", "diffusers-lora-to-safetensors.py"),
        "--file",
        path.join(values['output_dir'], 'pytorch_lora_weights.bin')
    ]

    subprocess.run(arg_convert, check=True)

    return 'Complete, check: ' + path.join(values['output_dir'], 'pytorch_lora_weights_converted.safetensors')


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
                button_start = gr.Button("Start")

            with gr.Box():
                gr.Markdown('Output message')
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
        fn=do_train,
        inputs=extract_inputs(),
        outputs=output_message
    )


def create_app() -> gr.Blocks:
    with gr.Blocks(
            title=TITLE,
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
