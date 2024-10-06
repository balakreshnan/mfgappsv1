# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import argparse
import os
import readline
import glob

import onnxruntime_genai as og
import time

def _complete(text, state):
    return (glob.glob(text + "*") + [None])[state]


def run(args: argparse.Namespace):
    print("Loading model...")
    model = og.Model(args.model_path)
    processor = model.create_multimodal_processor()
    tokenizer_stream = processor.create_stream()

    while True:
        readline.set_completer_delims(" \t\n;")
        readline.parse_and_bind("tab: complete")
        readline.set_completer(_complete)
        image_paths = [
            image_path.strip()
            for image_path in input(
                "Image Path (comma separated; leave empty if no image): "
            ).split(",")
        ]
        image_paths = [image_path for image_path in image_paths if len(image_path)]
        print(image_paths)

        images = None
        prompt = "<|user|>\n"
        if len(image_paths) == 0:
            print("No image provided")
        else:
            print("Loading images...")
            for i, image_path in enumerate(image_paths):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                prompt += f"<|image_{i+1}|>\n"

            images = og.Images.open(*image_paths)
            print("Images loaded")

        text = input("Prompt: ")
        prompt += f"{text}<|end|>\n<|assistant|>\n"
        print("Processing images and prompt...")
        inputs = processor(prompt, images=images)

        starttime = time.time()

        print("Generating response...")
        params = og.GeneratorParams(model)
        params.set_inputs(inputs)
        params.set_search_options(max_length=7680)

        generator = og.Generator(model, params)

        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end="", flush=True)

        endtime = time.time()
        #print('Time taken: ', endtime - starttime)
        print(f"Time taken: {endtime - starttime:.2f} seconds")

        for _ in range(3):
            print()

        # Delete the generator to free the captured graph before creating another one
        del generator

# python phi3vgpu.py -m cuda-int4-rtn-block-32 
# C:\Code\gradioapps\mfgappsv1\images\DunnesStoresImmuneClosedCups450gLabel.jpg
# C:\Code\gradioapps\mfgappsv1\images\highvisibility_blog_1200x628_8-1024x536.jpg
# C:\Code\gradioapps\mfgappsv1\images\istockphoto-1332558192-612x612.jpg
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", type=str, required=True, help="Path to the model"
    )
    args = parser.parse_args()
    run(args)
