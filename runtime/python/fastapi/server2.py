# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import os
import sys
import argparse
import logging
import torchaudio
import torch
import time
from pydantic import BaseModel

logging.getLogger("matplotlib").setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File, Response
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append("{}/../../..".format(ROOT_DIR))
sys.path.append("{}/../../../third_party/Matcha-TTS".format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav

app = FastAPI()
# set cross region allowance
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 非流式wav数据
def generate_data(model_output):
    tts_speeches = []
    for i in model_output:
        tts_speeches.append(i["tts_speech"])
    output = torch.concat(tts_speeches, dim=1)

    buffer = io.BytesIO()
    torchaudio.save(buffer, output, 22050, format="wav")
    buffer.seek(0)
    return buffer.read(-1)


def load_reference_path(ref_id: str):
    # Assuming 'references/xxx' is a directory containing files
    # This code reads the path of the first file in the directory
    directory_path = "references/" + ref_id
    files = os.listdir(directory_path)
    if files:  # Check if the directory is not empty
        return os.path.join(directory_path, files[0])
    else:
        print("Directory is empty.")


class InputData(BaseModel):
    text: str
    prompt_text: str = None
    spk_id: str = None
    reference_id: str = None
    instruct_text: str = None


@app.get("/inference_sft")
@app.post("/inference_sft")
async def inference_sft(input: InputData):
    model_output = cosyvoice.inference_sft(input.text, input.spk_id)
    return Response(generate_data(model_output))


@app.get("/inference_zero_shot")
@app.post("/inference_zero_shot")
async def inference_zero_shot(
input: InputData):
    path = load_reference_path(input.reference_id)
    prompt_speech_16k = load_wav(path, 16000)
    model_output = cosyvoice.inference_zero_shot(
        input.text, input.prompt_text, prompt_speech_16k
    )
    return Response(generate_data(model_output))


@app.get("/inference_cross_lingual")
@app.post("/inference_cross_lingual")
def inference_cross_lingual(input: InputData):
    path = load_reference_path(input.reference_id)
    prompt_speech_16k = load_wav(path, 16000)
    model_output = cosyvoice.inference_cross_lingual(input.text, prompt_speech_16k)
    return Response(generate_data(model_output), media_type="audio/wav")


@app.get("/inference_instruct")
@app.post("/inference_instruct")
async def inference_instruct(
   input: InputData 
):
    model_output = cosyvoice.inference_instruct(input.text, input.spk_id, input.instruct_text)
    return Response(generate_data(model_output))


@app.get("/inference_instruct2")
@app.post("/inference_instruct2")
async def inference_instruct2(
    tts_text: str = Form(), instruct_text: str = Form(), prompt_wav: UploadFile = File()
):
    prompt_speech_16k = load_wav(prompt_wav.file, 16000)
    model_output = cosyvoice.inference_instruct2(
        tts_text, instruct_text, prompt_speech_16k
    )
    return Response(generate_data(model_output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=50000)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="iic/CosyVoice-300M",
        help="local path or modelscope repo id",
    )
    args = parser.parse_args()
    try:
        cosyvoice = CosyVoice(args.model_dir)
    except Exception:
        try:
            cosyvoice = CosyVoice2(args.model_dir)
        except Exception:
            raise TypeError("no valid model_type!")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
