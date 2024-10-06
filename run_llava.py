import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

def load_model_processor(model_id):
  model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
  ).to(0)

  processor = AutoProcessor.from_pretrained(model_id)
  return model, processor


def create_chat_template(text_prompt):
  # Define a chat histiry and use `apply_chat_template` to get correctly formatted prompt
  # Each value in "content" has to be a list of dicts with types ("text", "image")
  conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": f'{text_prompt}'},
          {"type": "image"},
        ],
    },
  ]
  return conversation

def query_model(model, processor, text_prompt, image_file, temperature = 0):

  conversation = create_chat_template(text_prompt)
  prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

  if isinstance(image_file, str):
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
  else:
    raw_image = image_file
  inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
  with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=True if temperature > 0 else False,
            temperature = temperature,
            top_p = None,
            num_beams = 1,
            max_new_tokens = 200,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True
        )
  generated_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

  # Extract the part of the generated text after "ASSISTANT:"
  assistant_tag = "ASSISTANT:"
  if assistant_tag in generated_text:
      output_text = generated_text.split(assistant_tag)[1].strip()
  else:
    output_text = generated_text.strip()

  transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
  return output_text, outputs.scores, transition_scores