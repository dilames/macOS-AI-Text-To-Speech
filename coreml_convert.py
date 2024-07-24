import torch
import torchvision

from transformers import BarkProcessor, BarkModel, AutoTokenizer

# Load the processor and model
processor = BarkProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark", torchscript=True)
# Set the model in evaluation mode
model.eval()

# Trace: prepare example input
prompt = "Hello, my dog is cute"
voice_preset = "v2/en_speaker_6"

# Trace: model with prepared data.
example_inputs = processor(prompt, voice_preset=voice_preset)
# audio_array = model.generate(**example_inputs)
# audio_array = audio_array.cpu().numpy().squeeze()

# Traceable Inputs

example_inputs = processor(prompt, voice_preset=voice_preset)

# Trace Model
input_ids = example_inputs['input_ids']
attention_mask = example_inputs['attention_mask']
traced_model = torch.jit.trace(model, (input_ids, attention_mask))

# Run the traced model with inputs
output = traced_model(input_ids, attention_mask)

# traced_model = torch.jit.trace(
#     model, 
#     [
#         example_inputs['input_ids'], 
#         example_inputs['attention_mask']
#     ]
# )
# out = traced_model(example_inputs)

# Save waveform to .wav output audio file
# import scipy
# sample_rate = model.generation_config.sample_rate
# scipy.io.wavfile.write("bark_traced_out.wav", rate=sample_rate, data=audio_array)