# <span style="font-family: 'Courier New', monospace; font-weight: bold">T2V-Turbo</span>: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback

## Fast and High-Quality Text-to-video Generation 🚀

### 4-Step Results
<table class="center">
  <td><img src=assets/demo_videos/4steps/0273.gif width="320"></td></td>
  <td><img src=assets/demo_videos/4steps/0054.gif width="320"></td></td>
  <td><img src=assets/demo_videos/4steps/0262.gif width="320"></td></td></td>
  <tr>
  <td style="text-align:center;" width="320">With the style of low-poly game art, A majestic, white horse gallops gracefully across a moonlit beach.</td>
  <td style="text-align:center;" width="320">medium shot of Christine, a beautiful 25-year-old brunette resembling Selena Gomez, anxiously looking up as she walks down a New York street, cinematic style</td>
  <td style="text-align:center;" width="320">a cartoon pig playing his guitar, Andrew Warhol style</td>
  <tr>
</table >

<table class="center">
  <td><img src=assets/demo_videos/4steps/0023.gif width="320"></td></td>
  <td><img src=assets/demo_videos/4steps/0021.gif width="320"></td></td>
  <td><img src=assets/demo_videos/4steps/0064.gif width="320"></td></td>

  <tr>
  <td style="text-align:center;" width="320">a dog wearing vr goggles on a boat</td>
  <td style="text-align:center;" width="320">Pikachu snowboarding</td>
  <td style="text-align:center;" width="320">a girl floating underwater </td>
  <tr>
</table >

### 8-Step Results

<table class="center">
  <td><img src=assets/demo_videos/8steps/0026.gif width="320"></td></td>
  <td><img src=assets/demo_videos/8steps/0062.gif width="320"></td></td>
  <td><img src=assets/demo_videos/8steps/0065.gif width="320"></td></td></td>
  <tr>
  <td style="text-align:center;" width="320">Mickey Mouse is dancing on white background</td>
  <td style="text-align:center;" width="320">light wind, feathers moving, she moves her gaze, 4k</td>
  <td style="text-align:center;" width="320">fashion portrait shoot of a girl in colorful glasses, a breeze moves her hair </td>
  <tr>
</table >

<table class="center">
  <td><img src=assets/demo_videos/8steps/0348.gif width="320"></td></td>
  <td><img src=assets/demo_videos/8steps/0266.gif width="320"></td></td>
  <td><img src=assets/demo_videos/8steps/0278.gif width="320"></td></td>

  <tr>
  <td style="text-align:center;" width="320">With the style of abstract cubism, The flowers swayed in the gentle breeze, releasing their sweet fragrance.</td>
  <td style="text-align:center;" width="320">impressionist style, a yellow rubber duck floating on the wave on the sunset</td>
  <td style="text-align:center;" width="320">A Egyptian tomp hieroglyphics painting ofA regal lion, decked out in a jeweled crown, surveys his kingdom.</td>
  <tr>
</table >

## 🏭 Installation

```
pip install accelerate transformers diffusers webdataset loralib peft pytorch_lightning open_clip_torch hpsv2 peft wandb av einops packaging omegaconf opencv-python kornia

pip install flash-attn --no-build-isolation
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install csrc/fused_dense_lib csrc/layer_norm

pip install git+https://github.com/iejMac/video2dataset.git

conda install xformers
```
## 🛞 Model Checkpoints

|Model|Resolution|Checkpoints|
|:---------|:---------|:--------|
|T2V-Turbo (VC2)|320x512|[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2/blob/main/unet_lora.pt)
|T2V-Turbo (MS)|256x256|[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-MS/blob/main/unet_lora.pt)


## 🚀 Inference

We provide local demo codes supported with gradio (For MacOS users, need to set the device="mps" in app.py; For Intel GPU users, set device="xpu" in app.py).
1. Download the `unet_lora.pt` of our T2V-Turbo (VC2) [here](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2/blob/main/unet_lora.pt).

2. Download the model checkpoint of VideoCrafter2 [here](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt).

3. Launch the gradio demo with the following command:
```
pip install gradio==3.48.0
python app.py --unet_dir PATH_TO_UNET_LORA.pt --base_model_dir PATH_TO_VideoCrafter2_MODEL_CKPT
```

## 🏋️ Training

To train T2V-Turbo (VC2), run the following command

```
bash train_t2v_turbo_vc2.sh
```

To train T2V-Turbo (MS), run the following command

```
bash train_t2v_turbo_ms.sh
```
