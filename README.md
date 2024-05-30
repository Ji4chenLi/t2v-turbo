# <span style="font-family: 'Courier New', monospace; font-weight: bold">T2V-Turbo</span>: Breaking the Quality Bottleneck of Video Consistency Model with Mixed Reward Feedback

## 4-step Text-to-video Generation 🚀

<table class="center">
  <td>
  <video poster="" autoplay controls muted loop height="150%" playbackRate=2.0><source src="https://rg-lcd.s3.amazonaws.com/t2v-turbo-demo/t2v-turbo-vc2/4steps/0273.mp4" type="video/mp4"></video>
  </td>
  <td>
  <video poster="" autoplay controls muted loop height="150%" playbackRate=2.0><source src="https://rg-lcd.s3.amazonaws.com/t2v-turbo-demo/t2v-turbo-vc2/4steps/0054.mp4" type="video/mp4"></video>
  </td>
  <td>
  <video poster="" autoplay controls muted loop height="150%" playbackRate=2.0><source src="https://rg-lcd.s3.amazonaws.com/t2v-turbo-demo/t2v-turbo-vc2/4steps/0262.mp4" type="video/mp4"></video>
  </td>

  <tr>
  <td style="text-align:center;" width="320">With the style of low-poly game art, A majestic, white horse gallops gracefully across a moonlit beach.</td>
  <td style="text-align:center;" width="320">medium shot of Christine, a beautiful 25-year-old brunette resembling Selena Gomez, anxiously looking up as she walks down a New York street, cinematic style</td>
  <td style="text-align:center;" width="320">a cartoon pig playing his guitar, Andrew Warhol style</td>
  <tr>
</table >

<table class="center">
  <td>
  <video poster="" autoplay controls muted loop height="150%" playbackRate=2.0><source src="https://rg-lcd.s3.amazonaws.com/t2v-turbo-demo/t2v-turbo-vc2/4steps/0023.mp4" type="video/mp4"></video>
  </td>
  <td>
  <video poster="" autoplay controls muted loop height="150%" playbackRate=2.0><source src="https://rg-lcd.s3.amazonaws.com/t2v-turbo-demo/t2v-turbo-vc2/4steps/0021.mp4" type="video/mp4"></video>
  </td>
  <td>
  <video poster="" autoplay controls muted loop height="150%" playbackRate=2.0><source src="https://rg-lcd.s3.amazonaws.com/t2v-turbo-demo/t2v-turbo-vc2/4steps/0064.mp4" type="video/mp4"></video>
  </td>

  <tr>
  <td style="text-align:center;" width="320">a dog wearing vr goggles on a boat</td>
  <td style="text-align:center;" width="320">Pikachu snowboarding</td>
  <td style="text-align:center;" width="320">a girl floating underwater </td>
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
