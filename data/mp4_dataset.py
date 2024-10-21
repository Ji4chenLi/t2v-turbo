import pickle
import random

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

import boto3
from diffusers.utils import logging
import sys

from utils.common_utils import read_video_to_tensor

logger = logging.get_logger(__name__)


class MP4Dataset(Dataset):
    def __init__(
        self,
        path_to_csv,
        video_root="video_root",
        sample_fps=8,
        sample_frames=16,
        sample_size=[320, 512],
        bucket="BUCKET_NAME",
    ):
        self.video_root = video_root
        self.s3_client = boto3.client("s3")
        self.bucket = bucket

        logger.info(f"loading annotations from {path_to_csv} ...")
        self.video_df = pd.read_csv(path_to_csv)
        self.length = len(self.video_df)
        logger.info(f"data scale: {self.length}")

        self.sample_fps = sample_fps
        self.sample_frames = sample_frames

        sample_size = (
            tuple(sample_size)
            if not isinstance(sample_size, int)
            else (sample_size, sample_size)
        )
        self.pixel_transforms = transforms.Compose(
            [
                transforms.Resize(sample_size),
                transforms.CenterCrop(sample_size),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

    def get_video_text_pair(self, idx):
        video_dict = self.video_df.iloc[idx].to_dict()
        relpath, text = video_dict["relpath"], video_dict["text"]
        video_dir = f"{self.video_root}/{relpath.replace('.pkl', '.mp4')}"

        data_body = self.s3_client.get_object(Bucket=self.bucket, Key=video_dir).get(
            "Body"
        )
        pixel_values = read_video_to_tensor(
            data_body,
            self.sample_fps,
            self.sample_frames,
            uniform_sampling=False,
        )
        return pixel_values, text, relpath

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, text, relpath = self.get_video_text_pair(idx)
                break
            except Exception as e:
                idx = random.randint(0, self.length - 1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(mp4=pixel_values, txt=text, relpath=relpath)
        return sample


class MP4LatentDataset(Dataset):
    def __init__(
        self,
        path_to_csv,
        latent_root="latent_root",
        bucket="BUCKET_NAME",
    ):
        self.s3_resource = boto3.resource("s3")
        self.bucket = bucket
        self.latent_root = latent_root

        logger.info(f"loading annotations from {path_to_csv} ...")
        self.latent_df = pd.read_csv(path_to_csv)
        self.length = len(self.latent_df)
        logger.info(f"data scale: {self.length}")

    def get_latent_text_pair(self, idx):
        latent_dict = self.latent_df.iloc[idx].to_dict()
        relpath, text = latent_dict["relpath"], latent_dict["text"]
        if latent_dict.get("latent_root", None) is not None:
            latent_dir = f"{latent_dict['latent_root']}/{relpath}"
        else:
            latent_dir = f"{self.latent_root}/{relpath}"

        if "use_motion_guide" in latent_dict:
            use_motion_guide = bool(latent_dict["use_motion_guide"])
        else:
            use_motion_guide = True

        if "short_text" in latent_dict:
            short_text = latent_dict["short_text"]
        else:
            short_text = ""

        if str(short_text) == "nan":
            short_text = ""

        latent_dict = pickle.loads(
            self.s3_resource.Bucket(self.bucket).Object(latent_dir).get()["Body"].read()
        )
        if "webvid" in latent_dir:
            text = latent_dict.pop("text")
            short_text = text
        elif "text" in latent_dict:
            assert text == latent_dict.pop("text")
        return latent_dict, text, short_text, use_motion_guide

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # NOTE: Remove the while loop if you are debugging the dataset
        while True:
            try:
                latent_dict, text, short_text, use_motion_guide = (
                    self.get_latent_text_pair(idx)
                )
                for k in latent_dict.keys():
                    if isinstance(latent_dict[k], torch.Tensor):
                        latent_dict[k] = latent_dict[k].detach().cpu()
                sample = dict(
                    txt=text, short_txt=short_text, use_motion_guide=use_motion_guide
                )
                sample.update(latent_dict)
                break
            except Exception as e:
                idx = random.randint(0, self.length - 1)
        return sample


if __name__ == "__main__":
    import torchvision
    from torch.utils.data import DataLoader

    random_indx = list(range(10))
    dataset = MP4LatentDataset("data/mixed_motion_latent_128k_webvid.csv")
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for i, sample in enumerate(data_loader):
        print(sample["txt"])
        print(sample["short_txt"])
        print(sample["use_motion_guide"])
        print(sample["index"])
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
        break
