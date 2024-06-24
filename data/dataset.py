"""video dataset creation"""

import webdataset as wds
from functools import partial
from typing import List, Union

from webdataset import WebLoader
from video2dataset.dataloader.custom_wds import (
    dict_collation_fn,
    TorchDataWebdataset,
)
from video2dataset.dataloader.filters import KeyFilter
from video2dataset.dataloader.transform import VideoResizer, CustomTransforms
from video2dataset.dataloader.video_decode import VideoDecorder


def reassemble(x):
    """
    Process a dictionary by updating its values based on certain conditions.

    :param dict x: The input dictionary to process.
    :return: The processed dictionary.
    :rtype: dict
    """
    new_dict = {}

    for key in x:
        if key not in "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split():
            continue

        # this is updating the output of video decoders
        if isinstance(x[key], tuple) and len(x[key]) == 2:
            new_dict.update({f"{subk}": x[key][-1][subk] for subk in x[key][-1]})

        x[key] = x[key][0]
    x.update(new_dict)
    del new_dict
    return x


def get_video_dataset(
    urls: Union[str, List[str]],
    batch_size,
    shuffle=0,
    repeat=1,
    drop_last=False,
    video_key="mp4",
    decoder_kwargs=None,
    custom_transforms=None,
    resize_size=None,
    crop_size=None,
    random_crop=False,
    original_height_key="original_height",
    original_width_key="original_width",
    enforce_additional_keys=None,
    return_always: bool = False,
    handler=wds.warn_and_continue,
):
    """
    Generates a webdataset given the specified parameters.
    Parameters:
        urls (str, list(str)): The path to the dataset or a list of paths to the different locations of the dataset.
        batch_size (int): The number of samples per batch.
        shuffle (int, optional): Shuffle buffer size. Default is 0 means no shuffling.
        repeat (int, optional): Whether to repeat the dataset. Default is 1. -1 means repeating infinitely
        drop_last (bool, optional): Whether to drop the last incomplete batch. Default is False.
        video_key (str, optional): The key for video files. Default is 'mp4'.
        cuts_key (str, optional): The key for cut detection. Default is None.
        decoder_kwargs (dict, optional): Keyword arguments for the video decoder. Default is an empty dictionary.
        custom_transforms (dict, optional): Pairs of additional custom transforms to apply to samples.
        aesthetics_threshold (float, optional): Aesthetic threshold for filtering. Default is None.
        allowed_languages (list, optional): List of allowed languages. Default is None.
        p_unsafe_threshold (float, optional): Probability threshold for unsafe content filtering. Default is None.
        resize_size (tuple, optional): Tuple of (width, height) for resizing the video. Default is None.
        crop_size (tuple, optional): Tuple of (width, height) for cropping the video. Default is None.
        random_crop (bool, optional): Whether to apply random cropping. Default is False.
        original_height_key (str, optional): The key for the original video height. Default is 'original_height'.
        original_width_key (str, optional): The key for the original video width. Default is 'original_width'.
        keys_to_remove ((list, int), optional): Keys which, for the sake of speed, will be
        removed before decoding. Default is None which means nothing will be removed.
        enforce_additional_keys (list, optional): Which keys must be in each sample
    Returns:
        WebDataset: The processed webdataset.
    """

    assert decoder_kwargs is not None

    if enforce_additional_keys is None:
        enforce_additional_keys = ["txt"]

    if isinstance(urls, str):
        urls = [urls]
    # only use webdataset when using pipe
    use_torchdata = not urls[0].replace(" ", "").startswith("pipe:")

    if not use_torchdata:
        urls = urls[0]

    additional_decoder_kwargs = {}
    dataset_cls = (
        partial(
            wds.WebDataset,
            nodesplitter=wds.split_by_node,
        )
        if not use_torchdata
        else partial(
            TorchDataWebdataset,
            repeat=repeat,
            drop_last=drop_last,
            return_always=return_always,
            handler=handler,
        )
    )

    dset = dataset_cls(urls, shardshuffle=shuffle, handler=handler)

    if not use_torchdata:
        dset = dset.repeat(repeat).shuffle(shuffle, initial=shuffle)

    enforce_keys = [video_key] + enforce_additional_keys
    key_filter = KeyFilter(enforce_keys)
    dset = dset.select(key_filter)

    # Decoding
    dset = dset.decode(
        VideoDecorder(**decoder_kwargs),
        handler=handler,
        **additional_decoder_kwargs,
    ).map(reassemble, handler=handler)

    # Resizing
    dset = dset.map(
        VideoResizer(
            size=resize_size,
            crop_size=crop_size,
            random_crop=random_crop,
            key=video_key,
            width_key=original_width_key,
            height_key=original_height_key,
        ),
        handler=handler,
    )

    if custom_transforms:
        dset = dset.map(CustomTransforms(custom_transforms), handler=handler)

    dset = dset.batched(
        batch_size, partial=not drop_last, collation_fn=dict_collation_fn
    )
    return dset


if __name__ == "__main__":
    # WebVid validation split
    SHARDS = "PATH/TO/SHARDS"

    decoder_kwargs = {
        "n_frames": 16,  # get 8 frames from each video
        # "uniformly_sample": True,  # sample frames uniformly
        "fps": 16,  # sample frames at 16 fps
        "num_threads": 12,  # use 16 threads to decode the video
    }
    resize_size = crop_size = (320, 512)
    batch_size = 2

    dset = get_video_dataset(
        urls=SHARDS,
        batch_size=batch_size,
        decoder_kwargs=decoder_kwargs,
        resize_size=resize_size,
        crop_size=crop_size,
    )

    num_workers = 6  # 6 dataloader workers

    dl = WebLoader(dset, batch_size=None, num_workers=num_workers)

    for sample in dl:
        video_batch = sample["mp4"]
        print(video_batch.shape)  # torch.Size([32, 8, 256, 256, 3])

        # TODO: need to add option for text/metadata preprocessing (tokenization etc.)
        text_batch = sample["txt"]
        print(text_batch[0])
        metadata_batch = sample["json"]
        break
