from typing import Dict, List
import torch
import numpy as np
import copy
import cv2
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class OrbitImageDataset(BaseImageDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 shape_meta=None,
                 max_train_episodes=None,
                 task_name=None,
                 resize_image=(84, 84),
                 ):
        super().__init__()
        self.resize_image = resize_image
        self.task_name = task_name
        obs_meta = shape_meta.get('obs', {})
        self.image_keys = [key for key, meta in obs_meta.items() if meta.get('type') == 'rgb']
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['action', 'joint_pos'] + self.image_keys)
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        
       

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'joint_pos': self.replay_buffer['joint_pos'],
            'action': self.replay_buffer['action'],
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        for key in self.image_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):

        joint_pos = sample['joint_pos'].astype(np.float32)
        images = {}
        for key in self.image_keys:
            img = sample[key].squeeze(1) / 255
            resized_img = np.array([cv2.resize(im, self.resize_image) for im in img])
            images[key] = resized_img

        data = {
            'obs': {
                'joint_pos': joint_pos,
            },
            'action': sample['action'].astype(np.float32),
        }
        data['obs'].update(images)

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data