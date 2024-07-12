import pickle
import cv2
import os
import numpy as np


class Dataset:
    def __init__(self, name, path=None):
        self.storage_dir = os.path.join(path, name) if path else name
        os.makedirs(self.storage_dir, exist_ok=True)
        self.max_seed = -1
        self.seeds = []
        self._len = len(self)

    def __len__(self):
        if hasattr(self, '_len'):
            return self._len
        # Get the current index by finding the largest index in existing files
        content = os.listdir(self.storage_dir)
        if not content:
            return 0
        self.seeds = sorted(self._seed_list(content))
        self.max_seed = max(self.seeds)
        return len(self.seeds) if self.seeds else 0

    def _seed_list(self, existing_files):
        raise NotImplementedError(
            "This method should be implemented in a subclass.")

    def __getitem__(self, item):
        return self.read_sample(item)

    def read_sample(self, idx):
        seed = self.seeds[idx]
        sample = self.read_seed(seed)
        return sample

    def read_sample_at_idx(self, idx, p_idx):
        seed = self.seeds[idx]
        sample = self.read_perspective_at_seed(seed, p_idx)
        return sample

    def read_seed(self, seed):
        raise NotImplementedError(
            "This method should be implemented in a subclass.")

    def read_perspective_at_seed(self, seed, p_idx):
        raise NotImplementedError(
            "This method should be implemented in a subclass.")

    def add_sample(self, seed, sample):
        # idx = len(self)
        # Instead of adding the image to a list, save it directly to disk
        self.write_sample(seed, sample)
        self._len += 1
        self.seeds.append(seed)
        self.max_seed = seed

    def write_sample(self, seed, sample):
        raise NotImplementedError(
            "This method should be implemented in a subclass.")

    def store_metadata(self, name, metadata):
        file_path = os.path.join(self.storage_dir, f'{name}.npz')
        np.savez_compressed(file_path, **metadata)

    def load_metadata(self, name):
        file_path = os.path.join(self.storage_dir, f'{name}.npz')
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        try:
            with np.load(file_path) as data:
                metadata = {kw: data[kw] for kw in data.keys()}
            if metadata is None:
                raise ValueError(
                    f"The file {file_path} is not a valid metadata file.")
            return metadata
        except Exception as e:
            raise IOError(
                f"An error occurred while reading the file {file_path}: {str(e)}")


class ColorDataset(Dataset):
    def __init__(self, name='color', n_perspectives=1, path=None):
        super().__init__(name, path)
        self.n_perspectives = n_perspectives

    def _seed_list(self, content):
        seeds = [int(f) for f in content if os.path.isdir(
            os.path.join(self.storage_dir, f)) and f.isdigit()]
        return seeds

    def get_seed_dir(self, seed):
        seed_dir = os.path.join(self.storage_dir, f'{seed:06d}')
        return seed_dir

    def write_sample(self, seed, sample):
        seed_dir = self.get_seed_dir(seed)
        os.makedirs(seed_dir, exist_ok=True)

        for i, perspective in enumerate(sample):
            file_path = self.get_file_path(i, seed_dir)
            cv2.imwrite(file_path, perspective)

    @staticmethod
    def get_file_path(i, seed_dir):
        file_path = os.path.join(seed_dir, f'img_{i:06d}.png')
        return file_path

    def read_seed(self, seed):
        # Load a single sample from disk by index with error handling
        seed_dir = self.get_seed_dir(seed)
        if not os.path.isdir(seed_dir):
            raise FileNotFoundError(f"The file {seed_dir} does not exist.")
        images = []
        for i in range(self.n_perspectives):
            file_path = self.get_file_path(i, seed_dir)
            try:
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(
                        f"The file {file_path} is not a valid image.")
                images.append(image)
            except Exception as e:
                raise IOError(
                    f"An error occurred while reading the file {file_path}: {str(e)}")
        return images

    def read_perspective_at_seed(self, seed, p_idx):
        seed_dir = self.get_seed_dir(seed)
        if not os.path.isdir(seed_dir):
            raise FileNotFoundError(f"The file {seed_dir} does not exist.")
        file_path = self.get_file_path(p_idx, seed_dir)
        try:
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"The file {file_path} is not a valid image.")
        except Exception as e:
            raise IOError(
                f"An error occurred while reading the file {file_path}: {str(e)}")
        return image


class MNPZDataset(Dataset):
    def __init__(self, keywords, n_samples, name='npz', path='./data'):
        super().__init__(name, path)
        self.keywords = keywords
        self.n_samples = n_samples

    def _seed_list(self, content):
        seeds = [int(f) for f in content if os.path.isdir(
            os.path.join(self.storage_dir, f)) and f.isdigit()]
        return seeds

    def get_seed_dir(self, seed):
        seed_dir = os.path.join(self.storage_dir, f'{seed:06d}')
        return seed_dir

    def write_sample(self, seed, sample):
        seed_dir = self.get_seed_dir(seed)
        os.makedirs(seed_dir, exist_ok=True)
        for i, s in enumerate(sample):
            if not set(self.keywords) == set(s.keys()):
                raise ValueError(
                    f"Sample keys {s.keys()} do not match dataset keys {self.keywords}.")
            file_path = self.get_file_path(i, seed_dir)
            kwargs = {kw: s[kw] for kw in self.keywords}
            np.savez_compressed(file_path, **kwargs)

    def read_seed(self, seed):
        # Load a single sample from disk by index with error handling
        seed_dir = self.get_seed_dir(seed)
        if not os.path.isdir(seed_dir):
            raise FileNotFoundError(f"The file {seed_dir} does not exist.")
        samples = []
        for i in range(self.n_samples):
            file_path = self.get_file_path(i, seed_dir)
            try:
                with np.load(file_path) as data:
                    if not set(self.keywords) == set(data.keys()):
                        raise ValueError(
                            f"Sample keys {data.keys()} do not match dataset keys {self.keywords}.")
                    sample = {kw: data[kw] for kw in self.keywords}
                if sample is None:
                    raise ValueError(
                        f"The file {file_path} is not a valid sample.")
                samples.append(sample)
            except Exception as e:
                raise IOError(
                    f"An error occurred while reading the file {file_path}: {str(e)}")
        return samples

    def read_perspective_at_seed(self, seed, p_idx):
        seed_dir = self.get_seed_dir(seed)
        if not os.path.isdir(seed_dir):
            raise FileNotFoundError(f"The file {seed_dir} does not exist.")
        file_path = self.get_file_path(p_idx, seed_dir)
        try:
            with np.load(file_path) as data:
                if not set(self.keywords) == set(data.keys()):
                    raise ValueError(
                        f"Sample keys {data.keys()} do not match dataset keys {self.keywords}.")
                sample = {kw: data[kw] for kw in self.keywords}
            if sample is None:
                raise ValueError(
                    f"The file {file_path} is not a valid sample.")
        except Exception as e:
            raise IOError(
                f"An error occurred while reading the file {file_path}: {str(e)}")
        return sample

    @staticmethod
    def get_file_path(i, idx_dir):
        file_path = os.path.join(idx_dir, f'sample_{i:06d}.npz')
        return file_path


class NPZDataset(MNPZDataset):
    def __init__(self, keywords, name='npz', path='./data'):
        super().__init__(keywords, 1, name, path)

    def write_sample(self, seed, sample):
        super().write_sample(seed, [sample])

    def read_seed(self, seed):
        return super().read_seed(seed)[0]


class PickleDataset(Dataset):
    def __init__(self, name='pickle', path=None):
        super().__init__(name, path)

    def _seed_list(self, content):
        seeds = [int(f.split('_')[1].split('.')[0]) for f in content if
                 f.startswith('sample_') and f.endswith('.pkl')]
        return seeds

    def write_sample(self, seed, sample):
        file_path = self.get_file_path(seed)
        with open(file_path, 'wb') as f:
            pickle.dump(sample, f)

    def get_file_path(self, seed):
        file_path = os.path.join(self.storage_dir, f'sample_{seed:06d}.pkl')
        return file_path

    def read_seed(self, seed):
        # Load a single sample from disk by index with error handling
        file_path = self.get_file_path(seed)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        try:
            with open(file_path, 'rb') as f:
                sample = pickle.load(f)
            if sample is None:
                raise ValueError(
                    f"The file {file_path} is not a valid sample.")
            return sample
        except Exception as e:
            raise IOError(
                f"An error occurred while reading the file {file_path}: {str(e)}")


class LanguageDataset(Dataset):
    def __init__(self, name='language', path=None):
        super().__init__(name, path)

    def _seed_list(self, content):
        seeds = [int(f.split('_')[1].split('.')[0]) for f in content if
                 f.startswith('sample_') and f.endswith('.txt')]
        return seeds

    def write_sample(self, seed: int, sample: str):
        file_path = self.get_file_path(seed)
        with open(file_path, 'w') as text_file:
            text_file.write(sample)

    def get_file_path(self, seed: int):
        file_path = os.path.join(self.storage_dir, f'sample_{seed:06d}.txt')
        return file_path

    def read_seed(self, seed: int):
        # Load a single sample from disk by index with error handling
        file_path = self.get_file_path(seed)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        try:
            with open(file_path, 'r') as text_file:
                sample = text_file.read()
            if sample is None:
                raise ValueError(
                    f"The file {file_path} is not a valid sample.")
            return sample
        except Exception as e:
            raise IOError(
                f"An error occurred while reading the file {file_path}: {str(e)}")


class SynchronizedDatasets:
    def __init__(self, datasets):
        self.datasets = datasets
        self.max_seed = -1
        self._len = len(self)

    def __len__(self):
        if hasattr(self, '_len'):
            return self._len
        lengths = [len(dataset) for dataset in self.datasets.values()]
        max_seeds = [dataset.max_seed for dataset in self.datasets.values()]
        seeds_list = [dataset.seeds for dataset in self.datasets.values()]
        if not all([length == lengths[0] for length in lengths]):
            dataset_lengths = {dataset_name: len(
                dataset) for dataset_name, dataset in self.datasets.items()}
            debug_str = ', '.join(
                [f"{dataset_name}: {length}" for dataset_name, length in dataset_lengths.items()])
            print(f"Dataset lengths: {debug_str}")
            raise ValueError(
                f"All datasets must have the same length, but lengths are {lengths}.")
        if not all([max_seed == max_seeds[0] for max_seed in max_seeds]):
            raise ValueError(
                f"All datasets must have the same max_seed, but max_seeds are {max_seeds}")
        self.max_seed = max_seeds[0]
        self.seeds = seeds_list[0]
        return lengths[0]

    def __getitem__(self, item):
        return {dataset_name: dataset[item] for dataset_name, dataset in self.datasets.items()}

    def add_sample(self, seed, synchronized_samples):
        for dataset_name, dataset in self.datasets.items():
            dataset.add_sample(seed, synchronized_samples[dataset_name])
        self._len += 1


def load_dataset_nerf(n_perspectives, path):
    color_dataset = ColorDataset(name='color', n_perspectives=n_perspectives,
                                 path=path)
    camera_config_dataset = MNPZDataset(keywords=['intrinsics', 'pose'], name='camera_config',
                                        path=path,
                                        n_samples=n_perspectives)
    info_dataset = PickleDataset(name='info', path=path)
    datasets = {
        'color': color_dataset,
        'camera_config': camera_config_dataset,
        'info': info_dataset,
    }
    synchronized_dataset = SynchronizedDatasets(datasets)
    print(
        f"Dataset {path} loaded with {len(synchronized_dataset)} samples")
    return synchronized_dataset


def store_to_dataset_nerf(synchronized_dataset, observations, info, seed):
    images = [observation['color'] for observation in observations]
    camera_configs = [{'intrinsics': observation['intrinsics'], 'pose': observation['pose']} for observation
                      in observations]
    data_sample = {
        'color': images,
        'camera_config': camera_configs,
        'info': info
    }
    synchronized_dataset.add_sample(seed, data_sample)


def load_dataset_goal(n_perspectives, path):
    color_dataset = ColorDataset(name='color', n_perspectives=n_perspectives,
                                 path=path)
    camera_config_dataset = MNPZDataset(keywords=['intrinsics', 'pose'], name='camera_config',
                                        path=path,
                                        n_samples=n_perspectives)
    grasp_pose_dataset = NPZDataset(
        keywords=['grasp_pose'], name='grasp_pose', path=path)
    info_dataset = PickleDataset(name='info', path=path)
    datasets = {'color': color_dataset,
                'camera_config': camera_config_dataset,
                'grasp_pose': grasp_pose_dataset,
                'info': info_dataset}
    synchronized_dataset = SynchronizedDatasets(datasets)
    print(f"Dataset {path} loaded with {len(synchronized_dataset)} samples")
    return synchronized_dataset


def store_to_dataset_goal(synchronized_dataset, observations, grasp_pose, info, seed):
    images = [observation['color'] for observation in observations]
    camera_configs = [{'intrinsics': observation['intrinsics'], 'pose': observation['pose']} for observation
                      in observations]
    data_sample = {
        'color': images,
        'camera_config': camera_configs,
        'grasp_pose': {'grasp_pose': grasp_pose},
        'info': info
    }
    synchronized_dataset.add_sample(seed, data_sample)


def load_dataset_trajectory(n_perspectives, path):
    color_dataset = ColorDataset(name='color', n_perspectives=n_perspectives,
                                 path=path)
    camera_config_dataset = MNPZDataset(keywords=['intrinsics', 'pose'], name='camera_config',
                                        path=path,
                                        n_samples=n_perspectives)
    grasp_pose_dataset = NPZDataset(
        keywords=['grasp_pose'], name='grasp_pose', path=path)
    info_dataset = PickleDataset(name='info', path=path)
    trajectory_dataset = NPZDataset(keywords=['trajectory'], name='trajectory',
                                    path=path)
    datasets = {'color': color_dataset,
                'camera_config': camera_config_dataset,
                'grasp_pose': grasp_pose_dataset,
                'info': info_dataset,
                'trajectory': trajectory_dataset}
    synchronized_dataset = SynchronizedDatasets(datasets)
    print(f"Dataset {path} loaded with {len(synchronized_dataset)} samples")
    return synchronized_dataset


def store_to_dataset_trajectory(synchronized_dataset, observations, steps, grasp_pose, info, seed):
    images = [observation['color'] for observation in observations]
    camera_configs = [{'intrinsics': observation['intrinsics'], 'pose': observation['pose']} for observation
                      in observations]
    data_sample = {
        'color': images,
        'camera_config': camera_configs,
        'grasp_pose': {'grasp_pose': grasp_pose},
        'trajectory': {'trajectory': steps},
        'info': info
    }
    synchronized_dataset.add_sample(seed, data_sample)


def load_dataset_language(n_perspectives, path):
    color_dataset = ColorDataset(name='color', n_perspectives=n_perspectives,
                                 path=path)
    camera_config_dataset = MNPZDataset(keywords=['intrinsics', 'pose'], name='camera_config',
                                        path=path,
                                        n_samples=n_perspectives)
    grasp_pose_dataset = NPZDataset(
        keywords=['grasp_pose'], name='grasp_pose', path=path)
    info_dataset = PickleDataset(name='info', path=path)
    trajectory_dataset = NPZDataset(keywords=['trajectory'], name='trajectory',
                                    path=path)
    language_dataset = LanguageDataset(name='language', path=path)

    datasets = {'color': color_dataset,
                'camera_config': camera_config_dataset,
                'grasp_pose': grasp_pose_dataset,
                'info': info_dataset,
                'trajectory': trajectory_dataset,
                'language': language_dataset}
    synchronized_dataset = SynchronizedDatasets(datasets)
    print(f"Dataset {path} loaded with {len(synchronized_dataset)} samples")
    return synchronized_dataset


def store_to_dataset_language(synchronized_dataset, observations, steps, grasp_pose, info, language, seed):
    images = [observation['color'] for observation in observations]
    camera_configs = [{'intrinsics': observation['intrinsics'], 'pose': observation['pose']} for observation
                      in observations]
    data_sample = {
        'color': images,
        'camera_config': camera_configs,
        'grasp_pose': {'grasp_pose': grasp_pose},
        'trajectory': {'trajectory': steps},
        'info': info,
        'language': language
    }
    synchronized_dataset.add_sample(seed, data_sample)
