from typing import Union
from pathlib import Path
from dataclasses import dataclass
import uuid
import pickle
from functools import partialmethod
import torch
from datetime import datetime
import pandas as pd


def _create_unique_subpath(path, suffix="", dir=True):
    while True:
        try:
            new_path = Path(path) / Path(str(uuid.uuid4()) + suffix)
            if dir is True:
                new_path.mkdir()
            else:
                new_path.touch(exist_ok=False)
            break
        except FileExistsError:
            pass
    return new_path


Path.create_unique_dir = partialmethod(_create_unique_subpath, dir=True)
Path.create_unique_file = partialmethod(_create_unique_subpath, dir=False)

class FileLogger:

    def __init__(self, parent_dir: Union[str, Path]) -> None:
        self.parent_dir = Path(parent_dir)
        self.dir = self.parent_dir.create_unique_dir()

    def save(self, **kwargs) -> None:
        for k, v in kwargs.items():
            filename = self.dir / Path(k + ".pkl")
            with open(filename, "wb") as f:
                pickle.dump(v, f)

    def group(self, name: str) -> Path:
        group_dir = self.dir / Path(name)
        group_dir.mkdir(exist_ok=True)
        return group_dir

class FileLogReader:

    def __init__(self, dir: Union[str, Path]):
        self.dir = Path(dir)
        self.models = dict()
        self.files = dict()
        self.subdirs = dict()
        self.creation_time = datetime.fromtimestamp(self.dir.lstat().st_ctime)
        self.modification_time = datetime.fromtimestamp(self.dir.lstat().st_mtime)

    @property
    def shortname(self):
        return self.dir.stem.split('-')[0]

    def preload_attributes(self, depth=1):
        for path in self.dir.iterdir():
            if path.is_file():
                if path.suffix == ".pkl":
                    self._load_attribute_from_file(path)
                if path.suffix == ".pytorch_model":
                    self._load_model_from_file(path)
            else:
                self._load_subdir(path)
        if depth > 1:
            for subdir in self.subdirs.values():
                subdir.preload_attributes(depth-1)

    def list_subdirs(self, shortnames = True):
        subdirs_list = []
        for subdir in self.subdirs.values():
            subdirs_list.append([subdir, subdir.creation_time, subdir.modification_time])
        return pd.DataFrame(subdirs_list, columns=['subdir', 'creation', 'modification'])

    # def _set_if_not_exists(self, name, value):
    #     try:
    #         self.__getattribute__(name)
    #     except AttributeError:
    #         self.__setattr__(name, value)

    def _load_attribute_from_file(self, filepath: Path):
        with open(filepath, "rb") as f:
            loaded_object = pickle.load(f)
            # self._set_if_not_exists(filepath.stem, loaded_object)
            self.files[filepath.stem] = loaded_object

    def _load_model_from_file(self, filepath: Path, map_location="cpu"):
        loaded_model = torch.load(filepath, map_location=map_location)
        # self._set_if_not_exists(filepath.stem, loaded_model)
        self.models[filepath.stem] = loaded_model

    def _load_subdir(self, dirpath: Path):
        subdir_reader = FileLogReader(dirpath)
        # self._set_if_not_exists(subdir_reader.shortname, subdir_reader)
        self.subdirs[dirpath.stem] = subdir_reader

    # def __getattr__(self, item: str):
    #     if item not in self._attributes:
    #         attribute_filename = self.dir / Path(item + ".pkl")
    #         model_filename = self.dir / Path(item + ".pytorch_model")
    #         dirname = self.dir / Path(item)
    #         if attribute_filename.exists():
    #             self._load_attribute_from_file(attribute_filename)
    #         elif model_filename.exists():
    #             self._load_model_from_file(model_filename)
    #         elif dirname.exists():
    #             self._load_subdir(dirname)
    #         else:
    #             raise AttributeError
    #     return self._attributes[item]

    def __repr__(self):
        return self.shortname


