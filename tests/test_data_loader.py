import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import torch

from src.data import data_loader


class FakeHFDataset:
    def __len__(self):
        return 5994

    def __getitem__(self, index):
        return {
            "image": Image.new("RGB", (16, 16), (index % 256, 64, 128)),
            "label": index % 200,
            "description": "\n".join(f"Caption {number}" for number in range(10)),
        }


class DataLoaderTests(unittest.TestCase):
    def test_loaders_use_shared_split_and_preserve_batch_contract(self):
        with patch.object(
            data_loader, "load_dataset", return_value=FakeHFDataset()
        ) as load_dataset:
            train_loader, val_loader = data_loader.create_data_loaders()

        load_dataset.assert_called_once_with(
            data_loader.DATASET_NAME, split="train"
        )
        self.assertEqual(len(train_loader.dataset), 4795)
        self.assertEqual(len(val_loader.dataset), 1198)
        self.assertEqual(len(train_loader), 150)
        self.assertEqual(len(val_loader), 38)

        split_path = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "eda"
            / "split_indices.json"
        )
        split_data = json.loads(split_path.read_text(encoding="utf-8"))
        self.assertEqual(train_loader.dataset.indices, split_data["train_indices"])
        self.assertEqual(val_loader.dataset.indices, split_data["val_indices"])

        for loader in (train_loader, val_loader):
            batch = next(iter(loader))
            self.assertEqual(batch["image"].shape, (32, 3, 224, 224))
            self.assertEqual(batch["image"].dtype, torch.float32)
            self.assertGreaterEqual(batch["image"].min().item(), 0.0)
            self.assertLessEqual(batch["image"].max().item(), 1.0)
            self.assertEqual(batch["label"].shape, (32,))
            self.assertEqual(batch["label"].dtype, torch.long)
            self.assertEqual(len(batch["captions"]), 32)
            self.assertTrue(all(len(captions) == 10 for captions in batch["captions"]))
            self.assertTrue(
                all(
                    isinstance(caption, str)
                    for captions in batch["captions"]
                    for caption in captions
                )
            )

        expected_final_sizes = {train_loader: 27, val_loader: 14}
        for loader, expected_size in expected_final_sizes.items():
            final_indices = list(loader.batch_sampler)[-1]
            final_items = [loader.dataset[index] for index in final_indices]
            final_batch = data_loader.collate_cub_batch(final_items)
            self.assertEqual(final_batch["image"].shape[0], expected_size)
            self.assertEqual(final_batch["label"].shape[0], expected_size)
            self.assertEqual(len(final_batch["captions"]), expected_size)

    def test_rejects_missing_split_keys_before_loading_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            split_path = Path(directory) / "split.json"
            split_path.write_text(json.dumps({"train_indices": [0]}), encoding="utf-8")
            with patch.object(data_loader, "load_dataset") as load_dataset:
                with self.assertRaisesRegex(ValueError, "val_indices"):
                    data_loader.create_data_loaders(split_file=split_path)
            load_dataset.assert_not_called()

    def test_rejects_overlapping_indices_before_loading_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            split_path = Path(directory) / "split.json"
            split_path.write_text(
                json.dumps({"train_indices": [1], "val_indices": [1]}),
                encoding="utf-8",
            )
            with patch.object(data_loader, "load_dataset") as load_dataset:
                with self.assertRaisesRegex(ValueError, "overlap"):
                    data_loader.create_data_loaders(split_file=split_path)
            load_dataset.assert_not_called()

    def test_rejects_index_outside_loaded_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            split_path = Path(directory) / "split.json"
            split_path.write_text(
                json.dumps({"train_indices": [5994], "val_indices": [0]}),
                encoding="utf-8",
            )
            with patch.object(
                data_loader, "load_dataset", return_value=FakeHFDataset()
            ):
                with self.assertRaisesRegex(ValueError, "outside"):
                    data_loader.create_data_loaders(split_file=split_path)


if __name__ == "__main__":
    unittest.main()
