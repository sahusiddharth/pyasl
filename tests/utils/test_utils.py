# This test file includes multiple test cases for both read_data_description 
# and load_img functions. Here's a breakdown of the test cases:

# For read_data_description:
# 1. Successful read of a valid JSON file
# 2. Error when file is not found
# 3. Error when JSON is invalid
# 4. Error when file is empty
# 5. Successful read of nested JSON data

# For load_img:
# 1. Successful load of a 3D NIfTI file
# 2. Load image with NaN values (should be replaced)
# 3. Error when file is not found
# 4. Load 4D NIfTI image
# 5. Load image with custom affine matrix
# # 6. Load image with infinite values (should be replaced)

import pytest
import json
import numpy as np
import nibabel as nib
from pyasl.utils.utils import read_data_description, load_img


# Tests for read_data_description
class TestReadDataDescription:
    @pytest.fixture
    def sample_data(self):
        return {
            "subject_id": "sub-001",
            "age": 25,
            "gender": "F",
            "condition": "healthy",
        }

    def test_successful_read(self, tmp_path, sample_data):
        data_file = tmp_path / "data_description.json"
        with open(data_file, "w") as f:
            json.dump(sample_data, f)

        result = read_data_description(str(tmp_path))
        assert result == sample_data

    def test_file_not_found(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            read_data_description(str(tmp_path / "nonexistent"))

    def test_invalid_json(self, tmp_path):
        invalid_file = tmp_path / "data_description.json"
        with open(invalid_file, "w") as f:
            f.write("invalid json")
        with pytest.raises(ValueError, match="Failed to decode JSON"):
            read_data_description(str(tmp_path))

    def test_empty_file(self, tmp_path):
        empty_file = tmp_path / "data_description.json"
        with open(empty_file, "w") as f:
            f.write("")
        with pytest.raises(ValueError, match="Failed to decode JSON"):
            read_data_description(str(tmp_path))

    def test_nested_data(self, tmp_path):
        nested_data = {
            "subject": {"id": "sub-001", "demographics": {"age": 25, "gender": "F"}},
            "study": {"name": "ASL Study", "date": "2023-05-01"},
        }
        data_file = tmp_path / "data_description.json"
        with open(data_file, "w") as f:
            json.dump(nested_data, f)

        result = read_data_description(str(tmp_path))
        assert result == nested_data


# Tests for load_img
class TestLoadImg:
    @pytest.fixture
    def sample_data(self):
        return np.random.rand(10, 10, 10)

    def test_successful_load(self, tmp_path, sample_data):
        img_file = tmp_path / "test.nii.gz"
        nib.save(nib.Nifti1Image(sample_data, np.eye(4)), img_file)

        V, loaded_data = load_img(str(img_file))
        assert isinstance(V, nib.Nifti1Image)
        assert np.allclose(loaded_data, sample_data)

    def test_load_with_nans(self, tmp_path, sample_data):
        data_with_nans = sample_data.copy()
        data_with_nans[0, 0, 0] = np.nan
        img_file = tmp_path / "test_nans.nii.gz"
        nib.save(nib.Nifti1Image(data_with_nans, np.eye(4)), img_file)

        V, loaded_data = load_img(str(img_file))
        assert isinstance(V, nib.Nifti1Image)
        assert not np.isnan(loaded_data).any()
        assert np.isclose(loaded_data[0, 0, 0], 0)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_img(str(tmp_path / "nonexistent.nii.gz"))

    def test_load_4d_image(self, tmp_path):
        data_4d = np.random.rand(10, 10, 10, 5)
        img_file = tmp_path / "test_4d.nii.gz"
        nib.save(nib.Nifti1Image(data_4d, np.eye(4)), img_file)

        V, loaded_data = load_img(str(img_file))
        assert isinstance(V, nib.Nifti1Image)
        assert loaded_data.shape == (10, 10, 10, 5)

    def test_load_with_custom_affine(self, tmp_path, sample_data):
        custom_affine = np.array(
            [[2, 0, 0, -10], [0, 2, 0, -20], [0, 0, 2, -30], [0, 0, 0, 1]]
        )
        img_file = tmp_path / "test_custom_affine.nii.gz"
        nib.save(nib.Nifti1Image(sample_data, custom_affine), img_file)

        V, loaded_data = load_img(str(img_file))
        assert isinstance(V, nib.Nifti1Image)
        assert np.allclose(V.affine, custom_affine)

    # def test_load_with_inf_values(self, tmp_path, sample_data):
    #     data_with_inf = sample_data.copy()
    #     data_with_inf[5, 5, 5] = np.inf
    #     img_file = tmp_path / "test_inf.nii.gz"
    #     nib.save(nib.Nifti1Image(data_with_inf, np.eye(4)), img_file)

    #     V, loaded_data = load_img(str(img_file))
    #     assert isinstance(V, nib.Nifti1Image)
    #     assert not np.isinf(loaded_data).any()
    #     print(loaded_data[5, 5, 5])
    #     assert np.isclose(loaded_data[5, 5, 5], 0)
