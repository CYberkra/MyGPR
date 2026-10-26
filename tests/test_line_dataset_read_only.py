"""Read-only sessions must reject line writes before touching any artifact."""
import numpy as np
import pytest

from core.field_project_store import FieldProjectStore
from core.gpr_data_model import GPRDataSet
from core.storage_primitives import ProjectAccessMode, ProjectReadOnlyError
from core.trajectory_model import TrajectoryModel, TrajectoryPoint


@pytest.mark.parametrize("operation", ["transpose", "save", "trajectory"])
def test_read_only_line_dataset_write_preserves_all_project_files(tmp_path, operation):
    source = tmp_path / "source.npy"
    original = np.arange(24, dtype=np.float32).reshape(8, 3)
    np.save(source, original)
    root = tmp_path / "project"
    with FieldProjectStore.create_empty(root, name="read only") as store:
        store.import_line_file("L01", source)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with FieldProjectStore.open(root, access_mode=ProjectAccessMode.READ_ONLY) as reader:
        with pytest.raises(ProjectReadOnlyError):
            if operation == "transpose":
                reader.transpose_gpr_dataset("L01")
            elif operation == "save":
                reader.save_gpr_dataset("L01", GPRDataSet.from_matrix("L01", original.T))
            else:
                reader.save_trajectory("L01", TrajectoryModel([
                    TrajectoryPoint(distance_m=0., x=1., y=2., z=3.),
                ]))
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    with FieldProjectStore.open(root, access_mode=ProjectAccessMode.READ_ONLY) as reader:
        np.testing.assert_array_equal(reader.load_gpr_dataset("L01").matrix, original)
