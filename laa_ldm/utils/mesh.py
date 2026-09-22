"""Volume post-processing: isosurfacing and connected-component cleanup."""

import numpy as np
import SimpleITK as sitk
import trimesh
from scipy.ndimage import label
from skimage import measure

__all__ = ["volume_to_mesh", "save_mesh", "largest_connected_component", "save_nifti"]


def volume_to_mesh(volume, level=0.5):
    """Marching-cubes surface of ``volume`` as a :class:`trimesh.Trimesh`.

    Vertices are in voxel index coordinates; multiply by the voxel size to get
    millimetres.
    """
    verts, faces, _, _ = measure.marching_cubes(np.asarray(volume, dtype=np.float32), level=level)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def save_mesh(volume, path, level=0.5):
    """Mesh ``volume`` and write it to ``path``; on failure write a ``.txt`` next to it.

    Marching cubes fails for degenerate samples (e.g. an all-background volume),
    which must not abort a long generation run.
    """
    try:
        volume_to_mesh(volume, level=level).export(str(path))
        return True
    except Exception as exc:  # noqa: BLE001 - reported per sample, never fatal
        with open("{}_meshing_error.txt".format(path), "w") as f:
            f.write(str(exc))
        return False


def largest_connected_component(volume):
    """Keep only the largest 26-connected foreground component of a binary volume."""
    structure = np.ones((3, 3, 3), dtype=np.int8)
    labeled, num_features = label(volume, structure=structure)
    if num_features == 0:
        return volume
    component_sizes = np.bincount(labeled.flat)
    component_sizes[0] = 0  # background
    return (labeled == component_sizes.argmax()).astype(np.float32)


def save_nifti(arr, out_path, like_path=None, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
               direction=None):
    """Write a numpy volume (z, y, x) as a NIfTI image."""
    img = sitk.GetImageFromArray(arr)
    if like_path is not None:
        img.CopyInformation(sitk.ReadImage(str(like_path)))
    else:
        img.SetSpacing(tuple(map(float, spacing)))
        img.SetOrigin(tuple(map(float, origin)))
        if direction is not None:
            img.SetDirection(tuple(map(float, direction)))
    sitk.WriteImage(sitk.Cast(img, sitk.sitkFloat32), str(out_path))
