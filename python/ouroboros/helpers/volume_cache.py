from dataclasses import astuple
import os
import time

from cloudvolume import CloudVolume, VolumeCutout, Bbox
import numpy as np

from .bounding_boxes import BoundingBox, boxes_dim_range
from .memory_usage import calculate_gigabytes_from_dimensions
from .mem import SharedNPManager
from .shapes import NGOrder

FLUSH_CACHE = False


class VolumeCache:
    def __init__(
        self,
        bounding_boxes: list[BoundingBox],
        link_rects: list[int],
        cloud_volume_interface: "CloudVolumeInterface",
        mip=None,
        flush_cache=FLUSH_CACHE,
        use_shared: bool = False
    ) -> None:
        self.bounding_boxes = bounding_boxes

        self.link_rects = link_rects
        self.cv = cloud_volume_interface
        self.mip = mip
        self.flush_cache = flush_cache

        self.last_requested_slice = None

        # Stores the volume data for each bounding box
        self.volumes = [None] * len(bounding_boxes)

        self.use_shared = use_shared
        self.__shm_host = None

        # Indicates whether the a volume should be cached after the last slice to request it is processed
        self.cache_volume = [False] * len(bounding_boxes)
        self.cache_volume[link_rects[-1]] = VolumeCache.should_cache_last_volume(
            link_rects
        )

        self.init_cloudvolume()

    def to_dict(self) -> dict:
        return {
            "bounding_boxes": [bb.to_dict() for bb in self.bounding_boxes],
            "link_rects": self.link_rects,
            "cv": self.cv.to_dict(),
            "mip": self.mip,
            "flush_cache": self.flush_cache,
        }

    def connect_shm(self, address: str, authkey: str):
        self.__shm_host = SharedNPManager(address=address, authkey=authkey)
        self.__shm_host.connect()
        self.__authkey = authkey

    @staticmethod
    def from_dict(data: dict) -> "VolumeCache":
        bounding_boxes = [BoundingBox.from_dict(bb) for bb in data["bounding_boxes"]]
        link_rects = data["link_rects"]
        cv = CloudVolumeInterface.from_dict(data["cv"])
        mip = data["mip"]
        flush_cache = data["flush_cache"]

        return VolumeCache(bounding_boxes, link_rects, cv, mip, flush_cache)

    def init_cloudvolume(self):
        if self.mip is None:
            self.mip = min(self.cv.available_mips)

    def get_volume_gigabytes(self) -> float:
        return calculate_gigabytes_from_dimensions(
            self.get_volume_shape(), self.get_volume_dtype()
        )

    def get_volume_shape(self) -> tuple[int, ...]:
        return self.cv.get_volume_shape(self.mip)

    def has_color_channels(self) -> bool:
        return self.cv.has_color_channels

    def get_num_channels(self) -> int:
        return self.cv.num_channels

    def get_volume_dtype(self) -> np.dtype:
        return self.cv.dtype

    def get_volume_mip(self) -> int:
        return self.mip

    def set_volume_mip(self, mip: int):
        self.mip = mip

    def get_resolution_um(self):
        return self.cv.get_resolution_um(self.mip)

    @staticmethod
    def should_cache_last_volume(link_rects: list[int]):
        if link_rects[0] == link_rects[-1]:
            return True

        return False

    def volume_index(self, slice_index: int):
        return self.link_rects[slice_index]

    def request_volume_for_slice(self, slice_index: int):
        """
        Get the volume data for a slice index.

        Suitable for use in a loop that processes slices sequentially (not parallel).

        Download the volume if it is not already cached and remove the last requested volume if it is not to be cached.

        Parameters:
        ----------
            slice_index (int): The index of the slice to request.

        Returns:
        -------
            tuple: A tuple containing the volume data and the bounding box of the slice.
        """

        vol_index = self.volume_index(slice_index)
        bounding_box = self.bounding_boxes[vol_index]

        # Download the volume if it is not already cached
        if self.volumes[vol_index] is None:
            self.volumes[vol_index] = download_volume(self.cv, bounding_box, mip=self.mip)

        # Remove the last requested volume if it is not to be cached
        if (
            self.last_requested_slice is not None
            and self.last_requested_slice != vol_index
            and not self.cache_volume[self.last_requested_slice]
        ):
            self.remove_volume(self.last_requested_slice)

        self.last_requested_slice = vol_index

        return self.volumes[vol_index], bounding_box

    def remove_volume(self, volume_index: int, destroy_shared: bool = False):
        # Avoid removing the volume if it is cached for later
        if self.cache_volume[volume_index]:
            return

        if not self.use_shared:
            self.volumes[volume_index] = None
        elif destroy_shared:
            self.__shm_host.remove_termed(self.volumes[volume_index])
            self.volumes[volume_index] = None

    def get_slice_indices(self, volume_index: int):
        return [i for i, v in enumerate(self.link_rects) if v == volume_index]

    def flush_local_cache(self):
        if self.flush_cache:
            self.cv.flush_cache()


class CloudVolumeInterface:
    def __init__(self, source_url: str):
        self.source_url = source_url
        if os.environ.get('OUR_ENV') == "docker":
            self.source_url = self.source_url.replace("localhost", "host.docker.internal"
                                                      ).replace("127.0.0.1", "host.docker.internal")

        self.cv = CloudVolume(self.source_url, parallel=1, cache=True)

        self.available_mips = self.cv.available_mips
        self.dtype = self.cv.dtype

    def to_dict(self):
        return {"source_url": self.source_url}

    @staticmethod
    def from_dict(data: dict) -> "CloudVolumeInterface":
        source_url = data["source_url"]
        return CloudVolumeInterface(source_url)

    @property
    def has_color_channels(self) -> bool:
        return len(self.cv.shape) == 4

    @property
    def num_channels(self) -> int:
        return self.cv.shape[-1]

    def get_volume_shape(self, mip: int) -> tuple[int, ...]:
        return self.cv.mip_volume_size(mip)

    def get_resolution_nm(self, mip: int) -> tuple[float, ...]:
        return self.cv.mip_resolution(mip)

    def get_resolution_um(self, mip: int) -> tuple[float, ...]:
        return self.get_resolution_nm(mip) / 1000

    def flush_cache(self):
        self.cv.cache.flush()


def download_volume(
    cv: CloudVolumeInterface, bounding_box: BoundingBox, mip, parallel: int = 1,
    use_shared=False, shm_address: str = None, shm_authkey: str = None, **kwargs
) -> VolumeCutout:
    start = time.perf_counter()
    bbox = bounding_box.to_cloudvolume_bbox().astype(int)
    vol_shape = NGOrder(*bbox.size3(), cv.cv.num_channels)

    # Limit size of area we are grabbing, in case we go out of bounds.
    dl_box = Bbox.intersection(cv.cv.bounds, bbox)
    local_min = [int(start) for start in np.subtract(dl_box.minpt, bbox.minpt)]

    local_bounds = np.s_[*[slice(start, stop) for start, stop in
                           zip(local_min, np.sum([local_min, dl_box.size3()], axis=0))],
                         :]

    # Download the bounding box volume
    if use_shared:
        shm_host = SharedNPManager(address=shm_address, authkey=shm_authkey)
        shm_host.connect()
        volume = shm_host.TermedNPArray(vol_shape, np.float32)
        with volume as volume_data:
            volume_data[:] = 0     # Prob not most efficient but makes math much easier
            volume_data[local_bounds] = cv.cv.download(dl_box, mip=mip, parallel=parallel)
    else:
        volume = np.zeros(astuple(vol_shape))
        volume[local_bounds] = cv.cv.download(dl_box, mip=mip, parallel=parallel)

    # Return volume
    return volume, bounding_box, time.perf_counter() - start, *kwargs.values()


def get_mip_volume_sizes(source_url: str) -> dict:
    """
    Get the volume sizes for all available MIPs.

    Parameters:
    ----------
        source_url (str): The URL of the cloud volume.

    Returns:
    -------
        dict: A dictionary containing the volume sizes for all available MIPs.
    """

    try:
        cv = CloudVolumeInterface(source_url)
        result = {mip: cv.get_volume_shape(mip) for mip in cv.available_mips}
    except BaseException:
        return {}

    return result


def update_writable_rects(processed: np.ndarray, slice_rects: np.ndarray, min_dim: int, writeable: np.ndarray,
                          chunk_size: int):
    """
    Updates which z-stacks are writeable based on the slice rects that have been processed..

    Parameters:
    -----------
        processed (np.ndarray): Marker of which chunks are processed,
                                 by their (Z, Y, X) indicies.
        slice_rects: All full-size slice rects from the straightened volume.
        min_dim (int): Minimum (z) dimension of the full object.
        writeable (np.ndarray): Tracker of writable Z-stacks (index 0 = z min_dim).
                                Values: 0 (not writeable), 1 (writable), 2 (dispatched to writer).
        chunk_size (int): Size of 3D chunk in (z) dimension.

    Return:
    -------
        np.ndarray: Sorted values in the given dimension that ready to be written to.

    """
    # Each chunk covers part of chunk_size slice_rects in z (straightened) dimension,
    #  except last may be shorter (so capped to length of slice_rects).
    # A full slice_rect is ready if all (y, x) chunks for it are processed.
    processed_slices = np.repeat(np.all(processed, axis=(1, 2)), chunk_size)[:len(slice_rects)]
    if np.all(processed_slices):
        # All slice_rects processed, remaining z (backprojected) slices are to be written.
        writeable[:] = np.maximum(writeable, 1)
    elif np.any(processed_slices):
        # Get the z-indexes covered by processed bounding boxes.
        processed_z = boxes_dim_range([BoundingBox.from_rects(slice_rects[processed_slices])])
        remaining_z = boxes_dim_range([BoundingBox.from_rects(slice_rects[np.invert(processed_slices)])])

        # Update writable with any z-stacks that have been processed and have no chunks remaining,
        #  not overwriting any marker of those dispatched for writing.
        completed_index = processed_z[np.isin(processed_z, remaining_z, invert=True)] - min_dim
        writeable[completed_index] = np.maximum(writeable[completed_index], 1)


def update_writable_boxes(volume_cache, writeable, remaining, index):
    """
    Updates which boxes are writeable based on the current bounding box and ones remaining to be written.

    Parameters:
    -----------
        volume_cache (VolumeCache): VolumeCache holding all bounding boxes being processed.
        writeable (np.ndarray): What data is currently writeable.
        remaining (np.ndarray): Bounding boxes not yet processed.
        index:  Index in the volume cache of the current bounding box.

    Return:
    -------
        np.ndarray: Sorted values in the given dimension that ready to be written to.

    """
    current = boxes_dim_range([volume_cache.bounding_boxes[index]])
    remaining.remove(volume_cache.bounding_boxes[index])

    # Check for completed z-stacks
    writeable = np.concatenate([writeable, current[np.isin(current, boxes_dim_range(remaining), invert=True)]])

    # Sort available z-stacks
    writeable.sort()

    return writeable
