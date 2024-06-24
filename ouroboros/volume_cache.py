from .bounding_boxes import BoundingBox

from cloudvolume import CloudVolume, VolumeCutout

FLUSH_CACHE = False

# TODO: Progress hook of some kind

class VolumeCache:
    def __init__(self, bounding_boxes: list[BoundingBox], link_rects: list[int], source_url: str, mip=None, flush_cache=FLUSH_CACHE) -> None:
        self.bounding_boxes = bounding_boxes
        self.link_rects = link_rects
        self.source_url = source_url
        self.mip = mip
        self.flush_cache = flush_cache

        self.last_requested_slice = None

        # Stores the volume data for each bounding box
        self.volumes = [None] * len(bounding_boxes)

        # Indicates whether the a volume should be cached after the last slice to request it is processed
        self.cache_volume = [False] * len(bounding_boxes)
        self.cache_volume[link_rects[-1]] = VolumeCache.should_cache_last_volume(link_rects)

        self.init_cloudvolume()

    def init_cloudvolume(self):
        self.cv = CloudVolume(self.source_url, parallel=True, cache=True)

        available_mips = self.cv.available_mips

        if self.mip is None:
            self.mip = min(available_mips)

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
            self.download_volume(vol_index, bounding_box)

        # Remove the last requested volume if it is not to be cached
        if self.last_requested_slice is not None and self.last_requested_slice != vol_index and \
                not self.cache_volume[self.last_requested_slice]:
            self.remove_volume(self.last_requested_slice)

        self.last_requested_slice = vol_index

        return self.volumes[vol_index], bounding_box
    
    def remove_volume(self, volume_index: int):
        # Avoid removing the volume if it is cached for later
        if self.cache_volume[volume_index]:
            return

        self.volumes[volume_index] = None

    def download_volume(self, volume_index: int, bounding_box: BoundingBox) -> VolumeCutout:
        # TODO: Handle errors that occur here

        bbox = bounding_box.to_cloudvolume_bbox()

        # Download the bounding box volume
        volume = self.cv.download(bbox, mip=self.mip, parallel=False)

        # Store the volume in the cache
        self.volumes[volume_index] = volume

    def create_processing_data(self, volume_index: int):
        """
        Generate a data packet for processing a volume.

        Suitable for parallel processing.

        Parameters:
        ----------
            volume_index (int): The index of the volume to process.

        Returns:
        -------
            tuple: A tuple containing the volume data, the bounding box of the volume, 
                    the slice indices associated with the volume, and a function to remove the volume from the cache.
        """

        bounding_box = self.bounding_boxes[volume_index]

        # Download the volume if it is not already cached
        if self.volumes[volume_index] is None:
            self.download_volume(volume_index, bounding_box)

        # Get all slice indices associated with this volume
        slice_indices = [i for i, v in enumerate(self.link_rects) if v == volume_index]

        return self.volumes[volume_index], bounding_box, slice_indices, volume_index

    def __del__(self):
        if self.flush_cache:
            self.cv.cache.flush()