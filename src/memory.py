import os
import cv2


class CropMemory:
    """Stores unique object crops once per track identity.

    Every unique object (pedestrian, vehicle or license plate region)
    identified by a persistent ``track_id`` is extracted from the frame
    and persisted to disk exactly once — duplicate saves across
    subsequent frames are prevented by internal registries of already
    saved track ids. Face crops are derived from the top 40% of the
    pedestrian box (expanded 15% outward on each side) and are gated on
    consecutive-frame track stability; plate crops come from the
    bottom-center region of enclosed vehicles.
    """

    ENCLOSED_VEHICLE_CLASSES = ('car', 'bus', 'truck')

    def __init__(self, root_dir="data/outputs", faces_subdir="crops/faces",
                 vehicles_subdir="crops/vehicles", plates_subdir="crops/plates"):
        """Initialize the crop memory store.

        Args:
            root_dir: Base output directory (project-relative).
            faces_subdir: Sub-path (relative to ``root_dir``) where face
                crops are persisted.
            vehicles_subdir: Sub-path (relative to ``root_dir``) where
                vehicle crops are persisted.
            plates_subdir: Sub-path (relative to ``root_dir``) where
                license plate region crops are persisted.
        """
        self.root_dir = root_dir
        self.faces_dir = os.path.join(root_dir, faces_subdir)
        self.vehicles_dir = os.path.join(root_dir, vehicles_subdir)
        self.plates_dir = os.path.join(root_dir, plates_subdir)
        for directory in (self.faces_dir, self.vehicles_dir, self.plates_dir):
            os.makedirs(directory, exist_ok=True)
        self._saved_face_ids = set()
        self._saved_vehicle_ids = set()
        self._saved_plate_ids = set()
        self._face_frame_counts = {}

    def compute_head_region(self, bbox, head_fraction=0.40, width_expansion=0.15):
        """Compute the head/shoulder region of a pedestrian bounding box.

        The region spans the box width expanded outward by
        ``width_expansion`` (15% per side by default) and the top
        ``head_fraction`` of the box height (default top 40% captures the
        full facial geometry including chin and mouth). Out-of-frame
        expansion is clamped to the frame borders at save time.

        Args:
            bbox: (x1, y1, x2, y2) pedestrian bounding box.
            head_fraction: Fraction of the box height used for the crop.
            width_expansion: Fraction of the box width added on each side.

        Returns:
            Tuple (x1, y1, x2, y2) describing the head region.
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        expand = width * width_expansion
        head_y2 = y1 + height * head_fraction
        return (x1 - expand, y1, x2 + expand, head_y2)

    def compute_plate_region(self, bbox, bottom_fraction=0.35, center_fraction=0.60):
        """Compute the license plate region of a vehicle bounding box.

        Heuristic crop targeting where plates are typically mounted on
        enclosed vehicles: the bottom ``bottom_fraction`` of the box
        height, horizontally centered over the middle ``center_fraction``
        of the box width.

        Args:
            bbox: (x1, y1, x2, y2) vehicle bounding box.
            bottom_fraction: Fraction of the box height taken from the bottom.
            center_fraction: Fraction of the box width kept around the center.

        Returns:
            Tuple (x1, y1, x2, y2) describing the plate region.
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2.0
        half_span = width * center_fraction / 2.0
        return (center_x - half_span, y2 - height * bottom_fraction,
                center_x + half_span, y2)

    def _clamp_region(self, region, frame_shape):
        """Clamp a crop region to the frame bounds.

        Args:
            region: (x1, y1, x2, y2) crop region.
            frame_shape: Shape of the source frame.

        Returns:
            Clamped integer tuple (x1, y1, x2, y2).
        """
        x1, y1, x2, y2 = region
        h, w = frame_shape[:2]
        return (max(0, int(x1)), max(0, int(y1)),
                min(w, int(x2)), min(h, int(y2)))

    def _save_crop_once(self, frame, region, directory, filename,
                        track_id, saved_ids):
        """Persist a single crop region, exactly once per track id.

        Args:
            frame: Current video frame (BGR).
            region: (x1, y1, x2, y2) crop region.
            directory: Destination directory for the crop.
            filename: File name of the crop image.
            track_id: Unique tracking identifier of the object.
            saved_ids: Registry of track ids already saved to ``directory``.

        Returns:
            Crop path (relative to ``root_dir``) when the crop was saved
            for this ``track_id`` for the first time, otherwise None.
        """
        if track_id is None or track_id in saved_ids:
            return None

        cx1, cy1, cx2, cy2 = self._clamp_region(region, frame.shape)
        if cx2 <= cx1 or cy2 <= cy1:
            return None

        crop = frame[cy1:cy2, cx1:cx2]
        abs_path = os.path.join(directory, filename)
        cv2.imwrite(abs_path, crop)
        saved_ids.add(track_id)

        rel_path = os.path.join(os.path.relpath(directory, self.root_dir), filename)
        return rel_path

    def save_face_crop(self, frame, bbox, track_id, head_fraction=0.40,
                       width_expansion=0.15, min_stable_frames=5):
        """Save the face/head crop for a stable unique pedestrian, exactly once.

        The crop is only triggered once the ``track_id`` has been observed
        for at least ``min_stable_frames`` consecutive frames, so
        detections early in a track (low confidence or unstable boxes)
        never produce a crop. Each unique ``track_id`` is still saved
        exactly once.

        Args:
            frame: Current video frame (BGR).
            bbox: (x1, y1, x2, y2) pedestrian bounding box.
            track_id: Unique tracking identifier of the pedestrian.
            head_fraction: Fraction of the box height used for the crop.
            width_expansion: Fraction of the box width added on each side.
            min_stable_frames: Minimum consecutive frames the track must
                be observed for before the crop is saved.

        Returns:
            Crop path (relative to ``root_dir``) when the crop was saved
            for this ``track_id`` for the first time, otherwise None.
        """
        if track_id is None or track_id in self._saved_face_ids:
            return None

        count = self._face_frame_counts.get(track_id, 0) + 1
        self._face_frame_counts[track_id] = count
        if count < min_stable_frames:
            return None

        region = self.compute_head_region(bbox, head_fraction, width_expansion)
        filename = f"pedestrian_{track_id}.jpg"
        return self._save_crop_once(frame, region, self.faces_dir, filename,
                                    track_id, self._saved_face_ids)

    def mark_frame_observations(self, observed_track_ids):
        """Reset consecutive observation streaks for absent face tracks.

        Track ids present in the current frame keep their accumulated
        streak (incremented by ``save_face_crop``); ids absent from the
        current frame lose their streak entirely. This enforces strict
        consecutive-frame stability semantics and guards against
        ByteTrack id reuse for unrelated objects.

        Args:
            observed_track_ids: Iterable of person track ids detected in
                the current frame.
        """
        observed = set(observed_track_ids)
        for track_id in list(self._face_frame_counts.keys()):
            if track_id not in observed:
                del self._face_frame_counts[track_id]

    def save_vehicle_crop(self, frame, bbox, track_id, vehicle_class):
        """Save the full vehicle crop, exactly once per unique track id.

        Args:
            frame: Current video frame (BGR).
            bbox: (x1, y1, x2, y2) vehicle bounding box.
            track_id: Unique tracking identifier of the vehicle.
            vehicle_class: Detected vehicle class (car, bus, truck,
                motorcycle or bicycle).

        Returns:
            Crop path (relative to ``root_dir``) when the crop was saved
            for this ``track_id`` for the first time, otherwise None.
        """
        if track_id is None or track_id in self._saved_vehicle_ids:
            return None
        filename = f"{vehicle_class}_{track_id}.jpg"
        return self._save_crop_once(frame, bbox, self.vehicles_dir, filename,
                                    track_id, self._saved_vehicle_ids)

    def save_plate_crop(self, frame, bbox, track_id, vehicle_class=None,
                        bottom_fraction=0.35, center_fraction=0.60):
        """Save the license plate region for a unique vehicle, exactly once.

        Only enclosed vehicle classes (car, bus, truck) are eligible for
        plate regions; open micro-mobility classes return None immediately.

        Args:
            frame: Current video frame (BGR).
            bbox: (x1, y1, x2, y2) vehicle bounding box.
            track_id: Unique tracking identifier of the vehicle.
            vehicle_class: Detected vehicle class (validated against
                ``ENCLOSED_VEHICLE_CLASSES``).
            bottom_fraction: Fraction of the box height taken from the bottom.
            center_fraction: Fraction of the box width kept around the center.

        Returns:
            Crop path (relative to ``root_dir``) when the crop was saved
            for this ``track_id`` for the first time, otherwise None.
        """
        if vehicle_class is not None and vehicle_class not in self.ENCLOSED_VEHICLE_CLASSES:
            return None
        if track_id is None or track_id in self._saved_plate_ids:
            return None
        region = self.compute_plate_region(bbox, bottom_fraction, center_fraction)
        filename = f"plate_{track_id}.jpg"
        return self._save_crop_once(frame, region, self.plates_dir, filename,
                                    track_id, self._saved_plate_ids)

    def summary(self):
        """Report crop memory statistics.

        Returns:
            Dictionary with the number of unique crops persisted per
            category and the associated output directories.
        """
        return {
            'unique_faces_saved': len(self._saved_face_ids),
            'unique_vehicles_saved': len(self._saved_vehicle_ids),
            'unique_plates_saved': len(self._saved_plate_ids),
            'faces_dir': self.faces_dir,
            'vehicles_dir': self.vehicles_dir,
            'plates_dir': self.plates_dir
        }
