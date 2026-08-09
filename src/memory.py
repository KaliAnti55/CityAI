import os
import re
import threading
from datetime import datetime
import cv2


class CropMemory:
    """Stores unique object crops once per track identity.

    Every unique object (pedestrian, vehicle or license plate region)
    identified by a persistent ``track_id`` is extracted from the frame
    and persisted to disk exactly once — duplicate saves across
    subsequent frames are prevented by internal registries of already
    saved track ids. Face crops are derived from the top 40% of the
    pedestrian box (expanded 15% outward on each side) and are gated on
    consecutive-frame track stability; plate crops come from a secondary
    two-stage license plate model (with a geometric bottom-center
    fallback when no plate model is available).
    """

    ENCLOSED_VEHICLE_CLASSES = ('car', 'bus', 'truck')

    def __init__(self, root_dir="data/outputs", faces_subdir="crops/faces",
                 vehicles_subdir="crops/vehicles", plates_subdir="crops/plates",
                 plate_model=None, plate_conf_thresh=0.40, plate_min_aspect=1.5,
                 ocr_reader=None, ocr_conf_thresh=0.35,
                 target_plates=None, telegram_token=None, telegram_chat_id=None,
                 gemini_api_key=None, entry_line_y=0.5, crossing_cooldown_frames=30,
                 loitering_threshold=10):
        """Initialize the crop memory store.

        Args:
            root_dir: Base output directory (project-relative).
            faces_subdir: Sub-path (relative to ``root_dir``) where face
                crops are persisted.
            vehicles_subdir: Sub-path (relative to ``root_dir``) where
                vehicle crops are persisted.
            plates_subdir: Sub-path (relative to ``root_dir``) where
                license plate region crops are persisted.
            plate_model: Optional secondary license plate YOLO model —
                either a weights file path or an already-loaded YOLO
                instance. When None or unloadable, the geometric plate
                heuristic is used instead.
            plate_conf_thresh: Minimum confidence for plate detections.
            plate_min_aspect: Minimum width/height ratio for plate boxes.
            ocr_reader: Optional OCR reader instance exposing an
                easyocr-style ``readtext`` API. When None, easyocr is
                auto-initialized on first use with a pytesseract fallback.
            ocr_conf_thresh: Minimum easyocr confidence for plate text.
            target_plates: Optional iterable of hotlist license plate
                strings (case and separator insensitive) to alert on.
            telegram_token: Telegram Bot Token used for hotlist alerts.
            telegram_chat_id: Telegram target Chat ID for hotlist alerts.
            gemini_api_key: Optional Google Gemini API key enabling
                Gemini Vision API OCR for plate text (falls back to the
                local OCR backend when the API call fails or no key is
                provided).
            entry_line_y: Vertical position of the virtual ENTRY/EXIT
                crossing line as a fraction of the frame height (default
                0.5 = middle of the frame).
            crossing_cooldown_frames: Minimum frames between crossing
                events logged for the same ``track_id`` to suppress
                duplicates.
            loitering_threshold: Dwell time in seconds after which an
                employee is flagged as loitering and a loitering event
                is recorded (default 10).
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
        self.plate_conf_thresh = plate_conf_thresh
        self.plate_min_aspect = plate_min_aspect
        self._plate_best_conf = {}
        self.plate_model = self._load_plate_model(plate_model)
        self.ocr_reader = ocr_reader
        self.ocr_conf_thresh = ocr_conf_thresh
        self._ocr_backend = None
        self._plate_texts = {}
        self.gemini_api_key = gemini_api_key
        self._employee_records = {}
        self._employee_first_epochs = {}
        self.entry_line_y = entry_line_y
        self.crossing_cooldown_frames = crossing_cooldown_frames
        self.attendance_log = []
        self.total_entries = 0
        self.total_exits = 0
        self.current_occupancy = 0
        self._crossing_states = {}
        self.loitering_threshold = loitering_threshold
        self.loitering_events = []
        self.target_plates = set()
        if target_plates:
            for target in target_plates:
                cleaned = self._clean_plate_text(target)
                if cleaned:
                    self.target_plates.add(cleaned)
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.telegram_enabled = bool(telegram_token and telegram_chat_id)
        self._hotlist_matched_ids = set()
        self._http_requests = None
        self._http_urllib3 = None
        try:
            import requests
            self._http_requests = requests
        except ImportError:
            try:
                import urllib3
                self._http_urllib3 = urllib3
            except ImportError:
                pass

    @staticmethod
    def _load_plate_model(plate_model):
        """Load the secondary license plate detection model.

        Accepts a weights file path or an already-loaded YOLO instance.
        Returns None when nothing is provided or loading fails, enabling
        the caller to fall back gracefully to the geometric heuristic.

        Args:
            plate_model: Weights path or YOLO instance, or None.

        Returns:
            Loaded YOLO model, or None when unavailable.
        """
        if plate_model is None:
            return None
        try:
            if isinstance(plate_model, str):
                if not os.path.isfile(plate_model):
                    return None
                from ultralytics import YOLO
                return YOLO(plate_model)
            return plate_model
        except Exception:
            return None

    def _get_ocr_backend(self):
        """Resolve the OCR backend, initializing it lazily on first use.

        An explicitly provided reader instance (``ocr_reader``) is used
        as-is. Otherwise easyocr is attempted first, then the lightweight
        pytesseract fallback, and finally OCR is disabled. Load failures
        never raise — the pipeline degrades to ``plate_text`` = None.

        Returns:
            Backend name ('easyocr', 'pytesseract') or 'disabled'.
        """
        if self._ocr_backend is not None:
            return self._ocr_backend
        if self.ocr_reader is not None:
            self._ocr_backend = 'easyocr'
            return self._ocr_backend
        try:
            from easyocr import Reader
            self.ocr_reader = Reader(['en'], verbose=False)
            self._ocr_backend = 'easyocr'
        except Exception:
            self.ocr_reader = None
            try:
                import pytesseract  # noqa: F401
                self._ocr_backend = 'pytesseract'
            except Exception:
                self._ocr_backend = 'disabled'
        return self._ocr_backend

    @staticmethod
    def _clean_plate_text(text):
        """Normalize OCR output to an uppercase alphanumeric string.

        Strips all non-alphanumeric characters (spaces, dashes, dots,
        symbols). Returns None when the cleaned string is empty.

        Args:
            text: Raw OCR output string.

        Returns:
            Uppercase alphanumeric plate text, or None when empty.
        """
        if not text:
            return None
        cleaned = re.sub(r'[^A-Za-z0-9]', '', text).upper()
        return cleaned if cleaned else None

    @staticmethod
    def _ocr_with_gemini(crop_path, api_key):
        """Run Gemini Vision API OCR on a saved plate crop image.

        Sends the image to the ``gemini-3.5-flash-lite`` model via the REST
        generateContent endpoint using a strict plate-extraction prompt.
        Any API failure (network drop, bad key, quota) returns None so
        the caller can fall back to the local OCR backend.

        Args:
            crop_path: Path of the saved plate crop image.
            api_key: Google Gemini API key.

        Returns:
            Raw extracted text, or None on failure.
        """
        try:
            import base64
            import requests
            with open(crop_path, 'rb') as image_file:
                image_b64 = base64.b64encode(image_file.read()).decode('ascii')
            mime_type = 'image/png' if str(crop_path).lower().endswith('.png') else 'image/jpeg'
            url = ("https://generativelanguage.googleapis.com/v1beta/models/"
                   f"gemini-3.5-flash-lite:generateContent?key={api_key}")
            prompt = ("Extract ONLY the license plate alphanumeric characters "
                      "from this image. Return just the uppercase characters "
                      "without spaces, punctuation, or extra words.")
            payload = {
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                        {"text": prompt}
                    ]
                }]
            }
            response = requests.post(url, json=payload, timeout=20)
            response.raise_for_status()
            data = response.json()
            candidates = data.get('candidates') or []
            if not candidates:
                return None
            parts = candidates[0].get('content', {}).get('parts') or []
            if not parts:
                return None
            return parts[0].get('text')
        except Exception:
            return None

    def read_plate_text(self, plate_image):
        """Recognize and clean text from a plate crop image or path.

        Prefers the Gemini Vision API when a ``gemini_api_key`` is
        configured (falling back to the local OCR backend if the API
        call fails). Without a key, uses easyocr, then the lightweight
        pytesseract fallback. For easyocr, only the highest-confidence
        recognition is kept and gated by ``ocr_conf_thresh``; any OCR
        failure yields None.

        Args:
            plate_image: Plate crop image (BGR array) or saved file path.

        Returns:
            Cleaned uppercase alphanumeric plate text, or None.
        """
        if self.gemini_api_key and isinstance(plate_image, str) and os.path.isfile(plate_image):
            text = self._ocr_with_gemini(plate_image, self.gemini_api_key)
            if text is not None:
                return self._clean_plate_text(text)
        backend = self._get_ocr_backend()
        if backend == 'disabled':
            return None
        try:
            if backend == 'easyocr':
                results = self.ocr_reader.readtext(plate_image)
                if not results:
                    return None
                best = max(results, key=lambda r: r[2])
                if best[2] < self.ocr_conf_thresh:
                    return None
                text = best[1]
            else:
                import pytesseract
                text = pytesseract.image_to_string(plate_image, config='--psm 7')
        except Exception:
            return None
        return self._clean_plate_text(text)

    def _store_plate_text(self, crop_path, track_id):
        """Run OCR on a saved plate crop and record the text for the track.

        Uses the Gemini Vision API when a key is configured (falling back
        to the local OCR backend on failure); otherwise the local backend
        directly. The recorded value is the cleaned plate text, or None
        when OCR is unavailable or fails for this crop.

        Args:
            crop_path: Path of the saved plate crop image.
            track_id: Unique tracking identifier of the vehicle.

        Returns:
            The recorded (cleaned) plate text, or None.
        """
        text = self.read_plate_text(crop_path)
        self._plate_texts[track_id] = text
        return text

    def get_plate_text(self, track_id):
        """Return the last recognized plate text for a vehicle track.

        Args:
            track_id: Unique tracking identifier of the vehicle.

        Returns:
            Cleaned plate text, or None when OCR is unavailable, failed,
            or no plate has been recognized for this track.
        """
        return self._plate_texts.get(track_id)

    def is_hotlist_match(self, track_id):
        """Whether a vehicle track has matched a hotlist plate.

        Args:
            track_id: Unique tracking identifier of the vehicle.

        Returns:
            True once a target plate has been recognized for the track.
        """
        return track_id in self._hotlist_matched_ids

    def _is_hotlist_match(self, plate_text):
        """Check a cleaned plate text against the hotlist targets.

        Substring semantics are used so partial or noisy OCR output still
        matches (e.g., target '1234' matches text 'ABC1234').
        """
        if not plate_text:
            return False
        return any(target in plate_text for target in self.target_plates)

    def _check_hotlist(self, plate_text, track_id, image_path):
        """Trigger the once-per-track alert for hotlist plate matches.

        Marks the track as matched (driving visualizer highlighting and
        telemetry) and dispatches the Telegram alert on the first match
        for that track only.
        """
        if plate_text is None or track_id in self._hotlist_matched_ids:
            return
        if self._is_hotlist_match(plate_text):
            self._hotlist_matched_ids.add(track_id)
            self._send_telegram_alert(image_path, plate_text, track_id)

    def _send_telegram_alert(self, image_path, plate_text, track_id):
        """Dispatch a hotlist alert to Telegram without blocking the pipeline.

        Sends the incident plate image with an HTML-formatted caption to
        the configured chat via the sendPhoto API. Runs in a daemon
        thread; failures (network drop, invalid token, missing image)
        print a warning and never interrupt video processing.

        Args:
            image_path: Absolute path of the plate crop image to attach.
            plate_text: Recognized hotlist plate text.
            track_id: Unique tracking identifier of the matched vehicle.
        """
        if not self.telegram_enabled or not image_path or not os.path.isfile(image_path):
            return
        threading.Thread(target=self._telegram_alert_worker,
                         args=(image_path, plate_text, track_id),
                         daemon=True).start()

    def _telegram_alert_worker(self, image_path, plate_text, track_id):
        """Worker body for the Telegram alert (runs in a daemon thread).

        Attaches the plate crop image to a sendPhoto request with the
        hotlist message caption; any failure is reported and swallowed.
        """
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
            caption = (
                "🚨 <b>HOTLIST TARGET DETECTED</b>\n"
                f"<b>Plate:</b> <code>{plate_text}</code>\n"
                f"<b>Track ID:</b> #{track_id}"
            )
            with open(image_path, 'rb') as image_file:
                if self._http_requests is not None:
                    self._http_requests.post(
                        url,
                        data={'chat_id': self.telegram_chat_id,
                              'caption': caption, 'parse_mode': 'HTML'},
                        files={'photo': image_file}, timeout=10)
                elif self._http_urllib3 is not None:
                    self._http_urllib3.request(
                        'POST', url,
                        fields={'chat_id': self.telegram_chat_id,
                                'caption': caption, 'parse_mode': 'HTML',
                                'photo': (os.path.basename(image_path),
                                          image_file.read(), 'image/jpeg')})
                else:
                    print("[-] Telegram alert skipped: no HTTP client available")
        except Exception as exc:
            print(f"[-] Telegram alert failed (track #{track_id}): {exc}")

    def track_employee(self, track_id, timestamp_epoch, frame_index=None):
        """Update the employee access telemetry record for a person track.

        Creates the record on first sighting (including the
        ``zone_access_flag`` placeholder for future spatial restriction
        checks, and ``is_loitering`` for dwell-time alerting) and
        refreshes ``last_seen_timestamp`` and ``dwell_time_seconds`` on
        every subsequent observation. When an employee's dwell time
        exceeds ``loitering_threshold`` seconds, the record is marked
        with ``is_loitering = True`` and a single loitering event is
        appended to ``loitering_events`` (once per unique track ID).

        Args:
            track_id: Unique tracking identifier of the person.
            timestamp_epoch: Observation time as epoch seconds.
            frame_index: Optional frame number used in the loitering
                event log for traceability.
        """
        record = self._employee_records.get(track_id)
        if record is None:
            self._employee_records[track_id] = {
                'employee_track_id': track_id,
                'first_seen_timestamp': datetime.fromtimestamp(timestamp_epoch).isoformat(),
                'last_seen_timestamp': datetime.fromtimestamp(timestamp_epoch).isoformat(),
                'dwell_time_seconds': 0.0,
                'zone_access_flag': False,
                'is_loitering': False
            }
            self._employee_first_epochs[track_id] = timestamp_epoch
            return
        record['last_seen_timestamp'] = datetime.fromtimestamp(timestamp_epoch).isoformat()
        record['dwell_time_seconds'] = round(
            timestamp_epoch - self._employee_first_epochs[track_id], 2)
        if record.get('is_loitering'):
            return
        if record['dwell_time_seconds'] > self.loitering_threshold:
            record['is_loitering'] = True
            self.loitering_events.append({
                'track_id': track_id,
                'dwell_time_seconds': record['dwell_time_seconds'],
                'first_seen': record['first_seen_timestamp'],
                'timestamp': datetime.fromtimestamp(timestamp_epoch).isoformat(),
                'frame_index': frame_index
            })

    def get_employee_info(self, track_id):
        """Return the access telemetry record for one person track.

        Args:
            track_id: Unique tracking identifier of the person.

        Returns:
            Employee telemetry dictionary, or None when unknown.
        """
        return self._employee_records.get(track_id)

    def get_employee_records(self):
        """Return the aggregated access telemetry for all tracked employees.

        Returns:
            List of employee telemetry records (one per unique track id).
        """
        return list(self._employee_records.values())

    def update_employee_crossing(self, track_id, bbox, frame_height, frame_index):
        """Log ENTRY/EXIT crossing events and maintain occupancy counters.

        Compares the current bounding-box centroid (``cy = y1 + (y2 - y1)/2``)
        against the centroid from the previous observed frame, relative to
        the entry line at ``Y = frame_height * entry_line_y``:
        - a top-to-bottom crossing triggers ``ENTRY``;
        - a bottom-to-top crossing triggers ``EXIT``.
        A per-track cooldown suppresses duplicate events.

        Args:
            track_id: Unique tracking identifier of the person.
            bbox: (x1, y1, x2, y2) person bounding box.
            frame_height: Height of the current frame in pixels.
            frame_index: Global frame index used for event logging.
        """
        cy = bbox[1] + (bbox[3] - bbox[1]) / 2.0
        line_y = frame_height * self.entry_line_y

        state = self._crossing_states.get(track_id)
        if state is None:
            self._crossing_states[track_id] = {'frame': frame_index, 'cy': cy}
            return

        prev_cy = state['cy']
        state['cy'] = cy
        state['frame'] = frame_index

        if prev_cy < line_y and cy >= line_y:
            event_type = 'ENTRY'
        elif prev_cy >= line_y and cy < line_y:
            event_type = 'EXIT'
        else:
            return

        last_event_frame = state.get('last_event_frame')
        if last_event_frame is not None and \
                (frame_index - last_event_frame) < self.crossing_cooldown_frames:
            return

        state['last_event_frame'] = frame_index
        if event_type == 'ENTRY':
            self.total_entries += 1
            self.current_occupancy += 1
        else:
            self.total_exits += 1
            self.current_occupancy -= 1

        self.attendance_log.append({
            'track_id': track_id,
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'frame_index': frame_index
        })

    def get_occupancy_state(self):
        """Return the live entry/exit occupancy counters.

        Returns:
            Dictionary with ``total_entries``, ``total_exits`` and
            ``current_occupancy`` (entries minus exits).
        """
        return {
            'total_entries': self.total_entries,
            'total_exits': self.total_exits,
            'current_occupancy': self.current_occupancy
        }

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

    def _extract_crop(self, frame, region):
        """Extract a clamped crop region from the frame.

        Args:
            frame: Current video frame (BGR).
            region: (x1, y1, x2, y2) crop region.

        Returns:
            Crop image array, or None when the region is degenerate.
        """
        cx1, cy1, cx2, cy2 = self._clamp_region(region, frame.shape)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return frame[cy1:cy2, cx1:cx2]

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

        crop = self._extract_crop(frame, region)
        if crop is None:
            return None

        cv2.imwrite(os.path.join(directory, filename), crop)
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
        """Save the license plate crop for a unique vehicle, exactly once.

        When a secondary license plate model is loaded, runs the
        two-stage pipeline: the vehicle is cropped from the frame, the
        plate model runs on that crop only, and the best quality-gated
        detection is mapped back to frame coordinates and persisted.
        When no plate model is available, falls back gracefully to the
        geometric bottom-center heuristic.

        Only enclosed vehicle classes (car, bus, truck) are eligible for
        plate regions; open micro-mobility classes return None immediately.

        Args:
            frame: Current video frame (BGR).
            bbox: (x1, y1, x2, y2) vehicle bounding box.
            track_id: Unique tracking identifier of the vehicle.
            vehicle_class: Detected vehicle class (validated against
                ``ENCLOSED_VEHICLE_CLASSES``).
            bottom_fraction: Fraction of the box height taken from the
                bottom (geometric fallback only).
            center_fraction: Fraction of the box width kept around the
                center (geometric fallback only).

        Returns:
            Crop path (relative to ``root_dir``) when a plate crop was
            saved or improved for this ``track_id``, otherwise None.
        """
        if vehicle_class is not None and vehicle_class not in self.ENCLOSED_VEHICLE_CLASSES:
            return None
        if track_id is None:
            return None

        if self.plate_model is not None:
            try:
                return self._save_detected_plate(frame, bbox, track_id)
            except Exception:
                return None

        if track_id in self._saved_plate_ids:
            return None
        region = self.compute_plate_region(bbox, bottom_fraction, center_fraction)
        filename = f"plate_{track_id}.jpg"
        path = self._save_crop_once(frame, region, self.plates_dir, filename,
                                    track_id, self._saved_plate_ids)
        if path is not None:
            abs_plate_path = os.path.join(self.plates_dir, filename)
            text = self._store_plate_text(abs_plate_path, track_id)
            self._check_hotlist(text, track_id, abs_plate_path)
        return path

    def _save_detected_plate(self, frame, bbox, track_id):
        """Run the two-stage plate detection pipeline for a vehicle.

        Extracts the vehicle crop, runs the secondary plate model on it,
        filters detections by confidence (> ``plate_conf_thresh``) and
        aspect ratio (width/height > ``plate_min_aspect``), maps the best
        detection from relative to frame coordinates, and persists the
        best plate crop seen so far for this ``track_id`` (overwriting
        the file when a higher confidence detection arrives). OCR runs on
        every newly accepted plate crop and the result is recorded per
        track via ``_plate_texts``.

        Args:
            frame: Current video frame (BGR).
            bbox: (x1, y1, x2, y2) vehicle bounding box.
            track_id: Unique tracking identifier of the vehicle.

        Returns:
            Crop path (relative to ``root_dir``) when a plate crop was
            saved or improved for this ``track_id``, otherwise None.
        """
        vx1, vy1, vx2, vy2 = self._clamp_region(bbox, frame.shape)
        if vx2 <= vx1 or vy2 <= vy1:
            return None

        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        results = self.plate_model(vehicle_crop, conf=self.plate_conf_thresh,
                                   verbose=False)[0]

        best_conf, best_rel = 0.0, None
        if results.boxes is not None:
            for box in results.boxes:
                rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].item())
                plate_w = rx2 - rx1
                plate_h = ry2 - ry1
                if plate_h <= 0 or (plate_w / plate_h) <= self.plate_min_aspect:
                    continue
                if conf <= self.plate_conf_thresh or conf <= best_conf:
                    continue
                best_conf, best_rel = conf, (rx1, ry1, rx2, ry2)

        if best_rel is None:
            return None
        if self._plate_best_conf.get(track_id, 0.0) >= best_conf:
            return None

        rx1, ry1, rx2, ry2 = best_rel
        frame_region = (rx1 + vx1, ry1 + vy1, rx2 + vx1, ry2 + vy1)
        filename = f"plate_{track_id}.jpg"
        crop = self._extract_crop(frame, frame_region)
        if crop is None:
            return None

        abs_plate_path = os.path.join(self.plates_dir, filename)
        cv2.imwrite(abs_plate_path, crop)
        self._plate_best_conf[track_id] = best_conf
        self._saved_plate_ids.add(track_id)
        text = self._store_plate_text(abs_plate_path, track_id)
        self._check_hotlist(text, track_id, abs_plate_path)

        rel_path = os.path.join(os.path.relpath(self.plates_dir, self.root_dir), filename)
        return rel_path

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
            'employees_tracked': len(self._employee_records),
            'faces_dir': self.faces_dir,
            'vehicles_dir': self.vehicles_dir,
            'plates_dir': self.plates_dir
        }
