import hashlib
import io
import os
import random
import subprocess
import tempfile
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from PIL import Image

import scrypted_sdk
from scrypted_sdk import (
    ScryptedDeviceBase,
    VideoClips,
    VideoClip,
    VideoClipOptions,
    VideoClipThumbnailOptions,
    MediaObject,
)

def generate_png_bytes(width, height, color):
    """
    Generates a PNG image with the specified width, height, and color,
    and returns it as a byte array.

    Args:
        width (int): Width of the image.
        height (int): Height of the image.
        color (tuple or str): Color to fill the image. Can be a tuple (R, G, B) or string (e.g., "red").

    Returns:
        bytes: Byte array representing the PNG image.
    """
    # Create a new image with the specified size and color
    image = Image.new("RGB", (width, height), color)

    # Create a BytesIO object to hold the PNG data in memory
    byte_io = io.BytesIO()

    # Save the image as a PNG to the BytesIO object
    image.save(byte_io, format="PNG")

    # Get the byte data from the BytesIO object
    byte_data = byte_io.getvalue()

    # Close the BytesIO object
    byte_io.close()

    return byte_data


def generate_mp4_bytes(width, height, start_color, duration=5):
    """
    Generates an MP4 video with an animated color using ffmpeg.

    Args:
        width (int): Width of the video.
        height (int): Height of the video.
        start_color (tuple): Color as (R, G, B).
        duration (int): Duration of the video in seconds.

    Returns:
        bytes: Byte array representing the MP4 video.
    """
    r, g, b = start_color

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', f'color=c=#{r:02x}{g:02x}{b:02x}:s={width}x{height}', '-t', str(duration),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast', '-an',
            '-f', 'mp4', tmp_path
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr.decode()}")

        with open(tmp_path, 'rb') as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def get_color_from_seed(seed_string):
    """
    Generates a random color based on a given seed string.

    Args:
        seed_string (str): The seed string to generate the color.

    Returns:
        tuple: A tuple representing the (R, G, B) color.
    """
    # Create a deterministic hash from the seed string
    seed_hash = hashlib.md5(seed_string.encode()).hexdigest()

    # Use the hash to seed the random number generator
    random.seed(seed_hash)

    # Generate random values for R, G, B between 0 and 255
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    return (r, g, b)


def select_detection_classes(seed_string):
    """
    Randomly selects a subset of detection classes based on the given seed.

    Args:
        seed (str): The seed used to initialize the random generator.

    Returns:
        list: A list of randomly selected detection classes.
    """
    # List of possible detection classes
    detection_classes = ['person', 'vehicle', 'animal', 'package']

    # Create a deterministic hash from the seed string
    seed_hash = hashlib.md5(seed_string.encode()).hexdigest()

    # Seed the random generator for deterministic results
    random.seed(seed_hash)

    # Randomly select a subset of the detection classes (it could be empty or full)
    num_classes = random.randint(1, 2)  # Select between 0 and 2 classes
    selected_classes = random.sample(detection_classes, num_classes)

    # Select 'motion' class with a 50% chance, independent of the others
    if random.random() < 0.5:
        selected_classes.append('motion')

    return selected_classes


BLOCK_DURATION = 20 * 60   # 20-minute blocks in seconds
MAX_EVENTS_PER_BLOCK = 5


def generate_dummy_events(block_start, block_end):
    """
    Generates up to MAX_EVENTS_PER_BLOCK dummy object detection events for a single
    20-minute aligned block.  The number of events (0..MAX_EVENTS_PER_BLOCK) is chosen
    deterministically from the block boundaries so the same block always produces the
    same events.

    Args:
        block_start (int): Block start time in seconds (aligned to BLOCK_DURATION).
        block_end (int): Block end time in seconds (block_start + BLOCK_DURATION).

    Returns:
        list: A list of event dicts for this block.
    """
    events = []

    duration = block_end - block_start
    if duration <= 0:
        return events

    # Seed once per block so num_events and all event positions are deterministic.
    block_seed = f"block_{block_start}_{block_end}"
    seed_hash = hashlib.md5(block_seed.encode()).hexdigest()
    rng = random.Random(seed_hash)

    num_events = rng.randint(0, MAX_EVENTS_PER_BLOCK)

    for i in range(num_events):
        event_seed = f"event_{block_start}_{i}"

        event_start = rng.randint(block_start, block_end - 1)
        max_duration = block_end - event_start
        event_duration = rng.randint(1, min(30, max_duration))
        event_end = event_start + event_duration

        event_color = get_color_from_seed(event_seed)
        image_bytes = generate_png_bytes(200, 200, event_color)

        events.append({
            "start_time": event_start,
            "end_time": event_end,
            "snapshot": image_bytes,
            "detection_classes": select_detection_classes(event_seed),
        })

    return events


class VideoServer:
    def __init__(self, port=8080):
        self.port = port
        self.video_cache = {}
        self._server = None
        self._handler = None

    def register_video(self, video_id: str, mp4_bytes: bytes) -> None:
        self.video_cache[video_id] = mp4_bytes

    def has_video(self, video_id: str) -> bool:
        return video_id in self.video_cache

    def _create_handler(self):
        video_cache = self.video_cache

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed_path = urlparse(self.path)
                path = parsed_path.path.lstrip('/')
                if path in video_cache:
                    self.send_response(200)
                    self.send_header('Content-Type', 'video/mp4')
                    self.send_header('Content-Length', str(len(video_cache[path])))
                    self.end_headers()
                    self.wfile.write(video_cache[path])
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        return RequestHandler

    def start(self):
        self._handler = self._create_handler()
        self._server = HTTPServer(('', self.port), self._handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"Video server running on port {self.port}")

    def stop(self):
        if self._server:
            self._server.shutdown()


class EventManager:
    def __init__(self):
        # Maps block_start (int, seconds) -> list of event dicts for that block.
        # Blocks are BLOCK_DURATION-second windows aligned to multiples of BLOCK_DURATION.
        self._block_cache: dict[int, list[dict]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _block_start_for(self, t: int) -> int:
        """Return the aligned block start time that contains timestamp t."""
        return (t // BLOCK_DURATION) * BLOCK_DURATION

    def _blocks_for_range(self, start_time: int, end_time: int) -> list[int]:
        """
        Return an ordered list of block start times whose blocks overlap
        [start_time, end_time).
        """
        first_block = self._block_start_for(start_time)
        blocks = []
        b = first_block
        while b < end_time:
            blocks.append(b)
            b += BLOCK_DURATION
        return blocks

    def _get_or_generate_block(self, block_start: int) -> list[dict]:
        """
        Return the cached events for a block, generating and caching them first
        if this block has not yet been seen.
        """
        if block_start not in self._block_cache:
            block_end = block_start + BLOCK_DURATION
            self._block_cache[block_start] = generate_dummy_events(block_start, block_end)
        return self._block_cache[block_start]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_events_for_range(self, start_time: int, end_time: int,
                                   detection_classes: list[str] | None = None) -> list[dict]:
        """
        Return all events that fall within [start_time, end_time), optionally
        filtered by detection class.

        Each 20-minute aligned block touched by the range is generated on first
        access and cached permanently; subsequent calls for the same block reuse
        the cached events and only apply the filter.

        Args:
            start_time (int): Range start in seconds.
            end_time (int): Range end in seconds.
            detection_classes (list[str] | None): If provided, only events that
                share at least one class with this list are returned.

        Returns:
            list[dict]: Matching events sorted by start_time.
        """
        result = []

        for block_start in self._blocks_for_range(start_time, end_time):
            for event in self._get_or_generate_block(block_start):
                # Restrict to events that start within the requested range.
                if not (start_time <= event["start_time"] < end_time):
                    continue
                # Apply optional detection-class filter.
                if detection_classes:
                    if not any(c in event["detection_classes"] for c in detection_classes):
                        continue
                result.append(event)

        result.sort(key=lambda e: e["start_time"])
        return result

    def find_event_by_start_time(self, start_time: int) -> dict | None:
        """
        Finds an event by its start time.

        Args:
            start_time (int): Start time of the event to find (seconds).

        Returns:
            dict: The event dictionary if found, None otherwise.
        """
        block_start = self._block_start_for(start_time)
        for event in self._get_or_generate_block(block_start):
            if event["start_time"] == start_time:
                return event
        return None


class VideoClipsTester(ScryptedDeviceBase, VideoClips):
    def __init__(self, nativeId=None) -> None:
        super().__init__(nativeId)
        self.event_manager = EventManager()
        self.video_server = VideoServer(port=8765)
        self.video_server.start()

    async def getVideoClip(self, videoId: str) -> MediaObject:
        event = self.event_manager.find_event_by_start_time(int(videoId))
        if not event:
            return None

        mp4_key = videoId + ".mp4"
        if not self.video_server.has_video(mp4_key):
            color = get_color_from_seed(f"event_{event['start_time']}_{event['end_time']}")
            duration = event['end_time'] - event['start_time']
            video_bytes = generate_mp4_bytes(200, 200, color, max(duration, 1))
            self.video_server.register_video(mp4_key, video_bytes)

        video_url = f"http://localhost:{self.video_server.port}/{mp4_key}"
        mo = await scrypted_sdk.mediaManager.createFFmpegMediaObject({
            "url": video_url,
        })
        return mo

    async def getVideoClipThumbnail(self, thumbnailId: str, options: VideoClipThumbnailOptions = None) -> scrypted_sdk.MediaObject:
        event = self.event_manager.find_event_by_start_time(int(thumbnailId))
        if not event:
            return None

        mo = await scrypted_sdk.mediaManager.createMediaObject(event["snapshot"], "image/png")
        return mo

    async def getVideoClips(self, options: VideoClipOptions = None) -> list[VideoClip]:
        start_time = int(options["startTime"] / 1000)
        end_time = int(options["endTime"] / 1000)
        detection_classes = options.get("detectionClasses") if options else None

        events = self.event_manager.generate_events_for_range(
            start_time, end_time, detection_classes=detection_classes
        )
        clips = []
        for event in events:
            clip = {
                "id": str(event["start_time"]),
                "startTime": event["start_time"] * 1000,
                "endTime": event["end_time"] * 1000,
                "detectionClasses": event["detection_classes"],
                "thumbnailId": str(event["start_time"]),
                "videoId": str(event["start_time"]),
            }
            clips.append(clip)
        return clips

    async def removeVideoClips(self, videoClipIds: list[str]) -> None:
        raise NotImplementedError("Method not implemented")


def create_scrypted_plugin():
    return VideoClipsTester()