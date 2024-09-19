import hashlib
import io
import random

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
    detection_classes = ['motion', 'person', 'vehicle', 'animal', 'package']

    # Create a deterministic hash from the seed string
    seed_hash = hashlib.md5(seed_string.encode()).hexdigest()

    # Seed the random generator for deterministic results
    random.seed(seed_hash)

    # Randomly select a subset of the detection classes (it could be empty or full)
    num_classes = random.randint(0, len(detection_classes))  # Select between 0 and all classes
    selected_classes = random.sample(detection_classes, num_classes)

    return selected_classes


def scale_max_events(duration, min_interval=30, max_events=50):
    """
    Scales the maximum number of events based on the duration.

    Args:
        duration (int): The total duration in seconds.
        min_interval (int): The minimum time interval between events in seconds.
        max_events (int): The absolute maximum number of events to allow.

    Returns:
        int: The scaled maximum number of events based on the duration.
    """
    # Calculate how many events can reasonably fit within the duration
    scaled_events = duration // min_interval

    # Ensure the number of events doesn't exceed the absolute maximum
    return min(scaled_events, max_events)


def generate_dummy_events(start_time, end_time, min_interval=30, max_events=50):
    """
    Generates a list of dummy object detection events scaled according to the time range.

    Args:
        start_time (int): Start time in seconds.
        end_time (int): End time in seconds.
        min_interval (int): Minimum time interval between events in seconds.
        max_events (int): Absolute maximum number of events to generate.

    Returns:
        list: A list of dictionaries representing the dummy events.
              Each event includes a start time, end time, and a random image as bytes.
    """
    # List to store the generated events
    events = []

    # Calculate the duration of the period
    duration = end_time - start_time

    # Return an empty list if the time range is invalid
    if duration <= 0:
        return events

    # Scale the number of events based on the duration
    scaled_max_events = scale_max_events(duration, min_interval, max_events)

    # Randomly choose the number of events to generate (it could be zero)
    num_events = random.randint(0, scaled_max_events)

    for _ in range(num_events):
        # Randomly generate a start time within the time range
        event_start = random.randint(start_time, end_time - 1)

        # Randomly generate a duration for the event, ensuring it ends before the end_time
        max_duration = end_time - event_start
        event_duration = random.randint(1, min(min_interval, max_duration))  # Ensure short events
        event_end = event_start + event_duration

        # Generate a random color for the detection snapshot
        event_color = get_color_from_seed(f"event_{event_start}_{event_end}")

        # Generate an image as a snapshot for the event
        image_bytes = generate_png_bytes(200, 200, event_color)

        # Add the event to the list
        events.append({
            "start_time": event_start,
            "end_time": event_end,
            "snapshot": image_bytes,  # Image data in bytes
            "detection_classes": select_detection_classes(f"event_{event_start}_{event_end}"),
        })

    return events


class EventManager:
    def __init__(self):
        # This will store all generated events and their time ranges
        self.generated_events = []

    def _is_overlapping(self, range1, range2):
        """
        Check if two time ranges overlap.

        Args:
            range1 (tuple): (start_time, end_time) of the first range.
            range2 (tuple): (start_time, end_time) of the second range.

        Returns:
            bool: True if the ranges overlap, False otherwise.
        """
        return not (range1[1] <= range2[0] or range2[1] <= range1[0])

    def _find_overlapping_segments(self, start_time, end_time):
        """
        Find all segments of the requested time range that overlap with existing events.

        Args:
            start_time (int): Start time of the requested range.
            end_time (int): End time of the requested range.

        Returns:
            list: A list of tuples representing the overlapping segments
                  and their corresponding events.
        """
        overlapping_segments = []
        requested_range = (start_time, end_time)

        for event_range, events in self.generated_events:
            if self._is_overlapping(requested_range, event_range):
                # Calculate the overlapping part of the ranges
                overlap_start = max(start_time, event_range[0])
                overlap_end = min(end_time, event_range[1])
                overlapping_segments.append(((overlap_start, overlap_end), events))

        return overlapping_segments

    def _generate_and_cache_events(self, start_time, end_time):
        """
        Generate new events for the given time range and cache them.

        Args:
            start_time (int): Start time in seconds.
            end_time (int): End time in seconds.

        Returns:
            list: A list of newly generated events.
        """
        new_events = generate_dummy_events(start_time, end_time)
        self.generated_events.append(((start_time, end_time), new_events))
        return new_events

    def generate_events_for_range(self, start_time, end_time):
        """
        Generates events for a given time range if they don't already exist.
        In case of partial overlap, reuse existing events and generate for the remaining time range.

        Args:
            start_time (int): Start time in seconds.
            end_time (int): End time in seconds.

        Returns:
            list: A list of events (either existing or newly generated).
        """
        # Step 1: Find overlapping segments
        overlapping_segments = self._find_overlapping_segments(start_time, end_time)

        # Step 2: Track what time ranges still need event generation
        uncovered_ranges = []
        last_covered_time = start_time

        for overlap_range, _ in overlapping_segments:
            if last_covered_time < overlap_range[0]:
                # There's an uncovered range before the overlap
                uncovered_ranges.append((last_covered_time, overlap_range[0]))

            # Move the last covered time forward
            last_covered_time = overlap_range[1]

        # After the last overlap, check if there's still an uncovered range
        if last_covered_time < end_time:
            uncovered_ranges.append((last_covered_time, end_time))

        # Step 3: Generate events for the uncovered ranges and cache them
        new_events = []
        for uncovered_range in uncovered_ranges:
            uncovered_start, uncovered_end = uncovered_range
            new_events += self._generate_and_cache_events(uncovered_start, uncovered_end)

        # Step 4: Combine existing (overlapping) events with newly generated ones
        combined_events = []
        for overlap_range, events in overlapping_segments:
            combined_events += events

        combined_events += new_events

        return combined_events

    def find_event_by_start_time(self, start_time):
        """
        Finds an event by its start time.

        Args:
            start_time (int): Start time of the event to find.

        Returns:
            dict: The event dictionary if found, None otherwise.
        """
        for _, events in self.generated_events:
            for event in events:
                if event["start_time"] == start_time:
                    return event
        return None


class VideoClipsTester(ScryptedDeviceBase, VideoClips):
    def __init__(self, nativeId=None) -> None:
        super().__init__(nativeId)
        self.event_manager = EventManager()

    async def getVideoClip(self, videoId: str) -> MediaObject:
        raise NotImplementedError("Method not implemented")

    async def getVideoClipThumbnail(self, thumbnailId: str, options: VideoClipThumbnailOptions = None) -> scrypted_sdk.MediaObject:
        event = self.event_manager.find_event_by_start_time(int(thumbnailId))
        if not event:
            return None

        mo = await scrypted_sdk.mediaManager.createMediaObject(event["snapshot"], "image/png")
        return mo

    async def getVideoClips(self, options: VideoClipOptions = None) -> list[VideoClip]:
        events = self.event_manager.generate_events_for_range(int(options["startTime"] / 1000), int(options["endTime"] / 1000))
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