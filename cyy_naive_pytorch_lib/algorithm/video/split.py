import argparse
import os
import sys
import time
import warnings
from enum import IntEnum, auto

import cv2
import cyy_naive_cpp_extension
import torch
from cyy_naive_lib.log import get_logger

import numpy as np


class ProcessingState(IntEnum):
    READ_CONTENT = auto()
    DETECT_SIMILARITY = auto()


class VideoExtractor:
    def __init__(self, video_path):
        self.__video_path = video_path
        self.output_suffix = ".mp4"
        self.similarity_threshold = (0.96, 0.96, 0.96)
        self.__similarity_seconds = 5

    def __get_writer(self, first_frame_seq):
        return ()
        pass

    def extract(self):
        reader = cyy_naive_cpp_extension.video.FFmpegVideoReader(self.__video_path)
        if not reader.open(self.__video_path):
            raise RuntimeError("failed to open video " + self.__video_path)
        frame_width = reader.get_video_width()
        frame_height = reader.get_video_height()

        # create video writer
        self.writer = cyy_naive_cpp_extension.video.FFmpegVideoWriter()
        # res = self.writer.open(self.save_video_path, "mp4", frame_width, frame_height)
        assert res
        dense, interval = self.vdo.get_frame_rate()
        assert interval == 1
        get_logger.info("frame rate is ", dense, interval)
        content_begin_frame = None
        content_end_frame = None
        os.makedirs("output", exist_ok=True)
        last_frame = None
        processing_state = ProcessingState.READ_CONTENT
        similarity_cnt = 0
        while True:
            res, frame = reader.next_frame()
            if res <= 0:
                # TODO end
                break

            frame_seq = frame.seq

            similarity = None
            if content_begin_frame is None:
                assert processing_state == ProcessingState.READ_CONTENT
                content_begin_frame = frame
                last_frame = frame
                continue

            similarity = cyy_naive_cpp_extension.cv.Mat(last_frame.content).MSSIM(
                cyy_naive_cpp_extension.cv.Mat(frame.content)
            )
            if similarity < self.similarity_threshold:
                processing_state = ProcessingState.READ_CONTENT
                similarity_cnt = 0
                last_frame = frame
                continue
            similarity_cnt += 1

            if similarity_cnt >= self.__similarity_seconds * dense:
                pass

            last_frame = frame
            self.writer.write_frame(cyy_naive_cpp_extension.cv.Matrix(frame_content))
            for good_frame in frames:
                frame_seq, imgs = good_frame
                part_img, full_img = imgs

                totel_seconds = frame_seq / dense

                hour = int(totel_seconds) // 3600
                minute = (int(totel_seconds) - hour * 3600) // 60
                sec = int(totel_seconds) % 60
                mill_sec = int(totel_seconds * 1000) % 1000

                filename = "person_{:03d}_time_{:02d}h{:02d}m{:02d}s{:03d}ms_frame_{:06d}.jpg".format(
                    track_id, hour, minute, sec, mill_sec, frame_seq
                )
                filename = (
                    "full_time_{:02d}h{:02d}m{:02d}s{:03d}ms_frame_{:06d}.jpg".format(
                        hour, minute, sec, mill_sec, frame_seq
                    )
                )
        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
