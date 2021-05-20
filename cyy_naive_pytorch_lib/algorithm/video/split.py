import os
import shutil
import tempfile
from enum import IntEnum, auto

import cyy_naive_cpp_extension
from cyy_naive_lib.log import get_logger


class ProcessingState(IntEnum):
    READ_CONTENT = auto()
    DETECT_SIMILARITY = auto()
    IN_SIMILARITY = auto()


class VideoSpiltter:
    def __init__(
        self,
        video_path,
        similarity_threshold,
        similarity_seconds=5,
        content_seconds=1,
    ):
        self.__video_path: str = os.path.abspath(video_path)
        self.output_format = "mp4"
        self.similarity_threshold = similarity_threshold
        self.similarity_seconds = similarity_seconds
        self.content_seconds = content_seconds

    def __get_writer(self, reader):
        frame_width = reader.get_video_width()
        frame_height = reader.get_video_height()
        writer = cyy_naive_cpp_extension.video.FFmpegVideoWriter()
        tmp_path = tempfile.mktemp()
        assert writer.open(tmp_path, "mp4", frame_width, frame_height)
        return writer

    def __get_file_name(self, reader, start_frame_seq, end_frame_seq):
        dense, interval = reader.get_frame_rate()
        assert interval == 1
        filename = ""
        for frame_seq in [start_frame_seq, end_frame_seq]:
            totel_seconds = frame_seq / dense
            hour = int(totel_seconds) // 3600
            minute = (int(totel_seconds) - hour * 3600) // 60
            sec = int(totel_seconds) % 60
            mill_sec = int(totel_seconds * 1000) % 1000

            filename += "{:02d}h{:02d}m{:02d}s{:03d}ms_".format(
                hour, minute, sec, mill_sec
            )
        filename = filename[:-1]
        filename += "." + self.output_format
        filename = ".".join(self.__video_path.split(".")[:-1] + [filename])
        return filename

    def split(self):
        reader = cyy_naive_cpp_extension.video.FFmpegVideoReader()
        if not reader.open(self.__video_path):
            raise RuntimeError("failed to open video " + self.__video_path)

        dense, interval = reader.get_frame_rate()
        assert interval == 1
        get_logger().info("frame rate is %s %s", dense, interval)
        content_begin_frame = None
        content_end_frame = None
        similarity_begin_frame = None
        similarity_end_frame = None
        processing_state = ProcessingState.READ_CONTENT
        similarity_cnt = 0
        content_writer = None
        similarity_writer = None
        similarity_videos = set()
        content_videos = set()
        pending_content_frames = []

        def close_similarity_writer():
            nonlocal similarity_writer, similarity_begin_frame, similarity_end_frame
            nonlocal similarity_cnt, similarity_videos
            if processing_state is ProcessingState.IN_SIMILARITY:
                filename = self.__get_file_name(
                    reader, similarity_begin_frame.seq, similarity_end_frame.seq
                )
                similarity_writer.close()
                shutil.move(similarity_writer.get_url(), filename)
                similarity_videos.add(filename)
            elif similarity_writer is not None:
                similarity_writer.close()
                os.remove(similarity_writer.get_url())
            similarity_writer = None
            similarity_cnt = 0
            similarity_begin_frame = None
            similarity_end_frame = None

        def close_content_writer():
            nonlocal content_writer, content_begin_frame
            nonlocal content_end_frame, content_videos
            # save content video
            if content_writer is None:
                return
            assert content_begin_frame is not None
            assert content_end_frame is not None
            content_writer.close()
            if (
                content_end_frame.seq - content_begin_frame.seq
                >= self.content_seconds * dense
            ):
                filename = self.__get_file_name(
                    reader, content_begin_frame.seq, content_end_frame.seq
                )
                print(self.content_seconds * dense)
                shutil.move(content_writer.get_url(), filename)
                content_videos.add(filename)
            else:
                os.remove(content_writer.get_url())
            content_writer = None
            content_begin_frame = None
            content_end_frame = None

        while True:
            res, frame = reader.next_frame()
            if res <= 0:
                close_similarity_writer()
                close_content_writer()
                break

            assert frame.seq != 0

            # the first frame
            if frame.seq == 1:
                content_begin_frame = frame
                content_end_frame = frame
                similarity_begin_frame = frame
                similarity_end_frame = frame
                content_writer = self.__get_writer(reader)
                similarity_writer = self.__get_writer(reader)
                content_writer.write_frame(
                    cyy_naive_cpp_extension.cv.Matrix(frame.content)
                )
                similarity_writer.write_frame(
                    cyy_naive_cpp_extension.cv.Matrix(frame.content)
                )
                continue

            similarity = cyy_naive_cpp_extension.cv.Mat(
                similarity_end_frame.content
            ).MSSIM(cyy_naive_cpp_extension.cv.Mat(frame.content))
            similarity_end_frame = frame
            # Cv::Scalar has four emelemts, the last one is useless.
            similarity = list(similarity)[0:3]
            if sum(similarity) / len(similarity) < self.similarity_threshold:
                close_similarity_writer()
                if processing_state is ProcessingState.IN_SIMILARITY:
                    assert not pending_content_frames
                    assert content_writer is None
                    content_writer = self.__get_writer(reader)
                    content_begin_frame = frame
                else:
                    for f in pending_content_frames:
                        content_writer.write_frame(
                            cyy_naive_cpp_extension.cv.Matrix(f.content)
                        )
                pending_content_frames.clear()
                processing_state = ProcessingState.READ_CONTENT
                content_end_frame = frame
                similarity_begin_frame = frame
                content_writer.write_frame(
                    cyy_naive_cpp_extension.cv.Matrix(frame.content)
                )
                continue
            processing_state = ProcessingState.DETECT_SIMILARITY
            if similarity_writer is None:
                similarity_writer = self.__get_writer(reader)
            similarity_writer.write_frame(
                cyy_naive_cpp_extension.cv.Matrix(frame.content)
            )
            similarity_cnt += 1
            if similarity_cnt >= self.similarity_seconds * dense:
                pending_content_frames.clear()
                close_content_writer()
                processing_state = ProcessingState.IN_SIMILARITY
            else:
                pending_content_frames.append(frame)
        return content_videos, similarity_videos
