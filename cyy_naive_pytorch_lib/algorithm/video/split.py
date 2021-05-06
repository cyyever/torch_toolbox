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
    def __init__(self, video_path):
        self.__video_path: str = video_path
        self.output_format = "mp4"
        self.similarity_threshold = (0.96, 0.96, 0.96)
        self.similarity_seconds = 5  # at least
        self.content_seconds = 1  # at least

    def __get_writer(self, reader):
        frame_width = reader.get_video_width()
        frame_height = reader.get_video_height()
        writer = cyy_naive_cpp_extension.video.FFmpegVideoWriter()
        tmp_path = tempfile.mktemp()
        assert writer.open(tmp_path, "mp4", frame_width, frame_height)
        return (writer, tmp_path)

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

    def extract(self):
        reader = cyy_naive_cpp_extension.video.FFmpegVideoReader(self.__video_path)
        if not reader.open(self.__video_path):
            raise RuntimeError("failed to open video " + self.__video_path)

        dense, interval = reader.get_frame_rate()
        assert interval == 1
        get_logger().info("frame rate is %s %s", dense, interval)
        content_begin_frame = None
        content_end_frame = None
        similarity_begin_frame = None
        processing_state = ProcessingState.READ_CONTENT
        similarity_cnt = 0
        content_writer = None
        content_tmp_path = None
        similarity_writer = None
        similarity_tmp_path = None
        similarity_videos = []
        content_videos = []
        while True:
            res, frame = reader.next_frame()
            if res <= 0:
                # TODO end
                break

            assert frame.seq != 0

            # the first frame
            if frame.seq == 1:
                content_begin_frame = frame
                content_end_frame = frame
                similarity_begin_frame = frame
                content_writer, content_tmp_path = self.__get_writer(reader)
                similarity_writer, similarity_tmp_path = self.__get_writer(reader)
                content_writer.write_frame(
                    cyy_naive_cpp_extension.cv.Matrix(frame.content)
                )
                similarity_writer.write_frame(
                    cyy_naive_cpp_extension.cv.Matrix(frame.content)
                )
                continue

            similarity = cyy_naive_cpp_extension.cv.Mat(
                similarity_begin_frame.content
            ).MSSIM(cyy_naive_cpp_extension.cv.Mat(frame.content))
            if similarity < self.similarity_threshold:
                if processing_state is ProcessingState.IN_SIMILARITY:
                    filename = self.__get_file_name(
                        reader, similarity_begin_frame.seq, frame.seq - 1
                    )
                    similarity_writer.close()
                    shutil.move(similarity_tmp_path, filename)
                    similarity_videos.append(similarity_tmp_path)
                    content_begin_frame = frame
                elif similarity_writer is not None:
                    similarity_writer.close()
                    os.remove(similarity_tmp_path)
                similarity_writer, similarity_tmp_path = None, None

                processing_state = ProcessingState.READ_CONTENT
                similarity_cnt = 0
                content_end_frame = frame
                similarity_begin_frame = frame
                content_writer.write_frame(
                    cyy_naive_cpp_extension.cv.Matrix(frame.content)
                )
                continue
            processing_state = ProcessingState.DETECT_SIMILARITY
            if similarity_writer is None:
                similarity_writer, similarity_tmp_path = self.__get_writer(reader)
            similarity_writer.write_frame(
                cyy_naive_cpp_extension.cv.Matrix(frame.content)
            )
            similarity_cnt += 1
            if similarity_cnt >= self.similarity_seconds * dense:
                # save content video
                if content_writer is not None:
                    assert content_begin_frame is not None
                    assert content_end_frame is not None
                    content_writer.close()
                    content_writer = None
                    if (
                        content_end_frame.seq - content_begin_frame.seq
                        >= self.content_seconds * dense
                    ):
                        filename = self.__get_file_name(
                            reader, content_end_frame.seq, content_begin_frame.seq
                        )
                        shutil.move(content_tmp_path, filename)
                        content_videos.append(filename)
                    else:
                        os.remove(content_tmp_path)
                    content_tmp_path = None
                processing_state = ProcessingState.IN_SIMILARITY
        return content_videos, similarity_videos
