import cv2

from pathlib import Path


class MP4VideoLogger(object):
    def __init__(self, dir_path, filename, image_size, frame_rate=30.0):
        self.filename = f'{filename}.mp4' if filename[-4:].lower() != '.mp4' else filename
        
        self.dir_path = Path(dir_path)
        if not self.dir_path.exists():
            self.dir_path.mkdir(parents=True)

        out_file = f'{self.dir_path}/{self.filename}'
        self.writer = cv2.VideoWriter(out_file, 
                                      cv2.VideoWriter.fourcc(*'mp4v'),
                                      frame_rate,
                                      image_size)
        self._writer_active = True

    def write_image(self, image):
        if not self._writer_active:
            raise Exception(f'Video file {self.dir_path}/{self.filename} has been closed.')
        self.writer.write(image)

    def rename(self, new_name):
        if self._writer_active:
            self.writer.release()
        
        new_name = f'{new_name}.mp4' if new_name[-4:].lower() != '.mp4' else new_name
        new_path = Path(f'{self.dir_path}/{new_name}')
        old_path = Path(f'{self.dir_path}/{self.filename}')
        old_path.rename(new_path)
        self.filename = new_name

    def __del__(self):
        if self._writer_active:
            self.writer.release()
