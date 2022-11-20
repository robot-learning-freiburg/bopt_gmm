import cv2
import numpy as np

from argparse import ArgumentParser
from pathlib  import Path
from tqdm     import tqdm

from bopt_gmm.logging import MP4VideoLogger
from bopt_gmm.utils   import parse_list, \
                             power_set


class VideoReader():
    def __init__(self, file):
        self.src = cv2.VideoCapture(file)
        self._shape = (int(self.src.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.src.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._len   = int(self.src.get(cv2.CAP_PROP_FRAME_COUNT))
        self._last_frame = None

    def __del__(self):
        self.src.release()
    
    def __len__(self):
        return self._len

    def __next__(self):
        return self.get_frame()

    def __iter__(self):
        return self

    @property
    def shape(self):
        return self._shape

    def get_frame(self):
        # Keeps returning the last frame if the video has ended
        ret, frame = self.src.read()
        if ret:
            self._last_frame = frame
            return frame
        return self._last_frame


if __name__ == '__main__':
    parser = ArgumentParser(description='Tool for filtering experiment videos and collecting them in combined videos.')
    parser.add_argument('out', help='Prefix of the new video files to generate.')
    parser.add_argument('videos', nargs='+', help='Video files to process.')
    parser.add_argument('--scale', default=1, help='Factor by which to scale the input images.')
    parser.add_argument('--filter', default=None, choices=['s', 'f'], help='Filter videos by successes or failures.')
    parser.add_argument('--out-size', default='[1920,1080]', help='Output dimensions as list.')
    args = parser.parse_args()

    args.out_size = np.asarray(parse_list(args.out_size, tf=int))

    out_name = Path(args.out).name
    out_dir  = Path(args.out).parent
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    filtered_videos = [f for f in args.videos if f[-5].lower() == args.filter] if args.filter is not None else args.videos

    sample_vid =  VideoReader(filtered_videos[0])

    vs = sample_vid.shape
    del sample_vid

    print(args.out_size, vs)

    grid_dims  = args.out_size // vs

    batch_size = np.product(grid_dims)

    video_batches = []
    while len(filtered_videos) > 0:
        video_batches.append(filtered_videos[:batch_size])
        filtered_videos = filtered_videos[batch_size:]

    starting_coords = np.vstack(power_set(range(grid_dims[0]), range(grid_dims[1]))) * vs

    for x, b in tqdm(enumerate(video_batches), desc='Rendering videos'):
        writer = MP4VideoLogger(out_dir, f'{out_name}_{x:03d}', args.out_size)

        videos = [VideoReader(vf) for vf in b]
        vid_length = max([len(v) for v in videos])

        for fidx_frames in tqdm(zip(range(vid_length), *videos), desc=f'Rendering video {x + 1}/{len(video_batches)}'):
            frames = fidx_frames[1:]
            buffer = np.zeros((args.out_size[1], args.out_size[0], 3), dtype=np.uint8)
            
            for f, (sx, sy) in zip(frames, starting_coords):
                buffer[sy:sy + f.shape[0], sx:sx + f.shape[1]] = f
            
            writer.write_image(buffer)
