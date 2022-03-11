from os.path import join
from time import time
from argparse import ArgumentParser
from camera import VideoProcessor
from Materials.config import CONFIG


def main(id: int, path: str, interval: int, timeout: int) -> None:
    if path.isnumeric():
        path = int(path)

    vp = VideoProcessor(path, encode=True)

    filepath = join(CONFIG["camera_files_path"], str(id))
    count = 0
    start = time()

    try:
        for frame in vp:
            if count % interval == 0:
                if time() - start > timeout:
                    break
                with open(filepath, "w") as f:
                    f.write(frame)
            count += 1
    except KeyboardInterrupt:
        print(f"\n{count // interval} unique frames were written during {id} camera processing!\n")


if __name__ == "__main__":
    parser = ArgumentParser(usage="This is a program for writing camera frames to files that emulate the DB")
    parser.add_argument("--id", "-i", type=int, help="ID for camera objects")
    parser.add_argument("--path", "-p", type=str, help="Camera path for gaining access to frames")
    parser.add_argument("--interval", "-f", default=5, type=int, help="Each *interval*th frame is going to be written")
    parser.add_argument("--timeout", "-t", default=3600, type=int, help="Time in seconds for the app to run")
    args = parser.parse_args()

    main(args.id, args.path, args.interval, args.timeout)
