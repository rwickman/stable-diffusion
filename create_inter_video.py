import os
from moviepy.editor import *
import argparse
import numpy as np
from PIL import Image


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--smoothing_sec", type=float, default=1.0)#, help="number of images to be generated"
    parser.add_argument(
        "--mp4_fps", type=float, default=60.0)
    parser.add_argument(
        "--duration_sec", type=float, default=10.0)
    parser.add_argument(
        "--dir", required=True)
    parser.add_argument(
        "--inter", action="store_true", help="Interpolate between frames")

    parser.add_argument(
        "--inter_steps", type=int, default=4, help="Interpolate between frames")

    parser.add_argument("--out", default="ghost_inter.mp4")


    args = parser.parse_args()
    
    # Derive number of frames
    num_frames = int(np.rint(args.duration_sec * args.mp4_fps))
    fps=10

    img_files = [os.path.join(args.dir, img)
               for img in os.listdir(args.dir)
               if img.endswith(".png")]
    img_files = sorted(img_files, key=lambda x: int(x.split("/")[-1].split("_")[0]))

    frame_idx = 0
    #print("len(img_files)", len(img_files))
    def make_frame(t):
        global frame_idx
        if args.inter:
            actual_frame_idx = int(frame_idx // args.inter_steps)
            img = np.array(Image.open(img_files[actual_frame_idx]))
            inter_frame_step = frame_idx % args.inter_steps
            if inter_frame_step != 0 and actual_frame_idx + 1 < len(img_files):
                tgt_img = np.array(Image.open(img_files[actual_frame_idx + 1]))
                
                # Interpolate between current and next frame
                inter_factor = inter_frame_step / args.inter_steps
                img = (1 - inter_factor) * img + inter_factor * tgt_img
        else:
            img = np.array(Image.open(img_files[frame_idx]))

        frame_idx += 1

        return img

    if args.inter:
       duration = (args.inter_steps * len(img_files) - 1) / args.mp4_fps
    else:
        duration = (len(img_files) - 1) / args.mp4_fps
    
    print("duration", duration)
    clip = VideoClip(make_frame, duration=duration)
    mp4_bitrate='16M'
    clip.write_videofile(args.out, fps=args.mp4_fps, codec='libx264', bitrate=mp4_bitrate)
    # def make_frame(t):
    #     frame_idx = int(np.clip(np.round(t * args.mp4_fps), 0, num_frames - 1))

    #     #img = img.resize((1024, 512), Image.BICUBIC)
    #     img = np.array(img)
    #     #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    #     return img

    # # print(make_frame(0))
    # mp4_bitrate='16M'
    # print("args.mp4_fps", type(args.mp4_fps), args.mp4_fps + 1)
    # VideoClip(make_frame, duration=args.duration_sec).write_videofile(args.out, fps=60.0, codec='libx264', bitrate=mp4_bitrate)
