import cv2
from openni import openni2
import numpy as np

depth_camera_resolution = (512, 424) # px
depth_camera_fps = 30
color_camera_resolution = (1920, 1080) # px
color_camera_fps = 30
video_beam_resolution = (1920, 1080) # px
video_camera_fps = 30
window_size = (1050, 680) # mm

openni2.initialize("C:/Development Program Files/OpenNI2/Redist")
device = openni2.Device.open_any()

depth_stream = device.create_depth_stream()
color_stream = device.create_color_stream()

if depth_stream is None:
    print("No depth stream found")
    exit(1)

if color_stream is None:
    print("No color stream found")
    exit(1)

depth_stream.set_video_mode(
    openni2.VideoMode(
        pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM,
        resolutionX=depth_camera_resolution[0],
        resolutionY=depth_camera_resolution[1],
        fps=depth_camera_fps
    )
)
color_stream.set_video_mode(
    openni2.VideoMode(
        pixelFormat=openni2.PIXEL_FORMAT_RGB888,
        resolutionX=color_camera_resolution[0],
        resolutionY=color_camera_resolution[1],
        fps=color_camera_fps
    )
)
depth_stream.start()
color_stream.start()

color_frame = color_stream.read_frame()
color_image = np.frombuffer(color_frame.get_buffer_as_uint8(), dtype=np.uint8).reshape(video_beam_resolution[::-1] + (3,))
color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
color_image = cv2.flip(color_image, 1)

depth_frame = depth_stream.read_frame()
depth_image = np.frombuffer(depth_frame.get_buffer_as_uint16(), dtype=np.uint16).reshape(depth_camera_resolution[::-1])
depth_image = cv2.flip(depth_image, 1)
depth_vis = cv2.convertScaleAbs(depth_image, alpha=255.0/1000)
depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# Resize depth image to match color image size
depth_vis_resized = cv2.resize(depth_vis, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)

# Blend images (alpha controls transparency of depth overlay)
alpha = 0.7
overlay = cv2.addWeighted(color_image, 1 - alpha, depth_vis_resized, alpha, 0)

cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Overlay", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("Overlay", overlay)
cv2.waitKey()