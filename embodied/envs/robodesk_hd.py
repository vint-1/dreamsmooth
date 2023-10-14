import os
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'
import robodesk
import numpy as np
from dm_control import mujoco
from PIL import Image

class RoboDeskHD(robodesk.RoboDesk):
    def render(self, mode='rgb_array', resize=True):
        scale_factor = (self.image_size/120)
        params = {'distance': 1.8, 'azimuth': 90, 'elevation': -60,
                'crop_box': (16.75 * scale_factor, 25.0  * scale_factor, 105.0 * scale_factor, 88.75 * scale_factor), 'size': 120}
        camera = mujoco.Camera(
            physics=self.physics, height=self.image_size,
            width=self.image_size, camera_id=-1)
        camera._render_camera.distance = params['distance']  # pylint: disable=protected-access
        camera._render_camera.azimuth = params['azimuth']  # pylint: disable=protected-access
        camera._render_camera.elevation = params['elevation']  # pylint: disable=protected-access
        camera._render_camera.lookat[:] = [0, 0.535, 1.1]  # pylint: disable=protected-access

        image = camera.render(depth=False, segmentation=False)
        camera._scene.free()  # pylint: disable=protected-access
        image = Image.fromarray(image).crop(box=params['crop_box'])
        image = image.resize([self.image_size, self.image_size],
                           resample=Image.LANCZOS)
        image = np.asarray(image)
        return image
