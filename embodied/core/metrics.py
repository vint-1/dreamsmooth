import collections
import warnings

import numpy as np
import cv2


def annotate_frame(idx, frame, rew, info={}):
    """Renders a video frame and adds caption."""
    if np.max(frame) <= 1.0:
        frame *= 255.0
    frame = frame.astype(np.uint8)

    # Set the minimum size of frames to (`S`, `S`) for caption readibility.
    # S = 512
    S = 128
    if frame.shape[0] < S:
        frame = cv2.resize(frame, (int(S * frame.shape[1] / frame.shape[0]), S))
    h, w = frame.shape[:2]

    # Add caption.
    frame = np.concatenate([frame, np.zeros((64, w, 3), np.uint8)], 0)
    scale = h / S
    font_size = 0.4 * scale
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(5 * scale), h + int(10 * scale)
    add_text = lambda x, y, c, t: cv2.putText(
        frame, t, (x, y), font_face, font_size, c, 1, cv2.LINE_AA
    )

    add_text(x, y, (255, 255, 0), f"{idx:5} {rew}")
    for i, k in enumerate(info.keys()):
        key_text = f"{k}: "
        key_width = cv2.getTextSize(key_text, font_face, font_size, 1)[0][0]
        offset = int(12 * scale) * (i + 2)
        add_text(x, y + offset, (66, 133, 244), key_text)
        value_text = str(info[k])
        if isinstance(info[k], np.ndarray):
            value_text = np.array2string(
                info[k], precision=2, separator=", ", floatmode="fixed"
            )
        add_text(x + key_width, y + offset, (255, 255, 255), value_text)

    return frame


def annotate_video(video, reward):
    assert len(video) == len(reward), f"len(video): {len(video)} / len(reward): {len(reward)}"
    reward = np.cumsum(reward)
    return [annotate_frame(i, f, r) for i, (f, r) in enumerate(zip(video, reward))]


class Metrics:

  def __init__(self):
    self.scalars = collections.defaultdict(list)
    self.aggs = {}
    self.lasts = {}

  def scalar(self, key, value, agg='mean'):
    assert agg in ('mean', 'sum', 'min', 'max')
    self.scalars[key].append(value)
    self.aggs[key] = agg

  def image(self, key, value):
    self.lasts[key] = value

  def video(self, key, value):
    self.lasts[key] = value

  def add(self, mapping, prefix=None):
    for key, value in mapping.items():
      key = prefix + '/' + key if prefix else key
      if hasattr(value, 'shape') and len(value.shape) > 0:
        self.lasts[key] = value
      else:
        self.scalar(key, value)

  def result(self, reset=True):
    result = {
        k: annotate_video(*v) if isinstance(v, tuple) else v for k, v in self.lasts.items()
    }
    with warnings.catch_warnings():  # Ignore empty slice warnings.
      warnings.simplefilter('ignore', category=RuntimeWarning)
      for key, values in self.scalars.items():
        agg = self.aggs[key]
        value = {
            'mean': np.nanmean,
            'sum': np.nansum,
            'min': np.nanmin,
            'max': np.nanmax,
        }[agg](values, dtype=np.float64)
        result[key] = value
    reset and self.reset()
    return result

  def reset(self):
    self.scalars.clear()
    self.lasts.clear()
