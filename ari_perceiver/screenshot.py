# coding=utf-8
# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example script demonstrating usage of AndroidEnv with screenshot capture.

This script performs the following steps:
1. Builds an AndroidEnv configuration from command-line flags.
2. Loads the AndroidEnv environment using the dm_env interface.
3. Resets the environment and extracts the "pixels" from the observation.
4. Displays the captured screenshot using matplotlib.
5. Saves the screenshot as a PNG file using Pillow.
"""

from absl import app
from absl import flags
from absl import logging
from android_env import loader
from android_env.components import config_classes
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

FLAGS = flags.FLAGS

# Simulator arguments.
flags.DEFINE_string('avd_name', 'AndroidWorldAVD', 'Name of AVD to use.')
flags.DEFINE_string('android_avd_home', '~/.android/avd', 'Path to AVD.')
flags.DEFINE_string('android_sdk_root', '~/Library/Android/Sdk', 'Path to Android SDK.')
flags.DEFINE_string('emulator_path', '~/Library/Android/Sdk/emulator/emulator', 'Path to the emulator executable.')
flags.DEFINE_string('adb_path', '~/Library/Android/Sdk/platform-tools/adb', 'Path to the adb executable.')
flags.DEFINE_bool('run_headless', False, 'Whether to display the emulator window.')

# Environment arguments.
flags.DEFINE_string('task_path', None, 'Path to the task textproto file.')

# Experiment arguments.
flags.DEFINE_integer('num_steps', 1000, 'Number of steps to take (not used in this example).')

def main(_):
    # Build the AndroidEnv configuration.
    config = config_classes.AndroidEnvConfig(
        task=config_classes.FilesystemTaskConfig(path=FLAGS.task_path),
        simulator=config_classes.EmulatorConfig(
            emulator_launcher=config_classes.EmulatorLauncherConfig(
                emulator_path=FLAGS.emulator_path,
                android_sdk_root=FLAGS.android_sdk_root,
                android_avd_home=FLAGS.android_avd_home,
                avd_name=FLAGS.avd_name,
                run_headless=FLAGS.run_headless,
            ),
            adb_controller=config_classes.AdbControllerConfig(
                adb_path=FLAGS.adb_path
            ),
        ),
    )

    # Load the Android environment.
    # Since AndroidEnv follows the dm_env API, calling reset() returns a timestep
    # whose observation includes a "pixels" key containing the current screen image.
    with loader.load(config) as env:
        logging.info("Android environment created successfully.")

        # Reset the environment to start a new session.
        timestep = env.reset()
        logging.info("Environment reset complete.")

        # Extract the screenshot from the observation's 'pixels' field.
        for i in range (0,50000):
            screenshot = timestep.observation.get("pixels")
            if screenshot is None:
                logging.error("No 'pixels' field found in observation.")
                return
            logging.info("Screenshot captured with shape: %s", np.shape(screenshot))

    # Display the screenshot using matplotlib.
    plt.figure(figsize=(6, 10))
    plt.imshow(screenshot)
    plt.title("Current App Interface")
    plt.axis("off")
    plt.show()

    # Save the screenshot to a PNG file using Pillow.
    img = Image.fromarray(screenshot)
    img.save("input/screenshot.png")
    logging.info("Screenshot saved to screenshot.png")

if __name__ == '__main__':
    logging.set_verbosity('info')
    logging.set_stderrthreshold('info')
    flags.mark_flags_as_required(['avd_name', 'task_path'])
    app.run(main)
