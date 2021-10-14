# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.experimental.core.env."""

from absl.testing.absltest import mock
import tensorflow as tf
from tfx.orchestration.experimental.core import env
from tfx.orchestration.experimental.core import test_utils


class EnvTest(test_utils.TfxTest):

  def test_set_env(self):
    default_env = env.get_env()
    self.assertIsInstance(default_env, env._DefaultEnv)
    test_env = mock.create_autospec(spec=env.Env, instance=True)
    with env.set_env(test_env):
      self.assertIs(env.get_env(), test_env)
    self.assertIs(env.get_env(), default_env)


if __name__ == '__main__':
  tf.test.main()
