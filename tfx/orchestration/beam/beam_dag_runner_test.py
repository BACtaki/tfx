# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.portable.beam_dag_runner."""
import os
import tempfile
from typing import Any, Dict, List, Optional

from absl.testing import parameterized
import tensorflow as tf
from tfx import types
from tfx.dsl.compiler import compiler
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec as executor_spec_lib
from tfx.orchestration import pipeline as pipeline_py
from tfx.orchestration.beam import beam_dag_runner
from tfx.orchestration.beam.legacy import beam_dag_runner as legacy_beam_dag_runner
from tfx.orchestration.config import pipeline_config
from tfx.orchestration.metadata import sqlite_metadata_connection_config
from tfx.orchestration.portable import partial_run_utils
from tfx.proto.orchestration import local_deployment_config_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter

from google.protobuf import message
from google.protobuf import text_format

_executed_components = []
_COMPONENT_NAME = 'component_name'

_LOCAL_DEPLOYMENT_CONFIG = text_format.Parse(
    """
    metadata_connection_config {
        fake_database {}
    }
    """, local_deployment_config_pb2.LocalDeploymentConfig())

_INTERMEDIATE_DEPLOYMENT_CONFIG = text_format.Parse(
    """
    metadata_connection_config {
      [type.googleapis.com/ml_metadata.ConnectionConfig] {
        fake_database {}
      }
    }
    """, pipeline_pb2.IntermediateDeploymentConfig())


class _ArtifactTypeA(types.Artifact):
  TYPE_NAME = 'ArtifactTypeA'


class _ArtifactTypeB(types.Artifact):
  TYPE_NAME = 'ArtifactTypeB'


class _ArtifactTypeC(types.Artifact):
  TYPE_NAME = 'ArtifactTypeC'


class _ArtifactTypeD(types.Artifact):
  TYPE_NAME = 'ArtifactTypeD'


class _ArtifactTypeE(types.Artifact):
  TYPE_NAME = 'ArtifactTypeE'


# We define fake component spec classes below for testing.
class _FakeComponentSpecA(types.ComponentSpec):
  PARAMETERS = {_COMPONENT_NAME: ExecutionParameter(type=str)}
  INPUTS = {}
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeA)}


class _FakeComponentSpecB(types.ComponentSpec):
  PARAMETERS = {_COMPONENT_NAME: ExecutionParameter(type=str)}
  INPUTS = {'a': ChannelParameter(type=_ArtifactTypeA)}
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeB)}


class _FakeComponentSpecC(types.ComponentSpec):
  PARAMETERS = {_COMPONENT_NAME: ExecutionParameter(type=str)}
  INPUTS = {'a': ChannelParameter(type=_ArtifactTypeA)}
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeC)}


class _FakeComponentSpecD(types.ComponentSpec):
  PARAMETERS = {_COMPONENT_NAME: ExecutionParameter(type=str)}
  INPUTS = {
      'b': ChannelParameter(type=_ArtifactTypeB),
      'c': ChannelParameter(type=_ArtifactTypeC),
  }
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeD)}


class _FakeComponentSpecE(types.ComponentSpec):
  PARAMETERS = {_COMPONENT_NAME: ExecutionParameter(type=str)}
  INPUTS = {
      'a': ChannelParameter(type=_ArtifactTypeA),
      'b': ChannelParameter(type=_ArtifactTypeB),
      'd': ChannelParameter(type=_ArtifactTypeD),
  }
  OUTPUTS = {'output': ChannelParameter(type=_ArtifactTypeE)}


class _FakeExecutor(base_executor.BaseExecutor):

  def Do(self, input_dict: Dict[str, List[types.Artifact]],
         output_dict: Dict[str, List[types.Artifact]],
         exec_properties: Dict[str, Any]):
    _executed_components.append(exec_properties[_COMPONENT_NAME])


def _get_fake_component(spec: types.ComponentSpec):
  component_id = spec.__class__.__name__.replace('_FakeComponentSpec',
                                                 '').lower()

  class _FakeComponent(base_component.BaseComponent):
    SPEC_CLASS = types.ComponentSpec
    EXECUTOR_SPEC = executor_spec_lib.ExecutorClassSpec(_FakeExecutor)

  return _FakeComponent(spec=spec).with_id(component_id)


class BeamDagRunnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    _executed_components.clear()

  def _getTestPipeline(  # pylint: disable=invalid-name
      self, platform_config: message.Message) -> pipeline_py.Pipeline:
    component_a = _get_fake_component(
        _FakeComponentSpecA(
            output=types.Channel(type=_ArtifactTypeA),
            component_name='_FakeComponent.a'))
    component_b = _get_fake_component(
        _FakeComponentSpecB(
            a=component_a.outputs['output'],
            output=types.Channel(type=_ArtifactTypeB),
            component_name='_FakeComponent.b'))
    component_c = _get_fake_component(
        _FakeComponentSpecC(
            a=component_a.outputs['output'],
            output=types.Channel(type=_ArtifactTypeC),
            component_name='_FakeComponent.c'))
    component_c.add_upstream_node(component_b)
    component_d = _get_fake_component(
        _FakeComponentSpecD(
            b=component_b.outputs['output'],
            c=component_c.outputs['output'],
            output=types.Channel(type=_ArtifactTypeD),
            component_name='_FakeComponent.d'))
    component_e = _get_fake_component(
        _FakeComponentSpecE(
            a=component_a.outputs['output'],
            b=component_b.outputs['output'],
            d=component_d.outputs['output'],
            output=types.Channel(type=_ArtifactTypeE),
            component_name='_FakeComponent.e'))

    temp_path = tempfile.mkdtemp()
    pipeline_root_path = os.path.join(temp_path, 'pipeline_root')
    metadata_path = os.path.join(temp_path, 'metadata.db')
    test_pipeline = pipeline_py.Pipeline(
        pipeline_name='test_pipeline',
        pipeline_root=pipeline_root_path,
        metadata_connection_config=sqlite_metadata_connection_config(
            metadata_path),
        components=[
            component_d, component_c, component_a, component_b, component_e
        ],
        platform_config=platform_config)
    return test_pipeline

  def _getTestPipelineIR(  # pylint: disable=invalid-name
      self, platform_config: message.Message) -> pipeline_pb2.Pipeline:
    test_pipeline = self._getTestPipeline(platform_config)
    c = compiler.Compiler()
    return c.compile(test_pipeline)

  @parameterized.named_parameters([{
      'testcase_name': 'LocalDeploymentConfig',
      'platform_config': _LOCAL_DEPLOYMENT_CONFIG,
  }, {
      'testcase_name': 'IntermediateDeploymentConfig',
      'platform_config': _INTERMEDIATE_DEPLOYMENT_CONFIG,
  }])
  def testRun(self, platform_config: message.Message):
    test_pipeline = self._getTestPipeline(platform_config)
    beam_dag_runner.BeamDagRunner().run(test_pipeline)
    self.assertEqual(_executed_components, [
        '_FakeComponent.a', '_FakeComponent.b', '_FakeComponent.c',
        '_FakeComponent.d', '_FakeComponent.e'
    ])

  @parameterized.named_parameters([{
      'testcase_name':
          'LocalDeploymentConfig',
      'platform_config':
          text_format.Parse(
              """
              metadata_connection_config {
                  fake_database {}
              }
              """, local_deployment_config_pb2.LocalDeploymentConfig()),
      'partial_run_nodes':
          None,
      'expected_executed_components': [
          '_FakeComponent.a', '_FakeComponent.b', '_FakeComponent.c',
          '_FakeComponent.d', '_FakeComponent.e'
      ],
  }, {
      'testcase_name':
          'IntermediateDeploymentConfig',
      'platform_config':
          text_format.Parse(
              """
              metadata_connection_config {
                [type.googleapis.com/ml_metadata.ConnectionConfig] {
                  fake_database {}
                }
              }
              """, pipeline_pb2.IntermediateDeploymentConfig()),
      'partial_run_nodes':
          None,
      'expected_executed_components': [
          '_FakeComponent.a', '_FakeComponent.b', '_FakeComponent.c',
          '_FakeComponent.d', '_FakeComponent.e'
      ],
  }, {
      'testcase_name':
          'Partial_LocalDeploymentConfig',
      'platform_config':
          text_format.Parse(
              """
              metadata_connection_config {
                  fake_database {}
              }
              """, local_deployment_config_pb2.LocalDeploymentConfig()),
      'partial_run_nodes': ['a'],
      'expected_executed_components': ['_FakeComponent.a'],
  }, {
      'testcase_name':
          'Partial_IntermediateDeploymentConfig',
      'platform_config':
          text_format.Parse(
              """
              metadata_connection_config {
                [type.googleapis.com/ml_metadata.ConnectionConfig] {
                  fake_database {}
                }
              }
              """, pipeline_pb2.IntermediateDeploymentConfig()),
      'partial_run_nodes': ['a'],
      'expected_executed_components': ['_FakeComponent.a'],
  }])
  def testRunWithIR(self, platform_config: message.Message,
                    partial_run_nodes: Optional[List[str]],
                    expected_executed_components: List[str]):
    test_pipeline = self._getTestPipelineIR(platform_config)
    if partial_run_nodes is not None:
      partial_run_utils.mark_pipeline(
          test_pipeline,
          from_nodes=lambda node_id: node_id in partial_run_nodes,
          to_nodes=lambda node_id: node_id in partial_run_nodes)
    beam_dag_runner.BeamDagRunner().run_with_ir(test_pipeline)
    self.assertEqual(_executed_components, expected_executed_components)

  def testLegacyBeamDagRunnerConstruction(self):
    self.assertIsInstance(beam_dag_runner.BeamDagRunner(),
                          beam_dag_runner.BeamDagRunner)

    # Test that the legacy Beam DAG runner is used when a PipelineConfig is
    # specified.
    config = pipeline_config.PipelineConfig()
    runner = beam_dag_runner.BeamDagRunner(config=config)
    self.assertIs(runner.__class__, legacy_beam_dag_runner.BeamDagRunner)
    self.assertIs(runner._config, config)

    # Test that the legacy Beam DAG runner is used when beam_orchestrator_args
    # is specified.
    beam_orchestrator_args = ['--my-beam-option']
    runner = beam_dag_runner.BeamDagRunner(
        beam_orchestrator_args=beam_orchestrator_args)
    self.assertIs(runner.__class__, legacy_beam_dag_runner.BeamDagRunner)
    self.assertIs(runner._beam_orchestrator_args, beam_orchestrator_args)

    # Test that the legacy Beam DAG runner is used when both a PipelineConfig
    # and beam_orchestrator_args are specified.
    config = pipeline_config.PipelineConfig()
    beam_orchestrator_args = ['--my-beam-option']
    runner = beam_dag_runner.BeamDagRunner(
        config=config, beam_orchestrator_args=beam_orchestrator_args)
    self.assertIs(runner.__class__, legacy_beam_dag_runner.BeamDagRunner)
    self.assertIs(runner._config, config)
    self.assertIs(runner._beam_orchestrator_args, beam_orchestrator_args)


if __name__ == '__main__':
  tf.test.main()
