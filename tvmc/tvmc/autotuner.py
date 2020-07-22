# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
""" TVMC auto-tuning integration """
import logging
import time
from pathlib import Path

from tvm import autotvm
from tvm import relay
from tvm.autotvm.tuner import GATuner
from tvm.autotvm.tuner import GridSearchTuner
from tvm.autotvm.tuner import RandomTuner
from tvm.autotvm.tuner import XGBTuner

from tvmc.common import convert_graph_layout


def get_tuning_tasks(mod, params, target, target_host=None, alter_layout=None):
    """Get the tuning tasks for a given relay module.

    Parameters
    ----------
    mod : tvm.relay.Module
        The relay module from which to extract tuning tasks.
    params : dict
        The params for the relay module.
    target : str
        The compilation target.
    target_host : str, optional
        The compilation target for the host.
    alter_layout : str, optional
        The layout to convert the graph to. Note, the convert layout
        pass doesn't currently guarantee the whole of the graph will
        be converted to the chosen layout.

    Returns
    -------
    tasks : list
        A list of autotvm.Tasks.

    """
    if not target_host:
        target_host = target

    if alter_layout:
        mod = convert_graph_layout(mod, alter_layout)

    tasks = autotvm.task.extract_from_program(
        mod["main"],
        target=target,
        target_host=target_host,
        params=params,
        ops=(relay.op.get("nn.conv2d"),),
    )

    logging.debug("found %s tasks to be tuned", len(tasks))
    return tasks


def tune_tasks(
        tasks,
        log_file,
        measure_option,
        tuner,
        trials,
        early_stopping=None,
        use_transfer_learning=True,
):
    """Tune a list of tasks and output the history to a log file.

    Parameters
    ----------
    tasks : list
        A list of autotvm.Tasks to tune.
    log_file : str
        A file to output the tuning history.
    measure_option : autotvm.measure_option
        Options to build and run a tuning task.
    tuner : str
        Which tuner to use.
    trials : int
        The maximum number of tuning trials to perform.
    early_stopping : int, optional
        The minimum number of tuning trials to perform.
        This will be equal to 'trials' if not specified.
    template_keys : tuple, optional
        The template keys to try for each task.

    Returns
    -------

    """
    if not tasks:
        return

    if not early_stopping:
        early_stopping = trials

    logging.debug("tuner: %s", tuner)

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # Create a tuner
        if tuner in ("xgb", "xgb-rank"):
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # If transfer learning is being used, load the existing results
        if use_transfer_learning:
            if Path(log_file).is_file():
                start_time = time.time()
                tuner_obj.load_history(autotvm.record.load_from_file(log_file))
                logging.debug("loaded history in %s seconds", str(time.time() - start_time))

        tuner_obj.tune(
            n_trial=min(trials, len(tsk.config_space)),
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(trials, prefix=prefix),
                autotvm.callback.log_to_file(log_file),
            ],
        )
