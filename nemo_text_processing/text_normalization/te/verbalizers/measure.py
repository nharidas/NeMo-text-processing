# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.te.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.te.utils import get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure, e.g.
        measure { cardinal { integer: "ఒకటి" } units: "యోటాబైట్" } -> ఒక యోటాబైట్
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        number = pynini.union(
            pynutil.delete("cardinal {") + delete_space + cardinal.numbers + delete_space + pynutil.delete("}"),
            pynutil.delete("decimal {") + delete_space + decimal.numbers + delete_space + pynutil.delete("}"),
        )
        graph = number + pynini.closure(
            delete_space
            + insert_space
            + pynutil.delete('units: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"'),
            0,
            1,
        )
        verbalize_sandhi = pynini.string_file(get_abs_path("data/money/verbalize_sandhi.tsv"))
        postprocess = pynini.cdrewrite(verbalize_sandhi, "", "", NEMO_SIGMA).optimize()
        self.fst = self.delete_tokens(graph @ postprocess).optimize()