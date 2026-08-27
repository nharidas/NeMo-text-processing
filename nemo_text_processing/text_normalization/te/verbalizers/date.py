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
    NEMO_SPACE,
    GraphFst,
    delete_space,
)


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing Telugu date, e.g.
        date { day: "ఆరు" month: "మార్చి" year: "రెండు వేల పది" } -> "ఆరు మార్చి రెండు వేల పది"
        date { month: "మార్చి" day: "ఆరు" } -> "మార్చి ఆరు"
        date { era: "పంతొమ్మిది వందల తొంభైలో" } -> "పంతొమ్మిది వందల తొంభైలో"
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")

        day = pynutil.delete("day: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        month = pynutil.delete("month: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        year = pynutil.delete("year: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        graph_era = pynutil.delete("era: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        graph_dd_mm = day + NEMO_SPACE + month
        graph_mm_dd = month + NEMO_SPACE + day
        graph_dd_mm_yyyy = day + NEMO_SPACE + month + NEMO_SPACE + year
        graph_mm_dd_yyyy = month + NEMO_SPACE + day + NEMO_SPACE + year
        graph_mm_yyyy = month + NEMO_SPACE + year

        optional_preserve_order = pynini.closure(
            pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
            | pynutil.delete("field_order:")
            + delete_space
            + pynutil.delete("\"")
            + NEMO_NOT_QUOTE
            + pynutil.delete("\"")
            + delete_space
        )

        self.graph = (
            (graph_dd_mm | graph_mm_dd | graph_dd_mm_yyyy | graph_mm_dd_yyyy | graph_mm_yyyy | graph_era)
            + delete_space
            + optional_preserve_order
        )

        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
