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
    TE_DEDH,
    TE_DHAI,
    TE_HALF_WORD,
    TE_QUARTER_WORD,
    TE_THREE_QUARTERS_WORD,
    NEMO_SPACE,
    GraphFst,
)

TE_ONE_HALF = "౧/౨"           # 1/2
TE_ONE_QUARTER = "౧/౪"        # 1/4
TE_THREE_QUARTERS = "౩/౪"     # 3/4


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying Telugu fractions.

    "౨౩ ౪/౬" ->
    fraction { integer_part: "ఇరవై మూడు" numerator: "నాలుగు" denominator: "ఆరు" }

    "౪/౬" ->
    fraction { numerator: "నాలుగు" denominator: "ఆరు" }

    "౧/౨" ->
    fraction { morphosyntactic_features: "అర" }

    "౩ ౩/౪" ->
    fraction { integer_part: "మూడు" morphosyntactic_features: "ముప్పావు" }
    """

    def __init__(self, cardinal, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph

        self.optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ")
            + pynini.cross("-", "\"true\"")
            + pynutil.insert(NEMO_SPACE),
            0,
            1,
        )

        self.integer = (
            pynutil.insert("integer_part: \"")
            + cardinal_graph
            + pynutil.insert("\"")
        )

        self.numerator = (
            pynutil.insert("numerator: \"")
            + cardinal_graph
            + pynini.cross(pynini.union("/", NEMO_SPACE + "/" + NEMO_SPACE), "\"")
            + pynutil.insert(NEMO_SPACE)
        )

        self.denominator = (
            pynutil.insert("denominator: \"")
            + cardinal_graph
            + pynutil.insert("\"")
        )

        # General fractions:
        # ౪/౬ -> numerator: "నాలుగు" denominator: "ఆరు"
        final_graph = (
            self.optional_graph_negative
            + pynini.closure(self.integer + pynini.accep(NEMO_SPACE), 0, 1)
            + self.numerator
            + self.denominator
        )

        # Special Telugu lexicalized mixed fractions:
        # ౧ ౧/౨ -> ఒకటిన్నర
        # ౨ ౧/౨ -> రెండున్నర
        dedh_dhai_graph = pynini.string_map(
            [
                ("౧" + NEMO_SPACE + TE_ONE_HALF, TE_DEDH),
                ("౨" + NEMO_SPACE + TE_ONE_HALF, TE_DHAI),
            ]
        )

        graph_dedh_dhai = (
            pynutil.insert("morphosyntactic_features: \"")
            + dedh_dhai_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        # Common simple Telugu fraction words:
        # ౧/౨ -> అర
        # ౧/౪ -> పావు
        # ౩/౪ -> ముప్పావు
        common_fraction_graph = pynini.string_map(
            [
                (TE_ONE_HALF, TE_HALF_WORD),
                (TE_ONE_QUARTER, TE_QUARTER_WORD),
                (TE_THREE_QUARTERS, TE_THREE_QUARTERS_WORD),
            ]
        )

        graph_common_fraction = (
            pynutil.insert("morphosyntactic_features: \"")
            + common_fraction_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        # Mixed common fractions:
        # ౧౩౩ ౧/౨ -> integer_part: "నూట ముప్పై మూడు" morphosyntactic_features: "అర"
        # ౩ ౩/౪ -> integer_part: "మూడు" morphosyntactic_features: "ముప్పావు"
        graph_mixed_common_fraction = (
            pynutil.insert("integer_part: \"")
            + cardinal_graph
            + pynini.cross(NEMO_SPACE, "\" ")
            + pynutil.insert("morphosyntactic_features: \"")
            + common_fraction_graph
            + pynutil.insert("\"")
            + pynutil.insert(NEMO_SPACE)
        )

        weighted_graph = (
            final_graph
            | pynutil.add_weight(graph_dedh_dhai, -0.3)
            | pynutil.add_weight(graph_common_fraction, -0.2)
            | pynutil.add_weight(graph_mixed_common_fraction, -0.2)
        )

        self.graph = weighted_graph

        graph = self.graph
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()