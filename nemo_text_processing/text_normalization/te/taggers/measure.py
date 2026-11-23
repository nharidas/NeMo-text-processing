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
    NEMO_ALL_DIGIT,
    NEMO_DIGIT,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.te.utils import get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, e.g.
        -5 kg -> measure { cardinal { negative: "true" integer: "ఐదు" } units: "కిలోగ్రాములు" }
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        te_digit = pynini.difference(NEMO_ALL_DIGIT, NEMO_DIGIT).optimize()
        comma = pynini.accep(",")
        one = pynini.union(pynini.accep("1"), pynini.accep("౧"))
        ties_one_suffix = pynini.string_file(get_abs_path("data/numbers/ties_one_suffix.tsv"))

        singular_cardinal = pynini.compose(one, ties_one_suffix)
        plural_digits = pynini.union(
            pynini.closure(NEMO_ALL_DIGIT, 2),
            pynini.difference(NEMO_ALL_DIGIT, one),
        )
        plural_cardinal = pynini.compose(
            pynini.closure(plural_digits | pynutil.delete(comma), 1),
            cardinal.final_graph,
        )

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", '"true" '),
            0,
            1,
        )

        decimal_core = pynini.union(
            *[
                pynutil.insert('integer_part: "')
                + pynini.compose(
                    pynini.accep(zero) | (pynini.difference(digits, zero) + pynini.closure(digits | comma, 0)),
                    cardinal.final_graph,
                )
                + pynutil.insert('"')
                + pynutil.delete(".")
                + insert_space
                + pynutil.insert('fractional_part: "')
                + pynini.compose(pynini.closure(digits, 1), cardinal.single_digits_graph)
                + pynutil.insert('"')
                for zero, digits in (("0", NEMO_DIGIT), ("౦", te_digit))
            ]
        ).optimize()

        unit_singular = pynini.string_file(get_abs_path("data/measure/unit.tsv")).optimize()
        unit_plural = pynini.string_file(get_abs_path("data/measure/unit_plural.tsv")).optimize()
        opt_space = pynini.closure(delete_space, 0, 1)

        def units(body):
            return opt_space + pynutil.insert('units: "') + body + pynutil.insert('"')

        def cardinal_measure(number_graph, unit_graph):
            return (
                pynutil.insert("cardinal { ")
                + optional_graph_negative
                + pynutil.insert('integer: "')
                + number_graph
                + pynutil.insert('" }')
                + units(unit_graph)
            )

        decimal_measure = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal_core
            + pynutil.insert(" }")
            + units(unit_plural)
        )

        by_sep = pynini.string_file(get_abs_path("data/measure/dimension_separator.tsv"))
        dimension = (
            pynutil.insert('cardinal { integer: "')
            + plural_cardinal
            + by_sep
            + plural_cardinal
            + pynutil.insert('" }')
        )

        graph = (
            cardinal_measure(singular_cardinal, unit_singular)
            | cardinal_measure(plural_cardinal, unit_plural)
            | decimal_measure
            | dimension
        )
        self.fst = self.add_tokens(graph.optimize())