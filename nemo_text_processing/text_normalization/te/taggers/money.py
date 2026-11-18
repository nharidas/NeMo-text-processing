# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
    insert_space,
)
from nemo_text_processing.text_normalization.te.utils import get_abs_path

currency_graph = pynini.string_file(get_abs_path("data/money/currency.tsv"))
fraction_digit = pynini.string_file(get_abs_path("data/money/fraction_digit.tsv"))
normalize_fraction_digits = pynini.cdrewrite(
    fraction_digit, "", "", pynini.closure(NEMO_ALL_DIGIT)
).optimize()

two_digits_fractional_part = (
    pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(pynutil.delete("0"))
) @ (
    (pynutil.delete("0") + (NEMO_DIGIT - "0"))
    | ((NEMO_DIGIT - "0") + pynutil.insert("0"))
    | ((NEMO_DIGIT - "0") + NEMO_DIGIT)
)


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, 
    e.g. ₹1.50 -> money { currency_maj: "రూపాయలు" integer_part: "ఒకటి" fractional_part: "యాభై" currency_min: "centiles" }
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.final_graph
        optional_graph_negative = pynini.closure(
            pynutil.insert('negative: "') + pynini.cross("-", "true") + pynutil.insert('" ') + insert_space,
            0,
            1,
        )
        currency_major = pynutil.insert('currency_maj: "') + currency_graph + pynutil.insert('"')
        integer = (
            pynutil.insert('integer_part: "')
            + (pynini.closure(NEMO_ALL_DIGIT | pynutil.delete(",")) @ cardinal_graph)
            + pynutil.insert('"')
        )
        fraction = (
            pynutil.insert('fractional_part: "')
            + (
                pynini.closure(NEMO_ALL_DIGIT)
                @ normalize_fraction_digits
                @ two_digits_fractional_part
                @ cardinal_graph
            )
            + pynutil.insert('"')
        )
        currency_minor = pynutil.insert('currency_min: "') + pynutil.insert("centiles") + pynutil.insert('"')
        with_minor = (
            pynini.cross(".", " ") + fraction + insert_space + currency_minor
        )

        prefix_currency = currency_major + insert_space
        graph_major_only = (optional_graph_negative + prefix_currency + integer) | (
            prefix_currency + optional_graph_negative + insert_space + integer
        )
        graph_major_and_minor = (optional_graph_negative + prefix_currency + integer + with_minor) | (
            prefix_currency + optional_graph_negative + insert_space + integer + with_minor
        )

        self.fst = self.add_tokens((graph_major_only | graph_major_and_minor).optimize())