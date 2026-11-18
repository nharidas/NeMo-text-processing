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
    NEMO_NOT_QUOTE,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.text_normalization.te.utils import get_abs_path, load_labels

major_singular_tsv = pynini.string_file(get_abs_path("data/money/major_singular.tsv"))
minor_plural_tsv = pynini.string_file(get_abs_path("data/money/minor_plural.tsv"))
minor_singular_tsv = pynini.string_file(get_abs_path("data/money/minor_singular.tsv"))
verbalize_sandhi = pynini.string_file(get_abs_path("data/money/verbalize_sandhi.tsv"))

_MINUS = dict(load_labels(get_abs_path("data/decimal/symbols.tsv")))["minus"]
_ZERO = dict(load_labels(get_abs_path("data/numbers/zero.tsv")))["0"]
_DIGIT_ONE = dict(load_labels(get_abs_path("data/numbers/digit.tsv")))["1"]
_ONE_SUFFIX = dict(load_labels(get_abs_path("data/numbers/ties_one_suffix.tsv")))["1"]
_CRORES = dict(load_labels(get_abs_path("data/numbers/magnitudes.tsv")))

_sigma = pynini.closure(NEMO_NOT_QUOTE)
_number_rewrite = (
    pynini.cdrewrite(
        verbalize_sandhi
        | pynini.cross(NEMO_SPACE + _DIGIT_ONE, NEMO_SPACE + _ONE_SUFFIX),
        "",
        "",
        _sigma,
    )
    @ pynini.cdrewrite(
        pynini.cross(NEMO_SPACE + _CRORES["crores_plural"], NEMO_SPACE + _CRORES["crores_before"]),
        "",
        NEMO_SPACE,
        _sigma,
    )
    @ pynini.cdrewrite(
        pynini.cross(
            NEMO_SPACE + _ONE_SUFFIX + NEMO_SPACE + _CRORES["crores_plural"],
            NEMO_SPACE + _ONE_SUFFIX + NEMO_SPACE + _CRORES["crores_before"],
        ),
        "వై",
        "",
        _sigma,
    )
).optimize()


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, 
    e.g. money { integer_part: "ఒకటి" currency_maj: "రూపాయలు" fractional_part: "యాభై" currency_min: "centiles" } -> ఒక రూపాయి యాభై పైసలు
    """
    def __init__(self, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        space = pynini.accep(NEMO_SPACE)
        optional_negative = pynini.closure(
            pynutil.delete('negative: "') + pynini.cross("true", _MINUS + " ") + pynutil.delete('"') + delete_space,
            0,
            1,
        )
        unit_one = pynini.cross(_DIGIT_ONE, _ONE_SUFFIX) | pynini.accep(_ONE_SUFFIX)
        not_exact_one = pynini.difference(
            pynini.closure(NEMO_NOT_QUOTE, 1),
            pynini.union(pynini.accep(_DIGIT_ONE), pynini.accep(_ONE_SUFFIX)),
        ).optimize()

        integer_one = pynutil.delete('integer_part: "') + unit_one + pynutil.delete('"')
        integer_other = (
            pynutil.delete('integer_part: "')
            + (pynini.difference(not_exact_one, pynini.accep(_ZERO)) @ _number_rewrite)
            + pynutil.delete('"')
        )
        fraction_one = pynutil.delete('fractional_part: "') + unit_one + pynutil.delete('"')
        fractional_part = (
            pynutil.delete('fractional_part: "') + (not_exact_one @ _number_rewrite) + pynutil.delete('"')
        )

        major_minor_graphs = []
        minor_graphs = []
        major_only_graphs = []

        for maj_pl, _ in load_labels(get_abs_path("data/money/major_singular.tsv")):
            graph_major_pl = pynutil.delete('currency_maj: "') + pynini.accep(maj_pl) + pynutil.delete('"')
            graph_major_sg = pynutil.delete('currency_maj: "') + (pynini.accep(maj_pl) @ major_singular_tsv) + pynutil.delete('"')
            graph_minor_pl = (
                pynutil.delete('currency_min: "')
                + pynini.cross("centiles", pynini.cross(maj_pl, maj_pl) @ minor_plural_tsv)
                + pynutil.delete('"')
            )
            graph_minor_sg = (
                pynutil.delete('currency_min: "')
                + pynini.cross("centiles", pynini.cross(maj_pl, maj_pl) @ minor_singular_tsv)
                + pynutil.delete('"')
            )
            fraction = (fraction_one + space + graph_minor_sg) | (fractional_part + space + graph_minor_pl)
            major = (integer_one + space + graph_major_sg) | (integer_other + space + graph_major_pl)

            major_only_graphs.append(major)
            major_minor_graphs.append(major + space + fraction)
            minor_graphs.append(
                pynutil.delete('integer_part: "')
                + pynutil.delete(_ZERO)
                + pynutil.delete('"')
                + delete_space
                + pynutil.delete('currency_maj: "')
                + pynutil.delete(maj_pl)
                + pynutil.delete('"')
                + delete_space
                + fraction
            )

        graph = pynini.union(*major_only_graphs, *major_minor_graphs, *minor_graphs)
        self.fst = self.delete_tokens(optional_negative + graph).optimize()