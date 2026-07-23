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
    NEMO_ALL_ZERO,
    NEMO_DIGIT,
    NEMO_TE_DIGIT,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.te.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        -౨౩ -> cardinal { negative: "true"  integer: "ఇరవై మూడు" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        teens_te = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
        teens_en = pynini.string_file(get_abs_path("data/numbers/teens_and_ties_en.tsv"))
        ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))        
        teens_ties_thousand = pynutil.add_weight(
            pynini.string_file(get_abs_path("data/numbers/teens_and_ties_thousand.tsv")),
            -0.2,
        )

        magnitude = {}
        with open(get_abs_path("data/numbers/magnitudes.tsv"), encoding="utf-8") as magnitude_file:
            for magnitude_line in magnitude_file:
                magnitude_line = magnitude_line.rstrip("\r\n")
                if not magnitude_line:
                    continue
                magnitude_key, magnitude_word = magnitude_line.split("\t")
                magnitude[magnitude_key] = magnitude_word

        hundred = magnitude["hundred"]
        hundred_prefix = magnitude["hundred_prefix"] + " "
        thousand = magnitude["thousand"]
        lakh = magnitude["lakh"]
        crore = magnitude["crore"]
        hundreds_before_one = " " + magnitude["hundreds_before"] + " " + magnitude["one"]

        ins_hundreds_plural = pynutil.insert(" " + magnitude["hundreds_plural"])
        ins_hundreds_before = pynutil.insert(" " + magnitude["hundreds_before"])
        ins_thousand = pynutil.insert(thousand)
        ins_thousand_spaced = pynutil.insert(" " + thousand)
        ins_thousands_plural = pynutil.insert(" " + magnitude["thousands_plural"])
        ins_thousands_before = pynutil.insert(" " + magnitude["thousands_before"])
        ins_lakh = pynutil.insert(lakh)
        ins_lakh_spaced = pynutil.insert(" " + lakh)
        ins_lakha_digit = pynutil.insert(magnitude["lakh_before_digit"])
        ins_lakhs_plural = pynutil.insert(" " + magnitude["lakhs_plural"])
        ins_lakhs_before = pynutil.insert(" " + magnitude["lakhs_before"])
        ins_crore = pynutil.insert(crore)
        ins_crore_spaced = pynutil.insert(" " + crore)
        ins_crores_plural = pynutil.insert(" " + magnitude["crores_plural"])
        ins_crores_before = pynutil.insert(" " + magnitude["crores_before"])
        
        digit_en = (NEMO_DIGIT @ digit).optimize()
        digit_te = (NEMO_TE_DIGIT @ digit).optimize()
        ties_en = (NEMO_DIGIT @ ties).optimize()
        ties_te = (NEMO_TE_DIGIT @ ties).optimize()

        teens_ties_en = teens_en | (ties_en + pynutil.delete("0")) | (ties_en + insert_space + digit_en)
        teens_ties_te = teens_te | (ties_te + pynutil.delete("౦")) | (ties_te + insert_space + digit_te)
        teens_ties = pynini.union(teens_ties_te, teens_ties_en)

        single_digit_graph = digit | zero
        self.single_digits_graph = single_digit_graph + pynini.closure(insert_space + single_digit_graph)

        delete_zero = pynutil.delete(NEMO_ALL_ZERO)

        zero_pow = {0: pynini.accep("")}
        for _n in range(1, 8):
            zero_pow[_n] = (zero_pow[_n - 1] + delete_zero).optimize()

        def create_graph_suffix(digit_graph, suffix, zeros_counts):
            if zeros_counts == 0:
                return digit_graph + suffix

            return digit_graph + zero_pow[zeros_counts] + suffix

        def create_larger_number_graph(digit_graph, suffix, zeros_counts, sub_graph):
            if zeros_counts == 0:
                return digit_graph + suffix + insert_space + sub_graph

            return digit_graph + suffix + zero_pow[zeros_counts] + insert_space + sub_graph

        def build_group(prefix, rung_suffix, ladder, head_suffix=None, head_zeros=None):
            """Union over a "ladder" of (zeros_count, remainder_graph) rungs."""
            graph = create_graph_suffix(prefix, head_suffix, head_zeros) if head_suffix is not None else None
            for zeros, sub in ladder:
                rung = create_larger_number_graph(prefix, rung_suffix, zeros, sub)
                graph = rung if graph is None else graph | rung
            return graph

        one_digit = pynini.union("1", "౧")
        digit_except_one = (pynini.difference(NEMO_ALL_DIGIT, NEMO_ALL_ZERO | one_digit) @ digit).optimize()

        one_prefix = pynutil.delete(one_digit)

        graph_hundreds = pynini.cross("100", hundred) | pynini.cross("౧౦౦", hundred)
        graph_hundreds |= (pynini.cross("10", hundred_prefix) | pynini.cross("౧౦", hundred_prefix)) + digit
        graph_hundreds |= (pynini.cross("1", hundred_prefix) | pynini.cross("౧", hundred_prefix)) + teens_ties
        graph_hundreds |= create_graph_suffix(digit_except_one, ins_hundreds_plural, 2)
        graph_hundreds |= create_larger_number_graph(digit_except_one, ins_hundreds_before, 1, digit)
        graph_hundreds |= create_larger_number_graph(digit_except_one, ins_hundreds_before, 0, teens_ties)
        graph_hundreds = graph_hundreds.optimize()

        thousand_ladder = [
            (2, digit),
            (1, teens_ties),
            (0, graph_hundreds),
        ]

        graph_thousands = pynini.cross("1000", thousand) | pynini.cross("౧౦౦౦", thousand)
        graph_thousands |= build_group(one_prefix, ins_thousand, thousand_ladder)
        graph_thousands |= build_group(
            digit_except_one, ins_thousands_before, thousand_ladder, head_suffix=ins_thousands_plural, head_zeros=3
        )
        graph_thousands = graph_thousands.optimize()

        graph_ten_thousands = build_group(
            teens_ties_thousand, ins_thousands_before, thousand_ladder, head_suffix=ins_thousand_spaced, head_zeros=3
        )
        graph_ten_thousands |= build_group(
            teens_ties, ins_thousands_before, thousand_ladder, head_suffix=ins_thousands_plural, head_zeros=3
        )
        graph_ten_thousands = graph_ten_thousands.optimize()

        lakh_ladder = [
            (4, digit),
            (3, teens_ties),
            (2, graph_hundreds),
            (1, graph_thousands),
            (0, graph_ten_thousands),
        ]

        graph_lakhs = pynini.cross("100000", lakh) | pynini.cross("౧౦౦౦౦౦", lakh)
        graph_lakhs |= create_larger_number_graph(one_prefix, ins_lakha_digit, 4, digit)
        graph_lakhs |= build_group(one_prefix, ins_lakh, lakh_ladder[1:])
        graph_lakhs |= build_group(
            digit_except_one, ins_lakhs_before, lakh_ladder, head_suffix=ins_lakhs_plural, head_zeros=5
        )
        graph_lakhs = graph_lakhs.optimize()

        graph_ten_lakhs = build_group(
            teens_ties_thousand, ins_lakhs_before, lakh_ladder, head_suffix=ins_lakh_spaced, head_zeros=5
        )
        graph_ten_lakhs |= build_group(
            teens_ties, ins_lakhs_before, lakh_ladder, head_suffix=ins_lakhs_plural, head_zeros=5
        )
        graph_ten_lakhs = graph_ten_lakhs.optimize()

        crore_ladder = [
            (6, digit),
            (5, teens_ties),
            (4, graph_hundreds),
            (3, graph_thousands),
            (2, graph_ten_thousands),
            (1, graph_lakhs),
            (0, graph_ten_lakhs),
        ]

        graph_crores = pynini.cross("10000000", crore) | pynini.cross("౧౦౦౦౦౦౦౦", crore)
        graph_crores |= build_group(one_prefix, ins_crore, crore_ladder)
        graph_crores |= build_group(
            digit_except_one, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
        )
        graph_crores = graph_crores.optimize()

        graph_ten_crores = build_group(
            teens_ties_thousand, ins_crore_spaced, crore_ladder, head_suffix=ins_crore_spaced, head_zeros=7
        )
        graph_ten_crores |= build_group(
            teens_ties, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
        )
        graph_ten_crores = graph_ten_crores.optimize()

        hundred_crore_prefix = (
            pynini.cross("100", hundred)
            | pynini.cross("౧౦౦", hundred)
            | ((pynini.cross("10", hundred_prefix) | pynini.cross("౧౦", hundred_prefix)) + digit)
            | ((pynini.cross("1", hundred_prefix) | pynini.cross("౧", hundred_prefix)) + teens_ties)
            | create_graph_suffix(digit_except_one, ins_hundreds_before, 2)
            | create_larger_number_graph(digit_except_one, ins_hundreds_before, 1, digit_except_one)
            | create_larger_number_graph(digit_except_one, ins_hundreds_before, 0, teens_ties)
        ).optimize()

        hundred_one_crore_prefix = (
            digit_except_one
            + pynutil.delete(NEMO_ALL_ZERO | pynini.accep("౦"))
            + (pynini.cross("1", hundreds_before_one) | pynini.cross("౧", hundreds_before_one))
        )

        graph_arabs = build_group(
            hundred_one_crore_prefix, ins_crore_spaced, crore_ladder, head_suffix=ins_crore_spaced, head_zeros=7
        )
        graph_arabs |= build_group(
            hundred_crore_prefix, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
        )
        graph_arabs |= create_larger_number_graph(hundred_crore_prefix, ins_crores_before, 0, graph_crores)
        graph_arabs = graph_arabs.optimize()

        thousand_crore_ladder = [
            (2, digit),
            (1, teens_ties),
            (0, hundred_crore_prefix),
        ]

        thousand_crore_prefix = (
            pynini.cross("1000", thousand)
            | pynini.cross("౧౦౦౦", thousand)
            | build_group(
                digit_except_one, ins_thousands_before, thousand_crore_ladder, head_suffix=ins_thousands_before, head_zeros=3
            )
            | build_group(one_prefix, ins_thousand, thousand_crore_ladder)
        ).optimize()

        ten_thousand_crore_prefix = (
            build_group(
                teens_ties_thousand, ins_thousands_before, thousand_crore_ladder, head_suffix=ins_thousand_spaced, head_zeros=3
            )
            | build_group(
                teens_ties, ins_thousands_before, thousand_crore_ladder, head_suffix=ins_thousands_before, head_zeros=3
            )
        ).optimize()

        crore_count_prefix = (thousand_crore_prefix | ten_thousand_crore_prefix).optimize()

        graph_ten_arabs = pynutil.add_weight(
            build_group(
                crore_count_prefix, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
            ),
            -0.1,
        ).optimize()

        graph_kharabs = pynutil.add_weight(
            build_group(
                ten_thousand_crore_prefix, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
            ),
            -0.2,
        ).optimize()

        lakh_crore_ladder = [
            (4, digit),
            (3, teens_ties),
            (2, graph_hundreds),
            (1, graph_thousands),
            (0, ten_thousand_crore_prefix),
        ]

        lakh_crore_prefix = (
            pynini.cross("100000", lakh)
            | pynini.cross("౧౦౦౦౦౦", lakh)
            | build_group(one_prefix, ins_lakh, lakh_crore_ladder)
            | build_group(digit_except_one, ins_lakhs_before, lakh_crore_ladder, head_suffix=ins_lakhs_before, head_zeros=5)
        ).optimize()

        ten_lakh_crore_prefix = (
            build_group(teens_ties_thousand, ins_lakhs_before, lakh_crore_ladder, head_suffix=ins_lakh_spaced, head_zeros=5)
            | build_group(teens_ties, ins_lakhs_before, lakh_crore_ladder, head_suffix=ins_lakhs_before, head_zeros=5)
        ).optimize()

        lakh_crore_count_prefix = (lakh_crore_prefix | ten_lakh_crore_prefix).optimize()

        graph_ten_kharabs = pynutil.add_weight(
            build_group(
                lakh_crore_count_prefix, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
            ),
            -0.3,
        ).optimize()

        graph_nils = pynutil.add_weight(
            build_group(
                lakh_crore_count_prefix, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
            ),
            -0.4,
        ).optimize()

        ten_nil_lakh_remainder_before_kotlu = build_group(
            teens_ties, ins_lakhs_before, lakh_ladder, head_suffix=ins_lakhs_before, head_zeros=5
        ).optimize()

        koti_ladder = [
            (6, digit),
            (5, teens_ties),
            (4, graph_hundreds),
            (3, graph_thousands),
            (2, graph_ten_thousands),
            (1, graph_lakhs),
            (0, ten_nil_lakh_remainder_before_kotlu),
        ]

        ten_nil_crore_count_prefix = (
            graph_crores
            | graph_ten_crores
            | create_larger_number_graph(one_prefix, ins_crore, 0, ten_nil_lakh_remainder_before_kotlu)
            | create_larger_number_graph(digit_except_one, ins_crores_before, 0, ten_nil_lakh_remainder_before_kotlu)
            | create_larger_number_graph(teens_ties, ins_crores_before, 0, ten_nil_lakh_remainder_before_kotlu)
        ).optimize()

        graph_ten_nils = pynutil.add_weight(
            build_group(
                ten_nil_crore_count_prefix, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
            ),
            -0.5,
        ).optimize()

        padma_crore_count_prefix = (
            build_group(teens_ties_thousand, ins_crore_spaced, koti_ladder, head_suffix=ins_crore_spaced, head_zeros=7)
            | build_group(teens_ties, ins_crore_spaced, koti_ladder, head_suffix=ins_crore_spaced, head_zeros=7)
        ).optimize()

        graph_padmas = pynutil.add_weight(
            build_group(
                padma_crore_count_prefix, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
            ),
            -0.6,
        ).optimize()

        ten_padma_one_crore_count_prefix = create_graph_suffix(hundred_one_crore_prefix, ins_crore_spaced, 7).optimize()

        ten_padma_crore_count_prefix = (
            ten_padma_one_crore_count_prefix
            | build_group(hundred_crore_prefix, ins_crore_spaced, koti_ladder, head_suffix=ins_crore_spaced, head_zeros=7)
        ).optimize()

        graph_ten_padmas = pynutil.add_weight(
            build_group(
                ten_padma_crore_count_prefix, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
            ),
            -0.7,
        ).optimize()

        shankh_koti_count_prefix = build_group(
            crore_count_prefix, ins_crore_spaced, koti_ladder, head_suffix=ins_crore_spaced, head_zeros=7
        ).optimize()

        graph_shankhs = pynutil.add_weight(
            build_group(
                shankh_koti_count_prefix, ins_crores_before, crore_ladder, head_suffix=ins_crores_plural, head_zeros=7
            ),
            -0.8,
        ).optimize()

        graph_without_leading_zeros = (
            digit
            | zero
            | teens_ties
            | graph_hundreds
            | graph_thousands
            | graph_ten_thousands
            | graph_lakhs
            | graph_ten_lakhs
            | graph_crores
            | graph_ten_crores
            | graph_arabs
            | graph_ten_arabs
            | graph_kharabs
            | graph_ten_kharabs
            | graph_nils
            | graph_ten_nils
            | graph_padmas
            | graph_ten_padmas
            | graph_shankhs
        )

        cardinal_with_leading_zeros = pynini.compose(
            NEMO_ALL_ZERO + pynini.closure(NEMO_ALL_DIGIT), self.single_digits_graph
        )

        final_graph = graph_without_leading_zeros | cardinal_with_leading_zeros

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        self.final_graph = final_graph.optimize()
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph
