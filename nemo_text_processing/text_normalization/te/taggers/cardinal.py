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
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.te.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying Telugu cardinals.
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        teens_ties_hi = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
        teens_ties_en = pynini.string_file(get_abs_path("data/numbers/teens_and_ties_en.tsv"))
        teens_ties = pynini.union(teens_ties_hi, teens_ties_en)
        teens_and_ties = pynutil.add_weight(teens_ties, -0.1)

        self.digit = digit
        self.zero = zero
        self.teens_and_ties = teens_and_ties

        single_digit_graph = digit | zero
        self.single_digits_graph = single_digit_graph + pynini.closure(insert_space + single_digit_graph)

        def create_graph_suffix(digit_graph, suffix, zeros_counts):
            zero_delete = pynutil.add_weight(pynutil.delete(NEMO_ALL_ZERO | pynini.accep("౦")), -0.1)
            if zeros_counts == 0:
                return digit_graph + suffix
            return digit_graph + (zero_delete ** zeros_counts) + suffix

        def create_larger_number_graph(digit_graph, suffix, zeros_counts, sub_graph):
            zero_delete = pynutil.add_weight(pynutil.delete(NEMO_ALL_ZERO | pynini.accep("౦")), -0.1)
            if zeros_counts == 0:
                return digit_graph + suffix + pynutil.insert(" ") + sub_graph
            return digit_graph + suffix + (zero_delete ** zeros_counts) + pynutil.insert(" ") + sub_graph

        # -------------------------
        # Helpers
        # -------------------------

        one = (pynini.cross("1", "ఒక") | pynini.cross("౧", "ఒక")).optimize()
        one_empty = (pynini.cross("1", "") | pynini.cross("౧", "")).optimize()

        digit_except_one = (
            pynini.union(
                "2", "3", "4", "5", "6", "7", "8", "9",
                "౨", "౩", "౪", "౫", "౬", "౭", "౮", "౯"
            ) @ digit
        ).optimize()

        digit_for_scale = (one | digit_except_one).optimize()

        one_ties_for_scale = pynutil.add_weight(
            pynini.union(
                pynini.cross("21", "ఇరవై ఒక"),
                pynini.cross("31", "ముప్పై ఒక"),
                pynini.cross("41", "నలభై ఒక"),
                pynini.cross("51", "యాభై ఒక"),
                pynini.cross("61", "అరవై ఒక"),
                pynini.cross("71", "డెబ్బై ఒక"),
                pynini.cross("81", "ఎనభై ఒక"),
                pynini.cross("91", "తొంభై ఒక"),
                pynini.cross("౨౧", "ఇరవై ఒక"),
                pynini.cross("౩౧", "ముప్పై ఒక"),
                pynini.cross("౪౧", "నలభై ఒక"),
                pynini.cross("౫౧", "యాభై ఒక"),
                pynini.cross("౬౧", "అరవై ఒక"),
                pynini.cross("౭౧", "డెబ్బై ఒక"),
                pynini.cross("౮౧", "ఎనభై ఒక"),
                pynini.cross("౯౧", "తొంభై ఒక"),
            ),
            -0.2,
        ).optimize()

        teens_and_ties_for_scale = (one_ties_for_scale | teens_and_ties).optimize()

        # -------------------------
        # Hundreds
        # -------------------------

        graph_hundreds = pynini.cross("100", "వంద") | pynini.cross("౧౦౦", "వంద")
        graph_hundreds |= (pynini.cross("10", "నూట ") | pynini.cross("౧౦", "నూట ")) + digit
        graph_hundreds |= (pynini.cross("1", "నూట ") | pynini.cross("౧", "నూట ")) + teens_ties
        graph_hundreds |= create_graph_suffix(digit_except_one, pynutil.insert(" వందలు"), 2)
        graph_hundreds |= create_larger_number_graph(digit_except_one, pynutil.insert(" వందల"), 1, digit)
        graph_hundreds |= create_larger_number_graph(digit_except_one, pynutil.insert(" వందల"), 0, teens_ties)
        graph_hundreds = graph_hundreds.optimize()
        self.graph_hundreds = graph_hundreds

        # Count form before scale nouns:
        # రెండు వందల కోట్లు, not రెండు వందలు కోట్లు
        graph_hundreds_count = pynini.cross("100", "వంద") | pynini.cross("౧౦౦", "వంద")
        graph_hundreds_count |= (pynini.cross("10", "నూట ") | pynini.cross("౧౦", "నూట ")) + digit_for_scale
        graph_hundreds_count |= (pynini.cross("1", "నూట ") | pynini.cross("౧", "నూట ")) + teens_and_ties_for_scale
        graph_hundreds_count |= create_graph_suffix(digit_except_one, pynutil.insert(" వందల"), 2)
        graph_hundreds_count |= create_larger_number_graph(digit_except_one, pynutil.insert(" వందల"), 1, digit_for_scale)
        graph_hundreds_count |= create_larger_number_graph(
            digit_except_one, pynutil.insert(" వందల"), 0, teens_and_ties_for_scale
        )
        graph_hundreds_count = graph_hundreds_count.optimize()

        # -------------------------
        # Thousands / Ten thousands
        # -------------------------

        suffix_thousand = pynutil.insert(" వెయ్యి")
        suffix_thousands_spoken = pynutil.insert(" వేలు")
        suffix_thousands_oblique = pynutil.insert(" వేల")

        graph_thousands = pynini.cross("1000", "వెయ్యి") | pynini.cross("౧౦౦౦", "వెయ్యి")
        graph_thousands |= create_larger_number_graph(one_empty, suffix_thousand, 2, digit)
        graph_thousands |= create_larger_number_graph(one_empty, suffix_thousand, 1, teens_ties)
        graph_thousands |= create_larger_number_graph(one_empty, suffix_thousand, 0, graph_hundreds)

        graph_thousands |= create_graph_suffix(digit_except_one, suffix_thousands_spoken, 3)
        graph_thousands |= create_larger_number_graph(digit_except_one, suffix_thousands_oblique, 2, digit)
        graph_thousands |= create_larger_number_graph(digit_except_one, suffix_thousands_oblique, 1, teens_ties)
        graph_thousands |= create_larger_number_graph(digit_except_one, suffix_thousands_oblique, 0, graph_hundreds)
        graph_thousands = graph_thousands.optimize()
        self.graph_thousands = graph_thousands

        ten_thousands_ending_one = (
            pynini.union(
                "21", "31", "41", "51", "61", "71", "81", "91",
                "౨౧", "౩౧", "౪౧", "౫౧", "౬౧", "౭౧", "౮౧", "౯౧"
            ) @ one_ties_for_scale
        ).optimize()

        graph_ten_thousands = create_graph_suffix(ten_thousands_ending_one, suffix_thousand, 3)
        graph_ten_thousands |= create_graph_suffix(teens_and_ties, suffix_thousands_spoken, 3)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties_for_scale, suffix_thousands_oblique, 2, digit)
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties_for_scale, suffix_thousands_oblique, 1, teens_ties)
        graph_ten_thousands |= create_larger_number_graph(
            teens_and_ties_for_scale, suffix_thousands_oblique, 0, graph_hundreds
        )

        graph_ten_thousands |= pynini.cross("21000", "ఇరవై ఒక వెయ్యి")
        graph_ten_thousands |= pynini.cross("31000", "ముప్పై ఒక వెయ్యి")
        graph_ten_thousands |= pynini.cross("41000", "నలభై ఒక వెయ్యి")
        graph_ten_thousands |= pynini.cross("51000", "యాభై ఒక వెయ్యి")
        graph_ten_thousands |= pynini.cross("61000", "అరవై ఒక వెయ్యి")
        graph_ten_thousands |= pynini.cross("71000", "డెబ్బై ఒక వెయ్యి")
        graph_ten_thousands |= pynini.cross("81000", "ఎనభై ఒక వెయ్యి")
        graph_ten_thousands |= pynini.cross("91000", "తొంభై ఒక వెయ్యి")

        graph_ten_thousands = graph_ten_thousands.optimize()
        self.graph_ten_thousands = graph_ten_thousands

        # Count-only thousands before "కోట్లు"
        graph_thousands_count = pynini.cross("1000", "వెయ్యి") | pynini.cross("౧౦౦౦", "వెయ్యి")
        graph_thousands_count |= create_larger_number_graph(one_empty, suffix_thousand, 2, digit_for_scale)
        graph_thousands_count |= create_larger_number_graph(one_empty, suffix_thousand, 1, teens_and_ties_for_scale)
        graph_thousands_count |= create_larger_number_graph(one_empty, suffix_thousand, 0, graph_hundreds_count)
        graph_thousands_count |= create_graph_suffix(digit_except_one, suffix_thousands_oblique, 3)
        graph_thousands_count |= create_larger_number_graph(digit_except_one, suffix_thousands_oblique, 2, digit_for_scale)
        graph_thousands_count |= create_larger_number_graph(
            digit_except_one, suffix_thousands_oblique, 1, teens_and_ties_for_scale
        )
        graph_thousands_count |= create_larger_number_graph(
            digit_except_one, suffix_thousands_oblique, 0, graph_hundreds_count
        )
        graph_thousands_count = graph_thousands_count.optimize()

        graph_ten_thousands_count = create_graph_suffix(teens_and_ties_for_scale, suffix_thousands_oblique, 3)
        graph_ten_thousands_count |= create_larger_number_graph(
            teens_and_ties_for_scale, suffix_thousands_oblique, 2, digit_for_scale
        )
        graph_ten_thousands_count |= create_larger_number_graph(
            teens_and_ties_for_scale, suffix_thousands_oblique, 1, teens_and_ties_for_scale
        )
        graph_ten_thousands_count |= create_larger_number_graph(
            teens_and_ties_for_scale, suffix_thousands_oblique, 0, graph_hundreds_count
        )
        graph_ten_thousands_count = graph_ten_thousands_count.optimize()

        # -------------------------
        # Lakhs / Ten lakhs
        # -------------------------

        suffix_lakh = pynutil.insert(" లక్ష")
        suffix_lakhs = pynutil.insert(" లక్షల")
        suffix_lakha = pynutil.insert(" లక్షా")

        graph_lakhs = create_graph_suffix(one, suffix_lakh, 5)
        graph_lakhs |= create_larger_number_graph(one, suffix_lakha, 4, digit)
        graph_lakhs |= create_larger_number_graph(one, suffix_lakh, 3, teens_ties)
        graph_lakhs |= create_larger_number_graph(one, suffix_lakh, 2, graph_hundreds)
        graph_lakhs |= create_larger_number_graph(one, suffix_lakh, 1, graph_thousands)
        graph_lakhs |= create_larger_number_graph(one, suffix_lakh, 0, graph_ten_thousands)

        graph_lakhs |= create_graph_suffix(digit_except_one, suffix_lakhs, 5)
        graph_lakhs |= create_larger_number_graph(digit_except_one, suffix_lakhs, 4, digit)
        graph_lakhs |= create_larger_number_graph(digit_except_one, suffix_lakhs, 3, teens_ties)
        graph_lakhs |= create_larger_number_graph(digit_except_one, suffix_lakhs, 2, graph_hundreds)
        graph_lakhs |= create_larger_number_graph(digit_except_one, suffix_lakhs, 1, graph_thousands)
        graph_lakhs |= create_larger_number_graph(digit_except_one, suffix_lakhs, 0, graph_ten_thousands)

        graph_lakhs = graph_lakhs.optimize()
        self.graph_lakhs = graph_lakhs

        graph_ten_lakhs = create_graph_suffix(teens_and_ties_for_scale, suffix_lakhs, 5)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties_for_scale, suffix_lakhs, 4, digit)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties_for_scale, suffix_lakhs, 3, teens_ties)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties_for_scale, suffix_lakhs, 2, graph_hundreds)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties_for_scale, suffix_lakhs, 1, graph_thousands)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties_for_scale, suffix_lakhs, 0, graph_ten_thousands)
        graph_ten_lakhs = graph_ten_lakhs.optimize()
        self.graph_ten_lakhs = graph_ten_lakhs

        # Count-only lakhs before "కోట్లు"
        graph_lakhs_count = pynini.cross("100000", "లక్ష") | pynini.cross("౧౦౦౦౦౦", "లక్ష")
        graph_lakhs_count |= create_larger_number_graph(one_empty, suffix_lakh, 4, digit_for_scale)
        graph_lakhs_count |= create_larger_number_graph(one_empty, suffix_lakh, 3, teens_and_ties_for_scale)
        graph_lakhs_count |= create_larger_number_graph(one_empty, suffix_lakh, 2, graph_hundreds_count)
        graph_lakhs_count |= create_larger_number_graph(one_empty, suffix_lakh, 1, graph_thousands_count)
        graph_lakhs_count |= create_larger_number_graph(one_empty, suffix_lakh, 0, graph_ten_thousands_count)

        graph_lakhs_count |= create_graph_suffix(digit_except_one, suffix_lakhs, 5)
        graph_lakhs_count |= create_larger_number_graph(digit_except_one, suffix_lakhs, 4, digit_for_scale)
        graph_lakhs_count |= create_larger_number_graph(digit_except_one, suffix_lakhs, 3, teens_and_ties_for_scale)
        graph_lakhs_count |= create_larger_number_graph(digit_except_one, suffix_lakhs, 2, graph_hundreds_count)
        graph_lakhs_count |= create_larger_number_graph(digit_except_one, suffix_lakhs, 1, graph_thousands_count)
        graph_lakhs_count |= create_larger_number_graph(digit_except_one, suffix_lakhs, 0, graph_ten_thousands_count)
        graph_lakhs_count = graph_lakhs_count.optimize()

        graph_ten_lakhs_count = create_graph_suffix(teens_and_ties_for_scale, suffix_lakhs, 5)
        graph_ten_lakhs_count |= create_larger_number_graph(teens_and_ties_for_scale, suffix_lakhs, 4, digit_for_scale)
        graph_ten_lakhs_count |= create_larger_number_graph(
            teens_and_ties_for_scale, suffix_lakhs, 3, teens_and_ties_for_scale
        )
        graph_ten_lakhs_count |= create_larger_number_graph(
            teens_and_ties_for_scale, suffix_lakhs, 2, graph_hundreds_count
        )
        graph_ten_lakhs_count |= create_larger_number_graph(
            teens_and_ties_for_scale, suffix_lakhs, 1, graph_thousands_count
        )
        graph_ten_lakhs_count |= create_larger_number_graph(
            teens_and_ties_for_scale, suffix_lakhs, 0, graph_ten_thousands_count
        )
        graph_ten_lakhs_count = graph_ten_lakhs_count.optimize()

        # Remainder lakhs after "కోట్లు"
        # వంద కోట్లు లక్ష, వంద కోట్లు పది లక్షలు
        graph_lakhs_remainder = pynini.cross("100000", "లక్ష") | pynini.cross("౧౦౦౦౦౦", "లక్ష")
        graph_lakhs_remainder |= create_graph_suffix(digit_except_one, pynutil.insert(" లక్షలు"), 5)
        graph_lakhs_remainder |= create_larger_number_graph(digit_except_one, suffix_lakhs, 4, digit)
        graph_lakhs_remainder |= create_larger_number_graph(digit_except_one, suffix_lakhs, 3, teens_ties)
        graph_lakhs_remainder |= create_larger_number_graph(digit_except_one, suffix_lakhs, 2, graph_hundreds)
        graph_lakhs_remainder |= create_larger_number_graph(digit_except_one, suffix_lakhs, 1, graph_thousands)
        graph_lakhs_remainder |= create_larger_number_graph(digit_except_one, suffix_lakhs, 0, graph_ten_thousands)
        graph_lakhs_remainder = graph_lakhs_remainder.optimize()

        graph_ten_lakhs_remainder = create_graph_suffix(teens_and_ties_for_scale, pynutil.insert(" లక్షలు"), 5)
        graph_ten_lakhs_remainder |= create_larger_number_graph(teens_and_ties_for_scale, suffix_lakhs, 4, digit)
        graph_ten_lakhs_remainder |= create_larger_number_graph(teens_and_ties_for_scale, suffix_lakhs, 3, teens_ties)
        graph_ten_lakhs_remainder |= create_larger_number_graph(teens_and_ties_for_scale, suffix_lakhs, 2, graph_hundreds)
        graph_ten_lakhs_remainder |= create_larger_number_graph(teens_and_ties_for_scale, suffix_lakhs, 1, graph_thousands)
        graph_ten_lakhs_remainder |= create_larger_number_graph(
            teens_and_ties_for_scale, suffix_lakhs, 0, graph_ten_thousands
        )
        graph_ten_lakhs_remainder = graph_ten_lakhs_remainder.optimize()

        # -------------------------
        # Crores / Ten crores
        # -------------------------

        suffix_crore = pynutil.insert(" కోటి")
        suffix_crores = pynutil.insert(" కోట్లు")

        graph_crores = create_graph_suffix(one, suffix_crore, 7)
        graph_crores |= create_larger_number_graph(one, suffix_crore, 6, digit)
        graph_crores |= create_larger_number_graph(one, suffix_crore, 5, teens_ties)
        graph_crores |= create_larger_number_graph(one, suffix_crore, 4, graph_hundreds)
        graph_crores |= create_larger_number_graph(one, suffix_crore, 3, graph_thousands)
        graph_crores |= create_larger_number_graph(one, suffix_crore, 2, graph_ten_thousands)
        graph_crores |= create_larger_number_graph(one, suffix_crore, 1, graph_lakhs)
        graph_crores |= create_larger_number_graph(one, suffix_crore, 0, graph_ten_lakhs)

        graph_crores |= create_graph_suffix(digit_except_one, suffix_crores, 7)
        graph_crores |= create_larger_number_graph(digit_except_one, suffix_crores, 6, digit)
        graph_crores |= create_larger_number_graph(digit_except_one, suffix_crores, 5, teens_ties)
        graph_crores |= create_larger_number_graph(digit_except_one, suffix_crores, 4, graph_hundreds)
        graph_crores |= create_larger_number_graph(digit_except_one, suffix_crores, 3, graph_thousands)
        graph_crores |= create_larger_number_graph(digit_except_one, suffix_crores, 2, graph_ten_thousands)
        graph_crores |= create_larger_number_graph(digit_except_one, suffix_crores, 1, graph_lakhs)
        graph_crores |= create_larger_number_graph(digit_except_one, suffix_crores, 0, graph_ten_lakhs)
        graph_crores = graph_crores.optimize()
        self.graph_crores = graph_crores

        graph_ten_crores = create_graph_suffix(teens_and_ties_for_scale, suffix_crores, 7)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties_for_scale, suffix_crores, 6, digit)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties_for_scale, suffix_crores, 5, teens_ties)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties_for_scale, suffix_crores, 4, graph_hundreds)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties_for_scale, suffix_crores, 3, graph_thousands)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties_for_scale, suffix_crores, 2, graph_ten_thousands)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties_for_scale, suffix_crores, 1, graph_lakhs)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties_for_scale, suffix_crores, 0, graph_ten_lakhs)
        graph_ten_crores = graph_ten_crores.optimize()
        self.graph_ten_crores = graph_ten_crores

        # Count-only crore forms before final "కోట్లు"
        count_crores = create_graph_suffix(digit_for_scale | teens_and_ties_for_scale, suffix_crore, 7)
        count_crores |= create_larger_number_graph(
            digit_for_scale | teens_and_ties_for_scale, suffix_crore, 6, digit_for_scale
        )
        count_crores |= create_larger_number_graph(
            digit_for_scale | teens_and_ties_for_scale, suffix_crore, 5, teens_and_ties_for_scale
        )
        count_crores |= create_larger_number_graph(
            digit_for_scale | teens_and_ties_for_scale, suffix_crore, 4, graph_hundreds_count
        )
        count_crores |= create_larger_number_graph(
            digit_for_scale | teens_and_ties_for_scale, suffix_crore, 3, graph_thousands_count
        )
        count_crores |= create_larger_number_graph(
            digit_for_scale | teens_and_ties_for_scale, suffix_crore, 2, graph_ten_thousands_count
        )
        count_crores |= create_larger_number_graph(
            digit_for_scale | teens_and_ties_for_scale, suffix_crore, 1, graph_lakhs_count
        )
        count_crores |= create_larger_number_graph(
            digit_for_scale | teens_and_ties_for_scale, suffix_crore, 0, graph_ten_lakhs_count
        )
        count_crores = count_crores.optimize()

        count_hundred_crores = create_graph_suffix(graph_hundreds_count, suffix_crore, 7)
        count_hundred_crores |= create_larger_number_graph(graph_hundreds_count, suffix_crore, 6, digit_for_scale)
        count_hundred_crores |= create_larger_number_graph(
            graph_hundreds_count, suffix_crore, 5, teens_and_ties_for_scale
        )
        count_hundred_crores |= create_larger_number_graph(graph_hundreds_count, suffix_crore, 4, graph_hundreds_count)
        count_hundred_crores |= create_larger_number_graph(graph_hundreds_count, suffix_crore, 3, graph_thousands_count)
        count_hundred_crores |= create_larger_number_graph(
            graph_hundreds_count, suffix_crore, 2, graph_ten_thousands_count
        )
        count_hundred_crores |= create_larger_number_graph(graph_hundreds_count, suffix_crore, 1, graph_lakhs_count)
        count_hundred_crores |= create_larger_number_graph(graph_hundreds_count, suffix_crore, 0, graph_ten_lakhs_count)
        count_hundred_crores = count_hundred_crores.optimize()

        # -------------------------
        # Conversational large scales
        # అరబ్ / ఖరబ్ / నీల్ / పద్మ / శంఖ are expressed as కోట్లు.
        # -------------------------

        suffix_kotlu = pynutil.insert(" కోట్లు")

        def make_large_scale_graph(count_graph):
            g = create_graph_suffix(count_graph, suffix_kotlu, 7)
            g |= create_larger_number_graph(count_graph, suffix_kotlu, 6, digit)
            g |= create_larger_number_graph(count_graph, suffix_kotlu, 5, teens_ties)
            g |= create_larger_number_graph(count_graph, suffix_kotlu, 4, graph_hundreds)
            g |= create_larger_number_graph(count_graph, suffix_kotlu, 3, graph_thousands)
            g |= create_larger_number_graph(count_graph, suffix_kotlu, 2, graph_ten_thousands)
            g |= create_larger_number_graph(count_graph, suffix_kotlu, 1, graph_lakhs_remainder)
            g |= create_larger_number_graph(count_graph, suffix_kotlu, 0, graph_ten_lakhs_remainder)
            g |= create_larger_number_graph(count_graph, suffix_kotlu, 1, graph_crores)
            g |= create_larger_number_graph(count_graph, suffix_kotlu, 0, graph_ten_crores)
            return g.optimize()

        graph_arabs = make_large_scale_graph(graph_hundreds_count)
        graph_ten_arabs = make_large_scale_graph(graph_thousands_count | graph_ten_thousands_count)
        graph_kharabs = make_large_scale_graph(graph_ten_thousands_count)
        graph_ten_kharabs = make_large_scale_graph(graph_lakhs_count | graph_ten_lakhs_count)
        graph_nils = make_large_scale_graph(graph_ten_lakhs_count)
        graph_ten_nils = make_large_scale_graph(count_crores)
        graph_padmas = make_large_scale_graph(count_crores)
        graph_shankhs = make_large_scale_graph(count_hundred_crores)

        # Leading zero handling
        single_digit = digit | zero
        graph_leading_zero = zero + insert_space + single_digit
        graph_leading_zero = pynutil.add_weight(graph_leading_zero, 0.5)

        graph_without_leading_zeros = (
            digit
            | zero
            | teens_and_ties
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
            | graph_shankhs
        )

        self.graph_without_leading_zeros = graph_without_leading_zeros.optimize()

        cardinal_with_leading_zeros = pynini.compose(
            NEMO_ALL_ZERO + pynini.closure(NEMO_ALL_DIGIT), self.single_digits_graph
        )
        cardinal_with_leading_zeros = pynutil.add_weight(cardinal_with_leading_zeros, 0.5)

        final_graph = graph_without_leading_zeros | cardinal_with_leading_zeros

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1
        )

        self.final_graph = final_graph.optimize()
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph

