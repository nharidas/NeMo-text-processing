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

from nemo_text_processing.text_normalization.te.graph_utils import GraphFst
from nemo_text_processing.text_normalization.te.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.te.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Telugu ordinals, e.g.
        1వ -> ordinal { integer: "మొదటి" }
        12వ -> ordinal { integer: "పన్నెండవ" }
        21వ -> ordinal { integer: "ఇరవై ఒకటవ" }
        125వ -> ordinal { integer: "నూట ఇరవై ఐదవ" }
        9826వ -> ordinal { integer: "తొమ్మిది వేల ఎనిమిది వందల ఇరవై ఆరవ" }
    """

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        exceptions = pynini.string_file(get_abs_path("data/ordinal/exceptions.tsv"))
        suffixes_list = pynini.string_file(get_abs_path("data/ordinal/suffixes.tsv"))
        suffixes_map = pynini.string_file(get_abs_path("data/ordinal/suffixes_map.tsv"))
        suffixes_fst = pynini.union(suffixes_list, suffixes_map)

        ordinal_digit = pynini.string_file(get_abs_path("data/ordinal/ordinal_digit.tsv")).optimize()
        ordinal_teens = pynini.string_file(get_abs_path("data/ordinal/ordinal_teens.tsv")).optimize()
        ordinal_tens_prefix = pynini.string_file(
            get_abs_path("data/ordinal/ordinal_tens_prefix.tsv")
        ).optimize()

        te_zero = "౦"
        te_one = "౧"
        te_two = "౨"
        te_three = "౩"
        te_four = "౪"
        te_five = "౫"
        te_six = "౬"
        te_seven = "౭"
        te_eight = "౮"
        te_nine = "౯"

        zero = pynini.union("0", te_zero)
        one = pynini.union("1", te_one)
        two_to_nine = pynini.union(
            "2", "3", "4", "5", "6", "7", "8", "9",
            te_two, te_three, te_four, te_five, te_six, te_seven, te_eight, te_nine,
        )

        delete_suffix = pynutil.delete("వ")
        digit_except_one = (two_to_nine @ cardinal.digit).optimize()

        # 20వ -> ఇరవైవ, 40వ -> నలభైవ
        ordinal_tens_exact = (
            ordinal_tens_prefix
            + pynutil.delete(zero)
            + delete_suffix
            + pynutil.insert("వ")
        ).optimize()

        # 21వ -> ఇరవై ఒకటవ, 99వ -> తొంభై తొమ్మిదవ
        ordinal_two_digit_compound = (
            ordinal_tens_prefix
            + pynutil.insert(" ")
            + ordinal_digit
            + delete_suffix
        ).optimize()

        two_digit_ordinal = pynini.union(
            ordinal_teens + delete_suffix,
            ordinal_tens_exact,
            ordinal_two_digit_compound,
        ).optimize()

        # 100వ -> వందవ, 200వ -> రెండు వందలవ
        hundred_exact = pynini.union(
            pynini.cross("100", "వందవ"),
            pynini.cross(te_one + te_zero + te_zero, "వందవ"),
            digit_except_one
            + pynutil.delete(zero)
            + pynutil.delete(zero)
            + delete_suffix
            + pynutil.insert(" వందలవ"),
        ).optimize()

        # 101వ-109వ -> నూట ఒకటవ
        hundred_101_to_109 = (
            pynutil.delete(one)
            + pynutil.delete(zero)
            + pynutil.insert("నూట ")
            + ordinal_digit
            + delete_suffix
        ).optimize()

        # 110వ-199వ -> నూట పదవ / నూట పదకొండవ / నూట యాభై మూడవ
        hundred_110_to_199 = (
            pynutil.delete(one)
            + pynutil.insert("నూట ")
            + two_digit_ordinal
        ).optimize()

        # 201వ-909వ -> రెండు వందల ఒకటవ
        hundred_200_to_909 = (
            digit_except_one
            + pynutil.delete(zero)
            + pynutil.insert(" వందల ")
            + ordinal_digit
            + delete_suffix
        ).optimize()

        # 210వ-999వ -> రెండు వందల పదవ / మూడు వందల ఇరవై తొమ్మిదవ
        hundred_210_to_999 = (
            digit_except_one
            + pynutil.insert(" వందల ")
            + two_digit_ordinal
        ).optimize()

        hundreds_ordinal = pynini.union(
            hundred_exact,
            hundred_101_to_109,
            hundred_110_to_199,
            hundred_200_to_909,
            hundred_210_to_999,
        ).optimize()

        # 1000వ -> వెయ్యవ, 2000వ -> రెండు వేలవ
        thousand_exact = pynini.union(
            pynini.cross("1000", "వెయ్యవ"),
            pynini.cross(te_one + te_zero + te_zero + te_zero, "వెయ్యవ"),
            digit_except_one
            + pynutil.delete(zero)
            + pynutil.delete(zero)
            + pynutil.delete(zero)
            + delete_suffix
            + pynutil.insert(" వేలవ"),
        ).optimize()

        # 1001వ-1009వ -> వెయ్యి ఒకటవ
        thousand_1001_to_1009 = (
            pynutil.delete(one)
            + pynutil.delete(zero)
            + pynutil.delete(zero)
            + pynutil.insert("వెయ్యి ")
            + ordinal_digit
            + delete_suffix
        ).optimize()

        # 1010వ-1099వ -> వెయ్యి పదవ / వెయ్యి తొంభై ఒకటవ
        thousand_1010_to_1099 = (
            pynutil.delete(one)
            + pynutil.delete(zero)
            + pynutil.insert("వెయ్యి ")
            + two_digit_ordinal
        ).optimize()

        # 1100వ-1999వ -> వెయ్యి ఏడు వందల ఎనభై రెండవ
        thousand_1100_to_1999 = (
            pynutil.delete(one)
            + pynutil.insert("వెయ్యి ")
            + hundreds_ordinal
        ).optimize()

        # 2001వ-9009వ -> రెండు వేల ఒకటవ
        thousand_2001_to_9009 = (
            digit_except_one
            + pynutil.delete(zero)
            + pynutil.delete(zero)
            + pynutil.insert(" వేల ")
            + ordinal_digit
            + delete_suffix
        ).optimize()

        # 2010వ-9099వ -> రెండు వేల పదవ / తొమ్మిది వేల ఇరవై ఆరవ
        thousand_2010_to_9099 = (
            digit_except_one
            + pynutil.delete(zero)
            + pynutil.insert(" వేల ")
            + two_digit_ordinal
        ).optimize()

        # 2100వ-9999వ -> తొమ్మిది వేల ఎనిమిది వందల ఇరవై ఆరవ
        thousand_2100_to_9999 = (
            digit_except_one
            + pynutil.insert(" వేల ")
            + hundreds_ordinal
        ).optimize()

        thousands_ordinal = pynini.union(
            thousand_exact,
            thousand_1001_to_1009,
            thousand_1010_to_1099,
            thousand_1100_to_1999,
            thousand_2001_to_9009,
            thousand_2010_to_9099,
            thousand_2100_to_9999,
        ).optimize()

        # 10000వ-99000వ style.
        # Uses the existing cardinal ten-thousand prefix for the count, then ordinalizes the remainder.
        ten_thousand_exact = (
            cardinal.teens_and_ties
            + pynutil.delete(zero)
            + pynutil.delete(zero)
            + pynutil.delete(zero)
            + delete_suffix
            + pynutil.insert(" వేలవ")
        ).optimize()

        ten_thousand_0001_to_0009 = (
            cardinal.teens_and_ties
            + pynutil.delete(zero)
            + pynutil.delete(zero)
            + pynutil.insert(" వేల ")
            + ordinal_digit
            + delete_suffix
        ).optimize()

        ten_thousand_0010_to_0099 = (
            cardinal.teens_and_ties
            + pynutil.delete(zero)
            + pynutil.insert(" వేల ")
            + two_digit_ordinal
        ).optimize()

        ten_thousand_0100_to_9999 = (
            cardinal.teens_and_ties
            + pynutil.insert(" వేల ")
            + hundreds_ordinal
        ).optimize()

        ten_thousands_ordinal = pynini.union(
            ten_thousand_exact,
            ten_thousand_0001_to_0009,
            ten_thousand_0010_to_0099,
            ten_thousand_0100_to_9999,
        ).optimize()

        limited_cardinal_graph = (
            cardinal.digit
            | cardinal.zero
            | cardinal.teens_and_ties
            | cardinal.graph_hundreds
            | cardinal.graph_thousands
            | cardinal.graph_ten_thousands
        ).optimize()

        fallback_graph = pynutil.add_weight(limited_cardinal_graph + suffixes_fst, 20.0)
        exceptions = pynutil.add_weight(exceptions, -0.1)

        graph = pynini.union(
            exceptions,
            ten_thousands_ordinal,
            thousands_ordinal,
            hundreds_ordinal,
            two_digit_ordinal,
            ordinal_digit + delete_suffix,
            fallback_graph,
        ).optimize()

        self.graph = graph

        final_graph = pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()