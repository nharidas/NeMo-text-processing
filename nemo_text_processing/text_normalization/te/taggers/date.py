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
    NEMO_ALL_NON_ZERO,
    NEMO_ALL_ZERO,
    NEMO_CHAR,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.te.utils import get_abs_path

days = pynini.string_file(get_abs_path("data/date/days.tsv"))
months = pynini.string_file(get_abs_path("data/date/months.tsv"))
year_suffix = pynini.string_file(get_abs_path("data/date/year_suffix.tsv"))
suffixes = pynini.string_file(get_abs_path("data/date/suffixes.tsv"))
prefixes = pynini.string_file(get_abs_path("data/date/prefixes.tsv"))
separators = pynini.string_file(get_abs_path("data/date/separators.tsv"))
teens_ties_thousand = pynini.string_file(get_abs_path("data/numbers/teens_and_ties_thousand.tsv"))


class DateFst(GraphFst):
    """
    Finite state transducer for classifying Telugu dates (Kannada-style structure), e.g.
        "౦౧-౦౪-౨౦౨౪" -> date { day: "ఒకటి" month: "ఏప్రిల్" year: "రెండు వేల ఇరవై నాలుగు" }
        "౬ మార్చి, ౨౦౧౦" -> date { day: "ఆరు" month: "మార్చి" year: "రెండు వేల పది" }

    Args:
        cardinal: cardinal GraphFst
        ordinal: optional ordinal GraphFst for Nవ శతాబ్దం / దశకం
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst = None):
        super().__init__(name="date", kind="classify")

        spelling_fix = pynini.cdrewrite(
            pynini.cross("పధ్ధెనిమిది", "పద్దెనిమిది"), "", "", NEMO_SIGMA
        ).optimize()

        digit = cardinal.digit
        teens_and_ties = cardinal.teens_and_ties

        cardinal_graph = pynini.union(
            digit,
            teens_and_ties,
            cardinal.graph_hundreds,
            cardinal.graph_thousands,
        ).optimize()
        cardinal_graph = (cardinal_graph @ spelling_fix).optimize()

        teens_ties_excl_thousand = pynini.compose(
            pynini.difference(
                pynini.project(teens_and_ties, "input"),
                pynini.project(teens_ties_thousand, "input"),
            ),
            teens_and_ties,
        ).optimize()
        two_digit_year_prefix = pynini.union(teens_ties_thousand, teens_ties_excl_thousand).optimize()
        year_hundreds_raw = (
            (
                two_digit_year_prefix
                + pynutil.insert(" వందలు")
                + pynutil.delete(NEMO_ALL_ZERO)
                + pynutil.delete(NEMO_ALL_ZERO)
            )
            | (
                two_digit_year_prefix
                + pynutil.insert(" వందల ")
                + pynutil.delete(NEMO_ALL_ZERO)
                + digit
            )
            | (two_digit_year_prefix + pynutil.insert(" వందల ") + teens_and_ties)
        ).optimize()

        year_start_12 = pynini.union("1", "2", "౧", "౨")
        year_start_3_9 = pynini.union("3", "4", "5", "6", "7", "8", "9", "౩", "౪", "౫", "౬", "౭", "౮", "౯")

        graph_year_hundreds = pynini.compose(
            year_start_12 + NEMO_ALL_NON_ZERO + NEMO_ALL_DIGIT + NEMO_ALL_DIGIT, year_hundreds_raw
        ).optimize()
        graph_year_thousands = pynini.compose(
            NEMO_ALL_DIGIT + NEMO_ALL_ZERO + NEMO_ALL_DIGIT + NEMO_ALL_DIGIT, cardinal.graph_thousands
        ).optimize()
        graph_year_exact = pynini.compose(
            NEMO_ALL_DIGIT + NEMO_ALL_ZERO + NEMO_ALL_ZERO + NEMO_ALL_ZERO, cardinal.graph_thousands
        ).optimize()
        graph_year_old = pynini.compose(
            year_start_3_9 + NEMO_ALL_DIGIT + NEMO_ALL_DIGIT + NEMO_ALL_DIGIT, cardinal.graph_thousands
        ).optimize()

        graph_year = (
            graph_year_hundreds | graph_year_thousands | graph_year_exact | graph_year_old
        ).optimize()
        graph_year = (graph_year @ spelling_fix).optimize()

        graph_year_short = (digit | teens_and_ties | cardinal.graph_hundreds).optimize()
        graph_year_short = (graph_year_short @ spelling_fix).optimize()
        graph_year_any = pynini.union(graph_year, graph_year_short).optimize()

        telugu_chars = pynini.union(*[pynini.accep(chr(code)) for code in range(0x0C00, 0x0C7F)])
        space = pynini.accep(" ")
        to_stem = pynini.cdrewrite(
            pynini.union(pynini.cross(" వందలు", " వందల"), pynini.cross(" వేలు", " వేల")),
            "",
            "",
            pynini.closure(telugu_chars | space | NEMO_CHAR),
        ).optimize()
        vandalu_lo = pynini.cdrewrite(pynini.cross(" వందలులో", " వందలలో"), "", "", NEMO_SIGMA).optimize()

        def stem_form(number_graph):
            return (number_graph @ to_stem).optimize()

        def number_with_suffix(number_graph):
            """Kannada-style: number (+ optional space) + suffix."""
            return (
                stem_form(number_graph)
                + pynini.closure(pynutil.delete(" "), 0, 1)
                + suffixes
            ).optimize()

        days_verbalized = (days @ spelling_fix).optimize()
        days_graph = pynutil.insert('day: "') + days_verbalized + pynutil.insert('"') + insert_space
        months_graph = pynutil.insert('month: "') + months + pynutil.insert('"') + insert_space
        years_graph = pynutil.insert('year: "') + graph_year + pynutil.insert('"') + insert_space

        delete_separator = pynutil.delete(separators)
        dash_only = pynini.compose(separators, pynini.accep("-"))
        delete_dash = pynutil.delete(dash_only)
        delete_comma = pynutil.delete(",")
        delete_optional_space = pynini.closure(pynutil.delete(" "), 0, 1)
        delete_comma_sep = delete_comma + delete_optional_space
        delete_space = pynutil.delete(" ")

        DD_MM_PREFERENCE = -0.01

        graph_dd_mm = pynutil.add_weight(
            days_graph + delete_separator + months_graph, DD_MM_PREFERENCE
        )
        graph_mm_dd = months_graph + delete_separator + days_graph + pynutil.insert(" preserve_order: true ")

        graph_dd_mm_yyyy = pynutil.add_weight(
            days_graph + delete_separator + months_graph + delete_separator + years_graph,
            DD_MM_PREFERENCE,
        )
        graph_mm_dd_yyyy = (
            months_graph
            + delete_separator
            + days_graph
            + delete_separator
            + years_graph
            + pynutil.insert(" preserve_order: true ")
        )

        graph_mm_yyyy = months_graph + delete_dash + insert_space + years_graph

        month_name = pynini.union(
            pynini.project(months, "output"),
            pynini.cross("నవరి", "జనవరి"),
        ).optimize()
        month_name = (pynini.closure(pynutil.delete("ज"), 0, 1) + month_name).optimize()
        month_name_graph = pynutil.insert('month: "') + month_name + pynutil.insert('"') + insert_space

        devanagari_fix = pynini.cdrewrite(pynutil.delete("१"), "", "", NEMO_SIGMA).optimize()

        year_with_era = graph_year_any + pynini.closure(NEMO_SPACE, 0, 1) + year_suffix
        years_or_era_graph = (
            pynutil.insert('year: "')
            + pynini.union(graph_year, year_with_era)
            + pynutil.insert('"')
            + insert_space
        )
        years_suffix_graph = (
            pynutil.insert('year: "') + graph_year + suffixes + pynutil.insert('"') + insert_space
        )

        graph_d_month = pynutil.add_weight(
            days_graph + delete_space + month_name_graph, DD_MM_PREFERENCE
        )
        graph_month_d = (
            month_name_graph + delete_space + days_graph + pynutil.insert(" preserve_order: true ")
        )

        graph_d_month_yyyy = (
            days_graph + delete_space + month_name_graph + delete_comma_sep + years_or_era_graph
        )
        graph_d_month_yyyy_suf = (
            days_graph + delete_space + month_name_graph + delete_comma_sep + years_suffix_graph
        )
        graph_month_comma_yyyy = month_name_graph + delete_comma_sep + years_or_era_graph
        graph_month_space_yyyy = month_name_graph + delete_space + years_graph

        era_graph = pynutil.insert('era: "') + year_suffix + pynutil.insert('"') + insert_space

        word = pynini.closure(pynini.difference(NEMO_CHAR, pynini.union(NEMO_WHITE_SPACE, NEMO_ALL_DIGIT)), 1)
        rest_words = pynini.closure(pynini.accep(" ") + word, 0, 3)

        year_text = (
            pynutil.insert('era: "')
            + number_with_suffix(graph_year)
            + rest_words
            + pynutil.insert('"')
            + insert_space
        )
        year_text = (year_text @ vandalu_lo).optimize()

        short_with_suffix = (
            pynutil.insert('era: "')
            + number_with_suffix(graph_year_short @ spelling_fix)
            + rest_words
            + pynutil.insert('"')
            + insert_space
        )

        year_prefix = (
            pynutil.insert('era: "') + prefixes + insert_space + graph_year + pynutil.insert('"')
        )
        year_prefix_suffix = (
            pynutil.insert('era: "')
            + prefixes
            + insert_space
            + number_with_suffix(graph_year)
            + pynutil.insert('"')
        )
        year_prefix_era = (
            pynutil.insert('era: "')
            + prefixes
            + insert_space
            + graph_year
            + pynini.closure(NEMO_SPACE, 0, 1)
            + year_suffix
            + pynutil.insert('"')
        )

        verbalized_year = pynini.project(graph_year, "output").optimize()
        verbalized_bare = pynutil.insert('era: "') + verbalized_year + pynutil.insert('"') + insert_space
        verbalized_suffix = (
            pynutil.insert('era: "')
            + verbalized_year
            + suffixes
            + pynutil.insert('"')
            + insert_space
        )
        verbalized_prefix = (
            pynutil.insert('era: "') + prefixes + insert_space + verbalized_year + pynutil.insert('"')
        )
        verbalized_prefix_suffix = (
            pynutil.insert('era: "')
            + prefixes
            + insert_space
            + verbalized_year
            + suffixes
            + pynutil.insert('"')
        )

        year_era_4 = (
            pynutil.insert('era: "')
            + stem_form(graph_year)
            + pynini.closure(NEMO_SPACE, 0, 1)
            + year_suffix
            + pynutil.insert('"')
            + insert_space
        )
        year_suffix_short = pynini.union(
            pynini.accep("క్రీ.శ."),
            pynini.cross("క్రీ.పూ.", "క్రీస్తు పూర్వం"),
        )
        three_digit = NEMO_ALL_DIGIT ** 3
        year_era_3 = (
            pynutil.insert('era: "')
            + pynini.compose(three_digit, stem_form(graph_year_any))
            + pynini.closure(NEMO_SPACE, 0, 1)
            + year_suffix_short
            + pynutil.insert('"')
            + insert_space
        )
        year_era_short = (
            pynutil.insert('era: "')
            + stem_form(graph_year_short)
            + pynini.closure(NEMO_SPACE, 0, 1)
            + year_suffix
            + pynutil.insert('"')
            + insert_space
        )

        bare_year = pynutil.insert('era: "') + graph_year + pynutil.insert('"') + insert_space

        if ordinal is not None:
            ordinal_graph = ordinal.graph
        else:
            ordinal_graph = (cardinal_graph + pynutil.delete("వ") + pynutil.insert("వ")).optimize()

        ordinal_date = (
            ordinal_graph
            @ pynini.cdrewrite(pynini.cross("ఇరవైవ", "ఇరవయ్యవ"), "", "", NEMO_SIGMA)
        ).optimize()

        ordinal_text = (
            pynutil.insert('era: "')
            + ordinal_date
            + suffixes
            + pynutil.insert('"')
            + insert_space
        )

        century_text = (
            pynutil.insert('era: "')
            + ordinal_date
            + pynini.union(pynini.accep(" శతాబ్దం"), pynini.accep(" దశకం"))
            + pynutil.insert('"')
            + insert_space
        )

        dash_to_suffix = (pynini.accep("-") @ suffixes).optimize()

        range_with_era = (
            pynutil.insert('era: "')
            + stem_form(graph_year_any)
            + dash_to_suffix
            + graph_year_any
            + pynutil.delete(" ")
            + insert_space
            + year_suffix
            + pynutil.insert('"')
            + pynutil.insert(" preserve_order: true ")
        ).optimize()

        range_with_suffix = (
            pynutil.insert('era: "')
            + stem_form(graph_year_any)
            + dash_to_suffix
            + stem_form(graph_year_any)
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + suffixes
            + pynutil.insert('"')
            + pynutil.insert(" preserve_order: true ")
        ).optimize()

        month_ordinal_day = (
            month_name_graph
            + delete_space
            + pynutil.insert('day: "')
            + ordinal_date
            + pynini.accep(" తేదీ")
            + pynutil.insert('"')
            + insert_space
            + pynutil.insert(" preserve_order: true ")
        )
        day_tedi = (
            pynutil.insert('era: "')
            + days_verbalized
            + pynini.accep(" తేదీ")
            + pynini.closure(pynini.union(pynini.accep("న"), pynini.accep("లో")), 0, 1)
            + pynutil.insert('"')
            + insert_space
        )
        tedi_day = (
            pynutil.insert('era: "')
            + pynini.accep("తేదీ ")
            + days_verbalized
            + pynutil.insert('"')
            + insert_space
        )
        day_case = (
            pynutil.insert('era: "')
            + pynini.closure(word + NEMO_SPACE, 0, 3)
            + days_verbalized
            + pynini.union(pynini.accep("న"), pynini.accep("కి"), pynini.accep("లో"))
            + rest_words
            + pynutil.insert('"')
            + insert_space
        )

        weekdays = pynini.string_file(get_abs_path("data/date/weekdays.tsv"))
        d_month_weekday = (
            pynutil.insert('era: "')
            + days_verbalized
            + pynini.accep(" ")
            + month_name
            + pynini.accep(", ")
            + weekdays
            + pynutil.insert('"')
            + insert_space
        )

        final_graph = (
            range_with_era
            | range_with_suffix
            | graph_dd_mm_yyyy
            | graph_mm_dd_yyyy
            | graph_d_month_yyyy_suf
            | graph_d_month_yyyy
            | graph_month_comma_yyyy
            | graph_month_space_yyyy
            | graph_dd_mm
            | graph_mm_dd
            | graph_d_month
            | graph_month_d
            | graph_mm_yyyy
            | month_ordinal_day
            | d_month_weekday
            | century_text
            | ordinal_text
            | day_tedi
            | tedi_day
            | day_case
            | year_prefix_era
            | year_prefix_suffix
            | year_prefix
            | verbalized_prefix_suffix
            | verbalized_prefix
            | verbalized_suffix
            | verbalized_bare
            | year_text
            | short_with_suffix
            | year_era_4
            | year_era_3
            | year_era_short
            | era_graph
            | bare_year
        )

        final_graph = pynini.compose(devanagari_fix, final_graph).optimize()

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph)
