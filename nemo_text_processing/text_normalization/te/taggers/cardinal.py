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
    Finite state transducer for classifying cardinals, e.g.
        -౨౩ -> cardinal { negative: "true"  integer: "ఇరవై మూడు" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        teens_ties_hi = pynini.string_file(get_abs_path("data/numbers/teens_and_ties.tsv"))
        teens_ties_en = pynini.string_file(get_abs_path("data/numbers/teens_and_ties_en.tsv"))
        teens_ties_thousand = pynutil.add_weight(pynini.string_file(get_abs_path("data/numbers/teens_and_ties_thousand.tsv")),-0.2,)
        teens_ties = pynini.union(teens_ties_hi, teens_ties_en)
        teens_and_ties = pynutil.add_weight(teens_ties, -0.1)

        self.digit = digit
        self.zero = zero
        self.teens_and_ties = teens_and_ties

        # Single digit graph for digit-by-digit reading
        # e.g., "०७३" -> "शून्य सात तीन"
        single_digit_graph = digit | zero
        self.single_digits_graph = single_digit_graph + pynini.closure(insert_space + single_digit_graph)

        def create_graph_suffix(digit_graph, suffix, zeros_counts):
            zero = pynutil.add_weight(pynutil.delete(NEMO_ALL_ZERO | pynini.accep("౦")), -0.1)
            if zeros_counts == 0:
                return digit_graph + suffix

            return digit_graph + (zero**zeros_counts) + suffix

        def create_larger_number_graph(digit_graph, suffix, zeros_counts, sub_graph):
            insert_space = pynutil.insert(" ")
            zero = pynutil.add_weight(pynutil.delete(NEMO_ALL_ZERO | pynini.accep("౦")), -0.1)
            if zeros_counts == 0:
                return digit_graph + suffix + insert_space + sub_graph

            return digit_graph + suffix + (zero**zeros_counts) + insert_space + sub_graph

        # Hundred graph

        suffix_hundreds = pynutil.insert(" వంద")

        digit_except_one = (pynini.union("2", "3", "4", "5", "6", "7", "8", "9","౨", "౩", "౪", "౫", "౬", "౭", "౮", "౯")@ digit).optimize()

        graph_hundreds = pynini.cross("100", "వంద") | pynini.cross("౧౦౦", "వంద")
        graph_hundreds |= (pynini.cross("10", "నూట ") | pynini.cross("౧౦", "నూట ")) + digit
        graph_hundreds |= (pynini.cross("1", "నూట ") | pynini.cross("౧", "నూట ")) + teens_ties

        graph_hundreds |= create_graph_suffix(digit_except_one, pynutil.insert(" వందలు"), 2)
        graph_hundreds |= create_larger_number_graph(digit_except_one, pynutil.insert(" వందల"), 1, digit)
        graph_hundreds |= create_larger_number_graph(digit_except_one, pynutil.insert(" వందల"), 0, teens_ties)

        graph_hundreds = graph_hundreds.optimize()
        self.graph_hundreds = graph_hundreds

        # Transducer for eleven hundred -> 1100 or twenty one hundred eleven -> 2111
        graph_hundreds_as_thousand = create_graph_suffix(teens_and_ties, suffix_hundreds, 2)
        graph_hundreds_as_thousand |= create_larger_number_graph(teens_and_ties, suffix_hundreds, 1, digit)
        graph_hundreds_as_thousand |= create_larger_number_graph(teens_and_ties, suffix_hundreds, 0, teens_ties)
        self.graph_hundreds_as_thousand = graph_hundreds_as_thousand

        # Thousands and Ten thousands graph
        one_thousand_prefix = pynutil.delete(pynini.union("1", "౧"))
        digit_except_one = (pynini.union("2", "3", "4", "5", "6", "7", "8", "9","౨", "౩", "౪", "౫", "౬", "౭", "౮", "౯") @ digit).optimize()

        suffix_thousand_exact = pynutil.insert("వెయ్యి")
        suffix_thousand_plural = pynutil.insert(" వేలు")
        suffix_thousand_before = pynutil.insert(" వేల")

        # 1000
        graph_thousands = pynini.cross("1000", "వెయ్యి")
        graph_thousands |= pynini.cross("౧౦౦౦", "వెయ్యి")

        graph_thousands |= create_larger_number_graph(one_thousand_prefix, suffix_thousand_exact, 2, digit,)  # 1001–1009
        graph_thousands |= create_larger_number_graph(one_thousand_prefix, suffix_thousand_exact, 1, teens_ties,) # 1010–1099
        graph_thousands |= create_larger_number_graph(one_thousand_prefix, suffix_thousand_exact, 0, graph_hundreds,) # 1100–1999
        graph_thousands |= create_graph_suffix(digit_except_one, suffix_thousand_plural, 3,) # 2000–9000
        graph_thousands |= create_larger_number_graph(digit_except_one, suffix_thousand_before, 2, digit,) # 2001–2009, 3001–9009
        graph_thousands |= create_larger_number_graph(digit_except_one, suffix_thousand_before, 1, teens_ties,) # 2010–2099, 3010–9099
        graph_thousands |= create_larger_number_graph(digit_except_one, suffix_thousand_before, 0, graph_hundreds,) # 2100–9999

        graph_thousands = graph_thousands.optimize()
        self.graph_thousands = graph_thousands

        # Ten thousands graph
        suffix_ten_thousand_exact = pynutil.insert(" వెయ్యి")
        suffix_ten_thousand_plural = pynutil.insert(" వేలు")
        suffix_ten_thousand_before = pynutil.insert(" వేల")

        graph_ten_thousands = create_graph_suffix(teens_ties_thousand, suffix_ten_thousand_exact,3,) # 21000, 31000, 41000...
        graph_ten_thousands |= create_larger_number_graph(teens_ties_thousand, suffix_ten_thousand_before, 2, digit,) # 21001, 31001...
        graph_ten_thousands |= create_larger_number_graph(teens_ties_thousand, suffix_ten_thousand_before, 1, teens_ties,) # 21010, 41010...
        graph_ten_thousands |= create_larger_number_graph(teens_ties_thousand, suffix_ten_thousand_before, 0, graph_hundreds,) # 21100, 91234...
        graph_ten_thousands |= create_graph_suffix(teens_and_ties, suffix_ten_thousand_plural, 3,) # 10000, 11000, 12000, 19000, 20000, 22000...
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_ten_thousand_before, 2, digit,) # 10001, 22001...
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_ten_thousand_before, 1, teens_ties,) # 10010, 23010...
        graph_ten_thousands |= create_larger_number_graph(teens_and_ties, suffix_ten_thousand_before, 0, graph_hundreds,) # 10100, 99999...

        graph_ten_thousands = graph_ten_thousands.optimize()
        self.graph_ten_thousands = graph_ten_thousands

        # Lakhs graph and ten lakhs graph
        one_lakh_prefix = pynutil.delete(pynini.union("1", "౧"))

        suffix_lakh_exact = pynutil.insert("లక్ష")
        suffix_lakha_digit = pynutil.insert("లక్షా")
        suffix_lakh_plural = pynutil.insert(" లక్షలు")
        suffix_lakh_before = pynutil.insert(" లక్షల")

        # 100000
        graph_lakhs = pynini.cross("100000", "లక్ష")
        graph_lakhs |= pynini.cross("౧౦౦౦౦౦", "లక్ష")

        graph_lakhs |= create_larger_number_graph(one_lakh_prefix, suffix_lakha_digit, 4, digit,)  # 100001–100009 -> లక్షా ఒకటి
        graph_lakhs |= create_larger_number_graph(one_lakh_prefix, suffix_lakh_exact, 3, teens_ties,) # 100010–100099 -> లక్ష పది
        graph_lakhs |= create_larger_number_graph(one_lakh_prefix, suffix_lakh_exact, 2, graph_hundreds,) # 100100–100999 -> లక్ష వంద
        graph_lakhs |= create_larger_number_graph(one_lakh_prefix, suffix_lakh_exact, 1, graph_thousands,) # 101000–109999 -> లక్ష వెయ్యి
        graph_lakhs |= create_larger_number_graph(one_lakh_prefix, suffix_lakh_exact, 0, graph_ten_thousands,) # 110000–199999 -> లక్ష పది వేలు
        graph_lakhs |= create_graph_suffix(digit_except_one, suffix_lakh_plural, 5,) # 200000–900000 -> రెండు లక్షలు
        graph_lakhs |= create_larger_number_graph(digit_except_one, suffix_lakh_before, 4, digit,) # 200001–200009 -> రెండు లక్షల ఒకటి
        graph_lakhs |= create_larger_number_graph(digit_except_one, suffix_lakh_before, 3, teens_ties,) # 200010–200099
        graph_lakhs |= create_larger_number_graph(digit_except_one, suffix_lakh_before, 2, graph_hundreds,) # 200100–200999
        graph_lakhs |= create_larger_number_graph(digit_except_one, suffix_lakh_before, 1, graph_thousands,) # 201000–209999
        graph_lakhs |= create_larger_number_graph(digit_except_one, suffix_lakh_before, 0, graph_ten_thousands,) # 210000–999999
        graph_lakhs = graph_lakhs.optimize()
        self.graph_lakhs = graph_lakhs

        # Ten lakhs graph
        suffix_ten_lakh_exact = pynutil.insert(" లక్ష")
        suffix_ten_lakh_plural = pynutil.insert(" లక్షలు")
        suffix_ten_lakh_before = pynutil.insert(" లక్షల")

        graph_ten_lakhs = create_graph_suffix(teens_ties_thousand, suffix_ten_lakh_exact, 5,) # 2100000, 3100000, ... -> ఇరవై ఒక లక్ష
        graph_ten_lakhs |= create_larger_number_graph(teens_ties_thousand, suffix_ten_lakh_before, 4, digit,) # 2100001, 3100001, ... -> ఇరవై ఒక లక్షల ఒకటి
        graph_ten_lakhs |= create_larger_number_graph( teens_ties_thousand, suffix_ten_lakh_before, 3, teens_ties,)
        graph_ten_lakhs |= create_larger_number_graph(teens_ties_thousand, suffix_ten_lakh_before, 2, graph_hundreds,)
        graph_ten_lakhs |= create_larger_number_graph(teens_ties_thousand, suffix_ten_lakh_before, 1, graph_thousands,)
        graph_ten_lakhs |= create_larger_number_graph(teens_ties_thousand, suffix_ten_lakh_before, 0, graph_ten_thousands,)
        graph_ten_lakhs |= create_graph_suffix(teens_and_ties, suffix_ten_lakh_plural, 5,) # 1000000, 1100000, 2200000, 9900000 -> పది లక్షలు
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_ten_lakh_before, 4, digit,) # 1000001, 2200001
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_ten_lakh_before, 3, teens_ties,)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_ten_lakh_before, 2, graph_hundreds,)
        graph_ten_lakhs |= create_larger_number_graph( teens_and_ties, suffix_ten_lakh_before, 1, graph_thousands,)
        graph_ten_lakhs |= create_larger_number_graph(teens_and_ties, suffix_ten_lakh_before, 0, graph_ten_thousands,)

        graph_ten_lakhs = graph_ten_lakhs.optimize()
        self.graph_ten_lakhs = graph_ten_lakhs

        # Crores graph & ten crores graph
        one_crore_prefix = pynutil.delete(pynini.union("1", "౧"))

        suffix_crore_exact = pynutil.insert("కోటి")
        suffix_crore_plural = pynutil.insert(" కోట్లు")
        suffix_crore_before = pynutil.insert(" కోట్లు")

        # 10000000
        graph_crores = pynini.cross("10000000", "కోటి")
        graph_crores |= pynini.cross("౧౦౦౦౦౦౦౦", "కోటి")

        graph_crores |= create_larger_number_graph(one_crore_prefix, suffix_crore_exact, 6, digit,) # 10000001–10000009
        graph_crores |= create_larger_number_graph(one_crore_prefix, suffix_crore_exact, 5, teens_ties,) # 10000010–10000099
        graph_crores |= create_larger_number_graph(one_crore_prefix, suffix_crore_exact, 4, graph_hundreds,) # 10000100–10000999
        graph_crores |= create_larger_number_graph(one_crore_prefix, suffix_crore_exact, 3, graph_thousands,) # 10001000–10009999
        graph_crores |= create_larger_number_graph(one_crore_prefix, suffix_crore_exact, 2, graph_ten_thousands,) # 10010000–10099999
        graph_crores |= create_larger_number_graph(one_crore_prefix, suffix_crore_exact, 1, graph_lakhs,) # 10100000–10999999
        graph_crores |= create_larger_number_graph(one_crore_prefix, suffix_crore_exact, 0, graph_ten_lakhs,) # 11000000–19999999
        graph_crores |= create_graph_suffix(digit_except_one, suffix_crore_plural, 7,) # 20000000–90000000
        graph_crores |= create_larger_number_graph(digit_except_one, suffix_crore_before, 6, digit,) # 20000001–90000009
        graph_crores |= create_larger_number_graph(digit_except_one, suffix_crore_before, 5, teens_ties,) # 20000010–90000099
        graph_crores |= create_larger_number_graph(digit_except_one, suffix_crore_before, 4, graph_hundreds,) # 20000100–90000999    
        graph_crores |= create_larger_number_graph(digit_except_one, suffix_crore_before, 3, graph_thousands,) # 20001000–90009999
        graph_crores |= create_larger_number_graph(digit_except_one, suffix_crore_before, 2, graph_ten_thousands,) # 20010000–90099999
        graph_crores |= create_larger_number_graph(digit_except_one, suffix_crore_before, 1, graph_lakhs,) # 20100000–90999999
        graph_crores |= create_larger_number_graph(digit_except_one, suffix_crore_before, 0, graph_ten_lakhs,) # 21000000–99999999

        graph_crores = graph_crores.optimize()

        # Ten crores graph
        suffix_ten_crore_exact = pynutil.insert(" కోటి")
        suffix_ten_crore_plural = pynutil.insert(" కోట్లు")
        suffix_ten_crore_before = pynutil.insert(" కోట్లు")

        graph_ten_crores = create_graph_suffix(teens_ties_thousand, suffix_ten_crore_exact, 7,) # 210000000, 310000000, ... -> ఇరవై ఒక కోటి
        graph_ten_crores |= create_larger_number_graph(teens_ties_thousand, suffix_ten_crore_exact, 6, digit,) # 210000001, 310000001, ...
        graph_ten_crores |= create_larger_number_graph(teens_ties_thousand, suffix_ten_crore_exact, 5, teens_ties,)
        graph_ten_crores |= create_larger_number_graph(teens_ties_thousand, suffix_ten_crore_exact, 4, graph_hundreds,)
        graph_ten_crores |= create_larger_number_graph(teens_ties_thousand, suffix_ten_crore_exact, 3, graph_thousands,)
        graph_ten_crores |= create_larger_number_graph(teens_ties_thousand, suffix_ten_crore_exact, 2, graph_ten_thousands,)
        graph_ten_crores |= create_larger_number_graph(teens_ties_thousand, suffix_ten_crore_exact, 1, graph_lakhs,)
        graph_ten_crores |= create_larger_number_graph(teens_ties_thousand, suffix_ten_crore_exact, 0, graph_ten_lakhs,)   
        graph_ten_crores |= create_graph_suffix(teens_and_ties, suffix_ten_crore_plural, 7,) # 100000000, 110000000, 220000000, 990000000
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_ten_crore_before, 6, digit,)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_ten_crore_before, 5, teens_ties,)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_ten_crore_before, 4, graph_hundreds,)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_ten_crore_before, 3, graph_thousands,)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_ten_crore_before, 2, graph_ten_thousands,)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_ten_crore_before, 1, graph_lakhs,)
        graph_ten_crores |= create_larger_number_graph(teens_and_ties, suffix_ten_crore_before, 0, graph_ten_lakhs,)

        graph_ten_crores = graph_ten_crores.optimize()

        # Arabs graph as hundreds of crores
        suffix_hundred_crores_plural = pynutil.insert(" కోట్లు")
        suffix_hundred_crores_one = pynutil.insert(" కోటి")

        # 100,110,123,200,234 ... crore prefixes

        hundred_crore_prefix = (
            pynini.cross("100", "వంద")
            | pynini.cross("౧౦౦", "వంద")
            | ((pynini.cross("10", "నూట ") | pynini.cross("౧౦", "నూట ")) + digit)
            | ((pynini.cross("1", "నూట ") | pynini.cross("౧", "నూట ")) + teens_ties)
            | create_graph_suffix(
                digit_except_one,
                pynutil.insert(" వందల"),
                2,
            )
            | create_larger_number_graph(
                digit_except_one,
                pynutil.insert(" వందల"),
                1,
                digit_except_one,
            )
            | create_larger_number_graph(
                digit_except_one,
                pynutil.insert(" వందల"),
                0,
                teens_ties,
            )
        ).optimize()

        # 201,301,401...901 crore prefixes రెండు వందల ఒక
        hundred_one_crore_prefix = pynutil.add_weight(digit_except_one + pynutil.delete(NEMO_ALL_ZERO | pynini.accep("౦")) + (pynini.cross("1", " వందల ఒక")| pynini.cross("౧", " వందల ఒక")), -0.3,)
        # 2010000000 -> రెండు వందల ఒక కోటి
        graph_arabs = create_graph_suffix(hundred_one_crore_prefix, suffix_hundred_crores_one, 7,)
        graph_arabs |= create_larger_number_graph(hundred_one_crore_prefix, suffix_hundred_crores_one, 6, digit,)
        graph_arabs |= create_larger_number_graph(hundred_one_crore_prefix, suffix_hundred_crores_one, 5, teens_ties,)
        graph_arabs |= create_larger_number_graph(hundred_one_crore_prefix, suffix_hundred_crores_one, 4, graph_hundreds,)
        graph_arabs |= create_larger_number_graph(hundred_one_crore_prefix, suffix_hundred_crores_one, 3, graph_thousands,)
        graph_arabs |= create_larger_number_graph(hundred_one_crore_prefix, suffix_hundred_crores_one, 2, graph_ten_thousands,)
        graph_arabs |= create_larger_number_graph(hundred_one_crore_prefix, suffix_hundred_crores_one, 1, graph_lakhs,)
        graph_arabs |= create_larger_number_graph(hundred_one_crore_prefix, suffix_hundred_crores_one, 0, graph_ten_lakhs,)

        # Normal:
        # 100 కోట్లు
        # 110 కోట్లు
        # 123 కోట్లు
        # 200 కోట్లు
        graph_arabs |= create_graph_suffix(hundred_crore_prefix, suffix_hundred_crores_plural, 7,)
        graph_arabs |= create_larger_number_graph(hundred_crore_prefix, suffix_hundred_crores_plural, 6, digit,)
        graph_arabs |= create_larger_number_graph(hundred_crore_prefix, suffix_hundred_crores_plural, 5, teens_ties,)
        graph_arabs |= create_larger_number_graph(hundred_crore_prefix, suffix_hundred_crores_plural, 4, graph_hundreds,)
        graph_arabs |= create_larger_number_graph(hundred_crore_prefix, suffix_hundred_crores_plural, 3, graph_thousands,)
        graph_arabs |= create_larger_number_graph(hundred_crore_prefix, suffix_hundred_crores_plural, 2, graph_ten_thousands,)
        graph_arabs |= create_larger_number_graph(hundred_crore_prefix, suffix_hundred_crores_plural, 1, graph_lakhs,)
        graph_arabs |= create_larger_number_graph(hundred_crore_prefix, suffix_hundred_crores_plural, 0, graph_ten_lakhs,)
        graph_arabs |= create_larger_number_graph(hundred_crore_prefix, suffix_hundred_crores_plural, 0, graph_crores,)

        graph_arabs = graph_arabs.optimize()
        graph_ten_arabs = graph_arabs

        # Kharabs graph and ten kharabs graph
        suffix_kharabs = pynutil.insert(" ఖరబ్")
        graph_kharabs = create_graph_suffix(digit, suffix_kharabs, 11)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 10, digit)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 9, teens_ties)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 8, graph_hundreds)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 7, graph_thousands)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 6, graph_ten_thousands)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 5, graph_lakhs)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 4, graph_ten_lakhs)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 3, graph_crores)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 2, graph_ten_crores)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 1, graph_arabs)
        graph_kharabs |= create_larger_number_graph(digit, suffix_kharabs, 0, graph_ten_arabs)
        graph_kharabs.optimize()

        graph_ten_kharabs = create_graph_suffix(teens_and_ties, suffix_kharabs, 11)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 10, digit)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 9, teens_ties)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 8, graph_hundreds)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 7, graph_thousands)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 6, graph_ten_thousands)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 5, graph_lakhs)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 4, graph_ten_lakhs)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 3, graph_crores)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 2, graph_ten_crores)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 1, graph_arabs)
        graph_ten_kharabs |= create_larger_number_graph(teens_and_ties, suffix_kharabs, 0, graph_ten_arabs)
        graph_ten_kharabs.optimize()

        # Nils graph and ten nils graph
        suffix_nils = pynutil.insert(" నీల్")
        graph_nils = create_graph_suffix(digit, suffix_nils, 13)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 12, digit)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 11, teens_ties)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 10, graph_hundreds)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 9, graph_thousands)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 8, graph_ten_thousands)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 7, graph_lakhs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 6, graph_ten_lakhs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 5, graph_crores)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 4, graph_ten_crores)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 3, graph_arabs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 2, graph_ten_arabs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 1, graph_kharabs)
        graph_nils |= create_larger_number_graph(digit, suffix_nils, 0, graph_ten_kharabs)
        graph_nils.optimize()

        graph_ten_nils = create_graph_suffix(teens_and_ties, suffix_nils, 13)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 12, digit)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 11, teens_ties)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 10, graph_hundreds)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 9, graph_thousands)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 8, graph_ten_thousands)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 7, graph_lakhs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 6, graph_ten_lakhs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 5, graph_crores)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 4, graph_ten_crores)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 3, graph_arabs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 2, graph_ten_arabs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 1, graph_kharabs)
        graph_ten_nils |= create_larger_number_graph(teens_and_ties, suffix_nils, 0, graph_ten_kharabs)
        graph_ten_nils.optimize()

        # Padmas graph and ten padmas graph
        suffix_padmas = pynutil.insert(" పద్మ")
        graph_padmas = create_graph_suffix(digit, suffix_padmas, 15)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 14, digit)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 13, teens_ties)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 12, graph_hundreds)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 11, graph_thousands)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 10, graph_ten_thousands)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 9, graph_lakhs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 8, graph_ten_lakhs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 7, graph_crores)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 6, graph_ten_crores)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 5, graph_arabs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 4, graph_ten_arabs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 3, graph_kharabs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 2, graph_ten_kharabs)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 1, graph_nils)
        graph_padmas |= create_larger_number_graph(digit, suffix_padmas, 0, graph_ten_nils)
        graph_padmas.optimize()

        graph_ten_padmas = create_graph_suffix(teens_and_ties, suffix_padmas, 15)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 14, digit)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 13, teens_ties)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 12, graph_hundreds)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 11, graph_thousands)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 10, graph_ten_thousands)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 9, graph_lakhs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 8, graph_ten_lakhs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 7, graph_crores)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 6, graph_ten_crores)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 5, graph_arabs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 4, graph_ten_arabs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 3, graph_kharabs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 2, graph_ten_kharabs)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 1, graph_nils)
        graph_ten_padmas |= create_larger_number_graph(teens_and_ties, suffix_padmas, 0, graph_ten_nils)
        graph_ten_padmas.optimize()

        # Shankhs graph and ten shankhs graph
        suffix_shankhs = pynutil.insert(" పద్మ")
        graph_shankhs = create_graph_suffix(digit, suffix_shankhs, 17)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 16, digit)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 15, teens_ties)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 14, graph_hundreds)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 13, graph_thousands)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 12, graph_ten_thousands)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 11, graph_lakhs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 10, graph_ten_lakhs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 9, graph_crores)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 8, graph_ten_crores)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 7, graph_arabs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 6, graph_ten_arabs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 5, graph_kharabs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 4, graph_ten_kharabs)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 3, graph_nils)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 2, graph_ten_nils)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 1, graph_padmas)
        graph_shankhs |= create_larger_number_graph(digit, suffix_shankhs, 0, graph_ten_padmas)
        graph_shankhs.optimize()

        graph_ten_shankhs = create_graph_suffix(teens_and_ties, suffix_shankhs, 17)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 16, digit)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 15, teens_ties)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 14, graph_hundreds)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 13, graph_thousands)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 12, graph_ten_thousands)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 11, graph_lakhs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 10, graph_ten_lakhs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 9, graph_crores)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 8, graph_ten_crores)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 7, graph_arabs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 6, graph_ten_arabs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 5, graph_kharabs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 4, graph_ten_kharabs)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 3, graph_nils)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 2, graph_ten_nils)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 1, graph_padmas)
        graph_ten_shankhs |= create_larger_number_graph(teens_and_ties, suffix_shankhs, 0, graph_ten_padmas)
        graph_ten_shankhs.optimize()

        # Only match exactly 2 digits to avoid interfering with telephone numbers, decimals, etc.
        # e.g., "०५" -> "शून्य पाँच"
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
            | graph_ten_padmas
            | graph_shankhs
            | graph_ten_shankhs
        )
        self.graph_without_leading_zeros = graph_without_leading_zeros.optimize()

        # Handle numbers with leading zeros by reading digit-by-digit
        # e.g., English/arabic "073" -> "शून्य सात तीन", Hindi/devnagri "००५" -> "शून्य शून्य पाँच"
        cardinal_with_leading_zeros = pynini.compose(
            NEMO_ALL_ZERO + pynini.closure(NEMO_ALL_DIGIT), self.single_digits_graph
        )
        cardinal_with_leading_zeros = pynutil.add_weight(cardinal_with_leading_zeros, 0.5)

        # Full graph including leading zeros - for standalone cardinal matching
        final_graph = graph_without_leading_zeros | cardinal_with_leading_zeros

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        self.final_graph = final_graph.optimize()
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph

