# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.inverse_text_normalization.pt.utils import (
    get_abs_path,
)
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money
        e.g. doze dólares e cinco centavos -> money { integer_part: "12" fractional_part: "05" currency: "$" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="money", kind="classify")
        # quantity, integer_part, fractional_part, currency

        cardinal_graph = cardinal.graph_no_exception
        graph_decimal_final = decimal.final_graph_wo_negative

        unit_singular = pynini.string_file(
            get_abs_path("data/currency_singular.tsv")
        ).invert()
        unit_plural = pynini.string_file(
            get_abs_path("data/currency_plural.tsv")
        ).invert()

        graph_unit_singular = (
            pynutil.insert('currency: "')
            + convert_space(unit_singular)
            + pynutil.insert('"')
        )
        graph_unit_plural = (
            pynutil.insert('currency: "')
            + convert_space(unit_plural)
            + pynutil.insert('"')
        )

        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (
            pynutil.insert("0") + NEMO_DIGIT
        )
        # twelve dollars (and) fifty cents, zero cents
        cents_standalone = (
            pynutil.insert(
                'morphosyntactic_features: ","'
            )  # always use a comma in the decimal
            + insert_space
            + pynutil.insert('fractional_part: "')
            + pynini.union(
                pynutil.add_weight(
                    ((NEMO_SIGMA - "um" - "uma") @ cardinal_graph), -0.7
                )
                @ add_leading_zero_to_double_digit
                + delete_space
                + pynutil.delete(pynini.union("centavos")),
                pynini.cross("um", "01")
                + delete_space
                + pynutil.delete(pynini.union("centavo")),
            )
            + pynutil.insert('"')
        )

        optional_cents_standalone = pynini.closure(
            delete_space
            + pynini.closure(
                (pynutil.delete("com") | pynutil.delete("e")) + delete_space,
                0,
                1,
            )
            + insert_space
            + cents_standalone,
            0,
            1,
        )

        # twelve dollars fifty, only after integer
        # setenta e cinco dólares com sessenta e três ~ $75,63
        optional_cents_suffix = pynini.closure(
            delete_extra_space
            + pynutil.insert(
                'morphosyntactic_features: ","'
            )  # always use a comma in the decimal
            + insert_space
            + pynutil.insert('fractional_part: "')
            + pynini.closure(
                (pynutil.delete("com") | pynutil.delete("e")) + delete_space,
                0,
                1,
            )
            + pynutil.add_weight(
                cardinal_graph @ add_leading_zero_to_double_digit, -0.7
            )
            + pynutil.insert('"'),
            0,
            1,
        )

        graph_integer = (
            pynutil.insert('integer_part: "')
            + ((NEMO_SIGMA - "um" - "uma") @ cardinal_graph)
            + pynutil.insert('"')
            + delete_extra_space
            + graph_unit_plural
            + (optional_cents_standalone | optional_cents_suffix)
        )
        graph_integer |= (
            pynutil.insert('integer_part: "')
            + (pynini.cross("um", "1") | pynini.cross("uma", "1"))
            + pynutil.insert('"')
            + delete_extra_space
            + graph_unit_singular
            + (optional_cents_standalone | optional_cents_suffix)
        )

        graph_cents_standalone = pynini.union(
            pynutil.insert('currency: "R$" integer_part: "0" ')
            + cents_standalone,
            pynutil.add_weight(
                pynutil.insert('integer_part: "0" ')
                + cents_standalone
                + delete_extra_space
                + pynutil.delete("de")
                + delete_space
                + graph_unit_singular,
                -0.1,
            ),
        )

        graph_decimal = (
            graph_decimal_final
            + delete_extra_space
            + (pynutil.delete("de") + delete_space).ques
            + graph_unit_plural
        )
        graph_decimal |= graph_cents_standalone
        final_graph = graph_integer | graph_decimal
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
