import re
from copy import deepcopy

# from pykakasi import kakasi

# from nemo_text_processing.text_normalization.normalize import Normalizer
from .NeMoTN.nemo_text_processing.text_normalization.normalize import Normalizer

"""
Note : install following libraries before running this code.
# !pip install pykakasi
# !python -m pip install git+https://github.com/NVIDIA/NeMo-text-processing.git@main

This module is for normalizing text.

Done : normal(serial) number, math operator, alphabet, unit, currency, punctuation normalization
TBD : japanese numbering rule(月, 個, count, 人, 時, 枚 etc) / nemo speedup
updated by beomsookim :
  - get text argument at normalizer function
  - nemo normalizer is initialized in init function.
  - command line for testing is added.
"""


class TextNormalizer:
    # matching patterns for regex
    patterns = {
        "operation": r"(?P<operation>\d*[+,/,*]\d+)|(?P<minus>\d*\-\d+)|(?P<equal>\=)",
        "decimal": r"(?P<decimal>\d*\.\d+)",
        "serial": r"(?P<serial>(\d+-\d+-*\d*)+)(?!=\d+)",
        "comma": r"(?P<comma>\d+[\,]\d+)|(?P<dot>([A-Z]\.))(?==[A-Z])",
        "int": r"(?P<num>\d+)",
        "capital": r"(?P<capital>[A-Z]+.*)",
    }

    # dictionary for number and math operator normalization
    num_dict = {
        ".": {"ko": "쩜", "jp": "点"},
        "0": {"ko": "영", "jp": "零"},
        "1": {"ko": "일", "jp": "一"},
        "2": {"ko": "이", "jp": "二"},
        "3": {"ko": "삼", "jp": "三"},
        "4": {"ko": "사", "jp": "四"},
        "6": {"ko": "육", "jp": "六"},
        "5": {"ko": "오", "jp": "五"},
        "7": {"ko": "칠", "jp": "七"},
        "8": {"ko": "팔", "jp": "八"},
        "9": {"ko": "구", "jp": "九"},
        "10": {"ko": "십", "jp": "十"},
        "100": {"ko": "백", "jp": "百"},
        "1000": {"ko": "천", "jp": "千"},
        "10000": {"ko": "만", "jp": "万"},
        "100000000": {"ko": "억", "jp": "億"},
        "1000000000000": {"ko": "조", "jp": "兆"},
        "300": {"ko": "삼백", "jp": "三百"},
        "600": {"ko": "육백", "jp": "六百"},
        "800": {"ko": "팔백", "jp": "八百"},
        "3000": {"ko": "삼천", "jp": "三千"},
        "8000": {"ko": "팔천", "jp": "八千"},
        "01000": {"ko": "천", "jp": "一千"},
        "+": {"ko": "플러스", "jp": "プラス"},
        "-": {"ko": "마이너스", "jp": "マイナス"},
        "*": {"ko": "곱하기", "jp": "掛ける"},
        "/": {"ko": "분의", "jp": "分の"},
        "=": {"ko": "는", "jp": "は"},
    }

    # dictionary for alphabet normalization
    alphabet_dict = {
        "A": {"ko": "에이", "jp": "エイ"},
        "B": {"ko": "비", "jp": "ビー"},
        "C": {"ko": "씨", "jp": "シー"},
        "D": {"ko": "디", "jp": "ディー"},
        "E": {"ko": "이", "jp": "イー"},
        "F": {"ko": "에프", "jp": "エフ"},
        "G": {"ko": "쥐", "jp": "ジー"},
        "H": {"ko": "에이치", "jp": "エイチ"},
        "I": {"ko": "아이", "jp": "アイ"},
        "J": {"ko": "제이", "jp": "ジェイ"},
        "K": {"ko": "케이", "jp": "ケイ"},
        "L": {"ko": "엘", "jp": "エル"},
        "M": {"ko": "엠", "jp": "エム"},
        "N": {"ko": "엔", "jp": "エン"},
        "O": {"ko": "오", "jp": "オー"},
        "P": {"ko": "피", "jp": "ピー"},
        "Q": {"ko": "큐", "jp": "キュー"},
        "R": {"ko": "알", "jp": "アール"},
        "S": {"ko": "에스", "jp": "エス"},
        "T": {"ko": "티", "jp": "ティー"},
        "U": {"ko": "유", "jp": "ユー"},
        "V": {"ko": "브이", "jp": "ブイ"},
        "W": {"ko": "더블유", "jp": "ダブリュー"},
        "X": {"ko": "엑스", "jp": "エックス"},
        "Y": {"ko": "와이", "jp": "ワイ"},
        "Z": {"ko": "지", "jp": "ジー"},
    }

    # dictionary for unit normalization
    stdunit_dict = {
        "km": {"ko": "킬로미터", "jp": "キロメートル"},
        "cm": {"ko": "센티미터", "jp": "センチメートル"},
        "mm": {"ko": "밀리미터", "jp": "ミリメートル"},
        "ml": {"ko": "밀리리터", "jp": "ミリリットル"},
        "mg": {"ko": "밀리그램", "jp": "ミリグラム"},
        "kg": {"ko": "킬로그램", "jp": "キログラム"},
        "ton": {"ko": "톤", "jp": "トン"},
        "byte": {"ko": "바이트", "jp": "バイト"},
        "bit": {"ko": "비트", "jp": "ビット"},
        "tb": {"ko": "테라바이트", "jp": "テラバイト"},
        "gb": {"ko": "기가바이트", "jp": "ギガバイト"},
        "mb": {"ko": "메가바이트", "jp": "メガバイト"},
        "kcal": {"ko": "칼로리", "jp": "キロカロリー"},
        "cal": {"ko": "칼로리", "jp": "カロリー"},
        "mhz": {"ko": "메가헤르츠", "jp": "メガヘルツ"},
        "hz": {"ko": "헤르츠", "jp": "ヘルツ"},
        "w": {"ko": "와트", "jp": "ワット"},
        "v": {"ko": "볼트", "jp": "ボルト"},
        "a": {"ko": "암페어", "jp": "アンペア"},
        "m": {"ko": "미터", "jp": "メートル"},
        "g": {"ko": "그램", "jp": "グラム"},
        "l": {"ko": "리터", "jp": "リットル"},
        "t": {"ko": "톤", "jp": "トン"},
        "b": {"ko": "바이트", "jp": "バイト"},
    }

    # dictionary for punctuation normalization
    punc_dict = {
        "&": {"ko": "앤드", "jp": "アンド"},
        "#": {"ko": "샵", "jp": "シャープ"},
        "%": {"ko": "퍼센트", "jp": "パーセント"},
        "@": {"ko": "앳", "jp": "アット"},
        "°C": {"ko": "도", "jp": "度"},
        "°F": {"ko": "도 파렌하이트", "jp": "ディグリー ファレン"},
    }

    curr_dict = {
        "$": {"ko": "달러", "jp": "ドル"},
        "€": {"ko": "유로", "jp": "ユーロ"},
        "£": {"ko": "파운드", "jp": "ポンド"},
        "¥": {"ko": "엔", "jp": "円"},
        "₩": {"ko": "원", "jp": "ウォン"},
        "₹": {"ko": "루피", "jp": "ルピー"},
        "₽": {"ko": "루블", "jp": "ルーブル"},
    }

    def __init__(self, lang="ko"):
        self.lang = lang
        try:
            self.normalizer = Normalizer(input_case="cased", lang=self.lang)
            print("text normalizer from nvidia-NEMO is initialized.")
        except:
            self.normalizer = None
            print(f"text normalizer from nvidia-NEMO is empty for LANG:{lang}.")

    def _flatten(self, lst):
        # flatten nested list
        flat_list = []
        for item in lst:
            if isinstance(item, list):
                flat_list.extend(self._flatten(item))
            else:
                flat_list.append(item)
        return flat_list

    def _remove_anomaly_char(self, text):
        # remove anomalies not in korean, english, japanese, chinese, number, punctuation chars
        reg = re.compile(
            "[^ㄱ-ㅎ|ㅏ-ㅣ|가-힣|\u4E00-\u9FFF|\u3040-\u309F|\u30A0-\u30FF|&$%#?!~\.,\s\u2026|a-zA-Z|\d]"
        )
        return reg.sub("", text)

    @staticmethod
    def _replace_comma(match):
        """
        Remove comma in numbers.
        Remove dot used in capital letters.
        :param match: re.match instance
        :return: plain text without comma and dot
        """
        # two matching patterns for detecting comma
        match = tuple(x for x in match.groups() if x is not None)[0]
        return "".join([x for x in match if x not in [",", "."]])

    @staticmethod
    def _find_digit(num: str):
        """
        Split num to find the largest number of digits when pronouncing a number.
        ex) 3만 오천 -> '만' is maximum unit to pronounce
        :param num: str
        :return: 1230000 -> ['123', '10000'], 12000 -> ['12000']
        """
        digit = [
            2,
            3,
            4,
            5,
            9,
            13,
        ]  # digits of 10, 10^2, 10^3, 10^4, 10^8, 10^13
        assert len(num) < 17, "digit must be less than 17"
        if len(num) < 2:
            return [num]

        if len(num) in digit:
            return [num]
        else:
            tmp = sorted(digit + [len(num)])
            idx = tmp.index(len(num))
            return [
                num[: -(digit[idx - 1] - 1)],
                "1" + num[-(digit[idx - 1] - 1) :],
            ]

    def _decompose_num(self, num: str):
        """
        Converts a number to its phonetic matching form and returns it.
        :param num: str
        :return: ex) num = '22500' -> [['2'], '10000', '2', '1000', '5', '100']
        """
        if not num:
            pass
        if num in self.num_dict.keys():
            return [num]

        divisor = str(int(num) // (pow(10, len(num) - 1)))
        position_val = str(pow(10, len(num) - 1))
        residual = str(int(num) % (pow(10, len(num) - 1)))

        if divisor == "1":
            # okay to pronounce as it is. ex) 100 -> 백
            res = [position_val]
        else:
            res = [self._decompose_num(divisor), position_val]

        if residual != "0":
            residual = self._find_digit(residual)
            res.extend(
                self._flatten([self._decompose_num(x) for x in residual])
            )

        return res

    def _replace_int(self, match):
        """
        Transforms a normal integer into its pronunciation.
        :param match: can be str(int) or re.match instance
        :return: pronunciation of integer
        """

        # match can be either a str or re.match instance depending on the way the func is called.
        input_num = match
        if not isinstance(match, str):
            input_num = match.groups()[0]

        if len(input_num) in [9, 13]:
            res = self._flatten(
                ["1"]
                + [self._decompose_num(x) for x in self._find_digit(input_num)]
            )
        else:
            res = self._flatten(
                [
                    self._decompose_num(x)
                    for x in self._find_digit(input_num)
                    if x != ""
                ]
            )

        return "".join([self.num_dict[x][self.lang] for x in res])

    def _replace_serial(self, match):
        """
        Replace the serial number with its pronunciation.
        Temporarily mutate dict because 0 is pronounced differently when pronouncing decimal numbers and serial numbers. - is also pronounced differently in this case.
        :param match: re.match instance
        :return: pronunciation of serial number
        """
        original_num_dict = deepcopy(self.num_dict)

        self.num_dict["-"] = {"ko": "에", "jp": "の"}
        self.num_dict["0"] = {"ko": "영", "jp": "ゼロ"}

        match = match.groups()[0]
        match = list(match)
        res = []
        for item in match:
            try:
                res.append(self._replace_int(item))
            except KeyError:
                if item:
                    res.append(self.punc_dict[item][self.lang])
                else:
                    pass

        self.num_dict.clear()
        self.num_dict.update(original_num_dict)

        return "".join(res)

    def _replace_decimal(self, match):
        """
        Replace the decimal number with its pronunciation.
        :param match: re.match instance
        :return: pronounciation of decimal number
        """
        match = match.groups()[0]
        match = re.split(r"(\.)", match)

        # split numbers following after precision point
        match[-1] = list(match[-1])
        match = self._flatten(match)

        res = [self._replace_int(x) for x in match]
        return "".join(res)

    def _replace_equation(self, match):
        """
        Replace the equation with its pronunciation.
        :param match: re.match instance
        :return: pronounciation of equation and math operations
        """
        # two matching patterns for detecting math operations
        match = tuple(x for x in match.groups() if x is not None)[0]
        match = re.split(r"(\+|-|\*|/)", match)

        # swap positions of divisor and dividend to pronounce correctly ex) 3/2 => 2분의 3
        if "/" in match:
            pos = match.index("/")
            match[pos - 1], match[pos + 1] = match[pos + 1], match[pos - 1]

        res = []
        for item in match:
            try:
                res.append(self._replace_int(item))
            except KeyError:
                if item:
                    res.append(self.num_dict[item][self.lang])
                else:
                    pass

        res = "".join(res)

        return res

    def _remove_comma_and_replace_curr(self, text):
        """
        Remove comma and replace currency with its pronunciation.
        :return: text with replaced currency
        """
        text = re.sub(self.patterns["comma"], self._replace_comma, text)
        for symbol in self.curr_dict.keys():
            pattern = re.compile(r"({}\d+)".format(re.escape(symbol)))
            matches = pattern.findall(text)

            for match in matches:
                # 숫자와 기호의 순서를 변경
                switched = match[len(symbol) :] + match[: len(symbol)]
                text = text.replace(match, switched)
                print(text)
        pattern = "|".join(map(re.escape, self.curr_dict.keys()))
        text = re.sub(
            pattern, lambda m: self.curr_dict[m.group(0)][self.lang], text
        )
        return text

    def _replace_num(self, text):
        """
        Replace all types of number with its pronunciation.
        The order of replacement is important, because 0 and - pronounced differently when pronouncing decimal numbers and serial numbers.
        :return: text with replaced numbers
        """
        text = re.sub(self.patterns["serial"], self._replace_serial, text)
        text = re.sub(self.patterns["decimal"], self._replace_decimal, text)
        text = re.sub(self.patterns["operation"], self._replace_equation, text)
        text = re.sub(self.patterns["int"], self._replace_int, text)
        return text

    def _replace_cap_letters(self, text):
        """
        Replace capital letters with its pronunciation.
        :return: text with replaced capital letters
        """
        pattern = "|".join(map(re.escape, self.alphabet_dict.keys()))
        text = re.sub(
            pattern, lambda m: self.alphabet_dict[m.group(0)][self.lang], text
        )
        return text

    def _replace_unit(self, text):
        """
        Replace unit letters with its pronunciation.
        :return: text with replaced unit letters
        """
        pattern = "|".join(map(re.escape, self.stdunit_dict.keys()))
        text = re.sub(
            pattern, lambda m: self.stdunit_dict[m.group(0)][self.lang], text
        )
        return text

    def _replace_punc(self, text):
        """
        Replace punctuation with its pronunciation.
        :return: text with replaced punctuation
        """
        pattern = "|".join(map(re.escape, self.punc_dict.keys()))
        text = re.sub(
            pattern, lambda m: self.punc_dict[m.group(0)][self.lang], text
        )
        return text

    def normalize(self, text):
        if self.lang in ["ko", "jp"]:
            text = self._remove_comma_and_replace_curr(text)
            text = self._replace_num(text)
            text = self._replace_unit(text)
            text = self._replace_punc(text)
            text = self._replace_cap_letters(text)
            text = self._remove_anomaly_char(text)

            if self.lang == "jp":
                k = kakasi()
                result = k.convert(text)
                text = "".join(
                    [
                        x["kana"] if x["kana"] == x["orig"] else x["hira"]
                        for x in result
                    ]
                )
            elif self.lang == "ko":
                pass
            return text
        else:
            if self.lang not in [
                "en",
                "es",
                "fr",
                "de",
                "ar",
                "ru",
                "sv",
                "vi",
                "pt",
                "zh",
                "hu",
                "it",
            ]:
                raise NotImplementedError("Language not supported.")
            return self.normalizer.normalize(
                text, verbose=False, punct_post_process=False
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en")
    parser.add_argument("--text", default="i am 20 years old")
    args = parser.parse_args()

    normalizer = TextNormalizer(lang=args.lang)
    print(normalizer.normalize(args.text))
    """
    python normalize_text.py -> result : i am twenty years old
    """
