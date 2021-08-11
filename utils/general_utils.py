
import re
import math


def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


def get_module_name(mn):
    if not isinstance(mn, str):
        mn = str(mn)
    return re.sub(pattern="['<>]", repl="", string=mn).split()[-1].replace("dllib.", "")


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor