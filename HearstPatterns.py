"""
Problem 1 aims to detect Hearst patterns of a certain list of words (NPs) in a text corpus (s) and return a set of tuples (b, a),
where b is the superclass and a is the subclass. Patterns of type 1 refer to: “a is b”, “a is a type of b”, “a is a kind of b”,
“a was b”, “a was a type of b”, “a was a kind of b”, “a are b”, “a are a type of b”, and “a are a kind of b”. Patterns of type 2
refer to: “a[,] including b”, “a[,] including b, c, …[,] and d”, “a[,] including b, c, …[,] or d”, “a[,] such as b”,
“a[,] such as b, c, …[,] and d”, and “a[,] such as b, c, …[,] or d”.
"""
import re  # for using regex


def problem1(NPs, s):
    input_text = s.lower()  # convert input list to lowercase
    output_set = set()  # initialize output set
    np_word = '|'.join([re.escape(word.lower()) for word in NPs])  # create variable to store the noun phrases

    # parse through text using 2 cases of regex
    pattern_type1 = rf'\b({np_word})\b\s+(?:is|are|was)\s+(?:a|an)?\s*(?:type|kind)?\s*(?:of)?\s*(?:a|an)?\s*\b({np_word})\b'
    pattern_type2 = [
        rf'\b({np_word})\b[,]? (?:including|such as) ({np_word})',
        rf'\b({np_word})[,]? (?:including|such as) ({np_word})(?:, ({np_word}))*[,]? (?:and|or) ({np_word})\b'
    ]

    # find matches to pattern 1 and regardless of capitalization
    match1 = re.findall(pattern_type1, s, flags=re.IGNORECASE)
    for a, b in match1:
        a = a.lower()
        b = b.lower()
        if a != b and a in input_text and b in input_text:  # double checks that word is in input text
            output_set.add((b, a))
    for pt in pattern_type2:  # pattern 2 has 2 cases, so a loop is needed
        match2 = re.findall(pt, s, flags=re.IGNORECASE)
        for match in match2:
            b = match[0].lower()  # first word in match is the superclass
            a_group = [a.lower() for a in match[1:] if a]  # rest of the words in match are the subclasses
            for a in a_group:
                if b != a and b in input_text and a in input_text:
                    output_set.add((b, a))

    return output_set
