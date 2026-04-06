"""
Tests for lexicon_functions.py

Run with:  pytest test_lexicon_functions.py -v

These tests are self-contained and do not require any lexicon CSV files.
They test the core logic functions directly by constructing minimal inputs.
"""

import re
import pytest
import pandas as pd
import numpy as np

from lexicon_functions import (
    preproc_text,
    load_codebook_with_tokens,
    load_codebook_with_pairs,
    check_text_against_tokens,
    check_text_against_pairs,
    apply_codebook_to_column,
    get_matching_token,
    flag_suicide_related_emojis,
)


# ---------------------------------------------------------------------------
# preproc_text
# ---------------------------------------------------------------------------

class TestPreprocText:
    def test_lowercases(self):
        assert preproc_text("HELLO World") == "hello world"

    def test_removes_punctuation(self):
        result = preproc_text("hello, world!")
        assert "," not in result
        assert "!" not in result

    def test_preserves_pipe(self):
        # pipe is used as token separator in pair codebooks
        assert "|" in preproc_text("foo|bar")

    def test_empty_string(self):
        assert preproc_text("") == ""

    def test_numbers_preserved(self):
        assert "123" in preproc_text("abc 123")


# ---------------------------------------------------------------------------
# load_codebook_with_tokens
# ---------------------------------------------------------------------------

class TestLoadCodebookWithTokens:
    def _make_df(self, tokens, word_start=0, word_end=0):
        return pd.DataFrame({
            "token": tokens,
            "word_start": word_start,
            "word_end": word_end,
        })

    def test_returns_list_of_tuples(self):
        df = self._make_df(["kill", "die"])
        result = load_codebook_with_tokens(df)
        assert isinstance(result, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)

    def test_compiled_pattern_matches(self):
        df = self._make_df(["suicide"])
        patterns = load_codebook_with_tokens(df)
        compiled, original = patterns[0]
        assert compiled.search("I want to commit suicide today")
        assert original == "suicide"

    def test_case_insensitive(self):
        df = self._make_df(["Suicide"])
        patterns = load_codebook_with_tokens(df)
        compiled, _ = patterns[0]
        assert compiled.search("SUICIDE")
        assert compiled.search("suicide")

    def test_word_start_boundary(self):
        df = self._make_df(["hang"], word_start=1)
        patterns = load_codebook_with_tokens(df)
        compiled, _ = patterns[0]
        # "hang" at word start should match
        assert compiled.search("hang yourself")
        # "overhang" should NOT match (no word boundary at start)
        assert not compiled.search("overhang")

    def test_word_end_boundary(self):
        df = self._make_df(["die"], word_end=1)
        patterns = load_codebook_with_tokens(df)
        compiled, _ = patterns[0]
        assert compiled.search("I want to die")
        assert not compiled.search("died")  # 'd' follows immediately

    def test_regex_special_chars_escaped(self):
        # Token with regex metacharacter should not raise or misfire
        df = self._make_df(["end it all."])
        patterns = load_codebook_with_tokens(df)
        compiled, _ = patterns[0]
        assert compiled.search("end it all.")
        # The dot should be literal, not match any char
        assert not compiled.search("end it allX")


# ---------------------------------------------------------------------------
# load_codebook_with_pairs
# ---------------------------------------------------------------------------

class TestLoadCodebookWithPairs:
    def _make_df(self, tokens_col, pairs_col, freestanding_token=False, freestanding_pair=False):
        return pd.DataFrame({
            "tokens": tokens_col,
            "pairs": pairs_col,
            "freestanding_token": freestanding_token,
            "freestanding_pair": freestanding_pair,
        })

    def test_returns_list_of_tuples(self):
        df = self._make_df(["kill|end"], ["myself|life"])
        result = load_codebook_with_pairs(df)
        assert isinstance(result, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in result)

    def test_both_patterns_compiled(self):
        df = self._make_df(["kill"], ["myself"])
        result = load_codebook_with_pairs(df)
        means_re, pairs_re = result[0]
        assert hasattr(means_re, "search")
        assert hasattr(pairs_re, "search")

    def test_pipe_separated_tokens(self):
        df = self._make_df(["kill|end|finish"], ["myself|life"])
        result = load_codebook_with_pairs(df)
        means_re, _ = result[0]
        assert means_re.search("end")
        assert means_re.search("kill")
        assert means_re.search("finish")

    def test_freestanding_token_boundary(self):
        df = self._make_df(["die"], ["alone"], freestanding_token=True)
        result = load_codebook_with_pairs(df)
        means_re, _ = result[0]
        assert means_re.search("I want to die")
        assert not means_re.search("odies")  # embedded

    def test_missing_freestanding_columns_defaults_to_false(self):
        # Should not raise even if freestanding columns absent
        df = pd.DataFrame({"tokens": ["kill"], "pairs": ["myself"]})
        result = load_codebook_with_pairs(df)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# check_text_against_tokens
# ---------------------------------------------------------------------------

class TestCheckTextAgainstTokens:
    def _patterns(self, tokens):
        df = pd.DataFrame({"token": tokens, "word_start": 0, "word_end": 0})
        return load_codebook_with_tokens(df)

    def test_match_returns_1(self):
        patterns = self._patterns(["suicide"])
        assert check_text_against_tokens("thinking about suicide", patterns) == 1

    def test_no_match_returns_0(self):
        patterns = self._patterns(["suicide"])
        assert check_text_against_tokens("having a great day", patterns) == 0

    def test_empty_text_returns_0(self):
        patterns = self._patterns(["suicide"])
        assert check_text_against_tokens("", patterns) == 0

    def test_empty_patterns_returns_0(self):
        assert check_text_against_tokens("some text", []) == 0

    def test_first_match_short_circuits(self):
        # Both tokens present; should return 1 regardless
        patterns = self._patterns(["kill", "die"])
        assert check_text_against_tokens("kill and die", patterns) == 1

    def test_partial_match_without_boundary(self):
        # "die" inside "odies" should match when no boundary set
        patterns = self._patterns(["die"])
        assert check_text_against_tokens("bodies", patterns) == 1


# ---------------------------------------------------------------------------
# check_text_against_pairs
# ---------------------------------------------------------------------------

class TestCheckTextAgainstPairs:
    def _patterns(self, token_pairs):
        rows = [{"tokens": t, "pairs": p} for t, p in token_pairs]
        df = pd.DataFrame(rows)
        df["freestanding_token"] = False
        df["freestanding_pair"] = False
        return load_codebook_with_pairs(df)

    def test_both_present_returns_1(self):
        patterns = self._patterns([("kill", "myself")])
        assert check_text_against_pairs("i want to kill myself", patterns) == 1

    def test_only_first_present_returns_0(self):
        patterns = self._patterns([("kill", "myself")])
        assert check_text_against_pairs("i want to kill", patterns) == 0

    def test_only_second_present_returns_0(self):
        patterns = self._patterns([("kill", "myself")])
        assert check_text_against_pairs("hate myself", patterns) == 0

    def test_neither_present_returns_0(self):
        patterns = self._patterns([("kill", "myself")])
        assert check_text_against_pairs("having a great day", patterns) == 0

    def test_empty_text_returns_0(self):
        patterns = self._patterns([("kill", "myself")])
        assert check_text_against_pairs("", patterns) == 0


# ---------------------------------------------------------------------------
# get_matching_token
# ---------------------------------------------------------------------------

class TestGetMatchingToken:
    def _token_patterns(self, tokens):
        df = pd.DataFrame({"token": tokens, "word_start": 0, "word_end": 0})
        return load_codebook_with_tokens(df)

    def _pair_patterns(self, token_pairs):
        rows = [{"tokens": t, "pairs": p, "freestanding_token": False, "freestanding_pair": False}
                for t, p in token_pairs]
        return load_codebook_with_pairs(pd.DataFrame(rows))

    def test_paired_match_returns_pattern_tuple(self):
        patterns = self._pair_patterns([("kill", "myself")])
        result = get_matching_token("i want to kill myself", patterns, paired=True)
        assert result is not None
        assert isinstance(result, tuple)

    def test_paired_no_match_returns_none(self):
        patterns = self._pair_patterns([("kill", "myself")])
        result = get_matching_token("great day", patterns, paired=True)
        assert result is None

    def test_unpaired_match_returns_token_string(self):
        # This test will FAIL on the unpatched code (bug #1 from review).
        # After fix, get_matching_token should use compiled.search(), not `token in text`.
        patterns = self._token_patterns(["suicide"])
        result = get_matching_token("thinking about suicide", patterns, paired=False)
        assert result == "suicide"

    def test_unpaired_no_match_returns_none(self):
        patterns = self._token_patterns(["suicide"])
        result = get_matching_token("great day", patterns, paired=False)
        assert result is None

    def test_unpaired_respects_word_boundary(self):
        # Token compiled with word_start=1; should not match mid-word
        df = pd.DataFrame({"token": ["hang"], "word_start": 1, "word_end": 0})
        patterns = load_codebook_with_tokens(df)
        assert get_matching_token("overhang", patterns, paired=False) is None
        assert get_matching_token("hang yourself", patterns, paired=False) == "hang"


# ---------------------------------------------------------------------------
# apply_codebook_to_column
# ---------------------------------------------------------------------------

class TestApplyCodebookToColumn:
    def _df(self, texts):
        return pd.DataFrame({"text": texts})

    def _token_patterns(self, tokens):
        df = pd.DataFrame({"token": tokens, "word_start": 0, "word_end": 0})
        return load_codebook_with_tokens(df)

    def test_returns_series_of_correct_length(self):
        df = self._df(["hello", "suicide note", "fine"])
        patterns = self._token_patterns(["suicide"])
        result = apply_codebook_to_column(df, "text", patterns, paired=False)
        assert len(result) == 3

    def test_correct_flags_single_token(self):
        df = self._df(["I want to die", "having a good day", "kill myself"])
        patterns = self._token_patterns(["die", "kill"])
        result = apply_codebook_to_column(df, "text", patterns, paired=False)
        assert list(result) == [1, 0, 1]

    def test_correct_flags_pairs(self):
        rows = [{"tokens": "kill", "pairs": "myself",
                 "freestanding_token": False, "freestanding_pair": False}]
        pair_patterns = load_codebook_with_pairs(pd.DataFrame(rows))
        df = self._df(["kill myself", "kill the spider", "hate myself"])
        result = apply_codebook_to_column(df, "text", pair_patterns, paired=True)
        assert list(result) == [1, 0, 0]

    def test_all_zeros_no_match(self):
        df = self._df(["sunny day", "great weather", "love life"])
        patterns = self._token_patterns(["suicide"])
        result = apply_codebook_to_column(df, "text", patterns, paired=False)
        assert result.sum() == 0

    def test_stem_matches_variants(self):
        df = self._df(["suicide", "suicidal", "suicided"])
        patterns = self._token_patterns(["suicid"])  # actual shared stem
        result = apply_codebook_to_column(df, "text", patterns, paired=False)
        assert result.sum() == 3


# ---------------------------------------------------------------------------
# flag_suicide_related_emojis
# ---------------------------------------------------------------------------

class TestFlagSuicideRelatedEmojis:
    def _df(self, texts):
        return pd.DataFrame({"text": texts})

    def test_single_high_risk_emoji_flagged(self):
        df = self._df(["feeling 🔫 today"])
        result = flag_suicide_related_emojis(df, "text")
        assert result["emoji_flag"].iloc[0] == 1

    def test_no_emoji_not_flagged(self):
        df = self._df(["just a normal message"])
        result = flag_suicide_related_emojis(df, "text")
        assert result["emoji_flag"].iloc[0] == 0

    def test_one_combo_emoji_not_flagged(self):
        # A single combo-only emoji (not in single_emoji_flags) should NOT trigger
        df = self._df(["🚬 smoke"])
        result = flag_suicide_related_emojis(df, "text")
        assert result["emoji_flag"].iloc[0] == 0

    def test_two_combo_emojis_flagged(self):
        df = self._df(["🚬 🖤"])
        result = flag_suicide_related_emojis(df, "text")
        assert result["emoji_flag"].iloc[0] == 1

    def test_flag_col_in_output(self):
        df = self._df(["test"])
        result = flag_suicide_related_emojis(df, "text")
        assert "emoji_flag" in result.columns
        assert "emojis_found" in result.columns

    def test_custom_output_col_names(self):
        df = self._df(["test"])
        result = flag_suicide_related_emojis(df, "text",
                                              output_flag_col="my_flag",
                                              output_list_col="my_list")
        assert "my_flag" in result.columns
        assert "my_list" in result.columns

    def test_nan_text_handled(self):
        df = pd.DataFrame({"text": [None, float("nan"), "normal"]})
        result = flag_suicide_related_emojis(df, "text")
        assert result["emoji_flag"].iloc[0] == 0
        assert result["emoji_flag"].iloc[1] == 0

    def test_skull_in_found_list(self):
        df = self._df(["💀"])
        result = flag_suicide_related_emojis(df, "text")
        assert "skull" in result["emojis_found"].iloc[0]

    def test_same_combo_emoji_repeated_does_not_trigger_combo(self):
        # 💀 repeated — combo_count is still 1 (distinct types), not 2
        # 💀 IS in single_emoji_flags, so this is flagged via that path, not combo
        # This test documents the distinct-types behavior (see bug note in review)
        df = self._df(["💀💀💀"])
        result = flag_suicide_related_emojis(df, "text")
        assert result["emoji_flag"].iloc[0] == 1  # flagged via single, not combo

    def test_two_distinct_combo_emojis_triggers_combo(self):
        # 🚬 and 🖤 are combo-only; two distinct types should trigger
        df = self._df(["🚬🖤"])
        result = flag_suicide_related_emojis(df, "text")
        assert result["emoji_flag"].iloc[0] == 1

    def test_original_df_not_mutated_beyond_new_cols(self):
        df = self._df(["test message"])
        original_text = df["text"].iloc[0]
        flag_suicide_related_emojis(df, "text")
        assert df["text"].iloc[0] == original_text