#!/usr/bin/env python3
import argparse
import curses
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestConfig:
    words: int
    text_file: Optional[str]
    random_words_file: Optional[str]
    line_width: int
    cursor_style: str
    show_whitespace: bool


@dataclass
class TestResult:
    elapsed: float
    correct_chars: int
    total_chars: int
    mistakes: int
    stability: float


@dataclass
class TestState:
    target: str
    typed: List[Optional[str]]
    pos: int
    start_time: Optional[float]
    end_time: Optional[float]
    mistakes: int
    keypress_times: List[float]


def parse_args() -> TestConfig:
    parser = argparse.ArgumentParser(
        description="CLI typing trainer.",
        epilog=(
            "Examples:\n"
            "  typing_trainer.py --words 25\n"
            "  typing_trainer.py --text-file path/to/text.txt\n"
            "  typing_trainer.py --random-words-file path/to/words.txt --words 50\n"
            "\n"
            "Source selection:\n"
            "  If no file is provided, a random file from ./files is used\n"
            "  as the random-words source\n"
            "\n"
            "Whitespace visibility:\n"
            "  Use --show-whitespace to display symbols for space/tab/newline\n"
            "\n"
            "Tabs:\n"
            "  Tabs in text files are converted to four spaces\n"
            "  Pressing Tab types four spaces but counts as one keystroke\n"
            "\n"
            "Results:\n"
            "  WPM: words per minute (5 chars = 1 word, correct chars only)\n"
            "  CPM: characters per minute (correct chars only)\n"
            "  Mistakes: total incorrect keystrokes (includes corrected)\n"
            "  Stability: 0-100 based on average deviation from ideal rhythm\n"
            "    100 = perfectly even intervals (metronome-like)\n"
            "      0 = very uneven intervals (high variability)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--words", type=int, default=50, help="Number of words (default: 50).")
    parser.add_argument("--text-file", type=str, default=None, help="Full text file for the test.")
    parser.add_argument(
        "--random-words-file",
        type=str,
        default=None,
        help="Text file to sample random words from.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=75,
        help="Max line width for random words (default: 75).",
    )
    parser.add_argument(
        "--cursor-style",
        choices=["bar", "highlight", "underline"],
        default="underline",
        help="Cursor style: bar, highlight, or underline (default: underline).",
    )
    parser.add_argument(
        "--show-whitespace",
        action="store_true",
        help="Show visible symbols for spaces/tabs/newlines.",
    )
    args = parser.parse_args()

    if args.words <= 0:
        parser.error("--words must be a positive integer")
    if args.text_file and args.random_words_file:
        parser.error("Use only one of --text-file or --random-words-file")
    if args.line_width <= 0:
        parser.error("--line-width must be a positive integer")

    return TestConfig(
        words=args.words,
        text_file=args.text_file,
        random_words_file=args.random_words_file,
        line_width=args.line_width,
        cursor_style=args.cursor_style,
        show_whitespace=args.show_whitespace,
    )


def load_words_from_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    words = [w for w in content.split() if w.strip()]
    if not words:
        raise ValueError("No words found in file")
    return words


def load_full_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if not content.strip():
        raise ValueError("Text file is empty")
    return content.replace("\t", "    ")


def build_test_text(cfg: TestConfig) -> str:
    if cfg.text_file:
        return load_full_text(cfg.text_file)
    words_file = cfg.random_words_file or pick_words_file_from_dir()
    if words_file:
        words = load_words_from_file(words_file)
        chosen = [random.choice(words) for _ in range(cfg.words)]
        return " ".join(chosen)
    raise ValueError("No words source available. Add files to ./files or provide --random-words-file.")



def pick_words_file_from_dir() -> Optional[str]:
    dir_path = os.path.join(os.getcwd(), "files")
    if not os.path.isdir(dir_path):
        return None
    entries = [name for name in os.listdir(dir_path) if not name.startswith(".")]
    files = []
    for name in entries:
        path = os.path.join(dir_path, name)
        if os.path.isfile(path):
            files.append(path)
    if not files:
        return None
    return random.choice(files)


def clamp(n: int, min_n: int, max_n: int) -> int:
    return max(min_n, min(max_n, n))


def render_text(
    stdscr: curses.window,
    state: TestState,
    cursor_style: str,
    show_whitespace: bool,
    soft_wrap_width: Optional[int],
) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()
    wrap_width = width
    if soft_wrap_width is not None:
        wrap_width = min(width, max(1, soft_wrap_width))

    word_wrap = soft_wrap_width is not None

    def compute_layout() -> tuple[list[tuple[int, int]], int, int]:
        positions: list[tuple[int, int]] = [(0, 0)] * len(state.target)
        line = 0
        col = 0
        idx = 0
        while idx < len(state.target):
            ch = state.target[idx]
            if ch == "\n":
                positions[idx] = (line, col)
                line += 1
                col = 0
                idx += 1
                continue
            if word_wrap:
                if ch == " ":
                    positions[idx] = (line, col)
                    col += 1
                    if col >= wrap_width:
                        line += 1
                        col = 0
                    idx += 1
                    continue
                end = idx
                while end < len(state.target) and state.target[end] not in (" ", "\n"):
                    end += 1
                word_len = end - idx
                if col > 0 and col + word_len > wrap_width:
                    line += 1
                    col = 0
                for pos in range(idx, end):
                    positions[pos] = (line, col)
                    col += 1
                idx = end
                continue
            positions[idx] = (line, col)
            col += 1
            if col >= wrap_width:
                line += 1
                col = 0
            idx += 1
        if col >= wrap_width:
            line += 1
            col = 0
        return positions, line, col

    positions, end_line, end_col = compute_layout()

    def compute_scroll_state() -> tuple[int, int]:
        if state.pos < len(state.target):
            current_line = positions[state.pos][0]
        else:
            current_line = end_line
        max_line = end_line
        for line, _ in positions:
            if line > max_line:
                max_line = line
        total_lines = max_line + 1
        return current_line, total_lines

    current_line, total_lines = compute_scroll_state()
    max_top = max(0, total_lines - height)
    top_line = clamp(current_line - (height // 2), 0, max_top)

    def add_char_at(line: int, col: int, ch: str, attr: int) -> None:
        if top_line <= line < top_line + height:
            try:
                stdscr.addch(line - top_line, col, ch, attr)
            except curses.error:
                pass

    for idx, ch in enumerate(state.target):
        display_ch = ch
        if show_whitespace:
            if ch == " ":
                display_ch = "·"
            elif ch == "\t":
                display_ch = "→"
            elif ch == "\n":
                display_ch = "↵"
        elif ch == "\t":
            display_ch = " "

        line, col = positions[idx]

        if ch == "\n":
            if show_whitespace:
                attr = curses.A_DIM
                if idx < state.pos:
                    typed_ch = state.typed[idx]
                    if typed_ch == ch:
                        attr = curses.A_BOLD
                    else:
                        attr = curses.color_pair(1) | curses.A_BOLD
                elif idx == state.pos:
                    if cursor_style == "highlight":
                        attr = attr | curses.A_REVERSE
                    elif cursor_style == "underline":
                        attr = attr | curses.A_UNDERLINE
                if idx == state.pos and cursor_style == "bar":
                    add_char_at(line, col, "|", curses.A_BOLD)
                else:
                    add_char_at(line, col, display_ch, attr)
            else:
                if idx < state.pos:
                    typed_ch = state.typed[idx]
                    if typed_ch != ch:
                        add_char_at(line, col, "^", curses.color_pair(1) | curses.A_BOLD)
                elif idx == state.pos:
                    if cursor_style == "bar":
                        add_char_at(line, col, "|", curses.A_BOLD)
                    elif cursor_style == "highlight":
                        add_char_at(line, col, " ", curses.A_REVERSE)
                    elif cursor_style == "underline":
                        add_char_at(line, col, " ", curses.A_UNDERLINE)
            continue

        attr = curses.A_DIM
        if idx < state.pos:
            typed_ch = state.typed[idx]
            if typed_ch == ch:
                attr = curses.A_BOLD
            else:
                attr = curses.color_pair(1) | curses.A_BOLD
                if ch.isspace() and not show_whitespace:
                    display_ch = "."
        elif idx == state.pos:
            if cursor_style == "highlight":
                attr = attr | curses.A_REVERSE
            elif cursor_style == "underline":
                attr = attr | curses.A_UNDERLINE

        if idx == state.pos and cursor_style == "bar":
            add_char_at(line, col, "|", curses.A_BOLD)
        else:
            add_char_at(line, col, display_ch, attr)

    if state.pos >= len(state.target) and cursor_style == "bar":
        add_char_at(end_line, end_col, "|", curses.A_BOLD)

    stdscr.refresh()


def run_test(
    stdscr: curses.window,
    target: str,
    cursor_style: str,
    show_whitespace: bool,
    soft_wrap_width: Optional[int],
) -> TestResult:
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_RED, -1)

    typed: List[Optional[str]] = [None] * len(target)
    state = TestState(
        target=target,
        typed=typed,
        pos=0,
        start_time=None,
        end_time=None,
        mistakes=0,
        keypress_times=[],
    )

    render_text(stdscr, state, cursor_style, show_whitespace, soft_wrap_width)

    while state.pos < len(state.target):
        ch = stdscr.get_wch()
        if ch in ("\x1b",):
            raise KeyboardInterrupt
        if ch in ("\b", "\x7f", curses.KEY_BACKSPACE):
            if state.pos > 0:
                state.pos -= 1
                state.typed[state.pos] = None
            render_text(stdscr, state, cursor_style, show_whitespace, soft_wrap_width)
            continue
        if ch in ("\t", getattr(curses, "KEY_TAB", None)):
            if state.start_time is None:
                state.start_time = time.monotonic()
            mismatch = False
            for _ in range(4):
                if state.pos >= len(state.target):
                    break
                if state.target[state.pos] == " ":
                    state.typed[state.pos] = " "
                else:
                    state.typed[state.pos] = "\t"
                    mismatch = True
                state.pos += 1
            if mismatch:
                state.mistakes += 1
            state.keypress_times.append(time.monotonic())
            render_text(stdscr, state, cursor_style, show_whitespace, soft_wrap_width)
            continue
        if ch in ("\n", "\r", curses.KEY_ENTER):
            ch = "\n"
        if isinstance(ch, str) and ch.isprintable() or ch == "\n":
            idx = state.pos
            if idx >= len(state.target):
                break
            if state.start_time is None:
                state.start_time = time.monotonic()
            state.typed[idx] = ch
            if ch != state.target[idx]:
                state.mistakes += 1
            state.pos += 1
            state.keypress_times.append(time.monotonic())
            render_text(stdscr, state, cursor_style, show_whitespace, soft_wrap_width)

    state.end_time = time.monotonic() if state.start_time is not None else None
    elapsed = 0.0
    if state.start_time is not None and state.end_time is not None:
        elapsed = max(0.0001, state.end_time - state.start_time)
    correct_chars = sum(
        1
        for i, ch in enumerate(state.typed)
        if ch == state.target[i] and state.target[i] not in (" ", "\n", "\r", "\t")
    )
    total_chars = len(state.target)
    stability = compute_stability(state.keypress_times)

    return TestResult(
        elapsed=elapsed,
        correct_chars=correct_chars,
        total_chars=total_chars,
        mistakes=state.mistakes,
        stability=stability,
    )


def compute_stability(times: List[float]) -> float:
    if len(times) < 3:
        return 100.0
    intervals = [b - a for a, b in zip(times, times[1:]) if b - a > 0]
    if len(intervals) < 2:
        return 100.0
    total_time = sum(intervals)
    if total_time <= 0:
        return 0.0
    ideal = total_time / len(intervals)
    if ideal <= 0:
        return 0.0
    mean_abs_dev = sum(abs(x - ideal) for x in intervals) / len(intervals)
    score = 100.0 * (1.0 - (mean_abs_dev / ideal))
    return max(0.0, min(100.0, score))


def format_results(result: TestResult) -> str:
    minutes = result.elapsed / 60.0
    wpm = (result.correct_chars / 5.0) / minutes if minutes > 0 else 0.0
    cpm = result.correct_chars / minutes if minutes > 0 else 0.0
    lines = [
        "Results",
        f"Elapsed: {result.elapsed:.2f}s",
        f"WPM: {wpm:.2f}",
        f"CPM: {cpm:.2f}",
        f"Mistakes: {result.mistakes}",
        f"Stability: {result.stability:.2f}",
    ]
    return "\n".join(lines)


def main() -> int:
    cfg = parse_args()
    try:
        target = build_test_text(cfg)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    soft_wrap_width = None if cfg.text_file else cfg.line_width

    try:
        result = curses.wrapper(run_test, target, cfg.cursor_style, cfg.show_whitespace, soft_wrap_width)
    except KeyboardInterrupt:
        return 1

    print(format_results(result))
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
