"""Raw terminal, so a card can be clicked as well as typed.

Terminals will report mouse clicks if you ask them to, which most people don't
know. Turn on 1000h and they send a sequence on every press; turn on 1006h as
well and the coordinates come back in decimal rather than packed into bytes,
which matters the moment the window is wider than 223 columns. Then it's just
arithmetic to get from a column and row back to which card was under the mouse.

Falls back to typing on its own, since a terminal that ignores the request just
never sends any of these sequences and everything else carries on working.
"""

import re
import sys
import termios
import tty

MOUSE_ON = "\x1b[?1000h\x1b[?1006h"
MOUSE_OFF = "\x1b[?1006l\x1b[?1000l"
CLEAR = "\x1b[2J\x1b[H"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"

# ESC [ < button ; col ; row M   (M is press, m is release)
SGR = re.compile(r"<(\d+);(\d+);(\d+)([Mm])")


class Terminal:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        try:
            self.saved = termios.tcgetattr(self.fd)
            tty.setraw(self.fd)
            self.raw = True
        except termios.error:
            self.saved, self.raw = None, False  # piped in, not a real terminal
        self.write(MOUSE_ON)
        return self

    def __exit__(self, *_):
        self.write(MOUSE_OFF + SHOW_CURSOR)
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def write(self, s):
        # raw mode means a bare newline only drops a line, it doesn't return
        sys.stdout.write(s.replace("\n", "\r\n") if self.raw else s)
        sys.stdout.flush()

    def _escape(self):
        if sys.stdin.read(1) != "[":
            return ""
        out = ""
        while True:
            c = sys.stdin.read(1)
            if not c:
                return out
            out += c
            if c.isalpha():
                return out

    def read(self, prompt=""):
        """Blocks. Gives back ('click', (col, row)) or ('text', typed)."""
        self.write(prompt)
        buf = ""
        while True:
            ch = sys.stdin.read(1)
            if not ch:
                return "text", "q"
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch == "\x1b":
                m = SGR.match(self._escape())
                if m and m.group(4) == "M" and m.group(1) == "0":
                    return "click", (int(m.group(2)), int(m.group(3)))
                continue
            if ch in "\r\n":
                return "text", buf
            if ch in ("\x7f", "\b"):
                if buf:
                    buf = buf[:-1]
                    self.write("\b \b")
                continue
            if ch.isprintable():
                buf += ch
                self.write(ch)
