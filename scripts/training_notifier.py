"""Push a training run's milestones to a phone through ntfy.sh.

post_training_watchdog.sh runs this when NTFY_TOPIC is set, in two modes:

  training_notifier.py <training-pid>   a message on start, so a wrong topic
                                        shows now and not hours later, and one
                                        at the halfway step; exits with the
                                        training process
  training_notifier.py --report-exit    how training ended - finished, early
                                        stop or crash - with its losses

The watchdog sends the archive and pod-stop messages itself: only it knows
them, and the stop message has to leave before the pod stops.

Everything reported here comes from train.log. The trainer's metric dicts
reach that file only because training is launched with `python -u`.

The topic name is the only access control an ntfy.sh topic has: anyone who
knows it can read it. So it comes from the environment, never from a tracked
file, and messages carry steps and losses, never usernames.

Stdlib only, so it runs on the pod's system python.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import sys
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TRAIN_LOG = Path(os.environ.get("LITIGPT_TRAIN_LOG", "/workspace/train.log"))
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
if not NTFY_SERVER.startswith(("https://", "http://")):
    sys.exit(f"NTFY_SERVER must be an http(s) URL, not {NTFY_SERVER!r}")
POLL_SECONDS = 60
USAGE = "usage: NTFY_TOPIC=<topic> python training_notifier.py {<training-pid> | --report-exit}"

try:
    LOCAL_TZ = ZoneInfo(os.environ.get("NTFY_TZ", "Europe/Rome"))
except ZoneInfoNotFoundError:
    LOCAL_TZ = UTC

# The trainer's bar is unlabelled ("  10%|█   | 410/3908 [36:28<..."); dataset
# and shard-loading bars carry a "Map:" style label and are skipped. Eval bars
# are unlabelled too, so the training bar is told apart by its total.
BAR = re.compile(r"(?:^|[\r\n])[ \t]*\d+%\|[^|\r\n]*\| (\d+)/(\d+) \[(\d+(?::\d+)+)<")
EVAL = re.compile(r"\{'eval_loss'[^}]*\}")
SUMMARY = re.compile(r"\{'train_runtime'[^}]*\}")

log = logging.getLogger("training_notifier")


def notify(title: str, message: str, priority: str = "default",
           tags: str = "", attempts: int = 3) -> None:
    """Log the message and push it to ntfy.sh, retrying network failures."""
    log.info("%s: %s", title, message)
    headers = {"Title": title, "Priority": priority}
    if tags:
        headers["Tags"] = tags
    request = urllib.request.Request(  # noqa: S310  http(s) only, checked at import
        f"{NTFY_SERVER}/{os.environ['NTFY_TOPIC']}",
        data=message.encode("utf-8"),
        headers=headers,
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=10):  # noqa: S310  http(s) only, checked at import
                return
        except OSError as exc:
            log.warning("notification failed (attempt %d/%d): %s", attempt, attempts, exc)
            time.sleep(attempt)


def alive(pid: int) -> bool:
    """True while the process exists and has not exited (zombies count as exited)."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return False
    return not re.search(r"^State:\s+Z", status, re.MULTILINE)


def seconds(clock: str) -> int:
    """tqdm's "36:28" or "1:36:28" in seconds."""
    total = 0
    for part in clock.split(":"):
        total = total * 60 + int(part)
    return total


def duration(secs: float) -> str:
    hours, rest = divmod(int(secs), 3600)
    return f"{hours}h{rest // 60:02d}m"


def local_time(at: datetime) -> str:
    return at.astimezone(LOCAL_TZ).strftime("%H:%M")


def parse_dicts(pattern: re.Pattern[str], text: str) -> list[dict]:
    found = []
    for match in pattern.finditer(text):
        try:
            found.append(ast.literal_eval(match.group()))
        except (ValueError, SyntaxError):
            continue  # a nan loss prints as a bare name, which is not a literal
    return found


class Run:
    """What train.log says about the run so far."""

    def __init__(self, text: str):
        self.text = text
        bars = [(int(step), int(total), seconds(clock))
                for step, total, clock in BAR.findall(text)]
        # The trainer's bar is the first unlabelled one: evaluation starts only
        # after eval_steps of training.
        self.total = bars[0][1] if bars else 0
        training = [bar for bar in bars if bar[1] == self.total]
        self.step, _, self.elapsed = training[-1] if training else (0, 0, 0)
        self.evals = parse_dicts(EVAL, text)
        summaries = parse_dicts(SUMMARY, text)
        self.summary = summaries[-1] if summaries else None

    @classmethod
    def read(cls) -> Run:
        if not TRAIN_LOG.exists():
            return cls("")
        return cls(TRAIN_LOG.read_text(encoding="utf-8", errors="replace"))

    def progress(self) -> str:
        if not self.step:
            return "waiting for the first training step"
        line = f"step {self.step}/{self.total} ({100 * self.step // self.total}%)"
        per_step = self.elapsed / self.step
        left = per_step * (self.total - self.step)
        end = datetime.now(UTC) + timedelta(seconds=left)
        return (f"{line}, {duration(self.elapsed)} in, {per_step:.1f} s/step; "
                f"~{duration(left)} left, done ~{local_time(end)}")

    def eval_line(self) -> str:
        if not self.evals:
            return "no eval yet"
        last = self.evals[-1]
        best = min(self.evals, key=lambda e: e["eval_loss"])
        line = f"eval_loss {last['eval_loss']:.4f} at epoch {last.get('epoch', '?')}"
        if best is not last:
            line += f" (best {best['eval_loss']:.4f} at epoch {best.get('epoch', '?')})"
        return line

    def last_line(self) -> str:
        lines = [line.strip() for line in re.split(r"[\r\n]", self.text) if line.strip()]
        return lines[-1][:300] if lines else "(empty log)"


def report_exit(run: Run) -> None:
    """How training ended, sent once by the watchdog after the process exits."""
    if not run.text:
        notify("litiGPT: training ended", f"No log at {TRAIN_LOG}, so how is unknown.",
               priority="high", tags="warning")
        return
    if run.summary is None:
        where = f"at step {run.step}/{run.total}" if run.step else "before the first step"
        notify("litiGPT: training DIED",
               f"Exited {where} without finishing.\n"
               f"{run.last_line()}\n"
               "Archiving the latest checkpoint next.",
               priority="urgent", tags="rotating_light")
        return
    early = run.step < run.total
    stopped = f" (early stop at {run.step}/{run.total})" if early else ""
    message = (f"Finished in {duration(run.summary['train_runtime'])}{stopped}. "
               f"{run.eval_line()}; train_loss {run.summary['train_loss']:.4f}.\n"
               "Archiving next.")
    if "Traceback (most recent call last)" in run.text:
        notify("litiGPT: trained, then an error",
               f"{message}\nBut the pipeline raised afterwards:\n{run.last_line()}",
               priority="high", tags="warning")
    else:
        notify("litiGPT: done (early stop)" if early else "litiGPT: 100%", message,
               priority="high", tags="white_check_mark")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) != 2 or "NTFY_TOPIC" not in os.environ:
        sys.exit(USAGE)
    if sys.argv[1] == "--report-exit":
        report_exit(Run.read())
        return
    pid = int(sys.argv[1])
    if not alive(pid):
        sys.exit(f"no running process {pid}")

    run = Run.read()
    halfway_sent = run.total > 0 and 2 * run.step >= run.total
    notify("litiGPT: notifier armed",
           f"{run.progress()}. {run.eval_line()}.\n"
           f"Next: {'the end of the run' if halfway_sent else 'halfway'}.",
           tags="hourglass_flowing_sand")

    while alive(pid):
        if not halfway_sent:
            run = Run.read()
            if run.total and 2 * run.step >= run.total:
                notify("litiGPT: 50%", f"{run.progress()}. {run.eval_line()}.",
                       tags="hourglass")
                halfway_sent = True
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
