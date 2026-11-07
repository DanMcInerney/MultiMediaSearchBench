"""What an mm exec's agent is shown of its question by mm itself, and what of mm its isolation words catch. Every result of mm's tools
names the exec's project, and until 2026-11-07 the project was the question's id: A1's `leave-0829` and `return-1730` held their
answers, and every `absent-` question said there was none. And the isolation word `.multimem` matched `.multimem-project`, the marker
of the exec's own working folder, so a listing of that folder was an escape."""

from __future__ import annotations

import pytest

pytest.importorskip("multimem")

from mmmb import systems  # noqa: E402
from mmmb.core.agents.common import reached  # noqa: E402
from mmmb.systems.mm.home import PROJECT_PREFIX, exec_project  # noqa: E402


def test_an_execs_project_says_nothing_of_its_question():
    for qid in ("leave-0829", "return-1730", "vacuum_on-2204", "absent-return-2030", "absent-days-dog-2013-12", "first-deer"):
        project = exec_project(qid)
        digits = project.removeprefix(PROJECT_PREFIX)
        assert project.startswith(PROJECT_PREFIX) and len(digits) == 12 and set(digits) <= set("0123456789abcdef")
        assert qid not in project and "absent" not in project and qid.split("-")[-1] not in digits
    assert exec_project("leave-0829") == exec_project("leave-0829") != exec_project("leave-0830"), "one project per question"


def test_mms_isolation_words_catch_a_home_and_pass_over_the_marker_of_the_working_folder():
    deny = tuple(w.lower() for w in systems.isolation().deny_words)
    for reach in ("Get-ChildItem C:\\Users\\x\\.multimem", "Get-ChildItem C:\\Users\\x\\.multimem -Force",
                  'Get-Content "C:/Users/x/.multimem/index.sqlite"', "cat '/home/x/.multimem'", "ls ~/.multimem/"):
        assert reached(reach, deny) and reached(reach, deny, output=True), reach
    listing = "Name              Mode   Length\n.multimem-project -a---- 24"  # what Codex's listing printed, 2026-11-07
    assert reached(listing, deny, output=True) is None and reached("Get-Content .multimem-project", deny) is None
