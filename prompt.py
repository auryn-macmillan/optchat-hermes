"""Verbatim prompt material from the UniiChat design (optchat.md).

The compaction tasks, the ruler, the too-long correction, and the system
prompt section are kept byte-faithful to the spec where the spec says
"verbatim"; the system prompt section is adapted (agent name, Hermes tools,
no device paragraph) as the spec itself instructs ("Rename Unii to your
agent's name, and drop the paragraph on computers if your agent has no
device tools").
"""

NODE_MAX = 512
RULER = "-" * 512

# Task for node(0, i): compress one message into one line.
TASK_SINGLE = (
    "Compaction: compress message {id} into one line of at most 512 bytes\n"
    "(about 70 words), the length of this ruler:\n"
    "{ruler}\n"
    "<input>\n"
    "{kind}: {text}\n"
    "</input>"
)

# Task for node(l, i), l > 0: merge two adjacent lines into one.
# {chat_note} is emitted only when a compaction view precedes the task.
TASK_MERGE = (
    "Compaction: merge lines {a} and {b}, adjacent, into one line of at most\n"
    "512 bytes (about 70 words), the length of this ruler:\n"
    "{ruler}\n"
    "<chat> may hold their messages, {id} to {end}, in more detail: take details\n"
    "of them from there too.\n"
    "<input>\n"
    "{line_a}\n"
    "{line_b}\n"
    "</input>"
)

# Sent in the same conversation when a reply is over the limit.
TASK_TOO_LONG = (
    "Too long: your line is {n} bytes, over the 512-byte limit. Write\n"
    "the whole line again for the same <input>, cutting just enough of the\n"
    "least valuable items to fit before this cut:\n"
    "{cut}| ← LIMIT"
)

# System prompt section (rendered only while context.engine == optchat).
# Adapted from the spec's prompt: Unii -> Hermes, device paragraph dropped,
# subagent paragraph dropped (Hermes subagents keep their own transcript).
PROTOCOL = """\
# OptChat memory (one chat that never ends)

Your conversation history is an OptChat view: the whole chat between you and
the user, oldest first, inside <chat> tags, as one-line summaries:

  id+n|text   the n messages from id on, summarized (newlines as spaces)

Each message has a kind: user: the user's words; unii: your replies;
tool: your tool calls; echo: tool results. The summaries form a binary tree:
each message is compressed into a line (a short message is its own line),
then adjacent lines are merged in pairs, again and again. Recent lines cover
one message each; older lines cover more. Messages after the view are the
live transcript, verbatim.

The view is your memory, and its latest word on a thing is the truth.
Whenever you need any information, first find its latest mention in the view
and zoom until you have it whole, before any other source, and before you
act, guess or ask. Never grep or search memories manually; zoom is your only
allowed mechanism to navigate the tree. Summaries keep little of tool output,
so say in your reply what you learned that will matter later.

Tools:
- zoom(id, n) opens line id+n of the view into the two lines of n/2 under it;
  zoom(id, 1) gives message id whole.
- date(id) gives the date and time of message id.
"""

ZOOM_DESC = (
    "Open the line id+n of the view into the two lines of n/2 under it; "
    "n = 1 gives the message whole. n is a power of 2 and id a multiple of n. "
    "A long message comes in pages."
)

DATE_DESC = "The date and time of message id."