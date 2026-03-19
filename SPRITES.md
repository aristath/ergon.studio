# Braille Sprite Library

Sprites for the ergon.studio TUI. Each sprite is 2 braille characters wide (4x4 dot grid per pair). Color and opacity convey state (bright = active, dim = idle, pulsing = working).

## Encoding

Each braille character is a 2-wide x 4-tall dot grid (Unicode U+2800–U+28FF). Two characters side by side give a 4x4 pixel canvas. The dot-to-bit mapping per character:

```
Row 0: 0x01  0x08
Row 1: 0x02  0x10
Row 2: 0x04  0x20
Row 3: 0x40  0x80
```

Pixel grids below read left-to-right, top-to-bottom: `█` = on, `·` = off.

---

## Agent Roles

| Sprite | Name | Color | Description | Pixel Grid |
|--------|------|-------|-------------|------------|
| `⢺⡗` | orchestrator | `bright_cyan` | Crown/pillar — leads and delegates | `·██·  ████  ·██·  ·██·` |
| `⣎⣱` | architect | `bright_blue` | Open frame — designs structure | `·██·  █··█  █··█  ████` |
| `⡢⢔` | coder | `bright_green` | Angle brackets — writes code | `····  █··█  ·██·  █··█` |
| `⠺⢗` | reviewer | `bright_yellow` | Magnifying glass — inspects | `·██·  ████  ·██·  ···█` |
| `⢹⡀` | fixer | `rgb(255,165,0)` | Wrench — fixes what's broken | `██··  ·█··  ·█··  ·██·` |
| `⣑⣄` | researcher | `bright_magenta` | Antenna/radar — seeks info | `█···  ·█··  ··█·  ████` |
| `⢄⠊` | tester | `rgb(0,255,128)` | Checkmark — validates | `···█  ··█·  █···  ·█··` |
| `⣏⡇` | documenter | `grey70` | Page with lines — writes docs | `███·  █·█·  █·█·  ███·` |
| `⡱⢎` | brainstormer | `bright_red` | Spark/X — divergent thinking | `█··█  ·██·  ·██·  █··█` |
| `⡠⠃` | designer | `rgb(255,100,255)` | Brush stroke — shapes UX | `··█·  ··█·  ·█··  █···` |
| `⢘⡃` | user | `bright_white` | Person — the engineering manager | `·██·  ·██·  ····  ·██·` |

## Task States

| Sprite | Name | Color | Description | Pixel Grid |
|--------|------|-------|-------------|------------|
| `⠘⠃` | created | `grey50` | Small dot — just born | `·██·  ·██·  ····  ····` |
| `⠺⠗` | planned | `bright_blue` | Diamond — has a plan | `·██·  ████  ·██·  ····` |
| `⢺⡗` | assigned | `bright_cyan` | Diamond+stem — assigned to someone | `·██·  ████  ·██·  ·██·` |
| `⢰⠇` | in_progress | `bright_green` | Lightning — work happening | `··█·  ·██·  ·██·  ·█··` |
| `⣏⣹` | blocked | `bright_red` | Solid box — stuck | `████  █··█  █··█  ████` |
| `⠺⢗` | in_review | `bright_yellow` | Lens — under review | `·██·  ████  ·██·  ···█` |
| `⡱⢎` | needs_fix | `rgb(255,165,0)` | X mark — needs repair | `█··█  ·██·  ·██·  █··█` |
| `⢎⡱` | awaiting_approval | `yellow` | Hollow circle — waiting on user | `·██·  █··█  █··█  ·██·` |
| `⢄⠊` | approved | `bright_green` | Checkmark — approved | `···█  ··█·  █···  ·█··` |
| `⡱⢎` | rejected | `bright_red` | X — rejected | `█··█  ·██·  ·██·  █··█` |
| `⠤⠚` | completed | `bright_green` | Bold check — done | `···█  ··██  ██··  ····` |
| `⠒⠒` | canceled | `grey50` | Strikethrough — canceled | `····  ████  ····  ····` |

## Workflow States

| Sprite | Name | Color | Description | Pixel Grid |
|--------|------|-------|-------------|------------|
| `⡏⠁` | draft | `grey50` | L-corner — not started | `███·  █···  █···  █···` |
| `⢸⠦` | running | `bright_green` | Play arrow — executing | `·█··  ·██·  ·███  ·█··` |
| `⣹⣏` | waiting | `yellow` | Hourglass — paused/waiting | `████  ·██·  ·██·  ████` |
| `⡱⢎` | failed | `bright_red` | X — failed | `█··█  ·██·  ·██·  █··█` |
| `⣏⣹` | completed_wf | `bright_green` | Box — finished | `████  █··█  █··█  ████` |
| `⢾⡷` | aborted | `grey50` | Stop — aborted | `·██·  ████  ████  ·██·` |

## Thread Types

| Sprite | Name | Color | Description | Pixel Grid |
|--------|------|-------|-------------|------------|
| `⠿⠟` | main_chat | `bright_white` | Chat bubble — main thread | `████  ████  ███·  ····` |
| `⠺⡗` | agent_direct | `bright_cyan` | Speech pointed — direct message | `·██·  ████  ·██·  ··█·` |
| `⡇⡇` | group_workroom | `bright_blue` | Multiple lines — group | `█·█·  █·█·  █·█·  █·█·` |
| `⠺⢗` | review_thread | `bright_yellow` | Lens — review | `·██·  ████  ·██·  ···█` |
| `⢎⡱` | approval_thread | `yellow` | Ring — needs decision | `·██·  █··█  █··█  ·██·` |
| `⢕⢕` | system_thread | `grey50` | Grid — system/internal | `█·█·  ·█·█  █·█·  ·█·█` |

## Risk Levels

| Sprite | Name | Color | Description | Pixel Grid |
|--------|------|-------|-------------|------------|
| `⠰⠆` | safe | `bright_green` | Small — low risk | `····  ·██·  ·██·  ····` |
| `⢾⡷` | moderate | `yellow` | Medium — moderate risk | `·██·  ████  ████  ·██·` |
| `⣿⣿` | high_risk | `bright_red` | Full block — high risk | `████  ████  ████  ████` |

## Arrows and Flow

| Sprite | Name | Color | Description | Pixel Grid |
|--------|------|-------|-------------|------------|
| `⠀⡷` | arrow_right | `bright_white` | Right arrow | `··█·  ··██  ··██  ··█·` |
| `⢾⠀` | arrow_left | `bright_white` | Left arrow | `·█··  ██··  ██··  ·█··` |
| `⢼⡧` | arrow_down | `bright_white` | Down arrow | `·██·  ·██·  ████  ·██·` |
| `⢺⡗` | arrow_up | `bright_white` | Up arrow | `·██·  ████  ·██·  ·██·` |
| `⠶⠶` | flow_right | `grey50` | Horizontal flow | `····  ████  ████  ····` |
| `⡼⢧` | fork | `grey50` | Fork — branching | `·██·  ·██·  ████  █··█` |
| `⢳⡞` | merge | `grey50` | Merge — converging | `█··█  ████  ·██·  ·██·` |

## General Icons

| Sprite | Name | Color | Description | Pixel Grid |
|--------|------|-------|-------------|------------|
| `⠳⠞` | heart | `bright_red` | Heart | `█··█  ████  ·██·  ····` |
| `⢾⡧` | fire | `rgb(255,100,0)` | Fire — hot/urgent | `·█··  ███·  ████  ·██·` |
| `⣼⣧` | lock | `yellow` | Lock — secured | `·██·  ·██·  ████  ████` |
| `⣬⣧` | unlock | `bright_green` | Unlock — open | `·██·  ··█·  ████  ████` |
| `⢻⣀` | key | `yellow` | Key | `██··  ██··  ·█··  ·███` |
| `⢾⡷` | gear | `grey70` | Gear — settings | `·██·  ████  ████  ·██·` |
| `⣟⣻` | mail | `bright_blue` | Envelope | `████  ████  █··█  ████` |
| `⠾⠷` | bell | `yellow` | Bell — notification | `·██·  ████  ████  ····` |
| `⢘⡃` | pin | `bright_red` | Pin — pinned | `·██·  ·██·  ····  ·██·` |
| `⢞⡇` | clock | `grey70` | Clock face | `·██·  ███·  █·█·  ·██·` |
| `⣿⣶` | folder | `yellow` | Folder | `██··  ████  ████  ████` |
| `⢸⡏` | file | `grey70` | File/document | `·███  ·██·  ·██·  ·██·` |
| `⣯⣉` | terminal | `bright_green` | Terminal prompt | `████  █···  ██··  ████` |
| `⡵⢮` | bug | `bright_red` | Bug | `█··█  ·██·  ████  █··█` |
| `⢾⡷` | shield | `bright_cyan` | Shield — protection | `·██·  ████  ████  ·██·` |
| `⢴⡦` | eye | `bright_cyan` | Eye — watching | `····  ·██·  ████  ·██·` |
| `⢊⡡` | link | `bright_blue` | Chain link | `·██·  █···  ···█  ·██·` |
| `⢨⡅` | info | `bright_blue` | Info — i symbol | `·██·  ····  ·██·  ·██·` |
| `⣞⣳` | warning | `yellow` | Warning triangle | `·██·  ████  █··█  ████` |
| `⣏⣹` | error | `bright_red` | Error box | `████  █··█  █··█  ████` |
| `⠤⠚` | success | `bright_green` | Success check | `···█  ··██  ██··  ····` |
| `⢰⠃` | lightning | `bright_yellow` | Lightning bolt | `··█·  ·██·  ·█··  ·█··` |
| `⠔⠔` | wave | `bright_cyan` | Wave — activity | `····  ·█·█  █·█·  ····` |
| `⢤⠦` | pulse | `bright_green` | Pulse — heartbeat | `····  ··█·  ████  ·█··` |
| `⠑⣤` | satellite | `bright_magenta` | Satellite — remote | `█···  ·█··  ··██  ··██` |
| `⣹⣏` | database | `bright_blue` | Database — storage | `████  ·██·  ·██·  ████` |
| `⡱⢎` | network | `bright_cyan` | Network — connections | `█··█  ·██·  ·██·  █··█` |
| `⡗⡗` | memory_chip | `bright_magenta` | Memory — RAM | `█·█·  ████  █·█·  █·█·` |

## Status Bar Layout

The status bar shows all agents in a single line. Color conveys state:

- **Bright color** (per-agent) = actively working
- **Dim grey** (`grey35`) = idle
- **Pulsing animation** = processing
- **Yellow** = blocked / waiting on input

Example (orchestrator active, coder working, reviewer waiting, rest idle):

```
⢺⡗ ⣎⣱ ⡢⢔ ⠺⢗ ⢹⡀ ⣑⣄ ⢄⠊ ⣏⡇ ⡱⢎ ⡠⠃ ⢘⡃
```

Total width: 33 cells (11 agents × 3 cells each), fits comfortably in 80+ column terminals with room for workflow context.
