"""ASCII art visualization of climbing routes."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

from src.utils.constants import BOARD_WIDTH_CM, BOARD_HEIGHT_CM

console = Console()

# ASCII characters for different hold types
HOLD_SYMBOLS = {
    "start": "S",
    "finish": "F",
    "jug": "J",
    "crimp": "c",
    "sloper": "s",
    "pinch": "P",
    "pocket": "p",
    "volume": "V",
    "foothold": "f",
}

# Rich colors for LED colors
LED_STYLES = {
    "red": "red",
    "green": "green",
    "blue": "blue",
    "yellow": "yellow",
    "purple": "magenta",
    "off": "white",
}


def visualize_climb(holds: list[dict], grade: str, width: int = 40, height: int = 20) -> None:
    """Visualize a climb as ASCII art on a grid.

    Args:
        holds: List of hold dicts with x, y, hold_type, led_color
        grade: Target grade for display
        width: Grid width in characters
        height: Grid height in characters
    """
    # Create empty grid
    grid = [["." for _ in range(width)] for _ in range(height)]

    # Place holds on grid
    for hold in holds:
        col = int((hold["x"] / BOARD_WIDTH_CM) * (width - 1))
        row = int((1 - hold["y"] / BOARD_HEIGHT_CM) * (height - 1))  # Invert y (top = high)
        col = max(0, min(col, width - 1))
        row = max(0, min(row, height - 1))

        symbol = HOLD_SYMBOLS.get(hold["hold_type"], "?")
        grid[row][col] = symbol

    # Print header
    console.print(f"\n[bold]Kilter Board — {grade}[/bold]")
    console.print(f"[dim]{BOARD_WIDTH_CM}cm × {BOARD_HEIGHT_CM}cm @ 40°[/dim]\n")

    # Print grid with row labels
    for row_idx, row in enumerate(grid):
        y_pct = 1 - (row_idx / (height - 1))
        y_cm = y_pct * BOARD_HEIGHT_CM

        line = Text()
        for cell in row:
            if cell == ".":
                line.append("· ", style="dim")
            elif cell == "S":
                line.append(cell + " ", style="bold green")
            elif cell == "F":
                line.append(cell + " ", style="bold magenta")
            elif cell == "f":
                line.append(cell + " ", style="yellow")
            else:
                line.append(cell + " ", style="cyan")

        console.print(f"  {line}  [dim]{y_cm:4.0f}cm[/dim]")

    # Print legend
    console.print(f"\n[bold]Legend:[/bold]")
    console.print("  [green]S[/green] = Start  [magenta]F[/magenta] = Finish  [cyan]J[/cyan] = Jug  [cyan]c[/cyan] = Crimp")
    console.print("  [cyan]s[/cyan] = Sloper  [cyan]P[/cyan] = Pinch  [cyan]p[/cyan] = Pocket  [yellow]f[/yellow] = Foothold")
