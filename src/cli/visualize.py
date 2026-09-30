"""ASCII art visualization of climbing routes on the Kilter Board."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

from src.utils.constants import BOARD_WIDTH_CM, BOARD_HEIGHT_CM

console = Console()

# Symbols for different hold types
HOLD_SYMBOLS = {
    "start": "S",
    "finish": "F",
    "middle": "●",
    "foot": "○",
}

# Rich colors for hold types
HOLD_COLORS = {
    "start": "bold green",
    "finish": "bold magenta",
    "middle": "cyan",
    "foot": "yellow",
}

# Board dimensions in grid cells
GRID_WIDTH = 24
GRID_HEIGHT = 16


def visualize_climb(holds: list[dict], grade: str) -> None:
    """Visualize a climb on a grid mirroring the Kilter Board.

    The board is shown with:
    - Start holds as green S
    - Finish holds as magenta F
    - Middle holds as cyan dots
    - Foot holds as yellow circles
    - Numbers showing the order of holds

    Args:
        holds: List of hold dicts with x, y, hold_type
        grade: Target grade for display
    """
    # Create empty grid
    grid = [["." for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

    # Map holds to grid positions
    for i, hold in enumerate(holds):
        col = int((hold["x"] / BOARD_WIDTH_CM) * (GRID_WIDTH - 1))
        row = int((1 - hold["y"] / BOARD_HEIGHT_CM) * (GRID_HEIGHT - 1))
        col = max(0, min(col, GRID_WIDTH - 1))
        row = max(0, min(row, GRID_HEIGHT - 1))

        symbol = HOLD_SYMBOLS.get(hold["hold_type"], "?")
        grid[row][col] = (symbol, i + 1, hold["hold_type"])

    # Print header
    console.print(f"\n[bold cyan]Kilter Board — {grade}[/bold cyan]")
    console.print(f"[dim]{BOARD_WIDTH_CM}cm × {BOARD_HEIGHT_CM}cm @ 40°[/dim]\n")

    # Print column numbers
    col_header = Text()
    col_header.append("    ", style="dim")
    for col in range(GRID_WIDTH):
        col_header.append(f"{col:2d} ", style="dim")
    console.print(col_header)

    # Print grid with row labels
    for row_idx, row in enumerate(grid):
        y_pct = 1 - (row_idx / (GRID_HEIGHT - 1))
        y_cm = y_pct * BOARD_HEIGHT_CM

        line = Text()
        line.append(f"{y_cm:3.0f} ", style="dim")

        for cell in row:
            if cell == ".":
                line.append(" . ", style="dim")
            else:
                symbol, order, hold_type = cell
                color = HOLD_COLORS.get(hold_type, "white")
                # Show order number for first 9 holds, then just symbol
                if order <= 9:
                    line.append(f" {symbol}{order}", style=color)
                else:
                    line.append(f" {symbol} ", style=color)

        console.print(line)

    # Print legend
    console.print(f"\n[bold]Legend:[/bold]")
    console.print(f"  [bold green]S[/bold green] = Start  [bold magenta]F[/bold magenta] = Finish  [cyan]●[/cyan] = Hand  [yellow]○[/yellow] = Foot")
    console.print(f"  [dim]Numbers show hold order (1-9)[/dim]")
