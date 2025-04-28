# Grid

## Grid principles

grid-template-columns and grid-template-rows define the size of each of the columns and rows in the grid.

fr represents each column's (or row's) fraction unit. It works like flex-grow.

gap defines the amount of space to add to the gutter between each grid cell, just like with flexbox.

![The parts of a grid](../../assets/image/grid_parts.png)

The parts of a grid:

- Grid line: Can be vertical or horizontal and lie on either side of a row or column.
- Grid track: The space between two adjacent grid lines.
- Grid cell: A single space on the grid.
- Grid area: A rectangular area on the grid made up of one or more grid cells.

![Numbering grid lines](../../assets/image/grid_lines_number.png)

Grid lines are numbered beginning with 1 on the top left. Negative numbers refer to the position from the bottom right.

You can use the grid numbers to indicate where to place each grid item using the grid-column and grid-row properties.

span property indicate how much grid track the item to span. The item will be placed automatically using the grid item placement algorithm.

The flexbox layout and grid layout have two important distinctions:

1. Flexbox is basically one dimensional, whereas grid is two dimensional.
2. Flexbox works from the content out, whereas grid works from the layout in.

While the content of each grid item can influence the size of its grid track, this will affect the size of the entire track.

In practice, This often mean grid makes the most sense for a high-level layout of the page, and flexbox makes more sense for certain elements within each grid area.

Naming grid lines and naming grid area are two alternative syntaxes for laying out grid items.

## Explicit and implicit grid

If a grid item is placed outside the declared grid tracks, implicit tracks will be added to the grid until it can contain the item.

![implicit grid](../../assets/image/implicit_grid.png)

By default, implicit grid tracks will have a size of auto, meaning they’ll grow to the size necessary to contain the grid item contents. The properties grid-auto-columns and grid-auto-rows can be applied to the grid container to specify a different size for all implicit grid tracks (for example, grid-auto-columns: 1fr).

`minmax()` control the minimum and maximum values of a grid track.

`auto-fill` will place as many tracks onto the grid as it can fit.

`auto-fit` can cause the non-empty tracks to stretch to fill the available space.

`grid-auto-flow` can be used to manipulate the behavior of the placement algorithm.

## SubGrid

With subgrid, we can place a grid within a grid and then position the inner grid's items on the grid lines of the parent grid.

## Alignment

CSS provides three justify properties: justify-content, justify-items, justify-self. These properties control horizontal placement.

There are also three alignment properties: align-content, align-items, align-self. These control vertical placement.

![grid alignment](../../assets/image/grid_alignment.png)
