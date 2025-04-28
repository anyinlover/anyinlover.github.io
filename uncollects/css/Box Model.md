# Box model

## Normal document flow

Normal document flow refers to the default layout behavior. Inline elements flow along with the text of the page, block elements fall on individual lines.

block-level elements fill the width of their container by default.

The width of a parent element determines the width of its children, the heights of child elements determine the height of the parent.

To begin laying out a page, it is best to do so from the outside in.

double-container pattern: place the content inside two nested containers and set the margins on the inner container to center it.

| Classic properties | Logical properties    |
| ------------------ | --------------------- |
| horizontal         | inline base direction |
| vertical           | block flow direction  |
| width              | inline-size           |
| height             | block-size            |
| padding-left       | padding-inline-start  |
| padding-right      | padding-inline-end    |
| padding-top        | padding-block-start   |
| padding-bottom     | padding-block-end     |

Here is the [full list of logical properties](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_logical_properties_and_values)

margin-inline and margin-block are two useful shorthand logical properties.

## The box model

Top and bottom margins and paddings behave a little unusually on inline elements. They will still increase the height of the element, but they will not increase the height that the inline element contributes to its container; that height is derived from the inline element’s line-height. Using display: inline-block will change this behavior if necessary.

![border-box](../../assets/image/border-box.png)

border-box is a better way to control size.

## Element height

When you explicitly set an element's height, you run the risk of its contents overflowing the container.

You can control the exact behavior of the overflowing content with the overflow property:

![different overflow](../../assets/image/overflows.png)

Typically, auto is a prefer choice.

For percentage-based heights to work, the parent must have an explicitly defined height.

min-height and max-height is preferred to control a height.

## Negative margins

![negative margins](../../assets/image/negative_margins.png)

## Collapsed margins

When top and/or bottom margins are adjoining, they overlap, combining to form a single margin.

The size of the collapsed margin is equal to the largest of the joined margins.

Ways to prevent margins from collapsing:

- Applying overflow: auto (or any value other than visible) to the container prevents margins inside the container from collapsing with those outside the container. This is often the least intrusive solution.
- Adding a border or padding between two margins stops them from collapsing.
- Margins won’t collapse to the outside of a container that is an inline block, that is floated, or that has an absolute or fixed position.
- When using a flexbox or grid layout, margins won’t collapse between elements that are part of the flex layout.
- Elements with a table-cell display don’t have a margin, so they won’t collapse. This also applies to table-row and most other table display types. Exceptions are table, table-inline, and table-caption.

## Spacing elements

The interplay between the padding of a container and the margins of its contents can be tricky to work with.

lobotomized owl selector: `* + *`
