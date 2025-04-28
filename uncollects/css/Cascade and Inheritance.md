# Cascade and Inheritance

![High-level flowchart of the cascade](../../assets/image/cascade_flowchart.png)

## Cascade

### Stylesheet origin

1. author styles
2. user styles
3. user-agent styles(browser's default styles)

Different browsers' default styles:

- [Chromium](https://chromium.googlesource.com/chromium/blink/+/master/Source/core/css/html.css)
- [Firefox](https://dxr.mozilla.org/mozilla-central/source/layout/style/res/html.css)
- [Safari](https://github.com/WebKit/WebKit/blob/main/Source/WebCore/css/html.css)

Default styles summary:

1. Headings and paragraphs are given a top and bottom margin
2. Lists are given a left padding
3. Link colors are set
4. Default font sizes are set

### Selector specificity

The exact rules of specificity:

1. If a selector has more IDs, it wins.
2. If that results in a tie, the selector with the most classes wins.
3. If that results in a tie, the selector with the most tag names wins.

### Source order

Styling links should go in a certain order.

LoVe/HAte -- link, visited, hover, active

## Inheritance

If an element has no cascaded value for a given property, it may inherit one from an ancestor element.

By default, only certain ones are inherited. They are primarily properties pertaining to text, some are list properties and table border properties:

- color
- font
- font-family
- font-size
- font-weight
- font-variant
- font-style
- line-height
- letter-spacing
- text-align
- text-indent
- text-transform
- white-space
- word-spacing
- list-style
- list-style-type
- list-style-position
- list-style-image
- border-collapse
- border-spacing

Here is the [full list of inherited properties](https://stackoverflow.com/questions/5612302/which-css-properties-are-inherited)

## Special values

1. inherit -- Override the cascaded value and inherit the value from its parent.
2. initial -- Reset the value of that property to it's initial value.
3. unset -- Combination of inherit and initial, sets an inherited property to inherit, sets a non-inherited property to initial.
4. revert -- Override author styles but leave the user-agent styles intact.

The initial value is different to user-agent styles. We can lookup every property's initial value at [MDN](https://developer.mozilla.org/en-US/docs/Web/CSS)

## Shorthand properties

Most shorthand properties let you omit certain values. However, this sets the omitted values to their initial value and can silently override styles.

The order of shorthand values is clockwise order begin from top.

TRouBLe -- top, right, bottom, left
