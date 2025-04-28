# Flexbox

## Flexbox principles

A flex container asserts control over the layout of the elements within.

![flex container](../../assets/image/flex_container.png)

gap: add space between flex items

Flexbox allows you to use `margin: auto` to fill all available space between flex items.

## Flex item sizes

The `flex` property is shorthand for three different sizing properties: `flex-grow`, `flex-shrink` and `flex-basis`.

The flex basis defines a sort of starting point for the size of an element. It can be set to any value that would apply to width. Its initial value is auto, which means the browser will look to see if the element has a width declared. If so, the browser uses that size; if not, it determines the element’s size naturally by the contents.

The remaining space (or remainder) will be consumed by the flex items based on their flex-grow values, which are always specified as nonnegative integers. If an item has a flex-grow of 0, it won’t grow larger than its flex basis. If any items have a nonzero flex grow value, those items will grow until all of the remaining space is used up.

flex-shrink is similar to flex-grow.

![flex examples](../../assets/image/flex_examples.png)

## Flex direction

![flex direction](../../assets/image/flex_direction.png)

## Alignment and spacing

How to use a flexbox:

1. Identify a container and its items and use `display: flex` on the container.
2. If necessary, set a `gap` and/or `flex-direction` on the container.
3. Declare `flex` values for the flex items where necessary to control their size
4. Add other flexbox properties where necessary.

![flex-wrap](../../assets/image/flex_wrap.png)

![justify-content](../../assets/image/flex_justify_content.png)

![align-content](../../assets/image/flex_align_content.png)

![align-items](../../assets/image/flex_align_items.png)

![align-self](../../assets/image/flex_align_self.png)
