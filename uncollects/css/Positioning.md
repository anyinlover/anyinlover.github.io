# Positioning

The initial value of the position property is static.

Positioning removes elements from the document flow.

## Fixed positioning

Applying `position: fixed` to an element lets you position the element arbitrarily within the viewport.

A modal dialog box is a window that appears in front of the main page.

Because a fixed element is removed from the document flow, it no longer affects the position of other elements on the page.

## Absolute positioning

Absolute positioning works the same way like fixed positioning, except its position is based on the closest-positioned ancestor element.

## Relative positioning

The relative positioning shift the element from its original position, but they won't change the position of any elements around it. It can't change the size of the element.

Far more often, `position: relative` is used to establish the containing block for an absolutely positioned element inside it.

## Stacking

As the browser parses HTML into the DOM, it also creates another tree structure called the render tree. This represents the visual appearance and position of each element. It’s also responsible for determining the order in which the browser will paint the elements.

Under normal circumstances (that is, before positioning is applied), this order is determined by the order the elements appear in the HTML.

This behavior changes when you start positioning elements. The browser first paints all non-positioned elements; then it paints the positioned ones.

Typically, modals are added to the end of the page as the last bit of content before the closing </body> tag.

The z-index property can be set to any integer (positive or negative). z refers to the depth dimension in a Cartesian x,y,z coordinate system. Elements with a higher z-index appear in front of elements with a lower z-index. Elements with a negative z-index appear behind static elements.

![stacking contexts](../../assets/image/stacking_contexts.png)

Adding a z-index is not the only way to create a stacking context. An opacity below 1 creates one, as do the transform or filter properties. Fixed positioning and sticky positioning always create a stacking context, even without a z-index. The document root (`<html>`) also creates a top-level stacking context for the whole page.

positioning takes elements out of the document flow. Generally speaking, you should do this only when you need to stack elements in front of one another.

## Sticking positioning

It’s sort of a hybrid between relative and fixed positioning: the element scrolls normally with the page until it reaches a specified point on the screen, at which point it will “lock” in place as the user continues to scroll.

Note that the parent must be taller than the sticky element for it to stick into place.
